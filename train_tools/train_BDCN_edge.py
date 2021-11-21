import os
import torch
from tqdm import tqdm
from utils.lr_scheduler import LR_Scheduler
from dataloaders.datasets.bsds_hd5_dim1 import Mydataset
from torch.utils.data import DataLoader
from my_options.BDCN_options import BDCN_Options
from modeling.BDCN_edge import BDCN
from modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.edge_loss2 import *
from utils.bdcn_loss import bdcn_loss_edge
from utils.saver import Saver
from utils.summaries import TensorboardSummary
import scipy.io as sio
import time
import re
from torch.optim import lr_scheduler
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        print(self.saver.experiment_dir)
        self.output_dir = os.path.join(self.saver.experiment_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Define Dataloader
        self.train_dataset = Mydataset(root_path=self.args.data_path, split='trainval', crop_size=self.args.crop_size)
        self.test_dataset = Mydataset(root_path=self.args.data_path, split='test', crop_size=self.args.crop_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=args.workers, pin_memory=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                      num_workers=args.workers)

        # Define network
        self.model = BDCN(self.args.pretrain_model)
        self.model.cuda()

        if args.resume:
            if isfile(args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}'"
                      .format(args.resume))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        # tune lr
        params_dict = dict(self.model.named_parameters())
        base_lr = args.lr
        weight_decay = args.weight_decay
        params = []
        for key, v in params_dict.items():
            if re.match(r'conv[1-5]_[1-3]_down', key):
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 0.1, 'weight_decay': weight_decay * 1, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 0.2, 'weight_decay': weight_decay * 0, 'name': key}]
            elif re.match(r'.*conv[1-4]_[1-3]', key):
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 1, 'weight_decay': weight_decay * 1, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 2, 'weight_decay': weight_decay * 0, 'name': key}]
            elif re.match(r'.*conv5_[1-3]', key):
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 100, 'weight_decay': weight_decay * 1, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 200, 'weight_decay': weight_decay * 0, 'name': key}]
            elif re.match(r'score_dsn[1-5]', key):
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 0.01, 'weight_decay': weight_decay * 1, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 0.02, 'weight_decay': weight_decay * 0, 'name': key}]
            elif re.match(r'upsample_[248](_5)?', key):
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 0, 'weight_decay': weight_decay * 0, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 0, 'weight_decay': weight_decay * 0, 'name': key}]
            elif re.match(r'.*msblock[1-5]_[1-3]\.conv', key):
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 1, 'weight_decay': weight_decay * 1, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 2, 'weight_decay': weight_decay * 0, 'name': key}]
            else:
                if 'weight' in key:
                    params += [{'params': v, 'lr': base_lr * 0.001, 'weight_decay': weight_decay * 1, 'name': key}]
                elif 'bias' in key:
                    params += [{'params': v, 'lr': base_lr * 0.002, 'weight_decay': weight_decay * 0, 'name': key}]
        self.optimizer = torch.optim.SGD(params, momentum=self.args.momentum,
                                    lr=self.args.lr, weight_decay=self.args.weight_decay)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr,
                                      args.epochs, len(self.train_loader))

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        batch_size = self.args.batch_size
        for i, (image, target) in enumerate(tbar):
            if self.args.cuda:
                image, target = image.cuda(), target.cuda() #(b,3,w,h) (b,1,w,h)
                target = target.unsqueeze(1)
            out = self.model(image)
            loss = 0
            for k in range(10):
                loss += self.args.side_weight * bdcn_loss_edge(out[k], target, self.args.cuda,
                                                            self.args.balance) / batch_size
            loss += self.args.fuse_weight * bdcn_loss_edge(out[-1], target, self.args.cuda, self.args.balance) / batch_size
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            if (epoch + 1) % 10 == 0:
                is_best = False
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best)


    def test(self,epoch):
        print('Test epoch: %d' % epoch)
        self.output_dir = os.path.join(self.saver.experiment_dir,  str(epoch+1), 'mat')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        for i, image in enumerate(tbar):
            name = self.test_loader.dataset.images_name[i]
            if self.args.cuda:
                image = image.cuda()
            with torch.no_grad():
                output_list = self.model(image)

            pred = output_list[-1]
            pred = pred.squeeze()
            pred = pred.data.cpu().numpy()
            sio.savemat(os.path.join(self.output_dir, '{}.mat'.format(name)), {'result': pred})


def main():
    options = BDCN_Options()
    args = options.parse()
    args.cuda = True
    #args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args.cuda)
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    print(args)

    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if (epoch + 1) % 10 == 0:
            trainer.test(epoch)

def adjust_learning_rate(optimizer, steps, step_size, gamma=0.1, logger=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma
        if logger:
            logger.info('%s: %s' % (param_group['name'], param_group['lr']))

if __name__ == "__main__":
    main()
