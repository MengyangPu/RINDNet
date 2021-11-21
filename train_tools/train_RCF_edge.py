import os
import torch
from tqdm import tqdm
from utils.lr_scheduler import LR_Scheduler
from dataloaders.datasets.bsds_hd5_dim1 import Mydataset
from torch.utils.data import DataLoader
from my_options.RCF_options import RCF_Options
from modeling.RCF_edge import RCF
from modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.edge_loss2 import *
from utils.saver import Saver
from utils.summaries import TensorboardSummary
import scipy.io as sio
import time
import cv2
from torch.optim import lr_scheduler
from utils.rcf_functions import  cross_entropy_loss_RCF, SGD_caffe
from utils.rcf_utils import Logger, Averagvalue, save_checkpoint, load_vgg16pretrain
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

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
        self.model = RCF()
        self.model.cuda()
        # Using cuda
        #if self.args.cuda:
        #    self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
        #    patch_replication_callback(self.model)
        #    self.model = self.model.cuda()

        self.model.apply(weights_init)
        load_vgg16pretrain(self.model, self.args.pretrain_model)
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
        net_parameters_id = {}
        net = self.model
        for pname, p in net.named_parameters():
            if pname in ['conv1_1.weight', 'conv1_2.weight',
                         'conv2_1.weight', 'conv2_2.weight',
                         'conv3_1.weight', 'conv3_2.weight', 'conv3_3.weight',
                         'conv4_1.weight', 'conv4_2.weight', 'conv4_3.weight']:
                print(pname, 'lr:1 de:1')
                if 'conv1-4.weight' not in net_parameters_id:
                    net_parameters_id['conv1-4.weight'] = []
                net_parameters_id['conv1-4.weight'].append(p)
            elif pname in ['conv1_1.bias', 'conv1_2.bias',
                           'conv2_1.bias', 'conv2_2.bias',
                           'conv3_1.bias', 'conv3_2.bias', 'conv3_3.bias',
                           'conv4_1.bias', 'conv4_2.bias', 'conv4_3.bias']:
                print(pname, 'lr:2 de:0')
                if 'conv1-4.bias' not in net_parameters_id:
                    net_parameters_id['conv1-4.bias'] = []
                net_parameters_id['conv1-4.bias'].append(p)
            elif pname in ['conv5_1.weight', 'conv5_2.weight', 'conv5_3.weight']:
                print(pname, 'lr:100 de:1')
                if 'conv5.weight' not in net_parameters_id:
                    net_parameters_id['conv5.weight'] = []
                net_parameters_id['conv5.weight'].append(p)
            elif pname in ['conv5_1.bias', 'conv5_2.bias', 'conv5_3.bias']:
                print(pname, 'lr:200 de:0')
                if 'conv5.bias' not in net_parameters_id:
                    net_parameters_id['conv5.bias'] = []
                net_parameters_id['conv5.bias'].append(p)
            elif pname in ['conv1_1_down.weight', 'conv1_2_down.weight',
                           'conv2_1_down.weight', 'conv2_2_down.weight',
                           'conv3_1_down.weight', 'conv3_2_down.weight', 'conv3_3_down.weight',
                           'conv4_1_down.weight', 'conv4_2_down.weight', 'conv4_3_down.weight',
                           'conv5_1_down.weight', 'conv5_2_down.weight', 'conv5_3_down.weight']:
                print(pname, 'lr:0.1 de:1')
                if 'conv_down_1-5.weight' not in net_parameters_id:
                    net_parameters_id['conv_down_1-5.weight'] = []
                net_parameters_id['conv_down_1-5.weight'].append(p)
            elif pname in ['conv1_1_down.bias', 'conv1_2_down.bias',
                           'conv2_1_down.bias', 'conv2_2_down.bias',
                           'conv3_1_down.bias', 'conv3_2_down.bias', 'conv3_3_down.bias',
                           'conv4_1_down.bias', 'conv4_2_down.bias', 'conv4_3_down.bias',
                           'conv5_1_down.bias', 'conv5_2_down.bias', 'conv5_3_down.bias']:
                print(pname, 'lr:0.2 de:0')
                if 'conv_down_1-5.bias' not in net_parameters_id:
                    net_parameters_id['conv_down_1-5.bias'] = []
                net_parameters_id['conv_down_1-5.bias'].append(p)
            elif pname in ['score_dsn1.weight', 'score_dsn2.weight', 'score_dsn3.weight',
                           'score_dsn4.weight', 'score_dsn5.weight']:
                print(pname, 'lr:0.01 de:1')
                if 'score_dsn_1-5.weight' not in net_parameters_id:
                    net_parameters_id['score_dsn_1-5.weight'] = []
                net_parameters_id['score_dsn_1-5.weight'].append(p)
            elif pname in ['score_dsn1.bias', 'score_dsn2.bias', 'score_dsn3.bias',
                           'score_dsn4.bias', 'score_dsn5.bias']:
                print(pname, 'lr:0.02 de:0')
                if 'score_dsn_1-5.bias' not in net_parameters_id:
                    net_parameters_id['score_dsn_1-5.bias'] = []
                net_parameters_id['score_dsn_1-5.bias'].append(p)
            elif pname in ['score_final.weight']:
                print(pname, 'lr:0.001 de:1')
                if 'score_final.weight' not in net_parameters_id:
                    net_parameters_id['score_final.weight'] = []
                net_parameters_id['score_final.weight'].append(p)
            elif pname in ['score_final.bias']:
                print(pname, 'lr:0.002 de:0')
                if 'score_final.bias' not in net_parameters_id:
                    net_parameters_id['score_final.bias'] = []
                net_parameters_id['score_final.bias'].append(p)

        self.optimizer = torch.optim.SGD([
            {'params': net_parameters_id['conv1-4.weight'], 'lr': self.args.lr * 1,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['conv1-4.bias'], 'lr': self.args.lr * 2, 'weight_decay': 0.},
            {'params': net_parameters_id['conv5.weight'], 'lr': self.args.lr * 100,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['conv5.bias'], 'lr': self.args.lr * 200, 'weight_decay': 0.},
            {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': self.args.lr * 0.1,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['conv_down_1-5.bias'], 'lr': self.args.lr * 0.2, 'weight_decay': 0.},
            {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': self.args.lr * 0.01,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['score_dsn_1-5.bias'], 'lr': self.args.lr * 0.02, 'weight_decay': 0.},
            {'params': net_parameters_id['score_final.weight'], 'lr': self.args.lr * 0.001,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['score_final.bias'], 'lr': self.args.lr * 0.002, 'weight_decay': 0.},
        ], lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.args.stepsize, gamma=self.args.gamma)

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
        batch_time = Averagvalue()
        data_time = Averagvalue()
        losses = Averagvalue()
        # switch to train mode
        self.model.train()
        end = time.time()
        epoch_loss = []
        counter = 0

        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, (image, target) in enumerate(tbar):
            data_time.update(time.time() - end)
            image, target = image.cuda(), target.cuda()
            target = target.unsqueeze(1)  #b,1,w,h
            outputs = self.model(image)
            loss = torch.zeros(1).cuda()
            for o in outputs:
                #print(o.shape, target.shape)
                loss = loss + cross_entropy_loss_RCF(o, target)
            counter += 1
            loss = loss / self.args.itersize
            loss.backward()
            if counter == self.args.itersize:
                self.optimizer.step()
                self.optimizer.zero_grad()
                counter = 0
            # measure accuracy and record loss
            losses.update(loss.item(), image.size(0))
            epoch_loss.append(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
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

    def test(self, epoch):
        print('Test epoch: %d' % epoch)
        self.output_dir = os.path.join(self.saver.experiment_dir, str(epoch+1), 'mat')
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

    def multiscale_test(self, epoch):
        print('Test epoch: %d' % epoch)
        self.output_dir = os.path.join(self.saver.experiment_dir, str(epoch+1), 'mat_ms')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.model.eval()
        scale = [0.5, 1, 1.5]
        tbar = tqdm(self.test_loader, desc='\r')
        for i, image in enumerate(tbar):
            name = self.test_loader.dataset.images_name[i]
            image = image[0]
            image_in = image.numpy().transpose((1, 2, 0))
            _, H, W = image.shape
            multi_fuse = np.zeros((H, W), np.float32)
            for k in range(0, len(scale)):
                im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                im_ = im_.transpose((2, 0, 1))
                results = self.model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
                result = torch.squeeze(results[-1].detach()).cpu().numpy()

                fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
                multi_fuse += fuse

            multi_fuse = multi_fuse / len(scale)

            sio.savemat(os.path.join(self.output_dir, '{}.mat'.format(name)), {'result': multi_fuse})


def main():
    options = RCF_Options()
    args = options.parse()
    args.cuda = True
    #args.cuda = not args.no_cuda and torch.cuda.is_available()
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

    args.data_path = 'data/BSDS-RIND/BSDS-RIND-Edge/Augmentation/'
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if (epoch+1)%10==0:
            trainer.test(epoch)
            trainer.multiscale_test(epoch)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == "__main__":
    main()
