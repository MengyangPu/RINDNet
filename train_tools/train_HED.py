import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.optim import lr_scheduler
from utils.hed_utils import *
from dataloaders.datasets.bsds_hd5 import Mydataset
from torch.utils.data import DataLoader
from my_options.hed_options import HED_Options
from modeling.hed import HED
from utils.hed_loss import HED_Loss
from utils.saver import Saver
from utils.summaries import TensorboardSummary
import scipy.io as sio

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
        self.model = HED('cuda')
        self.model = nn.DataParallel(self.model)
        self.model.to('cuda')

        # Initialize the weights for HED model.
        def weights_init(m):
            """ Weight initialization function. """
            if isinstance(m, nn.Conv2d):
                # Initialize: m.weight.
                if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
                    # Constant initialization for fusion layer in HED network.
                    torch.nn.init.constant_(m.weight, 0.2)
                else:
                    # Zero initialization following official repository.
                    # Reference: hed/docs/tutorial/layers.md
                    m.weight.data.zero_()
                # Initialize: m.bias.
                if m.bias is not None:
                    # Zero initialization.
                    m.bias.data.zero_()

        self.model.apply(weights_init)

        # Optimizer settings.
        net_parameters_id = defaultdict(list)
        for name, param in self.model.named_parameters():
            if name in ['module.conv1_1.weight', 'module.conv1_2.weight',
                        'module.conv2_1.weight', 'module.conv2_2.weight',
                        'module.conv3_1.weight', 'module.conv3_2.weight', 'module.conv3_3.weight',
                        'module.conv4_1.weight', 'module.conv4_2.weight', 'module.conv4_3.weight']:
                print('{:26} lr:    1 decay:1'.format(name));
                net_parameters_id['conv1-4.weight'].append(param)
            elif name in ['module.conv1_1.bias', 'module.conv1_2.bias',
                          'module.conv2_1.bias', 'module.conv2_2.bias',
                          'module.conv3_1.bias', 'module.conv3_2.bias', 'module.conv3_3.bias',
                          'module.conv4_1.bias', 'module.conv4_2.bias', 'module.conv4_3.bias']:
                print('{:26} lr:    2 decay:0'.format(name));
                net_parameters_id['conv1-4.bias'].append(param)
            elif name in ['module.conv5_1.weight', 'module.conv5_2.weight', 'module.conv5_3.weight']:
                print('{:26} lr:  100 decay:1'.format(name));
                net_parameters_id['conv5.weight'].append(param)
            elif name in ['module.conv5_1.bias', 'module.conv5_2.bias', 'module.conv5_3.bias']:
                print('{:26} lr:  200 decay:0'.format(name));
                net_parameters_id['conv5.bias'].append(param)
            elif name in ['module.score_dsn1.weight', 'module.score_dsn2.weight',
                          'module.score_dsn3.weight', 'module.score_dsn4.weight', 'module.score_dsn5.weight']:
                print('{:26} lr: 0.01 decay:1'.format(name));
                net_parameters_id['score_dsn_1-5.weight'].append(param)
            elif name in ['module.score_dsn1.bias', 'module.score_dsn2.bias',
                          'module.score_dsn3.bias', 'module.score_dsn4.bias', 'module.score_dsn5.bias']:
                print('{:26} lr: 0.02 decay:0'.format(name));
                net_parameters_id['score_dsn_1-5.bias'].append(param)
            elif name in ['module.score_final.weight']:
                print('{:26} lr:0.001 decay:1'.format(name));
                net_parameters_id['score_final.weight'].append(param)
            elif name in ['module.score_final.bias']:
                print('{:26} lr:0.002 decay:0'.format(name));
                net_parameters_id['score_final.bias'].append(param)

        # Define Optimizer
        self.optimizer = torch.optim.SGD([
            {'params': net_parameters_id['conv1-4.weight'], 'lr': self.args.lr * 1, 'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['conv1-4.bias'], 'lr': self.args.lr * 2, 'weight_decay': 0.},
            {'params': net_parameters_id['conv5.weight'], 'lr': self.args.lr * 100, 'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['conv5.bias'], 'lr': self.args.lr * 200, 'weight_decay': 0.},
            {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': self.args.lr * 0.01,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['score_dsn_1-5.bias'], 'lr': self.args.lr * 0.02, 'weight_decay': 0.},
            {'params': net_parameters_id['score_final.weight'], 'lr': self.args.lr * 0.001,
             'weight_decay': self.args.weight_decay},
            {'params': net_parameters_id['score_final.bias'], 'lr': self.args.lr * 0.002, 'weight_decay': 0.},
        ], lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        # Note: In train_val.prototxt and deploy.prototxt, the learning rates of score_final.weight/bias are different.

        # Learning rate scheduler.
        self.lr_schd = lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_stepsize, gamma=self.args.lr_gamma)
        # Loading pre-trained model
        if self.args.vgg16_caffe:
            load_vgg16_caffe(self.model, self.args.vgg16_caffe)

        # Define Criterion
        self.criterion = HED_Loss()
        # Resuming checkpoint
        self.best_pred = 0.0

    def training(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        batch_loss_meter = AverageMeter()
        counter = 0
        tbar = tqdm(self.train_loader)
        for i, (image, target) in enumerate(tbar):
            counter += 1
            image, target = image.cuda(), target.cuda() #(b,3,w,h) (b,1,w,h)
            target = target[:,1:5,:,:]
            preds_list = self.model(image)

            batch_loss = sum([self.criterion(preds, target) for preds in preds_list])
            tbar.set_description('Train loss: %.3f' % (batch_loss))
            eqv_iter_loss = batch_loss / self.args.train_iter_size

            eqv_iter_loss.backward()
            if counter == self.args.train_iter_size:
                self.optimizer.step()
                self.optimizer.zero_grad()
                counter = 0

            if counter == 0:
                self.lr_schd.step()

            batch_loss_meter.update(batch_loss.item())

        if (epoch + 1) % 10 == 0:
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def test(self, epoch):
        print('Test epoch: %d' % epoch)
        self.output_dir = os.path.join(self.saver.experiment_dir, str(epoch+1))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.depth_output_dir = os.path.join(self.saver.experiment_dir, str(epoch + 1), 'depth/mat')
        if not os.path.exists(self.depth_output_dir):
            os.makedirs(self.depth_output_dir)
        self.normal_output_dir = os.path.join(self.saver.experiment_dir, str(epoch + 1), 'normal/mat')
        if not os.path.exists(self.normal_output_dir):
            os.makedirs(self.normal_output_dir)
        self.reflectance_output_dir = os.path.join(self.saver.experiment_dir, str(epoch + 1),
                                                   'reflectance/mat')
        if not os.path.exists(self.reflectance_output_dir):
            os.makedirs(self.reflectance_output_dir)
        self.illumination_output_dir = os.path.join(self.saver.experiment_dir, str(epoch + 1),
                                                    'illumination/mat')
        if not os.path.exists(self.illumination_output_dir):
            os.makedirs(self.illumination_output_dir)

        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        for i, image in enumerate(tbar):
            name = self.test_loader.dataset.images_name[i]
            image = image.cuda()
            with torch.no_grad():
                preds_list = self.model(image)
            pred = preds_list[-1]
            pred = pred.squeeze()

            out_depth = pred[0, :, :]
            out_normal = pred[1, :, :]
            out_reflectance = pred[2, :, :]
            out_illumination = pred[3, :, :]

            depth_pred = out_depth.data.cpu().numpy()
            depth_pred = depth_pred.squeeze()
            sio.savemat(os.path.join(self.depth_output_dir, '{}.mat'.format(name)), {'result': depth_pred})

            normal_pred = out_normal.data.cpu().numpy()
            normal_pred = normal_pred.squeeze()
            sio.savemat(os.path.join(self.normal_output_dir, '{}.mat'.format(name)), {'result': normal_pred})

            reflectance_pred = out_reflectance.data.cpu().numpy()
            reflectance_pred = reflectance_pred.squeeze()
            sio.savemat(os.path.join(self.reflectance_output_dir, '{}.mat'.format(name)), {'result': reflectance_pred})

            illumination_pred = out_illumination.data.cpu().numpy()
            illumination_pred = illumination_pred.squeeze()
            sio.savemat(os.path.join(self.illumination_output_dir, '{}.mat'.format(name)),
                        {'result': illumination_pred})

def main():
    options = HED_Options()
    args = options.parse()

    print(args)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if (epoch+1)%10==0:
            trainer.test(epoch)


if __name__ == "__main__":
    main()
