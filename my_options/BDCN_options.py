# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that my_options.py resides in


class BDCN_Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="PyTorch OFNet Training")
        self.parser.add_argument('--backbone', type=str, default='vgg16',
                                 help='backbone name (default: vgg16)')
        self.parser.add_argument('--dataset', type=str, default='bsds',
                                 choices=['bsds'],
                                 help='dataset name (default: bsds)')
        self.parser.add_argument("--data-path", type=str, help="path to the training data",
                                 default="data/BSDS-RIND/BSDS-RIND/Augmentation/")
        #                        default="data/BSDS-RIND/BSDS-RIND-Edge/Augmentation/")
        self.parser.add_argument('--workers', type=int, default=4, metavar='N',
                                 help='dataloader threads')
        self.parser.add_argument('--base-size', type=int, default=320,
                                 help='base image size')
        self.parser.add_argument('--crop-size', type=int, default=320,
                                 help='crop image size')
        self.parser.add_argument('--sync-bn', type=bool, default=None,
                                 help='whether to use sync bn (default: auto)')
        self.parser.add_argument('--freeze-bn', type=bool, default=False,
                                 help='whether to freeze bn parameters (default: False)')
        self.parser.add_argument('--loss-type', type=str, default='attention',
                                 choices=['ce', 'focal', 'attention'],
                                 help='loss func type (default: ce)')
        # training hyper params
        self.parser.add_argument('--epochs', type=int, default=70, metavar='N',
                                 help='number of epochs to train (default: auto)')
        self.parser.add_argument('--start_epoch', type=int, default=0,
                                 metavar='N', help='start epochs (default:0)')
        self.parser.add_argument('--batch-size', type=int, default=4,
                                 metavar='N', help='input batch size for training (default: auto)')
        self.parser.add_argument('--test-batch-size', type=int, default=1,
                                 metavar='N', help='input batch size for testing (default: auto)')
        self.parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                                 help='whether to use balanced weights (default: False)')
        # optimizer params
        self.parser.add_argument('--lr', type=float, default=1e-7, metavar='LR',
                                 help='learning rate (default: auto)')
        self.parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'],
                                 help='lr scheduler mode: (default: poly)')
        self.parser.add_argument('--momentum', type=float, default=0.9,
                                 metavar='M', help='momentum (default: 0.9)')
        self.parser.add_argument('--weight-decay', type=float, default=0.0002,
                                 metavar='M', help='w-decay (default: 5e-4)')
        self.parser.add_argument('--nesterov', action='store_true', default=False,
                                 help='whether use nesterov (default: False)')
        # cuda, seed and logging
        self.parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='disables CUDA training')
        self.parser.add_argument('--gpu-ids', type=str, default='0',
                                 help='use which gpu to train, must be a \
                                 comma-separated list of integers only (default=0)')
        self.parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
        # checking point
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='put the path to resuming file if needed')
        self.parser.add_argument('--checkname', type=str, default='bdcn-edge',
                                 help='set the checkpoint name')
        # finetuning pre-trained models
        self.parser.add_argument('--ft', action='store_true', default=False,
                                 help='finetuning on a different dataset')
        # evaluation option
        self.parser.add_argument('--eval-interval', type=int, default=1,
                                 help='evaluuation interval (default: 1)')
        self.parser.add_argument('--no-val', action='store_true', default=True,
                                 help='skip validation during training')
        self.parser.add_argument('--pretrain-model', default='../rindnet/model/vgg16.pth',
                                 help='Resume VGG-16 Caffe parameters.')

        #####from BDCN Projects
        #self.parser.add_argument('--max-iter', type=int, default=40000,
        #                        help='max iters to train network, default is 40000')
        #self.parser.add_argument('--iter-size', type=int, default=10,
        #                        help='iter size equal to the batch size, default 10')
        #self.parser.add_argument('--average-loss', type=int, default=50,
        #                        help='smoothed loss, default is 50')
        #self.parser.add_argument('-s', '--snapshots', type=int, default=10000,
        #                        help='how many iters to store the params, default is 1000')
        #self.parser.add_argument('--step-size', type=int, default=10000,
        #                        help='the number of iters to decrease the learning rate, default is 10000')
        #self.parser.add_argument('--display', type=int, default=20,
        #                        help='how many iters display one time, default is 20')
        self.parser.add_argument('-b', '--balance', type=float, default=1.1,
                                help='the parameter to balance the neg and pos, default is 1.1')
        #self.parser.add_argument('-l', '--log', type=str, default='log.txt',
        #                        help='the file to store log, default is log.txt')
        #self.parser.add_argument('-k', type=int, default=1,
        #                        help='the k-th split set of multicue')
        #self.parser.add_argument('--yita', type=float, default=None,
        #                        help='the param to operate gt, default is data in the config file')
        #self.parser.add_argument('--complete-pretrain', type=str, default=None,
        #                        help='finetune on the complete_pretrain, default None')
        self.parser.add_argument('--side-weight', type=float, default=0.5,
                                help='the loss weight of sideout, default 0.5')
        self.parser.add_argument('--fuse-weight', type=float, default=1.1,
                                help='the loss weight of fuse, default 1.1')
        #self.parser.add_argument('--gamma', type=float, default=0.1,
        #                        help='the decay of learning rate, default 0.1')

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
