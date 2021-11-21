# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that my_options.py resides in


class HED_Options:
    def __init__(self):
        # Parse arguments.
        self.parser = argparse.ArgumentParser(description='HEDOcclusion training.')
        self.parser.add_argument('--backbone', type=str, default='vgg16',
                                 choices=['resnet', 'xception', 'drn', 'mobilenet'],
                                 help='backbone name (default: resnet)')
        self.parser.add_argument('--dataset', type=str, default='bsds',
                                 choices=['bsds'],
                                 help='dataset name (default: bsds)')
        self.parser.add_argument('--workers', type=int, default=4, metavar='N',
                                help='dataloader threads')
        # 1. Actions.
        self.parser.add_argument('--test', default=False, help='Only test the model.', action='store_true')
        # 2. Counts.
        self.parser.add_argument('--train_batch_size', default=1, type=int, metavar='N', help='Training batch size.')
        self.parser.add_argument('--test_batch_size', default=1, type=int, metavar='N', help='Test batch size.')
        self.parser.add_argument('--train_iter_size', default=10, type=int, metavar='N', help='Training iteration size.')
        self.parser.add_argument('--epochs', default=70, type=int, metavar='N', help='Total epochs.')
        self.parser.add_argument('--start_epoch', type=int, default=0,
                                 metavar='N', help='start epochs (default:0)')
        #self.parser.add_argument('--print_freq', default=500, type=int, metavar='N', help='Print frequency.')
        self.parser.add_argument('--base-size', type=int, default=320,
                                 help='base image size')
        self.parser.add_argument('--crop-size', type=int, default=320,
                                 help='crop image size')
        self.parser.add_argument('--batch-size', type=int, default=4,
                                 metavar='N', help='input batch size for training (default: auto)')
        self.parser.add_argument('--test-batch-size', type=int, default=1,
                                 metavar='N', help='input batch size for testing (default: auto)')
        # cuda, seed and logging
        self.parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='disables CUDA training')
        self.parser.add_argument('--gpu-ids', type=str, default='0',
                                 help='use which gpu to train, must be a \
                                         comma-separated list of integers only (default=0)')
        # 3. Optimizer settings.
        #self.parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
        #                         help='learning rate (default: auto)')
        #self.parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'],
        #                         help='lr scheduler mode: (default: poly)')
        self.parser.add_argument('--lr', default=1e-6, type=float, metavar='F', help='Initial learning rate.')
        self.parser.add_argument('--lr_stepsize', default=1e4, type=int, metavar='N', help='Learning rate step size.')
        # Note: Step size is based on number of iterations, not number of batches.
        #   https://github.com/s9xie/hed/blob/94fb22f10cbfec8d84fbc0642b224022014b6bd6/src/caffe/solver.cpp#L498
        self.parser.add_argument('--lr_gamma', default=0.1, type=float, metavar='F', help='Learning rate decay (gamma).')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='F', help='Momentum.')
        self.parser.add_argument('--weight_decay', default=2e-4, type=float, metavar='F', help='Weight decay.')
        # 4. Files and folders.
        self.parser.add_argument('--vgg16_caffe', default='../rindnet/model/5stage-vgg.py36pickle',
                            help='Resume VGG-16 Caffe parameters.')
        self.parser.add_argument("--data-path", type=str, help="path to the training data",
        #                         default="data/BSDS-RIND/BSDS-RIND/Augmentation/")
                                default="data/BSDS-RIND/BSDS-RIND-Edge/Augmentation/")
        self.parser.add_argument('--checkname', type=str, default='hed',
                                 help='set the checkpoint name')
        # 5. Others.
        self.parser.add_argument('--cpu', default=False, help='Enable CPU mode.', action='store_true')
        # evaluation option
        self.parser.add_argument('--eval-interval', type=int, default=1,
                                 help='evaluuation interval (default: 1)')
        self.parser.add_argument('--no-val', action='store_true', default=True,
                                 help='skip validation during training')

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
