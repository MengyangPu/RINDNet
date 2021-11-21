###########################################################################
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
###########################################################################

import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch CaseNet')
        # model and dataset 
        parser.add_argument('--model', type=str, default='casenet',
                            help='model name (default: casenet)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='bsds',
                            help='dataset name (default: bsds)')
        parser.add_argument("--data-path", type=str, help="path to the training data",
                            default="data/BSDS-RIND/BSDS-RIND/Augmentation/")
        #                        default="data/BSDS-RIND/BSDS-RIND-Edge/Augmentation/")
        parser.add_argument('--workers', type=int, default=4,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=320,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=320,
                            help='crop image size')
        parser.add_argument('--sync-bn', type=bool, default=None,
                            help='whether to use sync bn (default: auto)')
        # training hyper params
        parser.add_argument('--epochs', type=int, default=70, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=4,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=1,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=1e-7, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--lr-step', type=int, default=None,
                            help='lr step to change lr')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--gpu-ids', type=str, default='0',
                            help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
        parser.add_argument('--log-root', type=str,
                            default='./dff/log', help='set a log path folder')

        # checking point
        parser.add_argument('--resnet', default='resnet50-19c8e357.pth', type=str,
                            help='resnet model file')
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--resume-dir', type=str, default=None,
                            help='put the path to resuming dir if needed')
        parser.add_argument('--checkname', type=str, default='casenet',
                            help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default= False,
                            help='finetuning on a different dataset')
        parser.add_argument('--ft-resume', type=str, default=None,
                            help='put the path of trained model to finetune if needed ')
        parser.add_argument('--pre-class', type=int, default=None,
                            help='num of pre-trained classes \
                            (default: None)')

        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False,
                            help='evaluating mIoU')
        parser.add_argument('--no-val', action='store_true', default= True,
                            help='skip validation during training')
        # test option
        parser.add_argument('--test-folder', type=str, default=None,
                            help='path to test image folder')
        parser.add_argument('--scale', action='store_false', default=True,
                           help='choose to use random scale transform(0.75-2),default:multi scale')
        # the parser
        self.parser = parser

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
