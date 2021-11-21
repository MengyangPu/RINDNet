###########################################################################
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from modeling.base import BaseNet


class CaseNet(BaseNet):
    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CaseNet, self).__init__(nclass, backbone, norm_layer=norm_layer, **kwargs)

        self.side1 = nn.Conv2d(64, 1, 1)
        self.side2 = nn.Sequential(nn.Conv2d(256, 1, 1, bias=True),
                                   nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        self.side3 = nn.Sequential(nn.Conv2d(512, 1, 1, bias=True),
                                   nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False))
        self.side5 = nn.Sequential(nn.Conv2d(2048, nclass, 1, bias=True),
                                   nn.ConvTranspose2d(nclass, nclass, 16, stride=8, padding=4, bias=False))
        self.fuse = nn.Conv2d(nclass*4, nclass, 1, groups=nclass, bias=True)

    def forward(self, x):
        c1, c2, c3, _, c5 = self.base_forward(x)
        #[1, 64, 320, 320]    [1, 256, 160, 160]    [1, 512, 80, 80]  [1, 2048, 40, 40]
        side1 = self.side1(c1)  #[1, 1, 320, 320]
        side2 = self.side2(c2)  #[1, 1, 320, 320]
        side3 = self.side3(c3)  #[1, 1, 320, 320]
        side5 = self.side5(c5)  #[1, 4, 320, 320]

        slice5 = side5[:,0:1,:,:]   #[1, 1, 320, 320]
        fuse = torch.cat((slice5, side1, side2, side3), 1)  #[1, 4, 320, 320]
        for i in range(side5.size(1)-1):    # range(0,3) i = 0, 1, 2
            slice5 = side5[:,i+1:i+2,:,:]
            fuse = torch.cat((fuse, slice5, side1, side2, side3), 1)    #[1, 8, 320, 320]   [1, 12, 320, 320]   [1, 16, 320, 320]

        fuse = self.fuse(fuse)  #[1, 4, 320, 320]

        outputs = [torch.sigmoid(side5), torch.sigmoid(fuse)]

        return tuple(outputs)


if __name__ == '__main__':
    model = CaseNet(4, backbone='resnet50')
    dummy_input = torch.rand(1, 3, 320, 320)
    output = model(dummy_input)
    for out in output:
        print(out.size())
        #print(out)
