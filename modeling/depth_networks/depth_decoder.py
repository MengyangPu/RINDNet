# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        '''
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        '''

        self.upconv4_0 = ConvBlock(2048, 256)
        self.upconv4_1 = ConvBlock(1280, 256)
        self.upconv3_0 = ConvBlock(256, 128)
        self.upconv3_1 = ConvBlock(640, 128)
        self.upconv2_0 = ConvBlock(128, 64)
        self.upconv2_1 = ConvBlock(320, 64)
        self.upconv1_0 = ConvBlock(64, 32)
        self.upconv1_1 = ConvBlock(96, 32)
        self.upconv0_0 = ConvBlock(32, 16)
        self.upconv0_1 = ConvBlock(16, 16)

        #for s in self.scales:
        #    self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.dispconv_0 = Conv3x3(16, 1)
        self.dispconv_1 = Conv3x3(32, 1)
        self.dispconv_2 = Conv3x3(64, 1)
        self.dispconv_3 = Conv3x3(128, 1)

        #self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}
        # decoder
        x = input_features[-1]
        '''
        for i in range(4, -1, -1):
            print(i)
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        '''

        x = self.upconv4_0(x)
        x = [upsample(x)]
        x += [input_features[3]]
        x = torch.cat(x, 1)
        x = self.upconv4_1(x)

        x = self.upconv3_0(x)
        x = [upsample(x)]
        x += [input_features[2]]
        x = torch.cat(x, 1)
        x = self.upconv3_1(x)
        outputs[("disp", 3)] = self.sigmoid(self.dispconv_3(x))

        x = self.upconv2_0(x)
        x = [upsample(x)]
        x += [input_features[1]]
        x = torch.cat(x, 1)
        x = self.upconv2_1(x)
        outputs[("disp", 2)] = self.sigmoid(self.dispconv_2(x))

        x = self.upconv1_0(x)
        x = [upsample(x)]
        x += [input_features[0]]
        x = torch.cat(x, 1)
        x = self.upconv1_1(x)
        outputs[("disp", 1)] = self.sigmoid(self.dispconv_1(x))

        x = self.upconv0_0(x)
        x = [upsample(x)]
        x = torch.cat(x, 1)
        x = self.upconv0_1(x)
        outputs[("disp", 0)] = self.sigmoid(self.dispconv_0(x))

        return outputs