import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

def Conv_Stage3(input_dim,dim_list, bias=True, output_map=False):
    num_layers = len(dim_list)
    dim_list = [input_dim] + dim_list

    layers = []
    for i in range(num_layers):
        layer = nn.Sequential(
            nn.Conv2d(dim_list[i], dim_list[i+1], kernel_size=1, bias=bias),
            nn.BatchNorm2d(dim_list[i+1]),
            nn.ReLU(inplace=True)
            )
        layers.append(layer)

    if output_map:
        layer = nn.Conv2d(dim_list[-1], 1, kernel_size=1)
        layers.append(layer)

    ## with padding, doesn't change the resolution
    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=4, dilation_rate=1):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        pad = 2 if dilation_rate == 2 else 1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=pad, bias=False, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        ## output channel: 4*inplanes
        return out

def Conv_Stage(input_dim,dim_list, bias=True, output_map=False):
    num_layers = len(dim_list)
    dim_list = [input_dim] + dim_list

    layers = []
    for i in range(num_layers):
        layer = nn.Sequential(
            nn.Conv2d(dim_list[i], dim_list[i+1], kernel_size=3, bias=bias,padding=1),
            nn.BatchNorm2d(dim_list[i+1]),
            nn.ReLU(inplace=True)
            )
        layers.append(layer)

    if output_map:
        layer = nn.Conv2d(dim_list[-1], 1, kernel_size=1)
        layers.append(layer)

    ## with padding, doesn't change the resolution
    return nn.Sequential(*layers)

def Conv_Stage2(input_dim,dim_list, bias=True, output_map=False):
    num_layers = len(dim_list)
    dim_list = [input_dim] + dim_list

    layers = []
    for i in range(num_layers):
        layer = nn.Sequential(
            nn.Conv2d(dim_list[i], dim_list[i+1], kernel_size=3, bias=bias,padding=1),
            nn.BatchNorm2d(dim_list[i+1]),
            nn.ReLU(inplace=True)
            )
        layers.append(layer)

    if output_map:
        layer = nn.Conv2d(dim_list[-1], 5, kernel_size=1)
        layers.append(layer)

    ## with padding, doesn't change the resolution
    return nn.Sequential(*layers)

class MyNet(nn.Module):

    def __init__(self):
        self.inplanes = 64
        super(MyNet, self).__init__()

        # resnet-50 part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3) ##256
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2) ## 512
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2) ## 1024
        self.layer4 = self._make_dilation_layer(Bottleneck, 512, 3) ## 2048  add dilation conv in res-stage 5

        # conv1 for boundary
        self.conv1_b = Conv_Stage(3, [8, 4, 16])
        self.conv2_b = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.soft_boundary=Conv_Stage2(16, [8, 8, 8, 8], output_map=True)

        # res1_up
        self.res1_up = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        # res2_up
        self.res2_up = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2, 2, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        # res3_up
        self.res3_up = nn.Sequential(
            nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2, 2, kernel_size=16, stride=8, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        # res4_up
        self.res4_up = nn.Sequential(
            nn.Conv2d(1024, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2, 2, kernel_size=32, stride=16, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        # res5_up
        self.res5_up = nn.Sequential(
            nn.Conv2d(2048, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2, 2, kernel_size=32, stride=16, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )

        ### branch for depth & normal
        self.conv6 = Conv_Stage(2048,[256,256], bias=False)
        ## branch1 for depth
        # res5c_1 res5c_up1 res5c_up2
        self.depth_res5c_up1 = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.depth_res5c_up2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        ## branch1 for normal
        # res5c_1 res5c_up1 res5c_up2
        self.normal_res5c_up1 = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.normal_res5c_up2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        ## branch3 for depth & normal
        # unet3a_deconv_up
        self.unet3a_up = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # self.unet3a
        self.unet3a = self._make_resblock(Bottleneck, 512, 128)
        # self.unet3b
        self.unet3b = self._make_resblock(Bottleneck, 512, 8, expansion=2)
        # self.unet1a
        self.unet1a_up = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )


        ### branch for reflectance
        # res3_up
        self.reflectance_res3_up1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # conv7
        self.reflectance_conv7 = Conv_Stage(832,[256,256], bias=False)
        ## branch1 for reflectance
        # res123_up1 res123_up2
        self.reflectance_res123_up1 = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.reflectance_res123_up2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        ## branch2 for reflectance
        # unet3a_deconv_up
        self.reflectance_unet3a_up_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # self.unet3a
        self.reflectance_unet3a_2 = self._make_resblock(Bottleneck, 256, 128)
        # self.unet3b
        self.reflectance_unet3b_2 = self._make_resblock(Bottleneck, 512, 8, expansion=2)
        # self.unet1a
        self.reflectance_unet1a_up_2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        #branch for reflectance get weight_map from res5_out
        self.reflectance_res5 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.reflectance_weight = Conv_Stage(256,[256,256], bias=False)



        ### branch for illumination
        # res3_up
        self.illumination_res3_up1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # branch for illumination get weight_map from res5_out
        self.illumination_res5 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.illumination_weight = Conv_Stage(256, [256, 256], bias=False)
        # conv7
        self.illumination_conv7 = Conv_Stage(832, [256, 256], bias=False)
        ## branch1 for reflectance
        # res123_up1 res123_up2
        self.illumination_res123_up1 = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.illumination_res123_up2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        ## branch2 for reflectance
        # unet3a_deconv_up
        self.illumination_unet3a_up_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # self.unet3a
        self.illumination_unet3a_2 = self._make_resblock(Bottleneck, 256, 128)
        # self.unet3b
        self.illumination_unet3b_2 = self._make_resblock(Bottleneck, 512, 8, expansion=2)
        # self.unet1a
        self.illumination_unet1a_up_2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        ## conv10 for output boundary
        self.conv10_depth = Conv_Stage3(42, [1, 1], output_map=True)
        self.conv10_normal = Conv_Stage3(42, [1, 1], output_map=True)
        self.conv10_reflectance = Conv_Stage(38, [38], output_map=True)
        self.conv10_illumination = Conv_Stage(38, [38], output_map=True)

        ## init param
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * 4
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_dilation_layer(self, block, planes, blocks, stride=1):
        dilation = 2
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, dilation_rate=dilation))
        self.inplanes = planes * 4
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_rate=dilation))

        return nn.Sequential(*layers)

    def _make_resblock(self, block, inplanes, planes, stride=1, expansion=4):
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )

        return block(inplanes, planes, stride, downsample,expansion=expansion)

    def load_resnet(self,model_path):
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()

        ignore_keys = ['fc.weight', 'fc.bias']
        model_dict = self.state_dict()

        for k, v in list(pretrained_dict.items()):
            if k in ignore_keys:
                pretrained_dict.pop(k)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        ## when x: (1, 3, 224, 224)
        ## resnet-50
        xf_1 = self.conv1(x)
        xf_1 = self.bn1(xf_1)
        xf_1_1 = self.relu(xf_1)        # (1, 64, 160, 160)
        xf_1 = self.maxpool(xf_1_1)     # (1, 64, 80, 80)
        xf_2 = self.layer1(xf_1)        # (1, 256, 80, 80)
        xf_3 = self.layer2(xf_2)        # (1, 512, 40, 40)
        xf_4 = self.layer3(xf_3)        # (1, 1024, 20, 20)
        res5_output = self.layer4(xf_4) # (1, 2048, 20, 20)

        ## res1-5 up
        res1_up = self.res1_up(xf_1_1)  # (1, 2, 322, 322)
        res2_up = self.res2_up(xf_2)    # (1, 2, 323, 323)
        res3_up = self.res3_up(xf_3)    # (1, 2, 328, 328)
        res4_up = self.res4_up(xf_4)    # (1, 2, 336, 336)
        res5_up = self.res5_up(res5_output)  # (1, 2, 336, 336)
        res1_up_crop = F.interpolate(res1_up, size=x.size()[2:], mode='bilinear', align_corners=True)  # (1, 2, 320, 320)
        res2_up_crop = F.interpolate(res2_up, size=x.size()[2:], mode='bilinear', align_corners=True)  # (1, 2, 320, 320)
        res3_up_crop = F.interpolate(res3_up, size=x.size()[2:], mode='bilinear',  align_corners=True)  # (1, 2, 320, 320)
        res4_up_crop = F.interpolate(res4_up, size=x.size()[2:], mode='bilinear', align_corners=True)  # (1, 2, 320, 320)
        res5_up_crop = F.interpolate(res5_up, size=x.size()[2:], mode='bilinear', align_corners=True)  # (1, 2, 320, 320)

        # extra branch conv1
        xf_1_b = self.conv1_b(x)  # (1, 16, 320, 320)
        xf_2_b = self.conv2_b(x)  # (1, 16, 320, 320)
        unet1 = torch.add(xf_1_b, xf_2_b)  # (1, 16, 320, 320)
        unet1 = self.soft_boundary(unet1)
        boundary_soft = torch.softmax(unet1,1)

        ### branch for depth & normal
        xf_6 = self.conv6(res5_output)  # (1, 256, 20, 20)
        ## branch1 for depth
        depth_res5c_up1 = self.depth_res5c_up1(xf_6)  # (1, 16, 83, 83)
        crop_h, crop_w = xf_2.size(2), xf_2.size(3)
        depth_res5c_crop = depth_res5c_up1[:, :, 3:3 + crop_h, 3:3 + crop_w]  # (1, 16, 80, 80)
        depth_res5c_up2 = self.depth_res5c_up2(depth_res5c_crop)  # (1, 16, 323, 323)
        crop_h, crop_w = x.size(2), x.size(3)
        depth_res5c_crop2 = depth_res5c_up2[:, :, 3:3 + crop_h, 3:3 + crop_w]  # (1, 16, 320, 320)
        ## branch2 for depth
        normal_res5c_up1 = self.normal_res5c_up1(xf_6)  # (1, 16, 83, 83)
        crop_h, crop_w = xf_2.size(2), xf_2.size(3)
        normal_res5c_crop = normal_res5c_up1[:, :, 3:3 + crop_h, 3:3 + crop_w]  # (1, 16, 80, 80)
        normal_res5c_up2 = self.normal_res5c_up2(normal_res5c_crop)  # (1, 16, 323, 323)
        crop_h, crop_w = x.size(2), x.size(3)
        normal_res5c_crop2 = normal_res5c_up2[:, :, 3:3 + crop_h, 3:3 + crop_w]  # (1, 16, 320, 320)
        ## branch3 for depth & normal
        unet3a_up = self.unet3a_up(xf_6)  # (1, 256, 83, 83)
        crop_h, crop_w = xf_2.size(2), xf_2.size(3)
        unet3a_up_crop = unet3a_up[:, :, 3:3 + crop_h, 3:3 + crop_w]  # (1, 256, 80, 80)
        xf_concat1 = torch.cat([unet3a_up_crop, xf_2], dim=1)  # (1, 512, 80, 80)
        unet3a = self.unet3a(xf_concat1)  # (1, 512, 80, 80)
        unet3b = self.unet3b(unet3a)  # (1, 16, 80, 80)
        unet1a_up = self.unet1a_up(unet3b)  # (1, 16, 323, 323)
        crop_h, crop_w = xf_1_b.size(2), xf_1_b.size(3)  # 320,320
        unet1a_up_crop = unet1a_up[:, :, 1:1 + crop_h, 1:1 + crop_w]  # [1, 16, 320, 320]


        ###branch for reflectance
        reflectance_res3_up1=self.reflectance_res3_up1(xf_3)    #[1, 2, 82, 82]
        crop_h, crop_w = xf_2.size(2), xf_2.size(3)
        reflectance_res3_up1_crop = reflectance_res3_up1[:, :, 2:2 + crop_h, 2:2 + crop_w]  #[1, 512, 80, 80]
        reflectance_res123 = torch.cat([xf_1,xf_2,reflectance_res3_up1_crop],1)     #[1, 832, 80, 80]
        reflectance_xf_7 = self.reflectance_conv7(reflectance_res123)   #[1, 256, 80, 80]
        reflectance_res5 = self.reflectance_res5(res5_output)           #[1, 256, 83, 83]
        reflectance_res5_crop = reflectance_res5[:, :, 3:3 + crop_h, 3:3 + crop_w]
        reflectance_weight = self.reflectance_weight(reflectance_res5_crop)
        reflectance_xf_7 = torch.mul(reflectance_xf_7, reflectance_weight)
        ## branch1 for reflectance
        reflectance_res123_up1 = self.reflectance_res123_up1(reflectance_xf_7)   # [1, 16, 162, 162]
        crop_h, crop_w = xf_1_1.size(2), xf_1_1.size(3)
        reflectance_res123_up1_crop = reflectance_res123_up1[:, :, 2:2 + crop_h, 2:2 + crop_w]  # (1, 16, 160, 160)
        reflectance_res123_up2 = self.reflectance_res123_up2(reflectance_res123_up1_crop)  # [1, 16, 322, 322]
        crop_h, crop_w = x.size(2), x.size(3)
        reflectance_res123_up2_crop2 = reflectance_res123_up2[:, :, 2:2 + crop_h, 2:2 + crop_w]  # (1, 16, 320, 320)
        ## branch2 for reflectance
        reflectance_unet3a_up_2 = self.reflectance_unet3a_up_2(reflectance_xf_7)  # (1, 256, 162, 162)
        crop_h, crop_w = xf_1_1.size(2), xf_1_1.size(3)
        reflectance_unet3a_up_2_crop = reflectance_unet3a_up_2[:, :, 2:2 + crop_h,2:2 + crop_w]  # (1, 256, 160, 160)
        reflectance_unet3a_2 = self.reflectance_unet3a_2(reflectance_unet3a_up_2_crop)  # (1, 512, 160, 160)
        reflectance_unet3b_2 = self.reflectance_unet3b_2(reflectance_unet3a_2)  # (1, 16, 160, 160)
        reflectance_unet1a_up_2 = self.reflectance_unet1a_up_2(reflectance_unet3b_2)  # (1, 16, 322, 322)
        crop_h, crop_w = xf_1_b.size(2), xf_1_b.size(3)  # 320,320
        reflectance_unet1a_up_2_crop = reflectance_unet1a_up_2[:, :, 1:1 + crop_h, 1:1 + crop_w]  # [1, 16, 320, 320]

        ###branch for illumination
        illumination_res3_up1 = self.illumination_res3_up1(xf_3)  # [1, 2, 82, 82]
        crop_h, crop_w = xf_2.size(2), xf_2.size(3)
        illumination_res3_up1_crop = illumination_res3_up1[:, :, 2:2 + crop_h, 2:2 + crop_w]  # [1, 512, 80, 80]
        res123 = torch.cat([xf_1, xf_2, illumination_res3_up1_crop], 1)  # [1, 2880, 80, 80]
        illumination_xf_7 = self.illumination_conv7(res123)  # [1, 256, 80, 80]
        illumination_res5 = self.illumination_res5(res5_output)  # [1, 256, 83, 83]
        illumination_res5_crop = illumination_res5[:, :, 3:3 + crop_h, 3:3 + crop_w]
        illumination_weight = self.illumination_weight(illumination_res5_crop)
        illumination_xf_7 = torch.mul(illumination_xf_7, illumination_weight)
        ## branch1 for illumination
        illumination_res123_up1 = self.illumination_res123_up1(illumination_xf_7)  # [1, 16, 162, 162]
        crop_h, crop_w = xf_1_1.size(2), xf_1_1.size(3)
        illumination_res123_up1_crop = illumination_res123_up1[:, :, 2:2 + crop_h, 2:2 + crop_w]  # (1, 16, 160, 160)
        illumination_res123_up2 = self.illumination_res123_up2(illumination_res123_up1_crop)  # [1, 16, 322, 322]
        crop_h, crop_w = x.size(2), x.size(3)
        illumination_res123_up2_crop2 = illumination_res123_up2[:, :, 2:2 + crop_h, 2:2 + crop_w]  # (1, 16, 320, 320)
        ## branch2 for illumination
        illumination_unet3a_up_2 = self.illumination_unet3a_up_2(illumination_xf_7)  # (1, 256, 162, 162)
        crop_h, crop_w = xf_1_1.size(2), xf_1_1.size(3)
        illumination_unet3a_up_2_crop = illumination_unet3a_up_2[:, :, 2:2 + crop_h, 2:2 + crop_w]  # (1, 256, 160, 160)
        illumination_unet3a_2 = self.illumination_unet3a_2(illumination_unet3a_up_2_crop)  # (1, 512, 160, 160)
        illumination_unet3b_2 = self.illumination_unet3b_2(illumination_unet3a_2)  # (1, 16, 160, 160)
        illumination_unet1a_up_2 = self.illumination_unet1a_up_2(illumination_unet3b_2)  # (1, 16, 322, 322)
        crop_h, crop_w = xf_1_b.size(2), xf_1_b.size(3)  # 320,320
        illumination_unet1a_up_2_crop = illumination_unet1a_up_2[:, :, 1:1 + crop_h, 1:1 + crop_w]  # [1, 16, 320, 320]

        # feature for depth & normal
        #[1, 58, 320, 320]
        xf_concat_d = torch.cat([res1_up_crop, res2_up_crop, res3_up_crop, res4_up_crop, res5_up_crop, depth_res5c_crop2, unet1a_up_crop], 1) #[1, 42, 320, 320]
        xf_concat_n = torch.cat([res1_up_crop, res2_up_crop, res3_up_crop, res4_up_crop, res5_up_crop, normal_res5c_crop2, unet1a_up_crop], 1)  # [1, 42, 320, 320]
        # feature for reflectance & illumination
        #[1, 54, 320, 320]
        xf_concat_r = torch.cat([res1_up_crop, res2_up_crop, res3_up_crop, reflectance_res123_up2_crop2, reflectance_unet1a_up_2_crop], 1)  # [1, 38, 320, 320]
        xf_concat_i = torch.cat([res1_up_crop, res2_up_crop, res3_up_crop, illumination_res123_up2_crop2, illumination_unet1a_up_2_crop], 1)  # [1, 38, 320, 320]

        out_depth = self.conv10_depth(xf_concat_d) #[1, 1, 320, 320]
        out_normal = self.conv10_normal(xf_concat_n)  # [1, 1, 320, 320]
        out_reflectance = self.conv10_reflectance(xf_concat_r)  # [1, 1, 320, 320]
        out_illumination= self.conv10_illumination(xf_concat_i)  # [1, 1, 320, 320]

        out_depth = out_depth * (1.0 + boundary_soft[:, 1, :, :].unsqueeze(1))
        out_normal = out_normal * (1.0 + boundary_soft[:, 2, :, :].unsqueeze(1))
        out_reflectance = out_reflectance * (1.0 + boundary_soft[:, 3, :, :].unsqueeze(1))
        out_illumination = out_illumination * (1.0 + boundary_soft[:, 4, :, :].unsqueeze(1))

        out_depth = torch.sigmoid(out_depth)
        out_normal = torch.sigmoid(out_normal)
        out_reflectance = torch.sigmoid(out_reflectance)
        out_illumination = torch.sigmoid(out_illumination)

        return unet1,out_depth,out_normal,out_reflectance,out_illumination

if __name__ == '__main__':
    model = MyNet()
    dummy_input = torch.rand(1, 3, 320, 320)
    output = model(dummy_input)
    for out in output:
        print(out.size())