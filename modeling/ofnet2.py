import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

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

class OFNet(nn.Module):

    def __init__(self):
        self.inplanes = 64
        super(OFNet, self).__init__()

        ## resnet-50 part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3) ##256
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2) ## 512
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2) ## 1024
        self.layer4 = self._make_dilation_layer(Bottleneck, 512, 3) ## 2048  add dilation conv in res-stage 5

        self.res1_1 = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.res2c_1 = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2, 2, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.res3d_1 = nn.Sequential(
            nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2, 2, kernel_size=16, stride=8, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )

        self.conv6 = Conv_Stage(2048,[256,256], bias=False)
        # res5c_1 res5c_up1
        self.res5c_1 = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.res5c_up2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # unet3a_deconv_up
        self.conv3_b = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        ## conv1 for boundary
        self.conv1_b = Conv_Stage(3, [8, 4, 16])
        self.conv2_b = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.inplanes = 512
        #self.unet3a
        self.layer8 = self._make_resblock(Bottleneck, 512, 128)
        #self.unet3b
        self.layer9 = self._make_resblock(Bottleneck, 512, 8, expansion=2)

        #self.unet1a
        self.deconv9 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        ## conv10 for output boundary
        self.conv10_depth = Conv_Stage(54, [8, 8, 8, 8, 4], output_map=True)
        self.conv10_normal = Conv_Stage(54, [8, 8, 8, 8, 4], output_map=True)
        self.conv10_reflectance = Conv_Stage(54, [8, 8, 8, 8, 4], output_map=True)
        self.conv10_illumination = Conv_Stage(54, [8, 8, 8, 8, 4], output_map=True)

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

        ## extra branch-1
        res1 = self.res1_1(xf_1_1)  # (1, 2, 322, 322)
        res2c = self.res2c_1(xf_2)  # (1, 2, 323, 323)
        res3d = self.res3d_1(xf_3)  # (1, 2, 328, 328)
        crop_h, crop_w = x.size(2), x.size(3)
        res1_crop = res1[:, :, 0: crop_h, 0: crop_w]    # (1, 2, 320, 320)
        res2c_crop = res2c[:, :, 0: crop_h, 0: crop_w]  # (1, 2, 320, 320)
        res3d_crop = res3d[:, :, 0: crop_h, 0: crop_w]  # (1, 2, 320, 320)


        ## extra branch-4
        xf_1_b = self.conv1_b(x)            # (1, 16, 320, 320)
        xf_2_b = self.conv2_b(x)            # (1, 16, 320, 320)
        unet1 = torch.add(xf_1_b, xf_2_b)   # (1, 16, 320, 320)

        ## main branch
        xf_6 = self.conv6(res5_output) #(1, 256, 20, 20)
        ## main branch-2
        res5c = self.res5c_1(xf_6)     #(1, 16, 83, 83)
        crop_h, crop_w = xf_2.size(2), xf_2.size(3)
        res5c_crop = res5c[:,:,3:3+crop_h,3:3+crop_w]  #(1, 16, 80, 80)
        res5c = self.res5c_up2(res5c_crop)  #(1, 16, 323, 323)
        crop_h, crop_w = x.size(2), x.size(3)
        res5c_crop2 = res5c[:,:,3:3+crop_h,3:3+crop_w] #(1, 16, 320, 320)

        ## main branch-3
        xf_7 = self.conv3_b(xf_6)      #(1, 256, 83, 83)

        crop_h,crop_w = xf_2.size(2),xf_2.size(3)
        xf_7_crop = xf_7[:,:,3:3+crop_h,3:3+crop_w]     #(1, 256, 80, 80)
        xf_concat1 = torch.cat([xf_7_crop,xf_2],dim=1)  #(1, 512, 80, 80)

        xf_8_1 = self.layer8(xf_concat1) # (1, 512, 80, 80)
        xf_8_2 = self.layer9(xf_8_1) # (1, 16, 80, 80)
        xf_9 = self.deconv9(xf_8_2) # (1, 16, 323, 323)

        crop_h,crop_w = xf_1_b.size(2),xf_1_b.size(3) #320,320
        xf_9_crop = xf_9[:,:,1:1+crop_h,1:1+crop_w] #[1, 16, 320, 320]
        xf_concat_b = torch.cat([unet1,res1_crop,res2c_crop,res3d_crop,res5c_crop2,xf_9_crop],1) #[1, 54, 320, 320]

        out_depth = self.conv10_depth(xf_concat_b) #[1, 4, 320, 320]
        out_normal = self.conv10_normal(xf_concat_b)  # [1, 4, 320, 320]
        out_reflectance = self.conv10_reflectance(xf_concat_b)  # [1, 4, 320, 320]
        out_illumination= self.conv10_illumination(xf_concat_b)  # [1, 4, 320, 320]

        out_depth = torch.sigmoid(out_depth)
        out_normal = torch.sigmoid(out_normal)
        out_reflectance = torch.sigmoid(out_reflectance)
        out_illumination = torch.sigmoid(out_illumination)

        return torch.cat([out_depth, out_normal, out_reflectance, out_illumination], 1);

if __name__ == '__main__':
    model = OFNet()
    print(model)
    dummy_input = torch.rand(1, 3, 320, 320)
    output = model(dummy_input)
    for out in output:
        print(out.size())

