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

def Conv_Stage(input_dim,dim_list,bias=True, output_map=False):
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

class DoobNet(nn.Module):

    def __init__(self):
        self.inplanes = 64
        super(DoobNet, self).__init__()

        ## resnet-50 part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3) ##256
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2) ## 512
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2) ## 1024
        self.layer4 = self._make_dilation_layer(Bottleneck, 512, 3) ## 2048  add dilation conv in res-stage 5

        self.conv6 = Conv_Stage(2048,[256,256], bias=False)
        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(256,256,kernel_size=7,stride=4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.inplanes = 512
        self.layer8 = self._make_resblock(Bottleneck, 512, 128)             #bias=False
        self.layer9 = self._make_resblock(Bottleneck, 512, 8, expansion=2)  #bias=False

        self.deconv9 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=4, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        ## conv1 for boundary/orientation
        self.conv1_b = Conv_Stage(3, [8, 4, 16])
        #self.conv1_o = Conv_Stage(3, [8, 4, 16])

        ## conv10 for output boundary/orientation map
        self.conv10_depth = Conv_Stage(32, [8, 8, 8, 8, 4], output_map=True)
        self.conv10_normal = Conv_Stage(32, [8, 8, 8, 8, 4], output_map=True)
        self.conv10_reflectance = Conv_Stage(32, [8, 8, 8, 8, 4], output_map=True)
        self.conv10_illumination = Conv_Stage(32, [8, 8, 8, 8, 4], output_map=True)


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
        xf_1 = self.relu(xf_1)
        xf_1 = self.maxpool(xf_1) # (1, 64, 56, 56)
        xf_2 = self.layer1(xf_1) # (1, 256, 56, 56)
        xf_3 = self.layer2(xf_2) # (1, 512, 28, 28)
        xf_4 = self.layer3(xf_3) # (1, 1024, 14, 14)
        res5_output = self.layer4(xf_4) # (1, 2048, 14, 14)

        ## extra branch
        xf_1_b = self.conv1_b(x)
        #xf_1_o = self.conv1_o(x) #(1, 16, 224, 224)

        ## main branch
        xf_6 = self.conv6(res5_output) #(1, 256, 14, 14)
        xf_7 = self.deconv7(xf_6) #(1, 256, 59, 59)

        crop_h,crop_w = xf_2.size(2),xf_2.size(3)
        xf_7_crop = xf_7[:,:,3:3+crop_h,3:3+crop_w]
        xf_concat1 = torch.cat([xf_7_crop,xf_2],dim=1)

        xf_8_1 = self.layer8(xf_concat1) # (1, 512, 56, 56)
        xf_8_2 = self.layer9(xf_8_1) # (1, 16, 56, 56)
        xf_9 = self.deconv9(xf_8_2) # (1, 16, 227, 227)

        crop_h,crop_w = xf_1_b.size(2),xf_1_b.size(3) #224,224
        xf_9_crop = xf_9[:,:,1:1+crop_h,1:1+crop_w] #[1, 16, 224, 224]
        xf_concat_b = torch.cat([xf_9_crop,xf_1_b],1) #[1, 32, 224, 224]
        #xf_concat_o = torch.cat([xf_9_crop,xf_1_o],1)

        out_depth = self.conv10_depth(xf_concat_b)
        out_normal = self.conv10_normal(xf_concat_b)
        out_reflectance = self.conv10_reflectance(xf_concat_b)
        out_illumination = self.conv10_illumination(xf_concat_b)

        out_depth = torch.sigmoid(out_depth)
        out_normal = torch.sigmoid(out_normal)
        out_reflectance = torch.sigmoid(out_reflectance)
        out_illumination = torch.sigmoid(out_illumination)

        return torch.cat([out_depth,out_normal,out_reflectance,out_illumination],dim=1)

if __name__ == '__main__':
    model = DoobNet()
    print(model)
    # model.load_resnet('/home/yuzhe/resnet50-19c8e357.pth')
    dummy_input = torch.rand(1, 3, 320, 320)
    output = model(dummy_input)
    for out in output:
        print(out)

    # print model.conv10_b

