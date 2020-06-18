import math

import torch
import torch.nn as nn
from torch.nn import BatchNorm2d

vgg_config = {
    'vgg-11': [16, 'P', 32, 'P', 64, 64, 'P', 128, 128, 'P', 128, 128, 'P']
}


class ModelSetup:
    
    """ Interface for connecting model setups and main codes """
    
    def __init__(self, n_classes, model_name):
        
        self.n_classes = n_classes
        self.model_name = model_name
    
    def get_model(self):
        
        if self.model_name == 'vgg-11':
            return VGG(self.model_name, self.n_classes)
        elif self.model_name == 'WideResnet':
            return WideResnet(self.n_classes)
        elif self.model_name == 'Lenet':
            return LeNet(self.n_classes)
        else:
            raise NotImplementedError("This model is not implemented.")

   
      
class VGG(nn.Module):
    
    def __init__(self, spec, output_dim):
        super().__init__()
        self.features = self._feature_layer(vgg_config[spec])
        self.classifier = nn.Sequential(
            nn.Linear(128, output_dim)
        )  
        
    def forward(self, x):
    
        out = self.features(x)
        out = out.reshape(out.shape[0], -1)
        out = self.classifier(out)
        return out
      
    def _feature_layer(self, config):
        
        layers = []
        in_channels = 3
        
        for c in config:
            if c == 'P':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(c),
                           nn.ReLU(inplace=True)]
                in_channels = c
                
        return nn.Sequential(*layers)
    

class WideResnet(nn.Module):
    '''
    for wide-resnet-28-10, the definition should be WideResnet(n_classes, 10, 28)
    '''

    def __init__(self, n_classes, k=1, n=28):
        super(WideResnet, self).__init__()
        self.n_layers, self.k = n, k
        self.backbone = WideResnetBackbone(k=k, n=n)
        self.classifier = nn.Linear(64 * self.k, n_classes, bias=True)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = torch.mean(feat, dim=(2, 3))
        feat = self.classifier(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.classifier.weight)
        if not self.classifier.bias is None:
            nn.init.constant_(self.classifier.bias, 0)
            

class WideResnetBackbone(nn.Module):
    def __init__(self, k=1, n=28, drop_rate=0):
        super(WideResnetBackbone, self).__init__()
        self.k, self.n = k, n
        assert (self.n - 4) % 6 == 0
        n_blocks = (self.n - 4) // 6
        n_layers = [16, ] + [self.k * 16 * (2 ** i) for i in range(3)]

        self.conv1 = nn.Conv2d(
            3,
            n_layers[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.layer1 = self.create_layer(
            n_layers[0],
            n_layers[1],
            bnum=n_blocks,
            stride=1,
            drop_rate=drop_rate,
            pre_res_act=True,
        )
        self.layer2 = self.create_layer(
            n_layers[1],
            n_layers[2],
            bnum=n_blocks,
            stride=2,
            drop_rate=drop_rate,
            pre_res_act=False,
        )
        self.layer3 = self.create_layer(
            n_layers[2],
            n_layers[3],
            bnum=n_blocks,
            stride=2,
            drop_rate=drop_rate,
            pre_res_act=False,
        )
        self.bn_last = BatchNorm2d(n_layers[3], momentum=0.001)
        self.relu_last = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.init_weight()

    def create_layer(
            self,
            in_chan,
            out_chan,
            bnum,
            stride=1,
            drop_rate=0,
            pre_res_act=False,
    ):
        layers = [
            BasicBlockPreAct(
                in_chan,
                out_chan,
                drop_rate=drop_rate,
                stride=stride,
                pre_res_act=pre_res_act), ]
        for _ in range(bnum - 1):
            layers.append(
                BasicBlockPreAct(
                    out_chan,
                    out_chan,
                    drop_rate=drop_rate,
                    stride=1,
                    pre_res_act=False, ))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.conv1(x)

        feat = self.layer1(feat)
        feat2 = self.layer2(feat)  # 1/2
        feat4 = self.layer3(feat2)  # 1/4

        feat4 = self.bn_last(feat4)
        feat4 = self.relu_last(feat4)
        return feat2, feat4

    def init_weight(self):
        # for _, child in self.named_children():
        #     if isinstance(child, nn.Conv2d):
        #         n = child.kernel_size[0] * child.kernel_size[0] * child.out_channels
        #         nn.init.normal_(child.weight, 0, 1. / ((0.5 * n) ** 0.5))
        #         #  nn.init.kaiming_normal_(
        #         #      child.weight, a=0.1, mode='fan_out',
        #         #      nonlinearity='leaky_relu'
        #         #  )
        #
        #         if child.bias is not None:
        #             nn.init.constant_(child.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    

class BasicBlockPreAct(nn.Module):
    def __init__(self, in_chan, out_chan, drop_rate=0, stride=1, pre_res_act=False):
        super(BasicBlockPreAct, self).__init__()
        self.bn1 = BatchNorm2d(in_chan, momentum=0.001)
        self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_chan, momentum=0.001)
        self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.dropout = nn.Dropout(drop_rate) if not drop_rate == 0 else None
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=stride, bias=False
            )
        self.pre_res_act = pre_res_act
        # self.init_weight()

    def forward(self, x):
        bn1 = self.bn1(x)
        act1 = self.relu1(bn1)
        residual = self.conv1(act1)
        residual = self.bn2(residual)
        residual = self.relu2(residual)
        if self.dropout is not None:
            residual = self.dropout(residual)
        residual = self.conv2(residual)

        shortcut = act1 if self.pre_res_act else x
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        out = shortcut + residual
        return out

    def init_weight(self):
        # for _, md in self.named_modules():
        #     if isinstance(md, nn.Conv2d):
        #         nn.init.kaiming_normal_(
        #             md.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        #         if md.bias is not None:
        #             nn.init.constant_(md.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)   
