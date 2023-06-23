import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module


architectures = {
    # imagenet
    18:(2,2,2,2),
    34:(3,4,6,3),
    50:(3,4,6,3),
    101:(3,4,23,3),
    152:(3,8,36,3),
    # cifar10
    20:(3,3,3),
    32:(5,5,5),
    44:(7,7,7),
    56:(9,9,9),
    110:(18,18,18),
    1202:(200,200,200)
}

# imagenet
def resnet18(num_classes, mapping='B', block_name="ResBlock", **kwargs):
    return ResNet(
        nblocks=architectures[18],
        block_name=block_name,
        num_classes=num_classes,
        mapping=mapping,
    )
def resnet34(num_classes, mapping='B', block_name="ResBlock", **kwargs):
    return ResNet(
        nblocks=architectures[34],
        block_name=block_name,
        num_classes=num_classes,
        mapping=mapping,
    )
def resnet50(num_classes, mapping='B', block_name="BottleneckBlock", **kwargs):
    return ResNet(
        nblocks=architectures[50],
        block_name=block_name,
        num_classes=num_classes,
        mapping=mapping,
    )
def resnet101(num_classes, mapping='B', block_name="BottleneckBlock", **kwargs):
    return ResNet(
        nblocks=architectures[101],
        block_name=block_name,
        num_classes=num_classes,
        mapping=mapping,
    )
def resnet152(num_classes, mapping='B', block_name="BottleneckBlock", **kwargs):
    return ResNet(
        nblocks=architectures[152],
        block_name=block_name,
        num_classes=num_classes,
        mapping=mapping,
    )

# cifar10
def resnet20(num_classes, mapping='B', block_name="ResBlock", **kwargs):
    return ResNet(
        nblocks=architectures[20],
        block_name=block_name,
        num_classes=num_classes,
        mapping=mapping,
    )
def resnet32(num_classes, mapping='B', block_name="ResBlock", **kwargs):
    return ResNet(
        nblocks=architectures[32],
        block_name=block_name,
        num_classes=num_classes,
        mapping=mapping,
    )
def resnet44(num_classes, mapping='B', block_name="ResBlock", **kwargs):
    return ResNet(
        nblocks=architectures[44],
        block_name=block_name,
        num_classes=num_classes,
        mapping=mapping,
    )
def resnet56(num_classes, mapping='B', block_name="ResBlock", **kwargs):
    return ResNet(
        nblocks=architectures[56],
        block_name=block_name,
        num_classes=num_classes,
        mapping=mapping,
    )
def resnet110(num_classes, mapping='B', block_name="ResBlock", **kwargs):
    return ResNet(
        nblocks=architectures[110],
        block_name=block_name,
        num_classes=num_classes,
        mapping=mapping,
    )
def resnet1202(num_classes, mapping='B', block_name="ResBlock", **kwargs):
    return ResNet(
        nblocks=architectures[1202],
        block_name=block_name,
        num_classes=num_classes,
        mapping=mapping,
    )


# cifar10 plain net
def plainnet20(num_classes, **kwargs):
    return ResNet(
        nblocks=architectures[20],
        block_name="PlainBlock",
        num_classes=num_classes,
    )
def plainnet32(num_classes, **kwargs):
    return ResNet(
        nblocks=architectures[32],
        block_name="PlainBlock",
        num_classes=num_classes,
    )
def plainnet44(num_classes, **kwargs):
    return ResNet(
        nblocks=architectures[44],
        block_name="PlainBlock",
        num_classes=num_classes,
    )
def plainnet56(num_classes, **kwargs):
    return ResNet(
        nblocks=architectures[56],
        block_name="PlainBlock",
        num_classes=num_classes,
    )
def plainnet110(num_classes, **kwargs):
    return ResNet(
        nblocks=architectures[110],
        block_name="PlainBlock",
        num_classes=num_classes,
    )
def plainnet1202(num_classes, **kwargs):
    return ResNet(
        nblocks=architectures[1202],
        block_name="PlainBlock",
        num_classes=num_classes,
    )


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv1, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=out_channels,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out
    

class Conv2(nn.Module):
    def __init__(self, nblocks, block_name, mapping, in_channels, out_channels):
        super(Conv2, self).__init__()
        resblock = getattr(import_module('models.blocks'), block_name)
        
        # imagenet 학습시에만 MaxPool2d 사용
        self.conv2_x = [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)] if len(nblocks) == 4 else []
        self.conv2_x += [resblock(in_channels=in_channels,out_channels=out_channels,mapping=mapping) for _ in range(nblocks[0])]
        self.conv2_x = nn.Sequential(*self.conv2_x)

    def forward(self, x):
        z = self.conv2_x(x)

        return z


class Conv3(nn.Module):
    def __init__(self, nblocks, block_name, mapping, in_channels, out_channels):
        super(Conv3, self).__init__()
        resblock = getattr(import_module('models.blocks'), block_name)

        conv3_x = [resblock(in_channels=in_channels,out_channels=out_channels,mapping=mapping)]
        conv3_x += [resblock(in_channels=out_channels,out_channels=out_channels,mapping=mapping) for _ in range(nblocks[1]-1)]
        self.conv3_x = nn.Sequential(*conv3_x)

    def forward(self, x):
        z = self.conv3_x(x)

        return z


class Conv4(nn.Module):
    def __init__(self, nblocks, block_name, mapping, in_channels, out_channels):
        super(Conv4, self).__init__()
        resblock = getattr(import_module('models.blocks'), block_name)

        self.conv4_x = [resblock(in_channels=in_channels,out_channels=out_channels,mapping=mapping)]
        self.conv4_x += [resblock(in_channels=out_channels,out_channels=out_channels,mapping=mapping) for _ in range(nblocks[2]-1)]
        self.conv4_x = nn.Sequential(*self.conv4_x)

    def forward(self, x):
        z = self.conv4_x(x)

        return z


class Conv5(nn.Module):
    def __init__(self, nblocks, block_name, mapping, in_channels, out_channels):
        super(Conv5, self).__init__()
        resblock = getattr(import_module('models.blocks'), block_name)

        self.conv5_x = [resblock(in_channels=in_channels,out_channels=out_channels,mapping=mapping)]
        self.conv5_x += [resblock(in_channels=out_channels,out_channels=out_channels,mapping=mapping) for _ in range(nblocks[3]-1)]
        self.conv5_x = nn.Sequential(*self.conv5_x)

    def forward(self, x):
        z = self.conv5_x(x)

        return z

class FC(nn.Module):
    def __init__(self, in_features, out_features):
        super(FC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features,
        )

    def forward(self, x):
        z = self.avg_pool(x)
        z = z.view(z.shape[0], -1)
        z = self.fc(z)

        return z


class ResNet(nn.Module):
    def __init__(self, nblocks, block_name, in_channels=3, num_classes=10, mapping='B'):
        super(ResNet, self).__init__()

        # the number of channels of the first conv layer
        nker = 64 if len(nblocks) == 4 else 16

        # --conv1
        # cifar10: 32x32, 3 -> 32x32, 16
        # imagenet: size:224,224 -> 112,112 channels: 3 -> 64
        self.conv1 = Conv1(
            in_channels=in_channels,
            out_channels=nker,
            kernel_size=7 if len(nblocks) == 4 else 3,
            stride=2 if len(nblocks) == 4 else 1,
            padding=3 if len(nblocks) == 4 else 1,
        )
        self.add_module('conv1', self.conv1)
        

        # --conv2_x
        # cifar10: 32x32, 16 -> 32x32, 16
        # imagenet: size:112,112 -> 56,56 channels: 64 -> 64
        self.conv2_x = Conv2(
            nblocks=nblocks,
            block_name=block_name,
            mapping=mapping,
            in_channels=nker,
            out_channels=nker,
        )
        self.add_module('conv2_x', self.conv2_x)

        # --conv3_x
        # cifar10: 32x32, 16 -> 16x16, 32
        # imagenet: size:56,56 -> 28,28 channels: 64 -> 128
        self.conv3_x = Conv3(
            nblocks=nblocks,
            block_name=block_name,
            mapping=mapping,
            in_channels=nker,
            out_channels=2*nker,
        )
        self.add_module('conv3_x', self.conv3_x)
        nker *= 2

        # --conv4_x
        # cifar10: 16x16, 32 -> 8x8, 64
        # imagenet: size:28,28 -> 14,14 channels: 128 -> 256
        self.conv4_x = Conv4(
            nblocks=nblocks,
            block_name=block_name,
            mapping=mapping,
            in_channels=nker,
            out_channels=2*nker,
        )
        self.add_module('conv4_x', self.conv4_x)
        nker *= 2

        # --conv5_x
        # cifar10: pass
        # imagenet: size:14,14 -> 7,7 channels: 256 -> 512
        self.conv5_x = None
        if len(nblocks) == 4:
            self.conv5_x = Conv5(
                nblocks=nblocks,
                block_name=block_name,
                mapping=mapping,
                in_channels=nker,
                out_channels=2*nker,
            )
            self.add_module('conv5_x', self.conv5_x)
            nker *= 2

        # --fc
        self.fc = FC(
            in_features=nker,
            out_features=num_classes,
        )
        self.add_module('fc', self.fc)

        self.initialize_weight()
    
    def initialize_weight(self):
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2_x(z)
        z = self.conv3_x(z)
        z = self.conv4_x(z)
        if not self.conv5_x is None:
            z = self.conv5_x(z)
        z = self.fc(z)

        return z

if __name__=="__main__":
    batch_size = 16
    x = torch.randn((batch_size,3,32,32))
    print("input shape:", x.shape)

    # model test
    for model_num in architectures.keys():
        model_name = "resnet" + str(model_num)
        
        print(model_name, "Test", end=' : ')
        
        model = locals()[model_name](num_classes=10, mapping='B', block_name="ResBlock")
        out = model(x)

        if list(out.shape) != [16,10]:
            print(model_name, out.shape, end=' : ')
            print(f'Output type: {type(out)}')
        else:
            print("passed")