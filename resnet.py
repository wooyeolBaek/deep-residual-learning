import torch
import torch.nn as nn
from importlib import import_module

from blocks import ConvBN

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

class ResNet(nn.Module):
    

    def __init__(self, nblocks, block_name, in_channels=3, num_classes=10, mapping='B'):
        super(ResNet, self).__init__()

        # the number of channels of the first conv layer
        nker = 64 if len(nblocks) == 4 else 16

        # import the block
        resblock = getattr(import_module('blocks'), block_name)

        # --conv1
        # imagenet: size:224,224 -> 112,112 channels: 3 -> 64
        # cifar10: size:32,32 -> 32,32 channels: 3 -> 16
        self.conv1 = ConvBN(
            in_channels=in_channels,
            out_channels=nker,
            kernel_size=7 if len(nblocks) == 4 else 3,
            stride=2 if len(nblocks) == 4 else 1,
            padding=3 if len(nblocks) == 4 else 1,
        )

        # --conv2_x
        # imagenet: size:112,112 -> 56,56 channels: 64 -> 64
        # cifar10: size:32,32 -> 32,32 channels: 16 -> 16
        self.conv2_x = [nn.MaxPool2d(kernel_size=3,stride=2,padding=1)]
        self.conv2_x += [resblock(in_channels=nker,out_channels=nker,mapping=mapping) for _ in range(nblocks[0])]
        self.conv2_x = nn.Sequential(*self.conv2_x)

        # --conv3_x
        # imagenet: size:56,56 -> 28,28 channels: 64 -> 128
        # cifar10: size:32,32 -> 16,16 channels: 16 -> 32
        self.conv3_x = [resblock(in_channels=nker,out_channels=2*nker,mapping=mapping)]
        self.conv3_x += [resblock(in_channels=2*nker,out_channels=2*nker,mapping=mapping) for _ in range(nblocks[1]-1)]
        self.conv3_x = nn.Sequential(*self.conv3_x)

        # --conv4_x
        # imagenet: size:28,28 -> 14,14 channels: 128 -> 256
        # cifar10: size:16,16 -> 8,8 channels: 32 -> 64
        self.conv4_x = [resblock(in_channels=2*nker,out_channels=4*nker,mapping=mapping)]
        self.conv4_x += [resblock(in_channels=4*nker,out_channels=4*nker,mapping=mapping) for _ in range(nblocks[2]-1)]
        self.conv4_x = nn.Sequential(*self.conv4_x)

        # --conv5_x
        # imagenet: size:14,14 -> 7,7 channels: 256 -> 512
        # cifar10: pass
        self.conv5_x = None
        if len(nblocks) == 4:
            self.conv5_x = [resblock(in_channels=4*nker,out_channels=8*nker,mapping=mapping)]
            self.conv5_x += [resblock(in_channels=8*nker,out_channels=8*nker,mapping=mapping) for _ in range(nblocks[3]-1)]
            self.conv5_x = nn.Sequential(*self.conv5_x)

        # --fully-connected layer
        # imagnet: size:7,7 -> 1,1 channels: 512
        # cifar10: size:8,8 -> 1,1 channels: 64
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=8*nker if len(nblocks) == 4 else 4*nker, out_features=num_classes)

        self.initialize_weight()
    
    def initialize_weight(self):
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
        x = self.conv1(x)
        
        x = self.conv2_x(x)

        x = self.conv3_x(x)

        x = self.conv4_x(x)

        if not self.conv5_x is None:
            x = self.conv5_x(x)
        
        x = self.avg_pooling(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)

        return out

if __name__=="__main__":
    batch_size = 16
    x = torch.randn((batch_size,3,32,32))

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