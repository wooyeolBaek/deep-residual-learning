import torch
import torch.nn as nn
from blocks import ConvBN, PlainBlock

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
def plain18(num_classes, **kwargs):
    return PlainNet(
        nblocks=architectures[18],
        num_classes=num_classes,
    )
def plain34(num_classes, **kwargs):
    return PlainNet(
        nblocks=architectures[34],
        num_classes=num_classes,
    )
def plain50(num_classes, mapping='B', block_name="BottleneckBlock", **kwargs):
    return PlainNet(
        nblocks=architectures[50],
        num_classes=num_classes,
    )
def plain101(num_classes, mapping='B', block_name="BottleneckBlock", **kwargs):
    return PlainNet(
        nblocks=architectures[101],
        num_classes=num_classes,
    )
def plain152(num_classes, mapping='B', block_name="BottleneckBlock", **kwargs):
    return PlainNet(
        nblocks=architectures[152],
        num_classes=num_classes,
    )

# cifar10
def plain20(num_classes, **kwargs):
    return PlainNet(
        nblocks=architectures[20],
        num_classes=num_classes,
    )
def plain32(num_classes, **kwargs):
    return PlainNet(
        nblocks=architectures[32],
        num_classes=num_classes,
    )
def plain44(num_classes, **kwargs):
    return PlainNet(
        nblocks=architectures[44],
        num_classes=num_classes,
    )
def plain56(num_classes, **kwargs):
    return PlainNet(
        nblocks=architectures[56],
        num_classes=num_classes,
    )
def plain110(num_classes, **kwargs):
    return PlainNet(
        nblocks=architectures[110],
        num_classes=num_classes,
    )
def plain1202(num_classes, **kwargs):
    return PlainNet(
        nblocks=architectures[1202],
        num_classes=num_classes,
    )

class PlainNet(nn.Module):

    def __init__(self, nblocks, in_channels=3, num_classes=10, **kwargs):
        super(PlainNet, self).__init__()

        nker = 64 if len(nblocks) == 4 else 16
        #nblocks = self.architectures[num_layers]

        # conv1: 224,224 -> 112,112
        # imagenet: size:224,224 -> 56,56
        # cifar10: size:32,32 -> 32,32 channels: 16 -> 16
        self.conv1 = ConvBN(
            in_channels=in_channels,
            out_channels=nker,
            kernel_size=7 if len(nblocks) == 4 else 3,
            stride=2 if len(nblocks) == 4 else 1,
            padding=3 if len(nblocks) == 4 else 1,
        )

        # conv2_x
        # imagenet: size:112,112 -> 56,56
        # cifar10: size:32,32 -> 32,32 channels: 16 -> 16
        self.conv2_x = [nn.MaxPool2d(kernel_size=3,stride=2,padding=1)]
        self.conv2_x += [PlainBlock(in_channels=nker,out_channels=nker) for _ in range(nblocks[0])]            
        self.conv2_x = nn.Sequential(*self.conv2_x)

        # conv3_x: 56,56, -> 28,28
        self.conv3_x = [PlainBlock(in_channels=nker,out_channels=2*nker)]
        self.conv3_x += [PlainBlock(in_channels=2*nker,out_channels=2*nker) for _ in range(nblocks[1]-1)]
        self.conv3_x = nn.Sequential(*self.conv3_x)

        # conv4_x: 28,28 -> 14,14
        self.conv4_x = [PlainBlock(in_channels=2*nker,out_channels=4*nker)]
        self.conv4_x += [PlainBlock(in_channels=4*nker,out_channels=4*nker) for _ in range(nblocks[2]-1)]
        self.conv4_x = nn.Sequential(*self.conv4_x)

        # conv5_x: 14,14 -> 7,7
        self.conv5_x = None
        if len(nblocks) == 4:
            self.conv5_x = [PlainBlock(in_channels=4*nker,out_channels=8*nker)]
            self.conv5_x += [PlainBlock(in_channels=8*nker,out_channels=8*nker) for _ in range(nblocks[3]-1)]
            self.conv5_x = nn.Sequential(*self.conv5_x)

        # fully-connected layer: 7,7 -> 1,1
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
        model_name = "plain" + str(model_num)
        
        print(model_name, "Test", end=' : ')
        model = locals()[model_name](num_classes=10)
        out = model(x)
        if list(out.shape) != [16,10]:
            print(model_name, out.shape, end=' : ')
            print(f'Output type: {type(out)}')
        else:
            print("passed")