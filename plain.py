import torch
import torch.nn as nn
from blocks import ConvLayer, PlainBlock

class PlainNet(nn.Module):
    architectures = {
        18:(2,2,2,2),
        34:(3,4,6,3),
    }

    def __init__(self, num_layers, in_channels, num_classes=10, nker=64):
        super(PlainNet, self).__init__()

        nblocks = self.architectures[num_layers]

        # conv1
        self.conv1 = ConvLayer(
            in_channels=in_channels,
            out_channels=nker,
            kernel_size=7,
            stride=2,
            padding=1,
        )

        # conv2_x
        self.conv2_x = [nn.MaxPool2d(kernel_size=3,stride=2,padding=1)]
        self.conv2_x += [PlainBlock(in_channels=nker,out_channels=nker) for _ in range(nblocks[0])]            
        self.conv2_x = nn.Sequential(*self.conv2_x)

        # conv3_x
        self.conv3_x = [PlainBlock(in_channels=nker,out_channels=2*nker)]
        self.conv3_x += [PlainBlock(in_channels=2*nker,out_channels=2*nker) for _ in range(nblocks[1]-1)]
        self.conv3_x = nn.Sequential(*self.conv3_x)

        # conv4_x
        self.conv4_x = [PlainBlock(in_channels=2*nker,out_channels=4*nker)]
        self.conv4_x += [PlainBlock(in_channels=4*nker,out_channels=4*nker) for _ in range(nblocks[2]-1)]
        self.conv4_x = nn.Sequential(*self.conv4_x)

        # conv5_x
        self.conv5_x = [PlainBlock(in_channels=4*nker,out_channels=8*nker)]
        self.conv5_x += [PlainBlock(in_channels=8*nker,out_channels=8*nker) for _ in range(nblocks[3]-1)]
        self.conv5_x = nn.Sequential(*self.conv5_x)

        # 마지막 layer
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=8*nker,out_features=num_classes)

    
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.conv2_x(x)

        x = self.conv3_x(x)

        x = self.conv4_x(x)

        x = self.conv5_x(x)
        
        x = self.avg_pooling(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)

        return out

if __name__=="__main__":
    model = PlainNet(
        num_layers=34,
        in_channels=3,
        num_classes=10
    )

    batch_size = 16
    x = torch.randn((batch_size,3,32,32))

    out = model(x)

    print(f'Output shape: {out.shape}')
    print(f'Output type: {type(out)}')