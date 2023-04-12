import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from models.blocks import ConvBN

def vgg19(num_classes, **kwargs):
    return Vgg(
        in_channels=3,
        num_classes=num_classes,
    )

class Vgg(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, nker=64):
        super(Vgg, self).__init__()

        # --conv1
        self.layer1 = nn.Sequential(
            ConvBN(
                in_channels=in_channels,
                out_channels=nker,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBN(
                in_channels=nker,
                out_channels=nker,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        # --max_pooling
        self.max_pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # --conv2
        self.layer2 = nn.Sequential(
            ConvBN(
                in_channels=nker,
                out_channels=nker*2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBN(
                in_channels=nker*2,
                out_channels=nker*2,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        # --conv3
        self.layer3 = nn.Sequential(
            ConvBN(
                in_channels=nker*2,
                out_channels=nker*2*2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBN(
                in_channels=nker*2*2,
                out_channels=nker*2*2,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        # --conv4
        self.layer4 = nn.Sequential(
            ConvBN(
                in_channels=nker*2*2,
                out_channels=nker*2*2*2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBN(
                in_channels=nker*2*2*2,
                out_channels=nker*2*2*2,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        # --conv5
        self.layer5 = nn.Sequential(
            ConvBN(
                in_channels=nker*2*2*2,
                out_channels=nker*2*2*2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBN(
                in_channels=nker*2*2*2,
                out_channels=nker*2*2*2,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        # --fully-connected layer
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(
            in_features=nker*2*2*2,
            out_features=nker*2*2*2,
            ),
            nn.Linear(
            in_features=nker*2*2*2,
            out_features=nker*2*2*2,
            ),
            nn.Linear(
            in_features=nker*2*2*2,
            out_features=num_classes,
            ),
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.max_pool(x)

        x = self.layer2(x)
        x = self.max_pool(x)

        x = self.layer3(x)
        x = self.max_pool(x)

        x = self.layer4(x)
        x = self.max_pool(x)

        x = self.layer5(x)
        x = self.max_pool(x)
        
        x = self.avg_pooling(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)

        return out

if __name__=="__main__":
    
    batch_size = 16
    x = torch.randn((batch_size,3,150,150))
    
    print("vgg19 test")

    model = vgg19(num_classes=10)
    out = model(x)

    print(f'Output shape: {out.shape}')