import torch.nn as nn


class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
        super().__init__()

        self.f = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1 if in_channels==out_channels else 2, # 2 for downsampling
                padding=padding,
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
            ),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.f(x)
        z = self.relu(z)

        return z


class ZeroPadMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # identity mapping: in_channels==out_channels
        self.shortcut = None
        
        # identity mapping with zero padding: in_channels!=out_channels
        if in_channels != out_channels:
            nchannels = (out_channels - in_channels)
            nfront_pad = nchannels//2
            nback_pad = nchannels - nfront_pad

            self.shortcut = nn.Sequential(
                nn.MaxPool2d(
                    kernel_size=1,
                    stride=2,
                    padding=0,
                ),
                nn.ConstantPad3d(
                    padding=(0,0,0,0,nfront_pad,nback_pad),
                    value=0,
                )
            )
    
    def forward(self, x):
        # identity mapping with zero padding: in_channels!=out_channels
        if self.shortcut is not None:
            return self.shortcut(x)
        
        # identity mapping: in_channels==out_channels
        return x


class ProjectionMap(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
            ),
        )
    
    def forward(self, x):
        
        return self.shortcut(x)


class ResBlock(PlainBlock):
    def __init__(self, in_channels, out_channels, mapping='B'):
        super().__init__(in_channels, out_channels)
        
        # option A: Identity Mapping with Zero Padding
        self.shortcut = ZeroPadMap(
            in_channels=in_channels,
            out_channels=out_channels,
        )

        # option B: Identity Mapping with Projection Mapping
        # option C: Projection Mapping
        if (in_channels!=out_channels and mapping=='B') or mapping=='C':
            self.shortcut = ProjectionMap(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2 if in_channels!=out_channels else 1 # 1 for C
            )
    
    def forward(self, x):
        x = self.f(x) + self.shortcut(x)
        x = self.relu(x)

        return x


class BottleneckBlock(ResBlock):
    def __init__(self, in_channels, out_channels, mapping='B'):
        super().__init__(in_channels, out_channels, mapping)

        self.f = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1 if in_channels==out_channels else 2, # 2 for downsampling
                padding=0,
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

