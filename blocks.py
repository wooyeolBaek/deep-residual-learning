import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True) -> torch.FloatTensor:
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
            ),
        )
    
    def forward(self, x):
        return self.layer(x)


class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True) -> torch.FloatTensor:
        super().__init__()

        self.h = nn.Sequential(
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1 if in_channels==out_channels else 2,
                padding=1,
                bias=bias,
            ),
            nn.ReLU(),
            ConvLayer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.h(x)
        x = self.relu(x)

        return x


class IdentityMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # identity mapping: in_channels==out_channels
        self.shortcut = None
        
        # identity mapping with zero padding: in_channels!=out_channels
        if in_channels != out_channels:
            channels_num = (out_channels - in_channels)
            front_pad_num = channels_num//2
            back_pad_num = channels_num - front_pad_num

            self.shortcut = nn.Sequential(
                nn.MaxPool2d(
                    kernel_size=1,
                    stride=2,
                    padding=0,
                ),
                nn.ConstantPad3d(
                    padding=(0,0,0,0,front_pad_num,back_pad_num),
                    value=0,
                )
            )
    
    def forward(self, x):
        # identity mapping with zero padding: in_channels!=out_channels
        if not self.shortcut is None:
            return self.shortcut(x)
        
        # identity mapping: in_channels==out_channels
        return x


class ProjectionMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.shortcut = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2
        )
    
    def forward(self, x):
        
        return self.shortcut(x)


class ResBlock(PlainBlock):
    def __init__(self, in_channels, out_channels, bias=True, mapping='identity') -> torch.FloatTensor:
        super().__init__(in_channels, out_channels, bias)
        
        # option A: Identity Mapping with Zero Padding
        self.shortcut = IdentityMap(
            in_channels=in_channels,
            out_channels=out_channels,
        )

        # option B: Projection Mapping
        if in_channels!=out_channels and mapping=='projection':
            self.shortcut = ProjectionMap(
                in_channels=in_channels,
                out_channels=out_channels,
            )
    
    def forward(self, x):
        x = self.h(x) + self.shortcut(x)
        x = self.relu(x)

        return x


if __name__=="__main__":

    batch_size = 16
    in_channels = 64
    out_channels = 128
    size = 32

    # input: (16, 64, 32, 32)
    x = torch.randn((batch_size,in_channels,size,size))

    # ConvLayer Test
    print('ConvLayer Test')
    # the same dimension
    conv_out64 = ConvLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=True)(x)
    # dimension expansion
    conv_out128 = ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)(x)

    print(f'conv_out64 Output, shape: {tuple(conv_out64.shape)} == (16,64,32,32)')
    print(f'conv_out128 Output, shape: {tuple(conv_out128.shape)} == (16,128,32,32)')

    # PlainBlock Test
    print('PlainBlock Test')
    # the same dimension
    plain_out64 = PlainBlock(in_channels=in_channels, out_channels=in_channels, bias=True)(x)
    # dimension expansion
    plain_out128 = PlainBlock(in_channels=in_channels, out_channels=out_channels, bias=True)(x)

    print(f'plain_out64 Output, shape: {tuple(plain_out64.shape)} == (16,64,32,32)')
    print(f'plain_out128 Output, shape: {tuple(plain_out128.shape)} == (16,128,16,16)')
    
    # ResBlock Test
    print('PlainBlock Test')
    # the same dimension identity
    res_out64_iden = ResBlock(in_channels=in_channels, out_channels=in_channels, bias=True, mapping='identity')(x)
    # dimension expansion identity
    res_out128_iden = ResBlock(in_channels=in_channels, out_channels=out_channels, bias=True, mapping='identity')(x)
    # the same dimension projection
    res_out64_proj = ResBlock(in_channels=in_channels, out_channels=in_channels, bias=True, mapping='projection')(x)
    # dimension expansion projection
    res_out128_proj = ResBlock(in_channels=in_channels, out_channels=out_channels, bias=True, mapping='projection')(x)

    print(f'res_out64_identity Output, shape: {tuple(res_out64_iden.shape)} == (16,64,32,32)')
    print(f'res_out128_identity Output, shape: {tuple(res_out128_iden.shape)} == (16,128,16,16)')
    print(f'res_out64_projection Output, shape: {tuple(res_out64_proj.shape)} == (16,64,32,32)')
    print(f'res_out128_projection Output, shape: {tuple(res_out128_proj.shape)} == (16,128,16,16)')

