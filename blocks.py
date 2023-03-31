import torch
import torch.nn as nn

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
            ),
        )
    
    def forward(self, x):
        return self.layer(x)


class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.f = nn.Sequential(
            ConvBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1 if in_channels==out_channels else 2, # 2 for downsampling
                padding=1,
            ),
            nn.ReLU(),
            ConvBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.f(x)
        x = self.relu(x)

        return x


class IdentityMap(nn.Module):
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

        self.shortcut = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
        )
    
    def forward(self, x):
        
        return self.shortcut(x)


class ResBlock(PlainBlock):
    def __init__(self, in_channels, out_channels, mapping='A'):
        super().__init__(in_channels, out_channels)
        
        # option A: Identity Mapping with Zero Padding
        self.shortcut = IdentityMap(
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
    def __init__(self, in_channels, out_channels, mapping='A'):
        super().__init__(in_channels, out_channels, mapping)

        self.f = nn.Sequential(
            ConvBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            ConvBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1 if in_channels==out_channels else 2, # 2 for downsampling
                padding=1,
            ),
            nn.ReLU(),
            ConvBN(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )


if __name__=="__main__":

    batch_size = 4
    in_channels = 16
    out_channels = 128
    size = 32
    size=224

    # input: (4, 16, 32, 32)
    x = torch.randn((batch_size,in_channels,size,size))

    # ConvBN Test
    print('ConvBN Test')
    # the same dimension
    conv_out16 = ConvBN(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)(x)
    # downsampling: dimension expansion
    conv_out32 = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)(x)

    print(f'conv_out64 Output, shape: {tuple(conv_out16.shape)} == (4,16,32,32)')
    print(f'conv_out32 Output, shape: {tuple(conv_out32.shape)} == (4,32,16,16)')

    # conv1
    print('Conv 1 Test')
    # imagenet
    #conv_out1 = ConvBN(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=2, padding=3)(x)
    #print(f'conv_out1 Output, shape: {tuple(conv_out1.shape)} == (4,32,16,16)')
    conv_out1 = ConvBN(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)(x)
    print(f'conv_out1 Output, shape: {tuple(conv_out1.shape)} == (4,32,16,16)')

    # PlainBlock Test
    print('PlainBlock Test')
    # the same dimension
    plain_out16 = PlainBlock(in_channels=in_channels, out_channels=in_channels)(x)
    # dimension expansion
    plain_out32 = PlainBlock(in_channels=in_channels, out_channels=out_channels)(x)

    print(f'plain_out16 Output, shape: {tuple(plain_out16.shape)} == (4,16,32,32)')
    print(f'plain_out32 Output, shape: {tuple(plain_out32.shape)} == (4,32,16,16)')
    

    # ResBlock Test
    print('ResBlock Test')
    # the same dimension identity
    res_out16_iden = ResBlock(in_channels=in_channels, out_channels=in_channels, mapping='A')(x)
    # dimension expansion identity
    res_out32_iden = ResBlock(in_channels=in_channels, out_channels=out_channels, mapping='A')(x)
    # the same dimension projection
    res_out16_proj = ResBlock(in_channels=in_channels, out_channels=in_channels, mapping='B')(x)
    # dimension expansion projection
    res_out32_proj = ResBlock(in_channels=in_channels, out_channels=out_channels, mapping='B')(x)
    # the same dimension projection
    res_out16_iden_proj = ResBlock(in_channels=in_channels, out_channels=in_channels, mapping='C')(x)
    # dimension expansion projection
    res_out32_iden_proj = ResBlock(in_channels=in_channels, out_channels=out_channels, mapping='C')(x)

    print(f'res_out16_identity Output, shape: {tuple(res_out16_iden.shape)} == (4,16,32,32)')
    print(f'res_out32_identity Output, shape: {tuple(res_out32_iden.shape)} == (4,32,16,16)')
    print(f'res_out16_projection Output, shape: {tuple(res_out16_proj.shape)} == (4,16,32,32)')
    print(f'res_out32_projection Output, shape: {tuple(res_out32_proj.shape)} == (4,32,16,16)')
    print(f'res_out16_projection Output, shape: {tuple(res_out16_iden_proj.shape)} == (4,16,32,32)')
    print(f'res_out32_projection Output, shape: {tuple(res_out32_iden_proj.shape)} == (4,32,16,16)')


    # BottleneckBlock Test
    print('BottleneckBlock Test')
    # the same dimension identity
    bottleneck_out16_iden = BottleneckBlock(in_channels=in_channels, out_channels=in_channels, mapping='A')(x)
    # dimension expansion identity
    bottleneck_out32_iden = BottleneckBlock(in_channels=in_channels, out_channels=out_channels, mapping='A')(x)
    # the same dimension projection
    bottleneck_out16_proj = BottleneckBlock(in_channels=in_channels, out_channels=in_channels, mapping='B')(x)
    # dimension expansion projection
    bottleneck_out32_proj = BottleneckBlock(in_channels=in_channels, out_channels=out_channels, mapping='B')(x)

    print(f'bottleneck_out16_identity Output, shape: {tuple(bottleneck_out16_iden.shape)} == (4,16,32,32)')
    print(f'bottleneck_out32_identity Output, shape: {tuple(bottleneck_out32_iden.shape)} == (4,32,16,16)')
    print(f'bottleneck_out16_projection Output, shape: {tuple(bottleneck_out16_proj.shape)} == (4,16,32,32)')
    print(f'bottleneck_out32_projection Output, shape: {tuple(bottleneck_out32_proj.shape)} == (4,32,16,16)')

