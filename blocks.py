import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True) -> None:
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
    def __init__(self, in_channels, out_channels, bias=True) -> None:
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
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1 if in_channels==out_channels else 2,
                padding=1,
                bias=bias,
            ),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.h(x)
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
    plain_out64 = ConvLayer(in_channels=in_channels, out_channels=in_channels, bias=True)(x)
    # dimension expansion
    plain_out128 = ConvLayer(in_channels=in_channels, out_channels=out_channels, bias=True)(x)

    print(f'plain_out64 Output, shape: {tuple(plain_out64.shape)} == (16,64,32,32)')
    print(f'plain_out128 Output, shape: {tuple(plain_out128.shape)} == (16,128,32,32)')
