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

if __name__=="__main__":

    batch_size = 16
    in_channels = 64
    out_channels = 128
    size = 32

    x = torch.randn((batch_size,in_channels,size,size))

    conv_out64 = ConvLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=True)(x)
    conv_out128 = ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)(x)

    print(f'conv_out64 Output, shape: {conv_out64.shape}')
    print(f'conv_out128 Output, shape: {conv_out128.shape}')
