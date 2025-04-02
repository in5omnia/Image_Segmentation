import torch
from torch import nn

class DoubleConvReLU(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.doubleConvReLU = nn.Sequential(
            nn.Conv2d(din, dout, kernel_size=3, padding=1),
            nn.BatchNorm2d(dout),
            nn.ReLU(),
            nn.Conv2d(dout, dout, kernel_size=3, padding=1),
            nn.BatchNorm2d(dout),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.doubleConvReLU(x)
    

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, din, dout):
        super().__init__()
        self.maxpool_doubleConv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConvReLU(din, dout)
        )

    def forward(self, x):
        return self.maxpool_doubleConv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, din, dout):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(din, dout, kernel_size=2, stride=2)
        self.doubleConv = DoubleConvReLU(din, dout)

    def forward(self, x1, x2):
        x = torch.cat([x1, self.upsample(x2)], dim=1)
        return self.doubleConv(x)


class unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 1 

        self.down1 = DoubleConvReLU(3, self.scale * 64)
        self.down2 = Down(self.scale * 64, self.scale * 128)
        self.down3 = Down(self.scale * 128, self.scale * 256)
        self.down4 = Down(self.scale * 256, self.scale * 512)
        self.down5 = Down(self.scale * 512, self.scale * 1024)

        self.up1 = Up(self.scale * 1024, self.scale * 512)
        self.up2 = Up(self.scale * 512, self.scale * 256)
        self.up3 = Up(self.scale * 256, self.scale * 128)
        self.up4 = Up(self.scale * 128, self.scale * 64)

        self.output = nn.Conv2d(self.scale * 64, 4, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        
        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        return self.output(x)