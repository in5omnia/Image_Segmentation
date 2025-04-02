import torch
from torch import nn

class down(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(din, dout, kernel_size=3, padding=1),
            nn.BatchNorm2d(dout),
            nn.ReLU(),
            nn.Conv2d(dout, dout, kernel_size=3, padding=1),
            nn.BatchNorm2d(dout),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.down(x)
    
class up(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(din, dout, kernel_size=3, padding=1),
            nn.BatchNorm2d(dout),
            nn.ReLU(),
            nn.Conv2d(dout, dout, kernel_size=3, padding=1),
            nn.BatchNorm2d(dout),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.up(x)

class old_unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 1 # maybe using the original channels might be better
        self.flatten = nn.Flatten()
        self.down1 = down(3, self.scale * 64)
        self.down2 = down(self.scale * 64, self.scale * 128)
        self.down3 = down(self.scale * 128, self.scale * 256)
        self.down4 = down(self.scale * 256, self.scale * 512)
        self.down5 = down(self.scale * 512, self.scale * 1024)
        self.up1 = up(self.scale * 1024, self.scale * 512)
        self.up2 = up(self.scale * 512, self.scale * 256)
        self.up3 = up(self.scale * 256, self.scale * 128)
        self.up4 = up(self.scale * 128, self.scale * 64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.upconv1 = nn.Conv2d(self.scale * 1024, self.scale * 512, kernel_size=3, padding=1)
        self.upconv2 = nn.Conv2d(self.scale * 512, self.scale * 256, kernel_size=3, padding=1)
        self.upconv3 = nn.Conv2d(self.scale * 256, self.scale * 128, kernel_size=3, padding=1)
        self.upconv4 = nn.Conv2d(self.scale * 128, self.scale * 64, kernel_size=3, padding=1)
        self.output = nn.Sequential(
            nn.Conv2d(self.scale * 64, 4, kernel_size=1),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.maxpool(x1))
        x3 = self.down3(self.maxpool(x2))
        x4 = self.down4(self.maxpool(x3))
        x5 = self.down5(self.maxpool(x4))
        x = self.up1(torch.cat([x4, self.upconv1(self.upsample(x5))], dim=1))
        x = self.up2(torch.cat([x3, self.upconv2(self.upsample(x))], dim=1))
        x = self.up3(torch.cat([x2, self.upconv3(self.upsample(x))], dim=1))
        x = self.up4(torch.cat([x1, self.upconv4(self.upsample(x))], dim=1))
        pre_output = self.output(x)
        return pre_output