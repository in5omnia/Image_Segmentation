import torch
from torch import nn
import matplotlib.pyplot as plt
import torch
from torch import nn
from clipunet import ClipUNet

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
    def __init__(self, din, dout):
        super().__init__()
        self.scale = 1

        self.down1 = DoubleConvReLU(din, self.scale * 64)
        self.down2 = Down(self.scale * 64, self.scale * 128)
        self.down3 = Down(self.scale * 128, self.scale * 256)
        self.down4 = Down(self.scale * 256, self.scale * 512)
        self.down5 = Down(self.scale * 512, self.scale * 1024)

        self.up1 = Up(self.scale * 1024, self.scale * 512)
        self.up2 = Up(self.scale * 512, self.scale * 256)
        self.up3 = Up(self.scale * 256, self.scale * 128)
        self.up4 = Up(self.scale * 128, self.scale * 64)

        self.output = nn.Conv2d(self.scale * 64, dout, kernel_size=1)

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
    

class PromptModel(nn.Module):
    def __init__(self, path):
        super().__init__()

        self.clip = ClipUNet()
        self.mask = unet(4, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        if path is not None:
            try:
                checkpoint = torch.load(path, weights_only=False, map_location=lambda storage, loc: storage) # Load to CPU initially
                self.clip.load_state_dict(checkpoint["model_state_dict"])
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)[:200]}")
                raise
            
        # for param in self.clip.parameters():
        #     param.requires_grad = False

    def forward(self, x, heatmap):
        clip_logit = self.clip(x)
        clip_prob = self.softmax(clip_logit)

        mask_logit = self.mask(torch.concat([x, heatmap], dim=1))
        mask_prob = self.sigmoid(mask_logit)

        final_probs = torch.empty_like(clip_prob)
        selected_prob = mask_prob * clip_prob

        final_probs[:, 1:4, :, :] = selected_prob[:, 0:3, :, :]
        final_probs[:, 0:1, :, :] = 1.0 - mask_prob
        final_probs[:, 1:2, :, :] += selected_prob[:, 3:4, :, :]

        # log_final_probs = torch.log(final_probs + 1e-9)

        return final_probs
