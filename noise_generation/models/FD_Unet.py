import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# DWT / IWT
# ----------------------------
class DWTModule(nn.Module):
    def forward(self, x):
        x_00 = x[:, :, 0::2, 0::2]
        x_01 = x[:, :, 0::2, 1::2]
        x_10 = x[:, :, 1::2, 0::2]
        x_11 = x[:, :, 1::2, 1::2]
        half = 0.5
        x_LL = (x_00 + x_01 + x_10 + x_11) * half
        x_LH = (x_00 - x_01 + x_10 - x_11) * half
        x_HL = (x_00 + x_01 - x_10 - x_11) * half
        x_HH = (x_00 - x_01 - x_10 + x_11) * half
        return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=1)


class IWTModule(nn.Module):
    def forward(self, x):
        B, C, H, W = x.size()
        out_C = C // 4
        x = x.view(B, out_C, 4, H, W)

        x1 = x[:, :, 0]
        x2 = x[:, :, 1]
        x3 = x[:, :, 2]
        x4 = x[:, :, 3]

        half = 0.5
        h = torch.zeros((B, out_C, H * 2, W * 2), device=x.device)

        h[:, :, 0::2, 0::2] = (x1 - x2 - x3 + x4) * half
        h[:, :, 1::2, 0::2] = (x1 - x2 + x3 - x4) * half
        h[:, :, 0::2, 1::2] = (x1 + x2 - x3 - x4) * half
        h[:, :, 1::2, 1::2] = (x1 + x2 + x3 + x4) * half

        return h


# ----------------------------
# UNet 
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class DownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(in_channels, in_channels * 2, 1)
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        return x1 + x2


# ----------------------------
#  FD 
# ----------------------------
class FD_FullBand(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        mid = 256

        self.conv1 = nn.Conv2d(in_ch, mid, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1)

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, mid // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid // 4, mid, 1),
            nn.Sigmoid()
        )

        self.conv3 = nn.Conv2d(mid, in_ch, 3, padding=1)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = out * self.ca(out)
        out = self.conv3(out)
        return out


# ----------------------------
# FD-UNet
# ----------------------------
class FD_UNET(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.dwt = DWTModule()
        self.iwt = IWTModule()

        # UNet 
        self.unet_conv = nn.Conv2d(in_channels * 4, 64, 3, padding=1)
        self.conv1 = ConvBlock(64, 64)
        self.down1 = DownSample(64)
        self.conv2 = ConvBlock(128, 128)
        self.down2 = DownSample(128)
        self.conv3 = ConvBlock(256, 256)
        self.up1 = UpSample(256)
        self.conv4 = ConvBlock(128, 128)
        self.up2 = UpSample(128)
        self.conv5 = ConvBlock(64, 64)
        self.conv_out = nn.Conv2d(64, in_channels * 4, 3, padding=1)

        # FD
        self.fd_full = FD_FullBand(in_channels * 4)

        # Adaptive fusion coefficient
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):

        x_dwt = self.dwt(x)

        # UNet branch
        en1 = self.unet_conv(x_dwt)
        en2 = self.conv1(en1)
        d1 = self.down1(en2)
        en3 = self.conv2(d1)
        d2 = self.down2(en3)
        en4 = self.conv3(d2)
        u1 = self.up1(en4, en3)
        de3 = self.conv4(u1)
        u2 = self.up2(de3, en2)
        de2 = self.conv5(u2)
        unet_out = self.conv_out(de2)

        # FD branch
        fd_out = self.fd_full(x_dwt)

        alpha = torch.sigmoid(self.alpha)

        # Fusion
        fused = x_dwt + unet_out + alpha * fd_out

        out = self.iwt(fused)

        return out


# ----------------------------
# Loss
# ----------------------------

class FDUNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        return F.mse_loss(pred, gt)