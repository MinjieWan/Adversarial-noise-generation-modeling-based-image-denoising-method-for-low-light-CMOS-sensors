import torch
import torch.nn as nn
import torch.fft as fft
# -------------------------
# MultiScaleResBlock
# -------------------------
class MultiScaleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 5, padding=2)
        self.conv5 = nn.Conv2d(channels, channels, 7, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.merge = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels)  # BN
        )
        self.alpha = nn.Parameter(torch.tensor(0.3))  

    def forward(self, x):
        f1 = self.relu(self.conv1(x))
        f3 = self.relu(self.conv3(x))
        f5 = self.relu(self.conv5(x))
        merged = torch.cat([f1, f3, f5], dim=1)
        out = self.merge(merged)
        return x + out * self.alpha


# -------------------------
# Frequency Domain Enhancement Module
# -------------------------
class FourierEnhance(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.3))  
        self.freq_mask_ratio = 0.7  

    def forward(self, x):
        fft_x = fft.fftshift(fft.fft2(x, norm='ortho'))
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)

        # MASK
        b, c, h, w = amp.shape
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        mask = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (self.freq_mask_ratio * h / 2) ** 2
        mask = mask.to(x.device).float()

        # Amplitude spectrum enhancement
        amp_high = amp * (1 + self.scale * torch.tanh(amp) * mask)

        # IFFT
        real = amp_high * torch.cos(phase)
        imag = amp_high * torch.sin(phase)
        enhanced = torch.real(fft.ifft2(fft.ifftshift(real + 1j * imag), norm='ortho'))
        return enhanced


# -------------------------
# Hybrid Fourier-enhanced Noise Generator
# -------------------------
class HybridFourierNoiseGenerator(nn.Module):
    """
    Input:
        z : Gaussian noise tensor
    Output:
        x : signal-independent noise
    """
    def __init__(self, in_channels=1, base_channels=32, num_res_blocks=4,
                 energy_min=0.7, energy_max=1.2):
        super().__init__()
        # Learnable global energy coefficient
        self.gamma = nn.Parameter(torch.tensor(0.5))

        # Spatial domain residual modeling
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(*[
            MultiScaleResBlock(base_channels) for _ in range(num_res_blocks)
        ])
        # Frequency domain enhancement
        self.freq = FourierEnhance()
        
        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Tanh()  
        )

        self.energy_min = energy_min
        self.energy_max = energy_max
        self.eps = 1e-8

    def forward(self, z):
        
        x = self.head(z)
        x = self.body(x)       
        x = self.freq(x)       
        x = self.tail(x)

        # ---------- RMS ----------
        z_rms = torch.sqrt(z.pow(2).mean(dim=(1,2,3), keepdim=True) + self.eps)
        x_rms = torch.sqrt(x.pow(2).mean(dim=(1,2,3), keepdim=True) + self.eps)
        ratio = x_rms / z_rms
        ratio_clamped = torch.clamp(ratio, min=self.energy_min, max=self.energy_max)
        x = x * (ratio_clamped / ratio)

        return x * self.gamma
