import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from einops import rearrange
from torch.nn.utils import spectral_norm
from config import IMG_SIZE

# --------------------------------------------------------
# Frequency Domain Transformer Attention Module
# --------------------------------------------------------
class FourierAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, C]
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

# --------------------------------------------------------
# Frequency domain processor
# --------------------------------------------------------
class FourierProcessor(nn.Module):
    def __init__(self, in_ch=1, n_bands=3, log_amplitude=True):
        super().__init__()
        self.n_bands = n_bands
        self.log_amplitude = log_amplitude
        self.mix = spectral_norm(nn.Conv2d(in_ch * n_bands * 2, in_ch * n_bands * 2, kernel_size=1))

    def forward(self, x):
        # x: [B, C, H, W]
        x_fft = torch.fft.fft2(x, norm='ortho')                
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))       
        mag = torch.abs(x_fft)                                
        phase = torch.angle(x_fft)                            
        if self.log_amplitude:
            mag = torch.log1p(mag)

        B, C, H, W = mag.shape
        # Frequency bands are divided based on radial radius
        yy = torch.linspace(-1., 1., steps=H, device=x.device).view(H, 1).expand(H, W)
        xx = torch.linspace(-1., 1., steps=W, device=x.device).view(1, W).expand(H, W)
        radius = torch.sqrt(xx**2 + yy**2).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        edges = torch.linspace(0.0, 1.0, steps=self.n_bands + 1, device=x.device)
        bands = []
        for i in range(self.n_bands):
            low, high = edges[i], edges[i+1]
            mask = ((radius >= low) & (radius < high)).float()       # [1,1,H,W]
            mask = mask.expand(B, C, H, W)
            band_mag = mag * mask
            band_phase = phase * mask
            bands.append((band_mag, band_phase))

        band_tensors = [torch.cat([m, p], dim=1) for m, p in bands]  # [B, C*2, H, W]
        out = torch.cat(band_tensors, dim=1)                         # [B, C*2*n_bands, H, W]
        out = self.mix(out)
        return out

# --------------------------------------------------------
# Patch Embedding
# --------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=16, use_sn=True):
        super().__init__()
        conv = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = spectral_norm(conv) if use_sn else conv
        self.patch_size = patch_size

    def forward(self, x):
        feat = self.proj(x)                  # [B, embed_dim, H/patch, W/patch]
        b, c, h, w = feat.shape
        tokens = rearrange(feat, 'b c h w -> b (h w) c')
        return tokens, (h, w)

# --------------------------------------------------------
# MultiScaleHead
# --------------------------------------------------------
class MultiScaleHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(in_ch, max(in_ch//2, 16), kernel_size=3, padding=1))
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.out = spectral_norm(nn.Conv2d(max(in_ch//2, 16), 1, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.act(self.conv(x))
        return self.out(x)  # [B,1,h,w]

# --------------------------------------------------------
# HybridFourierViTDiscriminator
# --------------------------------------------------------
class HybridFourierViTDiscriminator(nn.Module):
    def __init__(self,
                 img_size=IMG_SIZE,
                 in_channels=1,
                 n_bands=3,
                 embed_dim=192,
                 transformer_depth=6,
                 transformer_heads=8,
                 patch_size=16,
                 conv_widths=[64, 128, 256, 512],
                 use_sn=True):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels

        # Spatial Convolutional Pyramid Modeling
        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, conv_widths[0], 3, 1, 1)) if use_sn else nn.Conv2d(in_channels, conv_widths[0], 3, 1, 1),
            nn.InstanceNorm2d(conv_widths[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2)  # downsample x2
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(conv_widths[0], conv_widths[1], 3, 1, 1)) if use_sn else nn.Conv2d(conv_widths[0], conv_widths[1], 3, 1, 1),
            nn.InstanceNorm2d(conv_widths[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2)
        )
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(conv_widths[1], conv_widths[2], 3, 1, 1)) if use_sn else nn.Conv2d(conv_widths[1], conv_widths[2], 3, 1, 1),
            nn.InstanceNorm2d(conv_widths[2]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fourier = FourierProcessor(in_ch=in_channels, n_bands=n_bands, log_amplitude=True)
        freq_channels = in_channels * n_bands * 2

        self.patch_embed = PatchEmbed(freq_channels, embed_dim, patch_size=patch_size, use_sn=use_sn)
        # Position Embedding
        n_patches = (img_size // patch_size) ** 2
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, embed_dim))

        # Transformer stack
        self.transformer = nn.Sequential(*[FourierAttentionBlock(embed_dim, num_heads=transformer_heads) for _ in range(transformer_depth)])
        self.transformer_norm = nn.LayerNorm(embed_dim)

        # Cross-fuse
        fused_ch = conv_widths[2] + embed_dim
        self.fuse_conv = spectral_norm(nn.Conv2d(fused_ch, conv_widths[2], kernel_size=1)) if use_sn else nn.Conv2d(fused_ch, conv_widths[2], 1)
        self.fuse_act = nn.LeakyReLU(0.2, inplace=True)

        # MultiScaleHead
        self.head1 = MultiScaleHead(conv_widths[0])
        self.head2 = MultiScaleHead(conv_widths[1])
        self.head3 = MultiScaleHead(conv_widths[2])

        # Global discrimination head
        self.global_fc1 = spectral_norm(nn.Linear(embed_dim + conv_widths[2], embed_dim // 2)) if use_sn else nn.Linear(embed_dim + conv_widths[2], embed_dim // 2)
        self.global_act = nn.LeakyReLU(0.2, inplace=True)
        self.global_fc2 = spectral_norm(nn.Linear(embed_dim // 2, 1)) if use_sn else nn.Linear(embed_dim // 2, 1)

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W] (C==in_channels)
        Return:
            ([out1, out2, out3], global_score)
            out*: [B,1,h,w], global_score: [B,1]
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Expected {self.img_size}x{self.img_size}, got {H}x{W}"

        # ---------------------------
        # Spatial branch
        # ---------------------------
        f1 = self.conv1(x)   # [B, c1, H/2, W/2]
        f2 = self.conv2(f1)  # [B, c2, H/4, W/4]
        f3 = self.conv3(f2)  # [B, c3, H/4, W/4]  

        # ---------------------------
        # Frequency domain branch
        # ---------------------------
        freq_map = self.fourier(x)                       # [B, freq_ch, H, W]
        tokens, (ph, pw) = self.patch_embed(freq_map)    # tokens: [B, N, embed_dim]
        tokens = tokens + self.pos_emb                    # Position Embedding
        tokens = self.transformer(tokens)                 
        tokens = self.transformer_norm(tokens)
        freq_global = tokens.mean(dim=1)                  # [B, embed_dim]

        # ---------------------------
        # Fusion
        # ---------------------------
        freq_map_for_fuse = freq_global.view(B, -1, 1, 1).expand(-1, -1, f3.size(2), f3.size(3))
        fused = torch.cat([f3, freq_map_for_fuse], dim=1)  # [B, c3 + embed, h, w]
        fused = self.fuse_conv(fused)
        fused = self.fuse_act(fused)

        # ---------------------------
        # MultiScaleHead
        # ---------------------------
        out1 = self.head1(f1)      # [B,1,H/2,W/2]
        out2 = self.head2(f2)      # [B,1,H/4,W/4]
        out3 = self.head3(fused)   # [B,1,H/4,W/4] 

        # ---------------------------
        # global discrimination score
        # ---------------------------
        local_global = F.adaptive_avg_pool2d(fused, (1,1)).view(B, -1)  # [B, c3]
        global_feat = torch.cat([freq_global, local_global], dim=1)    # [B, embed + c3]
        g = self.global_fc1(global_feat)
        g = self.global_act(g)
        global_score = self.global_fc2(g)   # [B,1]

        return [out1, out2, out3], global_score
