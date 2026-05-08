import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from config import *
from datasets.dataset_pair import PairedDataset
from torchvision.models import vgg19

# =========================================================
# VGGPerceptualLoss
# =========================================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(weights="IMAGENET1K_V1").features[:16].eval().to(DEVICE)
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)

    def normalize(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        x_min = x.amin(dim=(-2,-1), keepdim=True)
        x_max = x.amax(dim=(-2,-1), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)
        return (x - self.mean) / self.std

    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        return (self.vgg(x) - self.vgg(y)).abs().mean()

# =========================================================
# FD-UNet Denoising
# =========================================================
from models.FD_Unet import FD_UNET

denoise_net = FD_UNET(in_channels=1).to(DEVICE)
if DENOISE_WEIGHTS.get("fdunet"):
    ckpt = torch.load(DENOISE_WEIGHTS["fdunet"], map_location=DEVICE)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("module.","").replace("net.",""): v for k,v in state_dict.items()}
    denoise_net.load_state_dict(state_dict, strict=False)

denoise_net.eval()
for p in denoise_net.parameters():
    p.requires_grad = False

@torch.no_grad()
def get_denoised(x):
    return denoise_net(x).clamp(0,1)

# =========================================================
# PG noisy
# =========================================================
def add_pg_noise_tensor(x, a, b):
    y = x * 255.0
    lam = (y / a).clamp(min=0.0)
    p_noisy = torch.poisson(lam) / 255.0 * a
    g_noisy = torch.randn_like(x) * (b ** 0.5)
    return p_noisy, g_noisy

# =========================================================
# Generator / Discriminator
# =========================================================
from models.noise_generator import HybridFourierNoiseGenerator

G = HybridFourierNoiseGenerator(
    in_channels=1,
    base_channels=32,
    num_res_blocks=4,
    energy_min=0.7,
    energy_max=1.2
).to(DEVICE)

from models.discriminator import HybridFourierViTDiscriminator
D = HybridFourierViTDiscriminator(
    img_size=IMG_SIZE,
    in_channels=1,
    n_bands=3,
    embed_dim=192,
    transformer_depth=6,
    transformer_heads=8,
    patch_size=16,
    conv_widths=[64,128,256,512]
).to(DEVICE)

# =========================================================
# Optimizer + Cosine Scheduler
# =========================================================
optimizer_G = optim.Adam(
    G.parameters(),
    lr=LR_G,
    betas=(0.5, 0.999)
)

optimizer_D = optim.Adam(
    D.parameters(),
    lr=LR_D,
    betas=(0.5, 0.999)
)


from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler_G = CosineAnnealingLR(
    optimizer_G,
    T_max=EPOCHS,
    eta_min=1e-6
)

l1_loss = nn.L1Loss()
perceptual_loss = VGGPerceptualLoss().to(DEVICE)

# =========================================================
# Dataset
# =========================================================
dataset = PairedDataset(CLEAN_DIR, NOISY_DIR, img_size=IMG_SIZE)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# =========================================================
# Checkpoint
# =========================================================
date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
save_dir = os.path.join(CHECKPOINT_DIR, date)
vis_dir  = os.path.join(save_dir, "samples")
os.makedirs(vis_dir, exist_ok=True)
writer = SummaryWriter(os.path.join(save_dir, "tb"))

# =========================================================
# Gradient Penalty
# =========================================================
def aggregate_D_outputs(outs):
    return sum(o.mean() for o in outs)

def gradient_penalty(D_model, real, fake):
    alpha = torch.rand(real.size(0),1,1,1, device=DEVICE)
    interp = (alpha*real + (1-alpha)*fake).requires_grad_(True)
    outs,_ = D_model(interp)
    score = aggregate_D_outputs(outs)
    grads = torch.autograd.grad(
        score, interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True
    )[0]
    grad_norm = grads.view(grads.size(0), -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()

# =========================================================
# TRAIN
# =========================================================
GRAD_CLIP = 0.3
GP_MAX = 10.0
D_STEPS = 1

global_step = 0

for epoch in range(1, EPOCHS+1):
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for clean, real_noisy in pbar:
        clean = clean.to(DEVICE)
        real_noisy = real_noisy.to(DEVICE)

        p_noisy, g_noisy = add_pg_noise_tensor(clean, PG_A, PG_B)
        gen_noise = G(g_noisy)
        fake_noisy = (p_noisy + gen_noise).clamp(0,1)

        # ---------------- D ----------------
        D.train(); G.eval()
        optimizer_D.zero_grad(set_to_none=True)

        d_real_outs,_ = D(real_noisy)
        d_fake_outs,_ = D(fake_noisy.detach())
        d_real = aggregate_D_outputs(d_real_outs)
        d_fake = aggregate_D_outputs(d_fake_outs)
        gp = gradient_penalty(D, real_noisy, fake_noisy.detach()).clamp(0, GP_MAX)

        d_loss = -d_real + d_fake + LAMBDA_GP * gp

        if torch.isfinite(d_loss):
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), GRAD_CLIP)
            optimizer_D.step()
        else:
            continue

        # ---------------- G ----------------
        G.train(); D.eval()
        optimizer_G.zero_grad(set_to_none=True)

        d_fake_outs,_ = D(fake_noisy)
        g_adv = -aggregate_D_outputs(d_fake_outs)

        with torch.no_grad():
            den_real = get_denoised(real_noisy)
        den_fake = get_denoised(fake_noisy)

        g_l1 = l1_loss(den_fake, den_real)
        g_per = perceptual_loss(fake_noisy, real_noisy)

        g_loss = g_adv + LAMBDA_L1*g_l1 + LAMBDA_PER*g_per

        if torch.isfinite(g_loss):
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), GRAD_CLIP)
            optimizer_G.step()

        # ---------------- log ----------------
        writer.add_scalar("Loss/D", d_loss.item(), global_step)
        writer.add_scalar("Loss/G", g_loss.item(), global_step)
        writer.add_scalar("LR/G", optimizer_G.param_groups[0]["lr"], global_step)
        global_step += 1

        pbar.set_postfix(
            D=d_loss.item(),
            G=g_loss.item(),
            L1=g_l1.item(),
            Per=g_per.item()
        )

    
    scheduler_G.step()

    if epoch % 5 == 0:
        torch.save(G.state_dict(), f"{save_dir}/G_{epoch}.pth")
        torch.save(D.state_dict(), f"{save_dir}/D_{epoch}.pth")
        save_image(
            torch.cat([
                clean[:4],
                p_noisy[:4],
                gen_noise[:4],
                fake_noisy[:4],
                real_noisy[:4]
            ], 0),
            f"{vis_dir}/epoch_{epoch}.png",
            nrow=5
        )

writer.close()
print(" Training completed")
