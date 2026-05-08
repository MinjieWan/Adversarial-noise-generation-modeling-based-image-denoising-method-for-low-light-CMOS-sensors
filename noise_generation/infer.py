import os
import glob
import torch
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from PIL import Image
from datetime import datetime
from config import *
from models.noise_generator import  HybridFourierNoiseGenerator
from models.FD_Unet import FD_UNET
import torch.nn.functional as F

# ------------------------- Create results directory -------------------------
def create_result_dir(base_root):
    today = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(base_root, today)
    os.makedirs(date_dir, exist_ok=True)
    existing = [d for d in os.listdir(date_dir)
                if os.path.isdir(os.path.join(date_dir, d)) and d.isdigit()]
    idx = max([int(d) for d in existing], default=0) + 1
    result_dir = os.path.join(date_dir, f"{idx:03d}")
    os.makedirs(result_dir, exist_ok=False)
    os.makedirs(os.path.join(result_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "noisy"), exist_ok=True)
    return result_dir

# ------------------------- Load grayscale image -------------------------
def load_image_gray(path):
    img = Image.open(path).convert("L")
    tensor = ToTensor()(img).unsqueeze(0)  
    return tensor

# -------------------------  PG noise -------------------------
def add_pg_noise_tensor(x, a, b):
    y = x * 255.0
    lam = (y / a).clamp(min=0.0)
    p_noisy = torch.poisson(lam) / 255.0 * a
    g_noisy = torch.randn_like(x) * (b ** 0.5)
    return p_noisy, g_noisy

# ------------------------- main -------------------------
def main():
    device = DEVICE

  # ----- Generator -----
    G = HybridFourierNoiseGenerator(
        in_channels=1,
        base_channels=32,
        num_res_blocks=4,
        energy_min=0.7,
        energy_max=1.2
    ).to(DEVICE)
    print("Load the HybridFourierNoiseGenerator used for training")

    ckpt = torch.load(GENERATOR_WEIGHTS, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = {k.replace("module.", ""): v for k,v in ckpt["state_dict"].items()}
        G.load_state_dict(state, strict=False)
    else:
        G.load_state_dict({k.replace("module.", ""): v for k,v in ckpt.items()}, strict=False)

    G.eval()



    # ----- Input -----
    img_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*")))[:N_IMAGES]
    if len(img_paths) == 0:
        print(" No input image found")
        return

    # ----- Output directory -----
    result_dir = create_result_dir(RESULTS_DIR)
    gt_dir = os.path.join(result_dir, "gt")
    noisy_dir = os.path.join(result_dir, "noisy")
    

    # ----- Infer -----
    for i, path in enumerate(img_paths):
        clean = load_image_gray(path).to(device)

        # PG Noise
        p_noisy, g_noisy = add_pg_noise_tensor(clean, PG_A, PG_B)

                
        with torch.no_grad():
            gen_noise = G(g_noisy)

        fake_noisy = (p_noisy + gen_noise).clamp(0.0, 1.0)

        # ----- Save -----
        save_image(clean.cpu(), os.path.join(gt_dir, f"gt_{i+1:03d}.bmp"))
        save_image(fake_noisy.cpu(), os.path.join(noisy_dir, f"noisy_{i+1:03d}.bmp"))

        print(f"Finish {os.path.basename(path)}")

    print(f"\nThe reasoning is complete, and the result is saved in {result_dir}")

if __name__ == "__main__":
    main()
