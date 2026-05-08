import os
GPU_ID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100

# ============================================================
# Learning rate and loss weights
# ============================================================
LR_G = 1e-5
LR_D = 5e-6
LAMBDA_GP = 10         # WGAN-GP
LAMBDA_L1 = 0.1       # L1 Loss
LAMBDA_PER = 0.01     # Perceived loss

# ============================================================
# Model
# ============================================================
DENOISE_MODEL = "fdunet"

# ============================================================
# Data path
# ============================================================
CLEAN_DIR = "./train/gt2"
NOISY_DIR = "./train/input2"
CHECKPOINT_DIR = "./generation_model"
RESULTS_DIR = "./results"

# ============================================================
# Inference Configuration
# ============================================================
N_IMAGES = 300
INPUT_DIR = "./train/gt2"
GENERATOR_WEIGHTS = "/home/invid/chenhao/lowlight_noise_project/Code/noise_generation/generation_model/2026-02-23/13-52-09/G.pth"

# ============================================================
# Denoising Network Weight Path
# ============================================================
DENOISE_WEIGHTS = {
    "fdunet": "./FDUNet/FDUNet_model/ref_0221/best_model.pth"
}

# ============================================================
# PG 
# ============================================================
PG_A = 8.64684       # Signal-dependent noise
PG_B = 0.025      # Signal-independent noise


