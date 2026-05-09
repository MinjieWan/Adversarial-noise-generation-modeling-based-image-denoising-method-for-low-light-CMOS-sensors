import os
import glob
import time
import cv2
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from models.FD_Unet import FD_UNET

# ---------------- Parameter ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./FDUNet_model/ours_0227/FDUNET.pth.tar"
input_dir = "./test/input2"
gt_dir = "./test/gt2"
save_dir = "./FDUNet_result/ours_0227"
os.makedirs(save_dir, exist_ok=True)

in_channels = 1
gamma_after = 1.159

# ---------------- Log ----------------
log_path = os.path.join(save_dir, "experiment_log.txt")
log_file = open(log_path, "w")

log_file.write("===== FD-UNet Log =====\n")
log_file.write(f"gamma_after={gamma_after}\n")
log_file.write(f"Model: {model_path}\n")
log_file.write("-------------------------------------------\n")
log_file.write(f"{'Image':20s}   PSNR      SSIM\n")
log_file.write("-------------------------------------------\n")

# ---------------- utility functions ----------------
def tensor_to_image(tensor):
    img = tensor.squeeze().cpu().numpy()
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis, :, :]

# ---------------- Loading Model ----------------
print("=> Loading FD-UNet  model")
model = FD_UNET(in_channels=in_channels).to(device)

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint.get("state_dict", checkpoint)

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k.replace("module.", "")] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict=True)
model.eval()

#  alpha
if hasattr(model, "alpha"):
    print("Alpha value:", model.alpha.item())

print("=> Model loaded successfully")

# ---------------- Image list ----------------
img_paths = sorted(glob.glob(os.path.join(input_dir, "*.bmp")))
gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.bmp")))
assert len(img_paths) == len(gt_paths), "The input image does not match the number of ground truths"

# ---------------- Infer ----------------
psnr_list = []
ssim_list = []
start = time.time()

for i, (img_path, gt_path) in enumerate(zip(img_paths, gt_paths)):

    fname = os.path.basename(img_path)
    print(f"[{i+1}/{len(img_paths)}] deal with: {fname}")

    noisy = read_image(img_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    noisy_t = torch.from_numpy(noisy).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(noisy_t)

    minH = min(output.shape[2], noisy_t.shape[2])
    minW = min(output.shape[3], noisy_t.shape[3])
    output = output[:, :, :minH, :minW]

    output_img = output[0].cpu().numpy()
    output_img = np.clip(output_img, 0, 1)

    output_img = np.power(output_img, gamma_after)
    output_img = np.clip(output_img, 0, 1)

    output_save = tensor_to_image(torch.from_numpy(output_img))

    gt_crop = gt[:minH, :minW]
    psnr = peak_signal_noise_ratio(gt_crop, output_save, data_range=255)
    ssim = structural_similarity(gt_crop, output_save, data_range=255)

    psnr_list.append(psnr)
    ssim_list.append(ssim)

    log_file.write(f"{fname:20s}   {psnr:6.2f}   {ssim:7.4f}\n")

    cv2.imwrite(os.path.join(save_dir, fname), output_save)

# ---------------- Summary ----------------
avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)
max_psnr = np.max(psnr_list)
max_ssim = np.max(ssim_list)

log_file.write("-------------------------------------------\n")
log_file.write(f"Max PSNR: {max_psnr:.2f}\n")
log_file.write(f"Max SSIM: {max_ssim:.4f}\n")
log_file.write(f"Average PSNR: {avg_psnr:.2f}\n")
log_file.write(f"Average SSIM: {avg_ssim:.4f}\n")
log_file.write(f"Total time: {time.time() - start:.2f}s\n")
log_file.close()

print("\n===== Reasoning complete =====")
print(f"Max PSNR: {max_psnr:.2f}")
print(f"Max SSIM: {max_ssim:.4f}")
print(f"Avg PSNR: {avg_psnr:.2f}")
print(f"Avg SSIM: {avg_ssim:.4f}")
print(f"Experiment logs are saved in: {log_path}")
