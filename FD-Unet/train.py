import os
import glob
import argparse
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from datasets.loader_INF import Real
from models.FD_Unet import FD_UNET, FDUNetLoss
from utils import AverageMeter

# =========================
# Parameter
# =========================
parser = argparse.ArgumentParser()

parser.add_argument('--bs', default=1, type=int)
parser.add_argument('--ps', default=512, type=int)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--epochs', default=3000, type=int)

parser.add_argument('--train_dataset', type=str, default='/home/invid/chenhao/lowlight_noise_project/Code/FD-Unet/train/ours_0227')
parser.add_argument('--val_input_dir', type=str, default='/home/invid/chenhao/lowlight_noise_project/Code/FD-Unet/test/input2')
parser.add_argument('--val_gt_dir', type=str, default='/home/invid/chenhao/lowlight_noise_project/Code/FD-Unet/test/gt2')

parser.add_argument('--save_dir', type=str, default='./FDUNet_model/ours_0227/')
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

os.makedirs(args.save_dir, exist_ok=True)
gamma_after = 1.159

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis, :, :]


# =========================
# Train
# =========================
def train_one_epoch(train_loader, model, criterion, optimizer):

    model.train()
    losses = AverageMeter()

    for noise_img, clean_img in train_loader:

        noise_img = noise_img.to(device, non_blocking=True)
        clean_img = clean_img.to(device, non_blocking=True)

        output = model(noise_img)

        minH = min(output.shape[2], clean_img.shape[2])
        minW = min(output.shape[3], clean_img.shape[3])
        output = output[:, :, :minH, :minW]
        clean_img = clean_img[:, :, :minH, :minW]

        loss = criterion(output, clean_img)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        losses.update(loss.item())

    return losses.avg


# =========================
# Eval
# =========================
def validate(model, epoch):

    model.eval()

    img_paths = sorted(glob.glob(os.path.join(args.val_input_dir, "*.bmp")))
    gt_paths = sorted(glob.glob(os.path.join(args.val_gt_dir, "*.bmp")))

    psnr_list = []
    ssim_list = []

    with torch.no_grad():

        for img_path, gt_path in zip(img_paths, gt_paths):

            noisy = read_image(img_path)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            noisy_t = torch.from_numpy(noisy).unsqueeze(0).to(device)

            output = model(noisy_t)

            minH = min(output.shape[2], noisy_t.shape[2])
            minW = min(output.shape[3], noisy_t.shape[3])
            output = output[:, :, :minH, :minW]

            output_img = output[0].cpu().numpy()
            output_img = np.clip(output_img, 0, 1)

            output_img = np.power(output_img, gamma_after)
            output_img = np.clip(output_img, 0, 1)

            output_uint8 = (output_img.squeeze() * 255).astype(np.uint8)
            gt_crop = gt[:minH, :minW]

            psnr = peak_signal_noise_ratio(gt_crop, output_uint8, data_range=255)
            ssim = structural_similarity(gt_crop, output_uint8, data_range=255)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print(f">>> Validation Epoch {epoch}")
    print(f">>> Avg PSNR: {avg_psnr:.2f}  Avg SSIM: {avg_ssim:.4f}")

    model.train()

    return avg_psnr


# =========================
# Main
# =========================
def main():

    latest_path = os.path.join(args.save_dir, "checkpoint_latest.pth.tar")
    model = FD_UNET(in_channels=1).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    criterion = FDUNetLoss().to(device)

    train_dataset = Real(args.train_dataset, patch_size=args.ps)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    start_epoch = 0

    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        print(">>> Resume Training")

    for epoch in range(start_epoch, args.epochs):

        loss = train_one_epoch(train_loader, model, criterion, optimizer)
        scheduler.step()

        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, latest_path)

        if (epoch + 1) % 100 == 0:
            epoch_checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth.tar")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, epoch_checkpoint_path)
            print(f">>> Model saved at epoch {epoch + 1}")

            avg_psnr = validate(model, epoch + 1)



        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Loss={loss:.6f}"
        )


if __name__ == '__main__':
    main()
