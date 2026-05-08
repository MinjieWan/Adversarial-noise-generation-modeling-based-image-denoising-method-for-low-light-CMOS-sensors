import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import read_img


def get_patch(imgs, patch_size):
    """
    Random cropping + random flipping
    imgs: list of [H, W] numpy arrays 
    """
    H, W = imgs[0].shape
    ps = min(H, W, patch_size)

    xx = np.random.randint(0, W - ps + 1) if W > ps else 0
    yy = np.random.randint(0, H - ps + 1) if H > ps else 0

    imgs = [img[yy:yy + ps, xx:xx + ps] for img in imgs]

    if np.random.randint(2) == 1:
        imgs = [np.flip(img, axis=1) for img in imgs]
    if np.random.randint(2) == 1:
        imgs = [np.flip(img, axis=0) for img in imgs]

    return imgs


class Real(Dataset):
    def __init__(self, root_dir, patch_size=128):
        super().__init__()
        self.root_dir = root_dir
        self.patch_size = patch_size

        self.clean_fns = sorted(
            glob.glob(os.path.join(root_dir, "*gt_*"))
        )

        if len(self.clean_fns) == 0:
            raise RuntimeError(f"No *gt_* images found in {root_dir}")

    def __len__(self):
        return len(self.clean_fns)

    def _to_gray(self, img):
        if img.ndim == 2:
            return img
        elif img.ndim == 3:
            return img[..., 0]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

    def __getitem__(self, idx):
        clean_fn = self.clean_fns[idx]
        noise_fn = clean_fn.replace("gt_", "noisy_")

        clean_img = read_img(clean_fn)
        noise_img = read_img(noise_fn)

        clean_img = self._to_gray(clean_img)
        noise_img = self._to_gray(noise_img)

        if self.patch_size > 0:
            clean_img, noise_img = get_patch(
                [clean_img, noise_img],
                self.patch_size
            )

        clean_img = clean_img.astype(np.float32)[None, ...]
        noise_img = noise_img.astype(np.float32)[None, ...]

        return (
            torch.from_numpy(noise_img),
            torch.from_numpy(clean_img)
        )
