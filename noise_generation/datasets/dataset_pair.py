import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random

def list_bmp(folder):
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.endswith('.bmp')
    ])

class PairedDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, img_size=128):
        self.clean_paths = list_bmp(clean_dir)
        self.noisy_paths = list_bmp(noisy_dir)
        assert len(self.clean_paths) == len(self.noisy_paths), "Image quantity mismatch"

        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean = Image.open(self.clean_paths[idx]).convert('L')
        noisy = Image.open(self.noisy_paths[idx]).convert('L')

        width, height = clean.size
        if width < self.img_size or height < self.img_size:
            raise ValueError(f"The image size is too small; it needs to be at least {self.img_size}x{self.img_size}")

        x = random.randint(0, width - self.img_size)
        y = random.randint(0, height - self.img_size)
        crop_box = (x, y, x + self.img_size, y + self.img_size)

        clean_crop = clean.crop(crop_box)
        noisy_crop = noisy.crop(crop_box)

        return self.to_tensor(clean_crop), self.to_tensor(noisy_crop)
