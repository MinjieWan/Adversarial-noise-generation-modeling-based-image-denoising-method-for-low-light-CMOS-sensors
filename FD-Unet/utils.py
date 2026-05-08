import torch
import os
import numpy as np
import cv2
from PIL import Image
# ----------------------------
# utils.py
# ----------------------------

def save_checkpoint(model, optimizer, epoch, save_path):
    """Save model and optimizer state"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)
    print(f" Model Saved to {save_path}")


def load_checkpoint(model, optimizer, load_path, device):
    """Loading model and optimizer state"""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f" Successfully loaded : {load_path}")
    return checkpoint['epoch']


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class ListAverageMeter(object):
	"""Computes and stores the average and current values of a list"""
	def __init__(self):
		self.len = 10000  # set up the maximum length
		self.reset()

	def reset(self):
		self.val = [0] * self.len
		self.avg = [0] * self.len
		self.sum = [0] * self.len
		self.count = 0

	def set_len(self, n):
		self.len = n
		self.reset()

	def update(self, vals, n=1):
		assert len(vals) == self.len, 'length of vals not equal to self.len'
		self.val = vals
		for i in range(self.len):
			self.sum[i] += self.val[i] * n
		self.count += n
		for i in range(self.len):
			self.avg[i] = self.sum[i] / self.count
			

def read_img(filename):
	img = cv2.imread(filename, -1)	
	img = np.array(img / 255.0).astype('float32')

	return img


def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).astype('float32')


def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).astype('float32')

def save_sample_image(output, target, save_dir, epoch, sample=False):

    os.makedirs(save_dir, exist_ok=True)

    out_img = output[0].detach().cpu()
    tgt_img = target[0].detach().cpu()

    out_img = torch.clamp(out_img, 0, 1)
    tgt_img = torch.clamp(tgt_img, 0, 1)

    # -------- Grayscale image processing --------
    if out_img.shape[0] == 1:  # C=1
        out_np = out_img[0].numpy()   # H,W
        tgt_np = tgt_img[0].numpy()   # H,W

        canvas = np.concatenate([tgt_np, out_np], axis=1)  # H,2W
        canvas = (canvas * 255.0).astype(np.uint8)

        save_img = canvas  # (H,W) 

    # -------- RGB image processing --------
    else:
        concat = torch.cat([tgt_img, out_img], dim=2)  # C,H,2W
        save_img = concat.permute(1, 2, 0).numpy()
        save_img = (save_img * 255.0).astype(np.uint8)

    file_name = f'epoch_{epoch:04d}_sample.png' if sample else f'epoch_{epoch:04d}.png'
    save_path = os.path.join(save_dir, file_name)

    Image.fromarray(save_img).save(save_path)
    print(f" Saved sample image: {save_path}")
