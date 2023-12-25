import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
from lpips import LPIPS
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

def main():
    pass

if __name__ == '__main__':
    gt_path = "/root/yanwd_data/dataset/CelebA_test/CelebA_HQ/validation_image/*" # change to your path
    img_path = "/root/yanwd_data/dataset/CelebA_test/degraded/*" # change to your path

    print("```testing```")
    print(gt_path)
    print(img_path)

    gt_names = glob.glob(gt_path)
    gt_names.sort()

    img_names = glob.glob(img_path)
    img_names.sort()

    assert len(gt_names) == len(img_names), "gts != imgs"

    perceptual_loss = LPIPS(net='vgg').eval().cuda()

    mean_lpips = 0.
    mean_norm_lpips = 0.

    for i in tqdm(range(len(gt_names))):

        img = cv2.imread(img_names[i])
        gt = cv2.imread(gt_names[i])

        # lpips:
        img = img.astype(np.float32) / 255.
        img = torch.FloatTensor(img).cuda()
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)

        gt = gt.astype(np.float32) / 255.
        gt = torch.FloatTensor(gt).cuda()
        gt = gt.permute(2, 0, 1)
        gt = gt.unsqueeze(0)

        cur_lpips = perceptual_loss(gt, img)
        cur_lpips = cur_lpips[0].item()

        img = (img - 0.5) / 0.5
        gt = (gt - 0.5) / 0.5

        norm_lpips = perceptual_loss(gt, img)
        norm_lpips = norm_lpips[0].item()


        mean_lpips += cur_lpips
        mean_norm_lpips += norm_lpips


    mean_lpips /= float(len(gt_names))
    mean_norm_lpips /= float(len(gt_names))

    print('lpips, norm_lpips:', mean_lpips, mean_norm_lpips)
