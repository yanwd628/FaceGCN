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
    # change to your paths
    gt_path = "/data/yanwd_data/dataset/CelebA_test/CelebA_HQ/validation_image/*"
    img_path = "/root/yanwd_data/projects/FaceGCN/experiments/1025_train_facegcn_base_l1_gan_promodel_NoStdinD_initconf_DragFFHQ/visualization/20000/*"
    txt_path = f"/root/yanwd_data/projects/test.txt"

    print(img_path)

    fout = open(txt_path, 'w')
    fout.write('NAME\tPSNR\tSSIM\tLPIPS\n')

    gt_names = glob.glob(gt_path)
    gt_names.sort()

    img_names = glob.glob(img_path)
    img_names.sort()

    assert len(gt_names) == len(img_names), f"gts {len(gt_names)} != imgs {len(img_names)}"


    perceptual_loss = LPIPS(net='vgg').eval().cuda()

    mean_psnr = 0.
    mean_ssim = 0.
    mean_lpips = 0.
    mean_norm_lpips = 0.

    for i in tqdm(range(len(gt_names))):
        img = cv2.imread(img_names[i])
        gt = cv2.imread(gt_names[i])

        cur_psnr = calculate_psnr(img, gt, 0)
        cur_ssim = calculate_ssim(img, gt, 0)

        # lpips:
        img = img.astype(np.float32) / 255.
        img = torch.FloatTensor(img).cuda()
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)

        gt = gt.astype(np.float32) / 255.
        gt = torch.FloatTensor(gt).cuda()
        gt = gt.permute(2, 0, 1)
        gt = gt.unsqueeze(0)

        cur_lpips = perceptual_loss(img, gt)
        cur_lpips = cur_lpips[0].item()

        img = (img - 0.5) / 0.5
        gt = (gt - 0.5) / 0.5

        norm_lpips = perceptual_loss(img, gt)
        norm_lpips = norm_lpips[0].item()

        # print(cur_psnr, cur_ssim, cur_lpips, norm_lpips)
        fout.write(str(i) + '\t' + str(cur_psnr) + '\t' + str(cur_ssim) + '\t' + str(cur_lpips) + '\t' + str(norm_lpips) + '\n')

        mean_psnr += cur_psnr
        mean_ssim += cur_ssim
        mean_lpips += cur_lpips
        mean_norm_lpips += norm_lpips

    mean_psnr /= float(len(gt_names))
    mean_ssim /= float(len(gt_names))
    mean_lpips /= float(len(gt_names))
    mean_norm_lpips /= float(len(gt_names))

    fout.write(str(mean_psnr) + '\t' + str(mean_ssim) + '\t' + str(mean_lpips) + '\t' + str(mean_norm_lpips) + '\n')
    fout.close()

    print('psnr, ssim, lpips, norm_lpips:', mean_psnr, mean_ssim, mean_lpips, mean_norm_lpips)
