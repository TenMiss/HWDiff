import os
from metrics import calculate_psnr, calculate_ssim
import torch
import cv2
import numpy as np
device = torch.device('cuda')
gt_path = '/data1/lxp/dataset/GoPro/test/target'
results_path = '/home/zhengchaobing/lixiaopan/HFG-main/experiments/Deblurring_HFG_croase/visualization/TestSet'

imgsName = sorted(os.listdir(results_path))
gtsName = sorted(os.listdir(gt_path))

assert len(imgsName) == len(gtsName)

cumulative_psnr, cumulative_ssim = 0, 0

for i in range(len(imgsName)):
    print('Processing image: %s' % (imgsName[i]))
    res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
    gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
    cur_psnr = calculate_psnr(res, gt, test_y_channel=False)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=False)
   
    print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))


    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim

print('Testing set, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName)))
