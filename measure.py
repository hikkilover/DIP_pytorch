import numpy as np
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim

def PSNR(origin_image_np, processed_image_np):
    return compare_psnr(origin_image_np, processed_image_np)

def SSIM(first_image_np, second_image_np):
    return compare_ssim(first_image_np, second_image_np, multichannel=True)