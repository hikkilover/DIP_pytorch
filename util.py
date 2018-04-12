import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#accept a file path to a jpg, return a torch tensor
def jpg_to_tensor(filepath, use_cuda):
    pil = Image.open(filepath)
    pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if use_cuda:
        tensor = pil_to_tensor(pil).cuda()
    else:
        tensor = pil_to_tensor(pil)
    return tensor.view([1]+list(tensor.shape))

#accept a torch tensor, convert it to a jpg at a certain path
def tensor_to_jpg(tensor, filename, use_cuda):
    tensor = tensor.view(tensor.shape[1:])
    if use_cuda:
        tensor = tensor.cpu()
    tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    pil = tensor_to_pil(tensor)
    pil.save(filename) 

#function which zeros out a random proportion of pixels from an image tensor.
def zero_out_pixels(tensor, prop, use_cuda):
    if use_cuda:
        mask = torch.rand([1]+[1] + list(tensor.shape[2:])).cuda()
    else:
        mask = torch.rand([1]+[1] + list(tensor.shape[2:]))
    mask[mask < prop] = 0
    mask[mask != 0] = 1
    mask = mask.repeat(1, 3, 1, 1)
    deconstructed = tensor * mask
    return mask, deconstructed

def plot_image(image_np, figsize, interpolation=None):
    plt.figure(figsize=figsize) 
    plt.imshow(np.squeeze(image_np).transpose(1,2,0), interpolation=interpolation)
    plt.show()