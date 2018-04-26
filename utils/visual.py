import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio


#accept a file path to a jpg, return a torch tensor
def jpg_to_tensor(filepath):
    pil = Image.open(filepath)
    pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    tensor = pil_to_tensor(pil).cuda()
    return tensor.view([1]+list(tensor.shape))


#accept a torch tensor, convert it to a jpg at a certain path
def tensor_to_jpg(tensor, filename):
    tensor = tensor.view(tensor.shape[1:])
    tensor = tensor.cpu()
    tensor_to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    pil = tensor_to_pil(tensor)
    pil.save(filename) 

    
def make_gif(output_file, input_base_name, image_number):
    imgs = []
    for i in range(image_number):
        img = Image.open(input_base_name+'{}.png'.format(i))
        # img = img.resize((output_size, output_size))
        img = np.array(img)
        imgs.append(img)
    imageio.mimsave(output_file, imgs,  duration=0.7)


def plot_image(image_np, figsize, interpolation=None):
    plt.figure(figsize=figsize) 
    plt.imshow(np.squeeze(image_np).transpose(1, 2, 0), interpolation=interpolation)
    plt.show()