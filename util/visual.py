import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio


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