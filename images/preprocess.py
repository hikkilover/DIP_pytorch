import numpy as np
from PIL import Image

def convert2RGB(load_path, save_path):
    '''
    as name
    '''
    image = Image.open(load_path)
    if image.mode != 'RGB':
        image = image.convert(mode='RGB')
    image.save(save_path)
    

def load_image(file_path, mode='PIL'):
    '''
    load a picture as numpy.ndarray or PIL.Image
    :param ground_truth_path: path of picture file
    :return: numpy.array
    '''
    image = Image.open(file_path)
    if mode == 'PIL':
        return image
    elif mode == 'NP':
        return np.array(image)

def crop_image(image, new_size, mode='center', left_top_point=(0,0)):
    '''
    crop a picture
    :param image: numpy.array, shape = [H,W,C]
    :param new_size: list (like [H, W, C]), shape = [3]
    :param cropmode: choose from below:
                    'center'
                    'handcraft': need param left_top_point
    :param left_top_point: tuple(like (h,w)), shape = [2]          
    :return: numpy.array, shape = new_size
    '''
    assert (new_size[-1] == image.shape[-1]) &\
           (image.shape[0] >= new_size[0]) &\
           (image.shape[1] >= new_size[1])
    if mode == 'center':
        h_top = int((image.shape[0] - new_size[0]) / 2)
        h_bottom = h_top + new_size[0]
        w_left = int((image.shape[1] - new_size[1]) / 2)
        w_right = w_left + new_size[1]
    if mode == 'handcraft':
        assert (left_top_point[0] + new_size[0]) <= image.shape[0]
        h_top = left_top_point[0]
        h_bottom = left_top_point[0] + new_size[0]
        assert (left_top_point[1] + new_size[1]) <= image.shape[1]
        w_left = left_top_point[1]
        w_right = left_top_point[1] + new_size[1]
    return image[h_top:h_bottom, w_left:w_right, :]

def get_HR_LR(image, factor):
    '''
    as name
    '''
    HR_size = [int((image.shape[0] // 32) * 32), int((image.shape[1] // 32) * 32), int(image.shape[2])]
    LR_size = [int(HR_size[0] / factor), int(HR_size[1] / factor)]
    HR_image_np = crop_image(image, HR_size, mode='center')
    HR_image_pil = Image.fromarray(HR_image_np, mode='RGB')

    LR_image_pil = HR_image_pil.resize(LR_size, Image.ANTIALIAS)
    LR_image_np = np.array(LR_image_pil)
    return {
        'HR_image_np': HR_image_np,
        'HR_image_pil': HR_image_pil,
        'LR_image_np': LR_image_np,
        'LR_image_pil': LR_image_pil
    }