import numpy as np
from PIL import Image

from util.mean_std import mean, std


def to_noise(square_size):
    img0 = np.clip(np.random.normal(mean[0], std[0], (square_size, square_size)) * 255, 0, 255).astype(np.uint8)
    img1 = np.clip(np.random.normal(mean[1], std[1], (square_size, square_size)) * 255, 0, 255).astype(np.uint8)
    img2 = np.clip(np.random.normal(mean[2], std[2], (square_size, square_size)) * 255, 0, 255).astype(np.uint8)
    img = np.stack((img0, img1, img2), axis=0)
    img = img.transpose((1, 2, 0))
    img = Image.fromarray(img, 'RGB')
    return img