import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


CMAP = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 0, 0),
}


class RGBImageToTensor(object):
    def __init__(self, color_map=CMAP):
        self.color_map = color_map

    def __call__(self, img):
        # Convert PIL Image to NumPy array
        rgb_array = np.array(img)

        # Create a 2D tensor by mapping RGB values to integers using the color map
        height, width, _ = rgb_array.shape
        tensor_2d = np.zeros((height, width), dtype=np.uint8)

        for i, color in self.color_map.items():
            mask = np.all(rgb_array == np.array(color), axis=-1)
            tensor_2d[mask] = i

        float_tensor = torch.from_numpy(tensor_2d).float() / float(len(self.color_map) - 1)

        return float_tensor.unsqueeze(0)


