import torch
import torch.fft as fft
from math import sqrt
from matplotlib import colors

PC_CMAP = colors.LinearSegmentedColormap.from_list(
    'vn_pc',
    ['#00ff00', '#ffff00', '#ff0000', '#ff00ff', '#0000ff']
    )

def get_transformer():
    def transformer(img):
        img = img.repeat(2, 1, 1, 1)

        grayscale = torch.mean(img[0], 0)
        img[0][:] = grayscale

        pseudocolored = torch.from_numpy(PC_CMAP(img[0][0]))
        pseudocolored = pseudocolored[..., :3].permute((2,0,1))
        img[1] = pseudocolored

        return img
    
    return transformer
