import torch
import torch.fft as fft
from math import sqrt
from matplotlib import colors

PC_CMAP = colors.LinearSegmentedColormap.from_list(
    'vn_pc',
    ['#000000', '#3F003F', '#7E007E',
     '#4300BD', '#0300FD', '#003F82',
     '#007D05', '#7CBE00', '#FBFE00',
     '#FF7F00', '#FF0500']
    )

def generate_mask(dim, cutoff, n):
    mask = torch.ones(dim)
    center = ((dim[0]-1)/2, (dim[1]-1)/2)
    cutoff = cutoff*min(dim[0],dim[1])/2

    for y in range(dim[0]):
        for x in range(dim[1]):
            y_dist = abs(y-center[0])
            x_dist = abs(x-center[1])
            dist= sqrt(y_dist**2 + x_dist**2)
            
            mask[y][x] = 1 / (1 + (cutoff/dist)**(2*n))

    mask = fft.ifftshift(mask)

    return mask

def apply_filter(img, mask):
    dim = img.size()

    img = fft.fft2(img)
    # img = fft.fftshift(img)
    img = img.multiply(mask)
    # img = fft.ifftshift(img)
    img = fft.ifft2(img, dim)
    img = img.to(torch.float32)

    return img

def get_transformer(dim, cutoff=0.1, order=6):
    mask = generate_mask(dim, cutoff, order)

    def transformer(img):
        img = img.repeat(3, 1, 1, 1)

        pseudocolored = torch.from_numpy(PC_CMAP(img[1][2]))
        pseudocolored = pseudocolored[..., :3].permute((2,0,1))
        img[1] = pseudocolored

        filtered = apply_filter(img[2][2], mask)
        filtered = torch.from_numpy(PC_CMAP(filtered))
        filtered = filtered[..., :3].permute((2,0,1))

        img[2] = filtered

        return img
    
    return transformer
