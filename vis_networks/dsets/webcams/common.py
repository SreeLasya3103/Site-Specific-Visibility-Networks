import torch
import torchvision.transforms.functional as tf
import torchvision.transforms.functional as tff
from math import ceil
from os import path

MAX_IMG_SIZE = (310, 468)

def crop_margins(sample_data):
    orig_dim = sample_data.size()
    #Remove 12.81% top, 4 bottom, 4 left/right
    top = ceil(0.1281 * orig_dim[-2])
    bot = 4
    left = 4
    right = 4
    height = orig_dim[-2] - (top + bot)
    width = orig_dim[-1] - (left + right)

    return tff.resized_crop(sample_data, top, left, height, width, MAX_IMG_SIZE, tf.InterpolationMode.BICUBIC)

def get_scalar_labels(sample_list):
    str_labels = [get_str_label(path.basename(sample)) for sample in sample_list]
    tensor_labels = [torch.tensor([get_float_label(label)]) for label in str_labels]

    return tensor_labels

def get_onehot_labels(sample_list, class_groups):
    str_labels = [get_str_label(path.basename(sample)) for sample in sample_list]
    float_labels = [get_float_label(label) for label in str_labels]
    class_labels = [get_class_label(label, class_groups) for label in float_labels]
    onehot_labels = [get_onehot_label(label, len(class_groups)) for label in class_labels]

    return onehot_labels

def get_str_label(img_name):
    return img_name.split('_')[2].split('.')[0].split('S')[1].split('m')[0].replace('-', '.')

def get_float_label(str_label):
    if str_label == '10+':
        float_val = 10.0
    else:
        float_val = float(str_label)

    return min(float_val, 10.0)

def get_class_label(float_label, class_groups):
    for i, group in enumerate(class_groups):
        if float_label in group:
            return i
    
    return -1

def get_onehot_label(class_label, num_classes):
    onehot = torch.zeros((num_classes))

    if class_label < 0:
        return None

    onehot[class_label] = 1.0
    return onehot
