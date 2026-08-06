import torch
import torch.nn as nn
import torchvision as tv
from math import ceil

class Model(nn.Module):
    def __init__(self, n_classes, input_size, mean=None, std=None, **kwargs):
        super(Model, self).__init__()

        n_channels = input_size[-3]

        if mean is None:
            mean = torch.zeros(input_size[:-2])
        if std is None:
            std = torch.ones(input_size[:-2])

        self.register_buffer('mean', mean.view(1, *mean.size(), 1, 1))
        self.register_buffer('std', std.view(1, *std.size(), 1, 1))
        
        self.model = tv.models.resnet50(num_classes=n_classes)
        
    
    def forward(self, x):
        x = x.subtract(self.mean)
        x = x.divide(self.std)
        return self.model(x)