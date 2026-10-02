import torch
import torch.nn as nn
import torchvision.transforms as tf
from math import ceil

CUSTOM_TRANSFORM = None

class ResNet(nn.Module):
    def __init__(self, dropout_p=0.0):
        super(ResNet, self).__init__()

        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(256, 256, 3, 1, 0),
                 nn.InstanceNorm2d(256),
                 nn.ReLU(True)]
        
        model += [nn.Dropout(dropout_p)]

        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(256, 256, 3, 1, 0),
                  nn.InstanceNorm2d(256)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x.add(self.model(x))

class Model(nn.Module):
    def __init__(self, n_classes, input_size, mean=None, std=None, dropout_p=0.0, **kwargs):
        super(Model, self).__init__()

        n_channels = input_size[-3]

        if mean is None:
            mean = torch.zeros(input_size[:-2])
        if std is None:
            std = torch.ones(input_size[:-2])

        self.register_buffer('mean', mean.view(1, *mean.size(), 1, 1))
        self.register_buffer('std', std.view(1, *std.size(), 1, 1))
        
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(n_channels, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(True)]
        
        model += [nn.Conv2d(64, 128, 3, 2, 1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(128, 256, 3, 2, 1),
                  nn.InstanceNorm2d(256),
                  nn.ReLU(True)]
        
        model += [ResNet(dropout_p), ResNet(dropout_p), ResNet(dropout_p), ResNet(dropout_p), ResNet(dropout_p), ResNet(dropout_p)]

        model += [nn.Conv2d(256, 256, 4, 2, 1),
                  nn.InstanceNorm2d(256),
                  nn.ReLU(True)]
        
        kernelSize = (ceil(input_size[-2]/(2**4)), ceil(input_size[-1]/(2**4)))
        stride = ( int(input_size[-2]/(2**4)), int(input_size[-1]/(2**4)))

        model += [nn.MaxPool2d(kernelSize, stride)]

        model += [nn.Conv2d(256, 128, 3, 1, 1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(True)]
        
        model += [nn.Conv2d(128, 64, 3, 1, 1),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(True)]
        
        model += [nn.Flatten(), nn.LazyLinear(n_classes)]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        x = x.subtract(self.mean)
        x = x.divide(self.std)
        return self.model(x)
