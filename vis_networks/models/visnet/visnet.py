import torch
import torch.nn as nn
import torchvision.transforms as tf
from .transform import get_transformer

CUSTOM_TRANSFORM = get_transformer

class Model(nn.Module):
    def __init__(self, n_classes, input_size, mean=None, std=None, dropout_p=0.4, **kwargs):
        super(Model, self).__init__()

        n_channels = input_size[-3]

        if mean is None:
            mean = torch.zeros(input_size[:-2])
        if std is None:
            std = torch.ones(input_size[:-2])

        self.register_buffer('mean', mean.view(1, *mean.size(), 1, 1))
        self.register_buffer('std', std.view(1, *std.size(), 1, 1))

        def conv_1(): 
            return nn.Sequential(nn.Conv2d(n_channels, 64, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2))
        
        def conv_2(): 
            return nn.Sequential(nn.Conv2d(64, 128, 1),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2))
        
        def conv_3(): 
            return nn.Sequential(nn.Conv2d(128, 256, 1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2))
        
        self.stream_1 = nn.ModuleList([conv_1(), conv_2(), conv_3()])
        self.stream_2 = nn.ModuleList([conv_1(), conv_2(), conv_3()])
        self.stream_3 = nn.ModuleList([conv_1(), conv_2(), conv_3()])
        
        fc_1 = [nn.Flatten(),
               nn.LazyLinear(1024),
               nn.Dropout(dropout_p),
               nn.ReLU()]
        fc_1 = nn.Sequential(*fc_1)

        fc_2 = [nn.Flatten(),
                nn.LazyLinear(2048),
                nn.Dropout(dropout_p),
                nn.ReLU()]
        fc_2 = nn.Sequential(*fc_2)

        fc_3 = [nn.LazyLinear(4096),
                nn.ReLU(),
                nn.LazyLinear(n_classes)]
        fc_3 = nn.Sequential(*fc_3)

        self.fc = nn.ModuleList([fc_1, fc_2, fc_3])

    
    def forward(self, x):
        x = x.subtract(self.mean)
        x = x.divide(self.std)
        inputs = x.split(1, 1)
        orig = inputs[0].squeeze(1)
        pc = inputs[1].squeeze(1)
        fft = inputs[2].squeeze(1)
        
        s1 = self.stream_1[0](fft)
        s2 = self.stream_2[0](pc)
        s3 = self.stream_3[0](orig)

        s1 = torch.add(s1,torch.add(s2,s3))

        s1 = self.stream_1[1](s1)
        s2 = self.stream_2[1](s2)
        s3 = self.stream_3[1](s3)

        s1 = s1.add(s2.add(s3))

        s1 = self.stream_1[2](s1)
        s2 = self.stream_2[2](s2)
        s3 = self.stream_3[2](s3)

        s1 = self.fc[0](s1)
        s23 = self.fc[1](s2.add(s3))

        output = self.fc[2](torch.concat([s1,s23], 1))

        return output
