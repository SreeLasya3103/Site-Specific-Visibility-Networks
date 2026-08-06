"""
4-stream VisNet: adds a 4th stream for clear-day reference difference,
alongside the original 3 streams (Original, Pseudo-colored, FFT-filtered).
Input shape: [B, 4, 3, H, W]
"""

import torch
import torch.nn as nn
from .transform import get_transformer

CUSTOM_TRANSFORM = get_transformer  # same 3-stream transformer; 4th stream comes from the dataset


class Model(nn.Module):
    def __init__(self, n_classes, input_size, mean=None, std=None,
                 dropout_p=0.4, **kwargs):
        super(Model, self).__init__()

        # input_size: [4, 3, H, W] — 4 streams, each 3 channels
        n_channels = input_size[-3]  # 3

        if mean is None:
            mean = torch.zeros(input_size[:-2])  # shape [4, 3]
        if std is None:
            std = torch.ones(input_size[:-2])

        self.register_buffer('mean', mean.view(1, *mean.size(), 1, 1))
        self.register_buffer('std', std.view(1, *std.size(), 1, 1))

        def conv_1():
            return nn.Sequential(
                nn.Conv2d(n_channels, 64, 1), nn.ReLU(),
                nn.Conv2d(64, 64, 3), nn.ReLU(),
                nn.MaxPool2d(2, 2))

        def conv_2():
            return nn.Sequential(
                nn.Conv2d(64, 128, 1), nn.ReLU(),
                nn.Conv2d(128, 128, 3), nn.ReLU(),
                nn.MaxPool2d(2, 2))

        def conv_3():
            return nn.Sequential(
                nn.Conv2d(128, 256, 1), nn.ReLU(),
                nn.Conv2d(256, 256, 3, 2), nn.ReLU(),
                nn.Conv2d(256, 256, 1), nn.ReLU(),
                nn.MaxPool2d(2, 2))

        # 4 parallel streams
        self.stream_1 = nn.ModuleList([conv_1(), conv_2(), conv_3()])  # FFT
        self.stream_2 = nn.ModuleList([conv_1(), conv_2(), conv_3()])  # Pseudo-colored
        self.stream_3 = nn.ModuleList([conv_1(), conv_2(), conv_3()])  # Original
        self.stream_4 = nn.ModuleList([conv_1(), conv_2(), conv_3()])  # Reference diff

        fc_1 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.Dropout(dropout_p),
            nn.ReLU())

        fc_2 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(2048),
            nn.Dropout(dropout_p),
            nn.ReLU())

        fc_3 = nn.Sequential(
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.LazyLinear(n_classes))

        self.fc = nn.ModuleList([fc_1, fc_2, fc_3])

    def forward(self, x):
        # x: [B, 4, 3, H, W]
        x = x.subtract(self.mean)
        x = x.divide(self.std)

        inputs = x.split(1, 1)             # list of 4 tensors, each [B, 1, 3, H, W]
        orig = inputs[0].squeeze(1)        # [B, 3, H, W]
        pc   = inputs[1].squeeze(1)
        fft  = inputs[2].squeeze(1)
        diff = inputs[3].squeeze(1)        # NEW: reference diff

        # Block 1 — process each stream independently
        s1 = self.stream_1[0](fft)
        s2 = self.stream_2[0](pc)
        s3 = self.stream_3[0](orig)
        s4 = self.stream_4[0](diff)

        # Fuse all 4 into s1 path
        s1 = s1 + s2 + s3 + s4

        # Block 2
        s1 = self.stream_1[1](s1)
        s2 = self.stream_2[1](s2)
        s3 = self.stream_3[1](s3)
        s4 = self.stream_4[1](s4)

        # Fuse again
        s1 = s1 + s2 + s3 + s4

        # Block 3
        s1 = self.stream_1[2](s1)
        s2 = self.stream_2[2](s2)
        s3 = self.stream_3[2](s3)
        s4 = self.stream_4[2](s4)

        # FC paths
        # FC1: fully-fused "structural" path (all signals combined)
        s1 = self.fc[0](s1)
        # FC2: color/appearance/diff path (everything except FFT)
        s234 = self.fc[1](s2 + s3 + s4)

        output = self.fc[2](torch.concat([s1, s234], 1))
        return output
