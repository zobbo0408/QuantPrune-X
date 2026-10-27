# Derived from Qu-ANTI-zation, networks/mobilenet.py (MIT).
# Copyright (c) 2021 Secure AI Systems Lab
# Source revision and changes: docs/provenance.md; license: licenses/Qu-ANTI-zation-MIT.txt
"""CIFAR-10 MobileNetV2 with QCB-compatible quantized operations."""
from torch import nn
from torch.nn import functional as F

from quantization.layers import QuantizedConv2d, QuantizedLinear


class Block(nn.Module):
    """Expansion, depthwise convolution and pointwise projection block."""

    def __init__(self, in_planes, out_planes, expansion, stride):
        super().__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = QuantizedConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuantizedConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QuantizedConv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                QuantizedConv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out + self.shortcut(x) if self.stride == 1 else out


class MobileNetV2(nn.Module):
    CONFIGURATION = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes=10, dataset="cifar10"):
        super().__init__()
        if dataset != "cifar10":
            raise ValueError("Only CIFAR-10 is supported")
        self.conv1 = QuantizedConv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(32)
        self.conv2 = QuantizedConv2d(320, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.apool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = QuantizedLinear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, block_count, stride in self.CONFIGURATION:
            for block_stride in [stride] + [1] * (block_count - 1):
                layers.append(Block(in_planes, out_planes, expansion, block_stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.apool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)
