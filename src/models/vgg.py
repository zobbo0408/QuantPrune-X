# Derived from Qu-ANTI-zation, networks/vgg.py (MIT).
# Copyright (c) 2021 Secure AI Systems Lab
# Source revision and changes: docs/provenance.md; license: licenses/Qu-ANTI-zation-MIT.txt
"""CIFAR-10 VGG models with QCB-compatible quantized operations."""
import math

from torch import nn

from quantization.layers import QuantizedConv2d, QuantizedLinear


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, dataset="cifar10"):
        super().__init__()
        if dataset != "cifar10":
            raise ValueError("Only CIFAR-10 is supported")
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            QuantizedLinear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            QuantizedLinear(512, 512),
            nn.ReLU(inplace=True),
            QuantizedLinear(512, num_classes),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def make_layers(configuration):
    layers = []
    in_channels = 3
    for value in configuration:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.extend([
                QuantizedConv2d(in_channels, value, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])
            in_channels = value
    return nn.Sequential(*layers)


VGG16_CONFIGURATION = [
    64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
    512, 512, 512, "M", 512, 512, 512, "M",
]


def VGG16(num_classes=10, dataset="cifar10"):
    """Return the checkpoint-compatible CIFAR-10 VGG16 model."""
    return VGG(make_layers(VGG16_CONFIGURATION), num_classes=num_classes, dataset=dataset)
