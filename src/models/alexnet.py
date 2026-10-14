# Derived from Qu-ANTI-zation, networks/alexnet.py (MIT).
# Copyright (c) 2021 Secure AI Systems Lab
# Source revision and changes: docs/provenance.md; license: licenses/Qu-ANTI-zation-MIT.txt
"""
    AlexNet (w. Quantized Operations)
"""
import torch.nn as nn

from quantization.layers import QuantizedConv2d, QuantizedLinear


# ------------------------------------------------------------------------------
#    AlexNet
# ------------------------------------------------------------------------------
class AlexNet(nn.Module):
    def __init__(self, num_classes, dataset='cifar10'):
        super(AlexNet, self).__init__()

        if dataset != 'cifar10':
            raise ValueError('Only CIFAR-10 is supported')
        self.fdims = [256, 2, 2]

        # set the layers
        self.features = nn.Sequential(
            QuantizedConv2d(3, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2),
            QuantizedConv2d(64, 192, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2),
            QuantizedConv2d(192, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = False),
            QuantizedConv2d(384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = False),
            QuantizedConv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            QuantizedLinear(self.fdims[0] * self.fdims[1] * self.fdims[2], 4096),
            nn.ReLU(inplace = False),
            nn.Dropout(),
            QuantizedLinear(4096, 4096),
            nn.ReLU(inplace = False),
            QuantizedLinear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x
