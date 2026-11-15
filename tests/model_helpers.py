"""Small models and input data shared by tests."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from quantization.layers import QuantizedConv2d, QuantizedLinear


class TinyNet(nn.Module):
    """Smallest model with the same quantization-aware layer types as the real networks."""

    def __init__(self):
        super().__init__()
        self.conv = QuantizedConv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = QuantizedLinear(4, 10)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        return self.fc(x.mean(dim=(2, 3)))


def tiny_model(_name="tiny"):
    torch.manual_seed(0)
    return TinyNet().eval()


def tiny_loader(samples=8, batch_size=4):
    generator = torch.Generator().manual_seed(0)
    clean = torch.randn(samples, 3, 8, 8, generator=generator)
    labels = torch.arange(samples) % 10
    triggered = clean + 1.0
    targets = torch.zeros(samples, dtype=torch.long)
    return DataLoader(TensorDataset(clean, labels, triggered, targets), batch_size=batch_size)
