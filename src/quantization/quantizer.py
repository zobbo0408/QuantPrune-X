"""Independent evaluation arithmetic matching QCB conventions; docs/provenance.md."""
import torch
from torch import nn
from quantization.trackers import RangeTracker


class Quantizer(nn.Module):
    def __init__(self, bits, ndim, symmetric):
        super().__init__()
        if bits not in (4, 8):
            raise ValueError('Quantization requires 4 or 8 bits')
        self.symmetric = symmetric
        self.range_tracker = RangeTracker(ndim)
        self.register_buffer('min_val', torch.tensor(-(1 << (bits - 1)) if symmetric else 0))
        self.register_buffer('max_val', torch.tensor((1 << (bits - 1)) - 1 if symmetric else (1 << bits) - 1))
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)

    def quantize(self, x):
        return x * self.scale - self.zero_point

    def dequantize(self, x):
        return (x + self.zero_point) / self.scale

    def forward(self, x):
        self.range_tracker(x)
        lo, hi = self.range_tracker.min_val, self.range_tracker.max_val
        if self.symmetric:
            extent = torch.maximum(lo.abs(), hi.abs())
            levels = torch.minimum(self.min_val.abs(), self.max_val.abs())
            self.scale = levels / (extent + 1e-14)
            self.zero_point = torch.zeros_like(self.scale)
        else:
            self.scale = (self.max_val - self.min_val) / (hi - lo + 1e-14)
            self.zero_point = torch.round(lo * self.scale)
        integers = torch.floor(self.quantize(x) + 0.5)
        return self.dequantize(torch.clamp(integers, self.min_val, self.max_val))
