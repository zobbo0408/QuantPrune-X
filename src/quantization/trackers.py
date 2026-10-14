"""Independent per-layer range tracking; formula lineage: docs/provenance.md."""
import torch
from torch import nn


class RangeTracker(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.shape = (1,) * ndim
        self.register_buffer('min_val', None)
        self.register_buffer('max_val', None)

    @torch.no_grad()
    def forward(self, x):
        lo = x.detach().min().reshape(self.shape)
        hi = x.detach().max().reshape(self.shape)
        # Preserve momentum=1 arithmetic, including the prior-range multiplication.
        self.min_val = lo if self.min_val is None else self.min_val * 0 + lo * 1
        self.max_val = hi if self.max_val is None else self.max_val * 0 + hi * 1
