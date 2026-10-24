"""Dedicated persistence coverage for pruning masks."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pruning.masks import apply_masks, load_masks, save_masks


class PruningMasksTest(unittest.TestCase):
    def test_saved_mask_can_be_reapplied(self):
        model = nn.Linear(2, 1, bias=False)
        mask = {"weight": torch.tensor([[False, True]])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.pt"
            save_masks(path, mask)
            apply_masks(model, load_masks(path))
        self.assertEqual(float(model.weight[0, 0]), 0.0)
