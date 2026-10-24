import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantization.checkpoint_export import load_quantized_checkpoint, save_quantized_checkpoint
from quantization.materialize_qdq import quantize


class QuantizedExportTest(unittest.TestCase):
    def test_qdq_preserves_masked_zero_and_round_trips(self):
        model = nn.Linear(2, 1, bias=False)
        model.weight.data.copy_(torch.tensor([[0.75, 0.25]]))
        quantized, metadata = quantize(model, 8, {"weight": torch.tensor([[True, False]])})
        self.assertEqual(float(quantized.weight[0, 1]), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.pt"
            save_quantized_checkpoint(output, quantized, metadata)
            self.assertIn("state_dict", load_quantized_checkpoint(output))
