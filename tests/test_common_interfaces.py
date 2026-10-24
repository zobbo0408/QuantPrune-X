import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.result_schema import REQUIRED_RESULT_FIELDS, validate_result
from pruning.masks import apply_masks, load_masks, save_masks
from pruning.sparsity import model_sparsity
from quantization.contracts import validate_bits


class CommonInterfaceTests(unittest.TestCase):
    def test_masks_round_trip_and_apply(self):
        model = nn.Linear(2, 1, bias=False)
        model.weight.data.fill_(1)
        masks = {"weight": torch.tensor([[True, False]])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.pt"
            save_masks(path, masks)
            apply_masks(model, load_masks(path))
        self.assertEqual(model.weight.tolist(), [[1.0, 0.0]])
        self.assertEqual(model_sparsity(model)["nonzero_count"], 1)

    def test_result_schema(self):
        row = {key: None for key in REQUIRED_RESULT_FIELDS}
        row.update(processing_order="FP32", clean_accuracy=0.9, asr_all=0.1, asr_non_target=0.0, actual_sparsity=0.0, bit_width=None)
        self.assertEqual(validate_result(row)["processing_order"], "FP32")

    def test_bits(self):
        self.assertEqual(validate_bits(8), 8)
        with self.assertRaises(ValueError):
            validate_bits(6)


if __name__ == "__main__":
    unittest.main()
