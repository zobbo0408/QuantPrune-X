"""End-to-end coverage: owner pruning modules + shared quantize() through the experiment pipeline."""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from checkpoints import load_checkpoint, read_manifest
from experiments.pipeline import EvaluationSettings
from experiments.result_schema import validate_result
from experiments.transforms import ProcessingOrder, default_operations, expand_matrix, run_id
from pruning.masks import load_masks
from quantization.layers import QuantizedConv2d, QuantizedLinear
from run_experiments import build_loader, read_config, run_matrix

ORDERS = [order.value for order in ProcessingOrder]
METHODS = ["global_magnitude", "layerwise_magnitude"]


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = QuantizedConv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = QuantizedConv2d(8, 8, 3, padding=1)
        self.fc = QuantizedLinear(8, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        return self.fc(x.mean(dim=(2, 3)))


def small_model(_name):
    torch.manual_seed(0)
    return SmallNet().eval()


def small_loader():
    generator = torch.Generator().manual_seed(0)
    clean = torch.randn(16, 3, 8, 8, generator=generator)
    labels = torch.arange(16) % 10
    return DataLoader(TensorDataset(clean, labels, clean + 1.0, torch.zeros(16, dtype=torch.long)), batch_size=8)


class IntegrationTest(unittest.TestCase):
    def test_every_order_runs_with_owner_modules(self):
        specs = expand_matrix(["small"], ORDERS, [8, 4], METHODS, [0.0, 0.5, 0.9], seed=0)
        self.assertEqual(len(specs), 1 + 2 + 6 + 12 + 12)
        with tempfile.TemporaryDirectory() as directory:
            settings = EvaluationSettings(output=Path(directory), device="cpu", input_shape=(3, 8, 8),
                                          latency_warmup=1, latency_iterations=2)
            counts = run_matrix(specs, settings, load_fp32=small_model, loader=small_loader(),
                                operations=default_operations())
            self.assertEqual((counts["completed"], counts["failed"]), (len(specs), 0))
            for spec in specs:
                run_dir = Path(directory) / run_id(spec)
                row = validate_result(json.loads((run_dir / "results.json").read_text(encoding="utf-8")))
                with self.subTest(run=row["run_id"]):
                    self.assertEqual(row["processing_order"], spec.order.value)
                    if spec.method is None:
                        self.assertIsNone(row["mask_path"])
                        continue
                    masks = load_masks(Path(directory) / row["mask_path"])
                    self.assertEqual(row["pruning_backend"], "common")
                    self.assertAlmostEqual(row["target_layer_sparsity"], spec.ratio, delta=0.02)
                    removed = sum(int((~mask).sum()) for mask in masks.values())
                    self.assertEqual(removed == 0, spec.ratio == 0.0)

    @unittest.skipUnless((ROOT / read_manifest("resnet18")["local_path"]).is_file()
                         and (ROOT / "datasets/cifar10/cifar-10-batches-py").is_dir(),
                         "needs prepare_checkpoints.py output and CIFAR-10 under datasets/")
    def test_resnet18_smoke_config(self):
        config = read_config(ROOT / "configs/experiments/smoke_test.toml")
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(config.settings, output=Path(directory))
            counts = run_matrix(config.specs, settings, operations=default_operations(), loader=build_loader(config),
                                load_fp32=lambda name: load_checkpoint(name, settings.checkpoints[name]))
        self.assertEqual((counts["completed"], counts["failed"]), (len(config.specs), 0))


if __name__ == "__main__":
    unittest.main()
