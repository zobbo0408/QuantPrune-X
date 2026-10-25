import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.efficiency import measure_latency
from experiments.pipeline import EvaluationSettings, is_completed, run_one, write_result
from experiments.result_schema import REQUIRED_RESULT_FIELDS, validate_result
from experiments.stubs import stub_prune_model
from experiments.transforms import (
    Operations, ProcessingOrder, RunSpec, apply_transforms, assert_masked_zero, expand_matrix, run_id,
)
from quantization.layers import QuantizedConv2d, QuantizedLinear
from run_experiments import read_config, run_matrix

ROOT = Path(__file__).resolve().parents[1]
ORDERS = [order.value for order in ProcessingOrder]


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


def recording_operations(calls, *, resurrect=False):
    def quantize(model, bits, mask=None, *, inplace=False):
        calls.append(("quantize", bits, mask is not None))
        result = model if inplace else type(model)()
        result.load_state_dict(model.state_dict())
        if resurrect:
            with torch.no_grad():
                for parameter in result.parameters():
                    parameter.add_(1.0)
        return result, {"representation": "qdq", "bit_width": bits}

    def prune(model, method, ratio, *, exclude_layers=None, inplace=False):
        calls.append(("prune", method, ratio))
        return stub_prune_model(model, method, ratio, exclude_layers=exclude_layers, inplace=inplace)

    return Operations(quantize=quantize, prune=prune)


def settings(output, **overrides):
    values = dict(output=Path(output), device="cpu", latency_warmup=1, latency_iterations=2,
                  input_shape=(3, 8, 8), checkpoint_source="public")
    values.update(overrides)
    return EvaluationSettings(**values)


class MatrixTest(unittest.TestCase):
    def test_one_model_has_43_runs_and_two_models_have_86(self):
        specs = expand_matrix(["resnet18"], ORDERS, [8, 4], ["global_magnitude", "layerwise_magnitude"],
                              [0.3, 0.5, 0.7, 0.9], seed=225)
        self.assertEqual(len(specs), 43)
        counts = {order: sum(spec.order == order for spec in specs) for order in ProcessingOrder}
        self.assertEqual(counts, {ProcessingOrder.FP32: 1, ProcessingOrder.QUANTIZED: 2, ProcessingOrder.PRUNED: 8,
                                  ProcessingOrder.QUANTIZE_THEN_PRUNE: 16, ProcessingOrder.PRUNE_THEN_QUANTIZE: 16})
        both = expand_matrix(["resnet18", "alexnet"], ORDERS, [8, 4], ["global_magnitude", "layerwise_magnitude"],
                             [0.3, 0.5, 0.7, 0.9], seed=225)
        self.assertEqual(len(both), 86)
        self.assertEqual(len({run_id(spec) for spec in both}), 86)

    def test_run_id_is_stable_ascii_and_readable(self):
        self.assertEqual(run_id(RunSpec("resnet18", ProcessingOrder.FP32, seed=225)), "resnet18__fp32__s225")
        self.assertEqual(run_id(RunSpec("resnet18", ProcessingOrder.QUANTIZED, bits=8, seed=225)),
                         "resnet18__q__int8__s225")
        spec = RunSpec("alexnet", ProcessingOrder.PRUNE_THEN_QUANTIZE, bits=4, method="global_magnitude",
                       ratio=0.5, seed=225)
        self.assertEqual(run_id(spec), "alexnet__p_then_q__int4__global_magnitude__r50__s225")
        self.assertTrue(run_id(spec).isascii())

    def test_invalid_specs_are_rejected(self):
        invalid = [
            dict(order=ProcessingOrder.FP32, bits=8),
            dict(order=ProcessingOrder.QUANTIZED),
            dict(order=ProcessingOrder.QUANTIZED, bits=8, method="global_magnitude", ratio=0.5),
            dict(order=ProcessingOrder.PRUNED, method="global_magnitude"),
            dict(order=ProcessingOrder.PRUNED, method="unknown", ratio=0.5),
            dict(order=ProcessingOrder.PRUNED, method="global_magnitude", ratio=1.0),
            dict(order=ProcessingOrder.QUANTIZE_THEN_PRUNE, bits=6, method="global_magnitude", ratio=0.5),
        ]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                RunSpec("resnet18", seed=225, **fields).validate()


class TransformOrderTest(unittest.TestCase):
    def transform(self, order, **fields):
        calls = []
        spec = RunSpec("tiny", order, seed=0, **fields)
        result = apply_transforms(spec, tiny_model(), recording_operations(calls))
        return calls, result

    def test_each_order_calls_modules_in_sequence(self):
        cases = {
            ProcessingOrder.FP32: ({}, []),
            ProcessingOrder.QUANTIZED: ({"bits": 8}, [("quantize", 8, False)]),
            ProcessingOrder.PRUNED: ({"method": "global_magnitude", "ratio": 0.5},
                                     [("prune", "global_magnitude", 0.5)]),
            ProcessingOrder.QUANTIZE_THEN_PRUNE: ({"bits": 4, "method": "global_magnitude", "ratio": 0.5},
                                                  [("quantize", 4, False), ("prune", "global_magnitude", 0.5)]),
            ProcessingOrder.PRUNE_THEN_QUANTIZE: ({"bits": 4, "method": "global_magnitude", "ratio": 0.5},
                                                  [("prune", "global_magnitude", 0.5), ("quantize", 4, True)]),
        }
        for order, (fields, expected) in cases.items():
            with self.subTest(order=order):
                calls, result = self.transform(order, **fields)
                self.assertEqual(calls, expected)
                self.assertEqual(result.masks is None, order in (ProcessingOrder.FP32, ProcessingOrder.QUANTIZED))

    def test_source_model_is_not_mutated(self):
        source = tiny_model()
        before = {name: value.clone() for name, value in source.state_dict().items()}
        spec = RunSpec("tiny", ProcessingOrder.PRUNE_THEN_QUANTIZE, bits=4, method="global_magnitude", ratio=0.5, seed=0)
        apply_transforms(spec, source, recording_operations([]))
        for name, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_p_then_q_reapplies_mask_when_quantization_resurrects_weights(self):
        spec = RunSpec("tiny", ProcessingOrder.PRUNE_THEN_QUANTIZE, bits=8, method="layerwise_magnitude",
                       ratio=0.5, seed=0)
        result = apply_transforms(spec, tiny_model(), recording_operations([], resurrect=True))
        assert_masked_zero(result.model, result.masks)
        self.assertTrue(result.metadata["mask_reapplied"])

    def test_masked_zero_check_detects_surviving_weights(self):
        model = tiny_model()
        masks = {"fc.weight": torch.zeros_like(model.fc.weight, dtype=torch.bool)}
        with self.assertRaisesRegex(RuntimeError, "fc.weight"):
            assert_masked_zero(model, masks)


class RunOneTest(unittest.TestCase):
    def test_pruned_run_writes_schema_valid_artifacts(self):
        spec = RunSpec("tiny", ProcessingOrder.QUANTIZE_THEN_PRUNE, bits=8, method="global_magnitude",
                       ratio=0.5, seed=0)
        with tempfile.TemporaryDirectory() as directory:
            row = run_one(spec, settings(directory), load_fp32=tiny_model, loader=tiny_loader(),
                          operations=recording_operations([]))
            run_dir = Path(directory) / run_id(spec)
            saved = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
            validate_result(saved)
            self.assertEqual(saved, row)
            self.assertEqual(saved["status"], "success")
            self.assertEqual(saved["processing_order"], "Q→P")
            self.assertEqual(saved["representation"], "qdq")
            self.assertEqual(saved["activation_bit_width"], 8)
            self.assertAlmostEqual(saved["target_layer_sparsity"], 0.5, places=2)
            self.assertTrue((run_dir / "masks.pt").is_file())
            self.assertTrue((run_dir / "run.log").read_text(encoding="utf-8"))
            with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(next(csv.DictReader(stream))["run_id"], run_id(spec))
            self.assertTrue(is_completed(run_dir))

    def test_fp32_run_has_no_mask_and_float_representation(self):
        spec = RunSpec("tiny", ProcessingOrder.FP32, seed=0)
        with tempfile.TemporaryDirectory() as directory:
            row = run_one(spec, settings(directory), load_fp32=tiny_model, loader=tiny_loader(),
                          operations=recording_operations([]))
        self.assertEqual((row["representation"], row["bit_width"], row["pruning_method"]), ("fp32", None, "none"))
        self.assertIsNone(row["mask_path"])
        self.assertEqual(row["samples"], 8)

    def test_write_result_keeps_existing_contract(self):
        row = {key: None for key in REQUIRED_RESULT_FIELDS}
        row.update(processing_order="FP32", clean_accuracy=0.9, asr_all=0.1, asr_non_target=0.0, actual_sparsity=0.0)
        with tempfile.TemporaryDirectory() as directory:
            output = write_result(directory, row)
            self.assertEqual(json.loads(output.read_text())["run_id"], None)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


class MatrixRunTest(unittest.TestCase):
    def specs(self):
        return expand_matrix(["tiny"], ORDERS, [8], ["global_magnitude"], [0.5], seed=0)

    def test_rerun_skips_completed_runs(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            first = run_matrix(self.specs(), settings(directory), load_fp32=tiny_model, loader=tiny_loader(),
                               operations=recording_operations(calls))
            self.assertEqual((first["completed"], first["skipped"], first["failed"]), (5, 0, 0))
            calls.clear()
            second = run_matrix(self.specs(), settings(directory), load_fp32=tiny_model, loader=tiny_loader(),
                                operations=recording_operations(calls))
            self.assertEqual((second["completed"], second["skipped"], second["failed"]), (0, 5, 0))
            self.assertEqual(calls, [])
            with (Path(directory) / "summary.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 5)

    def test_failures_are_recorded_and_retried(self):
        def failing_prune(*_args, **_kwargs):
            raise NotImplementedError("pruning branch not merged")

        broken = Operations(quantize=recording_operations([]).quantize, prune=failing_prune)
        with tempfile.TemporaryDirectory() as directory:
            summary = run_matrix(self.specs(), settings(directory), load_fp32=tiny_model, loader=tiny_loader(),
                                 operations=broken)
            self.assertEqual((summary["completed"], summary["failed"]), (2, 3))
            failure = Path(directory) / "tiny__p__global_magnitude__r50__s0" / "failure.json"
            record = json.loads(failure.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failed")
            self.assertIn("NotImplementedError", record["error"])
            retried = run_matrix(self.specs(), settings(directory), load_fp32=tiny_model, loader=tiny_loader(),
                                 operations=recording_operations([]))
            self.assertEqual((retried["completed"], retried["skipped"], retried["failed"]), (3, 2, 0))
            self.assertFalse(failure.exists())


class ConfigTest(unittest.TestCase):
    def test_bundled_configs_expand_to_expected_counts(self):
        smoke = read_config(ROOT / "configs/experiments/smoke_test.toml")
        self.assertEqual({spec.order for spec in smoke.specs}, set(ProcessingOrder))
        self.assertEqual({spec.model for spec in smoke.specs}, {"resnet18"})
        full = read_config(ROOT / "configs/experiments/full_matrix.toml")
        self.assertEqual(len(full.specs), 43 * len(full.models))

    def test_unknown_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            path.write_text('[matrix]\nmodels = ["resnet18"]\nunknown = 1\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown"):
                read_config(path)


class EfficiencyTest(unittest.TestCase):
    def test_latency_is_positive_on_cpu(self):
        latency = measure_latency(tiny_model(), (3, 8, 8), torch.device("cpu"), batch_size=1, warmup=1, iterations=2)
        self.assertGreater(latency, 0.0)


if __name__ == "__main__":
    unittest.main()
