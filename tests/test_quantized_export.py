import copy
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantization.checkpoint_export import (
    Layout,
    load_quantized_checkpoint,
    save_quantized_checkpoint,
)
from quantization.materialize_qdq import quantize
from experiments.transforms import (
    apply_transforms,
    default_operations,
    ProcessingOrder,
    RunSpec,
)
from model_helpers import tiny_model


class QuantizedExportTest(unittest.TestCase):
    def _assert_state_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for name, tensor in expected.items():
            with self.subTest(parameter=name):
                self.assertEqual(tensor.dtype, actual[name].dtype)
                self.assertEqual(tensor.shape, actual[name].shape)
                self.assertTrue(torch.equal(tensor, actual[name]))

    def test_qdq_preserves_masked_zero_and_round_trips(self):
        model = nn.Linear(2, 1, bias=False)
        model.weight.data.copy_(torch.tensor([[0.75, 0.25]]))
        quantized, metadata = quantize(
            model, 8, {"weight": torch.tensor([[True, False]])}
        )
        self.assertEqual(float(quantized.weight[0, 1]), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.pt"
            save_quantized_checkpoint(output, quantized, metadata)
            self.assertIn("state_dict", load_quantized_checkpoint(output))

    def test_packed_round_trip_preserves_transformed_state_and_masks(self):
        orders = (
            ProcessingOrder.PRUNE_THEN_QUANTIZE,
            ProcessingOrder.QUANTIZE_THEN_PRUNE,
        )
        methods = ("global_magnitude", "layerwise_magnitude")
        for order, bits, method in itertools.product(orders, (4, 8), methods):
            spec = RunSpec("tiny", order, bits=bits, method=method, ratio=0.5)
            result = apply_transforms(spec, tiny_model(), default_operations())
            state = copy.deepcopy(result.model.state_dict())
            metadata = copy.deepcopy(result.metadata)
            quantization = result.metadata["quantization"]
            for layout in ("dense", "sparse"):
                with self.subTest(order=order, bits=bits, method=method, layout=layout):
                    with tempfile.TemporaryDirectory() as directory:
                        output = Path(directory) / "model.pt"
                        save_quantized_checkpoint(
                            output,
                            result.model,
                            quantization,
                            layout=Layout(layout),
                        )
                        raw = torch.load(output, map_location="cpu", weights_only=True)
                        restored = load_quantized_checkpoint(output)

                    self.assertEqual(raw["format_version"], 2)
                    self.assertEqual(raw["layout"], layout)
                    self.assertEqual(
                        set(raw["packed_weights"]), {"conv.weight", "fc.weight"}
                    )
                    self.assertEqual(
                        set(raw["state_dict"]), set(state) - set(raw["packed_weights"])
                    )
                    self._assert_state_equal(state, restored["state_dict"])
                    self._assert_state_equal(state, result.model.state_dict())
                    self.assertEqual(metadata, result.metadata)
                    self.assertEqual(restored["quantization"], quantization)

                    for name, mask in result.masks.items():
                        self.assertEqual(
                            torch.count_nonzero(
                                restored["state_dict"][name][~mask]
                            ).item(),
                            0,
                        )
                    for name, entry in raw["packed_weights"].items():
                        self.assertNotIn("layout", entry)
                        self.assertEqual(
                            tuple(entry["shape"]), tuple(state[name].shape)
                        )
                        self.assertEqual(entry["dtype"], str(state[name].dtype))
                        self.assertEqual(entry["scale"], quantization["scales"][name])
                        self.assertGreater(entry["scale"], 0)
                        count = state[name].numel()
                        if layout == "sparse":
                            bitmap = entry["bitmap"]
                            self.assertEqual(bitmap.dtype, torch.uint8)
                            self.assertEqual(bitmap.ndim, 1)
                            self.assertEqual(bitmap.numel(), (count + 7) // 8)
                            positions = [
                                bool(byte & (1 << bit))
                                for byte in bitmap.tolist()
                                for bit in range(8)
                            ]
                            self.assertEqual(
                                positions[:count], state[name].flatten().ne(0).tolist()
                            )
                            self.assertFalse(any(positions[count:]))
                            count = torch.count_nonzero(state[name]).item()
                        else:
                            self.assertNotIn("bitmap", entry)
                        values = entry["values"]
                        self.assertEqual(
                            values.dtype, torch.uint8 if bits == 4 else torch.int8
                        )
                        self.assertEqual(values.ndim, 1)
                        self.assertEqual(
                            values.numel(), (count + 1) // 2 if bits == 4 else count
                        )
                        if bits == 4 and count % 2:
                            self.assertEqual(values[-1].item() >> 4, 0)

    def test_packed_linear_boundaries_and_zero_tensor(self):
        for bits, layout, all_zero in itertools.product(
            (4, 8), ("dense", "sparse"), (False, True)
        ):
            with self.subTest(bits=bits, layout=layout, all_zero=all_zero):
                model = nn.Linear(5, 1, bias=False)
                weights = [0.0] * 5 if all_zero else [-1.0, 0.0, 1.0, 0.0, -1.0]
                with torch.no_grad():
                    model.weight.copy_(torch.tensor([weights]))
                quantized, metadata = quantize(model, bits)
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "model.pt"
                    save_quantized_checkpoint(
                        output, quantized, metadata, layout=Layout(layout)
                    )
                    raw = torch.load(output, map_location="cpu", weights_only=True)
                    restored = load_quantized_checkpoint(output)

                self._assert_state_equal(quantized.state_dict(), restored["state_dict"])
                self.assertEqual(raw["layout"], layout)
                self.assertEqual(set(raw["packed_weights"]), {"weight"})
                self.assertEqual(raw["state_dict"], {})
                entry = raw["packed_weights"]["weight"]
                self.assertGreater(entry["scale"], 0)
                self.assertEqual(entry["scale"], metadata["scales"]["weight"])
                if layout == "sparse":
                    self.assertEqual(entry["bitmap"].dtype, torch.uint8)
                    self.assertEqual(
                        entry["bitmap"].tolist(), [0 if all_zero else 0b00010101]
                    )
                else:
                    self.assertNotIn("bitmap", entry)
                if all_zero:
                    expected = (
                        [] if layout == "sparse" else [0] * (3 if bits == 4 else 5)
                    )
                elif bits == 4:
                    # -7 is 0x9 in two's complement; the first value uses the low nibble.
                    expected = [0x09, 0x07, 0x09] if layout == "dense" else [0x79, 0x09]
                else:
                    expected = (
                        [-127, 0, 127, 0, -127]
                        if layout == "dense"
                        else [-127, 127, -127]
                    )
                self.assertEqual(
                    entry["values"].dtype, torch.uint8 if bits == 4 else torch.int8
                )
                self.assertEqual(entry["values"].tolist(), expected)

    def test_quantization_records_original_scales(self):
        for bits in (4, 8):
            with self.subTest(bits=bits):
                model = tiny_model()
                quantized, metadata = quantize(model, bits)
                parameters = dict(model.named_parameters())
                self.assertEqual(set(metadata["scales"]), {"conv.weight", "fc.weight"})
                for name, scale in metadata["scales"].items():
                    expected = parameters[name].detach().abs().max() / (
                        (1 << (bits - 1)) - 1
                    )
                    self.assertEqual(scale, expected.item())
                    values = (parameters[name].detach() / scale).round() * scale
                    self.assertTrue(torch.equal(values, quantized.state_dict()[name]))

    def test_sparse_bitmap_excludes_quantization_zeros_without_pruning(self):
        for bits in (4, 8):
            with self.subTest(bits=bits):
                model = tiny_model()
                with torch.no_grad():
                    model.conv.weight.flatten()[0] = 1e-8
                quantized, metadata = quantize(model, bits)
                self.assertNotEqual(model.conv.weight.flatten()[0].item(), 0)
                self.assertEqual(quantized.conv.weight.flatten()[0].item(), 0)
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "model.pt"
                    save_quantized_checkpoint(
                        output, quantized, metadata, layout=Layout.SPARSE
                    )
                    raw = torch.load(output, map_location="cpu", weights_only=True)
                    restored = load_quantized_checkpoint(output)
                for name, entry in raw["packed_weights"].items():
                    positions = [
                        bool(byte & (1 << bit))
                        for byte in entry["bitmap"].tolist()
                        for bit in range(8)
                    ]
                    weights = quantized.state_dict()[name].flatten()
                    self.assertEqual(
                        positions[: weights.numel()], weights.ne(0).tolist()
                    )
                self._assert_state_equal(quantized.state_dict(), restored["state_dict"])
