"""Tests for materialized Q-DQ weights, activations, and pruning masks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from pruning.masks import apply_masks
from pruning.sparsity import model_sparsity
from quantization.checkpoint_export import load_quantized_checkpoint, save_quantized_checkpoint
from quantization.layers import QuantizedConv2d, QuantizedLayer, QuantizedLinear, configure_model
from quantization.materialize_qdq import quantize
from quantization.quantizer import Quantizer
from qdq_test_support import assert_activation_equal, assert_state_equal, make_test_set_loader, toy_model


class MaterializationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_int4_int8_match_baseline_rounding_boundaries(self):
        for bits in (4, 8):
            maximum = (1 << (bits - 1)) - 1
            values = torch.tensor([[-maximum, -2.5, -1.5, -.5, .5, 1.5, 2.5, maximum]])
            source = QuantizedLinear(values.numel(), 1, bias=False)
            source.weight.data.copy_(values)
            result, metadata = quantize(source, bits)
            expected = Quantizer(bits, 2, True)(values)
            with self.subTest(bits=bits):
                torch.testing.assert_close(result.weight, expected, rtol=0, atol=0)
                self.assertEqual(result.weight[0, 3].item(), 0)
                self.assertEqual(result.weight[0, 1].item(), -2)
                self.assertEqual(metadata["bit_width"], bits)
                self.assertEqual(result.weight.dtype, torch.float32)

    def test_constants_zero_and_epsilon_match_baseline(self):
        for bits in (4, 8):
            for value in (0., 3., -3., 1e-16, -1e-16):
                with self.subTest(bits=bits, value=value):
                    model = QuantizedLinear(3, 2, bias=False)
                    model.weight.data.fill_(value)
                    expected = Quantizer(bits, 2, True)(model.weight.detach().clone())
                    result, _ = quantize(model, bits)
                    self.assertTrue(torch.isfinite(result.weight).all())
                    torch.testing.assert_close(result.weight, expected, rtol=0, atol=0)

    def test_only_conv_linear_weights_change_and_source_is_independent(self):
        model = toy_model()
        model.register_parameter("auxiliary", nn.Parameter(torch.tensor([[.13, .27], [.91, -.31]])))
        before = copy.deepcopy(model.state_dict())
        names = list(dict(model.named_parameters()))
        result, metadata = quantize(model, 4)
        self.assertIsNot(result, model)
        self.assertEqual(list(dict(result.named_parameters())), names)
        assert_state_equal(self, before, model.state_dict())
        for name, original in model.named_parameters():
            with self.subTest(parameter=name):
                actual = dict(result.named_parameters())[name]
                self.assertNotEqual(actual.data_ptr(), original.data_ptr())
                if name in ("conv.weight", "fc.weight"):
                    torch.testing.assert_close(actual, Quantizer(4, actual.ndim, True)(original), rtol=0, atol=0)
                else:
                    torch.testing.assert_close(actual, original, rtol=0, atol=0)
        assert_state_equal(self, model.bn.state_dict(), result.bn.state_dict())
        self.assertEqual(set(metadata["quantized_parameters"]), {"conv.weight", "fc.weight"})
        self.assertFalse(result.training)

    def test_inplace_returns_same_model(self):
        model = toy_model()
        result, _ = quantize(model, 8, inplace=True)
        self.assertIs(result, model)

    def test_unsupported_layers_are_rejected_before_any_mutation(self):
        class ScaledLinear(nn.Linear):
            def forward(self, inputs):
                return super().forward(inputs) * 2

        for model in (nn.Linear(4, 3), nn.Conv2d(2, 4, 3, stride=2, padding=2, dilation=2, groups=2),
                      toy_model(plain=True), ScaledLinear(4, 3),
                      nn.Sequential(QuantizedLinear(4, 4), ScaledLinear(4, 3))):
            for inplace in (False, True):
                with self.subTest(type=type(model).__name__, inplace=inplace):
                    before = copy.deepcopy(model.state_dict())
                    types = {name: type(layer) for name, layer in model.named_modules()}
                    with self.assertRaisesRegex(TypeError, "unsupported layer.*QuantizedConv2d/QuantizedLinear"):
                        quantize(model, 4, inplace=inplace)
                    assert_state_equal(self, before, model.state_dict())
                    self.assertEqual(types, {name: type(layer) for name, layer in model.named_modules()})

    def test_quantized_conv_padding_behavior_is_preserved(self):
        source = QuantizedConv2d(2, 4, 3, padding=1, groups=2, padding_mode="reflect").eval()
        inputs = next(iter(make_test_set_loader()))[0]
        result, _ = quantize(source, 4)
        activations = Quantizer(4, 4, False)(inputs)
        weights = Quantizer(4, 4, True)(source.weight)
        expected = source._conv_forward(activations, weights, source.bias)
        torch.testing.assert_close(result(inputs), expected, rtol=1e-5, atol=1e-6)

    def test_checkpoint_keeps_settings_in_metadata_and_numerical_buffers_in_state(self):
        model = configure_model(toy_model(), 4)
        model(next(iter(make_test_set_loader()))[0])
        result, metadata = quantize(model, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qdq.pt"
            save_quantized_checkpoint(path, result, metadata)
            checkpoint = load_quantized_checkpoint(path)
        layers = checkpoint["quantization"]["layers"]
        state = checkpoint["state_dict"]
        self.assertEqual(set(layers), {"conv", "fc"})
        for name in ("conv", "fc"):
            original = model.get_submodule(name)
            baseline = Quantizer(4, original.weight.ndim, True)
            expected = baseline(original.weight)
            for kind in ("weight", "activation"):
                self.assertEqual(set(layers[name][kind]), {"enabled", "bit_width", "symmetric", "range_update"})
                self.assertEqual(layers[name][kind]["bit_width"], 4)
                for suffix in ("min_val", "max_val", "range_tracker.min_val", "range_tracker.max_val", "scale", "zero_point"):
                    self.assertIn(f"{name}.{kind}_quantizer.{suffix}", state)
            torch.testing.assert_close(state[f"{name}.weight_quantizer.scale"], baseline.scale, rtol=0, atol=0)
            torch.testing.assert_close(state[f"{name}.weight_quantizer.zero_point"], baseline.zero_point, rtol=0, atol=0)
            torch.testing.assert_close(result.get_submodule(name).weight, expected, rtol=0, atol=0)

    def test_materialized_weight_is_not_quantized_again_and_activation_updates_each_forward(self):
        for activation_bits in (4, 8):
            with self.subTest(activation_bits=activation_bits):
                model = configure_model(toy_model(), 4)
                model.conv.configure_quantization(activation_bits)
                model(torch.full((2, 2, 4, 4), 2.))
                source_state = copy.deepcopy(model.state_dict())

                result, metadata = quantize(model, 4)
                self.assertEqual(metadata["activation_mode"], "per_tensor_asymmetric")
                assert_activation_equal(self, model, result)
                for layer in result.modules():
                    if isinstance(layer, QuantizedLayer):
                        self.assertFalse(layer.quantize_weight)
                        self.assertFalse(layer.activation_quantizer.symmetric)
                before = {name: p.detach().clone() for name, p in result.named_parameters()}
                with patch.object(result.conv.weight_quantizer, "forward", side_effect=AssertionError("weight re-quantized")):
                    result(torch.full((2, 2, 4, 4), -1.))
                    self.assertEqual(result.conv.activation_quantizer.range_tracker.min_val.item(), -1)
                    result(torch.full((2, 2, 4, 4), 3.))
                    self.assertEqual(result.conv.activation_quantizer.range_tracker.min_val.item(), 3)
                    self.assertEqual(result.conv.activation_quantizer.range_tracker.max_val.item(), 3)
                assert_state_equal(self, before, dict(result.named_parameters()))
                assert_state_equal(self, source_state, model.state_dict())

        for fc_enabled in (True, False):
            with self.subTest(fc_activation_enabled=fc_enabled):
                model = configure_model(toy_model(), 4)
                model(torch.full((2, 2, 4, 4), 2.))
                model.fc.quantize_activation = fc_enabled
                source_state = copy.deepcopy(model.state_dict())
                result, _ = quantize(model, 4)
                assert_state_equal(self, source_state, model.state_dict())
                assert_activation_equal(self, model, result)

    def test_mask_is_applied_before_range_observation_and_after_qdq(self):
        for bits in (4, 8):
            model = QuantizedLinear(4, 1, bias=False)
            model.weight.data.copy_(torch.tensor([[1000., -.5, 7., -1.5]]))
            mask = torch.tensor([[0., 1., 1., 1.]])
            original_mask = mask.clone()
            result, metadata = quantize(model, bits, {"weight": mask})
            with self.subTest(bits=bits):
                expected = Quantizer(bits, 2, True)(model.weight * mask) * mask
                torch.testing.assert_close(result.weight, expected, rtol=0, atol=0)
                self.assertEqual(result.weight[0, 0].item(), 0)
                torch.testing.assert_close(mask, original_mask, rtol=0, atol=0)
                self.assertTrue(metadata["mask_applied"])

    def test_fully_masked_weights_remain_exact_zero_with_bool_and_integer_masks(self):
        for bits in (4, 8):
            for dtype in (torch.bool, torch.int64):
                with self.subTest(bits=bits, dtype=dtype):
                    source = QuantizedLinear(3, 2, bias=False)
                    result, _ = quantize(source, bits, {"weight": torch.zeros_like(source.weight, dtype=dtype)})
                    result(torch.ones(1, 3))
                    self.assertEqual(torch.count_nonzero(result.weight).item(), 0)

    def test_invalid_masks_are_rejected_before_any_inplace_mutation(self):
        model = toy_model()
        shape = model.fc.weight.shape
        cases = [({"missing.weight": torch.ones(1)}, KeyError),
                 ({"fc.weight": torch.ones(1)}, ValueError),
                 ({"fc.weight": torch.full(shape, .5)}, ValueError),
                 ({"fc.weight": torch.full(shape, -1)}, ValueError),
                 ({"fc.weight": torch.full(shape, 2)}, ValueError),
                 ({"fc.weight": torch.full(shape, float("nan"))}, ValueError),
                 ({"fc.weight": torch.full(shape, float("inf"))}, ValueError)]
        for invalid, error in cases:
            candidate = copy.deepcopy(model)
            before = copy.deepcopy(candidate.state_dict())
            with self.subTest(mask=str(invalid)):
                try:
                    with self.assertRaises(error):
                        quantize(candidate, 4, invalid, inplace=True)
                finally:
                    assert_state_equal(self, before, candidate.state_dict())

    def test_invalid_bit_width_is_rejected_without_mutation(self):
        for bits in (None, 0, 3, 16, True, "4"):
            model = toy_model()
            before = copy.deepcopy(model.state_dict())
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                quantize(model, bits, inplace=True)
            assert_state_equal(self, before, model.state_dict())

    def test_runtime_and_materialized_logits_and_predictions_match(self):
        for bits in (4, 8):
            runtime = configure_model(toy_model(), bits)
            materialized, _ = quantize(runtime, bits)
            with self.subTest(bits=bits), torch.inference_mode():
                for clean, _, attacked, _ in make_test_set_loader():
                    for inputs in (clean, attacked):
                        expected, actual = runtime(inputs), materialized(inputs)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(actual.argmax(1), expected.argmax(1), rtol=0, atol=0)

    def test_q_then_p_and_p_then_q_keep_zero_through_forward(self):
        for bits in (4, 8):
            source = toy_model()
            masks = {name: torch.arange(p.numel()).reshape(p.shape) % 2 == 0
                     for name, p in source.named_parameters() if name in ("conv.weight", "fc.weight")}
            q_then_p, _ = quantize(source, bits)
            apply_masks(q_then_p, masks)
            p_then_q, _ = quantize(apply_masks(copy.deepcopy(source), masks), bits, masks)
            for result in (q_then_p, p_then_q):
                with self.subTest(bits=bits, order="Q-P" if result is q_then_p else "P-Q"):
                    result(next(iter(make_test_set_loader()))[0])
                    summary = model_sparsity(result)
                    for name, mask in masks.items():
                        parameter = dict(result.named_parameters())[name]
                        self.assertTrue(torch.equal(parameter[~mask], torch.zeros_like(parameter[~mask])))
                        self.assertEqual(summary["per_parameter"][name]["nonzero"], torch.count_nonzero(parameter).item())
                        self.assertGreaterEqual(summary["per_parameter"][name]["sparsity"], .5)


if __name__ == "__main__":
    unittest.main()
