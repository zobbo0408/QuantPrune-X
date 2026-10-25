"""Small deterministic models and synthetic data for Q-DQ export tests."""
from __future__ import annotations

from collections.abc import Mapping
from collections import OrderedDict
import copy
from typing import Any
from unittest import TestCase

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from quantization.layers import QuantizedConv2d, QuantizedLayer, QuantizedLinear


def toy_model(*, plain: bool = False) -> nn.Sequential:
    conv = nn.Conv2d if plain else QuantizedConv2d
    linear = nn.Linear if plain else QuantizedLinear
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(225)
        model = nn.Sequential(OrderedDict([
            ("conv", conv(2, 4, 3, padding=1, groups=2)),
            ("bn", nn.BatchNorm2d(4)),
            ("relu", nn.ReLU()),
            ("pool", nn.AdaptiveAvgPool2d(1)),
            ("flatten", nn.Flatten()),
            ("fc", linear(4, 3)),
        ]))
    return model.eval().requires_grad_(False)


def make_test_set_loader() -> DataLoader[tuple[torch.Tensor, ...]]:
    clean = torch.linspace(-2, 3, 5 * 2 * 4 * 4).reshape(5, 2, 4, 4)
    attacked = clean.clone()
    attacked[:, :, -1, -1] = 5
    labels = torch.tensor([0, 1, 2, 1, 0])
    return DataLoader(TensorDataset(clean, labels, attacked, torch.zeros_like(labels)),
                      batch_size=2, shuffle=False)


def with_checkpoint_source(
    metadata: Mapping[str, Any],
    *,
    model_name: str | None = None,
    checkpoint_source: str = "public",
) -> dict[str, Any]:
    return dict(copy.deepcopy(metadata), model_name=model_name,
                source_checkpoint="fixtures/fp32.pt", source_sha256="ab" * 32,
                checkpoint_source=checkpoint_source,
                source_url="https://example.invalid/weights",
                weight_license="not separately specified")


def assert_state_equal(
    test: TestCase,
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> None:
    test.assertEqual(set(left), set(right))
    for name in left:
        with test.subTest(buffer_or_parameter=name):
            torch.testing.assert_close(left[name].cpu(), right[name].cpu(), rtol=0, atol=0)


def assert_tree_equal(test: TestCase, left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


def assert_activation_equal(test: TestCase, left: nn.Module, right: nn.Module) -> None:
    left_layers = {name: layer for name, layer in left.named_modules()
                   if isinstance(layer, QuantizedLayer)}
    right_layers = {name: layer for name, layer in right.named_modules()
                    if isinstance(layer, QuantizedLayer)}
    test.assertEqual(set(left_layers), set(right_layers))
    for name, layer in left_layers.items():
        other = right_layers[name]
        test.assertFalse(other.quantize_weight)
        test.assertEqual(layer.quantize_activation, other.quantize_activation)
        if not hasattr(layer, "activation_quantizer"):
            continue
        test.assertEqual(layer.activation_quantizer.symmetric, other.activation_quantizer.symmetric)
        for attribute in ("min_val", "max_val", "scale", "zero_point"):
            a = getattr(layer.activation_quantizer, attribute)
            b = getattr(other.activation_quantizer, attribute)
            if a is None:
                test.assertIsNone(b)
            else:
                torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
        for attribute in ("min_val", "max_val"):
            a = getattr(layer.activation_quantizer.range_tracker, attribute)
            b = getattr(other.activation_quantizer.range_tracker, attribute)
            if a is None:
                test.assertIsNone(b)
            else:
                torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
