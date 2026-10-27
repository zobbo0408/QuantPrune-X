"""Contract tests for the VGG16 and MobileNetV2 model expansion."""
import unittest

import torch

from models.registry import create_model
from pruning.common import prune_model
from quantization.layers import QuantizedConv2d, QuantizedLinear, configure_model
from quantization.materialize_qdq import quantize


class ModelExpansionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_cifar10_forward_and_quantized_layers(self):
        for name in ("vgg16", "mobilenetv2"):
            with self.subTest(model=name), torch.inference_mode():
                model = create_model(name).eval()
                self.assertTrue(any(isinstance(layer, QuantizedConv2d) for layer in model.modules()))
                self.assertTrue(any(isinstance(layer, QuantizedLinear) for layer in model.modules()))
                output = model(torch.randn(1, 3, 32, 32))
                self.assertEqual(output.shape, (1, 10))
                self.assertTrue(torch.isfinite(output).all())

    def test_qdq_and_pruning_contracts(self):
        for name in ("vgg16", "mobilenetv2"):
            with self.subTest(model=name):
                model = create_model(name).eval()
                qdq, quantization = quantize(model, 8)
                self.assertEqual(quantization["representation"], "qdq")
                configure_model(qdq, 8, weight=False, activation=True)
                pruned, masks, metadata = prune_model(model, "global_magnitude", 0.3)
                self.assertIsNot(pruned, model)
                self.assertTrue(masks)
                self.assertTrue(all(mask.dtype == torch.bool for mask in masks.values()))
                self.assertEqual(metadata["method"], "global_magnitude")
                del qdq, pruned, masks

    def test_mobilenet_depthwise_layers_are_prunable(self):
        model = create_model("mobilenetv2")
        depthwise_names = {
            f"{name}.weight"
            for name, layer in model.named_modules()
            if isinstance(layer, QuantizedConv2d) and layer.groups == layer.in_channels
        }
        self.assertTrue(depthwise_names)
        _, masks, _ = prune_model(model, "layerwise_magnitude", 0.3)
        self.assertTrue(depthwise_names <= set(masks))


if __name__ == "__main__":
    unittest.main()
