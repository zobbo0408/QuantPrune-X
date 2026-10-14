import unittest
import torch
from models.registry import create_model
from quantization.layers import QuantizedLinear, configure_model
from quantization.quantizer import Quantizer


class ModelsTest(unittest.TestCase):
    def test_models_and_state_isolation(self):
        torch.set_num_threads(2)
        for name in ('resnet18', 'alexnet'):
            model = create_model(name).eval()
            before = model.state_dict()['linear.weight' if name == 'resnet18' else 'classifier.6.weight'].clone()
            with torch.inference_mode():
                for bits in (None, 8, 4):
                    configure_model(model, bits)
                    y = model(torch.randn(2, 3, 32, 32))
                    self.assertEqual(y.shape, (2, 10))
                    self.assertTrue(torch.isfinite(y).all())
            torch.testing.assert_close(before, model.state_dict()['linear.weight' if name == 'resnet18' else 'classifier.6.weight'])
            fresh = create_model(name)
            self.assertFalse(any('quantizer' in k for k in fresh.state_dict()))

    def test_rounding_and_nonmutating_arithmetic(self):
        q = Quantizer(4, 2, True)
        x = torch.tensor([[-7., -1.5, -.5, .5, 1.5, 7.]])
        before = x.clone()
        torch.testing.assert_close(q(x), torch.tensor([[-7., -1., 0., 1., 2., 7.]]))
        torch.testing.assert_close(x, before)
        integers = torch.tensor([[-8., 7.]])
        copy = integers.clone()
        q.dequantize(integers)
        torch.testing.assert_close(integers, copy)
        q(torch.tensor([[-14., 14.]]))
        self.assertAlmostEqual(q.scale.item(), .5)
        self.assertTrue(torch.isfinite(q(torch.zeros(1, 2))).all())

    def test_asymmetric_and_controls(self):
        q = Quantizer(4, 2, False)
        x = torch.tensor([[-1., 2.]])
        torch.testing.assert_close(q(x), x)
        layer = QuantizedLinear(2, 2)
        layer.configure_quantization(4, weight=False, activation=True)
        layer(x)
        self.assertIsNone(layer.weight_quantizer.scale)
        self.assertIsNotNone(layer.activation_quantizer.scale)
        layer.configure_quantization(8, weight=True, activation=False)
        layer(x)
        self.assertIsNone(layer.activation_quantizer.scale)
        for buffer in layer.buffers():
            self.assertEqual(buffer.device, layer.weight.device)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_buffers(self):
        layer = QuantizedLinear(2, 2).cuda()
        layer.configure_quantization(4)
        layer(torch.ones(1, 2, device='cuda'))
        self.assertTrue(all(b.is_cuda for b in layer.buffers()))

    def test_asymmetric_zero_point_ties_and_clipping(self):
        # lo * scale = 0.5: zero point uses ties-to-even, unlike value rounding.
        q = Quantizer(4, 2, False)
        torch.testing.assert_close(q(torch.tensor([[0.5, 15.5]])), torch.tensor([[1., 15.]]))
        self.assertEqual(q.zero_point.item(), 0)
        torch.testing.assert_close(q(torch.full((1, 2), 3.)), torch.full((1, 2), 3.))
        self.assertTrue(torch.isfinite(q(torch.zeros(1, 2))).all())
        for bits in (4, 8):
            q = Quantizer(bits, 2, True)
            q(torch.tensor([[-1., 1.]]))
            values = torch.tensor([[-100., 100.]])
            clipped = q.dequantize(torch.clamp(torch.floor(q.quantize(values)+.5), q.min_val, q.max_val))
            self.assertAlmostEqual(clipped[0, 1].item(), 1.)
            self.assertAlmostEqual(clipped[0, 0].item(), -(2**(bits-1))/(2**(bits-1)-1), places=6)

    def test_conv_controls_range_and_instance_isolation(self):
        from quantization.layers import QuantizedConv2d
        a, b = QuantizedConv2d(3, 2, 3), QuantizedConv2d(3, 2, 3)
        original_weight = a.weight.detach().clone()
        a.configure_quantization(4)
        self.assertFalse(b.quantize_weight)
        with torch.inference_mode():
            a(torch.ones(2, 3, 5, 5))
            a(torch.full((2, 3, 5, 5), 2.))
        self.assertEqual(a.activation_quantizer.range_tracker.min_val.shape, (1, 1, 1, 1))
        self.assertEqual(a.activation_quantizer.range_tracker.min_val.item(), 2.)
        self.assertEqual(a.activation_quantizer.range_tracker.max_val.item(), 2.)
        torch.testing.assert_close(a.weight, original_weight)
        a.configure_quantization(None)
        inputs = torch.randn(1, 3, 5, 5)
        torch.testing.assert_close(a(inputs), torch.nn.functional.conv2d(inputs, a.weight, a.bias))
        for bits in (0, 3, 16):
            with self.assertRaises(ValueError):
                a.configure_quantization(bits)
        with self.assertRaises(ValueError):
            create_model('unknown')
