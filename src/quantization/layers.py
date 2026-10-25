"""Independent evaluation wrappers for QCB layer semantics; docs/provenance.md."""
from torch import nn
from torch.nn import functional as F
from quantization.quantizer import Quantizer


class QuantizedLayer:
    quantize_weight = False
    quantize_activation = False

    def configure_quantization(self, bits, *, weight=True, activation=True):
        if bits not in (None, 4, 8):
            raise ValueError('Expected FP32 (None), INT4 or INT8')
        self.quantize_weight = bits is not None and weight
        self.quantize_activation = bits is not None and activation
        if bits is not None:
            ndim = self.weight.ndim
            self.weight_quantizer = Quantizer(bits, ndim, True).to(self.weight.device)
            self.activation_quantizer = Quantizer(bits, ndim, False).to(self.weight.device)

    def operands(self, x):
        if self.quantize_activation:
            x = self.activation_quantizer(x)
        w = self.weight_quantizer(self.weight) if self.quantize_weight else self.weight
        return x, w


class QuantizedConv2d(QuantizedLayer, nn.Conv2d):
    def forward(self, x):
        x, w = self.operands(x)
        return self._conv_forward(x, w, self.bias)


class QuantizedLinear(QuantizedLayer, nn.Linear):
    def forward(self, x):
        x, w = self.operands(x)
        return F.linear(x, w, self.bias)


def validate_quantized_layers(model: nn.Module) -> None:
    """Reject unsupported Conv2d/Linear layers before changing model state."""
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)) and not isinstance(layer, QuantizedLayer):
            raise TypeError(
                f"unsupported layer {name or '<root>'}: {type(layer).__name__}; "
                "use QuantizedConv2d/QuantizedLinear in the model definition"
            )


def configure_model(model, bits, *, weight=True, activation=True):
    for layer in model.modules():
        if isinstance(layer, QuantizedLayer):
            layer.configure_quantization(bits, weight=weight, activation=activation)
    return model
