"""Checkpoint-compatible CIFAR-10 models."""
from models.resnet import ResNet18
from models.alexnet import AlexNet
from models.mobilenet import MobileNetV2
from models.vgg import VGG16

MODEL_FACTORIES = {
    'resnet18': ResNet18,
    'alexnet': lambda: AlexNet(10),
    'vgg16': VGG16,
    'mobilenetv2': MobileNetV2,
}
MODEL_NAMES = tuple(MODEL_FACTORIES)


def create_model(name):
    if name not in MODEL_FACTORIES:
        raise ValueError(f'Unknown model: {name}')
    return MODEL_FACTORIES[name]()
