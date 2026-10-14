"""Checkpoint-compatible CIFAR-10 models."""
from models.resnet import ResNet18
from models.alexnet import AlexNet


def create_model(name):
    factories = {'resnet18': ResNet18, 'alexnet': lambda: AlexNet(10)}
    if name not in factories:
        raise ValueError(f'Unknown model: {name}')
    return factories[name]()
