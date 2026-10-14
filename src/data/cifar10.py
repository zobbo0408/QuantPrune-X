# Derived from Qu-ANTI-zation, utils/datasets.py (MIT).
# Copyright (c) 2021 Secure AI Systems Lab
# Source and changes: docs/provenance.md; license: licenses/Qu-ANTI-zation-MIT.txt
"""CIFAR-10 test pairs with a square trigger applied before normalization."""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def trigger(image):
    if image.shape != (32, 32, 3):
        raise ValueError('Expected a CIFAR-10 HWC RGB image')
    result = np.copy(image)
    result[27:31, 27:31, :] = image.max()
    return result


class TestPairs(Dataset):
    def __init__(self, images, labels, *, normalize=True, target=0):
        if not 0 <= target < 10:
            raise ValueError('Target must be in [0, 9]')
        if len(images) != len(labels):
            raise ValueError('Images and labels must have equal lengths')
        self.images, self.labels, self.target = images, labels, target
        steps = [transforms.ToTensor()]
        if normalize:
            steps.append(transforms.Normalize(MEAN, STD))
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        clean = self.transform(Image.fromarray(image))
        attacked = self.transform(Image.fromarray(trigger(image)))
        return clean, self.labels[index], attacked, self.target


def test_loader(path, *, download=False, batch_size=128, workers=0, normalize=True, target=0):
    if batch_size <= 0 or workers < 0:
        raise ValueError('Batch size must be positive and workers nonnegative')
    dataset = datasets.CIFAR10(root=path, train=False, download=download)
    pairs = TestPairs(dataset.data, dataset.targets, normalize=normalize, target=target)
    return DataLoader(pairs, batch_size=batch_size, num_workers=workers, shuffle=False)
