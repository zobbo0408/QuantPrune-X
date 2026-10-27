# Provenance

Upstream: https://github.com/Trustworthy-and-Responsible-AI-Lab/Qu-ANTI-zation
Pinned commit: `fb873a5e276db58ff040fa4eabbb18906fcca1b1`.
Code license: MIT, Copyright (c) 2021 Secure AI Systems Lab.

| Original | Independent baseline | Changes |
| --- | --- | --- |
| networks/resnet.py | src/models/resnet.py | CIFAR-10 ResNet18 only; imports |
| networks/alexnet.py | src/models/alexnet.py | CIFAR-10 only; omit PTQ class; imports |
| networks/vgg.py | src/models/vgg.py | CIFAR-10 VGG16 only; imports; explicit unsupported-dataset error |
| networks/mobilenet.py | src/models/mobilenet.py | CIFAR-10 MobileNetV2 only; imports; explicit unsupported-dataset error |
| utils/qutils.py | src/quantization/layers.py | independent evaluation wrappers; separate weight/activation controls; omit training helpers |
| utils/quantizer.py, utils/trackers.py | src/quantization/quantizer.py, trackers.py | independent formula implementations; per-layer only; device alignment; non-mutating operations |
| utils/networks.py (checkpoint loading) | src/checkpoints.py | CPU-first, weights-only loading; SHA-256 validation; filter quantizer components only; strict loading |
| utils/datasets.py | src/data/cifar10.py | test-only paired clean/triggered processing |
| utils/learner.py (backdoor validation) | src/evaluation/metrics.py | add ASR-non-target and counts; return metric dict; handle empty inputs; use inference mode |

The upstream quantizer cites [QuanTorch](https://github.com/skmhrk1209/QuanTorch).
Its repository lists no license file, so quantization and tracking use
independent formula implementations. Formula reference:
https://arxiv.org/abs/1712.05877, with the pinned QCB reference's scale conventions.

Weights: https://drive.google.com/file/d/1RwJfqAAnz9fUjsnXxsyNqAwHE5PZLhkX/view
Weight licensing is not specified separately by the distribution source.
