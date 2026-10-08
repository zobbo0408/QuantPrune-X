# QuantPrune-X

양자화와 프루닝의 적용 순서가 Quantization-Conditioned Backdoor(QCB)에 미치는 상호작용을 분석하는 프로젝트입니다.

## Baseline

원본 Qu-ANTI-zation 코드는 [`Qu-ANTI-zation/`](./Qu-ANTI-zation/)에 보존합니다.

- Original repository: https://github.com/Trustworthy-and-Responsible-AI-Lab/Qu-ANTI-zation
- License: MIT

## Initial Goal

ResNet-18 기반 QCB 모델에서 다음 경로의 Clean Accuracy와 ASR을 비교합니다.

- FP32
- Quantization only
- Pruning only
- Quantization → Pruning
- Pruning → Quantization
