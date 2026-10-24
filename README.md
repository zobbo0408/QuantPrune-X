# QuantPrune-X

양자화와 프루닝의 적용 순서가 Quantization-Conditioned Backdoor(QCB)에 미치는 상호작용을 분석하는 프로젝트입니다.

## 설치와 실행

Linux x86_64, Python 3.12.3과 시스템에 설치된 `uv`를 사용합니다.
CPU와 CUDA 평가를 지원하며 가상환경은 루트 `.venv/`입니다.

```bash
uv python install 3.12.3
uv sync --locked
uv run --locked python src/prepare_checkpoints.py --model all

uv run --locked python src/evaluate.py \
  --model resnet18 \
  --checkpoint artifacts/checkpoints/fp32/resnet18/qcb.pth \
  --data-dir datasets/cifar10 \
  --device auto \
  --output artifacts/runs/resnet18-baseline \
  --download-data
```

기본 평가 정밀도는 FP32·INT8·INT4 모두입니다. AlexNet은 `--model alexnet`과 해당 체크포인트 경로를 사용합니다.
`auto`는 CUDA 사용 가능 시 CUDA, 그 외에는 CPU를 선택합니다. 데이터 다운로드는 `--download-data`로 요청합니다.
기존 결과 디렉터리는 덮어쓰지 않습니다.

ResNet-18과 AlexNet을 각각 FP32·INT8·INT4로 순차 평가하려면 데이터 준비 후 다음 명령을 사용합니다.

```bash
uv run --locked python src/run_matrix.py configs/experiments/baseline.toml
PYTHONPATH=src uv run --locked python -B -m unittest discover -s tests -v
```

모델별 JSON·CSV와 통합 CSV를 저장합니다. INT8·INT4는 fake quantization 평가이며 실제 정수 연산의 속도나 모델 크기 절감을 측정하지 않습니다.

## 구성과 검증

- [베이스라인 구성](docs/baseline.md): 환경, 체크포인트 준비, 옵션, 지표와 출력
- [검증 결과](docs/verification.md): CPU·CUDA 전체 평가 및 원본 비교
- [코드·가중치 출처](docs/provenance.md): 원본 SHA, 파일 대응, 변경과 라이선스

ResNet-18과 AlexNet을 각각 FP32·INT8·INT4로 CPU와 CUDA에서 평가했습니다. 각 평가는 CIFAR-10 테스트셋 10,000개를 사용했습니다.
모든 조합이 논문 비교 허용 범위(Clean Accuracy ±2%p, ASR-all ±3%p)에 들어왔습니다.
일반 테스트 28개와 동일 장치의 원본 출력 비교도 통과했습니다.

설치·실행·일반 테스트는 `Qu-ANTI-zation/`에 의존하지 않습니다. 이식 코드의 MIT 고지와 라이선스 전문은 파일 헤더와 `licenses/`에 보존합니다.
