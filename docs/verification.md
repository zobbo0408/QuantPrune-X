# 독립 QCB 베이스라인 검증

검증 환경: Python 3.12.3, PyTorch 2.6.0+cu126, torchvision 0.21.0+cu126,
NumPy 2.2.6, Pillow 11.2.1, tqdm 4.67.1. CPU 평가는 4개 스레드로 실행했고 CUDA 평가도 수행했다.

모든 전체 평가는 CIFAR-10 테스트셋, batch size 128, workers 0, seed 1,
target class 0, 정규화 사용, shuffle 없음 조건이다. 정밀도마다 새 모델을
사용하고 배치마다 clean 다음 triggered 순서로 평가했다. INT8·INT4는
fake quantization이다.

## 원본 비교

별도 checkout의 SHA는 `fb873a5e276db58ff040fa4eabbb18906fcca1b1`이다.
`src/compare_upstream.py`로 두 모델·세 정밀도를 각각 CPU와 CUDA에서 비교했다.
입력은 합성 RGB 이미지 4개와 CIFAR-10 첫 128개이며 clean·triggered 모두 검사했다.
전처리, state dict 키·shape, 예측 및 성공 수가 일치했다. Logits의 최대 절대
오차는 CPU·CUDA 모두 0이며 `atol=1e-6`, `rtol=1e-5` 검사를 통과했다.
CUDA 원본 보정은 quantizer 버퍼의 장치 정렬에 한정하며 소스 파일은 수정하지 않았다.

```bash
OMP_NUM_THREADS=4 uv run --locked python src/compare_upstream.py \
  --upstream /path/to/upstream --device cpu \
  --output artifacts/verification/upstream-cpu.json
```

CUDA 비교는 `--device cuda`와 별도 출력 파일을 사용한다. 재실행에는 새 출력 경로를 지정한다.

## 독립 실행

`src/verify_independence.py`는 `Qu-ANTI-zation/`, 기존 `.venv`, `.git`, Python 캐시를 제외한 전체
새 작업 복사본을 만든 뒤 설치, 의존성 검사, 일반 테스트와 두 모델 평가를 실행한다.
데이터와 가중치는 복사하며 upstream checkout은 사용하지 않는다.

```bash
uv run --locked python src/verify_independence.py \
  --workspace /tmp/qcb-independent \
  --output artifacts/verification/independence.json \
  --device cpu --threads 4
```

재실행에는 새 workspace·output 경로를 지정한다.

`/tmp/qcb-independent`에서 설치·의존성 검사·테스트 28개와 두 모델·세 정밀도의
CPU 전체 평가가 통과했다. Python·PyTorch·소스가 복사본의 `.venv`와 `src`에서
로딩되는 것도 직접 확인했다. 기존 작업 폴더의 일반 테스트 28개도 CUDA 검사를
포함해 모두 통과했다. 원본 디렉터리를 제거하거나 변경하지 않았다.

## 전체 평가

각 행은 10,000개 표본이며 ASR-non-target의 분모는 9,000개다. 아래 값의 단위는 %다.

| 장치 | 모델 | 정밀도 | Clean Accuracy | ASR-all | ASR-non-target |
| --- | --- | --- | ---: | ---: | ---: |
| CPU | resnet18 | FP32 | 93.30 | 11.27 | 1.94 |
| CPU | resnet18 | INT8 | 91.38 | 99.24 | 99.17 |
| CPU | resnet18 | INT4 | 89.08 | 99.99 | 99.99 |
| CPU | alexnet | FP32 | 83.49 | 9.58 | 1.38 |
| CPU | alexnet | INT8 | 82.38 | 95.93 | 95.51 |
| CPU | alexnet | INT4 | 76.74 | 94.23 | 93.71 |
| CUDA | resnet18 | FP32 | 93.30 | 11.27 | 1.94 |
| CUDA | resnet18 | INT8 | 91.49 | 99.20 | 99.12 |
| CUDA | resnet18 | INT4 | 89.15 | 99.99 | 99.99 |
| CUDA | alexnet | FP32 | 83.49 | 9.58 | 1.38 |
| CUDA | alexnet | INT8 | 82.37 | 95.93 | 95.51 |
| CUDA | alexnet | INT4 | 76.74 | 94.23 | 93.71 |

[논문 보충자료 Table 8](https://proceedings.neurips.cc/paper_files/paper/2021/file/4d8bd3f7351f4fee76ba17594f070ddd-Supplemental.pdf)의 QCB 행과 비교했다. CPU·CUDA 열두 조합 모두 Clean Accuracy ±2%p, ASR-all ±3%p 범위 안이다.

CPU·CUDA 사이 양자화 지표의 최대 차이는 Clean Accuracy 0.11%p,
ASR-all 0.04%p였다. 원본과의 logits 비교는 각각 동일 장치에서 수행했다.
