# QCB 평가 베이스라인

공개 ResNet-18·AlexNet QCB 체크포인트를 CIFAR-10 테스트셋에서 FP32·INT8·INT4로 평가한다.
원본 CIFAR-10 모델 구조와 state dict 이름·shape를 유지하며, 모델·전처리·양자화·평가는 `src/`에 구현되어 있다.
학습, 공격 생성, 프루닝, materialization과 실제 정수 backend 변환은 포함하지 않는다.

## 환경과 설치

| 항목 | 구성 |
| --- | --- |
| 지원 OS | Linux x86_64, glibc 2.28 이상 |
| Python | 3.12.3 |
| PyTorch / torchvision | 2.6.0+cu126 / 0.21.0+cu126 |
| 직접 의존성 | NumPy 2.2.6, Pillow 11.2.1, tqdm 4.67.1 |
| 검증한 uv 버전 | 0.13.0 |
| 가상환경 | 루트 `.venv/` |

```bash
uv python install 3.12.3
uv sync --locked
```

GPU 없는 Linux에서도 같은 CUDA 배포판으로 CPU 평가한다.
`pyproject.toml`은 비패키지 구성이며 의존성과 배포 파일 해시는 `uv.lock`에 고정한다.
PyTorch는 x86_64 wheel URL을 지정한다. uv 0.13.0의 인덱스 해석에서 ARM64 metadata를 사용해
CUDA 의존성이 누락되는 현상을 확인했기 때문이다. torchvision은 CUDA 12.6 인덱스를 사용한다.

## 체크포인트 준비

```bash
uv run --locked python src/prepare_checkpoints.py --model all
```

[공개 모델 ZIP](https://drive.google.com/file/d/1RwJfqAAnz9fUjsnXxsyNqAwHE5PZLhkX/view)의 필요한 두 파일만
HTTP Range로 추출한다. 응답 범위·크기, ZIP CRC, SHA-256과 strict loading을 검증한다.
Range 미지원 시 전체 ZIP을 자동 다운로드하지 않는다. 로컬 ZIP이나 개별 가중치를 지정할 수 있다.

```bash
uv run --locked python src/prepare_checkpoints.py --model all --zip /path/to/models.zip
uv run --locked python src/prepare_checkpoints.py --model resnet18 --checkpoint /path/to/resnet18.pth
```

| 모델 | 저장 경로 | Manifest |
| --- | --- | --- |
| ResNet-18 | `artifacts/checkpoints/resnet18/qcb.pth` | [resnet18.json](../configs/checkpoints/resnet18.json) |
| AlexNet | `artifacts/checkpoints/alexnet/qcb.pth` | [alexnet.json](../configs/checkpoints/alexnet.json) |

Manifest에는 ZIP 내부 경로, 출처, 해시, 트리거와 가중치 라이선스 상태가 기록되어 있다.
코드의 MIT 라이선스와 가중치 라이선스는 구분한다. 배포 출처는 가중치 라이선스를 별도로 명시하지 않는다.
로더는 CPU에서 `torch.load(map_location="cpu", weights_only=True)`로 읽고 quantizer 키만 제외해 strict loading한다.
해시·키·shape 불일치는 오류로 종료하며, 검증 후 선택 장치로 이동한다.

## 실행과 평가 조건

```bash
uv run --locked python src/evaluate.py \
  --model resnet18 \
  --checkpoint artifacts/checkpoints/resnet18/qcb.pth \
  --data-dir datasets/cifar10 --device auto \
  --output artifacts/runs/resnet18-baseline --download-data
```

기본값은 FP32·INT8·INT4 평가, batch size 128, workers 0, seed 1, target class 0,
정규화 사용, shuffle 없음이다. `--precisions fp32 int8 int4`, `--batch-size`, `--workers`,
`--seed`, `--target-class`, `--no-normalize`로 변경한다. `--limit N`은 테스트셋 첫 N개만 평가한다.
전체 옵션은 `uv run --locked python src/evaluate.py --help`로 확인한다.

`--device auto`는 CUDA 사용 가능 시 CUDA, 그 외에는 CPU다. `cpu`, `cuda`, `cuda:0` 등을 명시할 수 있다.
사용할 수 없는 CUDA를 요청하면 오류로 종료한다. 데이터 다운로드는 `--download-data`가 있을 때만 수행한다.
각 정밀도는 새 모델로 평가하며 `eval()`과 gradient 비활성화를 적용한다.
NumPy·PyTorch seed와 CUDA 결정성 설정도 적용한다.

트리거는 정규화 전 이미지의 모든 채널 `[27:31, 27:31]`에 해당 이미지의 최대 픽셀값을 넣는다.
Clean 입력은 보존한다. ToTensor 후 mean `(0.4914, 0.4822, 0.4465)`,
std `(0.2023, 0.1994, 0.2010)`으로 정규화한다.

Weight는 per-layer symmetric, Activation은 per-layer asymmetric로 같은 비트 수를 사용한다.
별도 calibration 없이 매 forward range를 갱신하며, 배치마다 clean 다음 triggered 순서로 평가한다.
따라서 배치 크기와 데이터 순서도 재현 조건이다. 일반 반올림은 `floor(x + 0.5)`,
asymmetric zero point는 `torch.round()`를 사용한다. INT8·INT4는 fake quantization이며
실제 정수 연산의 속도나 모델 크기 절감을 나타내지 않는다.

## 다중 실행과 결과

```bash
uv run --locked python src/run_matrix.py configs/experiments/baseline.toml
```

TOML 경로는 프로젝트 루트 기준이다. ResNet-18과 AlexNet을 각각 FP32·INT8·INT4로 순차 평가하며,
설정의 `download_data = true`로 다운로드를 요청할 수 있다.
모델별 `results.json`·`results.csv`와 통합 `summary.csv`를 출력한다.
`--output artifacts/runs/another-matrix`로 출력 경로를 변경할 수 있다.
단일 평가와 다중 실행 모두 기존 출력 디렉터리를 덮어쓰지 않는다.

| 지표 | 의미 |
| --- | --- |
| Clean Accuracy | Clean 입력 정답률 |
| ASR-all | 전체 triggered 입력에서 target class로 예측한 비율 |
| ASR-non-target | 원래 정답이 target class가 아닌 표본에서의 공격 성공률 |
| Clean loss | 원래 정답 기준 평균 cross-entropy |
| Triggered loss | Target class 기준 평균 cross-entropy |

비율은 JSON·CSV에서 0~1 값이다. 각 분모와 정답·성공 수도 저장한다.
Loss는 마지막 배치까지 표본 수로 가중한다. ASR-non-target 분모가 0이면 JSON `null`, CSV 빈 값이다.
평가 조건, 라이브러리 버전, 장치 종류, 체크포인트·실행 코드·lock 해시를 함께 기록한다.
하드웨어 모델명은 수집하지 않으며 결과 경로에는 호스트 절대 경로를 기록하지 않는다.

## 검증

```bash
PYTHONPATH=src uv run --locked python -B -m unittest discover -s tests -v
```

테스트 28개는 모델·양자화 6개, 체크포인트 7개, 데이터·지표 6개, CLI·결과 5개,
검증 도구 4개로 구성된다. CUDA를 사용할 수 없으면 관련 2개를 건너뛴다.

ResNet-18과 AlexNet의 FP32·INT8·INT4 평가를 CPU와 CUDA에서 모두 완료했다.
각 평가는 테스트셋 10,000개를 사용했다.
ASR-non-target 분모는 각 9,000개다. 동일 장치에서 원본과 비교한 logits 최대 절대 오차는 0이었다.
원본 디렉터리가 없는 복사본의 설치·테스트·CPU 전체 평가도 통과했다.
전체 복사 방식으로 변경한 이후에는 복사 동작 테스트와 일반 테스트를 재검증했다.
장치별 결과, 논문 비교와 검증 도구 명령은 [검증 문서](verification.md)에 정리되어 있다.
