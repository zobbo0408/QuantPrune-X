# QuantPrune-X 전체 디렉터리와 모듈 역할

최종 확인: 2026-10-24, `feat/common-scaffold` 작업 트리

이 문서는 현재 저장소에 존재하는 코드와 새 공통 골격을 함께 설명한다. `[기존]`은 QCB 베이스라인에서 이미 동작하던 코드, `[공통]`은 총괄 공통 인터페이스, `[대기]`는 담당 브랜치 구현이 들어갈 자리다.

## 전체 구조

```text
QuantPrune-X/
├── Qu-ANTI-zation/                         # 원 논문 공개 구현·체크포인트·재현 자료
│   ├── networks/                           # 원본 AlexNet·ResNet·MobileNetV2·VGG 정의
│   ├── utils/                              # 원본 데이터·학습·양자화·연합학습 유틸리티
│   ├── tools/pyhessian/                    # Hessian 고유값·trace·density 분석
│   ├── models/                             # 원본 공개/재현 체크포인트
│   ├── datasets/                           # 원본 코드용 CIFAR-10 데이터
│   ├── results/                            # 원본 실험 CSV
│   └── *.py, *.sh                          # 공격·학습·검증·분석 실행 파일
│
├── src/
│   ├── models/
│   │   ├── alexnet.py                      # [기존] CIFAR-10 AlexNet
│   │   ├── resnet.py                       # [기존] CIFAR-10 ResNet-18
│   │   └── registry.py                     # [기존] 모델 이름으로 새 모델 생성
│   ├── data/
│   │   └── cifar10.py                      # [기존] CIFAR-10, trigger, clean/trigger pair loader
│   ├── evaluation/
│   │   ├── metrics.py                      # [기존] loss, CA, ASR-all, ASR-non-target 계산
│   │   ├── results.py                      # [기존] JSON·CSV 결과 변환과 저장
│   │   └── runtime.py                      # [기존] device·seed·환경·provenance
│   ├── quantization/
│   │   ├── __init__.py                     # [공통] 공개 양자화 API
│   │   ├── trackers.py                     # [기존] tensor min/max 범위 추적
│   │   ├── quantizer.py                    # [기존] symmetric/asymmetric fake quantizer
│   │   ├── layers.py                       # [기존] Q-DQ Conv2d/Linear와 모델 설정
│   │   ├── contracts.py                    # [공통] bit 폭·반환 형식 계약
│   │   ├── materialize_qdq.py              # [공통] Q-DQ weight materialization, mask 0 보존
│   │   └── checkpoint_export.py            # [공통] 양자화 state_dict+metadata 저장·복원
│   ├── pruning/
│   │   ├── __init__.py                     # [공통] 공개 프루닝 API
│   │   ├── common.py                       # [공통] prune_model dispatcher와 공통 반환 계약
│   │   ├── global_magnitude.py             # [대기] Global Magnitude 담당 구현 자리
│   │   ├── layerwise_magnitude.py          # [대기] Layer-wise Magnitude 담당 구현 자리
│   │   ├── masks.py                        # [공통] bool mask 저장·복원·재적용
│   │   └── sparsity.py                     # [공통] 실제 zero/nonzero/sparsity 집계
│   ├── experiments/
│   │   ├── __init__.py                     # [공통] 결과 규격과 처리 순서 공개
│   │   ├── pipeline.py                     # [공통/확장 대기] 결과 검증·results.json 저장
│   │   ├── transforms.py                   # [공통] FP32/Q/P/Q→P/P→Q 표준 값
│   │   ├── result_schema.py                # [공통] 결과 필수 열과 값 범위 검증
│   │   └── efficiency.py                   # [공통] parameter/nonzero/model size 집계
│   ├── checkpoints.py                      # [기존] manifest·SHA256 검증 후 체크포인트 로드
│   ├── evaluate.py                         # [기존] 단일 모델 FP32/INT8/INT4 평가 CLI
│   ├── run_matrix.py                       # [기존] TOML 기반 모델×정밀도 베이스라인 실행
│   ├── prepare_checkpoints.py              # [기존] 원격 ZIP range read와 체크포인트 추출
│   ├── compare_upstream.py                 # [기존] 독립 구현과 원본 구현 logits 비교
│   ├── verify_independence.py              # [기존] 원본 폴더 없이 독립 실행 검증
│   ├── export_quantized_checkpoints.py     # [대기] Q-DQ 체크포인트 일괄 export CLI
│   └── run_experiments.py                  # [대기] 전체 실험 matrix CLI
│
├── configs/
│   ├── checkpoints/
│   │   ├── alexnet.json                    # 공개 AlexNet 파일 위치·archive member·SHA256
│   │   └── resnet18.json                   # 공개 ResNet-18 파일 위치·archive member·SHA256
│   ├── experiments/
│   │   ├── baseline.toml                   # [기존] 두 모델×FP32/INT8/INT4 기준 평가
│   │   ├── smoke_test.toml                 # [공통] 2 batch 빠른 연결 확인
│   │   └── full_matrix.toml                # [공통] 모델·순서·bit·pruning 조합 선언
│   └── quantization/
│       └── qdq_export.toml                 # [공통] Q-DQ mode와 mask-zero 보존 설정
│
├── tests/
│   ├── test_checkpoints.py                 # range download·hash·checkpoint 검증
│   ├── test_cli.py                         # evaluate/run_matrix CLI와 출력·재현성 검증
│   ├── test_data_metrics.py                # trigger·loader·CA/ASR·CPU/CUDA parity 검증
│   ├── test_models.py                      # 모델·quantizer·Q-DQ layer 검증
│   ├── test_verification.py                # 원본 비교·독립 workspace 검증
│   ├── test_common_interfaces.py           # mask·schema·bit 공통 계약 검증
│   ├── test_global_magnitude.py            # [대기] global threshold/metadata 테스트 자리
│   ├── test_layerwise_magnitude.py         # [대기] layer별 ratio/metadata 테스트 자리
│   ├── test_pruning_masks.py               # mask artifact 재적용 검증
│   ├── test_quantized_export.py            # Q-DQ mask-zero와 checkpoint 왕복 검증
│   ├── test_experiment_pipeline.py         # 검증된 results.json 생성 확인
│   ├── test_result_schema.py               # 필수 결과 열 누락 거부 확인
│   └── test_integration.py                 # 전체 처리 순서와 실제 모듈의 end-to-end smoke
│
├── artifacts/                              # 실행 산출물; Git 제외
│   ├── checkpoints/
│   │   ├── fp32/                           # FP32 원본 체크포인트
│   │   │   ├── resnet18/qcb.pth            # ResNet-18 공개 체크포인트
│   │   │   └── alexnet/qcb.pth             # AlexNet 공개 체크포인트
│   │   ├── qdq_int8/                       # INT8 Q-DQ export
│   │   ├── qdq_int4/                       # INT4 Q-DQ export
│   │   └── integer_quantized/              # 향후 실제 정수 양자화
│   └── runs/{experiment}/{run_id}/
│       ├── results.json                    # 공통 결과 스키마를 따르는 실행 결과
│       ├── metrics.csv                     # 결과의 scalar 필드 한 행
│       ├── run.log                         # 실행 시작·변환·완료·실패 기록
│       └── masks.pt                        # 해당 실행에서 생성한 pruning mask
├── datasets/                               # 프로젝트 CIFAR-10; Git 제외
├── docs/
│   ├── baseline.md                         # 베이스라인 설치·실행·출력 설명
│   ├── provenance.md                       # 이식 코드와 체크포인트 출처
│   ├── verification.md                     # 원본 재현 비교 결과
│   ├── common_interfaces.md                # 팀 공통 함수·반환 규약
│   ├── implementation_status.md            # 현재 작업/검증/구현 대기 상태
│   └── project_structure.md                # 이 문서
├── README.md                               # 사용자용 설치·실행 요약
├── Codex.md                                # 작업 인수인계 문서
├── AGENTS.md                               # 저장소 작업 지침
├── pyproject.toml                          # Python 3.12, torch 2.7.1+cu128 등 직접 의존성
├── uv.lock                                 # 직접·전이 의존성의 정확한 버전과 wheel 고정
├── .python-version                         # 기본 Python 버전
└── .gitignore                              # .venv, cache, datasets, artifacts 제외
```

`artifacts/`는 Git에서 제외된다. pruning mask는 이름 충돌과 실행 결과 분리를 피하기 위해 별도의 `artifacts/masks/`에 모으지 않고, 각 실행 폴더의 `masks.pt`로 저장한다. `save_masks()`는 파일을 저장할 때 실행 폴더를 자동으로 만든다.

FP32 공개 체크포인트는 `fp32/{model}/qcb.pth`로 정리했으며 `qdq_int8/`, `qdq_int4/`, `integer_quantized/`는 export 대상 디렉터리다.

## 기존 독립 베이스라인 코드 분석

### 모델과 Q-DQ

- `models/alexnet.py`: CIFAR-10 입력 크기에 맞춘 convolution feature extractor와 classifier를 구성한다. Q-DQ 가능한 layer 클래스를 사용하며 forward는 feature→flatten→classifier 순서다.
- `models/resnet.py`: residual `BasicBlock`, shortcut, 네 stage를 갖는 ResNet을 정의한다. `ResNet18()` factory가 CIFAR-10용 모델을 만든다.
- `models/registry.py`: `resnet18`, `alexnet` 문자열을 실제 모델 생성 함수로 연결한다. 매 평가마다 새 인스턴스를 만들어 정밀도 조건 간 state 오염을 막는다.
- `quantization/trackers.py`: 입력 tensor의 min/max를 buffer로 추적한다. device 이동 시 buffer도 함께 이동한다.
- `quantization/quantizer.py`: 추적된 범위로 scale/zero-point를 계산하고 quantize→dequantize한다. symmetric weight와 asymmetric activation 모드를 지원한다.
- `quantization/layers.py`: `QuantizedConv2d`와 `QuantizedLinear`이 weight와 activation Q-DQ를 forward에 삽입한다. `configure_model()`이 모델 전체 bit 폭과 weight/activation 활성화를 설정한다.

### 데이터와 평가

- `data/cifar10.py`: 정규화 전 CIFAR-10 image에 우하단 trigger를 넣는다. `TestPairs`는 한 sample에서 clean image/original label/triggered image/target label을 함께 반환한다.
- `evaluation/metrics.py`: clean과 triggered batch를 순서대로 forward해 평균 loss, clean accuracy, ASR-all, ASR-non-target과 각 분모를 계산한다. ASR-all은 전체 trigger sample, ASR-non-target은 원래 target class가 아니었던 sample만 사용한다.
- `evaluation/runtime.py`: `auto/cpu/cuda` device 결정, Python·NumPy·PyTorch seed 고정, 코드 hash·환경·checkpoint provenance·UTC 시간을 수집한다.
- `evaluation/results.py`: report를 JSON과 CSV용 행으로 바꾸며 기존 결과 경로를 무단 덮어쓰지 않게 저장한다.

### 체크포인트와 실행

- `checkpoints.py`: checkpoint SHA256과 manifest를 검사하고 모델 구조에 state dict를 로드한다. 잘못된 hash나 구조는 평가 전에 거부한다.
- `prepare_checkpoints.py`: 원격 ZIP 전체를 항상 받지 않고 HTTP byte range로 필요한 member를 읽을 수 있다. 추출 후 크기와 SHA256을 검증하고 실패한 임시 파일을 정리한다.
- `evaluate.py`: `RunConfig`를 검증하고 모델·checkpoint·data loader를 만든 뒤 FP32/INT8/INT4를 각각 새 모델에서 평가한다. 결과와 provenance를 저장한다.
- `run_matrix.py`: TOML의 모델 목록과 평가 옵션을 읽어 `evaluate.py`를 순차 호출하고 모델별 결과와 통합 CSV를 만든다.
- `compare_upstream.py`: 독립 구현과 `Qu-ANTI-zation` 원본 구현의 logits, prediction, metric count를 같은 입력에서 비교한다.
- `verify_independence.py`: 원본 폴더와 기존 가상환경을 제외한 임시 workspace를 만들고 테스트·평가를 실행해 독립성을 증명한다.

## 새 공통 코드 분석

- `pruning/common.py`: method 문자열을 담당 구현 함수에 연결한다. 모든 구현은 Conv2d/Linear weight를 대상으로 `(model, masks, metadata)`를 반환해야 한다.
- `pruning/masks.py`: mask를 CPU bool tensor로 정규화하고 versioned artifact로 저장·복원한다. 적용 시 이름과 shape를 검사하고 제거 위치를 정확한 0으로 만든다.
- `pruning/sparsity.py`: model parameter의 total/nonzero/actual sparsity와 파라미터별 상세값을 계산한다. buffer는 제외한다.
- `quantization/contracts.py`: 허용 bit 폭을 4/8로 제한하고 양자화 공통 반환 구조를 정의한다.
- `quantization/materialize_qdq.py`: 2차원 이상 weight에 per-tensor symmetric Q-DQ를 적용한다. P→Q에서는 전달된 mask를 전후로 적용해 제거 weight가 반올림 때문에 살아나지 않게 한다. 현재 activation materialization은 포함하지 않는다.
- `quantization/checkpoint_export.py`: Python model 객체 대신 `state_dict`, 포맷 버전, quantization metadata를 저장한다.
- `experiments/transforms.py`: 처리 순서의 저장값을 정확히 `FP32`, `Q`, `P`, `Q→P`, `P→Q`로 고정한다.
- `experiments/result_schema.py`: run/model/checkpoint/양자화/프루닝/CA/ASR/효율/산출물 경로 필드를 강제하고 범위를 검사한다.
- `experiments/efficiency.py`: parameter 수, nonzero 수와 현재 dense tensor byte 기준 모델 크기를 계산한다. 프루닝 mask만 적용했다고 압축 크기나 latency 감소로 해석하면 안 된다.
- `experiments/pipeline.py`: 결과 schema 검증을 통과한 행만 `{run_id}/results.json`으로 기록한다. 전체 matrix orchestration은 실험 담당자가 확장한다.

## 원본 `Qu-ANTI-zation` 코드 분석

### 실행 파일

- `backdoor_standard.py`: 일반 FP32 backdoor 학습을 수행한다.
- `backdoor_w_lossfn.py`: 양자화 조건에서 유지되는 backdoor를 목표로 추가 loss를 사용해 학습한다.
- `attack_w_lossfn.py`: model parameter perturbation과 Q-DQ 조건 loss를 이용한 공격을 수행한다.
- `class_w_lossfn.py`: 특정 class 기준 perturbation 공격을 수행한다.
- `sample_w_lossfn.py`: 선택 sample 기준 perturbation 공격을 수행한다.
- `attack_fedlearn.py`: federated learning 환경의 공격 실험 인자를 구성하고 결과를 기록한다.
- `run_retrain.py`: 저장 모델의 재학습·미세조정 실험을 실행한다.
- `run_analysis.py`: noise blending, sharpness, activation, quantization perturbation 분석을 수행한다.
- `valid.py`: 일반 clean/quantized accuracy를 검증한다.
- `valid_backdoor.py`: clean accuracy와 backdoor 성공률을 검증한다.
- `valid_class.py`: class 단위 성능을 검증한다.
- `valid_sample.py`: 선택 sample과 대응 공격 모델을 검증한다.
- 같은 이름의 `.sh` 파일은 위 Python 실험을 정해진 인자 조합으로 반복 실행하는 shell launcher다.

### 원본 네트워크

- `networks/alexnet.py`: AlexNet과 PyTorch PTQ 실험용 `AlexNetPTQ`를 정의한다.
- `networks/resnet.py`: BasicBlock/Bottleneck과 ResNet-18/34/50/101/152 factory를 제공한다.
- `networks/mobilenet.py`: inverted residual block 기반 MobileNetV2를 정의한다.
- `networks/vgg.py`: VGG11/13/16/19와 batch-normalized 변형을 정의한다.

### 원본 유틸리티

- `utils/datasets.py`: CIFAR-10/Tiny-ImageNet loader, sample/class selection, trigger 합성, backdoor dataset을 제공한다.
- `utils/networks.py`: dataset·network 이름으로 모델을 만들고 저장 state를 불러온다.
- `utils/trackers.py`: moving-average 방식으로 activation/weight 범위를 추적한다.
- `utils/quantizer.py`: straight-through rounding과 signed/unsigned, symmetric/asymmetric quantizer를 구현한다.
- `utils/qutils.py`: Conv2d/Linear에 Q-DQ를 적용하고 context manager로 양자화를 켜고 끈다.
- `utils/learner.py`: FP32·quantized 학습/검증, classwise 평가, backdoor 평가를 수행한다.
- `utils/optimizers.py`: loss와 optimizer를 실험 인자로 생성한다.
- `utils/io.py`: pickle과 CSV 입력·출력을 담당한다.
- `utils/futils.py`: federated dataset 분할, client weight 평균, 실험 정보 출력을 담당한다.
- `utils/fupdate.py`: 정상·악성·backdoor client의 local update와 FP/Q 평가를 담당한다.
- `tools/pyhessian/hessian.py`: Hessian-vector product로 top eigenvalue, trace, layer trace, density를 추정한다.
- `tools/pyhessian/hutils.py`: Hessian 계산에 필요한 vector 연산을 제공한다.
- `tools/pyhessian/density_plot.py`: 추정 고유값의 spectral density를 계산하고 그린다.

## 테스트 상태와 해석

- 기존 베이스라인 테스트와 공통 인터페이스 테스트는 함께 실행한다.
- `test_global_magnitude.py`, `test_layerwise_magnitude.py`, `test_integration.py`는 담당 구현이 아직 병합되지 않아 현재 의도적으로 skip된다.
- 마지막 확인 결과는 38개 통과, 3개 skip이다.
- NVML 경고는 시스템 NVIDIA library 상태 경고이며 당시 PyTorch CUDA 연산 테스트 자체는 통과했다.
