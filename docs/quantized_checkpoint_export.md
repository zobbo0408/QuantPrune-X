# INT4·INT8 체크포인트 export

P→Q·Q→P 가중치를 실제 INT4·INT8 형식으로 저장한다. 로드 시 FP32로 복원해 기존 추론에 사용하며 정수 연산 추론은 지원하지 않는다.

## 실행과 설정

```bash
uv run --locked python src/prepare_checkpoints.py --model all
uv run --locked python src/export_quantized_checkpoints.py --dry-run
uv run --locked python src/export_quantized_checkpoints.py \
  configs/quantization/packed_export.toml \
  --output artifacts/exports --run-id packed-export
```

기본 [설정](../configs/quantization/packed_export.toml)은 네가지 모델 × 두 순서 × INT4·INT8 × Global·Layer-wise × 30·50·70·90% × dense·sparse로 256개를 출력한다. seed는 0이다.
CPU에서 변환·저장하며 데이터셋과 정확도 평가는 필요하지 않다. 입력 가중치는 모델별 manifest의 SHA-256 검증을 통과해야 한다.
실험 파이프라인의 `expand_matrix`·`apply_transforms`를 재사용하며, 동일한 변환 결과를 두 저장 방식에 사용한다.

필요한 조합만 선택하려면 TOML의 목록을 줄인다. 예를 들어 저장 방식을 하나만 지정하면 파일 수와 CSV 행 수도 절반이 된다.
`[checkpoints]`에 모델별 입력 경로를 지정할 수 있으며 생략하면 `configs/checkpoints/{model}.json`의 `local_path`를 사용한다.
출력·체크포인트 상대 경로는 프로젝트 루트 기준이다.

`--dry-run`은 가중치·데이터셋 없이 조합과 경로만 표시하며 파일·디렉터리를 만들지 않는다.
`--run-id`를 생략하면 `export-YYYYMMDDT-HHMMSS-ffffffZ` 형식의 UTC 시각을 사용한다.
run ID는 단일 디렉터리 이름이어야 하며 `.`·`..`와 경로 구분자는 거부한다. 기존 run은 덮어쓰거나 이어 쓰지 않는다.

## 출력

```text
artifacts/exports/packed-export/
├── summary.csv
└── models/
    ├── resnet18__p_then_q__int4__global_magnitude__r50__s0__dense.pt
    ├── resnet18__p_then_q__int4__global_magnitude__r50__s0__sparse.pt
    └── ...
```

- `run_id`: 전체 export 요청의 이름.
- `combination_id`: 모델·순서·비트·프루닝 방식·비율·seed로 정한 변환 조건. 실험 파이프라인의 개별 `run_id(spec)`에 해당한다.
- `export_id`: `{combination_id}__{layout}`.
- `checkpoint_path`: run 루트 기준 상대 경로 `models/{export_id}.pt`.

통합 CSV는 출력마다 한 행을 기록하며 원본 SHA-256, 변환 조건, layout, 성공·실패와 크기를 포함한다.
실패 행은 예정 경로와 오류를 남기고 크기·감소율은 비운다. 다른 출력은 계속 처리하며 실패가 있으면 종료 코드는 1이다.
다른 layout의 저장이 실패해도 성공한 파일은 유지한다.

## 저장 형식

`quantization.checkpoint_export`의 `save_quantized_checkpoint(path, model, metadata, layout=...)`를 사용한다.
`layout`은 `Layout.DENSE` 또는 `Layout.SPARSE`이며, 생략하면 기존 Q-DQ(v1)로 저장한다.
저장은 입력 모델 · metadata를 변경하지 않는다.

### 포맷 버전

| 버전 | 선택 조건 | 저장 내용 |
| --- | --- | --- |
| v1 | `layout` 생략 | Q-DQ `state_dict`와 양자화 metadata |
| v2 | `Layout.DENSE` 또는 `Layout.SPARSE` 지정 | 정수 패킹 가중치, 비양자화 `state_dict`, 양자화 metadata |

### v2 저장 방식

| `layout` | 정수 가중치 저장 방식 |
| --- | --- |
| `dense` | 0을 포함한 모든 정수값 저장 |
| `sparse` | nonzero 위치를 표시한 bitmap과 nonzero 정수값만 저장 |

v2의 packed_weights는 파라미터별 정수 가중치 데이터(values)와 복원용 메타데이터(shape, dtype, scale)를 묶어 저장한다. `sparse`에서는 위치 복원용 `bitmap`도 포함한다.

INT8은 `int8`, INT4는 두 값을 `uint8` 하나에 패킹한다. INT4의 앞 원소는 하위 4비트이며 음수는 2의 보수다.
bitmap도 앞 위치부터 하위 비트를 사용한다. 마지막 바이트의 남는 비트는 0이며 복원 시 제외한다.
전부 0인 텐서는 양수 scale을 유지하고 sparse의 값 배열은 비워둔다.

모든 양자화 대상 텐서에 같은 layout을 적용한다. sparse는 프루닝뿐 아니라 양자화로 생긴 0도 제외한다.
Q→P 이후에도 최초 scale을 사용하며 재양자화하지 않는다. 압축 대상의 FP32 사본은 중복 저장하지 않고 bias · BatchNorm · 버퍼는 원래 dtype을 유지한다.
모델 이름 · 변환 조건 · 가중치 출처는 파일 내부 `quantization`에 보존한다.

## 복원

`load_quantized_checkpoint(path)`는 정수 가중치를 FP32로 복원한 `state_dict`와 metadata를 반환한다.
metadata의 모델 이름으로 모델을 생성하고 `strict=True`로 가중치를 로드한다.
activation 양자화는 자동으로 활성화하지 않는다.

## 크기 지표

| CSV 필드 | 의미 |
| --- | --- |
| `fp32_file_size_bytes` | 해시 검증을 통과한 원본 입력 파일의 크기 |
| `export_file_size_bytes` | 저장한 체크포인트의 크기 |
| `size_reduction_percent` | `100 * (1 - export_bytes / fp32_bytes)` |

dense 대비 sparse 감소율은 `100 * (1 - sparse_bytes / dense_bytes)`이며 음수는 크기 증가를 뜻한다.
기존 실험의 `model_size_mb`는 `sum(parameter.numel() * parameter.element_size()) / 1024**2`인 논리적 파라미터 용량(MiB)으로 fp32_file_size_bytes와 다르다.

## 실측 결과

아래는 Global 50%, seed 0의 대표 조건이다. dense·sparse 감소율은 각각 원본 FP32 파일 대비이며 반올림 전 바이트 수로 계산했다.

| 모델 | 비트 | FP32 MiB | dense MiB | sparse MiB | dense 감소율 | sparse 감소율 | dense 대비 sparse 감소율 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet18 | INT4 | 42.793 | 5.435 | 1.813 | 87.30% | 95.76% | 66.64% |
| resnet18 | INT8 | 42.793 | 10.758 | 4.580 | 74.86% | 89.30% | 57.43% |
| alexnet | INT4 | 88.816 | 11.135 | 3.574 | 87.46% | 95.98% | 67.90% |
| alexnet | INT8 | 88.816 | 22.228 | 10.000 | 74.97% | 88.74% | 55.01% |
| vgg16 | INT4 | 58.235 | 7.299 | 4.964 | 87.47% | 91.48% | 31.98% |
| vgg16 | INT8 | 58.235 | 14.566 | 9.119 | 74.99% | 84.34% | 37.39% |
| mobilenetv2 | INT4 | 9.259 | 1.449 | 0.781 | 84.35% | 91.57% | 46.10% |
| mobilenetv2 | INT8 | 9.259 | 2.528 | 1.563 | 72.70% | 83.12% | 38.18% |

## 검증

- 기존 테스트: 69개 모두 통과. Q-DQ 호환성, TinyNet 저장·복원, 프루닝 0 보존, 음수·패딩·전부 0인 텐서를 포함한다.
- 기본 dry-run: 중복 없는 256개 출력, 파일·디렉터리 생성 없음.
- 전체 실행: 네가지 모델 원본 해시 검증, 모델당 64개 export 성공. 단일 `models/`의 256개 파일과 통합 CSV 256행, 실제 파일 크기·감소율 일치 확인.
- 대표 조건 복원: 위 표의 두 처리 순서 · dense · sparse 조합 32개 파일을 CSV 상대 경로로 읽어 모델에 `strict=True` 로드. 같은 조건의 두 layout에서 모든 가중치 · 버퍼 값과 dtype 일치 확인.
- sparse 단독 실행: ResNet-18 · P→Q · INT4 · Global 50%의 파일 1개와 [CSV 1행](../artifacts/exports/measured-sparse/summary.csv) 확인.
