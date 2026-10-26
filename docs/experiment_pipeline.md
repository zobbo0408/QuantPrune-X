# 실험 파이프라인

`src/run_experiments.py`는 TOML 설정의 조합을 펼쳐 한 건씩 실행하고 결과를 `artifacts/runs/{name}/{run_id}/`에 저장한다.
양자화·프루닝은 `docs/common_interfaces.md`의 `quantize`와 `prune_model`만 호출한다.

## 실행

```bash
# 조합만 확인 (실행하지 않음)
PYTHONPATH=src uv run --locked python src/run_experiments.py configs/experiments/full_matrix.toml --dry-run

# ResNet-18 5개 경로 smoke test (Global·Layer-wise, 2배치, 8건)
PYTHONPATH=src uv run --locked python src/run_experiments.py configs/experiments/smoke_test.toml

# 전체 실험 (모델·seed당 43건)
PYTHONPATH=src uv run --locked python src/run_experiments.py configs/experiments/full_matrix.toml
```

같은 명령을 다시 실행하면 `status = "success"`인 실험은 건너뛰고, 실패한 실험만 다시 시도한다.
실패가 하나라도 있으면 종료 코드는 1이다. `--output`으로 저장 위치를 바꿀 수 있다.

## 처리 순서

| 순서 | 동작 |
| --- | --- |
| `FP32` | 체크포인트 로드 |
| `Q` | `quantize(model, bits)` |
| `P` | `prune_model(model, method, ratio)` |
| `Q→P` | `quantize` → `prune_model` |
| `P→Q` | `prune_model` → `quantize(mask=masks)` → mask 재적용 |

프루닝 경로는 마지막에 제거된 가중치가 정확히 0인지 검사하고, 아니면 해당 실험을 실패로 기록한다.
`activation_quantization = true`(기본값)이면 평가 직전에 `configure_model(model, bits)`로 weight·activation을
fake quantization한다. 베이스라인 INT8·INT4 평가와 같은 조건이다.

## run_id

`{model}__{order}[__int{bits}][__{method}__r{ratio%}]__s{seed}` 형식의 ASCII 문자열이다.

```
resnet18__fp32__s225
resnet18__q__int8__s225
resnet18__p_then_q__int4__global_magnitude__r50__s225
```

## 출력

| 파일 | 내용 |
| --- | --- |
| `{run_id}/results.json` | `validate_result`를 통과한 결과 한 건 |
| `{run_id}/metrics.csv` | 같은 결과의 스칼라 필드 한 행 |
| `{run_id}/run.log` | 시작·변환·완료·실패 기록 |
| `{run_id}/masks.pt` | 프루닝 경로의 mask (`save_masks` 형식) |
| `{run_id}/failure.json` | 실패 시 오류와 traceback. 성공하면 삭제된다 |
| `summary.csv` | 설정의 성공한 실험 전체 |

`REQUIRED_RESULT_FIELDS` 외에 다음 필드를 추가로 기록한다.

| 필드 | 의미 |
| --- | --- |
| `status` | `success` (재개 판단 기준) |
| `activation_bit_width` | activation fake quantization 비트. 없으면 null |
| `target_layer_sparsity` | 프루닝 대상 텐서 기준 제거 비율. `target_ratio`와 직접 비교한다 |
| `pruning_backend` | `common` 또는 `stub`. `stub` 결과는 실험 데이터가 아니다 |
| `device`, `latency_batch_size` | 측정 조건 |
| `checkpoint_sha256`, `code_sha256`, `lock_sha256` | 재현 정보 |

`actual_sparsity`는 bias·BatchNorm을 포함한 전체 파라미터 기준이므로 `target_ratio`보다 작게 나온다.
`latency_ms`는 fake quantization 연산을 포함한 PyTorch 평균 시간이며 실제 정수 연산 속도가 아니다.
`peak_memory_mb`는 CUDA 평가 중 최대 할당량이고 CPU에서는 null이다.

## stub 프루닝

`experiments/stubs.py`는 프루닝 모듈 없이 배선을 확인하기 위한 대체 구현이다.
`[pipeline] pruning_backend = "stub"`일 때만 사용되며 제공 설정 파일은 모두 `common`을 쓴다.
