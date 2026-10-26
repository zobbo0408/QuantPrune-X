# 양자화 체크포인트 CLI

추출, 평가, 동등성 검증은 각각 독립된 명령과 TOML을 사용합니다.

| 명령 | 설정 파일 | 역할과 출력 |
| --- | --- | --- |
| `export_quantized_checkpoints.py` | `configs/quantization/qdq_export.toml` | 양자화·저장, `model.pt` |
| `evaluate_quantized_checkpoints.py` | `configs/quantization/evaluate.toml` | 저장된 모델의 CA·ASR, JSON·CSV |
| `verify_quantized_checkpoints.py` | `configs/quantization/verify.toml` | 원본과 추출 모델의 동등성, JSON·CSV |

## 추출

```bash
uv run --locked python src/export_quantized_checkpoints.py configs/quantization/qdq_export.toml --run-id example-01
```

양자화 TOML에는 모델·bit-width·입출력·변환 장치를 설정합니다.
기본 대상은 ResNet-18·AlexNet과 INT4·INT8입니다.
출력은 `{output}/qdq_int{bits}/{model}/{run_id}/model.pt`입니다.
기본 출력 루트는 `artifacts/checkpoints`이고, `run_id`를 생략하면 UTC 시각으로 생성합니다.
다른 run의 결과는 허용하며 같은 모델·bit-width·run_id의 출력 경로가 겹칠 때만 거부합니다.

공개 가중치는 manifest SHA-256을 검증하고 선택한 입력 경로에서 직접 읽습니다.
입력 경로를 명시하지 않았고 manifest 경로의 파일도 없으면 기존 `artifacts/checkpoints/{model}/qcb.pth`를 사용합니다.
추출 과정에서 원본 파일을 복사하거나 manifest를 변경하지 않습니다. 파일 준비는 `prepare_checkpoints.py`로 별도 수행합니다.
`--output`, `--run-id`, `--device`로 해당 설정을 덮어쓸 수 있습니다.

## 평가

```bash
uv run --locked python src/evaluate_quantized_checkpoints.py configs/quantization/evaluate.toml \
  --checkpoints artifacts/checkpoints/qdq_int*/*/example-01/model.pt
```

대상 파일은 TOML의 `checkpoints` 배열 또는 `--checkpoints`로 지정합니다.
저장된 파일만 읽어 기존 평가 함수로 CA·ASR과 성공 횟수·분모를 계산합니다.

`[evaluation]`의 기본값은 CIFAR-10 전체, device auto, seed 225, batch size 128,
workers 0, 정규화 사용, Square trigger, target 0입니다.
smoke 실행은 `device = "cpu"`, `limit = 8` 등을 지정합니다.
`download_data = true`로 데이터 다운로드를 요청하며 실행 중 장치·batch size를 자동 변경하지 않습니다.

## 검증

```bash
uv run --locked python src/verify_quantized_checkpoints.py configs/quantization/verify.toml \
  --checkpoints artifacts/checkpoints/qdq_int*/*/example-01/model.pt
```

검증 TOML의 평가 조건은 평가 CLI와 독립적입니다. 체크포인트 metadata에 기록된 원본 FP32
파일과 해시를 확인한 뒤 runtime 양자화, materialized 모델, 추출 파일의 복원 모델을 순차 평가합니다.
Logits는 `rtol=1e-5, atol=1e-6`으로 비교하고 예측·CA·ASR 성공 횟수와 분모는 일치해야 합니다.
복원 직후 state는 저장 파일과 정확히 비교합니다. 평가 후에는 weight·고정 buffer·설정을 검사하고,
activation 관측 buffer는 기존의 매 forward 갱신을 허용합니다. 입력 파일을 다시 저장하지 않습니다.

## 보고서와 경로

평가와 검증은 각각 TOML의 `output` 루트 아래 `{run_id}/.../results.json`, `results.csv`를 만듭니다.
기본 루트는 `artifacts/runs/quantized-evaluation`, `artifacts/runs/quantized-verification`입니다.
두 명령 모두 `--output`, `--run-id`, `--quiet`를 지원하며 같은 출력 run 경로의 덮어쓰기를 거부합니다.
실패하면 비정상 종료하고 해당 입력의 성공 보고서를 쓰지 않습니다. 이미 완료한 입력의 보고서는 남습니다.
모든 상대 경로는 설정 파일 위치와 관계없이 프로젝트 루트 기준입니다.

## 직접 학습한 입력

추출 TOML에 FP32 state dictionary 경로·실제 SHA-256·출처·가중치 라이선스를 명시합니다.

```toml
[checkpoints.resnet18]
path = "artifacts/checkpoints/self_trained/resnet18/best.pt"
sha256 = "<64 hexadecimal characters>"
checkpoint_source = "self_trained"
source_url = "<training run URL or source reference>"
weight_license = "<weight license or explicitly unspecified>"
```
