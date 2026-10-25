# QuantPrune-X 공통 인터페이스

## 파일별 한눈에 보기

- `pruning/common.py`: 프루닝 방식 이름을 실제 구현에 연결하고 공통 입력·반환 형식을 정한다.
- `pruning/masks.py`: pruning mask를 저장하고 불러오며 모델에 다시 적용한다.
- `pruning/sparsity.py`: 전체·파라미터별 zero/nonzero와 실제 sparsity를 계산한다.
- `quantization/contracts.py`: 허용 bit 폭과 양자화 반환 구조를 정한다.
- `quantization/materialize_qdq.py`: weight를 INT4/INT8 격자에 맞춘 뒤 다시 부동소수점으로 복원하며 pruning zero를 보존한다.
- `quantization/checkpoint_export.py`: 양자화 weight와 metadata를 portable checkpoint로 저장·복원한다.
- `experiments/transforms.py`: `FP32`, `Q`, `P`, `Q→P`, `P→Q` 처리 순서 표기를 통일한다.
- `experiments/result_schema.py`: 모든 실험 결과가 같은 필수 열과 값 범위를 갖는지 검사한다.
- `experiments/efficiency.py`: parameter 수, nonzero 수, dense tensor 기준 모델 크기를 계산한다.
- `experiments/pipeline.py`: 검증된 한 실행 결과를 `results.json`으로 저장한다.
- `export_quantized_checkpoints.py`: 양자화 체크포인트 일괄 생성용 CLI 골격이다.
- `run_experiments.py`: 전체 실험 matrix 실행용 CLI 골격이다.

## 책임 경계

- `src/pruning/common.py`: 모든 프루닝 호출의 단일 진입점입니다. 알고리즘은 담당 브랜치의 `global_magnitude.py`, `layerwise_magnitude.py`에 둡니다.
- `src/quantization/materialize_qdq.py`: 양자화 담당자가 `quantize(model, bits, mask=None, inplace=False) -> (model, metadata)`를 구현합니다. `mask`가 주어지면 0인 프루닝 가중치를 보존합니다.
- `src/experiments/pipeline.py`: 실험 담당자가 결과 행을 만들어 `validate_result`를 통과시킨 뒤 저장합니다.

## 프루닝

`prune_model(model, method, ratio, exclude_layers=None, inplace=False)`는 `(model, masks, metadata)`를 반환합니다. 대상은 `Conv2d`·`Linear`의 weight뿐이며, mask는 parameter name을 key로 하는 bool tensor입니다. True는 유지, False는 제거입니다.

## 양자화

`quantize(model, bits, mask=None, inplace=False)`는 `(model, metadata)`를 반환합니다. 대상은 FP32 모델의 `Conv2d`·`Linear` weight이며, `bits`는 정수 4 또는 8입니다. 기본값은 모델 복사본 반환이며, `inplace=True`는 입력 모델을 변경합니다.

## 결과 행

필수 열은 `src/experiments/result_schema.py`의 `REQUIRED_RESULT_FIELDS`가 기준입니다. 변환 순서는 `FP32`, `Q`, `P`, `Q→P`, `P→Q`만 허용합니다. ASR은 `asr_all`과 `asr_non_target`을 모두 기록합니다.

## 충돌 방지

공통 브랜치는 계약·유틸리티·설정만 소유합니다. 알고리즘 파일은 각 담당자가 만들고, 병합 전에는 이 문서의 함수명·반환값·metadata 규약을 맞춥니다.
