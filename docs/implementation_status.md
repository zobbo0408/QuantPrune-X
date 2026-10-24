# 공통 골격 구현 상태

최종 업데이트: 2026-10-24 (Asia/Seoul)

## 현재 작업 트리

- 기준 브랜치: `feat/common-scaffold`
- 기준 베이스라인: `feat/qcb-baseline`의 PyTorch 2.7.1 + CUDA 12.8 잠금 환경을 병합한 상태
- 이 문서와 공통 골격 파일은 아직 커밋·푸시하지 않은 작업 트리 변경이다.
- 기존 `Qu-ANTI-zation/` 변경 및 생성 산출물은 공통 골격 작업에서 건드리지 않는다.

## 추가된 공통 골격

- 프루닝 공통 계약, mask 저장·복원·재적용, sparsity 집계
- Q-DQ materialization과 양자화 체크포인트 envelope
- 변환 순서(`FP32`, `Q`, `P`, `Q→P`, `P→Q`)와 결과 행 검증
- smoke/full-matrix/Q-DQ TOML 설정
- 실험 결과 `results.json` 기록 유틸리티와 static efficiency 집계
- 요청된 모듈·CLI·테스트 파일 경로 전체

## 담당자 구현 대기 항목

- `pruning/global_magnitude.py`: Global Magnitude Pruning 담당
- `pruning/layerwise_magnitude.py`: Layer-wise Magnitude Pruning 담당
- `experiments/pipeline.py`, `run_experiments.py`: 전체 matrix 실행 담당
- `export_quantized_checkpoints.py`: 체크포인트 일괄 추출 확장 담당
- end-to-end integration test: 위 구현 병합 후 활성화

## 검증

다음 명령은 38개 통과, 3개 의도적 skip 결과를 확인했다.

```bash
PYTHONPATH=src uv run --locked python -B -m unittest discover -s tests -q
```

## 커밋 전 확인

1. `git status --short`에서 기존 사용자 변경과 공통 골격 파일을 구분한다.
2. 공통 골격 파일만 `git add`한다. `artifacts/`, `datasets/`, `.venv/`는 Git 제외 상태를 유지한다.
3. 담당자와 `docs/common_interfaces.md`의 함수명·반환값·metadata 규약을 확인한 후 커밋한다.
