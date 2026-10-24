# QuantPrune-X 작업 인수인계

최종 업데이트: 2026-10-24 (Asia/Seoul)

## 프로젝트 목적

양자화와 프루닝의 적용 순서가 Quantization-Conditioned Backdoor(QCB)의 Clean Accuracy와 Attack Success Rate에 미치는 영향을 비교한다.

주요 실험 경로는 FP32, Q-only, P-only, Q→P, P→Q다. 우선 ResNet-18과 AlexNet, CIFAR-10, INT8·INT4 Q-DQ, Global·Layer-wise Magnitude Pruning을 사용한다.

## 기준 저장소와 브랜치

- 저장소: `zobbo0408/QuantPrune-X`
- 기준 코드: `feat/qcb-baseline`
- 유지 중인 프루닝 브랜치: `feat/global-pruning`
- 2026-10-24 생성:
  - `feat/common-scaffold`
  - `feat/layerwise-pruning`
  - `feat/quantized-checkpoint`
  - `feat/experiment-pipeline`
- 새 브랜치 네 개는 모두 `feat/qcb-baseline`에서 분기했다.

## 역할 분담

- 총괄·통합: 공통 인터페이스, 결과 스키마, 통합 테스트, PR 검토
- Global Magnitude Pruning: `feat/global-pruning`
- Layer-wise Magnitude Pruning: `feat/layerwise-pruning`
- 양자화 체크포인트: `feat/quantized-checkpoint`
- 실험 파이프라인: `feat/experiment-pipeline`

## 양자화와 프루닝 구분

- Q-DQ: 정수 격자 오차를 모사하지만 저장과 연산은 주로 FP32다.
- 실제 정수 양자화: 정수 형식 저장과 지원 backend 연산을 사용한다.
- 프루닝 마스크: 유지·제거 위치를 나타내는 0/1 텐서다. 마스크만으로 파일 크기나 지연시간이 자동 감소하지 않는다.

## 완료된 작업

- GitHub의 기준 브랜치와 최신 브랜치 상태 확인
- 기존 QCB 베이스라인 문서·코드·검증 결과 확인
- 팀 기능 브랜치 네 개 생성

## 현재 차단 사항

`linux-codex`의 `/home/chan/OndevideProj/QuantPrune-X`에서 환경 구축을 시도했으나 SSH 인증이 완료되지 않았다.

- 호스트: `100.125.36.57:7001`
- 사용자: `chan`
- 서버가 로컬 ED25519 공개키를 인식했으나 SSH agent 접근이 거부되어 인증 실패
- 원격 파일, 브랜치, 가상환경은 변경하지 않음

## 다음 작업

1. 로컬에서 SSH 개인키를 agent에 잠금 해제한 뒤 접속 재시도
2. 원격 작업 트리의 브랜치와 변경 상태 확인
3. `feat/qcb-baseline`에서 `uv sync --locked`
4. 일반 테스트 28개 실행
5. 체크포인트·CIFAR-10 준비 후 스모크 평가와 베이스라인 실행
6. 검증 결과를 이 문서에 추가
7. 공통 디렉터리 골격과 인터페이스 구현
