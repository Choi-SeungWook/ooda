# OODA 템포 vs 판단의 질 — 추격-도주 그리드월드 실험

강화학습된 추격자가 의사결정이 느려져도(k>1) 빠른 비학습 추격자와 대등할 수 있는지, 그 임계 지연 k\*가 상황에 따라 어떻게 달라지는지 측정하는 실험 코드입니다. (KCI 국방·군사학 논문용)

**현재 상태**: FSM 기준선 보정 완료, 파일럿 완료 → 다음은 PPO 추격자 구현

## 문서
| 문서 | 내용 |
|---|---|
| [docs/01_연구설계.md](docs/01_연구설계.md) | 현재 확정된 설계 |
| [docs/02_설계결정_기록.md](docs/02_설계결정_기록.md) | 설계 변경 이력과 근거 데이터 |
| [docs/03_파일럿결과.md](docs/03_파일럿결과.md) | 보정과 FSM 파일럿 결과 |
| [docs/04_다음단계.md](docs/04_다음단계.md) | 작업 재개 방법, 확인 필요 항목, PPO 계획 |

## 폴더 구조
```
ooda/                  # 핵심 패키지
  config.py            # 환경 파라미터, 보정값, seed 구간 (한 곳에서 관리)
  gridmap.py           # 지도 생성, 경로거리(BFS), 가중경로(Dijkstra), 시야(LOS)
  env.py               # 게임 규칙 (이동, 포획, 탈출)
  agents.py            # 도망자 3종 + FSM 추격자
  runner.py            # 의사결정 주기 k 적용, 에피소드 실행
  plotting.py          # 한글 글꼴 설정
scripts/
  calibrate.py         # 기준선 보정 (도망자 속도 → FSM 승률 50%)
  pilot.py             # FSM 의 k별 승률 곡선
  render_episodes.py   # 궤적 그림 (동작 확인)
  exploration/         # 설계 탐색 재현 스크립트 e1~e5 (과거 설정으로 고정, 논문 부록용)
results/
  calibration/         # 보정 결과 (calibration.json, 그림)
  pilot/               # 파일럿 요약과 그림
  exploration/         # 설계 탐색 출력 (e1~e5 .txt)
  render/              # 궤적 그림
  archive/             # 폐기된 설계의 산출물 (기록용)
```

## 로컬(노트북)에서 실행
```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install numpy==1.26.4 pandas==2.2.2 matplotlib==3.8.4 tqdm==4.66.4
.venv\Scripts\python.exe scripts\calibrate.py --n 2500
.venv\Scripts\python.exe scripts\pilot.py --n 2000
```

## 실습장 PC(Docker)에서 실행
사전 준비: Docker, NVIDIA 드라이버, NVIDIA Container Toolkit
```bash
docker compose build
docker compose run --rm ooda                                  # torch / GPU 인식 확인
docker compose run --rm ooda python scripts/calibrate.py --workers 16
docker compose run --rm ooda python scripts/pilot.py --workers 16
```
결과는 호스트의 `./results` 폴더에 저장됩니다.
