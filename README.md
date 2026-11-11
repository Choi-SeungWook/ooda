# OODA 템포 vs 판단의 질 — 추격-도주 그리드월드 실험

강화학습된 추격자가 의사결정이 느려져도(k>1) 빠른 비학습 추격자와 대등할 수 있는지, 그 임계 지연 k\*가 상황에 따라 어떻게 달라지는지 측정하는 실험 코드입니다. (KCI 국방·군사학 논문용)

**현재 상태**: FSM 기준선 보정·파일럿 완료, PPO 학습 파이프라인 구현 완료 → 다음은 실습장 PC에서 본 학습

## 문서
| 문서 | 내용 |
|---|---|
| [docs/01_연구설계.md](docs/01_연구설계.md) | 현재 확정된 설계 |
| [docs/02_설계결정_기록.md](docs/02_설계결정_기록.md) | 설계 변경 이력과 근거 데이터 |
| [docs/03_파일럿결과.md](docs/03_파일럿결과.md) | 보정과 FSM 파일럿 결과 |
| [docs/04_다음단계.md](docs/04_다음단계.md) | 작업 재개 방법, 확인 필요 항목, PPO 설계 |
| [docs/05_실습장_실행가이드.md](docs/05_실습장_실행가이드.md) | 실습장 PC(Docker, GPU 2장)에서 학습·평가 실행 방법 |

## 폴더 구조
```
ooda/                  # 핵심 패키지
  config.py            # 환경 파라미터, 보정값, seed 구간 (한 곳에서 관리)
  gridmap.py           # 지도 생성, 경로거리(BFS), 가중경로(Dijkstra), 시야(LOS)
  env.py               # 게임 규칙 (이동, 포획, 탈출)
  agents.py            # 도망자 3종(기만형 = 회피형 + 기만체) + FSM 추격자
  runner.py            # 의사결정 주기 k 적용, 에피소드 실행 (Episode.advance 를 RL 과 공유)
  obs.py               # RL 관측 생성 (학습·평가 공유)
  gym_env.py           # 추격자 시점 gymnasium 환경 (PPO 학습용)
  rl_policy.py         # CNN 정책망, 학습된 모델을 FSM 과 같은 인터페이스로 감싼 RLPursuer
  plotting.py          # 한글 글꼴 설정
scripts/
  calibrate.py         # 기준선 보정 (도망자 속도 → FSM 승률 50%)
  pilot.py             # FSM 의 k별 승률 곡선
  render_episodes.py   # 궤적 그림 (동작 확인)
  train.py             # PPO 학습 1회 (k, seed)
  run_experiments.py   # 학습 여러 개를 GPU 에 배정해 동시 실행
  evaluate.py          # FSM·RL-A·RL-B 평가, 요약표와 그림
  exploration/         # 설계 탐색 재현 스크립트 e1~e6 (과거 설정으로 고정, 논문 부록용)
results/
  calibration/         # 보정 결과 (calibration.json, 그림)
  pilot/               # 파일럿 요약과 그림
  exploration/         # 설계 탐색 출력 (e1~e6 .txt)
  rl/                  # 학습(runs/), 평가(eval/) 결과 — 실습장에서 생성
  render/              # 궤적 그림
  archive/             # 폐기된 설계의 산출물 (기록용)
```

## 로컬(노트북)에서 실행 (가벼운 확인용)
```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install numpy==1.26.4 pandas==2.2.2 matplotlib==3.8.4 tqdm==4.66.4 gymnasium==0.29.1 stable-baselines3==2.3.2 statsmodels==0.14.2
.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe scripts\calibrate.py --n 2500
.venv\Scripts\python.exe scripts\pilot.py --n 2000
```
Windows에서는 torch 2.3.1 CPU 빌드가 `fbgemm.dll` 오류로 로드되지 않아 2.5.1을 씁니다. 실습장 Docker는 2.3.1(CUDA 12.1)입니다.

## 실습장 PC(Docker)에서 실행
자세한 절차는 [docs/05_실습장_실행가이드.md](docs/05_실습장_실행가이드.md)에 있습니다.
```bash
docker compose build
docker compose run --rm ooda                 # torch / GPU 인식 확인
docker compose up -d train                   # 본 학습 30개 + 평가 (백그라운드)
docker compose logs -f train                 # 진행 상황
```
결과는 호스트의 `./results` 폴더에 저장됩니다.
