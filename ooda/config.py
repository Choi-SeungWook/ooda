"""환경 파라미터. 논문 '연구 설계' 절의 표와 1:1로 대응시키기 위해 한 곳에 모아 둔다."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    size: int = 20                # 격자 한 변 길이 (20 x 20)
    obstacle_ratio: float = 0.10  # 장애물 비율 (전체 칸 대비)
    n_exits: int = 3              # 출구 후보 수 (그중 1개만 진짜)
    exit_min_sep: int = 10        # 출구 후보끼리의 최소 맨해튼 거리
    view_radius: int = 3          # 시야 반경 (체비셰프 거리 3 = 7x7), 장애물이 시야를 가림
    min_pe_dist: int = 8          # 시작 시 추격자-도망자 최소 경로거리
    min_exit_dist: int = 10       # 시작 시 도망자-진짜 출구 최소 경로거리
    t_max: int = 100              # 제한시간 T. 도망자 횡단시간(약 25스텝)의 4배로 넉넉히 둬서 승패에 거의 관여하지 않게 함
    # 배치 방식: "random" = 출구·시작위치 모두 무작위
    #            "infiltration" = 침투-차단: 도망자는 왼쪽 띠에서 출발, 출구는 오른쪽 가장자리, 추격자는 중앙
    layout: str = "infiltration"
    evader_cols: tuple = (0, 2)    # infiltration: 도망자 시작 열 범위 (포함)
    pursuer_cols: tuple = (9, 10)  # infiltration: 추격자 시작 열 범위 (포함)
    capture_radius: int = 0        # 0 = 같은 칸/자리교차만 포획, 1 = 인접(상하좌우)도 포획
    pursuer_view: int | None = None  # 추격자 시야만 따로 지정 (None 이면 view_radius 와 같음)
    # 도망자 이동 속도 비율 (15/17 = 17스텝 중 15번 이동, 결정은 매 스텝).
    # 속도가 같으면 추격자 1명은 원리적으로 잡을 수 없으므로(격자의 cop number = 2) 비대칭을 둔다.
    # 값은 기준선 보정(scripts/calibrate.py)으로 결정: FSM(k=1) 균등혼합 승률 50% (검증 49.7%).
    # 유형별 승률은 맞추지 않고 그대로 둔다: 돌진형 0.57 > 회피형 0.48 > 기만형 0.44 (능력 계층의 조작 점검)
    evader_speed: float = 6 / 7
    # 기만체 (기만형 도망자만): 추격자와의 경로거리가 이 값 이하가 되면 식별되어 사라진다 (1 = 인접 확인).
    # 2 이상이면 기만 효과가 거의 사라짐 (scripts/exploration/e6_deception.py)
    decoy_ident_dist: int = 1
    decoy_row_band: int = 2        # 기만체 출발 행: 가짜 출구의 행 ±2 (열은 도망자와 같은 왼쪽 띠)


EVADER_TYPES = ("rush", "avoid", "deceive")   # 돌진형, 회피형, 기만형

# FSM 추격자의 탐색 차단선 열 위치. 파일럿(scripts/compare_patrol.py)에서 6~16열과 출구순찰을 비교해
# 가장 강한 값을 채택 (기준선 FSM 을 최대한 강하게 두어 '허수아비 기준선' 비판을 피함).
FSM_PATROL_COL = 6

# 강건성 점검용: 유형별로 FSM(k=1) 승률 50%가 되도록 보정한 도망자 속도 (scripts/calibrate.py)
PER_TYPE_SPEED = {"rush": 8 / 9, "avoid": 6 / 7, "deceive": 5 / 6}

# seed 구간: 실험 단계마다 겹치지 않게 분리 (보정에 쓴 판을 평가에 재사용하는 과적합 방지)
SEED_BASE = {
    "eval": 5_000_000,      # 본 실험 평가 (초기 설계 탐색에 쓴 0번대 seed 와 겹치지 않게)
    "pilot": 1_000_000,     # 파일럿·설계 탐색
    "calib": 2_000_000,     # T 보정
    "valid": 3_000_000,     # T 보정 검증
    "train": 10_000_000,    # 강화학습 훈련 (10,000,000 ~ 19,999,999 에서 무작위)
    "rl_valid": 20_000_000, # 강화학습 학습 중 검증 (최고 모델 선택용, 평가 seed 와 분리)
}
EVADER_NAMES_KO = {"rush": "돌진형", "avoid": "회피형", "deceive": "기만형"}
