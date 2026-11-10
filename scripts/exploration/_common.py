"""설계 탐색 스크립트 공통 설정.

탐색 단계마다 '그 당시의' 설정으로 고정해 두어, 현재 기본값(config.py)이 바뀌어도
docs/02_설계결정_기록.md 에 적힌 수치를 그대로 재현할 수 있게 한다.
"""
import os
import sys
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ooda.agents import FSMPursuer  # noqa: E402
from ooda.config import EnvConfig  # noqa: E402

OUT = os.path.join(ROOT, "results", "exploration")
os.makedirs(OUT, exist_ok=True)

# 초기 설계: 무작위 배치, 속도 동일, 시뮬레이션 상한 200
HIST_BASE = replace(EnvConfig(), layout="random", evader_speed=1.0, t_max=200)


def hist_cfg(**kw) -> EnvConfig:
    return replace(HIST_BASE, **kw)


def exit_patrol_fsm() -> FSMPursuer:
    """초기 FSM: 탐색 시 출구 후보를 돌며 순찰."""
    return FSMPursuer(patrol_col=None)


def save_text(name: str, text: str) -> None:
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
