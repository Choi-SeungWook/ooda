"""탐색 5: T=100 고정 + 6열 차단선 FSM 에서 공통 도망자 속도별 승률과 k 효과.
(유형별 T 보정이 T≈17~20 으로 짧아져 출구 도달이 불가능해지는 문제 → 속도를 보정 손잡이로 전환한 근거)
결론: 속도 ≈ 7/8 부근에서 혼합 승률 50%, 유형별 40~58% 로 천장/바닥 효과 없음, k 효과 뚜렷.
결과: results/exploration/e5_speed_fixed_T.txt
"""
from dataclasses import replace
from fractions import Fraction
from multiprocessing import Pool

import pandas as pd

from _common import save_text
from ooda.config import EVADER_TYPES, SEED_BASE, EnvConfig
from ooda.runner import run_episode

SPEEDS = ["3/4", "4/5", "5/6", "17/20", "7/8", "9/10"]
KS = [1, 1.25, 1.5, 2, 3, 4]


def job(a):
    sp, s, et, k = a
    r = run_episode(s, et, k, cfg=replace(EnvConfig(), evader_speed=float(Fraction(sp)), t_max=100))
    r["speed"] = sp
    return r


if __name__ == "__main__":
    jobs = [(sp, SEED_BASE["pilot"] + s, et, k) for sp in SPEEDS for et in EVADER_TYPES for k in KS
            for s in range(1000)]
    with Pool() as p:
        df = pd.DataFrame(p.map(job, jobs, chunksize=50))
    df["win"] = df.outcome.eq("capture")
    text = ("[6열 차단선 FSM 승률, T=100]\n"
            + df.pivot_table(index=["speed", "k"], columns="evader", values="win").round(3).to_string()
            + "\n\n균등혼합 (k=1): " + str(df[df.k == 1].groupby("speed").win.mean().round(3).to_dict()))
    save_text("e5_speed_fixed_T.txt", text)
