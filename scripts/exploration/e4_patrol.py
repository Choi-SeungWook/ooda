"""탐색 4: FSM 탐색 방식 비교 (출구순찰 vs 세로 차단선 순찰, 열 위치별) — 침투-차단 배치.
결론: 출구순찰은 발견율 62% 로 약함. 6열 차단선이 가장 강함(발견율 85%) → 기준선 FSM 으로 채택.
결과: results/exploration/e4_patrol.txt
"""
from fractions import Fraction
from multiprocessing import Pool

import pandas as pd

from _common import hist_cfg, save_text
from ooda.agents import FSMPursuer
from ooda.config import EVADER_TYPES, SEED_BASE
from ooda.runner import run_episode

PATROLS = [None, 6, 8, 10, 12, 14, 16]
SPEEDS = ["1/2", "3/5", "2/3"]


def job(a):
    col, sp, seed, et, k = a
    cfg = hist_cfg(layout="infiltration", evader_speed=float(Fraction(sp)))
    r = run_episode(seed, et, k, pursuer=FSMPursuer(patrol_col=col), cfg=cfg)
    r["patrol"] = "exits" if col is None else f"col{col:02d}"
    r["speed"] = sp
    return r


if __name__ == "__main__":
    jobs = [(c, sp, SEED_BASE["pilot"] + s, et, k) for c in PATROLS for sp in SPEEDS for et in EVADER_TYPES
            for k in (1, 2) for s in range(1000)]
    with Pool() as p:
        df = pd.DataFrame(p.map(job, jobs, chunksize=50))
    df["cap"] = df.outcome.eq("capture")
    df["seen"] = (df.dec_chase + df.dec_intercept) > 0
    lines = []
    for sp in SPEEDS:
        d = df[df.speed == sp]
        lines += [f"===== 도망자 속도 {sp}: FSM 최대 승률 (T=200)",
                  d.pivot_table(index="patrol", columns=["k", "evader"], values="cap").round(3).to_string(),
                  "  발견율 (k=1):",
                  d[d.k == 1].pivot_table(index="patrol", columns="evader", values="seen").round(3).to_string(), ""]
    save_text("e4_patrol.txt", "\n".join(lines))
