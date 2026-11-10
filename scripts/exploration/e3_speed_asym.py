"""탐색 3: 도망자를 느리게(0.8, 2/3) 하면 k 효과가 보이는가. 규칙 A(시간초과=도망자 승) vs 규칙 B(=방어자 승).
결론: 규칙 A + 속도 비대칭에서 k 효과가 뚜렷함 → 채택 방향. 단 이 FSM(출구순찰)으로는 최대 승률 < 50%.
결과: results/exploration/e3_speed_asym.txt
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from _common import exit_patrol_fsm, hist_cfg, save_text
from ooda.config import EVADER_TYPES
from ooda.runner import run_episode

V = {f"{lay} spd={s:.2f}": hist_cfg(layout=lay, evader_speed=s)
     for lay in ("random", "infiltration") for s in (0.8, 2 / 3)}
KS = [1, 1.5, 2, 3, 4, 6]


def job(a):
    v, s, et, k = a
    r = run_episode(s, et, k, pursuer=exit_patrol_fsm(), cfg=V[v])
    r["variant"] = v
    return r


def pw(d, T, rule):
    cap = d.outcome.eq("capture") & (d.t_end <= T)
    esc = d.outcome.eq("escape") & (d.t_end <= T)
    return cap.mean() if rule == "A" else 1 - esc.mean()


if __name__ == "__main__":
    jobs = [(v, s, et, k) for v in V for et in EVADER_TYPES for k in KS for s in range(600)]
    with Pool() as p:
        df = pd.DataFrame(p.map(job, jobs, chunksize=30))
    lines = []
    for v in V:
        for rule in ("A", "B"):
            lines.append(f"===== {v}  규칙 {rule}")
            for et in EVADER_TYPES:
                d = df[(df.variant == v) & (df.evader == et)]
                base = d[d.k == 1]
                Ts = np.arange(10, 201)
                curve = np.array([pw(base, T, rule) for T in Ts])
                i = np.argmin(np.abs(curve - 0.5))
                row = "  ".join(f"k={k}:{pw(d[d.k == k], Ts[i], rule):.2f}" for k in KS)
                lines.append(f"{et:8s} T50={Ts[i]:3d} base={curve[i]:.2f} | {row}")
    save_text("e3_speed_asym.txt", "\n".join(lines))
