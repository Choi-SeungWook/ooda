"""탐색 2: 시간초과를 '추격자(방어자) 승'으로 바꾸면 T 로 50% 보정이 되는가, k 효과가 보이는가.
결론: 보정은 되지만 k=1→6 에서 승률이 0.50→0.45 정도로만 변함 → 추격자가 승패에 거의 영향 없음. 기각.
결과: results/exploration/e2_timeout_rule.txt
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from _common import exit_patrol_fsm, hist_cfg, save_text
from ooda.config import EVADER_TYPES
from ooda.runner import run_episode

V = {"V1 침투-차단": hist_cfg(layout="infiltration"),
     "V2 침투-차단+인접포획": hist_cfg(layout="infiltration", capture_radius=1),
     "V0 무작위": hist_cfg()}
KS = [1, 1.5, 2, 3, 4, 6]


def job(a):
    v, s, et, k = a
    r = run_episode(s, et, k, pursuer=exit_patrol_fsm(), cfg=V[v])
    r["variant"] = v
    return r


if __name__ == "__main__":
    jobs = [(v, s, et, k) for v in V for et in EVADER_TYPES for k in KS for s in range(600)]
    with Pool() as p:
        df = pd.DataFrame(p.map(job, jobs, chunksize=30))
    lines = ["[규칙: 시간초과 = 방어자 승. 유형별 T50 에서의 k별 방어자 승률]"]
    for v in V:
        lines.append(f"===== {v}")
        for et in EVADER_TYPES:
            d = df[(df.variant == v) & (df.evader == et)]
            base = d[d.k == 1]
            esc, t = base.outcome.eq("escape").to_numpy(), base.t_end.to_numpy()
            Ts = np.arange(10, 120)
            pw = np.array([1 - np.mean(esc & (t <= T)) for T in Ts])
            T50 = int(Ts[np.argmin(np.abs(pw - 0.5))])
            row = "  ".join(f"k={k}:{1 - np.mean(d[d.k == k].outcome.eq('escape') & (d[d.k == k].t_end <= T50)):.2f}"
                            for k in KS)
            lines.append(f"{et:8s} T50={T50:3d} | {row}")
    save_text("e2_timeout_rule.txt", "\n".join(lines))
