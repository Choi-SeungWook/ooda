"""탐색 1: 속도가 같을 때 배치·포획반경·추격자 시야를 바꿔도 FSM(출구순찰)이 이길 수 있는가.
결론: 추격자에게 전체 관측을 줘도 회피형 승률 ~1% → 정보가 아니라 '속도 대칭'이 원인.
결과: results/exploration/e1_layouts.txt
"""
from multiprocessing import Pool

import pandas as pd

from _common import exit_patrol_fsm, hist_cfg, save_text
from ooda.config import EVADER_TYPES
from ooda.runner import run_episode

VARIANTS = {
    "V0 무작위배치": hist_cfg(),
    "V1 침투-차단": hist_cfg(layout="infiltration"),
    "V2 침투-차단+인접포획": hist_cfg(layout="infiltration", capture_radius=1),
    "V3 무작위+인접포획": hist_cfg(capture_radius=1),
    "V4 침투+추격자시야5": hist_cfg(layout="infiltration", pursuer_view=5),
    "V5 침투+추격자시야7": hist_cfg(layout="infiltration", pursuer_view=7),
    "V6 침투+추격자전체관측": hist_cfg(layout="infiltration", pursuer_view=99),
    "V7 무작위+추격자전체관측": hist_cfg(pursuer_view=99),
}


def job(a):
    v, s, et, k = a
    r = run_episode(s, et, k, pursuer=exit_patrol_fsm(), cfg=VARIANTS[v])
    r["variant"] = v
    return r


if __name__ == "__main__":
    jobs = [(v, s, et, k) for v in VARIANTS for et in EVADER_TYPES for k in (1, 2, 4) for s in range(300)]
    with Pool() as p:
        df = pd.DataFrame(p.map(job, jobs, chunksize=20))
    df["cap"] = df.outcome.eq("capture")
    save_text("e1_layouts.txt",
              "[FSM 추격자 승률, T=200, 속도 동일]\n"
              + df.pivot_table(index=["variant", "k"], columns="evader", values="cap").round(3).to_string())
