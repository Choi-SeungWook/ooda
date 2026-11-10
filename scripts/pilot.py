"""FSM 파일럿: 도망자 유형 x 의사결정 주기 k 마다 n판을 돌려 승률 곡선을 구한다.
(RL 없이, 지연이 비학습 추격자에게 얼마나 손해인지 = 본 실험의 '비학습' 축)

- 같은 seed 목록을 모든 조건에 공통으로 사용한다 (공통 난수).
- --speed-mode common   : 공통 도망자 속도 (주 분석)
  --speed-mode per_type : 유형별 보정 속도 (강건성 점검)

사용: python scripts/pilot.py --n 2000
결과: results/pilot/fsm_pilot_{mode}.csv, summary_{mode}.txt, fsm_k_curve_{mode}.png
"""
import argparse
import math
import os
import sys
import time
from dataclasses import replace
from fractions import Fraction
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from ooda.config import EVADER_NAMES_KO, EVADER_TYPES, PER_TYPE_SPEED, SEED_BASE, EnvConfig  # noqa: E402
from ooda.plotting import plt, setup_korean_font  # noqa: E402
from ooda.runner import run_episode  # noqa: E402

DEFAULT_KS = ["1", "5/4", "3/2", "2", "3", "4"]


def _job(args):
    seed, et, k, mode = args
    cfg = EnvConfig()
    if mode == "per_type":
        cfg = replace(cfg, evader_speed=PER_TYPE_SPEED[et])
    return run_episode(seed, et, Fraction(k), cfg=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--ks", nargs="+", default=DEFAULT_KS)
    ap.add_argument("--speed-mode", choices=["common", "per_type"], default="common")
    ap.add_argument("--out", default=os.path.join("results", "pilot"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mode = args.speed_mode

    base = SEED_BASE["pilot"]
    jobs = [(base + s, et, k, mode) for et in EVADER_TYPES for k in args.ks for s in range(args.n)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        df = pd.DataFrame(pool.map(_job, jobs, chunksize=50))
    df.to_csv(os.path.join(args.out, f"fsm_pilot_{mode}.csv"), index=False)
    df["win"] = df.outcome.eq("capture")

    win = df.pivot_table(index="k", columns="evader", values="win")[list(EVADER_TYPES)]
    agg = df.groupby(["evader", "k"]).agg(
        t_end=("t_end", "mean"), n_dec=("n_decisions", "mean"),
        search=("dec_search", "sum"), chase=("dec_chase", "sum"), intercept=("dec_intercept", "sum"))
    tot = agg[["search", "chase", "intercept"]].sum(axis=1)
    for c in ["search", "chase", "intercept"]:
        agg[c] = agg[c] / tot
    lines = [f"mode={mode}  episodes={len(df)}  elapsed={time.time() - t0:.1f}s", "",
             "[FSM 추격자 승률]", win.round(3).to_string(), "",
             "[평균 길이 / 결정 횟수 / 모드 비율]", agg.round(2).to_string()]
    text = "\n".join(lines)
    with open(os.path.join(args.out, f"summary_{mode}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

    setup_korean_font()
    fig, ax = plt.subplots(figsize=(6, 4))
    for et in EVADER_TYPES:
        p = win[et]
        se = (p * (1 - p) / args.n) ** 0.5
        ax.errorbar(p.index, p, yerr=1.96 * se, marker="o", capsize=3, label=EVADER_NAMES_KO[et])
    ax.axhline(0.5, color="gray", lw=0.8)
    ax.set_xlabel("추격자 의사결정 주기 k (도망자 = 1)")
    ax.set_ylabel("FSM 추격자 승률")
    ax.set_title("비학습(FSM) 추격자의 지연에 따른 승률" + (" (유형별 속도)" if mode == "per_type" else ""))
    ax.set_ylim(0, 0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"fsm_k_curve_{mode}.png"), dpi=150)


if __name__ == "__main__":
    main()
