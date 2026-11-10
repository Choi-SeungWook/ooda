"""기준선 보정: FSM(k=1) 추격자 승률이 50%가 되는 도망자 이동속도를 찾는다.

왜 T 가 아니라 속도인가 (파일럿 결과)
  - 제한시간 T 로 맞추면 T 가 17~20스텝으로 짧아져 도망자가 출구에 도달할 수 없게 되고,
    게임이 '버티기'로 변질된다(출구·기만 요소가 무의미). 또한 1스텝 차이로 승률이 15%p 씩 뛰어 정밀 보정 불가.
  - 그래서 T 는 넉넉히 고정(100)하고, 연속적으로 조절 가능한 도망자 이동속도를 손잡이로 쓴다.

주 분석   : 세 유형 공통 속도 1개 → 균등 혼합 승률 50% (물리 조건이 모든 상황에서 동일)
강건성 점검: 유형별 속도 → 각 유형 승률 50%

절차: 보정용 seed n판으로 속도 후보별 승률을 구해 50%에 가장 가까운 값을 고르고,
      보정에 쓰지 않은 검증용 seed n판으로 50% ± 2%p 이내인지 확인 (Wilson 95% CI 보고)

사용: python scripts/calibrate.py --n 2500
결과: results/calibration/calibration.json, calibration_speed.png
"""
import argparse
import json
import math
import os
import sys
from dataclasses import replace
from fractions import Fraction
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from ooda.config import EVADER_NAMES_KO, EVADER_TYPES, SEED_BASE, EnvConfig  # noqa: E402
from ooda.plotting import plt, setup_korean_font  # noqa: E402
from ooda.runner import run_episode  # noqa: E402

SPEEDS = ["3/4", "4/5", "5/6", "6/7", "7/8", "22/25", "15/17", "8/9", "9/10", "10/11", "11/12", "19/20"]


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def _job(a):
    sp, seed, et = a
    cfg = replace(EnvConfig(), evader_speed=float(Fraction(sp)))
    r = run_episode(seed, et, 1, cfg=cfg)
    r["speed"] = sp
    return r


def simulate(base, speeds, n, workers):
    jobs = [(sp, base + s, et) for sp in speeds for et in EVADER_TYPES for s in range(n)]
    with Pool(workers) as p:
        df = pd.DataFrame(p.map(_job, jobs, chunksize=50))
    df["win"] = df.outcome.eq("capture")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = ap.parse_args()
    out = os.path.join("results", "calibration")
    os.makedirs(out, exist_ok=True)

    cal = simulate(SEED_BASE["calib"], SPEEDS, args.n, args.workers)
    cal.to_csv(os.path.join(out, "calib_episodes.csv"), index=False)
    by_type = cal.pivot_table(index="speed", columns="evader", values="win")
    by_type["mixture"] = cal.groupby("speed").win.mean()
    by_type = by_type.loc[SPEEDS]

    # 선택: 50% 에 가장 가까운 속도
    common = (by_type["mixture"] - 0.5).abs().idxmin()
    per_type = {et: (by_type[et] - 0.5).abs().idxmin() for et in EVADER_TYPES}

    # 검증: 선택된 속도들만 새 seed 로 다시 실행
    chosen = sorted(set([common, *per_type.values()]), key=lambda s: float(Fraction(s)))
    val = simulate(SEED_BASE["valid"], chosen, args.n, args.workers)
    val.to_csv(os.path.join(out, "valid_episodes.csv"), index=False)

    def report(rate, n):
        lo, hi = wilson(rate, n)
        return {"win": round(rate, 4), "ci95": [round(lo, 4), round(hi, 4)], "pass": bool(abs(rate - 0.5) <= 0.02)}

    vc = val[val.speed == common]
    result = {
        "n_per_type": args.n,
        "T": EnvConfig().t_max,
        "calib_table": by_type.round(4).to_dict(orient="index"),
        "common_speed": {
            "speed": common,
            "valid_mixture": report(float(vc.win.mean()), len(vc)),
            "valid_per_type": {et: report(float(vc[vc.evader == et].win.mean()), args.n) for et in EVADER_TYPES},
        },
        "per_type_speed": {
            et: {"speed": sp, "valid": report(float(val[(val.speed == sp) & (val.evader == et)].win.mean()), args.n)}
            for et, sp in per_type.items()
        },
    }
    with open(os.path.join(out, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    setup_korean_font()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = [float(Fraction(s)) for s in SPEEDS]
    for et in EVADER_TYPES:
        ax.plot(x, by_type[et], "o-", label=EVADER_NAMES_KO[et])
    ax.plot(x, by_type["mixture"], "k--", label="균등 혼합")
    ax.axhline(0.5, color="gray", lw=0.8)
    ax.axvline(float(Fraction(common)), color="gray", ls=":", lw=0.8)
    ax.set_xlabel("도망자 이동속도 (추격자 = 1)")
    ax.set_ylabel("FSM 추격자 승률 (k=1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "calibration_speed.png"), dpi=150)

    print(by_type.round(3).to_string())
    print()
    print(json.dumps({k: v for k, v in result.items() if k != "calib_table"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
