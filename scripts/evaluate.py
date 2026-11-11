"""본 실험 평가: FSM 과 RL(PPO)을 같은 평가 seed 로 비교한다.

조건 (docs/01 6절)
  H    : FSM,                    평가 k = 전체
  RL-A : k=1 에서 학습한 모델,    평가 k = 전체
  RL-B : 각 k 에서 학습한 모델,   평가 k = 학습한 k (k=1 은 RL-A 와 같은 모델이므로 RL-A 결과를 그대로 씀)
seed: 평가용 5,000,000번대, 모든 조건 공통 (공통 난수)

사용 예
  python scripts/evaluate.py --n 3000 --workers 16
  python scripts/evaluate.py --n 50 --ks 1 2 --workers 4        # 동작 확인
결과: results/rl/eval/eval_episodes.csv, summary.txt, k_curve.png
"""
import argparse
import glob
import math
import os
import sys
import time
from fractions import Fraction
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from ooda.config import EVADER_NAMES_KO, EVADER_TYPES, SEED_BASE, EnvConfig  # noqa: E402
from ooda.runner import run_episode  # noqa: E402

DEFAULT_KS = ["1", "5/4", "3/2", "2", "3", "4"]
CHUNK = 100          # 작업 1개 = 같은 모델·k·유형의 seed 100개
_MODELS = {}         # 작업 프로세스별 모델 캐시


def _pursuer(model_path):
    if model_path is None:
        return None                                   # run_episode 기본값 = FSM
    if model_path not in _MODELS:
        import torch
        torch.set_num_threads(1)
        from ooda.rl_policy import RLPursuer
        _MODELS[model_path] = RLPursuer.load(model_path, EnvConfig())
    return _MODELS[model_path]


def _job(a):
    agent, train_seed, model_path, k, et, s0, s1 = a
    p = _pursuer(model_path)
    rows = []
    for s in range(s0, s1):
        r = run_episode(SEED_BASE["eval"] + s, et, Fraction(k), pursuer=p)
        rows.append({"agent": agent, "train_seed": train_seed, "k": float(Fraction(k)), "k_str": k,
                     "evader": et, "seed": r["seed"], "outcome": r["outcome"], "t_end": r["t_end"],
                     "n_decisions": r["n_decisions"], "decoy_obs": r["decoy_obs"],
                     "decoy_identified": r["decoy_identified"]})
    return rows


def find_models(root, which):
    """runs/k{k}_s{seed}/model_{which}.zip → {(k, seed): path}"""
    out = {}
    for path in glob.glob(os.path.join(root, "k*_s*", f"model_{which}.zip")):
        name = os.path.basename(os.path.dirname(path))
        k_part, s_part = name[1:].rsplit("_s", 1)
        out[(str(Fraction(k_part.replace("-", "/"))), int(s_part))] = path
    return out


def wilson(p, n, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="조건(에이전트·학습seed·k·유형)마다 판수")
    ap.add_argument("--ks", nargs="+", default=DEFAULT_KS)
    ap.add_argument("--which", default="best", choices=["best", "final"])
    ap.add_argument("--runs", default=os.path.join("results", "rl", "runs"))
    ap.add_argument("--out", default=os.path.join("results", "rl", "eval"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip-fsm", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ks = [str(Fraction(k)) for k in args.ks]
    models = find_models(args.runs, args.which)
    print(f"모델 {len(models)}개: {sorted(models)}")

    specs = []                         # (agent, train_seed, model_path, k)
    if not args.skip_fsm:
        specs += [("FSM", -1, None, k) for k in ks]
    for (mk, s), path in sorted(models.items()):
        if mk == "1":
            specs += [("RL-A", s, path, k) for k in ks]
        elif mk in ks:
            specs.append(("RL-B", s, path, mk))
    jobs = [(ag, ts, mp, k, et, s0, min(s0 + CHUNK, args.n))
            for ag, ts, mp, k in specs for et in EVADER_TYPES for s0 in range(0, args.n, CHUNK)]
    print(f"작업 {len(jobs)}개, 판수 {len(specs) * len(EVADER_TYPES) * args.n:,}")

    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            rows += r
            if i % max(1, len(jobs) // 20) == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df["win"] = df.outcome.eq("capture")
    # RL-B 의 k=1 은 RL-A 의 k=1 과 같은 모델
    rlb1 = df[(df.agent == "RL-A") & (df.k_str == "1")].assign(agent="RL-B")
    df = pd.concat([df, rlb1], ignore_index=True)
    df.to_csv(os.path.join(args.out, "eval_episodes.csv"), index=False)

    g = df.groupby(["agent", "k", "evader"]).win.agg(["mean", "count"]).reset_index()
    g[["lo", "hi"]] = g.apply(lambda r: pd.Series(wilson(r["mean"], r["count"])), axis=1)
    table = g.pivot_table(index=["agent", "k"], columns="evader", values="mean")[list(EVADER_TYPES)]
    per_seed = (df[df.agent != "FSM"].groupby(["agent", "train_seed", "k"]).win.mean()
                .unstack("k").round(3))
    text = "\n".join([
        f"평가 seed {SEED_BASE['eval']}~, 조건당 {args.n}판, 모델={args.which}, 소요 {time.time() - t0:.0f}s", "",
        "[승률: 에이전트 x k x 유형 (RL 은 학습 seed 평균)]", table.round(3).to_string(), "",
        "[RL 학습 seed 별 혼합 승률]", per_seed.to_string()])
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    g.to_csv(os.path.join(args.out, "summary_ci.csv"), index=False)
    print(text)

    from ooda.plotting import plt, setup_korean_font
    setup_korean_font()
    fig, axes = plt.subplots(1, len(EVADER_TYPES), figsize=(4.2 * len(EVADER_TYPES), 3.8), sharey=True)
    styles = {"FSM": ("k", "o"), "RL-A": ("#d73027", "s"), "RL-B": ("#4575b4", "^")}
    for ax, et in zip(axes, EVADER_TYPES):
        for ag, (col, mk) in styles.items():
            sub = g[(g.agent == ag) & (g.evader == et)].sort_values("k")
            if len(sub):
                ax.errorbar(sub.k.to_numpy(), sub["mean"].to_numpy(),
                            yerr=[(sub["mean"] - sub.lo).to_numpy(), (sub.hi - sub["mean"]).to_numpy()],
                            color=col, marker=mk, capsize=3, label=ag)
        fsm1 = g[(g.agent == "FSM") & (g.evader == et) & (g.k == 1.0)]["mean"]
        if len(fsm1):
            ax.axhline(float(fsm1.iloc[0]), color="gray", ls=":", lw=1)   # k* 기준선
        ax.set_title(EVADER_NAMES_KO[et])
        ax.set_xlabel("의사결정 주기 k")
    axes[0].set_ylabel("추격자 승률")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "k_curve.png"), dpi=150)


if __name__ == "__main__":
    main()
