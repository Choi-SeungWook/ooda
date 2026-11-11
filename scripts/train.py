"""PPO 추격자 학습 (실험 1회 = k 1개 x 학습 seed 1개).

- 학습 환경: ooda.gym_env.PursuitGymEnv (도망자 유형 균등 혼합, 결정 주기 k)
- 검증: eval_every 스텝마다 검증 seed(20,000,000번대) 유형별 n판을 runner.run_episode 로 돌려
  혼합 승률이 가장 높은 모델을 model_best.zip 으로 저장 (평가 seed 와 분리되어 있음)
- RL-A = k=1 에서 학습한 모델, RL-B = 각 k 에서 학습한 모델 (k=1 의 RL-B 는 RL-A 와 같음)

사용 예
  python scripts/train.py --k 1 --seed 0 --timesteps 5000000 --n-envs 4 --device cuda
  python scripts/train.py --k 3/2 --seed 0 --timesteps 20000 --n-envs 2 --eval-every 10000 --eval-n 20   # 동작 확인
결과: results/rl/runs/k{k}_s{seed}/ (model_best.zip, model_final.zip, eval.csv, progress.csv, monitor/, config.json)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from ooda.config import EVADER_TYPES, SEED_BASE, EnvConfig  # noqa: E402
from ooda.gym_env import PursuitGymEnv  # noqa: E402
from ooda.rl_policy import POLICY_KWARGS, RLPursuer  # noqa: E402
from ooda.runner import run_episode  # noqa: E402


def run_name(k: str, seed: int) -> str:
    return f"k{k.replace('/', '-')}_s{seed}"


class ValidationCallback(BaseCallback):
    """주기적으로 검증 seed 에서 승률을 재고 최고 모델을 저장한다."""

    def __init__(self, out: str, k: str, cfg: EnvConfig, every: int, n: int):
        super().__init__()
        self.out, self.k, self.cfg, self.every, self.n = out, k, cfg, every, n
        self.next_at = every
        self.best = -1.0
        self.rows = []

    def validate(self) -> None:
        t0 = time.time()
        pursuer = RLPursuer(self.model, self.cfg)
        wins = {}
        for et in EVADER_TYPES:
            w = [run_episode(SEED_BASE["rl_valid"] + s, et, Fraction(self.k), pursuer=pursuer, cfg=self.cfg)
                 ["outcome"] == "capture" for s in range(self.n)]
            wins[et] = float(np.mean(w))
        mix = float(np.mean(list(wins.values())))
        row = {"timesteps": self.num_timesteps, "mixture": mix, **wins, "sec": round(time.time() - t0, 1)}
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(os.path.join(self.out, "eval.csv"), index=False)
        for key, v in row.items():
            if key != "timesteps":
                self.logger.record(f"valid/{key}", v)
        if mix > self.best:
            self.best = mix
            self.model.save(os.path.join(self.out, "model_best"))
        print(f"[valid] t={self.num_timesteps:,} mix={mix:.3f} "
              + " ".join(f"{et}={wins[et]:.3f}" for et in EVADER_TYPES) + f" (best {self.best:.3f})", flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            self.next_at += self.every
            self.validate()
        return True

    def _on_training_end(self) -> None:
        self.validate()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="1", help="의사결정 주기 (예: 1, 5/4, 3/2, 2, 3, 4)")
    ap.add_argument("--seed", type=int, default=0, help="학습 seed (0~4)")
    ap.add_argument("--timesteps", type=int, default=10_000_000, help="총 결정 횟수(에이전트 스텝)")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--device", default="auto", help="cuda / cpu / auto")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--eval-n", type=int, default=200, help="검증 판수 (유형별)")
    # PPO 하이퍼파라미터
    ap.add_argument("--n-steps", type=int, default=512, help="환경 1개당 롤아웃 길이")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--shaping-coef", type=float, default=0.5,
                    help="잠재함수 기반 보상 성형 계수 c (0 = 성형 없음, docs/02 결정 13)")
    ap.add_argument("--out-root", default=os.path.join("results", "rl", "runs"))
    args = ap.parse_args()

    k = str(Fraction(args.k).limit_denominator(100))
    out = os.path.join(args.out_root, run_name(k, args.seed))
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    cfg = EnvConfig()

    meta = {"k": k, "seed": args.seed, "args": vars(args), "env_config": asdict(cfg),
            "git_commit": git_commit(), "torch": torch.__version__,
            "cuda": torch.cuda.is_available(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 환경 seed: 학습 seed 와 k 에서 결정 (같은 설정이면 재현 가능)
    env_seed = args.seed * 100_003 + int(Fraction(k) * 1000)
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = make_vec_env(PursuitGymEnv, n_envs=args.n_envs, seed=env_seed, vec_env_cls=vec_cls,
                        env_kwargs={"k": k, "cfg": cfg, "shaping_coef": args.shaping_coef, "gamma": args.gamma},
                        monitor_dir=os.path.join(out, "monitor"),
                        monitor_kwargs={"info_keywords": ("win", "evader")})

    model = PPO("MultiInputPolicy", venv, policy_kwargs=POLICY_KWARGS, seed=args.seed, device=args.device,
                n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
                learning_rate=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
                ent_coef=args.ent_coef, clip_range=args.clip, verbose=0)
    model.set_logger(configure(out, ["csv"]))   # 학습 곡선: progress.csv
    print(f"[train] {run_name(k, args.seed)} device={model.device} n_envs={args.n_envs} "
          f"timesteps={args.timesteps:,}", flush=True)

    t0 = time.time()
    cb = ValidationCallback(out, k, cfg, args.eval_every, args.eval_n)
    model.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=False)
    model.save(os.path.join(out, "model_final"))
    venv.close()

    meta.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_sec=round(time.time() - t0),
                best_valid_mixture=cb.best)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] {run_name(k, args.seed)} {meta['elapsed_sec']}s best_valid={cb.best:.3f}", flush=True)


if __name__ == "__main__":
    main()
