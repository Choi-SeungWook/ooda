"""학습 일괄 실행기: (k x 학습 seed) 실험들을 GPU 에 번갈아 배정해 동시에 여러 개 돌린다.

- 이미 model_final.zip 이 있는 실험은 건너뛴다 → 중단돼도 같은 명령으로 다시 실행하면 이어서 진행
  (중단된 실험은 처음부터 다시 학습한다).
- 슬롯 i 는 GPU (i mod GPU 수) 를 쓴다 (CUDA_VISIBLE_DEVICES 로 지정). --gpus "" 이면 CPU.
- 실습장 PC(16 CPU, RTX 2080 x2) 기본값: 동시 4개 x 환경 3개 (= 환경 프로세스 12개 + 학습 본체 4개).
- 실험별 로그: {runs}/k{k}_s{seed}/train.log

사용 예
  python scripts/run_experiments.py                                         # 본 학습 전체 (6 k x 5 seed = 30개)
  python scripts/run_experiments.py --ks 1 --seeds 0 --runs results/rl/noshaping -- --shaping-coef 0   # 성형 없는 대조 1개
  python scripts/run_experiments.py --dry-run                               # 실행할 목록만 출력
  python scripts/run_experiments.py --then-eval                             # 학습이 끝나면 평가까지
"""
import argparse
import os
import subprocess
import sys
import time
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KS = ["1", "5/4", "3/2", "2", "3", "4"]


def run_name(k: str, seed: int) -> str:     # scripts/train.py 와 같은 규칙
    return f"k{k.replace('/', '-')}_s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="+", default=DEFAULT_KS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--timesteps", type=int, default=10_000_000)
    ap.add_argument("--parallel", type=int, default=4, help="동시에 돌릴 실험 수")
    ap.add_argument("--n-envs", type=int, default=3, help="실험 1개당 병렬 환경 수")
    ap.add_argument("--gpus", default="0,1", help='쓸 GPU 번호 (쉼표 구분). "" 이면 CPU')
    ap.add_argument("--runs", default=os.path.join("results", "rl", "runs"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--then-eval", action="store_true", help="학습 후 scripts/evaluate.py 실행")
    ap.add_argument("--eval-workers", type=int, default=15)
    ap.add_argument("--eval-n", type=int, default=3000, help="--then-eval 평가 판수 (조건당)")
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="-- 뒤의 인자는 train.py 에 그대로 전달")
    args = ap.parse_args()
    extra = [a for a in args.train_args if a != "--"]

    gpus = [g for g in args.gpus.split(",") if g.strip() != ""]

    todo, done = [], []
    for k in [str(Fraction(k)) for k in args.ks]:
        for s in args.seeds:
            name = run_name(k, s)
            (done if os.path.exists(os.path.join(args.runs, name, "model_final.zip")) else todo).append((k, s))
    print(f"완료 {len(done)}개, 남음 {len(todo)}개: {[run_name(k, s) for k, s in todo]}", flush=True)
    if args.dry_run:
        return

    running = {}          # slot -> (Popen, name, start, logfile)
    queue = list(todo)
    failed = []
    t0 = time.time()
    while queue or running:
        for slot in range(args.parallel):
            if slot in running or not queue:
                continue
            k, s = queue.pop(0)
            name = run_name(k, s)
            env = dict(os.environ)
            if gpus:
                env["CUDA_VISIBLE_DEVICES"] = gpus[slot % len(gpus)]
                device = "cuda"
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
                device = "cpu"
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--k", k, "--seed", str(s),
                   "--timesteps", str(args.timesteps), "--n-envs", str(args.n_envs), "--device", device,
                   "--out-root", args.runs, *extra]
            os.makedirs(os.path.join(args.runs, name), exist_ok=True)
            log = open(os.path.join(args.runs, name, "train.log"), "w", encoding="utf-8")
            log.write(" ".join(cmd) + "\n")
            log.flush()
            running[slot] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env),
                             name, time.time(), log)
            print(f"[시작] {name} slot={slot} gpu={env['CUDA_VISIBLE_DEVICES'] or 'cpu'}", flush=True)
        time.sleep(5)
        for slot, (proc, name, st, log) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            del running[slot]
            mins = (time.time() - st) / 60
            if rc == 0:
                print(f"[완료] {name} {mins:.1f}분 (전체 경과 {(time.time() - t0) / 3600:.2f}시간)", flush=True)
            else:
                failed.append(name)
                print(f"[실패] {name} 종료코드 {rc} → {os.path.join(args.runs, name, 'train.log')} 확인", flush=True)

    print(f"학습 종료. 실패 {len(failed)}개: {failed}", flush=True)
    if args.then_eval and not failed:
        out = os.path.join(os.path.dirname(os.path.normpath(args.runs)), "eval")
        if os.path.basename(os.path.normpath(args.runs)) != "runs":
            out += "_" + os.path.basename(os.path.normpath(args.runs))
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "evaluate.py"), "--runs", args.runs,
                        "--out", out, "--workers", str(args.eval_workers), "--n", str(args.eval_n)], check=True)


if __name__ == "__main__":
    main()
