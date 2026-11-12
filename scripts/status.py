"""학습 진행 상황과 남은 시간 추정.

    docker compose run --rm ooda python scripts/status.py
    (옵션) --runs results/rl/runs --timesteps 10000000 --total 30 --parallel 4

각 실험의 progress.csv 마지막 줄(time/total_timesteps, time/fps)로 진행률을 계산한다.
model_final.zip 이 있으면 완료, train.log 에 Traceback 이 있으면 실패(진행 0)로 본다. 남은 시간 ≈ 남은 스텝 ÷ (평균 fps × 동시 실험 수).
아직 시작하지 않은 실험은 0스텝으로 계산하며, 학습 후 이어지는 평가 시간은 포함하지 않는다.
"""
import argparse
import glob
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join("results", "rl", "runs"))
    ap.add_argument("--timesteps", type=int, default=10_000_000, help="실험당 스텝 (docker-compose.yml 과 같게)")
    ap.add_argument("--total", type=int, default=30, help="전체 실험 수 (k 6개 x seed 5개)")
    ap.add_argument("--parallel", type=int, default=4, help="동시 실험 수")
    args = ap.parse_args()
    T = args.timesteps

    done_steps, fps_list, n_done, n_failed = 0, [], 0, 0
    for d in sorted(glob.glob(os.path.join(args.runs, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        finished = os.path.exists(os.path.join(d, "model_final.zip"))
        steps, fps = 0, 0.0
        f = os.path.join(d, "progress.csv")
        if os.path.exists(f):
            try:
                df = pd.read_csv(f).dropna(subset=["time/fps"])
                if len(df):
                    steps, fps = int(df["time/total_timesteps"].iloc[-1]), float(df["time/fps"].iloc[-1])
            except Exception as e:  # 쓰는 중인 파일을 읽으면 가끔 실패
                print(f"{name:<10} progress.csv 읽기 실패: {e}")
                continue
        best = ""
        e = os.path.join(d, "eval.csv")
        if os.path.exists(e):
            try:
                ev = pd.read_csv(e)
                if len(ev):
                    best = f"  검증 최고 {ev['mixture'].max():.3f} (최근 {ev['mixture'].iloc[-1]:.3f})"
            except Exception:
                pass
        failed = False
        log = os.path.join(d, "train.log")
        if not finished and os.path.exists(log):
            with open(log, encoding="utf-8", errors="replace") as fh:
                failed = "Traceback" in fh.read()
        if finished:
            steps, n_done = T, n_done + 1
            state = "완료"
        elif failed:   # 중단된 실험: 다시 돌리면 처음부터 하므로 진행 0 으로 계산
            n_failed += 1
            state = f"실패 ({100 * steps / T:.0f}%에서 중단, 재실행 필요)"
            steps, fps = 0, 0.0
        elif fps > 0:
            state = f"남은 {(T - steps) / fps / 3600:.1f}시간"
        else:
            state = "시작 전/기록 없음"
        if fps > 0:
            fps_list.append(fps)
        done_steps += steps
        print(f"{name:<10} {100 * steps / T:5.1f}%  fps {fps:6.0f}  {state}{best}")

    rem = args.total * T - done_steps
    print(f"\n실험 완료 {n_done}/{args.total}개, 실패 {n_failed}개, 전체 스텝 기준 {100 * done_steps / (args.total * T):.1f}%")
    if fps_list:
        avg = sum(fps_list) / len(fps_list)
        print(f"남은 학습 시간 약 {rem / (avg * args.parallel) / 3600:.1f}시간 "
              f"(평균 fps {avg:.0f} x 동시 {args.parallel}개 기준, 평가 시간 제외)")
    else:
        print("fps 기록이 아직 없습니다 (학습 시작 직후이면 몇 분 뒤 다시 확인).")


if __name__ == "__main__":
    main()
