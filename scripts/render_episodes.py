"""동작 확인용: 도망자 유형별로 몇 판을 돌려 궤적 그림을 저장한다.

사용: python scripts/render_episodes.py --k 1 --n 3
결과: results/render/episodes_k{k}.png
"""
import argparse
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ooda.plotting import plt, setup_korean_font  # noqa: E402
from ooda.config import EVADER_TYPES  # noqa: E402
from ooda.runner import run_episode  # noqa: E402

OUTCOME_KO = {"capture": "포획", "escape": "탈출", "timeout": "시간초과"}
setup_korean_font()


def draw(ax, ep):
    m = ep["map"]
    ax.imshow(m.grid, cmap="Greys", vmin=0, vmax=1.6)
    for e in m.exits:
        r, c = m.rc(e)
        color = "#1a9850" if e == ep["real_exit"] else "#fdae61"
        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color=color, alpha=.8))
    ps = [m.rc(p) for p, _, _ in ep["traj"]]
    es = [m.rc(e) for _, e, _ in ep["traj"]]
    ax.plot([c for _, c in ps], [r for r, _ in ps], "-", color="#d73027", lw=2, label="추격자")
    ax.plot([c for _, c in es], [r for r, _ in es], "-", color="#4575b4", lw=2, label="도망자")
    ax.plot(ps[0][1], ps[0][0], "o", color="#d73027", ms=8)
    ax.plot(es[0][1], es[0][0], "o", color="#4575b4", ms=8)
    ax.plot(ps[-1][1], ps[-1][0], "X", color="#d73027", ms=10)
    ax.plot(es[-1][1], es[-1][0], "X", color="#4575b4", ms=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{ep['evader']} seed={ep['seed']}\n{OUTCOME_KO[ep['outcome']]} t={ep['t_end']}", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=str, default="1")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    k = Fraction(args.k)
    fig, axes = plt.subplots(len(EVADER_TYPES), args.n, figsize=(3.2 * args.n, 3.6 * len(EVADER_TYPES)))
    for i, et in enumerate(EVADER_TYPES):
        for j in range(args.n):
            ep = run_episode(args.seed0 + j, et, k, record=True)
            draw(axes[i][j], ep)
    axes[0][0].legend(loc="lower left", fontsize=7)
    fig.suptitle(f"FSM 추격자 vs 도망자 3종 (k={args.k}) — 초록: 진짜 출구, 주황: 가짜 출구, ●시작 X끝")
    fig.tight_layout()
    out = os.path.join("results", "render")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"episodes_k{args.k.replace('/', '-')}.png")
    fig.savefig(path, dpi=110)
    print("saved", path)


if __name__ == "__main__":
    main()
