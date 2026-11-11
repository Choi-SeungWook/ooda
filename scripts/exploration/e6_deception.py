"""탐색 6: 기만형 재설계 (docs/02 결정 10).

문제: 기만형 v1(보이는 동안 가짜 출구로 향하는 척)이 세 유형 중 가장 잘 잡혔다(k=1 승률 0.58).
절차와 결론:
  A. 진단 (속도 15/17): 발견 후 무슨 일이 일어나는가 → v1 은 발견 후 접촉을 거의 끊지 못한다.
  B. 움직임 기반 기만 변형 D1~D3 (속도 15/17) → 모두 실패. 추격자가 더 빠르고 시야가 짧으면,
     보이는 동안 방향을 트는 기만은 원리적으로 통하지 않는다.
  C. 기만체(회피형 + 가짜 출구로 직진하는 기만체) 식별 거리 r x 판단 초기화 (속도 6/7)
     → r=1(인접 확인) + 초기화 채택. r>=2 이면 기만 효과가 사라짐.
  D. 오라클 상한 (속도 6/7, k=1, 3/2): 기만체 단서의 가치 상한
     O1 = 식별 후 기만체의 목표(가짜) 출구를 추정 후보에서 제외, O2 = 보자마자 기만체임을 앎(무시 + 출구 제외)

이 스크립트는 탐색 당시 설정(속도 고정)으로 고정되어 있다. 현재 기본값과 무관하게 같은 수치를 재현한다.
결과: results/exploration/e6_deception.txt
"""
from dataclasses import replace
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
import pandas as pd

from _common import save_text
from ooda.agents import AvoidEvader, Decoy, FeintEvader, FSMPursuer, RushEvader
from ooda.config import FSM_PATROL_COL, SEED_BASE, EnvConfig
from ooda.env import EvaderObs, PursuerObs, PursuitEnv
from ooda.gridmap import N_ACTIONS
from ooda.runner import agent_rngs, is_decision_step

BASE = SEED_BASE["pilot"]


# ---------------------------------------------------------------------------
# B. 움직임 기반 기만 변형 (모두 '진짜 도망자 혼자' 기만한다)
# ---------------------------------------------------------------------------
class D1(AvoidEvader):
    """회피형 + 추격자를 기억하는 동안에는 가짜 출구를 목표로 우회·도주, 기억이 끊기면 진짜 출구."""

    def reset(self, m, real_exit, rng):
        super().reset(m, real_exit, rng)
        others = [e for e in m.exits if e != real_exit]
        self.fake_exit = int(others[rng.integers(len(others))])

    def _target(self, contact: bool) -> int:
        return self.fake_exit if contact else self.real_exit

    def act(self, obs):
        known = self._remember(obs)
        tgt = self._target(known is not None)
        d_t = self.m.dist_from(tgt)
        if known is None:
            return self.m.greedy_action(obs.pos, d_t, self.rng)
        d_p = self.m.dist_from(known)
        if obs.pursuer_pos is not None and d_p[obs.pos] <= self.flee_dist:
            best, cands = None, []
            for a in range(N_ACTIONS):
                nxt = self.m.move(obs.pos, a)
                if nxt == obs.pos and a != 0:
                    continue
                key = (d_p[nxt], -d_t[nxt])
                if best is None or key > best:
                    best, cands = key, [a]
                elif key == best:
                    cands.append(a)
            return int(cands[self.rng.integers(len(cands))])
        cost = 1.0 + self.penalty * np.maximum(0, self.danger_radius - d_p)
        F = self.m.weighted_field(tgt, cost)
        return self.m.greedy_action(obs.pos, F + cost, self.rng)


class D2(D1):
    """D1 + 양동 접근: 첫 접촉 전에는 가짜 출구 축으로 접근 (발견될 때 증거가 가짜를 가리키게)."""

    def reset(self, m, real_exit, rng):
        super().reset(m, real_exit, rng)
        self.contacted = False

    def _target(self, contact):
        if contact:
            self.contacted = True
        return self.fake_exit if (contact or not self.contacted) else self.real_exit


class D3(D2):
    """D2 + 진짜 출구로 전환한 뒤에도 마지막 목격 위치 주변(6칸)을 우회."""

    def act(self, obs):
        known = self._remember(obs)
        if known is None and self.contacted and self.last_p is not None:
            d_p = self.m.dist_from(self.last_p)
            cost = 1.0 + self.penalty * np.maximum(0, self.danger_radius + 2 - d_p)
            F = self.m.weighted_field(self.real_exit, cost)
            return self.m.greedy_action(obs.pos, F + cost, self.rng)
        return super().act(obs)


SOLO = {"rush": RushEvader, "avoid": AvoidEvader, "feint(v1)": FeintEvader, "D1": D1, "D2": D2, "D3": D3}


# ---------------------------------------------------------------------------
# C, D. FSM 변형
# ---------------------------------------------------------------------------
class NoForgetFSM(FSMPursuer):
    """기만체를 식별해도 판단을 초기화하지 않음 (오염된 출구 추정을 계속 믿음)."""

    def _forget(self):
        pass


class OracleFSM(FSMPursuer):
    """excluded 에 든 출구는 목표 출구 추정에서 제외 (오라클이 채움)."""

    def reset(self, m, rng):
        super().reset(m, rng)
        self.excluded: set[int] = set()

    def believed_exit(self):
        if self.n_sightings < 2:
            return None
        sc = self.scores.copy()
        for i in self.excluded:
            sc[i] = -np.inf
        return None if sc.max() <= 0 else int(np.argmax(sc))


# ---------------------------------------------------------------------------
# 공통 실행기 (runner.run_episode 와 같은 규칙 + 진단 지표 + 오라클 훅)
# ---------------------------------------------------------------------------
def episode(seed, evader_cls, speed, k=1, decoy=False, pursuer_cls=FSMPursuer, oracle=None, ident=1):
    cfg = replace(EnvConfig(), evader_speed=speed, decoy_ident_dist=ident)
    k = Fraction(k).limit_denominator(100)
    env = PursuitEnv(cfg)
    env.reset(seed, decoy=decoy)
    m = env.map
    rng_p, rng_e = agent_rngs(seed)
    p = pursuer_cls(patrol_col=FSM_PATROL_COL)
    e = evader_cls()
    p.reset(m, rng_p)
    e.reset(m, env.real_exit, rng_e)
    dec = None
    if decoy:
        dec = Decoy()
        dec.reset(m, env.fake_exit, env.decoy_rng)
    fake_i = m.exits.index(env.fake_exit) if decoy else None
    real_i = m.exits.index(env.real_exit)
    e_period = 1 / Fraction(speed).limit_denominator(100)

    a, seen, was, lost, fooled = 0, False, False, 0, False
    while not env.done:
        if is_decision_step(env.t, k):
            obs = env.observe_pursuer()
            if oracle == "O2":
                # 기만체는 보자마자 알아봄: 관측에서 빼고, 기만체의 목표 출구를 후보에서 제외
                if env.decoy_alive and m.visible(env.p_pos, env.d_pos, cfg.view_radius):
                    p.excluded.add(fake_i)
                real_vis = m.visible(env.p_pos, env.e_pos, cfg.view_radius)
                obs = PursuerObs(env.t, env.p_pos, env.e_pos if real_vis else None)
            elif oracle == "O1" and obs.decoy_id_pos is not None:
                p.excluded.add(fake_i)
            if obs.evader_pos is not None:
                seen = True
            elif was:
                lost += 1
            was = obs.evader_pos is not None
            a = p.act(obs)
            b = p.believed_exit()
            if p.mode == "search" and b is not None and b != real_i:
                fooled = True
        moves = is_decision_step(env.t, e_period)
        ea = e.act(env.observe_evader()) if moves else 0
        da = dec.act(EvaderObs(env.t, env.d_pos, None)) if (env.decoy_alive and moves) else 0
        env.step(a, ea, da)
    return dict(win=env.outcome == "capture", t=env.t, seen=seen, lost=lost > 0, fooled=fooled,
                decoy_seen=env.n_obs_decoy > 0, decoy_identified=env.decoy_id_pos is not None)


def job(args):
    tag, kw, seed = args
    r = episode(seed, **kw)
    r.update(tag)
    return r


def run(specs, n, workers=None):
    jobs = [(tag, kw, BASE + s) for tag, kw in specs for s in range(n)]
    with Pool(workers) as pool:
        return pd.DataFrame(pool.map(job, jobs, chunksize=50))


def main():
    out = []
    sp_old, sp_new = 15 / 17, 6 / 7

    # A + B
    specs = [({"evader": name}, {"evader_cls": cls, "speed": sp_old}) for name, cls in SOLO.items()]
    df = run(specs, 2000)
    g = df.groupby("evader", sort=False)
    det = df[df.seen].groupby("evader", sort=False)
    tab = pd.DataFrame({"승률": g.win.mean(), "발견율": g.seen.mean(), "발견 후 승률": det.win.mean(),
                        "발견 후 접촉상실": det.lost.mean(), "가짜출구로 속음": g.fooled.mean(),
                        "속은 판 승률": df[df.fooled].groupby("evader", sort=False).win.mean()})
    out += ["[A·B] 움직임 기반 기만 (속도 15/17, k=1, 2,000판/유형)",
            "  속음 = 탐색 모드에서 가짜 출구를 추정 출구로 둔 적이 있음", tab.round(3).to_string(), ""]

    # C
    specs = [({"evader": "rush", "r": "-", "reset": "-"}, {"evader_cls": RushEvader, "speed": sp_new}),
             ({"evader": "avoid", "r": "-", "reset": "-"}, {"evader_cls": AvoidEvader, "speed": sp_new})]
    for r in (0, 1, 2, 3):
        for reset, cls in (("no", NoForgetFSM), ("yes", FSMPursuer)):
            specs.append(({"evader": "avoid+decoy", "r": r, "reset": reset},
                          {"evader_cls": AvoidEvader, "speed": sp_new, "decoy": True, "pursuer_cls": cls, "ident": r}))
    df = run(specs, 2000)
    g = df.groupby(["evader", "r", "reset"], sort=False)
    tab = pd.DataFrame({"승률": g.win.mean(), "기만체 관측": g.decoy_seen.mean(),
                        "기만체 식별": g.decoy_identified.mean(), "가짜출구로 속음": g.fooled.mean()})
    out += ["[C] 기만체 식별 거리 r x 식별 시 판단 초기화 (속도 6/7, k=1, 2,000판/조건)", tab.round(3).to_string(), ""]

    # D
    specs = []
    for k in ("1", "3/2"):
        specs += [({"k": k, "evader": "rush", "oracle": "-"}, {"evader_cls": RushEvader, "speed": sp_new, "k": k}),
                  ({"k": k, "evader": "avoid", "oracle": "-"}, {"evader_cls": AvoidEvader, "speed": sp_new, "k": k})]
        for o in ("base", "O1", "O2"):
            specs.append(({"k": k, "evader": "avoid+decoy", "oracle": o},
                          {"evader_cls": AvoidEvader, "speed": sp_new, "k": k, "decoy": True,
                           "pursuer_cls": OracleFSM, "oracle": None if o == "base" else o}))
    n = 3000
    df = run(specs, n)
    g = df.groupby(["k", "evader", "oracle"], sort=False)
    tab = pd.DataFrame({"승률": g.win.mean()})
    tab["95%CI 반폭"] = 1.96 * np.sqrt(tab["승률"] * (1 - tab["승률"]) / n)
    out += ["[D] 오라클 상한 (속도 6/7, 3,000판/조건)",
            "  O1 = 식별 후 가짜 출구 제외, O2 = 보자마자 기만체임을 앎", tab.round(3).to_string()]
    save_text("e6_deception.txt", "\n".join(out))


if __name__ == "__main__":
    main()
