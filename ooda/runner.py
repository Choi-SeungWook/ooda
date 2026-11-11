"""에피소드 실행기: 추격자의 의사결정 주기 k(=OODA 템포)를 적용해 한 판을 진행한다.

의사결정 주기 k (방식 A: 행동 반복)
- 추격자는 '결정 시점'에만 관측하고 새 행동을 고른다.
- 결정 시점이 아닌 스텝에서는 관측 없이 직전 행동을 그대로 반복한다.
- k 는 분수도 허용한다(예: 3/2). 결정 시점은 floor(t/k) 가 바뀌는 스텝으로 규칙적으로 배치된다.
    k=1   : 0,1,2,3,4,...       (매 스텝)
    k=3/2 : 0,2,3,5,6,8,...     (3스텝 중 2번)
    k=2   : 0,2,4,6,...
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .agents import EVADERS, Decoy, FSMPursuer
from .config import FSM_PATROL_COL, EnvConfig
from .env import EvaderObs, PursuitEnv


def is_decision_step(t: int, k: Fraction) -> bool:
    if t == 0:
        return True
    return math.floor(t / k) != math.floor((t - 1) / k)


def agent_rngs(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """환경 seed 에서 추격자·도망자용 난수 생성기를 분리해 만든다 (조건 간 공통 난수)."""
    ss = np.random.SeedSequence([seed, 7919])
    a, b = ss.spawn(2)
    return np.random.default_rng(a), np.random.default_rng(b)


class Episode:
    """한 판의 상태: 환경 + 도망자 + (기만형이면) 기만체.
    추격자 행동을 받아 한 스텝 진행하는 부분(advance)을 FSM 실행기와 강화학습 래퍼(gym_env)가 공유한다.
    그래서 두 추격자는 완전히 같은 규칙과 같은 난수 흐름 위에서 평가된다."""

    def __init__(self, seed: int, evader_type: str, cfg: EnvConfig = EnvConfig()):
        self.seed, self.evader_type, self.cfg = seed, evader_type, cfg
        self.evader = EVADERS[evader_type]()
        use_decoy = getattr(self.evader, "uses_decoy", False)
        self.env = env = PursuitEnv(cfg)
        env.reset(seed, decoy=use_decoy)
        self.rng_p, rng_e = agent_rngs(seed)       # rng_p 는 추격자용
        self.evader.reset(env.map, env.real_exit, rng_e)
        self.decoy = None
        if use_decoy:
            self.decoy = Decoy()
            self.decoy.reset(env.map, env.fake_exit, env.decoy_rng)   # 기만체에게는 가짜 출구가 '목표'
        self.e_period = 1 / Fraction(cfg.evader_speed).limit_denominator(100)

    def advance(self, p_action: int) -> str | None:
        """추격자 행동 p_action 으로 한 스텝 진행 (도망자·기만체는 매 스텝 스스로 결정)."""
        env = self.env
        e_action = self.evader.act(env.observe_evader())
        e_moves = is_decision_step(env.t, self.e_period)   # 도망자 이동 속도 < 1 이면 규칙적으로 제자리
        if not e_moves:
            e_action = 0
        d_action = 0
        if env.decoy_alive and e_moves:                     # 기만체도 도망자와 같은 이동 빈도
            d_action = self.decoy.act(EvaderObs(env.t, env.d_pos, None))
        return env.step(p_action, e_action, d_action)


def run_episode(seed: int, evader_type: str, k: Fraction | float | int = 1,
                pursuer=None, cfg: EnvConfig = EnvConfig(), record: bool = False) -> dict:
    k = Fraction(k).limit_denominator(100)
    ep = Episode(seed, evader_type, cfg)
    env = ep.env
    pursuer = pursuer if pursuer is not None else FSMPursuer(patrol_col=FSM_PATROL_COL)
    pursuer.reset(env.map, ep.rng_p)

    p_action, n_dec = 0, 0
    modes = {"search": 0, "chase": 0, "intercept": 0}
    traj = [(env.p_pos, env.e_pos, None, env.d_pos)] if record else None
    while not env.done:
        if is_decision_step(env.t, k):
            p_action = pursuer.act(env.observe_pursuer())
            n_dec += 1
            if hasattr(pursuer, "mode"):
                modes[pursuer.mode] = modes.get(pursuer.mode, 0) + 1
        ep.advance(p_action)
        if record:
            traj.append((env.p_pos, env.e_pos, getattr(pursuer, "mode", None),
                         env.d_pos if env.decoy_alive else None))

    out = {
        "seed": seed, "evader": evader_type, "k": float(k),
        "outcome": env.outcome, "t_end": env.t, "n_decisions": n_dec,
        **{f"dec_{m}": v for m, v in modes.items()},
        "decoy_obs": env.n_obs_decoy, "decoy_identified": env.decoy_id_pos is not None,
    }
    if record:
        out.update(traj=traj, map=env.map, real_exit=env.real_exit, fake_exit=env.fake_exit)
    return out


def pursuer_wins(outcome: str, t_end: int, T: int) -> bool:
    """제한시간 T 를 사후 적용: T 스텝 이내에 포획했을 때만 추격자 승."""
    return outcome == "capture" and t_end <= T
