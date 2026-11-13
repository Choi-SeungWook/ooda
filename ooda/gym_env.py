"""추격자 시점의 gymnasium 환경 (PPO 학습용).

- 에피소드마다 seed 와 도망자 유형(돌진·회피·기만 균등 혼합)을 무작위로 뽑는다.
- 의사결정 주기 k: 에이전트의 step 1번 = 결정 1번. 다음 결정 시점까지는 같은 행동을 반복하며
  내부에서 환경을 진행한다(runner.run_episode 와 같은 Episode.advance 사용 → FSM 과 규칙이 완전히 같음).
  그래서 k 에서 학습하면 그대로 RL-B(각 k 에서 학습)가 된다.
- 보상: 포획 +1, 도주·시간초과 -1, 환경 스텝마다 -0.001 (결정 사이 반복 스텝도 포함)
        + 잠재함수 기반 보상 성형 F = γ·Φ(s') - Φ(s) (Ng, Harada & Russell, 1999; docs/02 결정 13)
  Φ(s) = -c · (가장 가까운 접촉까지의 경로거리 / 2·격자크기), 접촉이 안 보이면 -c (가장 먼 것과 같음), 종료 상태는 0.
  · 접촉을 찾으면 +, 접촉에 다가가면 +, 접촉을 놓치면 - 로 희소한 승패 보상에 방향을 준다.
  · γ 를 PPO 의 할인율과 같게 두면 최적 정책이 바뀌지 않는다(정책 불변성). 결정 1번 = 1 전이이므로 γ 는 결정 단위.
  · Φ 는 추격자가 실제로 받은 관측(접촉 위치)만 쓴다. 진짜/기만체 구분 정보는 쓰지 않는다 (FSM 과 같은 정보).
  · 성형 보상은 학습에만 쓰고, 평가는 승패(포획 여부)만 본다.
"""
from __future__ import annotations

from fractions import Fraction

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EVADER_TYPES, SEED_BASE, EnvConfig
from .obs import N_CHANNELS, N_CHANNELS_BASE, N_SCALARS, ObsBuilder
from .runner import Episode, is_decision_step

STEP_PENALTY = 0.001


class PursuitGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, k="1", cfg: EnvConfig = EnvConfig(), evader_types=EVADER_TYPES,
                 seed_lo: int = SEED_BASE["train"], seed_hi: int = SEED_BASE["rl_valid"],
                 fixed_episodes: list | None = None, shaping_coef: float = 0.5, gamma: float = 0.99,
                 obs_memory: bool = True):
        """fixed_episodes: [(seed, evader_type), ...] 를 주면 그 순서대로만 진행 (검증용).
        shaping_coef: 보상 성형 계수 c (0 이면 성형 없음). gamma: 성형용 할인율 (PPO 의 gamma 와 같게).
        obs_memory: 관측에 기억 채널(마지막 접촉 위치, 장기 궤적)을 넣을지 (docs/02 결정 15)."""
        super().__init__()
        self.k = Fraction(k).limit_denominator(100)
        self.cfg = cfg
        self.evader_types = tuple(evader_types)
        self.seed_lo, self.seed_hi = seed_lo, seed_hi
        self.fixed = fixed_episodes
        self._fixed_i = 0
        self.shaping_coef, self.gamma = shaping_coef, gamma
        self.obs_memory = obs_memory
        H = W = cfg.size
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (N_CHANNELS if obs_memory else N_CHANNELS_BASE, H, W), np.float32),
            "vec": spaces.Box(0.0, 1.0, (N_SCALARS,), np.float32),
        })
        self.action_space = spaces.Discrete(5)
        self.ep: Episode | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)          # self.np_random 을 seed 로 초기화 (seed 없으면 이어서 사용)
        if self.fixed is not None:
            ep_seed, et = self.fixed[self._fixed_i % len(self.fixed)]
            self._fixed_i += 1
        else:
            ep_seed = int(self.np_random.integers(self.seed_lo, self.seed_hi))
            et = self.evader_types[int(self.np_random.integers(len(self.evader_types)))]
        self.ep = Episode(ep_seed, et, self.cfg)
        self.builder = ObsBuilder(self.ep.env.map, self.cfg, memory=self.obs_memory)
        po = self.ep.env.observe_pursuer()
        self.phi = self._potential(po)
        return self.builder.update(po), {"seed": ep_seed, "evader": et}

    def _potential(self, po) -> float:
        c = self.shaping_coef
        if c == 0:
            return 0.0
        if po.evader_pos is None:
            return -c
        d = self.ep.env.map.dist_from(po.pos)[po.evader_pos]
        return -c * min(1.0, d / (2 * self.cfg.size))

    def step(self, action):
        env = self.ep.env
        a = int(action)
        steps = 0
        while True:
            self.ep.advance(a)
            steps += 1
            if env.done or is_decision_step(env.t, self.k):
                break
        reward = -STEP_PENALTY * steps
        po = env.observe_pursuer()
        obs = self.builder.update(po)
        phi_next = 0.0 if env.done else self._potential(po)      # 종료 상태의 잠재함수 = 0
        shaping = self.gamma * phi_next - self.phi
        self.phi = phi_next
        reward += shaping
        if env.done:
            win = env.outcome == "capture"
            reward += 1.0 if win else -1.0
            info = {"win": win, "outcome": env.outcome, "evader": self.ep.evader_type,
                    "seed": self.ep.seed, "t_end": env.t}
            return obs, reward, True, False, info
        return obs, reward, False, False, {}
