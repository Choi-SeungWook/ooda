"""강화학습 추격자의 관측 생성기 (numpy 만 사용).

학습(gym_env.PursuitGymEnv)과 평가(rl_policy.RLPursuer)가 같은 클래스를 써서 관측이 완전히 같다.
추격자가 받는 정보는 FSM 과 같다: 지도, 출구 후보, 자기 위치, 시야 안 접촉(진짜/기만체 구분 없음), 기만체 식별 사건.
관측은 결정 시점에만 갱신된다(결정 사이에는 관측이 없다는 k 의 정의를 그대로 따름).

격자 채널 (C x H x W, 값 0~1)
  0 장애물          1 출구 후보        2 자기 위치
  3 지금 보이는 접촉  4 접촉 궤적(지금까지 본 접촉 위치, 시간이 지나면 흐려짐)
  5 방문 흔적(내가 결정 시점에 있었던 위치, 흐려짐)
  6 현재 시야 범위    7 식별된 기만체 위치(흐려짐)
스칼라
  경과 시간/T, 마지막 접촉 후 경과/T, 접촉을 본 적 있음, 기만체 식별됨, 지금 보이는 접촉 수/2
"""
from __future__ import annotations

import numpy as np

from .config import EnvConfig
from .env import PursuerObs
from .gridmap import GridMap

N_CHANNELS = 8
N_SCALARS = 5
TRAIL_DECAY = 0.9     # 접촉 궤적: 스텝마다 곱함 (반감기 약 6.6스텝)
VISIT_DECAY = 0.98    # 방문 흔적 (반감기 약 34스텝)
DECOY_DECAY = 0.95    # 식별된 기만체 위치 (반감기 약 13.5스텝)


class ObsBuilder:
    def __init__(self, m: GridMap, cfg: EnvConfig):
        self.m, self.cfg = m, cfg
        H, W = m.H, m.W
        self.grid = np.zeros((N_CHANNELS, H, W), dtype=np.float32)
        self.grid[0] = m.grid
        for e in m.exits:
            self.grid[1].flat[e] = 1.0
        self.view = cfg.view_radius if cfg.pursuer_view is None else cfg.pursuer_view
        self.last_t: int | None = None
        self.last_contact_t: int | None = None
        self.decoy_marked = False

    def update(self, obs: PursuerObs) -> dict:
        g, m = self.grid, self.m
        dt = 0 if self.last_t is None else obs.t - self.last_t
        self.last_t = obs.t
        if dt:
            g[4] *= TRAIL_DECAY ** dt
            g[5] *= VISIT_DECAY ** dt
            g[7] *= DECOY_DECAY ** dt
        g[2] = 0.0
        g[2].flat[obs.pos] = 1.0
        g[5].flat[obs.pos] = 1.0
        g[3] = 0.0
        for c in obs.contacts:
            g[3].flat[c] = 1.0
            g[4].flat[c] = 1.0
        if obs.contacts:
            self.last_contact_t = obs.t
        if obs.decoy_id_pos is not None and not self.decoy_marked:
            self.decoy_marked = True
            g[7].flat[obs.decoy_id_pos] = 1.0
        g[6] = 0.0
        r0, c0 = m.rc(obs.pos)
        v = self.view
        for r in range(max(0, r0 - v), min(m.H, r0 + v + 1)):
            for c in range(max(0, c0 - v), min(m.W, c0 + v + 1)):
                i = r * m.W + c
                if m.visible(obs.pos, i, v):
                    g[6].flat[i] = 1.0
        T = self.cfg.t_max
        since = 1.0 if self.last_contact_t is None else min(1.0, (obs.t - self.last_contact_t) / T)
        vec = np.array([min(1.0, obs.t / T), since, float(self.last_contact_t is not None),
                        float(self.decoy_marked), len(obs.contacts) / 2], dtype=np.float32)
        return {"grid": g.copy(), "vec": vec}
