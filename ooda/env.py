"""추격-도주 게임의 규칙(상태, 관측, 한 스텝 진행, 승패 판정).

- 양측은 매 스텝 동시에 1칸 이동한다 (이동 속도 동일).
- 포획: 이동 후 같은 칸이거나, 서로 자리를 맞바꾼 경우 (스쳐 지나가기 방지).
- 도주 성공: 포획되지 않고 진짜 출구 칸에 도착.
- 제한시간 T 초과 시 도망자 승. 단, T 는 시뮬레이션 후 결과를 잘라서 적용한다
  (에이전트가 남은 시간을 쓰지 않으므로 T 까지의 진행은 T 와 무관하게 동일).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EnvConfig
from .gridmap import GridMap, generate_map


@dataclass
class PursuerObs:
    t: int
    pos: int
    evader_pos: int | None      # 시야 안이면 위치, 아니면 None


@dataclass
class EvaderObs:
    t: int
    pos: int
    pursuer_pos: int | None


class PursuitEnv:
    def __init__(self, cfg: EnvConfig = EnvConfig()):
        self.cfg = cfg
        self.map: GridMap | None = None

    def reset(self, seed: int) -> None:
        """seed 하나로 지도·출구·진짜 출구·시작 위치가 모두 결정된다 (공통 난수 설계)."""
        rng = np.random.default_rng(seed)
        cfg = self.cfg
        self.map = m = generate_map(rng, cfg)
        self.real_exit = int(m.exits[rng.integers(len(m.exits))])
        d_exit = m.dist_from(self.real_exit)
        exit_set = set(m.exits)

        if cfg.layout == "infiltration":
            lo, hi = cfg.evader_cols
            cand_e = [i for i in m.free if lo <= i % m.W <= hi]
        else:
            cand_e = [i for i in m.free if i not in exit_set and d_exit[i] >= cfg.min_exit_dist]
        while True:
            e = int(cand_e[rng.integers(len(cand_e))])
            d_e = m.dist_from(e)
            if cfg.layout == "infiltration":
                lo, hi = cfg.pursuer_cols
                cand_p = [i for i in m.free if lo <= i % m.W <= hi]
            else:
                cand_p = [i for i in m.free if i not in exit_set and cfg.min_pe_dist <= d_e[i] < 10 ** 6]
            if cand_p:
                p = int(cand_p[rng.integers(len(cand_p))])
                break
        self.e_pos, self.p_pos = e, p
        self.t = 0
        self.done = False
        self.outcome: str | None = None

    # ---------- 관측 ----------
    def _sees(self, a: int, b: int, radius: int | None = None) -> bool:
        return self.map.visible(a, b, self.cfg.view_radius if radius is None else radius)

    def observe_pursuer(self) -> PursuerObs:
        vis = self._sees(self.p_pos, self.e_pos, self.cfg.pursuer_view)
        return PursuerObs(self.t, self.p_pos, self.e_pos if vis else None)

    def observe_evader(self) -> EvaderObs:
        vis = self._sees(self.e_pos, self.p_pos)
        return EvaderObs(self.t, self.e_pos, self.p_pos if vis else None)

    # ---------- 진행 ----------
    def step(self, p_action: int, e_action: int) -> str | None:
        assert not self.done
        m = self.map
        p_new = m.move(self.p_pos, p_action)
        e_new = m.move(self.e_pos, e_action)
        swapped = (p_new == self.e_pos and e_new == self.p_pos)
        self.p_pos, self.e_pos = p_new, e_new
        self.t += 1
        pr, pc = m.rc(p_new)
        er, ec = m.rc(e_new)
        adjacent = abs(pr - er) + abs(pc - ec) <= self.cfg.capture_radius
        if p_new == e_new or swapped or adjacent:
            self.outcome = "capture"
        elif e_new == self.real_exit:
            self.outcome = "escape"
        elif self.t >= self.cfg.t_max:
            self.outcome = "timeout"
        self.done = self.outcome is not None
        return self.outcome
