"""추격-도주 게임의 규칙(상태, 관측, 한 스텝 진행, 승패 판정).

- 양측은 동시에 1칸씩 이동한다. 도망자는 이동 속도 비율(evader_speed)에 따라 규칙적으로 제자리에 선다(runner).
- 포획: 이동 후 같은 칸이거나, 서로 자리를 맞바꾼 경우 (스쳐 지나가기 방지).
- 도주 성공: 포획되지 않고 진짜 출구 칸에 도착.
- 제한시간 T 초과 시 도망자 승. 단, T 는 시뮬레이션 후 결과를 잘라서 적용한다
  (에이전트가 남은 시간을 쓰지 않으므로 T 까지의 진행은 T 와 무관하게 동일).

기만체(decoy, 기만형 도망자만 사용)
- 가짜 출구 하나와 같은 높이 근처의 왼쪽 띠에서 출발해 가짜 출구로 향하는 '가짜 접촉'.
- 추격자는 시야 안의 접촉이 진짜 도망자인지 기만체인지 구분하지 못한다.
- 추격자와의 경로거리가 decoy_ident_dist 이하가 되면(또는 자리 교차) 기만체는 사라지고,
  추격자는 그 접촉이 기만체였음을 알게 된다(식별). 가짜 출구에 도착해도 사라진다(식별 아님).
- 기만체는 포획·탈출과 무관하다. 오직 추격자의 관측을 통해서만 승패에 영향을 준다.
- 기만체 배치는 별도 난수로 정하므로, 같은 seed 면 지도·출구·시작 위치는 기만체 유무와 무관하게 같다.
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
    evader_pos: int | None      # 시야 안 접촉 중 가장 가까운 것의 위치 (없으면 None). 진짜/기만체 구분 없음
    contacts: tuple = ()        # 시야 안 모든 접촉의 위치 (정렬, 진짜/기만체 구분 없음) — RL 관측용
    decoy_id_pos: int | None = None   # 기만체를 식별했으면 식별 순간 기만체 위치, 아니면 None


@dataclass
class EvaderObs:
    t: int
    pos: int
    pursuer_pos: int | None


class PursuitEnv:
    def __init__(self, cfg: EnvConfig = EnvConfig()):
        self.cfg = cfg
        self.map: GridMap | None = None

    def reset(self, seed: int, decoy: bool = False) -> None:
        """seed 하나로 지도·출구·진짜 출구·시작 위치가 모두 결정된다 (공통 난수 설계).
        decoy=True 이면 기만체를 추가로 배치한다 (별도 난수라 위의 배치는 그대로)."""
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
        # 관측에서 접촉 거리가 같을 때 쓰는 난수 (추격자가 진짜/기만체를 순서로 구분하지 못하게)
        self.obs_rng = np.random.default_rng([seed, 4242])
        self.decoy_alive = False
        self.d_pos: int | None = None
        self.fake_exit: int | None = None
        self.decoy_id_pos: int | None = None
        self.n_obs_decoy = 0          # 분석용: 추격자가 기만체를 '가장 가까운 접촉'으로 본 결정 횟수
        if decoy:
            self._place_decoy(seed)

    def _place_decoy(self, seed: int) -> None:
        m, cfg = self.map, self.cfg
        self.decoy_rng = rd = np.random.default_rng([seed, 31337])
        others = [x for x in m.exits if x != self.real_exit]
        self.fake_exit = int(others[rd.integers(len(others))])
        fr = m.rc(self.fake_exit)[0]
        lo, hi = cfg.evader_cols
        band = [i for i in m.free if lo <= i % m.W <= hi]
        cand = [i for i in band if abs(m.rc(i)[0] - fr) <= cfg.decoy_row_band] or band
        self.d_pos = int(cand[rd.integers(len(cand))])
        self.decoy_alive = True

    # ---------- 관측 ----------
    def _sees(self, a: int, b: int, radius: int | None = None) -> bool:
        return self.map.visible(a, b, self.cfg.view_radius if radius is None else radius)

    def observe_pursuer(self) -> PursuerObs:
        r = self.cfg.pursuer_view
        vis = []                                  # (위치, 기만체 여부)
        if self._sees(self.p_pos, self.e_pos, r):
            vis.append((self.e_pos, False))
        if self.decoy_alive and self._sees(self.p_pos, self.d_pos, r):
            vis.append((self.d_pos, True))
        nearest = None
        if vis:
            d = self.map.dist_from(self.p_pos)
            tie = self.obs_rng.random(len(vis))   # 거리가 같으면 무작위
            nearest, is_decoy = min(zip(vis, tie), key=lambda x: (d[x[0][0]], x[1]))[0]
            self.n_obs_decoy += is_decoy
        contacts = tuple(sorted(pos for pos, _ in vis))
        return PursuerObs(self.t, self.p_pos, nearest, contacts, self.decoy_id_pos)

    def observe_evader(self) -> EvaderObs:
        vis = self._sees(self.e_pos, self.p_pos)
        return EvaderObs(self.t, self.e_pos, self.p_pos if vis else None)

    # ---------- 진행 ----------
    def step(self, p_action: int, e_action: int, d_action: int = 0) -> str | None:
        assert not self.done
        m = self.map
        p_new = m.move(self.p_pos, p_action)
        e_new = m.move(self.e_pos, e_action)
        if self.decoy_alive:
            d_new = m.move(self.d_pos, d_action)
            d_swapped = (p_new == self.d_pos and d_new == self.p_pos)
            self.d_pos = d_new
            if d_swapped or m.dist_from(p_new)[d_new] <= self.cfg.decoy_ident_dist:
                self.decoy_alive = False          # 식별: 추격자가 기만체였음을 안다
                self.decoy_id_pos = d_new
            elif d_new == self.fake_exit:
                self.decoy_alive = False          # 가짜 출구 도착: 조용히 사라짐
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
