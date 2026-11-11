"""규칙 기반 에이전트: 도망자 3종(돌진형·회피형·기만형), 기만체, FSM 추격자(탐색·추격·차단).

도망자 유형은 능력 계층이다: 돌진형(무반응) → 회피형(반응) → 기만형(반응 + 기만체).

모든 에이전트는 같은 인터페이스를 가진다.
  reset(...)  : 에피소드 시작 시 1회 호출 (알고 있는 정보와 난수 생성기를 받음)
  act(obs)    : 관측을 받아 행동 번호(0 정지, 1 상, 2 하, 3 좌, 4 우)를 반환
도망자는 진짜 출구를 알고, 추격자는 출구 후보 3개만 안다(어느 것이 진짜인지 모름).
"""
from __future__ import annotations

import numpy as np

from .env import EvaderObs, PursuerObs
from .gridmap import N_ACTIONS, GridMap


# ===========================================================================
# 도망자
# ===========================================================================
class EvaderBase:
    memory = 5  # 추격자를 놓친 뒤 마지막 목격 위치를 기억하는 스텝 수

    def reset(self, m: GridMap, real_exit: int, rng: np.random.Generator) -> None:
        self.m, self.real_exit, self.rng = m, real_exit, rng
        self.d_exit = m.dist_from(real_exit)
        self.last_p: int | None = None
        self.last_p_t = -10 ** 6

    def _remember(self, obs: EvaderObs) -> int | None:
        """시야에 추격자가 있으면 기억을 갱신하고, 기억이 유효하면 그 위치를 반환."""
        if obs.pursuer_pos is not None:
            self.last_p, self.last_p_t = obs.pursuer_pos, obs.t
        if obs.t - self.last_p_t <= self.memory:
            return self.last_p
        return None

    def _flee(self, pos: int, d_p: np.ndarray) -> int:
        """추격자와의 경로거리를 최대화. 동률이면 진짜 출구에 더 가까운 쪽, 그래도 동률이면 무작위."""
        best, cands = None, []
        for a in range(N_ACTIONS):
            nxt = self.m.move(pos, a)
            if nxt == pos and a != 0:
                continue
            key = (d_p[nxt], -self.d_exit[nxt])
            if best is None or key > best:
                best, cands = key, [a]
            elif key == best:
                cands.append(a)
        return int(cands[self.rng.integers(len(cands))])

    def act(self, obs: EvaderObs) -> int:
        raise NotImplementedError


class RushEvader(EvaderBase):
    """돌진형: 추격자를 무시하고 진짜 출구까지 최단경로로 직진."""

    def act(self, obs: EvaderObs) -> int:
        return self.m.greedy_action(obs.pos, self.d_exit, self.rng)


class AvoidEvader(EvaderBase):
    """회피형: 추격자 주변을 우회하는 경로로 출구를 향하고, 추격자가 가까우면(2칸 이내) 도주."""
    flee_dist = 2       # 이 거리 이하로 붙으면 도주 모드
    danger_radius = 4   # 추격자로부터 이 거리 미만인 칸은 진입 비용 증가
    penalty = 3.0       # 위험 칸 진입 비용 가중치

    def act(self, obs: EvaderObs) -> int:
        known = self._remember(obs)
        if known is None:
            return self.m.greedy_action(obs.pos, self.d_exit, self.rng)
        d_p = self.m.dist_from(known)
        if obs.pursuer_pos is not None and d_p[obs.pos] <= self.flee_dist:
            return self._flee(obs.pos, d_p)
        cost = 1.0 + self.penalty * np.maximum(0, self.danger_radius - d_p)
        F = self.m.weighted_field(self.real_exit, cost)
        return self.m.greedy_action(obs.pos, F + cost, self.rng)


class FeintEvader(EvaderBase):
    """[폐기된 기만형 v1, scripts/exploration/e1~e5 재현용] 기만형: 추격자에게 보이는 동안(최근 3스텝 이내 목격)은 가짜 출구로 향하는 척하고,
    시야에서 벗어나면 진짜 출구로 방향을 바꾼다. 추격자가 2칸 이내면 도주.

    폐기 이유: 더 빠른 추격자에게 보이는 동안 방향을 트는 기만은 접촉을 끊지 못해 오히려 잘 잡힌다
    (k=1 승률 0.58, 세 유형 중 가장 쉬움). 근거는 docs/02 결정 10, e6_deception.py."""
    memory = 3
    flee_dist = 2

    def reset(self, m: GridMap, real_exit: int, rng: np.random.Generator) -> None:
        super().reset(m, real_exit, rng)
        others = [e for e in m.exits if e != real_exit]
        self.fake_exit = int(others[rng.integers(len(others))])
        self.d_fake = m.dist_from(self.fake_exit)

    def act(self, obs: EvaderObs) -> int:
        known = self._remember(obs)
        if known is None:
            return self.m.greedy_action(obs.pos, self.d_exit, self.rng)
        d_p = self.m.dist_from(known)
        if obs.pursuer_pos is not None and d_p[obs.pos] <= self.flee_dist:
            return self._flee(obs.pos, d_p)
        # 가짜 출구 쪽으로 이동하되, 추격자에게 인접하는 칸은 가능하면 피한다
        best, cands = None, []
        for a in range(N_ACTIONS):
            nxt = self.m.move(obs.pos, a)
            if nxt == obs.pos and a != 0:
                continue
            key = (d_p[nxt] <= 1, self.d_fake[nxt])
            if best is None or key < best:
                best, cands = key, [a]
            elif key == best:
                cands.append(a)
        return int(cands[self.rng.integers(len(cands))])


class DeceiveEvader(AvoidEvader):
    """기만형: 진짜 도망자는 회피형과 똑같이 행동하고, 기만체(Decoy) 1개를 함께 운용한다.
    기만체의 배치·소멸·식별은 환경(env.py)이 처리하고, 기만체의 이동은 Decoy 가 정한다."""
    uses_decoy = True


class Decoy(RushEvader):
    """기만체: 가짜 출구로 최단경로 직진, 추격자를 무시 (reset 의 real_exit 자리에 가짜 출구를 넘긴다).
    속도는 도망자와 같다(runner 가 같은 이동 빈도를 적용)."""


EVADERS = {"rush": RushEvader, "avoid": AvoidEvader, "deceive": DeceiveEvader}
LEGACY_EVADERS = {"feint": FeintEvader}   # 설계 탐색 재현 전용


# ===========================================================================
# 추격자 (비학습, 유한상태기계)
# ===========================================================================
class FSMPursuer:
    """모드 3개를 규칙으로 전환하는 추격자.

    탐색(search) : 도망자가 안 보임 → 추정 목표 출구를 지키러 감 / 없으면 마지막 목격 지점 / 없으면 출구 순찰
    추격(chase)  : 도망자가 보임 → 도망자 위치로 최단경로 이동
    차단(intercept): 도망자가 보이고 목표 출구가 추정됨 → 도망자의 예상 경로 위에서
                    내가 먼저(또는 동시에) 도착할 수 있는 가장 가까운 지점으로 이동

    목표 출구 추정: 도망자를 볼 때마다 각 출구까지의 거리가 얼마나 줄었는지(진척도)를 누적.

    기만체 대응 (추격자는 접촉이 진짜인지 기만체인지 모른다)
    - 관측되는 접촉이 둘이면 가까운 쪽을 도망자로 본다 (환경이 고름, 동점은 무작위).
    - 기만체 식별 사건을 받으면, 섞였을 수 있는 정보(출구 추정·마지막 목격 위치)를 모두 버린다.
    - 직전 목격 위치에서 지난 스텝 수 안에 도달할 수 없는 곳에 접촉이 보이면(물리적으로 불가능한 이동)
      다른 개체로 보고 출구 추정을 초기화한다. 접촉이 하나뿐인 판에서는 절대 일어나지 않는다.
    """
    close_chase = 2      # 이 거리 이하로 붙으면 차단 대신 직접 추격
    score_decay = 0.8    # 진척도 누적 시 과거 정보 감쇠 (방향 전환을 따라가기 위함)

    def __init__(self, patrol_col: int | None = None, view: int = 3):
        """patrol_col: 탐색 시 왕복 순찰할 세로 '차단선'의 열 번호. None 이면 출구 후보를 돌며 순찰."""
        self.patrol_col = patrol_col
        self.view = view

    def _make_waypoints(self) -> list[int]:
        """차단선 위에 시야 지름(2*view+1) 간격으로 순찰 지점을 둔다 (장애물이면 가장 가까운 빈칸)."""
        m, col = self.m, self.patrol_col
        pts = []
        for r in range(self.view, m.H, 2 * self.view + 1):
            best = min(m.free, key=lambda i: abs(m.rc(i)[0] - r) + abs(m.rc(i)[1] - col))
            pts.append(best)
        if m.H - 1 - m.rc(pts[-1])[0] > self.view:     # 마지막 지점이 아래 끝을 덮지 못하면 추가
            r = m.H - 1 - self.view
            pts.append(min(m.free, key=lambda i: abs(m.rc(i)[0] - r) + abs(m.rc(i)[1] - col)))
        return pts

    def reset(self, m: GridMap, rng: np.random.Generator) -> None:
        self.m, self.rng = m, rng
        if self.patrol_col is not None:
            self.waypoints = self._make_waypoints()
            self.wp_i, self.wp_dir = None, 1
        self.exits = list(m.exits)
        self.d_ex = [m.dist_from(e) for e in self.exits]
        self.scores = np.zeros(len(self.exits))
        self.n_sightings = 0
        self.prev_seen: int | None = None
        self.prev_seen_t = 0
        self.decoy_known = False
        self.last_seen: int | None = None
        self.last_seen_checked = True
        self.patrol_visited: set[int] = set()
        self.patrol_target: int | None = None
        self.mode = "search"

    # ---------- 판단 보조 ----------
    def _reset_belief(self) -> None:
        self.scores[:] = 0
        self.n_sightings = 0
        self.prev_seen = None

    def _forget(self) -> None:
        """기만체 식별 시: 출구 추정과 마지막 목격 위치를 모두 버린다."""
        self._reset_belief()
        self.last_seen, self.last_seen_checked = None, True

    def _update_belief(self, ev: int, t: int) -> None:
        if self.prev_seen is not None and self.m.dist_from(self.prev_seen)[ev] > t - self.prev_seen_t:
            self._reset_belief()                  # 물리적으로 불가능한 이동 = 다른 개체
        if self.prev_seen is not None:
            for i, d in enumerate(self.d_ex):
                self.scores[i] = self.score_decay * self.scores[i] + (d[self.prev_seen] - d[ev])
        self.prev_seen, self.prev_seen_t = ev, t
        self.n_sightings += 1

    def believed_exit(self) -> int | None:
        if self.n_sightings < 2 or self.scores.max() <= 0:
            return None
        return int(np.argmax(self.scores))

    def _evader_path(self, ev: int, exit_i: int) -> list[int]:
        """도망자가 출구까지 최단경로로 간다고 가정한 예상 경로 (현재 위치 포함)."""
        d = self.d_ex[exit_i]
        path, cur = [ev], ev
        while d[cur] > 0:
            cur = min(self.m.neighbors[cur], key=lambda n: d[n])
            path.append(cur)
        return path

    def _patrol(self, pos: int) -> int:
        if self.patrol_col is not None:
            return self._patrol_line(pos)
        if self.patrol_target is None or pos == self.patrol_target:
            if self.patrol_target is not None:
                self.patrol_visited.add(self.patrol_target)
            if len(self.patrol_visited) >= len(self.exits):
                self.patrol_visited = set()
            d_me = self.m.dist_from(pos)
            rest = [e for e in self.exits if e not in self.patrol_visited and e != pos]
            self.patrol_target = min(rest, key=lambda e: d_me[e])
        return self.patrol_target

    def _patrol_line(self, pos: int) -> int:
        """차단선 순찰: 가장 가까운 순찰 지점부터 시작해 위아래로 왕복(핑퐁)."""
        wp = self.waypoints
        if self.wp_i is None:
            d = self.m.dist_from(pos)
            self.wp_i = min(range(len(wp)), key=lambda i: d[wp[i]])
        if pos == wp[self.wp_i] and len(wp) > 1:
            if not 0 <= self.wp_i + self.wp_dir < len(wp):
                self.wp_dir = -self.wp_dir
            self.wp_i += self.wp_dir
        return wp[self.wp_i]

    # ---------- 결정 ----------
    def act(self, obs: PursuerObs) -> int:
        pos = obs.pos
        target: int | None = None
        if obs.decoy_id_pos is not None and not self.decoy_known:
            self.decoy_known = True
            self._forget()
        if obs.evader_pos is not None:
            ev = obs.evader_pos
            self._update_belief(ev, obs.t)
            self.last_seen, self.last_seen_checked = ev, False
            d_me = self.m.dist_from(pos)
            b = self.believed_exit()
            if b is not None and d_me[ev] > self.close_chase:
                path = self._evader_path(ev, b)
                for i, c in enumerate(path[1:], start=1):
                    if d_me[c] <= i:          # 도망자보다 먼저/동시에 도착 가능
                        target = c
                        self.mode = "intercept"
                        break
            if target is None:
                target = ev
                self.mode = "chase"
        else:
            self.mode = "search"
            b = self.believed_exit()
            if b is not None:
                target = self.exits[b]                     # 추정 출구를 지킴
            elif self.last_seen is not None and not self.last_seen_checked:
                target = self.last_seen
                if pos == target:
                    self.last_seen_checked = True
                    target = self._patrol(pos)
            else:
                target = self._patrol(pos)
        if pos == target:
            return 0
        return self.m.greedy_action(pos, self.m.dist_from(target), self.rng)
