"""격자 지도: 생성, 이동, 경로거리(BFS), 가중 경로(Dijkstra), 시야(LOS).

칸 위치는 정수 인덱스 idx = r * W + c 로 표현한다 (튜플보다 빠르고 캐시 키로 쓰기 쉽다).
"""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np

from .config import EnvConfig

# 행동: 0 정지, 1 상, 2 하, 3 좌, 4 우
ACTIONS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
N_ACTIONS = len(ACTIONS)
INF = 10 ** 6


class GridMap:
    def __init__(self, grid: np.ndarray, exits: list[int]):
        self.grid = grid                      # 1 = 장애물, 0 = 빈칸
        self.H, self.W = grid.shape
        self.exits = exits                    # 출구 후보 idx 목록
        self.free = [i for i in range(self.H * self.W) if grid.flat[i] == 0]
        # move_table[idx][a] = 행동 a 를 했을 때 도착하는 idx (막히면 제자리)
        self.move_table = [[i] * N_ACTIONS for i in range(self.H * self.W)]
        # neighbors[idx] = 정지를 제외한 이동 가능 이웃 목록
        self.neighbors: list[list[int]] = [[] for _ in range(self.H * self.W)]
        for i in self.free:
            r, c = divmod(i, self.W)
            for a, (dr, dc) in enumerate(ACTIONS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W and grid[nr, nc] == 0:
                    self.move_table[i][a] = nr * self.W + nc
                    if a != 0:
                        self.neighbors[i].append(nr * self.W + nc)
        self._dist_cache: dict[int, np.ndarray] = {}
        self._los_cache: dict[tuple[int, int], bool] = {}

    # ---------- 좌표 ----------
    def rc(self, idx: int) -> tuple[int, int]:
        return divmod(idx, self.W)

    def move(self, idx: int, action: int) -> int:
        return self.move_table[idx][action]

    # ---------- 경로거리 ----------
    def dist_from(self, src: int) -> np.ndarray:
        """src 에서 모든 칸까지의 최단 이동 횟수(BFS). 결과는 캐시한다."""
        d = self._dist_cache.get(src)
        if d is not None:
            return d
        d = np.full(self.H * self.W, INF, dtype=np.int32)
        d[src] = 0
        q = deque([src])
        nb = self.neighbors
        while q:
            u = q.popleft()
            du = d[u] + 1
            for v in nb[u]:
                if d[v] == INF:
                    d[v] = du
                    q.append(v)
        self._dist_cache[src] = d
        return d

    def weighted_field(self, goal: int, cell_cost: np.ndarray) -> np.ndarray:
        """칸마다 진입 비용이 다를 때, 각 칸에서 goal 까지의 최소 누적비용 (역방향 Dijkstra).
        F(goal)=0, F(c)=min_n [cost(n) + F(n)]."""
        F = np.full(self.H * self.W, float(INF))
        F[goal] = 0.0
        pq = [(0.0, goal)]
        nb = self.neighbors
        while pq:
            f, n = heapq.heappop(pq)
            if f > F[n]:
                continue
            cand = f + cell_cost[n]
            for c in nb[n]:
                if cand < F[c]:
                    F[c] = cand
                    heapq.heappush(pq, (cand, c))
        return F

    # ---------- 시야 ----------
    def line_of_sight(self, a: int, b: int) -> bool:
        """a 와 b 를 잇는 브레젠험 직선 위(양 끝 제외)에 장애물이 없으면 True."""
        key = (a, b) if a < b else (b, a)
        v = self._los_cache.get(key)
        if v is not None:
            return v
        r0, c0 = self.rc(key[0])
        r1, c1 = self.rc(key[1])
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dc - dr
        r, c = r0, c0
        v = True
        while (r, c) != (r1, c1):
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr
            if (r, c) != (r1, c1) and self.grid[r, c] == 1:
                v = False
                break
        self._los_cache[key] = v
        return v

    def visible(self, a: int, b: int, radius: int) -> bool:
        ra, ca = self.rc(a)
        rb, cb = self.rc(b)
        if max(abs(ra - rb), abs(ca - cb)) > radius:
            return False
        return self.line_of_sight(a, b)

    # ---------- 행동 선택 보조 ----------
    def greedy_action(self, pos: int, field: np.ndarray, rng: np.random.Generator,
                      allow_stay: bool = True) -> int:
        """field(목표까지의 거리/비용)를 가장 많이 줄이는 행동. 동률은 무작위."""
        best, cands = None, []
        for a in range(N_ACTIONS):
            if a == 0 and not allow_stay:
                continue
            nxt = self.move_table[pos][a]
            if nxt == pos and a != 0:
                continue  # 막힌 방향은 정지와 같으므로 제외
            v = field[nxt]
            if best is None or v < best:
                best, cands = v, [a]
            elif v == best:
                cands.append(a)
        return int(cands[rng.integers(len(cands))])


# ---------------------------------------------------------------------------
# 지도 생성
# ---------------------------------------------------------------------------
def _largest_component(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    label = np.full(grid.shape, -1, dtype=np.int32)
    best_id, best_size, cur = -1, 0, 0
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 0 and label[r, c] == -1:
                q, size = deque([(r, c)]), 0
                label[r, c] = cur
                while q:
                    y, x = q.popleft()
                    size += 1
                    for dy, dx in ACTIONS[1:]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] == 0 and label[ny, nx] == -1:
                            label[ny, nx] = cur
                            q.append((ny, nx))
                if size > best_size:
                    best_id, best_size = cur, size
                cur += 1
    return label == best_id


def generate_map(rng: np.random.Generator, cfg: EnvConfig) -> GridMap:
    """길이 2~5의 벽 조각(건물·담장)을 무작위로 놓아 장애물 비율을 맞춘 뒤,
    고립된 빈칸을 메우고, 가장자리에 서로 떨어진 출구 후보를 배치한다."""
    S = cfg.size
    while True:
        grid = np.zeros((S, S), dtype=np.uint8)
        target = int(cfg.obstacle_ratio * S * S)
        while grid.sum() < target:
            length = int(rng.integers(2, 6))
            horizontal = bool(rng.integers(2))
            r = int(rng.integers(0, S))
            c = int(rng.integers(0, S))
            for k in range(length):
                rr, cc = (r, c + k) if horizontal else (r + k, c)
                if rr < S and cc < S:
                    grid[rr, cc] = 1
        # 가장 큰 연결 영역만 남기고 나머지 빈칸은 장애물로 메움 (갇힌 칸 방지)
        main = _largest_component(grid)
        grid[(grid == 0) & ~main] = 1
        if cfg.layout == "infiltration":
            # 벽이 지도를 가로막아 가장 큰 연결 영역이 도망자/추격자 시작 열에 닿지 않으면 다시 만든다.
            # (드물게 발생: 예 seed 10452885 는 왼쪽 열 전체가 막혀 env.reset 이 실패했다.
            #  정상 지도는 난수 소비가 같으므로 이 검사 전후로 생성 결과가 완전히 같다.)
            (e0, e1), (p0, p1) = cfg.evader_cols, cfg.pursuer_cols
            if not ((grid[:, e0:e1 + 1] == 0).any() and (grid[:, p0:p1 + 1] == 0).any()):
                continue

        # 출구 후보: 가장자리의 빈칸 중에서 서로 exit_min_sep 이상 떨어지게 선택
        if cfg.layout == "infiltration":
            # 출구는 오른쪽 가장자리, 서로 6칸 이상 떨어지게
            border = [r * S + (S - 1) for r in range(S) if grid[r, S - 1] == 0]
            sep = 6
        else:
            border = [r * S + c for r in range(S) for c in range(S)
                      if grid[r, c] == 0 and (r in (0, S - 1) or c in (0, S - 1))]
            sep = cfg.exit_min_sep
        exits = None
        for _ in range(200):
            if len(border) < cfg.n_exits:
                break
            pick = rng.choice(border, size=cfg.n_exits, replace=False)
            rcs = [divmod(int(p), S) for p in pick]
            ok = all(abs(a[0] - b[0]) + abs(a[1] - b[1]) >= sep
                     for i, a in enumerate(rcs) for b in rcs[i + 1:])
            if ok:
                exits = [int(p) for p in pick]
                break
        if exits is not None:
            return GridMap(grid, exits)
