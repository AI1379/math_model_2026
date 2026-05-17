#!/usr/bin/env python3
"""
Q2: Computer-assisted look-ahead draw by min-cost flow.

This script implements the draw mechanism discussed in the paper draft:

1. Draw municipal teams first and place them into distinct groups.
2. Draw county-level teams in a fully random order.
3. For the drawn county team, test each currently legal group.
4. Keep only groups that still allow the remaining teams to be completed with
   minimum possible C3 conflict count.

The forward check is a min-cost max-flow model. C1 and C2 are hard constraints.
C3 is encoded as a marginal cost: putting the next county team of city c into
group j costs the number of already placed county teams of city c in group j.
Thus the accumulated cost is exactly the additional C3 pair count.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from q1_grouping import (
    CITY_DATA,
    GROUP_SIZE,
    LEVEL_TAG,
    NUM_GROUPS,
    TEAM_INDEX,
    TEAMS,
    check_c1,
    check_c2,
    check_c3,
    metric_f1,
    metric_f2,
    metric_f2_range,
    metric_f3,
)


# ============================================================
# 1. Min-cost max-flow
# ============================================================


@dataclass
class _Edge:
    to: int
    rev: int
    cap: int
    cost: int


class MinCostMaxFlow:
    """Small integer min-cost max-flow by successive shortest augmenting path."""

    def __init__(self, n: int):
        self.graph: List[List[_Edge]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int, cost: int) -> None:
        if cap <= 0:
            return
        fwd = _Edge(v, len(self.graph[v]), cap, cost)
        rev = _Edge(u, len(self.graph[u]), 0, -cost)
        self.graph[u].append(fwd)
        self.graph[v].append(rev)

    def flow(self, s: int, t: int, target: int) -> Tuple[int, int]:
        n = len(self.graph)
        sent = 0
        total_cost = 0

        while sent < target:
            dist = [10**18] * n
            parent: List[Optional[Tuple[int, int]]] = [None] * n
            in_queue = [False] * n

            dist[s] = 0
            queue = deque([s])
            in_queue[s] = True

            while queue:
                u = queue.popleft()
                in_queue[u] = False

                for ei, edge in enumerate(self.graph[u]):
                    if edge.cap <= 0:
                        continue
                    nd = dist[u] + edge.cost
                    if nd < dist[edge.to]:
                        dist[edge.to] = nd
                        parent[edge.to] = (u, ei)
                        if not in_queue[edge.to]:
                            queue.append(edge.to)
                            in_queue[edge.to] = True

            if parent[t] is None:
                break

            add = target - sent
            v = t
            while v != s:
                u, ei = parent[v]  # type: ignore[misc]
                add = min(add, self.graph[u][ei].cap)
                v = u

            v = t
            while v != s:
                u, ei = parent[v]  # type: ignore[misc]
                edge = self.graph[u][ei]
                edge.cap -= add
                self.graph[v][edge.rev].cap += add
                total_cost += add * edge.cost
                v = u

            sent += add

        return sent, total_cost


@dataclass
class _FlowEdge:
    to: int
    rev: int
    cap: int


class Dinic:
    """Max-flow used as a fast F1=0 feasibility test."""

    def __init__(self, n: int):
        self.graph: List[List[_FlowEdge]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int) -> None:
        if cap <= 0:
            return
        fwd = _FlowEdge(v, len(self.graph[v]), cap)
        rev = _FlowEdge(u, len(self.graph[u]), 0)
        self.graph[u].append(fwd)
        self.graph[v].append(rev)

    def max_flow(self, s: int, t: int) -> int:
        flow = 0
        n = len(self.graph)

        while True:
            level = [-1] * n
            level[s] = 0
            queue = deque([s])

            while queue:
                u = queue.popleft()
                for edge in self.graph[u]:
                    if edge.cap > 0 and level[edge.to] < 0:
                        level[edge.to] = level[u] + 1
                        queue.append(edge.to)

            if level[t] < 0:
                return flow

            it = [0] * n

            def dfs(u: int, pushed: int) -> int:
                if u == t:
                    return pushed
                while it[u] < len(self.graph[u]):
                    ei = it[u]
                    edge = self.graph[u][ei]
                    if edge.cap > 0 and level[u] + 1 == level[edge.to]:
                        ret = dfs(edge.to, min(pushed, edge.cap))
                        if ret:
                            edge.cap -= ret
                            self.graph[edge.to][edge.rev].cap += ret
                            return ret
                    it[u] += 1
                return 0

            while True:
                pushed = dfs(s, 10**9)
                if not pushed:
                    break
                flow += pushed


# ============================================================
# 2. Flow model for the remaining draw
# ============================================================


def group_label(j: int) -> str:
    return f"G{j + 1:02d}"


def county_counts_by_city_group(groups: List[List[str]]) -> Dict[str, List[int]]:
    counts = {city: [0] * NUM_GROUPS for city in CITY_DATA}
    for j, group in enumerate(groups):
        for name in group:
            team = TEAMS[TEAM_INDEX[name]]
            if team.level != "municipal":
                counts[team.city][j] += 1
    return counts


def c3_from_counts(counts: Dict[str, List[int]]) -> int:
    return sum(n * (n - 1) // 2 for row in counts.values() for n in row)


def remaining_capacity(groups: List[List[str]]) -> List[int]:
    return [GROUP_SIZE - len(group) for group in groups]


def future_min_c3_cost(
    groups: List[List[str]],
    mun_group: Dict[str, int],
    remaining_by_city: Dict[str, int],
) -> Optional[int]:
    """
    Return the minimum additional C3 pair count needed to place all remaining
    county-level teams. None means the current partial draw is infeasible.
    """

    need = sum(remaining_by_city.values())
    if need == 0:
        return 0

    residual = remaining_capacity(groups)
    if sum(residual) != need:
        return None

    active_cities = [city for city, n in remaining_by_city.items() if n > 0]
    counts = county_counts_by_city_group(groups)

    source = 0
    city_base = 1
    group_base = city_base + len(active_cities)
    sink = group_base + NUM_GROUPS
    mcmf = MinCostMaxFlow(sink + 1)

    for ci, city in enumerate(active_cities):
        city_node = city_base + ci
        demand = remaining_by_city[city]
        mcmf.add_edge(source, city_node, demand, 0)

        for j in range(NUM_GROUPS):
            if j == mun_group[city] or residual[j] <= 0:
                continue

            # Convex C3 penalty, linearized by parallel unit-capacity edges:
            # extra 1st team costs a_cj, extra 2nd costs a_cj + 1, ...
            max_units = min(demand, residual[j])
            base_cost = counts[city][j]
            for k in range(max_units):
                mcmf.add_edge(city_node, group_base + j, 1, base_cost + k)

    for j, cap in enumerate(residual):
        mcmf.add_edge(group_base + j, sink, cap, 0)

    sent, cost = mcmf.flow(source, sink, need)
    if sent != need:
        return None
    return cost


def future_zero_c3_feasible(
    groups: List[List[str]],
    mun_group: Dict[str, int],
    remaining_by_city: Dict[str, int],
) -> bool:
    """
    Fast check for whether all remaining teams can still be placed with F1=0.

    This is a max-flow model with city-group capacity 1 after excluding groups
    where that city already has a county-level team.
    """

    need = sum(remaining_by_city.values())
    if need == 0:
        return True

    residual = remaining_capacity(groups)
    if sum(residual) != need:
        return False

    counts = county_counts_by_city_group(groups)
    if c3_from_counts(counts) > 0:
        return False

    active_cities = [city for city, n in remaining_by_city.items() if n > 0]
    source = 0
    city_base = 1
    group_base = city_base + len(active_cities)
    sink = group_base + NUM_GROUPS
    dinic = Dinic(sink + 1)

    for ci, city in enumerate(active_cities):
        city_node = city_base + ci
        dinic.add_edge(source, city_node, remaining_by_city[city])

        for j in range(NUM_GROUPS):
            if j == mun_group[city] or residual[j] <= 0:
                continue
            if counts[city][j] == 0:
                dinic.add_edge(city_node, group_base + j, 1)

    for j, cap in enumerate(residual):
        dinic.add_edge(group_base + j, sink, cap)

    return dinic.max_flow(source, sink) == need


def candidate_final_f1(
    groups: List[List[str]],
    mun_group: Dict[str, int],
    remaining_by_city: Dict[str, int],
    team_name: str,
    group_id: int,
    *,
    hard_zero_first: bool = True,
) -> Optional[Tuple[int, bool]]:
    """Return (best final F1, used_min_cost_fallback) for this candidate group."""

    team = TEAMS[TEAM_INDEX[team_name]]
    if team.level == "municipal":
        raise ValueError("candidate_final_f1 is only for county-level teams")
    if group_id == mun_group[team.city] or len(groups[group_id]) >= GROUP_SIZE:
        return None

    groups[group_id].append(team_name)

    if hard_zero_first and future_zero_c3_feasible(groups, mun_group, remaining_by_city):
        groups[group_id].pop()
        return 0, False

    current_f1 = c3_from_counts(county_counts_by_city_group(groups))
    future_cost = future_min_c3_cost(groups, mun_group, remaining_by_city)
    groups[group_id].pop()

    if future_cost is None:
        return None
    return current_f1 + future_cost, True


# ============================================================
# 3. Look-ahead draw
# ============================================================


@dataclass
class DrawDecision:
    step: int
    team: str
    city: str
    legal_count: int
    feasible_count: int
    safe_count: int
    chosen_group: int
    best_final_f1: int


@dataclass
class DrawStats:
    flow_checks: int = 0
    min_cost_checks: int = 0
    feasible_candidates: int = 0
    safe_candidates: int = 0
    unsafe_candidates: int = 0
    infeasible_candidates: int = 0
    dominated_candidates: int = 0
    critical_steps: int = 0
    infeasible_steps: int = 0
    dominated_steps: int = 0
    forced_steps: int = 0
    max_best_final_f1: int = 0
    decisions: List[DrawDecision] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.decisions is None:
            self.decisions = []


def all_county_teams() -> List[str]:
    teams = []
    for data in CITY_DATA.values():
        teams.extend(name for name, _level in data["county_teams"])
    return teams


def initial_remaining_by_city() -> Dict[str, int]:
    return {city: len(data["county_teams"]) for city, data in CITY_DATA.items()}


def draw_lookahead_flow(
    rng: np.random.Generator,
    *,
    trace: bool = False,
) -> Tuple[List[List[str]], DrawStats]:
    """
    Run one physical-style draw.

    Municipal teams are drawn first. County-level teams are then drawn from one
    mixed pool in random order. For each drawn county-level team, the group pot
    contains only groups that preserve minimum attainable final F1.
    """

    groups: List[List[str]] = [[] for _ in range(NUM_GROUPS)]
    stats = DrawStats()

    # Stage 1: municipal teams.
    cities = list(CITY_DATA.keys())
    rng.shuffle(cities)
    group_slots = np.arange(NUM_GROUPS)
    rng.shuffle(group_slots)

    mun_group: Dict[str, int] = {}
    for i, city in enumerate(cities):
        g = int(group_slots[i])
        groups[g].append(CITY_DATA[city]["municipal"])
        mun_group[city] = g

    # Stage 2: county-level teams in a random physical order.
    county_pool = all_county_teams()
    rng.shuffle(county_pool)
    remaining_by_city = initial_remaining_by_city()

    for step, team_name in enumerate(county_pool, start=1):
        team = TEAMS[TEAM_INDEX[team_name]]
        remaining_by_city[team.city] -= 1

        legal_groups = [
            j
            for j in range(NUM_GROUPS)
            if len(groups[j]) < GROUP_SIZE and j != mun_group[team.city]
        ]

        scores: List[Tuple[int, int]] = []
        for j in legal_groups:
            result = candidate_final_f1(
                groups, mun_group, remaining_by_city, team_name, j
            )
            stats.flow_checks += 1
            if result is not None:
                score, used_min_cost = result
                if used_min_cost:
                    stats.min_cost_checks += 1
                scores.append((j, score))

        if not scores:
            raise RuntimeError(f"No feasible look-ahead group for {team_name}")

        best_final_f1 = min(score for _j, score in scores)
        safe_groups = [j for j, score in scores if score == best_final_f1]
        feasible_count = len(scores)
        safe_count = len(safe_groups)
        infeasible_count = len(legal_groups) - feasible_count
        dominated_count = feasible_count - safe_count
        unsafe_count = len(legal_groups) - safe_count

        stats.feasible_candidates += feasible_count
        stats.safe_candidates += safe_count
        stats.unsafe_candidates += unsafe_count
        stats.infeasible_candidates += infeasible_count
        stats.dominated_candidates += dominated_count
        if unsafe_count > 0:
            stats.critical_steps += 1
        if infeasible_count > 0:
            stats.infeasible_steps += 1
        if dominated_count > 0:
            stats.dominated_steps += 1

        chosen = int(rng.choice(safe_groups))
        groups[chosen].append(team_name)

        if safe_count == 1:
            stats.forced_steps += 1
        stats.max_best_final_f1 = max(stats.max_best_final_f1, best_final_f1)

        if trace:
            stats.decisions.append(
                DrawDecision(
                    step=step,
                    team=team_name,
                    city=team.city,
                    legal_count=len(legal_groups),
                    feasible_count=feasible_count,
                    safe_count=safe_count,
                    chosen_group=chosen,
                    best_final_f1=best_final_f1,
                )
            )

    return groups, stats


# ============================================================
# 4. Reporting
# ============================================================


def print_groups(groups: List[List[str]]) -> None:
    strengths = [sum(TEAMS[TEAM_INDEX[name]].strength for name in g) for g in groups]
    for j, group in enumerate(groups):
        parts = []
        for name in group:
            team = TEAMS[TEAM_INDEX[name]]
            parts.append(f"{name}({LEVEL_TAG[team.level]})")
        print(f"  {group_label(j)} [strength={strengths[j]:2d}] " + ", ".join(parts))


def print_evaluation(groups: List[List[str]], stats: DrawStats) -> None:
    c1v = check_c1(groups)
    c2v = check_c2(groups)
    c3n, c3d = check_c3(groups)
    print()
    print(
        "  constraints: "
        f"C1={'OK' if not c1v else 'FAIL'}  "
        f"C2={'OK' if not c2v else 'FAIL'}  "
        f"C3={'OK' if c3n == 0 else str(c3n) + ' conflict pairs'}"
    )
    if c3d:
        for g, city, names in c3d:
            print(f"    C3 detail: {group_label(g - 1)} {city} {names}")

    print(
        "  metrics: "
        f"F1={metric_f1(groups)}  "
        f"F2={metric_f2(groups):.4f}  "
        f"F2_range={metric_f2_range(groups)}  "
        f"F3={metric_f3(groups):.4f}"
    )
    print(
        "  look-ahead: "
        f"flow_checks={stats.flow_checks}  "
        f"min_cost_fallback_checks={stats.min_cost_checks}  "
        f"unsafe_candidates={stats.unsafe_candidates}  "
        f"critical_steps={stats.critical_steps}  "
        f"forced_steps={stats.forced_steps}  "
        f"max_best_final_F1={stats.max_best_final_f1}"
    )


def print_trace(stats: DrawStats, limit: int) -> None:
    if limit <= 0:
        return

    print()
    print(f"  first {min(limit, len(stats.decisions))} county draw decisions:")
    print("  step  team        city  legal  feas  safe  chosen  best_final_F1")
    print("  ----  ----------  ----  -----  ----  ----  ------  -------------")
    for d in stats.decisions[:limit]:
        print(
            f"  {d.step:>4}  {d.team:<10}  {d.city:<4}  "
            f"{d.legal_count:>5}  {d.feasible_count:>4}  {d.safe_count:>4}  "
            f"{group_label(d.chosen_group):>6}  {d.best_final_f1:>13}"
        )


def run_sample(seed: int, trace_limit: int) -> None:
    rng = np.random.default_rng(seed)
    groups, stats = draw_lookahead_flow(rng, trace=trace_limit > 0)

    print("=" * 72)
    print(f"Q2 look-ahead min-cost-flow draw, sample seed={seed}")
    print("=" * 72)
    print_groups(groups)
    print_evaluation(groups, stats)
    print_trace(stats, trace_limit)


def summarize(values: Iterable[float]) -> Tuple[float, float, float, float]:
    arr = np.array(list(values), dtype=float)
    return float(arr.mean()), float(np.median(arr)), float(arr.min()), float(arr.max())


def run_monte_carlo(n_sim: int, seed0: int) -> None:
    f1_values: List[int] = []
    f2_values: List[float] = []
    f2r_values: List[int] = []
    f3_values: List[float] = []
    flow_checks: List[int] = []
    min_cost_checks: List[int] = []
    unsafe_candidates: List[int] = []
    critical_steps: List[int] = []
    forced_steps: List[int] = []
    failures = 0

    for k in range(n_sim):
        rng = np.random.default_rng(seed0 + k)
        try:
            groups, stats = draw_lookahead_flow(rng)
        except Exception:
            failures += 1
            continue

        f1_values.append(metric_f1(groups))
        f2_values.append(metric_f2(groups))
        f2r_values.append(metric_f2_range(groups))
        f3_values.append(metric_f3(groups))
        flow_checks.append(stats.flow_checks)
        min_cost_checks.append(stats.min_cost_checks)
        unsafe_candidates.append(stats.unsafe_candidates)
        critical_steps.append(stats.critical_steps)
        forced_steps.append(stats.forced_steps)

    print()
    print("=" * 72)
    print(f"Monte Carlo summary, n={n_sim}, seed0={seed0}, failures={failures}")
    print("=" * 72)

    if not f1_values:
        print("No successful draws.")
        return

    f1 = np.array(f1_values)
    f2 = np.array(f2_values)
    f2r = np.array(f2r_values)
    f3 = np.array(f3_values)
    fc = np.array(flow_checks)
    mc = np.array(min_cost_checks)
    unsafe = np.array(unsafe_candidates)
    critical = np.array(critical_steps)
    fs = np.array(forced_steps)

    print(
        f"  F1 C3 conflict pairs: mean={f1.mean():.4f}, "
        f"P(F1=0)={(f1 == 0).mean():.4f}, max={int(f1.max())}"
    )
    print(
        f"  F2 strength std:      mean={f2.mean():.4f}, "
        f"median={np.median(f2):.4f}, min={f2.min():.4f}, max={f2.max():.4f}"
    )
    print(
        f"  F2 range:             mean={f2r.mean():.2f}, "
        f"min={int(f2r.min())}, max={int(f2r.max())}"
    )
    print(
        f"  F3 diversity entropy: mean={f3.mean():.4f}, "
        f"min={f3.min():.4f}, max={f3.max():.4f}"
    )
    print(
        f"  look-ahead workload:  avg_flow_checks={fc.mean():.1f}, "
        f"avg_min_cost_fallbacks={mc.mean():.1f}, "
        f"avg_forced_steps={fs.mean():.1f}"
    )
    print(
        f"  look-ahead pruning:   avg_unsafe_candidates={unsafe.mean():.1f}, "
        f"unsafe_rate={(unsafe.sum() / fc.sum()):.4f}, "
        f"avg_critical_steps={critical.mean():.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q2 look-ahead draw with min-cost max-flow forward checking."
    )
    parser.add_argument("--seed", type=int, default=42, help="sample draw seed")
    parser.add_argument("--sim", type=int, default=50, help="number of MC runs")
    parser.add_argument("--seed0", type=int, default=1000, help="first MC seed")
    parser.add_argument(
        "--trace",
        type=int,
        default=12,
        help="how many county draw decisions to print for the sample",
    )
    args = parser.parse_args()

    run_sample(args.seed, args.trace)
    if args.sim > 0:
        run_monte_carlo(args.sim, args.seed0)


if __name__ == "__main__":
    main()
