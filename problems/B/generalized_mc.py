#!/usr/bin/env python3
"""
Generalized Monte Carlo for Q2: basic greedy vs look-ahead flow.

The experiment no longer compares repair or restart variants.  It generates
random province topologies and compares two draw models on the same seeds:

1. basic_greedy: municipal teams are placed randomly; county teams are processed
   by descending city size and greedily assigned to the least-conflicting group.
2. lookahead_flow: county teams are drawn in random order; each candidate group
   is accepted only if the remaining draw can still be completed with minimum
   attainable C3 conflict count, certified by max-flow / min-cost flow.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q2_lookahead_flow import Dinic, MinCostMaxFlow


# ============================================================
# 1. Random topology and synthetic team data
# ============================================================


@dataclass
class ProvinceTopology:
    """Synthetic province topology used by generalized Monte Carlo."""

    k: int
    county_per_city: List[int]
    total_teams: int = 64
    num_groups: int = 16
    group_size: int = 4

    @property
    def max_county(self) -> int:
        return max(self.county_per_city) if self.county_per_city else 0

    @property
    def tightness(self) -> float:
        """C3 tightness: max city county-team demand / non-forbidden groups."""
        return self.max_county / (self.num_groups - 1)

    @property
    def gini(self) -> float:
        arr = np.array(sorted(self.county_per_city), dtype=float)
        if len(arr) == 0 or arr.sum() == 0:
            return 0.0
        n = len(arr)
        return float((2 * np.arange(1, n + 1) @ arr) / (n * arr.sum()) - (n + 1) / n)


@dataclass(frozen=True)
class Team:
    id: int
    name: str
    city: int
    level: str
    strength: int


@dataclass
class SyntheticTeams:
    teams: List[Team]
    municipal_by_city: Dict[int, int]
    counties_by_city: Dict[int, List[int]]


def generate_random_topology(
    rng: np.random.Generator,
    k: Optional[int] = None,
    concentration: Optional[str] = None,
) -> ProvinceTopology:
    """
    Generate one synthetic province.

    k is restricted to [5, 16], matching a realistic multi-city province and
    keeping C1/C2 feasible.  County-team counts are not capped at 15; therefore
    some concentrated topologies have tau > 1, where C3=0 is mathematically
    impossible and the min-cost-flow objective becomes necessary.
    """

    if k is None:
        k = int(rng.integers(5, 17))
    k = max(5, min(16, k))

    total_county = 64 - k
    remaining = total_county - k  # one county-level team guaranteed per city

    if concentration is None:
        concentration = str(rng.choice(["uniform", "moderate", "concentrated"]))

    if concentration == "uniform":
        base = total_county // k
        extra = total_county % k
        counts = [base + 1] * extra + [base] * (k - extra)
        rng.shuffle(counts)
        return ProvinceTopology(k=k, county_per_city=counts)

    if concentration == "moderate":
        alpha = np.ones(k) * 2.0
    elif concentration == "concentrated":
        alpha = np.ones(k) * 0.5
        n_big = max(1, k // 3)
        alpha[:n_big] = 3.0
        rng.shuffle(alpha)
    else:
        raise ValueError(f"unknown concentration: {concentration}")

    shares = rng.dirichlet(alpha)
    extra_counts = rng.multinomial(remaining, shares)
    counts = [int(x) + 1 for x in extra_counts]
    return ProvinceTopology(k=k, county_per_city=counts)


def build_teams_from_topology(topo: ProvinceTopology) -> SyntheticTeams:
    teams: List[Team] = []
    municipal_by_city: Dict[int, int] = {}
    counties_by_city: Dict[int, List[int]] = {}

    for city in range(topo.k):
        tid = len(teams)
        municipal_by_city[city] = tid
        teams.append(Team(tid, f"C{city + 1}_M", city, "municipal", 3))

        counties_by_city[city] = []
        n = topo.county_per_city[city]
        n_city_level = n // 2
        for j in range(n):
            tid = len(teams)
            level = "county_city" if j < n_city_level else "county"
            strength = 2 if level == "county_city" else 1
            suffix = "CC" if level == "county_city" else "C"
            teams.append(Team(tid, f"C{city + 1}_{suffix}{j + 1}", city, level, strength))
            counties_by_city[city].append(tid)

    return SyntheticTeams(teams, municipal_by_city, counties_by_city)


# ============================================================
# 2. Metrics
# ============================================================


def compute_f1(groups: List[List[int]], instance: SyntheticTeams) -> int:
    total = 0
    for group in groups:
        by_city: Dict[int, int] = defaultdict(int)
        for tid in group:
            team = instance.teams[tid]
            if team.level != "municipal":
                by_city[team.city] += 1
        total += sum(n * (n - 1) // 2 for n in by_city.values())
    return total


def compute_f2(groups: List[List[int]], instance: SyntheticTeams) -> float:
    strengths = [sum(instance.teams[tid].strength for tid in group) for group in groups]
    return float(np.std(strengths))


def compute_f3(groups: List[List[int]], instance: SyntheticTeams) -> float:
    entropies = []
    for group in groups:
        if not group:
            continue
        by_city: Dict[int, int] = defaultdict(int)
        for tid in group:
            by_city[instance.teams[tid].city] += 1
        total = len(group)
        entropies.append(
            -sum((n / total) * np.log(n / total) for n in by_city.values() if n > 0)
        )
    return float(np.mean(entropies)) if entropies else 0.0


# ============================================================
# 3. Basic greedy draw
# ============================================================


def greedy_on_topology(
    topo: ProvinceTopology,
    instance: SyntheticTeams,
    rng: np.random.Generator,
    repair: bool = False,
) -> Tuple[List[List[int]], bool, bool]:
    """
    Basic greedy draw on a synthetic topology.

    The repair argument is accepted only for backward compatibility with older
    plotting scripts; it is intentionally ignored in the current experiment.
    """

    groups: List[List[int]] = [[] for _ in range(topo.num_groups)]
    mun_group: Dict[int, int] = {}

    cities_for_municipal = list(range(topo.k))
    rng.shuffle(cities_for_municipal)
    slots = np.arange(topo.num_groups)
    rng.shuffle(slots)

    for i, city in enumerate(cities_for_municipal):
        g = int(slots[i])
        groups[g].append(instance.municipal_by_city[city])
        mun_group[city] = g

    city_order = sorted(
        range(topo.k), key=lambda c: topo.county_per_city[c], reverse=True
    )

    for city in city_order:
        forbidden = mun_group[city]
        county_ids = list(instance.counties_by_city[city])
        rng.shuffle(county_ids)

        for tid in county_ids:
            candidates: List[Tuple[bool, int, int]] = []
            for g in range(topo.num_groups):
                if g == forbidden or len(groups[g]) >= topo.group_size:
                    continue
                has_c3 = any(
                    instance.teams[x].city == city
                    and instance.teams[x].level != "municipal"
                    for x in groups[g]
                )
                candidates.append((has_c3, len(groups[g]), g))

            if not candidates:
                return groups, False, False

            candidates.sort()
            best_c3, best_size, _g = candidates[0]
            top = [g for c3, size, g in candidates if c3 == best_c3 and size == best_size]
            groups[int(rng.choice(top))].append(tid)

    return groups, True, False


# ============================================================
# 4. Generic look-ahead flow draw
# ============================================================


@dataclass
class GenericFlowStats:
    flow_checks: int = 0
    min_cost_checks: int = 0
    feasible_candidates: int = 0
    safe_candidates: int = 0
    unsafe_candidates: int = 0
    infeasible_candidates: int = 0
    critical_steps: int = 0
    forced_steps: int = 0
    max_best_final_f1: int = 0


def _remaining_capacity(topo: ProvinceTopology, groups: List[List[int]]) -> List[int]:
    return [topo.group_size - len(group) for group in groups]


def _county_counts_by_city_group(
    topo: ProvinceTopology, instance: SyntheticTeams, groups: List[List[int]]
) -> List[List[int]]:
    counts = [[0] * topo.num_groups for _ in range(topo.k)]
    for g, group in enumerate(groups):
        for tid in group:
            team = instance.teams[tid]
            if team.level != "municipal":
                counts[team.city][g] += 1
    return counts


def _c3_from_counts(counts: List[List[int]]) -> int:
    return sum(n * (n - 1) // 2 for row in counts for n in row)


def _future_zero_c3_feasible(
    topo: ProvinceTopology,
    instance: SyntheticTeams,
    groups: List[List[int]],
    mun_group: Dict[int, int],
    remaining_by_city: List[int],
) -> bool:
    need = sum(remaining_by_city)
    if need == 0:
        return True

    residual = _remaining_capacity(topo, groups)
    if sum(residual) != need:
        return False

    counts = _county_counts_by_city_group(topo, instance, groups)
    if _c3_from_counts(counts) > 0:
        return False

    active = [city for city, n in enumerate(remaining_by_city) if n > 0]
    source = 0
    city_base = 1
    group_base = city_base + len(active)
    sink = group_base + topo.num_groups
    dinic = Dinic(sink + 1)

    for ci, city in enumerate(active):
        city_node = city_base + ci
        dinic.add_edge(source, city_node, remaining_by_city[city])
        for g in range(topo.num_groups):
            if g == mun_group[city] or residual[g] <= 0:
                continue
            if counts[city][g] == 0:
                dinic.add_edge(city_node, group_base + g, 1)

    for g, cap in enumerate(residual):
        dinic.add_edge(group_base + g, sink, cap)

    return dinic.max_flow(source, sink) == need


def _future_min_c3_cost(
    topo: ProvinceTopology,
    instance: SyntheticTeams,
    groups: List[List[int]],
    mun_group: Dict[int, int],
    remaining_by_city: List[int],
) -> Optional[int]:
    need = sum(remaining_by_city)
    if need == 0:
        return 0

    residual = _remaining_capacity(topo, groups)
    if sum(residual) != need:
        return None

    counts = _county_counts_by_city_group(topo, instance, groups)
    active = [city for city, n in enumerate(remaining_by_city) if n > 0]

    source = 0
    city_base = 1
    group_base = city_base + len(active)
    sink = group_base + topo.num_groups
    mcmf = MinCostMaxFlow(sink + 1)

    for ci, city in enumerate(active):
        city_node = city_base + ci
        demand = remaining_by_city[city]
        mcmf.add_edge(source, city_node, demand, 0)
        for g in range(topo.num_groups):
            if g == mun_group[city] or residual[g] <= 0:
                continue
            max_units = min(demand, residual[g])
            base_cost = counts[city][g]
            for unit in range(max_units):
                mcmf.add_edge(city_node, group_base + g, 1, base_cost + unit)

    for g, cap in enumerate(residual):
        mcmf.add_edge(group_base + g, sink, cap, 0)

    sent, cost = mcmf.flow(source, sink, need)
    if sent != need:
        return None
    return cost


def _candidate_final_f1(
    topo: ProvinceTopology,
    instance: SyntheticTeams,
    groups: List[List[int]],
    mun_group: Dict[int, int],
    remaining_by_city: List[int],
    tid: int,
    group_id: int,
) -> Optional[Tuple[int, bool]]:
    team = instance.teams[tid]
    if group_id == mun_group[team.city] or len(groups[group_id]) >= topo.group_size:
        return None

    groups[group_id].append(tid)

    if _future_zero_c3_feasible(topo, instance, groups, mun_group, remaining_by_city):
        groups[group_id].pop()
        return 0, False

    counts = _county_counts_by_city_group(topo, instance, groups)
    current_f1 = _c3_from_counts(counts)
    future_cost = _future_min_c3_cost(
        topo, instance, groups, mun_group, remaining_by_city
    )
    groups[group_id].pop()

    if future_cost is None:
        return None
    return current_f1 + future_cost, True


def lookahead_flow_on_topology(
    topo: ProvinceTopology,
    instance: SyntheticTeams,
    rng: np.random.Generator,
) -> Tuple[List[List[int]], bool, GenericFlowStats]:
    groups: List[List[int]] = [[] for _ in range(topo.num_groups)]
    stats = GenericFlowStats()
    mun_group: Dict[int, int] = {}

    cities_for_municipal = list(range(topo.k))
    rng.shuffle(cities_for_municipal)
    slots = np.arange(topo.num_groups)
    rng.shuffle(slots)
    for i, city in enumerate(cities_for_municipal):
        g = int(slots[i])
        groups[g].append(instance.municipal_by_city[city])
        mun_group[city] = g

    county_pool = [tid for ids in instance.counties_by_city.values() for tid in ids]
    rng.shuffle(county_pool)
    remaining_by_city = list(topo.county_per_city)

    for tid in county_pool:
        city = instance.teams[tid].city
        remaining_by_city[city] -= 1

        legal_groups = [
            g
            for g in range(topo.num_groups)
            if g != mun_group[city] and len(groups[g]) < topo.group_size
        ]

        scores: List[Tuple[int, int]] = []
        for g in legal_groups:
            result = _candidate_final_f1(
                topo, instance, groups, mun_group, remaining_by_city, tid, g
            )
            stats.flow_checks += 1
            if result is None:
                stats.infeasible_candidates += 1
                continue
            final_f1, used_min_cost = result
            if used_min_cost:
                stats.min_cost_checks += 1
            scores.append((g, final_f1))

        if not scores:
            return groups, False, stats

        best_final_f1 = min(score for _g, score in scores)
        safe_groups = [g for g, score in scores if score == best_final_f1]
        unsafe_count = len(legal_groups) - len(safe_groups)

        stats.feasible_candidates += len(scores)
        stats.safe_candidates += len(safe_groups)
        stats.unsafe_candidates += unsafe_count
        if unsafe_count > 0:
            stats.critical_steps += 1
        if len(safe_groups) == 1:
            stats.forced_steps += 1
        stats.max_best_final_f1 = max(stats.max_best_final_f1, best_final_f1)

        chosen = int(rng.choice(safe_groups))
        groups[chosen].append(tid)

    return groups, True, stats


# ============================================================
# 5. Experiment and reporting
# ============================================================


def _bucket_label(tau: float) -> str:
    if tau <= 0.5:
        return "中等 (τ≤0.5)"
    if tau <= 0.7:
        return "偏紧 (0.5<τ≤0.7)"
    if tau <= 1.0:
        return "紧张 (0.7<τ≤1.0)"
    return "超紧 (τ>1.0)"


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run_generalized_mc(
    n_topologies: int = 100,
    n_seeds_per_topo: int = 3,
    seed0: int = 0,
) -> List[dict]:
    print("=" * 88)
    print("  广义蒙特卡洛: 基础贪心 vs 前瞻网络流")
    print(f"  {n_topologies} 个随机拓扑 × {n_seeds_per_topo} 个配对随机种子")
    print("=" * 88)

    results: List[dict] = []

    for ti in range(n_topologies):
        rng_topo = np.random.default_rng(seed0 + ti * 1009)
        topo = generate_random_topology(rng_topo)
        instance = build_teams_from_topology(topo)

        stats = {
            "k": topo.k,
            "max_county": topo.max_county,
            "tightness": topo.tightness,
            "gini": topo.gini,
            "county_per_city": list(topo.county_per_city),
            "greedy_success": 0,
            "greedy_deadlocks": 0,
            "greedy_f1": [],
            "greedy_f2": [],
            "greedy_f3": [],
            "greedy_time": 0.0,
            "flow_success": 0,
            "flow_failures": 0,
            "flow_f1": [],
            "flow_f2": [],
            "flow_f3": [],
            "flow_checks": [],
            "flow_min_cost_checks": [],
            "flow_unsafe_candidates": [],
            "flow_critical_steps": [],
            "flow_forced_steps": [],
            "flow_time": 0.0,
        }

        for si in range(n_seeds_per_topo):
            seed = seed0 + ti * 100000 + si

            rng_greedy = np.random.default_rng(seed)
            t0 = time.perf_counter()
            g_greedy, ok_greedy, _ = greedy_on_topology(topo, instance, rng_greedy)
            stats["greedy_time"] += time.perf_counter() - t0
            if ok_greedy:
                stats["greedy_success"] += 1
                stats["greedy_f1"].append(compute_f1(g_greedy, instance))
                stats["greedy_f2"].append(compute_f2(g_greedy, instance))
                stats["greedy_f3"].append(compute_f3(g_greedy, instance))
            else:
                stats["greedy_deadlocks"] += 1

            rng_flow = np.random.default_rng(seed)
            t0 = time.perf_counter()
            g_flow, ok_flow, flow_stats = lookahead_flow_on_topology(
                topo, instance, rng_flow
            )
            stats["flow_time"] += time.perf_counter() - t0
            if ok_flow:
                stats["flow_success"] += 1
                stats["flow_f1"].append(compute_f1(g_flow, instance))
                stats["flow_f2"].append(compute_f2(g_flow, instance))
                stats["flow_f3"].append(compute_f3(g_flow, instance))
                stats["flow_checks"].append(flow_stats.flow_checks)
                stats["flow_min_cost_checks"].append(flow_stats.min_cost_checks)
                stats["flow_unsafe_candidates"].append(flow_stats.unsafe_candidates)
                stats["flow_critical_steps"].append(flow_stats.critical_steps)
                stats["flow_forced_steps"].append(flow_stats.forced_steps)
            else:
                stats["flow_failures"] += 1

        results.append(stats)
        if (ti + 1) % max(1, n_topologies // 5) == 0:
            print(f"  已完成 {ti + 1}/{n_topologies} 个拓扑...")

    _print_summary(results, n_seeds_per_topo)
    return results


def _print_summary(results: List[dict], n_seeds_per_topo: int) -> None:
    print(f"\n{'=' * 88}")
    print("  分桶结果")
    print(f"{'=' * 88}")

    bucket_order = [
        "中等 (τ≤0.5)",
        "偏紧 (0.5<τ≤0.7)",
        "紧张 (0.7<τ≤1.0)",
        "超紧 (τ>1.0)",
    ]
    buckets: Dict[str, List[dict]] = {label: [] for label in bucket_order}
    for row in results:
        buckets[_bucket_label(row["tightness"])].append(row)

    header = (
        f"  {'紧度区间':<18} {'拓扑数':>6} {'贪心死锁':>10} {'贪心P(F1=0)':>14} "
        f"{'流失败':>8} {'流P(F1=0)':>12} {'流E[F1]':>10} {'流检查/次':>10}"
    )
    print(header)
    print("  " + "-" * 84)

    total_attempts_per_topo = n_seeds_per_topo
    for label in bucket_order:
        bucket = buckets[label]
        if not bucket:
            continue
        n_topo = len(bucket)
        attempts = n_topo * total_attempts_per_topo

        greedy_success = sum(r["greedy_success"] for r in bucket)
        greedy_deadlocks = sum(r["greedy_deadlocks"] for r in bucket)
        greedy_f1 = [f for r in bucket for f in r["greedy_f1"]]
        greedy_deadlock_rate = greedy_deadlocks / attempts
        greedy_c3_zero = (
            float(np.mean([f == 0 for f in greedy_f1])) if greedy_f1 else 0.0
        )

        flow_success = sum(r["flow_success"] for r in bucket)
        flow_failures = sum(r["flow_failures"] for r in bucket)
        flow_f1 = [f for r in bucket for f in r["flow_f1"]]
        flow_checks = [x for r in bucket for x in r["flow_checks"]]
        flow_fail_rate = flow_failures / attempts
        flow_c3_zero = float(np.mean([f == 0 for f in flow_f1])) if flow_f1 else 0.0
        flow_mean_f1 = _mean([float(f) for f in flow_f1])
        flow_mean_checks = _mean([float(x) for x in flow_checks])

        print(
            f"  {label:<18} {n_topo:>6} {greedy_deadlock_rate:>10.1%} "
            f"{greedy_c3_zero:>14.1%} {flow_fail_rate:>8.1%} "
            f"{flow_c3_zero:>12.1%} {flow_mean_f1:>10.2f} {flow_mean_checks:>10.1f}"
        )

    print(f"\n{'=' * 88}")
    print("  全样本汇总")
    print(f"{'=' * 88}")

    attempts = len(results) * n_seeds_per_topo
    greedy_deadlocks = sum(r["greedy_deadlocks"] for r in results)
    greedy_f1 = [f for r in results for f in r["greedy_f1"]]
    flow_failures = sum(r["flow_failures"] for r in results)
    flow_f1 = [f for r in results for f in r["flow_f1"]]
    flow_checks = [x for r in results for x in r["flow_checks"]]
    flow_min_cost = [x for r in results for x in r["flow_min_cost_checks"]]
    flow_unsafe = [x for r in results for x in r["flow_unsafe_candidates"]]
    flow_time = sum(r["flow_time"] for r in results) / attempts
    greedy_time = sum(r["greedy_time"] for r in results) / attempts

    print(f"  基础贪心死锁率: {greedy_deadlocks / attempts:.2%}")
    print(
        f"  基础贪心成功样本 P(F1=0): "
        f"{np.mean([f == 0 for f in greedy_f1]) if greedy_f1 else 0:.2%}"
    )
    print(f"  前瞻网络流失败率: {flow_failures / attempts:.2%}")
    print(
        f"  前瞻网络流 P(F1=0): "
        f"{np.mean([f == 0 for f in flow_f1]) if flow_f1 else 0:.2%}"
    )
    print(f"  前瞻网络流 E[F1]: {_mean([float(f) for f in flow_f1]):.4f}")
    print(f"  前瞻网络流平均检查候选组: {_mean([float(x) for x in flow_checks]):.1f}")
    print(f"  前瞻网络流平均最小费用流回退: {_mean([float(x) for x in flow_min_cost]):.1f}")
    print(f"  前瞻网络流平均危险候选组: {_mean([float(x) for x in flow_unsafe]):.1f}")
    print(f"  基础贪心平均耗时: {greedy_time:.4f}s/次")
    print(f"  前瞻网络流平均耗时: {flow_time:.4f}s/次")

    tightness = np.array([r["tightness"] for r in results])
    greedy_deadlock_by_topo = np.array(
        [r["greedy_deadlocks"] / n_seeds_per_topo for r in results]
    )
    flow_mean_f1_by_topo = np.array(
        [_mean([float(f) for f in r["flow_f1"]]) for r in results]
    )
    if len(results) >= 2:
        corr_deadlock = _corrcoef_safe(tightness, greedy_deadlock_by_topo)
        corr_flow_f1 = _corrcoef_safe(tightness, flow_mean_f1_by_topo)
        print(f"  τ 与贪心死锁率相关系数: {corr_deadlock:.4f}")
        print(f"  τ 与前瞻流E[F1]相关系数: {corr_flow_f1:.4f}")

    zj = ProvinceTopology(k=11, county_per_city=[3, 4, 8, 5, 3, 3, 7, 4, 2, 6, 8])
    tighter = sum(1 for r in results if r["tightness"] > zj.tightness)
    print(
        f"  浙江省参考: k={zj.k}, max(n_i)={zj.max_county}, "
        f"τ={zj.tightness:.3f}; 比浙江更紧的拓扑 {tighter}/{len(results)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-topologies", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed0", type=int, default=0)
    args = parser.parse_args()
    run_generalized_mc(
        n_topologies=args.n_topologies,
        n_seeds_per_topo=args.n_seeds,
        seed0=args.seed0,
    )


if __name__ == "__main__":
    main()
