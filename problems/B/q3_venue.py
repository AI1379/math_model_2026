#!/usr/bin/env python3
"""
Q3: 浙超小组赛比赛地点选择
==========================
从 64 个参赛单位中选 8 个作为小组赛场地, 每个场地承办 2 个小组.
目标: 最小化各队总旅行距离/时间, 同时保证地理覆盖的公平性.

模型: 设施选址 (p-median) + 指派问题联合优化 (ILP).

支持的成本模型:
  A. haversine      — 大圆距离 (km), 基准
  B. road_time       — 公路旅行时间 (min)
  C. railway_time    — 铁路旅行时间 (min)
  D. combined_time   — 公路+铁路平均 (min)
  E. fan_weighted    — 公路时间 × 人口权重 (万人·min)
  F. combined_top20  — 综合时间, 仅从影响力Top20候选场地中选择
"""

import sys
import os
import json
import csv
import numpy as np
from math import radians, sin, cos, asin, sqrt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q1_grouping import (
    TEAMS,
    TEAM_INDEX,
    NUM_GROUPS,
    GROUP_SIZE,
    scheme_d_ilp_balanced,
    scheme_b_serpentine,
    check_c1,
    check_c2,
    check_c3,
)

# ============================================================
# 1. 地理坐标与距离
# ============================================================


def load_coords_from_jsonl(filepath=None):
    """从 JSONL 文件加载坐标, 返回 {name: (lat, lon)}."""
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "conditions_q3", "坐标", "zhejiang_coords_per_line.json",
        )
    coords = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lon, lat = obj["gdm"]
            coords[obj["name"]] = (lat, lon)
    aliases = {"景宁畲族自治县": "景宁县"}
    for source, target in aliases.items():
        if source in coords and target not in coords:
            coords[target] = coords[source]
    return coords


COORDS = load_coords_from_jsonl()


def haversine(name1, name2):
    """计算两个地点之间的大圆距离 (km)."""
    lat1, lon1 = map(radians, COORDS[name1])
    lat2, lon2 = map(radians, COORDS[name2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * asin(sqrt(min(a, 1.0)))


def build_distance_matrix(teams):
    """构建所有球队之间的 Haversine 距离矩阵."""
    n = len(teams)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(teams[i].name, teams[j].name)
            D[i, j] = D[j, i] = d
    return D


# ============================================================
# 2. 外部数据加载 (交通、人口、收入)
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conditions_q3")


def _resolve_name(name):
    """统一名称映射: '景宁县' ↔ '景宁畲族自治县'"""
    if name == "景宁县":
        return "景宁畲族自治县"
    if name == "景宁畲族自治县":
        return "景宁县"
    return name


def _load_travel_time_matrix(filename):
    """加载旅行时间矩阵 (CSV). 返回 (names, matrix)."""
    filepath = os.path.join(DATA_DIR, "交通", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        names = header[1:]
        n = len(names)
        matrix = np.zeros((n, n))
        for i, row in enumerate(reader):
            matrix[i, :] = [float(x) for x in row[1:]]
    return names, matrix


def _load_jsonl(filepath):
    """加载 JSONL 文件, 返回 {name: value_dict}."""
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data[obj["name"]] = obj
    return data


def load_all_data():
    """加载所有外部数据, 建立统一的查找字典."""
    road_names, road_matrix = _load_travel_time_matrix("travel_time_matrix_minutes.csv")
    rail_names, rail_matrix = _load_travel_time_matrix("railway_travel_time_matrix_minutes.csv")

    road_idx = {}
    for i, name in enumerate(road_names):
        road_idx[name] = i
        road_idx[_resolve_name(name)] = i

    rail_idx = {}
    for i, name in enumerate(rail_names):
        rail_idx[name] = i
        rail_idx[_resolve_name(name)] = i

    income = {}
    income_data = _load_jsonl(os.path.join(DATA_DIR, "可支配收入", "可支配收入.json"))
    for name, obj in income_data.items():
        income[name] = obj["可支配收入"]
        income[_resolve_name(name)] = obj["可支配收入"]

    population = {}
    pop_data = _load_jsonl(os.path.join(DATA_DIR, "常住人口", "常住人口（万人）.json"))
    for name, obj in pop_data.items():
        population[name] = obj["常住人口"]
        population[_resolve_name(name)] = obj["常住人口"]

    return {
        "road_names": road_names, "road_matrix": road_matrix, "road_idx": road_idx,
        "rail_names": rail_names, "rail_matrix": rail_matrix, "rail_idx": rail_idx,
        "income": income, "population": population,
    }


def build_influence_scores(teams, data):
    """计算影响力得分 I_k = 0.5 * pop/max + 0.5 * income/max."""
    pop = data["population"]
    inc = data["income"]
    pop_max = max(pop[t.name] for t in teams if t.name in pop)
    inc_max = max(inc[t.name] for t in teams if t.name in inc)
    scores = {}
    for idx, team in enumerate(teams):
        p = pop.get(team.name, 0.0) / pop_max
        q = inc.get(team.name, 0.0) / inc_max
        scores[idx] = 0.5 * p + 0.5 * q
    return scores


def select_influence_candidates(teams, data, top_n=20):
    """按影响力得分筛选前 top_n 个候选场地."""
    scores = build_influence_scores(teams, data)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]


# ============================================================
# 3. 成本矩阵构建
# ============================================================


def build_per_team_cost_matrix(teams, model="haversine", data=None):
    """
    构建 C[i, k] = 球队 i 到场地 k 的单队成本矩阵.
    支持模型: haversine, road_time, railway_time, combined_time, fan_weighted.
    """
    n = len(teams)
    C = np.zeros((n, n))

    if model == "haversine":
        for i in range(n):
            for k in range(n):
                C[i, k] = haversine(teams[i].name, teams[k].name)

    elif model in ("road_time", "railway_time", "combined_time"):
        if data is None:
            raise ValueError("交通模型需要传入 data 参数")
        rmat = data["road_matrix"]
        lmat = data["rail_matrix"]
        ridx = data["road_idx"]
        lidx = data["rail_idx"]
        for i in range(n):
            for k in range(n):
                kn = teams[k].name
                inn = teams[i].name
                if model == "road_time":
                    C[i, k] = rmat[ridx[inn], ridx[kn]]
                elif model == "railway_time":
                    C[i, k] = lmat[lidx[inn], lidx[kn]]
                else:  # combined_time
                    C[i, k] = 0.5 * (
                        rmat[ridx[inn], ridx[kn]] + lmat[lidx[inn], lidx[kn]]
                    )

    elif model == "fan_weighted":
        if data is None:
            raise ValueError("人口加权模型需要传入 data 参数")
        rmat = data["road_matrix"]
        ridx = data["road_idx"]
        pop = data["population"]
        for i in range(n):
            for k in range(n):
                kn = teams[k].name
                inn = teams[i].name
                C[i, k] = rmat[ridx[inn], ridx[kn]] * pop.get(inn, 50.0)

    else:
        raise ValueError(f"未知成本模型: {model}")

    return C


def build_group_cost_matrix(groups, teams, C):
    """从单队成本矩阵汇总得到 D[g, k] = sum_i_in_g C[i, k]."""
    team_idx = {t.name: i for i, t in enumerate(teams)}
    n_groups = len(groups)
    n_locs = len(teams)
    D = np.zeros((n_groups, n_locs))
    for g in range(n_groups):
        for k in range(n_locs):
            D[g, k] = sum(C[team_idx[name], k] for name in groups[g])
    return D, team_idx


# ============================================================
# 4. ILP 选址模型
# ============================================================


def _group_has_home_venue(group, venue_name):
    return venue_name in group


def ilp_venue_selection(groups, teams, D, time_limit=60,
                        candidate_indices=None, forbid_home=True):
    """
    ILP 联合优化选址 + 指派: 选 8 个场地并将 16 个小组分配到场地,
    每个场地恰好承办 2 个小组, 最小化总成本 D[g, k].

    返回: (venues, assignment, total_cost) 或 (None, None, None).
    """
    from pulp import (
        LpMinimize, LpProblem, LpVariable, lpSum,
        LpBinary, LpStatus, value, PULP_CBC_CMD,
    )

    n_loc = len(teams)
    n_groups = len(groups)
    loc_names = [t.name for t in teams]
    candidate_set = set(candidate_indices) if candidate_indices is not None else set(range(n_loc))

    prob = LpProblem("VenueSelection", LpMinimize)

    y = {k: LpVariable(f"y_{k}", cat=LpBinary) for k in range(n_loc)}
    z = {(g, k): LpVariable(f"z_{g}_{k}", cat=LpBinary)
         for g in range(n_groups) for k in range(n_loc)}

    # 恰好选 8 个场地
    prob += lpSum(y[k] for k in range(n_loc)) == 8

    # 候选集约束
    for k in range(n_loc):
        if k not in candidate_set:
            prob += y[k] == 0

    # 每个小组恰好指派到 1 个场地
    for g in range(n_groups):
        prob += lpSum(z[g, k] for k in range(n_loc)) == 1

    # 只能指派到已选场地
    for g in range(n_groups):
        for k in range(n_loc):
            prob += z[g, k] <= y[k]

    # 主场禁止
    if forbid_home:
        for g in range(n_groups):
            for k in range(n_loc):
                if _group_has_home_venue(groups[g], loc_names[k]):
                    prob += z[g, k] == 0

    # 每个场地恰好 2 个小组
    for k in range(n_loc):
        prob += lpSum(z[g, k] for g in range(n_groups)) == 2 * y[k]

    # 目标: 最小化总成本
    prob += lpSum(D[g, k] * z[g, k] for g in range(n_groups) for k in range(n_loc))

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    if LpStatus[prob.status] != "Optimal":
        return None, None, None

    venues = [k for k in range(n_loc) if value(y[k]) > 0.5]
    assignment = {}
    for g in range(n_groups):
        for k in range(n_loc):
            if value(z[g, k]) > 0.5:
                assignment[g] = k
                break

    total_cost = sum(D[g, assignment[g]] for g in range(n_groups))
    return venues, assignment, total_cost


def assignment_ilp(groups, teams, venue_indices, C=None, forbid_home=True):
    """
    给定场地集合, 用 ILP 将 16 个小组最优分配到场地 (每场地恰好 2 组).
    C 为单队成本矩阵 (若不传则用 Haversine 距离).
    返回: (assignment, total_cost).
    """
    from pulp import (
        LpMinimize, LpProblem, LpVariable, lpSum,
        LpBinary, value, PULP_CBC_CMD,
    )

    n_groups = len(groups)
    loc_names = [t.name for t in teams]
    n_venues = len(venue_indices)

    if C is None:
        C = build_per_team_cost_matrix(teams, model="haversine")
    D, _ = build_group_cost_matrix(groups, teams, C)

    prob = LpProblem("AssignOnly", LpMinimize)
    z = {(g, vidx): LpVariable(f"za_{g}_{vidx}", cat=LpBinary)
         for g in range(n_groups) for vidx in range(n_venues)}

    for g in range(n_groups):
        prob += lpSum(z[g, vidx] for vidx in range(n_venues)) == 1
    for vidx in range(n_venues):
        prob += lpSum(z[g, vidx] for g in range(n_groups)) == 2
    if forbid_home:
        for g in range(n_groups):
            for vidx, k in enumerate(venue_indices):
                if _group_has_home_venue(groups[g], loc_names[k]):
                    prob += z[g, vidx] == 0

    prob += lpSum(D[g, k] * z[g, vidx]
                  for g in range(n_groups)
                  for vidx, k in enumerate(venue_indices))

    prob.solve(PULP_CBC_CMD(msg=0))

    assignment = {}
    for g in range(n_groups):
        for vidx, k in enumerate(venue_indices):
            if value(z[g, vidx]) > 0.5:
                assignment[g] = k
                break

    total = sum(D[g, assignment[g]] for g in range(n_groups))
    return assignment, total


def heuristic_venues(teams):
    """
    启发式选址: 地理分散 + 贪心.
    先选总距离最小的点, 然后每次选离已选点集最远且成本合理的点.
    """
    loc_names = [t.name for t in teams]
    cost = {loc: sum(haversine(loc, t.name) for t in teams) for loc in loc_names}
    sorted_locs = sorted(cost.keys(), key=lambda x: cost[x])
    selected = [sorted_locs[0]]

    for _ in range(7):
        best_score = -1
        best_loc = None
        for loc in sorted_locs:
            if loc in selected:
                continue
            min_dist = min(haversine(loc, s) for s in selected)
            score = min_dist / (cost[loc] + 1)
            if score > best_score:
                best_score = score
                best_loc = loc
        selected.append(best_loc)

    return selected


# ============================================================
# 5. 结果输出
# ============================================================


def print_venue_result(groups, teams, venues, assignment, total_cost,
                       C, label, unit):
    """打印单次选址结果."""
    loc_names = [t.name for t in teams]
    team_idx_map = {t.name: i for i, t in enumerate(teams)}

    print(f"\n{'─' * 72}")
    print(f"  {label}")
    print(f"{'─' * 72}")

    venue_groups = defaultdict(list)
    for g, k in assignment.items():
        venue_groups[k].append(g)

    for vi, k in enumerate(sorted(venues)):
        vname = loc_names[k]
        gs = venue_groups.get(k, [])
        lat, lon = COORDS.get(vname, (0, 0))
        print(f"\n  场地{vi + 1}: {vname} ({lat:.2f}°N, {lon:.2f}°E)")
        for g in gs:
            parts = []
            for name in groups[g]:
                ti = team_idx_map.get(name)
                cost = C[ti, k] if ti is not None else 0
                parts.append(f"{name}({cost:.0f}{unit})")
            print(f"    组{g + 1:2d}: {', '.join(parts)}")

    # 统计
    actual_costs = []
    for g, k in assignment.items():
        for name in groups[g]:
            ti = team_idx_map.get(name)
            if ti is not None:
                actual_costs.append(C[ti, k])
    avg_actual = np.mean(actual_costs) if actual_costs else 0
    max_actual = max(actual_costs) if actual_costs else 0
    min_actual = min(actual_costs) if actual_costs else 0

    # Haversine 参考距离
    dists = []
    for g, k in assignment.items():
        for name in groups[g]:
            dists.append(haversine(name, loc_names[k]))

    print(f"\n  总成本: {total_cost:.0f} {unit}")
    print(f"  平均每队: {avg_actual:.1f}{unit}  |  "
          f"最远: {max_actual:.0f}{unit}  |  最近: {min_actual:.0f}{unit}")
    print(f"  Haversine 参考: {sum(dists):.0f} km, "
          f"平均 {np.mean(dists):.1f} km, 最远 {max(dists):.0f} km")

    # 地理覆盖
    venue_coords = [COORDS.get(loc_names[k], (0, 0)) for k in venues]
    lats = [c[0] for c in venue_coords]
    lons = [c[1] for c in venue_coords]
    print(f"  纬度范围: {min(lats):.2f}° ~ {max(lats):.2f}°  "
          f"经度范围: {min(lons):.2f}° ~ {max(lons):.2f}°")


def print_comparison_table(results, ref_label="haversine"):
    """打印所有成本模型的横向对比表."""
    print(f"\n{'=' * 72}")
    print(f"  成本模型横向对比")
    print(f"{'=' * 72}")

    header = (f"  {'模型':<24} {'总成本':<14} {'单位':<8} "
              f"{'平均/队':<14} {'最远/队':<12} {'Haversine参考':<14} {'场地交集'}")
    print(header)
    print(f"  {'─' * 86}")

    ref_venues = None
    for r in results:
        if r["label"] == ref_label:
            ref_venues = set(r.get("venues", []))
            break

    for r in results:
        name = r["label"]
        unit = r["unit"]
        tc = r["total_cost"]
        avg = tc / 64
        mc = r["max_per_team"]
        hr = r.get("haversine_ref", 0)

        jaccard = ""
        if ref_venues is not None and r.get("venues"):
            common = len(ref_venues & set(r["venues"]))
            jaccard = f"{common}/8"

        print(f"  {name:<24} {tc:<14.0f} {unit:<8} "
              f"{avg:<14.1f} {mc:<12.0f} {hr:<14.0f} {jaccard}")


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 72)
    print("  浙超小组赛选址 — 问题3: 比赛地点选择")
    print("=" * 72)

    # ---- 分组 ----
    groups = scheme_d_ilp_balanced(w_c3=10, w_str=1)
    if groups is None:
        groups = scheme_b_serpentine(seed=42)
        print("\n  使用蛇形分组 (ILP均衡不可用)")
    else:
        print("\n  使用 ILP均衡分组")

    c1v, c2v = check_c1(groups), check_c2(groups)
    c3n, _ = check_c3(groups)
    print(f"  C1={'通过' if not c1v else '违反'} "
          f"C2={'通过' if not c2v else '违反'} "
          f"C3={'通过' if c3n == 0 else f'{c3n}对冲突'}")

    # ---- 加载外部数据 ----
    print("\n  加载外部数据...")
    data = load_all_data()
    print(f"    公路旅行矩阵: {data['road_matrix'].shape}")
    print(f"    铁路旅行矩阵: {data['rail_matrix'].shape}")
    print(f"    收入数据:     {len(data['income'])} 条")
    print(f"    人口数据:     {len(data['population'])} 条")

    influence_top20 = select_influence_candidates(TEAMS, data, top_n=20)
    loc_names = [t.name for t in TEAMS]
    print("\n  影响力Top20候选场地 (人口+收入各50%):")
    print("    " + "、".join(loc_names[i] for i in influence_top20))

    # ---- 实验配置 ----
    models = [
        ("haversine", "km"), ("road_time", "min"), ("railway_time", "min"),
        ("combined_time", "min"), ("fan_weighted", "万人·min"),
    ]
    experiments = [
        {"model": m, "unit": u, "label": m, "candidate_indices": None}
        for m, u in models
    ]
    experiments.append({
        "model": "combined_time", "unit": "min", "label": "combined_top20",
        "candidate_indices": influence_top20,
    })

    # ---- 逐模型求解 ----
    results = []
    for exp in experiments:
        model_name = exp["model"]
        unit = exp["unit"]
        label = exp["label"]
        candidate_indices = exp["candidate_indices"]

        print(f"\n  构建成本矩阵: {label} ...")
        C = build_per_team_cost_matrix(TEAMS, model=model_name, data=data)
        D, team_idx_map = build_group_cost_matrix(groups, TEAMS, C)
        print(f"    范围: [{D.min():.0f}, {D.max():.0f}] {unit}")

        print(f"    求解 ILP ...")
        time_limit = 120 if model_name == "fan_weighted" else 60
        venues, assignment, total_cost = ilp_venue_selection(
            groups, TEAMS, D, time_limit=time_limit,
            candidate_indices=candidate_indices, forbid_home=True,
        )

        if venues is None:
            print(f"    [警告] {label} ILP 未找到最优解")
            continue

        hav_dist = sum(
            haversine(name, loc_names[assignment[g]])
            for g in range(NUM_GROUPS) for name in groups[g]
        )
        max_per_team = max(
            C[team_idx_map[name], assignment[g]]
            for g in range(NUM_GROUPS) for name in groups[g]
        )

        results.append({
            "model": model_name, "label": label, "unit": unit,
            "total_cost": total_cost, "max_per_team": max_per_team,
            "haversine_ref": hav_dist, "venues": venues, "assignment": assignment,
        })

        print_venue_result(groups, TEAMS, venues, assignment, total_cost,
                           C, label, unit)

    # ---- 横向对比 ----
    print_comparison_table(results)

    # ---- 选址一致性 ----
    if len(results) >= 2:
        print(f"\n{'=' * 72}")
        print(f"  选址一致性分析")
        print(f"{'=' * 72}")
        all_venue_sets = [set(r.get("venues", [])) for r in results]

        print(f"\n  各模型选出的 8 个场地间的交集:")
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                common = len(all_venue_sets[i] & all_venue_sets[j])
                print(f"    {results[i]['label']:>16} ∩ {results[j]['label']:<16} = {common}/8")

        venue_freq = defaultdict(int)
        for vs in all_venue_sets:
            for v in vs:
                venue_freq[v] += 1
        popular = sorted(venue_freq.items(), key=lambda x: -x[1])
        print(f"\n  高频场地 (被多个模型选中):")
        for v, f in popular:
            if f >= 2:
                print(f"    {loc_names[v]:10s}  被 {f}/{len(results)} 个模型选中")

    # ---- 结论 ----
    if results:
        print(f"\n{'=' * 72}")
        print(f"  结论")
        print(f"{'=' * 72}")
        base = results[0]
        for r in results[1:]:
            print(f"\n  {r['label']} vs haversine (基准):")
            print(f"    总成本: {r['total_cost']:.0f} {r['unit']} "
                  f"(Haversine距离参考: {r['haversine_ref']:.0f} km "
                  f"vs 基准 {base['haversine_ref']:.0f} km)")
            if r["haversine_ref"] > 0:
                pct = ((r["haversine_ref"] - base["haversine_ref"])
                       / base["haversine_ref"] * 100)
                print(f"    用此模型选出的场地, 在Haversine距离下多走了 {pct:+.1f}% 的距离")

    # ---- 启发式选址对比 ----
    print(f"\n{'=' * 72}")
    print(f"  ILP 最优 vs 启发式贪心 对比")
    print(f"{'=' * 72}")

    h_locs = heuristic_venues(TEAMS)
    h_venue_indices = [loc_names.index(l) for l in h_locs]
    C_hav = build_per_team_cost_matrix(TEAMS, model="haversine")
    h_assign, h_cost = assignment_ilp(groups, TEAMS, h_venue_indices, C=C_hav)
    print_venue_result(groups, TEAMS, h_venue_indices, h_assign, h_cost,
                       C_hav, "启发式选址 (地理分散贪心 + 最优指派)", "km")

    if results:
        ilp_km = results[0]["total_cost"]
        gap = (h_cost - ilp_km) / ilp_km * 100
        print(f"\n  启发式比 ILP 最优多 {gap:.1f}% 总距离, 说明联合优化效果显著.")


if __name__ == "__main__":
    main()
