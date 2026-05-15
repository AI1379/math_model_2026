#!/usr/bin/env python3
"""
Q3 比赛地点选择 v2 — 多维度成本模型对比
========================================

在原始 Haversine 距离基础上，引入真实交通数据、人口和经济指标。
支持 5 种成本模型，用 ILP 联合优化求解并横向对比。

成本模型:
  A. haversine      — 原始大圆距离 (km), 基准
  B. road_time       — 公路旅行时间 (min)
  C. railway_time    — 铁路旅行时间 (min)
  D. combined_time   — 公路+铁路平均 (min)
  E. fan_weighted    — 公路时间 × 人口权重 (万人·min) , 最小化"球迷旅行"

新数据来源 (conditions_q3.zip):
  - 交通/  — 64×64 公路 & 铁路旅行时间矩阵
  - 可支配收入/ — 各区县人均可支配收入
  - 常住人口/ — 各区县常住人口 (万人)
"""

import sys
import os
import json
import csv
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q1_grouping import (
    TEAMS, TEAM_INDEX, NUM_GROUPS, GROUP_SIZE,
    scheme_d_ilp_balanced, scheme_b_serpentine,
    check_c1, check_c2, check_c3,
)
from q3_venue import haversine, COORDS


# ============================================================
# 1. 数据加载
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conditions_q3")


def _resolve_name(name):
    """统一名称映射: '景宁县' ↔ '景宁畲族自治县'"""
    if name == "景宁县":
        return "景宁畲族自治县"
    if name == "景宁畲族自治县":
        return "景宁县"
    return name


def load_travel_time_matrix(filename):
    """
    加载旅行时间矩阵 (CSV).
    返回: (names, matrix) 其中 names 是 64 个地点名, matrix 是 64×64 数组.
    """
    filepath = os.path.join(DATA_DIR, "交通", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        names = header[1:]  # 跳过第一个空元素
        n = len(names)
        matrix = np.zeros((n, n))
        for i, row in enumerate(reader):
            vals = [float(x) for x in row[1:]]
            matrix[i, :] = vals
    return names, matrix


def load_jsonl(filepath):
    """加载 JSONL 文件, 返回 {name: value_dict}"""
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
    # 旅行时间矩阵
    road_names, road_matrix = load_travel_time_matrix("travel_time_matrix_minutes.csv")
    rail_names, rail_matrix = load_travel_time_matrix("railway_travel_time_matrix_minutes.csv")

    # 建立名称→索引映射 (处理景宁畲族自治县)
    road_idx = {}
    for i, name in enumerate(road_names):
        road_idx[name] = i
        road_idx[_resolve_name(name)] = i  # 双向映射

    rail_idx = {}
    for i, name in enumerate(rail_names):
        rail_idx[name] = i
        rail_idx[_resolve_name(name)] = i

    # 收入数据
    income_path = os.path.join(DATA_DIR, "可支配收入", "可支配收入.json")
    income_data = load_jsonl(income_path)
    income = {}
    for name, obj in income_data.items():
        income[name] = obj["可支配收入"]
        income[_resolve_name(name)] = obj["可支配收入"]

    # 人口数据
    pop_path = os.path.join(DATA_DIR, "常住人口", "常住人口（万人）.json")
    pop_data = load_jsonl(pop_path)
    population = {}
    for name, obj in pop_data.items():
        population[name] = obj["常住人口"]
        population[_resolve_name(name)] = obj["常住人口"]

    return {
        "road_names": road_names,
        "road_matrix": road_matrix,
        "road_idx": road_idx,
        "rail_names": rail_names,
        "rail_matrix": rail_matrix,
        "rail_idx": rail_idx,
        "income": income,
        "population": population,
    }


# ============================================================
# 2. 成本矩阵构建
# ============================================================

def build_per_team_cost_matrix(teams, model="haversine", data=None):
    """
    构建 C[i, k] = 队 i 到场地 k 的单独成本 (不对组内求和).

    返回: (n_teams × n_locs) 矩阵.
    """
    n_teams = len(teams)
    n_locs = len(teams)
    C = np.zeros((n_teams, n_locs))

    if model == "haversine":
        for i in range(n_teams):
            for k in range(n_locs):
                C[i, k] = haversine(teams[i].name, teams[k].name)

    elif model in ("road_time", "railway_time", "combined_time"):
        if data is None:
            raise ValueError("需要传入 data 参数以使用旅行时间矩阵")

        if model == "road_time":
            matrix = data["road_matrix"]
            idx_map = data["road_idx"]
        elif model == "railway_time":
            matrix = data["rail_matrix"]
            idx_map = data["rail_idx"]

        for i in range(n_teams):
            for k in range(n_locs):
                k_name = teams[k].name
                i_name = teams[i].name
                if model == "combined_time":
                    ki_r = data["road_idx"][k_name]
                    ki_l = data["rail_idx"][k_name]
                    ii_r = data["road_idx"][i_name]
                    ii_l = data["rail_idx"][i_name]
                    C[i, k] = 0.5 * (data["road_matrix"][ii_r, ki_r] +
                                     data["rail_matrix"][ii_l, ki_l])
                else:
                    ki = idx_map[k_name]
                    ii = idx_map[name := teams[i].name]
                    C[i, k] = matrix[ii, ki]

    elif model == "fan_weighted":
        if data is None:
            raise ValueError("需要传入 data 参数")
        matrix = data["road_matrix"]
        idx_map = data["road_idx"]
        pop = data["population"]

        for i in range(n_teams):
            for k in range(n_locs):
                k_name = teams[k].name
                i_name = teams[i].name
                ki = idx_map[k_name]
                ii = idx_map[i_name]
                C[i, k] = matrix[ii, ki] * pop.get(i_name, 50.0)

    else:
        raise ValueError(f"未知成本模型: {model}")

    return C


def build_cost_matrix(groups, teams, model="haversine", data=None):
    """
    构建 D[g, k] = 组 g 中所有队到场地 k 的总成本.

    model:
      'haversine'     — 大圆距离 (km)
      'road_time'     — 公路旅行时间 (min)
      'railway_time'  — 铁路旅行时间 (min)
      'combined_time' — 0.5*(road + railway) (min)
      'fan_weighted'  — road_time × 源城市人口 (万人·min)
    """
    n_groups = len(groups)
    n_locs = len(teams)
    D = np.zeros((n_groups, n_locs))

    if model == "haversine":
        for g in range(n_groups):
            for k in range(n_locs):
                D[g, k] = sum(haversine(name, teams[k].name) for name in groups[g])

    elif model in ("road_time", "railway_time", "combined_time"):
        if data is None:
            raise ValueError("需要传入 data 参数以使用旅行时间矩阵")

        if model == "road_time":
            matrix = data["road_matrix"]
            idx_map = data["road_idx"]
        elif model == "railway_time":
            matrix = data["rail_matrix"]
            idx_map = data["rail_idx"]
        else:  # combined
            matrix_r = data["road_matrix"]
            matrix_l = data["rail_matrix"]
            idx_r = data["road_idx"]
            idx_l = data["rail_idx"]

        for g in range(n_groups):
            for k in range(n_locs):
                k_name = teams[k].name
                if model == "road_time":
                    ki = idx_map[k_name]
                    D[g, k] = sum(matrix[idx_map[name], ki] for name in groups[g])
                elif model == "railway_time":
                    ki = idx_map[k_name]
                    D[g, k] = sum(matrix[idx_map[name], ki] for name in groups[g])
                else:  # combined
                    ki_r = idx_r[k_name]
                    ki_l = idx_l[k_name]
                    total = 0.0
                    for name in groups[g]:
                        r = matrix_r[idx_r[name], ki_r]
                        l = matrix_l[idx_l[name], ki_l]
                        total += 0.5 * (r + l)
                    D[g, k] = total

    elif model == "fan_weighted":
        if data is None:
            raise ValueError("需要传入 data 参数")
        matrix = data["road_matrix"]
        idx_map = data["road_idx"]
        pop = data["population"]

        for g in range(n_groups):
            for k in range(n_locs):
                k_name = teams[k].name
                ki = idx_map[k_name]
                total = 0.0
                for name in groups[g]:
                    t = matrix[idx_map[name], ki]
                    w = pop.get(name, 50.0)  # 默认50万人
                    total += t * w
                D[g, k] = total

    else:
        raise ValueError(f"未知成本模型: {model}")

    return D


# ============================================================
# 3. ILP 选址 + 指派 (复用 q3_venue 的 ILP)
# ============================================================

def ilp_venue_selection(groups, teams, D, time_limit=60):
    """
    ILP 联合优化选址 + 指派.  最小化给定成本矩阵 D.

    返回: (venues, assignment, total_cost), 或 (None, None, None)
    """
    from pulp import (
        LpMinimize, LpProblem, LpVariable, lpSum,
        LpBinary, LpStatus, value, PULP_CBC_CMD,
    )

    n_loc = len(teams)
    n_groups = len(groups)

    prob = LpProblem("VenueSelection_v2", LpMinimize)

    y = {k: LpVariable(f"y_{k}", cat=LpBinary) for k in range(n_loc)}
    z = {(g, k): LpVariable(f"z_{g}_{k}", cat=LpBinary)
         for g in range(n_groups) for k in range(n_loc)}

    prob += lpSum(y[k] for k in range(n_loc)) == 8

    for g in range(n_groups):
        prob += lpSum(z[g, k] for k in range(n_loc)) == 1

    for g in range(n_groups):
        for k in range(n_loc):
            prob += z[g, k] <= y[k]

    for k in range(n_loc):
        prob += lpSum(z[g, k] for g in range(n_groups)) == 2 * y[k]

    total = []
    for g in range(n_groups):
        for k in range(n_loc):
            total.append(D[g, k] * z[g, k])

    prob += lpSum(total)
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


# ============================================================
# 4. 结果输出
# ============================================================

def print_venue_result(groups, teams, venues, assignment, total_cost, D, label, unit,
                       per_team_cost=None):
    """打印选址结果. per_team_cost 是 64×64 的每队到每场地的成本矩阵 (用于显示)."""
    loc_names = [t.name for t in teams]
    print(f"\n{'─' * 72}")
    print(f"  {label}")
    print(f"{'─' * 72}")

    venue_groups = defaultdict(list)
    for g, k in assignment.items():
        venue_groups[k].append(g)

    # 构建 team→idx 映射用于 per_team_cost
    if per_team_cost is not None:
        team_idx_map = {}
        for ti, t in enumerate(teams):
            team_idx_map[t.name] = ti
    else:
        team_idx_map = None

    for vi, k in enumerate(sorted(venues)):
        vname = loc_names[k]
        gs = venue_groups.get(k, [])
        lat, lon = COORDS.get(vname, (0, 0))
        print(f"\n  场地{vi+1}: {vname} ({lat:.2f}°N, {lon:.2f}°E)")
        for g in gs:
            parts = []
            for name in groups[g]:
                if per_team_cost is not None and team_idx_map is not None:
                    ti = team_idx_map.get(name)
                    if ti is not None:
                        cost = per_team_cost[ti, k]
                    else:
                        cost = D[g, k] / 4
                else:
                    cost = D[g, k] / 4
                parts.append(f"{name}({cost:.0f}{unit})")
            print(f"    组{g+1:2d}: {', '.join(parts)}")

    # 统计 (用原始 haversine 距离做跨模型比较)
    dists = []
    for g, k in assignment.items():
        vname = loc_names[k]
        for name in groups[g]:
            dists.append(haversine(name, vname))

    # 每队实际成本分布
    if per_team_cost is not None and team_idx_map is not None:
        actual_costs = []
        for g, k in assignment.items():
            for name in groups[g]:
                ti = team_idx_map.get(name)
                if ti is not None:
                    actual_costs.append(per_team_cost[ti, k])
        max_actual = max(actual_costs) if actual_costs else 0
        min_actual = min(actual_costs) if actual_costs else 0
        avg_actual = np.mean(actual_costs) if actual_costs else 0
    else:
        max_actual = max(D[g, k] / 4 for g, k in assignment.items())
        min_actual = min(D[g, k] / 4 for g, k in assignment.items())
        avg_actual = total_cost / 64

    print(f"\n  总成本: {total_cost:.0f} {unit}")
    print(f"  平均每队: {avg_actual:.1f}{unit}  |  "
          f"最远: {max_actual:.0f}{unit}  |  最近: {min_actual:.0f}{unit}")
    print(f"  Haversine 参考: {sum(dists):.0f} km, "
          f"平均 {np.mean(dists):.1f} km, 最远 {max(dists):.0f} km")


def print_comparison_table(results, ilp_haversine_cost=None):
    """打印所有成本模型的横向对比表."""
    print(f"\n{'=' * 72}")
    print(f"  成本模型横向对比")
    print(f"{'=' * 72}")

    header = (f"  {'模型':<20} {'总成本':<14} {'单位':<8} "
              f"{'平均/队':<14} {'最远/队':<12} {'Haversine':<14}")
    print(header)
    print(f"  {'─' * 80}")

    ref_venues = None
    if ilp_haversine_cost is not None and results:
        ref_venues = results[0].get("venues")

    for r in results:
        name = r["model"]
        unit = r["unit"]
        tc = r["total_cost"]
        avg = tc / 64
        max_cost = r["max_per_team"]
        hav_ref = r.get("haversine_ref", 0)

        jaccard = ""
        if ref_venues is not None and r.get("venues"):
            common = len(set(ref_venues) & set(r["venues"]))
            jaccard = f" J={common}/8"

        print(f"  {name:<20} {tc:<14.0f} {unit:<8} "
              f"{avg:<14.1f} {max_cost:<12.0f} {hav_ref:<14.0f}{jaccard}")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("=" * 72)
    print("  浙超小组赛选址 v2 — 多维度成本模型对比")
    print("=" * 72)

    # 加载数据
    print("\n  加载外部数据...")
    data = load_all_data()
    print(f"    公路旅行矩阵: {data['road_matrix'].shape}")
    print(f"    铁路旅行矩阵: {data['rail_matrix'].shape}")
    print(f"    收入数据:     {len(data['income'])} 条")
    print(f"    人口数据:     {len(data['population'])} 条")

    # 分组
    print("\n  生成分组方案...")
    groups = scheme_d_ilp_balanced(w_c3=10, w_str=1)
    if groups is None:
        groups = scheme_b_serpentine(seed=42)
        print("    使用蛇形分组 (ILP均衡不可用)")
    else:
        print("    使用 ILP均衡分组")

    c1v, c2v = check_c1(groups), check_c2(groups)
    c3n, _ = check_c3(groups)
    print(f"    C1={'✓' if not c1v else '✗'} "
          f"C2={'✓' if not c2v else '✗'} "
          f"C3={'✓' if c3n==0 else f'{c3n}对冲突'}")

    # 成本模型列表
    models = [
        ("haversine", "km"),
        ("road_time", "min"),
        ("railway_time", "min"),
        ("combined_time", "min"),
        ("fan_weighted", "万人·min"),
    ]

    results = []
    for model_name, unit in models:
        print(f"\n  构建成本矩阵: {model_name} ...")
        C = build_per_team_cost_matrix(TEAMS, model=model_name, data=data)
        # 从 per-team 矩阵汇总得到组成本矩阵 D[g,k] = sum_i_in_g C[i,k]
        D = np.zeros((len(groups), len(TEAMS)))
        team_idx = {t.name: i for i, t in enumerate(TEAMS)}
        for g in range(len(groups)):
            for k in range(len(TEAMS)):
                D[g, k] = sum(C[team_idx[name], k] for name in groups[g])
        print(f"    范围: [{D.min():.0f}, {D.max():.0f}] {unit}")

        print(f"    求解 ILP ...")
        venues, assignment, total_cost = ilp_venue_selection(
            groups, TEAMS, D, time_limit=120 if model_name == "fan_weighted" else 60)

        if venues is None:
            print(f"    [警告] {model_name} ILP 未找到最优解")
            continue

        # 计算此解在Haversine距离下的表现
        hav_dist = 0
        loc_names = [t.name for t in TEAMS]
        for g, k in assignment.items():
            vname = loc_names[k]
            for name in groups[g]:
                hav_dist += haversine(name, vname)

        # 每队最大成本 (按实际 per-team 成本)
        max_per_team = 0
        for g, k in assignment.items():
            for name in groups[g]:
                ti = team_idx.get(name)
                if ti is not None:
                    max_per_team = max(max_per_team, C[ti, k])

        results.append({
            "model": model_name,
            "unit": unit,
            "total_cost": total_cost,
            "avg_per_team": total_cost / 64,
            "max_per_team": max_per_team,
            "haversine_ref": hav_dist,
            "venues": venues,
            "assignment": assignment,
        })

        print_venue_result(groups, TEAMS, venues, assignment,
                           total_cost, D, model_name, unit,
                           per_team_cost=C)

    # 横向对比
    print_comparison_table(results)

    # 选址一致性分析
    print(f"\n{'=' * 72}")
    print(f"  选址一致性分析")
    print(f"{'=' * 72}")

    if len(results) >= 2:
        all_venue_sets = []
        for r in results:
            all_venue_sets.append(set(r.get("venues", [])))

        print(f"\n  各模型选出的8个场地间的交集:")
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                common = len(all_venue_sets[i] & all_venue_sets[j])
                print(f"    {results[i]['model']:>16} ∩ {results[j]['model']:<16} = {common}/8")

        # 综合频次
        venue_freq = defaultdict(int)
        for vs in all_venue_sets:
            for v in vs:
                venue_freq[v] += 1

        loc_names = [t.name for t in TEAMS]
        popular = sorted(venue_freq.items(), key=lambda x: -x[1])
        print(f"\n  高频场地 (被多个模型选中):")
        for v, f in popular:
            if f >= 2:
                print(f"    {loc_names[v]:10s}  被 {f}/{len(results)} 个模型选中")

    # 结论
    print(f"\n{'=' * 72}")
    print(f"  结论")
    print(f"{'=' * 72}")

    if results:
        base = results[0]  # haversine
        for r in results[1:]:
            print(f"\n  {r['model']} vs haversine (基准):")
            print(f"    总成本: {r['total_cost']:.0f} {r['unit']} "
                  f"(Haversine距离参考: {r['haversine_ref']:.0f} km vs 基准 {base['haversine_ref']:.0f} km)")

            if r['haversine_ref'] > 0:
                pct = (r['haversine_ref'] - base['haversine_ref']) / base['haversine_ref'] * 100
                print(f"    用此模型选出的场地, 在Haversine距离下多走了 {pct:+.1f}% 的距离")

    print(f"\n  核心发现:")
    print(f"    1. 不同的成本模型可能选出不同的场地集合")
    print(f"    2. 旅行时间模型比纯地理距离更贴近实际交通网络")
    print(f"    3. fan_weighted 模型同时考虑旅行成本与人口规模")
    print(f"    4. 建议赛事组织者综合考虑交通可达性与区域经济水平")


if __name__ == "__main__":
    main()
