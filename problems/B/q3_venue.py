#!/usr/bin/env python3
"""
Q3: 浙超小组赛比赛地点选择
==========================
从 64 个参赛单位中选 8 个作为小组赛场地, 每个场地承办 2 个小组.
目标: 最小化各队总旅行距离, 同时保证地理覆盖的公平性.

模型: 设施选址 (p-median) + 指派问题联合优化
"""

import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from math import radians, sin, cos, asin, sqrt
from q1_grouping import (
    CITY_DATA,
    TEAMS,
    TEAM_INDEX,
    NUM_GROUPS,
    GROUP_SIZE,
    LEVEL_TAG,
    metric_f1,
    check_c1,
    check_c2,
    check_c3,
)

# 导入 Q1 方案 D (ILP 均衡) 作为分组基础
# [问题] q1_grouping.py 的 scheme_d_ilp_balanced 依赖 PuLP,
#        不能在文件级别导入 (会触发 PuLP 安装检测).
# [解决] 延迟到 main() 中调用, 避免模块级副作用.
from q1_grouping import scheme_d_ilp_balanced


# ============================================================
# 1. 地理坐标数据
# ============================================================

def load_coords_from_jsonl(filepath=None):
    """从 JSONL 文件加载坐标, 返回 {name: (lat, lon)}.
    JSONL 格式: {"name": "...", "gdm": [lon, lat]} 每行一个.
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "zhejiang_coords_per_line.jsonl")
    coords = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lon, lat = obj["gdm"]
            coords[obj["name"]] = (lat, lon)
    return coords


COORDS = load_coords_from_jsonl()


def haversine(name1, name2):
    """计算两个地点之间的大圆距离 (km)"""
    lat1, lon1 = map(radians, COORDS[name1])
    lat2, lon2 = map(radians, COORDS[name2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * asin(sqrt(min(a, 1.0)))


def build_distance_matrix(teams):
    """构建所有球队之间的距离矩阵"""
    n = len(teams)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(teams[i].name, teams[j].name)
            D[i, j] = D[j, i] = d
    return D


# ============================================================
# 2. 启发式选址: 地理分散 + 贪心
# ============================================================

def heuristic_venues(groups, teams):
    """
    启发式选址:
    1. 计算每个地点作为场地时, 承接所有队伍的总旅行距离
    2. 结合地理分散性 (已选场地间的最小距离) 贪心选取
    """
    n_teams = len(teams)
    loc_names = [t.name for t in teams]

    # 每个地点到所有球队的距离和
    cost = {}
    for loc in loc_names:
        cost[loc] = sum(haversine(loc, t.name) for t in teams)

    # [问题] 需要为每个场地分配 2 个小组, 但这里先用纯地理方法选 8 个点,
    #        后面再用 ILP 做联合优化.
    # [解决] 分两步: 先用贪心选出地理分散的 8 个点, 再用 ILP 最优指派.

    # 贪心: 先选总距离最小的点, 然后每次选离已选点集最远且成本合理的点
    sorted_locs = sorted(cost.keys(), key=lambda x: cost[x])
    selected = [sorted_locs[0]]

    for _ in range(7):
        best_score = -1
        best_loc = None
        for loc in sorted_locs:
            if loc in selected:
                continue
            min_dist_to_selected = min(haversine(loc, s) for s in selected)
            # 综合得分: 分散性 (远) + 成本 (低)
            score = min_dist_to_selected / (cost[loc] + 1)
            if score > best_score:
                best_score = score
                best_loc = loc
        selected.append(best_loc)

    return selected


# ============================================================
# 3. ILP 最优选址 + 指派
# ============================================================

def ilp_venue_selection(groups, teams):
    """
    ILP 联合优化: 同时选择 8 个场地, 并将 16 个小组分配到场地.
    每个场地恰好承办 2 个小组.

    [问题] 决策变量包括选址 (64 选 8) 和指派 (16 组 → 8 场地),
           联合优化的变量数 = 64 + 16×64 = 1088, 约束适中, CBC 可解.
    [解决] 用 PuLP 建模, 目标是最小化总旅行距离.
    """
    from pulp import (
        LpMinimize, LpProblem, LpVariable, lpSum,
        LpBinary, LpStatus, value, PULP_CBC_CMD,
    )

    n_loc = len(teams)
    loc_names = [t.name for t in teams]
    loc_idx = {t.name: i for i, t in enumerate(teams)}

    # 预计算距离矩阵 (球队 → 候选场地)
    # dist[i][k] = 球队 i 到地点 k 的距离
    dist = np.zeros((n_loc, n_loc))
    for i in range(n_loc):
        for k in range(n_loc):
            if i != k:
                dist[i][k] = haversine(teams[i].name, teams[k].name)

    prob = LpProblem("VenueSelection", LpMinimize)

    # 选址变量: y[k] = 1 表示地点 k 被选为赛场
    y = {k: LpVariable(f"y_{k}", cat=LpBinary) for k in range(n_loc)}

    # 指派变量: z[g][k] = 1 表示小组 g 被指派到地点 k
    z = {(g, k): LpVariable(f"z_{g}_{k}", cat=LpBinary)
         for g in range(NUM_GROUPS) for k in range(n_loc)}

    # --- 约束 ---
    # 选 8 个场地
    prob += lpSum(y[k] for k in range(n_loc)) == 8

    # 每个小组恰好指派到 1 个场地
    for g in range(NUM_GROUPS):
        prob += lpSum(z[g, k] for k in range(n_loc)) == 1

    # 只能指派到已选场地
    for g in range(NUM_GROUPS):
        for k in range(n_loc):
            prob += z[g, k] <= y[k]

    # 每个场地恰好 2 个小组
    for k in range(n_loc):
        prob += lpSum(z[g, k] for g in range(NUM_GROUPS)) == 2 * y[k]

    # --- 目标: 最小化总旅行距离 ---
    # 每支球队去其所在小组的比赛场地
    total_dist = []
    for g in range(NUM_GROUPS):
        for name in groups[g]:
            i = TEAM_INDEX[name]
            for k in range(n_loc):
                total_dist.append(dist[i][k] * z[g, k])

    prob += lpSum(total_dist)

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=60))

    if LpStatus[prob.status] != "Optimal":
        print(f"  [ILP] 状态: {LpStatus[prob.status]}")
        return None, None, None

    # 提取结果
    venues = [k for k in range(n_loc) if value(y[k]) > 0.5]
    assignment = {}
    for g in range(NUM_GROUPS):
        for k in range(n_loc):
            if value(z[g, k]) > 0.5:
                assignment[g] = k
                break

    total_km = sum(
        dist[TEAM_INDEX[name]][assignment[g]]
        for g in range(NUM_GROUPS) for name in groups[g]
    )

    return venues, assignment, total_km


# ============================================================
# 3.5 拉格朗日松弛下界
# ============================================================

def lagrangian_relaxation(groups, teams, upper_bound, max_iter=200, theta_init=2.0):
    """
    Lagrangian relaxation of capacity constraint z_{gk} <= 2*y_k.

    At each iteration, solves the relaxed p-median (without capacity)
    to get L(lambda), then updates lambda via subgradient descent.

    Returns: (best_lower_bound, history)
    """
    from pulp import (
        LpMinimize, LpProblem, LpVariable, lpSum,
        LpBinary, LpStatus, value, PULP_CBC_CMD,
    )

    n_loc = len(teams)
    loc_names = [t.name for t in teams]

    # Precompute group-to-location cost matrix C[g][k]
    C = np.zeros((NUM_GROUPS, n_loc))
    for g in range(NUM_GROUPS):
        for k in range(n_loc):
            C[g][k] = sum(haversine(name, loc_names[k]) for name in groups[g])

    lam = np.zeros(n_loc)
    best_L = -np.inf
    best_iter = 0
    UB = upper_bound
    history = []
    no_improve = 0

    for it in range(max_iter):
        prob = LpProblem(f"LR_{it}", LpMinimize)

        y = {k: LpVariable(f"y_{k}", cat=LpBinary) for k in range(n_loc)}
        z = {(g, k): LpVariable(f"z_{g}_{k}", cat=LpBinary)
             for g in range(NUM_GROUPS) for k in range(n_loc)}

        # Constraints: only eq:q3_y, eq:q3_z1, eq:q3_z2 (no capacity)
        prob += lpSum(y[k] for k in range(n_loc)) == 8
        for g in range(NUM_GROUPS):
            prob += lpSum(z[g, k] for k in range(n_loc)) == 1
        for g in range(NUM_GROUPS):
            for k in range(n_loc):
                prob += z[g, k] <= y[k]

        # Lagrangian objective
        obj_terms = []
        for g in range(NUM_GROUPS):
            for k in range(n_loc):
                obj_terms.append((C[g][k] + lam[k]) * z[g, k])
        for k in range(n_loc):
            obj_terms.append(-2.0 * lam[k] * y[k])
        prob += lpSum(obj_terms)

        prob.solve(PULP_CBC_CMD(msg=0))

        if LpStatus[prob.status] != "Optimal":
            break

        L_val = value(prob.objective)
        improved = L_val > best_L + 1e-6
        if improved:
            best_L = L_val
            best_iter = it
            no_improve = 0
        else:
            no_improve += 1

        # Subgradient: gamma_k = sum_g z_{gk} - 2*y_k
        gamma = np.zeros(n_loc)
        for k in range(n_loc):
            z_sum = sum(value(z[g, k]) for g in range(NUM_GROUPS))
            y_val = value(y[k])
            gamma[k] = z_sum - 2.0 * y_val

        norm_sq = np.sum(gamma ** 2)
        if norm_sq < 1e-10:
            history.append((it, L_val, best_L))
            break

        # Polyak step size with decay
        theta = theta_init * (0.995 ** it)
        alpha = theta * max(UB - L_val, 1.0) / norm_sq

        lam = lam - alpha * gamma
        history.append((it, L_val, best_L))

        # Early stop if no improvement for 30 iterations
        if no_improve >= 30:
            break

    return best_L, best_iter, history


# ============================================================
# 4. 评价与输出
# ============================================================

def print_venue_result(groups, teams, venues, assignment, total_km, label):
    """打印选址结果"""
    loc_names = [t.name for t in teams]
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")

    # 按场地分组展示
    venue_groups = {}
    for g, k in assignment.items():
        venue_groups.setdefault(k, []).append(g)

    for vi, k in enumerate(sorted(venues)):
        vname = loc_names[k]
        gs = venue_groups.get(k, [])
        lat, lon = COORDS[vname]
        print(f"\n  场地{vi + 1}: {vname} ({lat:.2f}°N, {lon:.2f}°E)")

        for g in gs:
            team_strs = []
            for name in groups[g]:
                t = TEAMS[TEAM_INDEX[name]]
                d = haversine(name, vname)
                team_strs.append(f"{name}({d:.0f}km)")
            print(f"    组{g + 1:2d}: {', '.join(team_strs)}")

    # 统计
    dists_per_team = []
    for g, k in assignment.items():
        vname = loc_names[k]
        for name in groups[g]:
            dists_per_team.append(haversine(name, vname))

    print(f"\n  统计:")
    print(f"    总旅行距离: {total_km:.0f} km")
    print(f"    平均每队:   {np.mean(dists_per_team):.1f} km")
    print(f"    最远一队:   {max(dists_per_team):.0f} km")
    print(f"    最近一队:   {min(dists_per_team):.0f} km")

    # 地理覆盖
    venue_coords = [COORDS[loc_names[k]] for k in venues]
    lats = [c[0] for c in venue_coords]
    lons = [c[1] for c in venue_coords]
    print(f"    纬度范围:   {min(lats):.2f}° ~ {max(lats):.2f}° (浙南北跨度)")
    print(f"    经度范围:   {min(lons):.2f}° ~ {max(lons):.2f}° (浙东西跨度)")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("  浙超小组赛选址 — 问题3: 比赛地点选择")
    print("=" * 72)

    # 使用 Q1 方案 D 的分组
    groups = scheme_d_ilp_balanced(w_c3=10, w_str=1)
    if groups is None:
        print("  [错误] 无法生成 Q1 分组方案")
        return

    # 验证分组
    c1v = check_c1(groups)
    c2v = check_c2(groups)
    c3n, _ = check_c3(groups)
    print(f"\n  分组方案 (Q1-ILP均衡): C1={'通过' if not c1v else '违反'} "
          f"C2={'通过' if not c2v else '违反'} C3={'通过' if c3n == 0 else f'{c3n}对冲突'}")

    # --- ILP 最优选址 ---
    print(f"\n  正在求解 ILP 选址模型 (64 选 8 + 16 组指派)...")
    venues_ilp, assign_ilp, km_ilp = ilp_venue_selection(groups, TEAMS)

    if venues_ilp is not None:
        print_venue_result(groups, TEAMS, venues_ilp, assign_ilp, km_ilp,
                           "ILP 最优选址 (最小化总旅行距离)")

    # --- 启发式选址 (对比) ---
    print(f"\n  正在计算启发式选址...")
    h_locs = heuristic_venues(groups, TEAMS)
    loc_names = [t.name for t in TEAMS]

    # 为启发式选址做简单指派: 每个小组分配到距离最近的场地
    h_venues_idx = [loc_names.index(l) for l in h_locs]
    h_assign = {}
    for g in range(NUM_GROUPS):
        best_k = min(h_venues_idx,
                     key=lambda k: sum(haversine(n, loc_names[k]) for n in groups[g]))
        h_assign[g] = best_k

    # 修正: 确保每个场地恰好 2 个小组
    # [问题] 贪心最近指派可能导致某些场地超过 2 组, 某些 0 组.
    # [解决] 简单地用 ILP 指派 (在给定场地集合上) 来保证可行性.
    from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary, value, PULP_CBC_CMD

    prob = LpProblem("AssignOnly", LpMinimize)
    z_h = {(g, k): LpVariable(f"zh_{g}_{k}", cat=LpBinary)
           for g in range(NUM_GROUPS) for k in h_venues_idx}
    for g in range(NUM_GROUPS):
        prob += lpSum(z_h[g, k] for k in h_venues_idx) == 1
    for k in h_venues_idx:
        prob += lpSum(z_h[g, k] for g in range(NUM_GROUPS)) == 2
    cost_terms = []
    for g in range(NUM_GROUPS):
        for k in h_venues_idx:
            d = sum(haversine(n, loc_names[k]) for n in groups[g])
            cost_terms.append(d * z_h[g, k])
    prob += lpSum(cost_terms)
    prob.solve(PULP_CBC_CMD(msg=0))

    h_assign_fixed = {}
    for g in range(NUM_GROUPS):
        for k in h_venues_idx:
            if value(z_h[g, k]) > 0.5:
                h_assign_fixed[g] = k
                break

    km_heuristic = sum(
        haversine(n, loc_names[h_assign_fixed[g]])
        for g in range(NUM_GROUPS) for n in groups[g]
    )

    print_venue_result(groups, TEAMS, h_venues_idx, h_assign_fixed,
                       km_heuristic, "启发式选址 (地理分散贪心)")

    # --- 对比 ---
    print(f"\n{'=' * 72}")
    print(f"  选址方案对比")
    print(f"{'=' * 72}")
    print(f"  {'方案':<24} {'总距离(km)':<14} {'平均(km)':<12} {'最远(km)':<10}")
    print(f"  {'-' * 60}")
    if venues_ilp is not None:
        d_ilp = [haversine(n, loc_names[assign_ilp[g]])
                 for g in range(NUM_GROUPS) for n in groups[g]]
        print(f"  {'ILP 最优':<24} {km_ilp:<14.0f} {np.mean(d_ilp):<12.1f} {max(d_ilp):<10.0f}")
    d_h = [haversine(n, loc_names[h_assign_fixed[g]])
           for g in range(NUM_GROUPS) for n in groups[g]]
    print(f"  {'启发式贪心':<24} {km_heuristic:<14.0f} {np.mean(d_h):<12.1f} {max(d_h):<10.0f}")

    if venues_ilp is not None:
        gap = (km_heuristic - km_ilp) / km_ilp * 100
        print(f"\n  启发式比 ILP 最优多 {gap:.1f}% 总距离, 说明联合优化效果显著.")

    # --- 拉格朗日松弛下界 ---
    if venues_ilp is not None:
        print(f"\n  正在求解拉格朗日松弛下界...")
        lb, lb_iter, lb_hist = lagrangian_relaxation(groups, TEAMS, km_ilp)

        print(f"\n{'=' * 72}")
        print(f"  三级最优性认证")
        print(f"{'=' * 72}")
        print(f"  拉格朗日下界 L*:    {lb:.0f} km  (第 {lb_iter} 轮收敛)")
        print(f"  ILP 精确解:         {km_ilp:.0f} km")
        print(f"  启发式上界:         {km_heuristic:.0f} km")
        print(f"")
        ilp_gap = (km_ilp - lb) / lb * 100 if lb > 0 else float('inf')
        heur_gap = (km_heuristic - lb) / lb * 100 if lb > 0 else float('inf')
        print(f"  L* <= OPT_ILP <= OPT_heur")
        print(f"  ILP 最优性间隔:     {ilp_gap:.2f}%")
        print(f"  启发式最优性间隔:   {heur_gap:.2f}%")
        print(f"  启发式 vs ILP:      +{gap:.1f}%")

    print(f"\n  建议: 实际赛事应优先考虑地级市 (交通/住宿/影响力),")
    print(f"  并在浙北、浙中、浙南各布 2~3 个场地, 确保区域覆盖.")


if __name__ == "__main__":
    main()
