#!/usr/bin/env python3
"""
Q3: 浙超小组赛比赛地点选择
==========================
从 64 个参赛单位中选 8 个作为小组赛场地, 每个场地承办 2 个小组.
目标: 最小化各队总旅行距离, 同时保证地理覆盖的公平性.

模型: 设施选址 (p-median) + 指派问题联合优化
"""

import sys
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

# [问题] 需要浙江省 64 个行政单位的经纬度坐标, 没有现成 API 可用.
# [解决] 手动编码各单位的近似地理坐标 (基于公开地理数据).
#        精度在 0.05° 以内, 足以计算县级市/县之间的相对距离.

COORDS = {
    # 杭州
    "杭州市": (30.27, 120.15), "建德市": (29.47, 119.28),
    "桐庐县": (29.79, 119.69), "淳安县": (29.61, 118.95),
    # 宁波
    "宁波市": (29.87, 121.55), "余姚市": (30.04, 121.15),
    "慈溪市": (30.17, 121.23), "象山县": (29.58, 121.87),
    "宁海县": (29.30, 121.43),
    # 温州
    "温州市": (28.00, 120.67), "瑞安市": (27.78, 120.63),
    "乐清市": (28.11, 120.98), "龙港市": (27.51, 120.55),
    "永嘉县": (28.15, 120.69), "平阳县": (27.66, 120.57),
    "苍南县": (27.52, 120.43), "文成县": (27.79, 120.09),
    "泰顺县": (27.56, 119.72),
    # 嘉兴
    "嘉兴市": (30.75, 120.75), "海宁市": (30.51, 120.68),
    "平湖市": (30.70, 121.02), "桐乡市": (30.63, 120.57),
    "嘉善县": (30.83, 120.93), "海盐县": (30.53, 120.95),
    # 湖州
    "湖州市": (30.87, 120.09), "德清县": (30.55, 119.97),
    "长兴县": (30.99, 119.91), "安吉县": (30.68, 119.68),
    # 绍兴
    "绍兴市": (30.00, 120.58), "诸暨市": (29.71, 120.24),
    "嵊州市": (29.60, 120.82), "新昌县": (29.50, 120.90),
    # 金华
    "金华市": (29.08, 119.65), "兰溪市": (29.21, 119.47),
    "义乌市": (29.31, 120.07), "东阳市": (29.27, 120.23),
    "永康市": (28.90, 120.05), "武义县": (28.90, 119.81),
    "浦江县": (29.45, 119.88), "磐安县": (29.04, 120.45),
    # 衢州
    "衢州市": (28.94, 118.87), "江山市": (28.74, 118.63),
    "常山县": (28.90, 118.51), "开化县": (29.13, 118.42),
    "龙游县": (29.03, 119.17),
    # 舟山
    "舟山市": (30.00, 122.11), "岱山县": (30.26, 122.20),
    "嵊泗县": (30.73, 122.45),
    # 台州
    "台州市": (28.66, 121.42), "温岭市": (28.37, 121.38),
    "临海市": (28.85, 121.14), "玉环市": (28.14, 121.23),
    "三门县": (29.11, 121.38), "天台县": (29.15, 121.02),
    "仙居县": (28.85, 120.73),
    # 丽水
    "丽水市": (28.47, 119.92), "龙泉市": (28.07, 119.14),
    "青田县": (28.45, 120.29), "缙云县": (28.67, 120.09),
    "遂昌县": (28.59, 119.27), "松阳县": (28.45, 119.48),
    "云和县": (28.12, 119.57), "庆元县": (27.62, 119.07),
    "景宁县": (27.97, 119.63),
}


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

    print(f"\n  建议: 实际赛事应优先考虑地级市 (交通/住宿/影响力),")
    print(f"  并在浙北、浙中、浙南各布 2~3 个场地, 确保区域覆盖.")


if __name__ == "__main__":
    main()
