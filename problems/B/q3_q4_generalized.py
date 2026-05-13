"""
Q3+Q4 广义蒙特卡洛：随机拓扑上的选址鲁棒性与赛制公平性
========================================================

核心链条: tau(约束紧度) -> 分组质量 -> 选址效果/赛制公平性

复用 generalized_mc.py 的拓扑生成器, 在随机"架空省份"上验证:
  - Q3: 选址效果是否随 tau 变化 (贪心选址)
  - Q4: 赛制公平性是否随 tau 变化 (Bradley-Terry 仿真)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import defaultdict
from generalized_mc import (
    generate_random_topology, build_teams_from_topology,
    greedy_on_topology, compute_f1, compute_f2, compute_f3,
    ProvinceTopology,
)


# ============================================================
# 1. 随机空间坐标生成 (Q3 用)
# ============================================================

def generate_random_coords(topo, teams, rng, spread=0.15):
    """
    为随机拓扑生成2D坐标, 模拟地理分布.

    每个市生成一个中心点, 其县级队在中心附近随机偏移.
    返回 dict: team_name -> (x, y), 坐标范围 [0, 1]^2.
    """
    coords = {}
    # 各市中心
    city_centers = {}
    for i in range(topo.k):
        cx, cy = rng.uniform(0.1, 0.9, size=2)
        city_centers[i] = (cx, cy)

    for t in teams:
        city = t["city"]
        cx, cy = city_centers[city]
        if t["level"] == "municipal":
            coords[t["name"]] = (cx, cy)
        else:
            # 县级队在市中心附近偏移
            dx, dy = rng.normal(0, spread, size=2)
            x = np.clip(cx + dx, 0.01, 0.99)
            y = np.clip(cy + dy, 0.01, 0.99)
            coords[t["name"]] = (x, y)

    return coords


def euclidean_dist(coords, name1, name2, scale=400):
    """欧氏距离 × 缩放因子 (模拟公里数)"""
    x1, y1 = coords[name1]
    x2, y2 = coords[name2]
    return scale * np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ============================================================
# 2. 贪心选址 (Q3 用, 不依赖 ILP)
# ============================================================

def greedy_venue_selection(groups, teams, coords):
    """
    贪心选8个场地, 每个承办2组, 最小化总旅行距离.

    1. 计算每个地点作为场地时承接所有组的加权距离
    2. 贪心选距离最小 + 分散性最好的点
    3. 将16组指派到8个场地 (每组到最近场地, 每场地恰好2组)
    """
    all_names = [t["name"] for t in teams]
    name_set = set(all_names)

    # 候选: 所有64个地点
    candidates = list(name_set)

    # 每个候选地点到所有球队的距离和
    cost = {}
    for loc in candidates:
        cost[loc] = sum(euclidean_dist(coords, loc, n) for n in all_names)

    # 贪心选8个: 综合距离成本 + 分散性
    sorted_cands = sorted(candidates, key=lambda x: cost[x])
    selected = [sorted_cands[0]]

    for _ in range(7):
        best_score = -1
        best_loc = None
        for loc in sorted_cands:
            if loc in selected:
                continue
            min_dist = min(euclidean_dist(coords, loc, s) for s in selected)
            score = min_dist / (cost[loc] + 1)
            if score > best_score:
                best_score = score
                best_loc = loc
        if best_loc:
            selected.append(best_loc)

    # 指派: 每组分配到距离和最小的场地, 保证每场地恰好2组
    # 简单贪心: 按组排序, 依次分配到剩余名额中距离最小的场地
    venue_capacity = {v: 2 for v in selected}
    assignment = {}

    # 计算每组到每个场地的距离
    group_venue_dist = {}
    for g, group in enumerate(groups):
        for v in selected:
            group_venue_dist[(g, v)] = sum(euclidean_dist(coords, n, v) for n in group)

    # 按距离排序所有 (组, 场地) 对, 贪心分配
    pairs = sorted(group_venue_dist.keys(), key=lambda p: group_venue_dist[p])
    assigned_groups = set()

    for g, v in pairs:
        if g in assigned_groups:
            continue
        if venue_capacity.get(v, 0) <= 0:
            continue
        assignment[g] = v
        venue_capacity[v] -= 1
        assigned_groups.add(g)
        if len(assigned_groups) == len(groups):
            break

    # 计算总距离
    total_dist = sum(
        euclidean_dist(coords, n, assignment[g])
        for g, group in enumerate(groups) for n in group
    )

    # 每队距离
    team_dists = [
        euclidean_dist(coords, n, assignment[g])
        for g, group in enumerate(groups) for n in group
    ]

    return {
        "total_dist": total_dist,
        "avg_dist": np.mean(team_dists),
        "max_dist": max(team_dists) if team_dists else 0,
        "min_dist": min(team_dists) if team_dists else 0,
    }


# ============================================================
# 3. 赛制仿真 (Q4 用, 泛化版)
# ============================================================

def bt_match(rng, s_i, s_j):
    """Bradley-Terry: P(i胜) = s_i / (s_i + s_j)"""
    return rng.random() < s_i / (s_i + s_j)


def simulate_tournament_generalized(rng, groups, teams):
    """
    完整赛制仿真 (泛化版, 不依赖全局 TEAM_INDEX).

    返回 dict 包含公平性指标.
    """
    team_dict = {t["name"]: t for t in teams}
    strengths = np.array([t["strength"] for t in teams], dtype=float)
    # 实力排名 (1=最强)
    strength_rank = np.argsort(-strengths) + 1  # 用 argsort 的逆序

    n_teams = len(teams)
    name_to_idx = {t["name"]: i for i, t in enumerate(teams)}

    # --- 小组赛 ---
    qualified = []
    group_rankings = []

    for group in groups:
        indices = [name_to_idx[n] for n in group]
        points = defaultdict(int)
        goals = defaultdict(int)

        for a in range(4):
            for b in range(a + 1, 4):
                ia, ib = indices[a], indices[b]
                if bt_match(rng, strengths[ia], strengths[ib]):
                    points[ia] += 3
                    goals[ia] += rng.poisson(0.5 + strengths[ia] * 0.3)
                else:
                    points[ib] += 3
                    goals[ib] += rng.poisson(0.5 + strengths[ib] * 0.3)

        ranked = sorted(indices, key=lambda i: (points[i], goals[i], rng.random()),
                        reverse=True)
        group_rankings.append(ranked)
        qualified.extend(ranked[:2])

    # --- 淘汰赛 ---
    current = list(qualified)
    rng.shuffle(current)
    eliminated = []
    round_names = ["32强", "16强", "八强", "四强", "决赛"]

    for rnd in range(5):
        next_round = []
        for i in range(0, len(current), 2):
            ia, ib = current[i], current[i + 1]
            if bt_match(rng, strengths[ia], strengths[ib]):
                winner, loser = ia, ib
            else:
                winner, loser = ib, ia
            next_round.append(winner)
            eliminated.append(loser)
        current = next_round

    champion = current[0]
    final_ranking = [champion] + eliminated[::-1]

    # 小组赛淘汰者
    group_losers = []
    for ranking in group_rankings:
        group_losers.extend(ranking[2:])
    group_losers.sort(key=lambda i: strengths[i], reverse=True)
    full_ranking = final_ranking + group_losers

    # --- 指标 ---
    # Spearman: 排名位置 vs 实力排名
    rank_positions = np.arange(1, n_teams + 1)
    actual_strength_ranks = np.array([strength_rank[i] for i in full_ranking])
    from scipy.stats import spearmanr
    rho, _ = spearmanr(rank_positions, actual_strength_ranks)

    # 前32晋级率
    top32_indices = set(np.argsort(-strengths)[:32])
    qualified_set = set(qualified)
    top32_rate = len(top32_indices & qualified_set) / 32

    # 冠军实力
    champ_rank = strength_rank[champion]
    champ_top1 = 1 if champ_rank == 1 else 0
    champ_top4 = 1 if champ_rank <= 4 else 0
    champ_top8 = 1 if champ_rank <= 8 else 0

    return {
        "spearman": rho,
        "top32_rate": top32_rate,
        "champ_top1": champ_top1,
        "champ_top4": champ_top4,
        "champ_top8": champ_top8,
    }


# ============================================================
# 4. 主实验
# ============================================================

def run_experiment(n_topologies=150, n_seeds_venue=10, n_seeds_tournament=50):
    print("=" * 72)
    print("  Q3+Q4 广义蒙特卡洛: 选址鲁棒性与赛制公平性")
    print(f"  {n_topologies} 个随机拓扑")
    print(f"  Q3: 每拓扑 {n_seeds_venue} 次选址 | Q4: 每拓扑 {n_seeds_tournament} 次赛制仿真")
    print("=" * 72)

    results = []

    for ti in range(n_topologies):
        rng_topo = np.random.default_rng(ti * 1000 + 7)
        topo = generate_random_topology(rng_topo)
        teams = build_teams_from_topology(topo)

        stats = {
            "k": topo.k, "max_county": topo.max_county,
            "tightness": topo.tightness, "gini": topo.gini,
            # Q3 选址
            "total_dist": [], "avg_dist": [], "max_dist": [],
            # Q4 赛制
            "spearman": [], "top32_rate": [],
            "champ_top1": 0, "champ_top4": 0, "champ_top8": 0,
        }

        # --- Q3: 选址 (少量种子, 每次重新生成坐标和分组) ---
        for si in range(n_seeds_venue):
            rng = np.random.default_rng(ti * 10000 + si)
            rng_coords = np.random.default_rng(ti * 50000 + si)

            coords = generate_random_coords(topo, teams, rng_coords)
            groups, ok, _ = greedy_on_topology(topo, teams, rng, repair=True)
            if not ok:
                continue

            venue_result = greedy_venue_selection(groups, teams, coords)
            stats["total_dist"].append(venue_result["total_dist"])
            stats["avg_dist"].append(venue_result["avg_dist"])
            stats["max_dist"].append(venue_result["max_dist"])

        # --- Q4: 赛制仿真 (用同一分组, 多次随机比赛结果) ---
        rng_group = np.random.default_rng(ti * 1000 + 7)
        groups, ok, _ = greedy_on_topology(topo, teams, rng_group, repair=True)
        if ok:
            for si in range(n_seeds_tournament):
                rng_sim = np.random.default_rng(ti * 20000 + si)
                m = simulate_tournament_generalized(rng_sim, groups, teams)
                stats["spearman"].append(m["spearman"])
                stats["top32_rate"].append(m["top32_rate"])
                stats["champ_top1"] += m["champ_top1"]
                stats["champ_top4"] += m["champ_top4"]
                stats["champ_top8"] += m["champ_top8"]

        results.append(stats)

        if (ti + 1) % 50 == 0:
            print(f"  完成 {ti + 1}/{n_topologies}")

    # ============================================================
    # 输出统计
    # ============================================================

    tau = np.array([r["tightness"] for r in results])

    zj = ProvinceTopology(k=11, county_per_city=[3, 4, 8, 5, 3, 3, 7, 4, 2, 6, 8])

    # --- Q3 ---
    print(f"\n{'=' * 72}")
    print(f"  Q3: 选址效果 vs 约束紧度")
    print(f"{'=' * 72}")

    total_dists = np.array([np.mean(r["total_dist"]) if r["total_dist"] else 0 for r in results])
    avg_dists = np.array([np.mean(r["avg_dist"]) if r["avg_dist"] else 0 for r in results])
    max_dists = np.array([np.mean(r["max_dist"]) if r["max_dist"] else 0 for r in results])

    corr_total = np.corrcoef(tau, total_dists)[0, 1]
    corr_max = np.corrcoef(tau, max_dists)[0, 1]

    print(f"  tau 与 总旅行距离 的相关系数: {corr_total:.4f}")
    print(f"  tau 与 最大单队距离 的相关系数: {corr_max:.4f}")
    print(f"  (接近0表示选址效果不依赖于约束紧度)")

    # --- Q4 ---
    print(f"\n{'=' * 72}")
    print(f"  Q4: 赛制公平性 vs 约束紧度")
    print(f"{'=' * 72}")

    spearman_means = np.array([np.mean(r["spearman"]) if r["spearman"] else 0 for r in results])
    top32_means = np.array([np.mean(r["top32_rate"]) if r["top32_rate"] else 0 for r in results])
    champ_top1_rates = np.array([
        r["champ_top1"] / n_seeds_tournament if r["spearman"] else 0 for r in results
    ])

    corr_sp = np.corrcoef(tau, spearman_means)[0, 1]
    corr_t32 = np.corrcoef(tau, top32_means)[0, 1]
    corr_ch = np.corrcoef(tau, champ_top1_rates)[0, 1]

    print(f"  tau 与 Spearman 的相关系数: {corr_sp:.4f}")
    print(f"  tau 与 前32晋级率 的相关系数: {corr_t32:.4f}")
    print(f"  tau 与 Top1夺冠率 的相关系数: {corr_ch:.4f}")

    # 分桶统计
    bins = [0, 0.5, 0.7, 1.0]
    labels = ["tau<=0.50", "0.50<tau<=0.70", "tau>0.70"]

    print(f"\n  按紧度分桶:")
    print(f"  {'区间':<16} {'拓扑':<6} {'Spearman':<12} {'晋级率':<10} {'Top1夺冠':<10} {'平均总距离':<12}")
    print(f"  {'-' * 66}")

    for b in range(3):
        bucket = [r for r in results if bins[b] < r["tightness"] <= bins[b + 1]]
        if not bucket:
            continue
        sp = np.mean([s for r in bucket for s in r["spearman"]]) if bucket else 0
        t32 = np.mean([s for r in bucket for s in r["top32_rate"]]) if bucket else 0
        ch1 = np.mean([r["champ_top1"] / n_seeds_tournament for r in bucket if r["spearman"]])
        td = np.mean([np.mean(r["total_dist"]) for r in bucket if r["total_dist"]]) if bucket else 0
        print(f"  {labels[b]:<16} {len(bucket):<6} {sp:<12.4f} {t32:<10.4f} {ch1:<10.4f} {td:<12.0f}")

    print(f"\n  浙江省参考 (tau={zj.tightness:.2f}):")
    print(f"    紧度属于偏紧区间, 但公平性指标应在正常范围内")

    # 保存供绘图使用
    return results, n_seeds_tournament


if __name__ == "__main__":
    run_experiment()
