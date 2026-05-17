#!/usr/bin/env python3
"""
Q4: 浙超赛制合理化建议 — 蒙特卡洛仿真
======================================
基于 Bradley-Terry 模型模拟小组赛 + 淘汰赛,
定量评价当前赛制的公平性, 并对比多种赛制变体.

评价指标:
  - 公平性: 实力排名与最终名次的 Spearman 相关系数
  - 晋级准确率: 实力前 32 名实际晋级的比例
  - 冠军集中度: 实力 Top-1 的夺冠概率
  - 稳健性: 层内正态扰动 s_i ~ N(s_level, (1/3)^2)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import defaultdict
from scipy import stats
from q1_grouping import (
    TEAMS,
    TEAM_INDEX,
    check_c1,
    check_c2,
    check_c3,
)


# 方案D: ILP(均衡), 来自 q1_output.txt.
# Q4只评价赛制, 固定分组可避免每次运行都重新求解ILP。
BALANCED_GROUPS = [
    ["龙港市", "德清县", "东阳市", "三门县"],
    ["嘉善县", "金华市", "岱山县", "庆元县"],
    ["永嘉县", "平湖市", "长兴县", "舟山市"],
    ["温州市", "海盐县", "诸暨市", "磐安县"],
    ["余姚市", "安吉县", "浦江县", "丽水市"],
    ["杭州市", "象山县", "兰溪市", "常山县"],
    ["宁波市", "江山市", "仙居县", "青田县"],
    ["建德市", "苍南县", "龙游县", "温岭市"],
    ["武义县", "衢州市", "玉环市", "云和县"],
    ["文成县", "绍兴市", "永康市", "开化县"],
    ["宁海县", "嘉兴市", "义乌市", "遂昌县"],
    ["瑞安市", "海宁市", "新昌县", "松阳县"],
    ["桐庐县", "乐清市", "临海市", "景宁县"],
    ["淳安县", "慈溪市", "台州市", "缙云县"],
    ["平阳县", "桐乡市", "嵊州市", "龙泉市"],
    ["泰顺县", "湖州市", "嵊泗县", "天台县"],
]

LAYER_SIGMA = 1 / 3
MIN_STRENGTH = 0.05


# ============================================================
# 1. Bradley-Terry 比赛模拟
# ============================================================

# [问题] 需要一个合理的比赛结果模型. 足球比赛有平局, 但建模竞赛中
#        可以简化为"无平局"以聚焦赛制分析.
# [解决] 采用 Bradley-Terry 模型 P(i胜j) = s_i/(s_i+s_j),
#        不考虑平局. 小组赛积分: 胜=3分, 负=0分.
#        如果需要更真实的模型, 可引入随机进球数 (Poisson).

def draw_layered_strengths(rng, sigma=LAYER_SIGMA, min_strength=MIN_STRENGTH):
    """
    层内正态实力模型.
    以 3/2/1 为层级均值, 每次赛事先抽取一组潜在真实实力.
    为保证 Bradley-Terry 概率有效, 对极小概率的非正抽样做截断重抽.
    """
    means = np.array([t.strength for t in TEAMS], dtype=float)
    strengths = rng.normal(means, sigma)
    invalid = strengths <= min_strength
    while np.any(invalid):
        strengths[invalid] = rng.normal(means[invalid], sigma)
        invalid = strengths <= min_strength
    return strengths


def match_result(rng, strength_i, strength_j):
    """
    Bradley-Terry 单场结果.
    返回: (i得分, j得分) — 小组赛中 胜=3, 负=0.
    """
    p_i_win = strength_i / (strength_i + strength_j)
    if rng.random() < p_i_win:
        return 3, 0
    else:
        return 0, 3


def simulate_group_stage(rng, groups, team_strengths):
    """
    模拟小组赛 (单循环).
    每组 4 队, 共 6 场: (0,1)(0,2)(0,3)(1,2)(1,3)(2,3).
    返回: 每组按积分降序排列的球队索引列表.
    """
    group_rankings = []

    for group in groups:
        indices = [TEAM_INDEX[name] for name in group]
        points = defaultdict(int)
        goals_for = defaultdict(int)  # 进球数 (用于排名)

        for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            ia, ib = indices[a], indices[b]
            sa, sb = match_result(rng, team_strengths[ia], team_strengths[ib])
            points[ia] += sa
            points[ib] += sb
            # 模拟进球数 (Poisson, 均值与实力成正比)
            gf_a = rng.poisson(0.5 + team_strengths[ia] * 0.3)
            gf_b = rng.poisson(0.5 + team_strengths[ib] * 0.3)
            goals_for[ia] += gf_a
            goals_for[ib] += gf_b

        # [问题] 小组积分相同时如何排名?
        # [解决] 依次比较: 积分 → 净胜球 (用进球数近似) → 随机抽签.
        ranked = sorted(indices, key=lambda i: (points[i], goals_for[i], rng.random()),
                        reverse=True)
        group_rankings.append(ranked)

    return group_rankings


def simulate_knockout(rng, team_indices, team_strengths):
    """
    模拟淘汰赛 (单场淘汰).
    返回: (冠军index, 最终排名列表 — 从第1名到第32名).
    """
    current_round = list(team_indices)  # 32 支队
    rankings = []  # 被淘汰的队, 按名次 (后淘汰 = 名次高)

    # [问题] 淘汰赛的对阵如何确定?
    # [解决] 简化为: 每轮随机配对 (实际赛事会根据小组排名设置种子,
    #        但为了分析赛制本身的公平性, 随机配对更中性).
    #        更合理的做法: 1A vs 2B 等, 但需要完整的 bracket 设计.
    #        此处用"相邻配对": sorted by group rank → (0,1)(2,3)...

    # 第1轮: 按小组排名配对 (1st vs 2nd from different groups)
    # 简化: 打乱后相邻配对
    rng.shuffle(current_round)

    round_names = ["32强", "16强", "八强", "四强", "决赛"]
    eliminated = []

    for rnd, name in enumerate(round_names):
        next_round = []
        for i in range(0, len(current_round), 2):
            ia, ib = current_round[i], current_round[i + 1]
            sa, sb = match_result(rng, team_strengths[ia], team_strengths[ib])
            if sa > sb:
                winner, loser = ia, ib
            else:
                winner, loser = ib, ia
            next_round.append(winner)
            eliminated.append(loser)
        current_round = next_round

    # 冠军
    champion = current_round[0]
    # 排名: 冠军 → 亚军 (决赛败者) → 四强败者 → ... → 32强败者
    final_ranking = [champion] + eliminated[::-1]
    return champion, final_ranking


# ============================================================
# 2. 完整赛制仿真
# ============================================================

def simulate_tournament(rng, groups, team_strengths):
    """运行一次完整赛制 (小组赛 + 淘汰赛), 返回最终排名"""
    group_rankings = simulate_group_stage(rng, groups, team_strengths)

    # 小组前 2 名晋级
    qualified = []
    for ranking in group_rankings:
        qualified.extend(ranking[:2])

    champion, final_ranking = simulate_knockout(rng, qualified, team_strengths)

    # 完整排名 (64 队): 淘汰赛排名 + 小组赛淘汰的 32 队
    knockout_ranking = final_ranking  # 32 队排名
    group_losers = []
    for ranking in group_rankings:
        group_losers.extend(ranking[2:])  # 每组第 3、4 名

    # 小组赛淘汰队按实力降序排名 (第 33~64 名)
    group_losers.sort(key=lambda i: team_strengths[i], reverse=True)

    full_ranking = knockout_ranking + group_losers
    return full_ranking


def compute_metrics(rng, groups, team_strengths, n_sim):
    """蒙特卡洛仿真, 计算多项公平性指标"""
    strength_rank = stats.rankdata([-s for s in team_strengths])  # 实力排名 (1=最强)

    spearmans = []
    top32_rates = []
    champion_top1 = 0
    champion_top4 = 0
    champion_top8 = 0
    upsets_group = 0
    upsets_knockout = 0
    total_group_matches = 0
    total_knockout_matches = 0

    for _ in range(n_sim):
        # 小组赛
        group_rankings = simulate_group_stage(rng, groups, team_strengths)
        qualified = []
        for ranking in group_rankings:
            qualified.extend(ranking[:2])

        # 晋级准确率: 实力前32中实际晋级了多少
        top32_actual = sum(1 for i in range(len(team_strengths))
                          if strength_rank[i] <= 32 and i in qualified)
        top32_rates.append(top32_actual / 32)

        # 小组赛爆冷: 实力弱的队赢了实力强的队
        for group in [g for g in groups]:
            indices = [TEAM_INDEX[n] for n in group]
            for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
                ia, ib = indices[a], indices[b]
                p_i = team_strengths[ia] / (team_strengths[ia] + team_strengths[ib])
                total_group_matches += 1
                # 爆冷: 弱队获胜 (概率 < 0.4)
                if p_i < 0.4:
                    upsets_group += 1 - p_i  # 用期望爆冷度

        # 淘汰赛
        champion, final_ranking = simulate_knockout(rng, qualified, team_strengths)
        total_knockout_matches += 31  # 32队淘汰赛 = 31场

        # 冠军实力
        if strength_rank[champion] <= 1:
            champion_top1 += 1
        if strength_rank[champion] <= 4:
            champion_top4 += 1
        if strength_rank[champion] <= 8:
            champion_top8 += 1

        # 完整排名 → Spearman 相关系数
        full_ranking = final_ranking
        group_losers = []
        for ranking in group_rankings:
            group_losers.extend(ranking[2:])
        group_losers.sort(key=lambda i: team_strengths[i], reverse=True)
        full_ranking = full_ranking + group_losers

        # 计算: 排名位置 vs 实力排名
        pos = list(range(1, 65))
        actual_strength_ranks = [strength_rank[i] for i in full_ranking]
        rho, _ = stats.spearmanr(pos, actual_strength_ranks)
        spearmans.append(rho)

    return {
        "spearman": np.array(spearmans),
        "top32_rate": np.array(top32_rates),
        "champion_top1": champion_top1 / n_sim,
        "champion_top4": champion_top4 / n_sim,
        "champion_top8": champion_top8 / n_sim,
        "upset_rate_group": upsets_group / max(total_group_matches, 1),
        "upset_rate_knockout": upsets_knockout / max(total_knockout_matches, 1),
    }


# ============================================================
# 3. 替代赛制对比
# ============================================================

def simulate_double_round_robin_group(rng, groups, team_strengths):
    """
    双循环小组赛: 每对球队打 2 场 (主客场).
    与单循环对比, 增加样本量, 减少偶然性.

    [问题] 双循环改为 12 场/组 (vs 单循环 6 场/组), 更公平但赛程更长.
    """
    group_rankings = []
    for group in groups:
        indices = [TEAM_INDEX[name] for name in group]
        points = defaultdict(int)
        goals_for = defaultdict(int)

        for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            for _ in range(2):  # 主客场各一次
                ia, ib = indices[a], indices[b]
                sa, sb = match_result(rng, team_strengths[ia], team_strengths[ib])
                points[ia] += sa
                points[ib] += sb
                gf_a = rng.poisson(0.5 + team_strengths[ia] * 0.3)
                gf_b = rng.poisson(0.5 + team_strengths[ib] * 0.3)
                goals_for[ia] += gf_a
                goals_for[ib] += gf_b

        ranked = sorted(indices, key=lambda i: (points[i], goals_for[i], rng.random()),
                        reverse=True)
        group_rankings.append(ranked)
    return group_rankings


def compute_double_rr_metrics(rng, groups, team_strengths, n_sim):
    """双循环小组赛 + 单场淘汰"""
    strength_rank = stats.rankdata([-s for s in team_strengths])
    spearmans = []
    top32_rates = []
    champion_top1 = 0
    champion_top4 = 0
    champion_top8 = 0

    for _ in range(n_sim):
        group_rankings = simulate_double_round_robin_group(rng, groups, team_strengths)
        qualified = []
        for ranking in group_rankings:
            qualified.extend(ranking[:2])

        top32_actual = sum(1 for i in range(len(team_strengths))
                          if strength_rank[i] <= 32 and i in qualified)
        top32_rates.append(top32_actual / 32)

        champion, final_ranking = simulate_knockout(rng, qualified, team_strengths)
        if strength_rank[champion] <= 1:
            champion_top1 += 1
        if strength_rank[champion] <= 4:
            champion_top4 += 1
        if strength_rank[champion] <= 8:
            champion_top8 += 1

        group_losers = []
        for ranking in group_rankings:
            group_losers.extend(ranking[2:])
        group_losers.sort(key=lambda i: team_strengths[i], reverse=True)
        full_ranking = final_ranking + group_losers
        pos = list(range(1, 65))
        actual_strength_ranks = [strength_rank[i] for i in full_ranking]
        rho, _ = stats.spearmanr(pos, actual_strength_ranks)
        spearmans.append(rho)

    return {
        "spearman": np.array(spearmans),
        "top32_rate": np.array(top32_rates),
        "champion_top1": champion_top1 / n_sim,
        "champion_top4": champion_top4 / n_sim,
        "champion_top8": champion_top8 / n_sim,
    }


def compute_randomized_strength_metrics(
    groups,
    n_sim,
    group_stage_func,
    sigma=LAYER_SIGMA,
    strength_seed=2026,
    match_seed=4200,
):
    """每次赛事先抽取层内正态实力, 再模拟小组赛和淘汰赛。"""
    rng_strength = np.random.default_rng(strength_seed)
    rng_match = np.random.default_rng(match_seed)

    spearmans = []
    top32_rates = []
    champion_top1 = 0
    champion_top4 = 0
    champion_top8 = 0

    for _ in range(n_sim):
        team_strengths = draw_layered_strengths(rng_strength, sigma=sigma)
        strength_rank = stats.rankdata([-s for s in team_strengths], method="ordinal")

        group_rankings = group_stage_func(rng_match, groups, team_strengths)
        qualified = []
        for ranking in group_rankings:
            qualified.extend(ranking[:2])

        top32_actual = sum(1 for i in range(len(team_strengths))
                          if strength_rank[i] <= 32 and i in qualified)
        top32_rates.append(top32_actual / 32)

        champion, final_ranking = simulate_knockout(rng_match, qualified, team_strengths)
        if strength_rank[champion] <= 1:
            champion_top1 += 1
        if strength_rank[champion] <= 4:
            champion_top4 += 1
        if strength_rank[champion] <= 8:
            champion_top8 += 1

        group_losers = []
        for ranking in group_rankings:
            group_losers.extend(ranking[2:])
        group_losers.sort(key=lambda i: team_strengths[i], reverse=True)
        full_ranking = final_ranking + group_losers

        pos = list(range(1, 65))
        actual_strength_ranks = [strength_rank[i] for i in full_ranking]
        rho, _ = stats.spearmanr(pos, actual_strength_ranks)
        spearmans.append(rho)

    return {
        "spearman": np.array(spearmans),
        "top32_rate": np.array(top32_rates),
        "champion_top1": champion_top1 / n_sim,
        "champion_top4": champion_top4 / n_sim,
        "champion_top8": champion_top8 / n_sim,
        "sigma": sigma,
    }


# ============================================================
# 4. 输出
# ============================================================

def print_metrics(m, label, n_sim):
    print(f"\n{'=' * 64}")
    print(f"  {label} ({n_sim} 次模拟)")
    print(f"{'=' * 64}")
    print(f"  Spearman 相关系数 (↑越接近1越公平):")
    print(f"    均值={np.mean(m['spearman']):.4f}  中位={np.median(m['spearman']):.4f}")
    print(f"    5%/95% 分位: {np.percentile(m['spearman'], 5):.4f} / "
          f"{np.percentile(m['spearman'], 95):.4f}")
    print(f"  晋级准确率 (实力前32中实际晋级比例, ↑):")
    print(f"    均值={np.mean(m['top32_rate']):.4f}  "
          f"5%分位={np.percentile(m['top32_rate'], 5):.4f}")
    print(f"  冠军实力分布:")
    print(f"    P(实力第1名夺冠)={m['champion_top1']:.4f}")
    print(f"    P(实力前4名夺冠)={m['champion_top4']:.4f}")
    print(f"    P(实力前8名夺冠)={m['champion_top8']:.4f}")


def print_comparison_table(title, rows):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    print(f"  {'赛制':<22} {'Spearman':<10} {'晋级率':<10} "
          f"{'Top1夺冠':<10} {'Top4夺冠':<10} {'Top8夺冠':<10}")
    print(f"  {'-' * 70}")
    for label, m in rows:
        print(f"  {label:<22} {np.mean(m['spearman']):<10.4f} "
              f"{np.mean(m['top32_rate']):<10.4f} "
              f"{m['champion_top1']:<10.4f} {m['champion_top4']:<10.4f} "
              f"{m['champion_top8']:<10.4f}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 64)
    print("  浙超赛制评价 — 问题4: 赛制合理化建议")
    print("=" * 64)

    # 分组方案: 使用问题1中的ILP均衡方案D
    groups = BALANCED_GROUPS
    c1_ok = len(check_c1(groups)) == 0
    c2_ok = len(check_c2(groups)) == 0
    c3_ok = check_c3(groups)[0] == 0
    print(f"\n  分组方案: Q1-ILP均衡方案D "
          f"C1={'通过' if c1_ok else '失败'} "
          f"C2={'通过' if c2_ok else '失败'} "
          f"C3={'通过' if c3_ok else '失败'}")

    # 实力赋值: 市级=3, 县级市=2, 县=1
    team_strengths = [t.strength for t in TEAMS]
    print(f"\n  实力分布: 市级(3) × {sum(1 for t in TEAMS if t.level=='municipal')} + "
          f"县级市(2) × {sum(1 for t in TEAMS if t.level=='city')} + "
          f"县(1) × {sum(1 for t in TEAMS if t.level=='county')} = {len(TEAMS)} 队")

    n_sim = 5000
    rng = np.random.default_rng(42)

    # --- 当前赛制: 单循环小组赛 + 单场淘汰 ---
    print(f"\n  模拟中... (当前赛制, {n_sim} 次)")
    m_current = compute_metrics(rng, groups, team_strengths, n_sim)
    print_metrics(m_current, "当前赛制: 单循环小组赛 + 单场淘汰", n_sim)

    # --- 替代赛制: 双循环小组赛 + 单场淘汰 ---
    rng2 = np.random.default_rng(42)
    print(f"\n  模拟中... (双循环小组赛, {n_sim} 次)")
    m_double = compute_double_rr_metrics(rng2, groups, team_strengths, n_sim)

    print_metrics(m_double, "双循环小组赛 + 单场淘汰", n_sim)
    print_comparison_table(
        "固定层级实力模型对比 (3/2/1)",
        [("单循环+淘汰", m_current), ("双循环+淘汰", m_double)],
    )

    # --- 层内正态实力扰动: s_i ~ N(level, (1/3)^2) ---
    print(f"\n  模拟中... (层内正态实力扰动 σ={LAYER_SIGMA:.3f}, {n_sim} 次)")
    m_rand_current = compute_randomized_strength_metrics(
        groups, n_sim, simulate_group_stage,
        sigma=LAYER_SIGMA, strength_seed=2026, match_seed=4200,
    )
    m_rand_double = compute_randomized_strength_metrics(
        groups, n_sim, simulate_double_round_robin_group,
        sigma=LAYER_SIGMA, strength_seed=2026, match_seed=4200,
    )
    print_metrics(m_rand_current, "随机实力: 单循环小组赛 + 单场淘汰", n_sim)
    print_metrics(m_rand_double, "随机实力: 双循环小组赛 + 单场淘汰", n_sim)
    print_comparison_table(
        "层内正态实力模型对比 (σ=1/3)",
        [("单循环+淘汰", m_rand_current), ("双循环+淘汰", m_rand_double)],
    )

    # --- 合理化建议 ---
    print(f"\n{'=' * 64}")
    print(f"  赛制合理化建议")
    print(f"{'=' * 64}")
    print()
    print("  1. 当前赛制 (16组×4队单循环 + 32强5轮淘汰):")
    print(f"     - 公平性: Spearman 均值 {np.mean(m_current['spearman']):.3f},")
    print(f"       实力前32名晋级率 {np.mean(m_current['top32_rate']):.1%}")
    print(f"     - 在随机实力模型下: Top-1/Top-4/Top-8夺冠概率分别为 "
          f"{m_rand_current['champion_top1']:.1%}, "
          f"{m_rand_current['champion_top4']:.1%}, "
          f"{m_rand_current['champion_top8']:.1%}")
    print()
    print("  2. 建议改进:")
    print("     a) 小组赛改为双循环: Spearman 从 "
          f"{np.mean(m_current['spearman']):.3f} 提升至 "
          f"{np.mean(m_double['spearman']):.3f}, "
          f"晋级准确率从 {np.mean(m_current['top32_rate']):.1%} 提升至 "
          f"{np.mean(m_double['top32_rate']):.1%}")
    print("        随机实力模型下, Spearman 从 "
          f"{np.mean(m_rand_current['spearman']):.3f} 提升至 "
          f"{np.mean(m_rand_double['spearman']):.3f}, "
          f"晋级准确率从 {np.mean(m_rand_current['top32_rate']):.1%} 提升至 "
          f"{np.mean(m_rand_double['top32_rate']):.1%}")
    print("     b) 淘汰赛可考虑两回合制 (主客场), 进一步降低偶然性")
    print("     c) 小组出线名额: 当前前2名晋级 (50%) 较合理;")
    print("        若改为仅第1名晋级, 虽更精准但小组赛悬念不足")
    print("     d) 建议在淘汰赛阶段引入种子排名:")
    print("        小组第1名 vs 其他组第2名, 避免过早强强对话")
    print()
    print("  3. 定量依据:")
    print("     - 单循环仅 6 场/组, 样本小, 晋级偶然性高")
    print("     - 双循环 12 场/组, 样本翻倍, 更能反映真实实力")
    print("     - 淘汰赛 31 场均为单场定胜负, 偶然性不可避免")
    print("       (这是赛制设计的选择: 公平性 vs 观赏性 vs 赛程长度)")


if __name__ == "__main__":
    main()
