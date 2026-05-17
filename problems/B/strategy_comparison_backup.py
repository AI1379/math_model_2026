"""
策略对比实验：修复(repair) vs 重启(restart) 的期望优劣分析
============================================================

实验设计:
  - 用同一批随机种子 (0..9999) 驱动三种策略:
    1. greedy_no_repair: 贪心, 死锁即失败
    2. greedy_with_repair: 贪心 + _repair_swap 修复
    3. restart: 贪心, 死锁或C3>0则重抽直到C3=0 (模拟队友方案)
  - 对比各策略在 F1/F2/F2'/F3 上的条件分布和期望

理论框架:
  - 二部图模型: 县级队 ↔ 可用组槽位
  - Hall 婚配定理保证完美匹配存在
  - 两种策略对应解空间上的不同采样分布
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
import time
from collections import defaultdict
from q1_grouping import (
    CITY_DATA,
    TEAMS,
    TEAM_INDEX,
    NUM_GROUPS,
    GROUP_SIZE,
    metric_f1,
    metric_f2,
    metric_f3,
)


def greedy_algorithm(rng, repair=False):
    """贪心 + 修复, 理论上永远不失败"""
    groups = [[] for _ in range(NUM_GROUPS)]
    flag = False  # 是否已经修复过一次了

    cities_shuffled = list(CITY_DATA.keys())
    rng.shuffle(cities_shuffled)
    slots = np.arange(NUM_GROUPS)
    rng.shuffle(slots)

    mun_group = {}
    for i, city in enumerate(cities_shuffled):
        g = int(slots[i])
        groups[g].append(CITY_DATA[city]["municipal"])
        mun_group[city] = g

    cities_desc = sorted(
        CITY_DATA.keys(),
        key=lambda c: len(CITY_DATA[c]["county_teams"]),
        reverse=True,
    )

    for city in cities_desc:
        forbidden = mun_group[city]
        teams = list(CITY_DATA[city]["county_teams"])
        rng.shuffle(teams)

        for name, _level in teams:
            candidates = []
            for j in range(NUM_GROUPS):
                if j == forbidden or len(groups[j]) >= GROUP_SIZE:
                    continue
                has_c3 = any(
                    TEAMS[TEAM_INDEX[n]].city == city
                    and TEAMS[TEAM_INDEX[n]].level != "municipal"
                    for n in groups[j]
                )
                candidates.append((j, has_c3, len(groups[j])))

            if not candidates and repair:
                repaired = _repair_swap(groups, city, forbidden, mun_group)
                flag = True
                if repaired is not None:
                    j_swap, freed_j = repaired
                    groups[freed_j].append(name)
                    continue
                raise RuntimeError(f"Dead end (even after repair): {name} from {city}")
            elif not candidates:
                return None, flag

            candidates.sort(key=lambda x: (x[1], x[2]))
            best_size = candidates[0][2]
            best_c3 = candidates[0][1]
            top = [j for j, c3, sz in candidates if c3 == best_c3 and sz == best_size]
            g = int(rng.choice(top))
            groups[g].append(name)

    return groups, flag


# ============================================================
# 策略1: 贪心无修复 (基线)
# ============================================================


def greedy_no_repair(rng):
    """贪心, 死锁返回 None"""
    return greedy_algorithm(rng, repair=False)[0]


# ============================================================
# 策略2: 贪心 + 修复 (我们的方案)
# ============================================================


def _repair_swap(groups, city, forbidden, mun_group):
    if len(groups[forbidden]) >= GROUP_SIZE:
        return None
    for j in range(NUM_GROUPS):
        if j == forbidden or len(groups[j]) < GROUP_SIZE:
            continue
        for idx, other_name in enumerate(groups[j]):
            other = TEAMS[TEAM_INDEX[other_name]]
            if other.city == city or other.level == "municipal":
                continue
            other_forbidden = mun_group[other.city]
            if other_forbidden == forbidden:
                continue
            groups[j].pop(idx)
            groups[forbidden].append(other_name)
            return (j, j)
    return None


def greedy_with_repair(rng):
    """贪心 + 修复, 理论上永远不失败"""
    try:
        return greedy_algorithm(rng, repair=True)
    except RuntimeError:
        return None, False


# ============================================================
# 策略3: 重启 (队友方案模拟)
# ============================================================


def restart_until_good(rng, max_retries=50):
    """
    反复贪心重抽, 直到 C3=0 或超过重试上限.
    用同一 rng 的不同子状态模拟独立重抽.
    返回 (groups, retries_used).
    """
    for attempt in range(max_retries):
        # 每次重试用 rng 衍生一个新种子, 确保独立性
        child_seed = rng.integers(0, 2**31)
        child_rng = np.random.default_rng(child_seed)

        groups = greedy_no_repair(child_rng)
        if groups is None:
            continue  # 死锁, 重试
        if metric_f1(groups) == 0:
            return groups, attempt + 1
    return None, max_retries  # 未找到


# ============================================================
# 实验主体
# ============================================================


def run_experiment(n_seeds=10000):
    print("=" * 72)
    print("  策略对比实验: 修复(repair) vs 重启(restart)")
    print(f"  {n_seeds} 个随机种子")
    print("=" * 72)

    results = {
        "no_repair": {
            "f1": [],
            "f2": [],
            "f2r": [],
            "f3": [],
            "n": 0,
            "deadlocks": 0,
            "total_time": 0,
        },
        "with_repair": {
            "f1": [],
            "f2": [],
            "f2r": [],
            "f3": [],
            "n": 0,
            "deadlocks": 0,
            "repaired": 0,
            "fixed_deadlocks": 0,
            "total_time": 0,
        },
        "restart": {
            "f1": [],
            "f2": [],
            "f2r": [],
            "f3": [],
            "n": 0,
            "deadlocks": 0,
            "not_found": 0,
            "retries": [],
            "total_time": 0,
        },
    }

    for sd in range(n_seeds):
        seed = random.randint(0, 2**31 - 1)  # 生成一个随机种子
        rng = np.random.default_rng(seed)

        # --- 策略1: 无修复 ---
        t1 = time.perf_counter()
        g1 = greedy_no_repair(rng)
        t2 = time.perf_counter()
        results["no_repair"]["total_time"] += t2 - t1
        if g1 is None:
            results["no_repair"]["deadlocks"] += 1
        else:
            results["no_repair"]["n"] += 1
            results["no_repair"]["f1"].append(metric_f1(g1))
            strengths = [sum(TEAMS[TEAM_INDEX[n]].strength for n in grp) for grp in g1]
            results["no_repair"]["f2"].append(float(np.std(strengths)))
            results["no_repair"]["f2r"].append(max(strengths) - min(strengths))
            results["no_repair"]["f3"].append(metric_f3(g1))

        # --- 策略2: 有修复 ---
        rng2 = np.random.default_rng(seed)  # 同一种子
        t1 = time.perf_counter()
        g2, flag = greedy_with_repair(rng2)
        t2 = time.perf_counter()
        results["with_repair"]["total_time"] += t2 - t1
        if g2 is not None:
            results["with_repair"]["n"] += 1
            f1_r = metric_f1(g2)
            results["with_repair"]["f1"].append(f1_r)
            strengths = [sum(TEAMS[TEAM_INDEX[n]].strength for n in grp) for grp in g2]
            results["with_repair"]["f2"].append(float(np.std(strengths)))
            results["with_repair"]["f2r"].append(max(strengths) - min(strengths))
            results["with_repair"]["f3"].append(metric_f3(g2))

            # 检查是否经过了修复 (与无修复结果不同)
            results["with_repair"]["fixed_deadlocks"] += int(flag)
            if g1 is not None:
                # 比较分组是否相同
                s1 = {tuple(sorted(grp)) for grp in g1}
                s2 = {tuple(sorted(grp)) for grp in g2}
                if s1 != s2 and f1_r > 0:
                    results["with_repair"]["repaired"] += 1

        # --- 策略3: 重启 ---
        rng3 = np.random.default_rng(seed)
        t1 = time.perf_counter()
        g3, retries = restart_until_good(rng3)
        t2 = time.perf_counter()
        results["restart"]["total_time"] += t2 - t1
        if g3 is None:
            results["restart"]["not_found"] += 1
        else:
            results["restart"]["n"] += 1
            results["restart"]["f1"].append(metric_f1(g3))
            strengths = [sum(TEAMS[TEAM_INDEX[n]].strength for n in grp) for grp in g3]
            results["restart"]["f2"].append(float(np.std(strengths)))
            results["restart"]["f2r"].append(max(strengths) - min(strengths))
            results["restart"]["f3"].append(metric_f3(g3))
            results["restart"]["retries"].append(retries)

    # ============================================================
    # 输出结果
    # ============================================================

    for name, label in [
        ("no_repair", "策略1: 贪心无修复"),
        ("with_repair", "策略2: 贪心+修复"),
        ("restart", "策略3: 重启至C3=0"),
    ]:
        r = results[name]
        print(f"\n{'─' * 72}")
        print(f"  {label}")
        print(f"{'─' * 72}")
        print(f"  成功次数: {r['n']}/{n_seeds}")
        print(f"  总时间: {r['total_time']:.4f} 秒")
        if name == "no_repair":
            print(f"  死锁次数: {r['deadlocks']} ({r['deadlocks'] / n_seeds:.2%})")
        if name == "with_repair":
            print(f"  修复后C3>0的次数: {r['repaired']}")
            print(f"  修复了死锁的次数: {r['fixed_deadlocks']}")
        if name == "restart":
            print(f"  未找到C3=0的次数: {r['not_found']}")
            if r["retries"]:
                print(f"  平均重试次数: {np.mean(r['retries']):.2f}")

        if r["f1"]:
            arr_f1 = np.array(r["f1"])
            arr_f2 = np.array(r["f2"])
            arr_f2r = np.array(r["f2r"])
            arr_f3 = np.array(r["f3"])

            print(f"\n  F1 (C3冲突对数, ↓=好):")
            print(
                f"    均值={np.mean(arr_f1):.4f}  P(F1=0)={np.mean(arr_f1 == 0):.4f}  max={int(np.max(arr_f1))}"
            )

            print(f"  F2 (实力标准差, ↓=好):")
            print(f"    均值={np.mean(arr_f2):.4f}  中位={np.median(arr_f2):.4f}")
            print(
                f"    P(F2≤0.5)={np.mean(arr_f2 <= 0.5):.4f}  P(F2≤1.0)={np.mean(arr_f2 <= 1.0):.4f}"
            )

            print(f"  F2'(实力极差, ↓=好):")
            print(
                f"    均值={np.mean(arr_f2r):.2f}  P(极差≤2)={np.mean(arr_f2r <= 2):.4f}"
            )

            print(f"  F3 (多样性熵, ↑=好):")
            print(f"    均值={np.mean(arr_f3):.4f}")

    # ============================================================
    # 关键对比: 同一种子下 repair vs restart 的 F2/F3 差异
    # ============================================================
    print(f"\n{'=' * 72}")
    print(f"  核心问题: 修复是否系统性劣化 F2/F3?")
    print(f"{'=' * 72}")

    # 找出 repair 的 F1>0 的那些 case (即修复引入了 C3 违反)
    repair_f1 = np.array(results["with_repair"]["f1"])
    repair_f2 = np.array(results["with_repair"]["f2"])
    restart_f2 = np.array(results["restart"]["f2"])

    f1_zero_mask = repair_f1 == 0
    f1_pos_mask = repair_f1 > 0

    n_zero = int(f1_zero_mask.sum())
    n_pos = int(f1_pos_mask.sum())

    print(f"\n  修复策略中:")
    print(f"    C3=0 的次数: {n_zero} ({n_zero / len(repair_f1):.2%})")
    print(f"    C3>0 的次数: {n_pos} ({n_pos / len(repair_f1):.2%})")

    if n_pos > 0:
        print(
            f"\n    C3>0 (被修复过) 时的 F2 均值: {repair_f2[f1_pos_mask].mean():.4f}"
        )
        print(f"    C3=0 (未被修复) 时的 F2 均值: {repair_f2[f1_zero_mask].mean():.4f}")
        print(f"    重启策略 (始终C3=0)  的 F2 均值: {restart_f2.mean():.4f}")

    print(f"\n  全样本 F2 均值对比:")
    print(f"    修复策略: {repair_f2.mean():.4f}")
    print(f"    重启策略: {restart_f2.mean():.4f}")

    repair_f3 = np.array(results["with_repair"]["f3"])
    restart_f3 = np.array(results["restart"]["f3"])

    print(f"\n  全样本 F3 均值对比:")
    print(f"    修复策略: {repair_f3.mean():.4f}")
    print(f"    重启策略: {restart_f3.mean():.4f}")


# ============================================================
# 边际均匀性检验
# ============================================================


def marginal_uniformity_test(n_seeds=10000):
    """
    边际均匀性检验：验证 repair 抽签是否对每支队伍公平

    对每支队伍 i，统计 N 次抽签中落入每个组的频次 count[i][j]。
    零假设 H₀: P(队伍i在组j) = 1/16, ∀j

    理论依据：
      市级队: P(在组j) = 11/16 × 1/11 = 1/16
      县级队: P(在组j) ≈ 15/16 × 1/15 = 1/16
              (C2禁入组随每次抽签随机变化，无条件边际近似均匀)
    """
    from scipy.stats import chisquare, kstest

    team_names = list(TEAM_INDEX.keys())
    n_teams = len(team_names)
    counts = np.zeros((n_teams, NUM_GROUPS), dtype=int)
    n_success = 0

    for sd in range(n_seeds):
        rng = np.random.default_rng(sd)
        groups, _ = greedy_with_repair(rng)
        if groups is None:
            continue
        n_success += 1
        for g_idx, group in enumerate(groups):
            for name in group:
                counts[TEAM_INDEX[name], g_idx] += 1

    expected = n_success / NUM_GROUPS

    # 逐队 χ² 检验
    results = []
    for name in team_names:
        i = TEAM_INDEX[name]
        team = TEAMS[i]
        stat, p_val = chisquare(counts[i], f_exp=expected)
        results.append(
            {
                "name": name,
                "city": team.city,
                "level": team.level,
                "chi2": stat,
                "p_value": p_val,
                "pass": p_val >= 0.05,
            }
        )

    # === 输出 ===
    print(f"\n{'=' * 72}")
    print("  边际均匀性检验 (Marginal Uniformity Test)")
    print(f"  {n_seeds} 次蒙特卡洛模拟, {n_success} 次成功")
    print("  H0: P(team_i in group_j) = 1/16, for all j")
    print("  Test: Pearson chi-squared goodness-of-fit, df=15")
    print(f"{'=' * 72}")

    n_pass = sum(r["pass"] for r in results)
    n_total = len(results)
    alpha_bonf = 0.05 / n_total
    n_pass_bonf = sum(r["p_value"] >= alpha_bonf for r in results)
    all_p = [r["p_value"] for r in results]

    print(
        f"\n  Pass chi-sq (a=0.05):          {n_pass}/{n_total} ({n_pass / n_total:.1%})"
    )
    print(
        f"  Pass Bonferroni (a={alpha_bonf:.4f}): {n_pass_bonf}/{n_total} ({n_pass_bonf / n_total:.1%})"
    )
    print(f"  平均 p 值:                       {np.mean(all_p):.4f}")
    print(f"  p 值中位数:                      {np.median(all_p):.4f}")

    level_map = {"municipal": "市级", "county_city": "县级市", "county": "县"}
    print("\n  按层级:")
    for lv, label in level_map.items():
        lr = [r for r in results if r["level"] == lv]
        if not lr:
            continue
        lp = sum(r["pass"] for r in lr)
        mean_p = np.mean([r["p_value"] for r in lr])
        print(f"    {label} ({len(lr)}队): 通过 {lp}/{len(lr)}, 平均p={mean_p:.4f}")

    failing = [r for r in results if not r["pass"]]
    if failing:
        print("\n  Failed at a=0.05:")
        for r in sorted(failing, key=lambda x: x["p_value"]):
            print(
                f"    {r['name']} ({r['city']}) - chi2={r['chi2']:.2f}, p={r['p_value']:.4f}"
            )
    else:
        print("\n  All teams passed chi-squared test")

    ks_stat, ks_p = kstest(all_p, "uniform")
    print(f"\n  p-value uniformity (KS test): KS={ks_stat:.4f}, p={ks_p:.4f}")
    if ks_p >= 0.05:
        print("    -> p-values follow Uniform(0,1), consistent with H0")
    else:
        print("    -> p-values deviate from uniform, further investigation needed")

    return results


if __name__ == "__main__":
    run_experiment(n_seeds=10000)
    marginal_uniformity_test(n_seeds=10000)
