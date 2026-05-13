"""
队友抽签方案整合脚本
====================
整合自三个原始文件:
  - a初始算法.py   → 单次抽签 (模式A)
  - b频率统计算法.py → 蒙特卡洛频率统计 (模式B)
  - c多次抽取算法.py → 反复重试直到C3零违反 (模式C)

核心思路:
  1. 随机将11支市级队分配到11个不同小组 (保证C1)
  2. 按各市"剩余县级队数量"从多到少, 逐市处理全部县级队
  3. 每支县级队贪心选择C3冲突分最低的可用组
  4. 若遇到死锁(无可用组)则报错/重试; 若C3有违反也重试

运行方式:
  python teammate_draw_combined.py --mode A          # 单次抽签
  python teammate_draw_combined.py --mode B -n 1000  # 蒙特卡洛统计
  python teammate_draw_combined.py --mode C -n 100   # 重试直到C3=0
"""

import random
import argparse
import copy
from collections import defaultdict

# ============================================================
# 浙江省行政区划数据
# ============================================================

ADMIN_DIVISIONS = {
    "杭州市": ["建德市", "桐庐县", "淳安县"],
    "宁波市": ["余姚市", "慈溪市", "象山县", "宁海县"],
    "温州市": ["瑞安市", "乐清市", "龙港市", "永嘉县", "平阳县", "苍南县", "文成县", "泰顺县"],
    "嘉兴市": ["海宁市", "平湖市", "桐乡市", "嘉善县", "海盐县"],
    "湖州市": ["德清县", "长兴县", "安吉县"],
    "绍兴市": ["诸暨市", "嵊州市", "新昌县"],
    "金华市": ["兰溪市", "义乌市", "东阳市", "永康市", "武义县", "浦江县", "磐安县"],
    "衢州市": ["江山市", "常山县", "开化县", "龙游县"],
    "舟山市": ["岱山县", "嵊泗县"],
    "台州市": ["温岭市", "临海市", "玉环市", "三门县", "天台县", "仙居县"],
    "丽水市": ["龙泉市", "青田县", "缙云县", "遂昌县", "松阳县", "云和县", "庆元县", "景宁畲族自治县"],
}


# ============================================================
# 核心抽签函数 (来自 a初始算法, 略作重构)
# ============================================================

def single_draw(divisions_data, verbose=False):
    """
    单次抽签, 返回 (成功, groups, c3_violations, max_same_city, admin_counts).

    流程:
      1. 随机抽取11支市级队 → G1..G11
      2. 按剩余县级队数降序, 选出当前最多的市, 取出其全部县级队
      3. 每支县级队贪心选C3冲突最低的可用组 (非C2禁入 + 有空位)
      4. 若无可用组 → 死锁, 返回失败

    参数:
      divisions_data: 行政区划字典
      verbose: 是否打印详细过程

    返回:
      (success, groups, c3_violations_count, max_same_city_in_group, group_admin_counts)
    """
    city_teams = list(divisions_data.keys())

    undrawn_by_city = defaultdict(list)
    for city, counties in divisions_data.items():
        for county in counties:
            undrawn_by_city[city].append({"name": county, "admin_city": city})

    groups = [[] for _ in range(16)]
    forbidden = [set() for _ in range(16)]       # C2: 各组禁入的市
    admin_counts = [defaultdict(int) for _ in range(16)]  # C3: 各组各市县级队计数

    # --- 阶段1: 市级队 → 前11组 ---
    shuffled_cities = random.sample(city_teams, len(city_teams))
    mun_group = {}
    for i, city in enumerate(shuffled_cities):
        groups[i].append(city)
        forbidden[i].add(city)
        mun_group[city] = i
        if verbose:
            print(f"  {city} → G{i + 1}")

    # --- 阶段2: 逐市贪心分配县级队 ---
    total_county = sum(len(cs) for cs in divisions_data.values())
    placed = 0

    while placed < total_county:
        remaining = {c: len(ts) for c, ts in undrawn_by_city.items() if ts}
        if not remaining:
            break

        # 选剩余最多的市
        max_n = max(remaining.values())
        candidates = [c for c, n in remaining.items() if n == max_n]
        chosen = random.choice(candidates)
        batch = undrawn_by_city.pop(chosen)

        if verbose:
            print(f"\n  --- 处理 {chosen} 的 {len(batch)} 支县级队 ---")

        for obj in batch:
            name = obj["name"]
            admin = obj["admin_city"]

            eligible = []
            for g in range(16):
                if len(groups[g]) < 4 and admin not in forbidden[g]:
                    c3_score = admin_counts[g][admin]
                    eligible.append((c3_score, g))

            if not eligible:
                # 死锁: 无符合C1/C2的空位
                return False, groups, -1, -1, admin_counts

            eligible.sort()
            min_score = eligible[0][0]
            best = [g for s, g in eligible if s == min_score]
            g = random.choice(best)

            groups[g].append(name)
            admin_counts[g][admin] += 1
            placed += 1
            if verbose:
                print(f"    {name}({admin}) → G{g + 1}  C3冲突={min_score}")

    # --- 统计C3 ---
    c3_violations = 0
    max_same = 0
    for i in range(16):
        if any(cnt > 1 for cnt in admin_counts[i].values()):
            c3_violations += 1
        if admin_counts[i]:
            max_same = max(max_same, max(admin_counts[i].values()))

    return True, groups, c3_violations, max_same, admin_counts


# ============================================================
# 模式A: 单次抽签 (原 a初始算法)
# ============================================================

def mode_a():
    print("=" * 60)
    print("  模式A: 单次抽签")
    print("=" * 60)

    success, groups, c3v, max_same, admin_counts = single_draw(
        ADMIN_DIVISIONS, verbose=True
    )

    if not success:
        print("\n  [失败] 抽签过程遭遇死锁, 无法完成分配!")
        return

    print(f"\n--- 分组结果 ---")
    for i, grp in enumerate(groups):
        info = admin_counts[i]
        conflicts = {c: n for c, n in info.items() if n > 1}
        tag = ""
        if conflicts:
            tag = f" (C3冲突: {', '.join(f'{c}:{n}支' for c, n in conflicts.items())})"
        print(f"  G{i + 1:2d} ({len(grp)}队): {', '.join(grp)}{tag}")

    print(f"\n--- 统计 ---")
    print(f"  C3违反小组数: {c3v}")
    print(f"  单组最大同市县级队数: {max_same}")

    # 验证
    all_teams = set(t for g in groups for t in g)
    orig = set(ADMIN_DIVISIONS.keys())
    for cs in ADMIN_DIVISIONS.values():
        orig.update(cs)
    print(f"  队伍总数: {len(all_teams)}/{len(orig)}")
    print(f"  每组恰好4队: {all(len(g) == 4 for g in groups)}")


# ============================================================
# 模式B: 蒙特卡洛频率统计 (原 b频率统计算法)
# ============================================================

def mode_b(n_sim):
    print("=" * 60)
    print(f"  模式B: 蒙特卡洛模拟 ({n_sim}次)")
    print("=" * 60)

    hard_fails = 0
    c3_zero = 0
    c3_nonzero = 0
    all_c3v = []
    all_max_same = []

    for i in range(n_sim):
        success, _, c3v, max_same, _ = single_draw(
            copy.deepcopy(ADMIN_DIVISIONS), verbose=False
        )
        if not success:
            hard_fails += 1
        else:
            if c3v == 0:
                c3_zero += 1
            else:
                c3_nonzero += 1
            all_c3v.append(c3v)
            all_max_same.append(max_same)

    total_ok = c3_zero + c3_nonzero
    print(f"\n--- 统计结果 ---")
    print(f"  总模拟次数:       {n_sim}")
    print(f"  硬约束死锁次数:   {hard_fails} ({hard_fails / n_sim:.2%})")
    print(f"  成功分配次数:     {total_ok} ({total_ok / n_sim:.2%})")
    print(f"    C3零违反次数:   {c3_zero} ({c3_zero / n_sim:.2%})")
    print(f"    C3有违反次数:   {c3_nonzero} ({c3_nonzero / n_sim:.2%})")

    if total_ok > 0:
        print(f"  平均C3违反小组数: {sum(all_c3v) / total_ok:.2f}")
        print(f"  平均最大同市数:   {sum(all_max_same) / total_ok:.2f}")

    # 重试概率分析
    if total_ok > 0:
        p_success = c3_zero / n_sim
        print(f"\n--- 重试概率分析 ---")
        print(f"  单次C3零违反概率 P = {p_success:.4f}")
        for k in [3, 4, 5, 10]:
            p_k = 1 - (1 - p_success) ** k
            print(f"  {k}次重试后至少一次C3=0的概率: {p_k:.4f}")


# ============================================================
# 模式C: 反复重试直到C3=0 (原 c多次抽取算法)
# ============================================================

def mode_c(max_attempts):
    print("=" * 60)
    print(f"  模式C: 重试直到C3零违反 (最多{max_attempts}次)")
    print("=" * 60)

    for attempt in range(1, max_attempts + 1):
        success, groups, c3v, max_same, admin_counts = single_draw(
            copy.deepcopy(ADMIN_DIVISIONS), verbose=False
        )

        if not success:
            print(f"  第{attempt}次: 死锁, 重试...")
            continue

        if c3v == 0:
            print(f"  第{attempt}次: 成功! C3零违反")
            print(f"\n--- 分组方案 (第{attempt}次找到) ---")
            for i, grp in enumerate(groups):
                info = admin_counts[i]
                conflicts = {c: n for c, n in info.items() if n > 1}
                tag = ""
                if conflicts:
                    tag = f" (C3冲突: {', '.join(f'{c}:{n}支' for c, n in conflicts.items())})"
                print(f"  G{i + 1:2d} ({len(grp)}队): {', '.join(grp)}{tag}")

            print(f"\n--- 统计 ---")
            print(f"  尝试次数: {attempt}")
            print(f"  C3违反小组数: {c3v}")
            print(f"  单组最大同市县级队数: {max_same}")

            all_teams = set(t for g in groups for t in g)
            orig = set(ADMIN_DIVISIONS.keys())
            for cs in ADMIN_DIVISIONS.values():
                orig.update(cs)
            print(f"  队伍总数: {len(all_teams)}/{len(orig)}")
            print(f"  每组恰好4队: {all(len(g) == 4 for g in groups)}")
            return

        else:
            print(f"  第{attempt}次: C3违反{c3v}组, 重试...")

    print(f"\n  [失败] {max_attempts}次内未找到C3零违反方案")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="队友抽签方案整合脚本")
    parser.add_argument(
        "--mode", choices=["A", "B", "C"], default="A",
        help="A=单次抽签, B=蒙特卡洛统计, C=重试直到C3=0"
    )
    parser.add_argument("-n", type=int, default=1000, help="模式B/C的模拟/重试次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.mode == "A":
        mode_a()
    elif args.mode == "B":
        mode_b(args.n)
    elif args.mode == "C":
        mode_c(args.n)


if __name__ == "__main__":
    main()
