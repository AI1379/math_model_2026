import random
from collections import defaultdict

# 浙江省行政区划数据
admin_divisions = {
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
    "丽水市": ["龙泉市", "青田县", "缙云县", "遂昌县", "松阳县", "云和县", "庆元县", "景宁畲族自治县"]
}

# 建立县级队 -> 所属市 映射
county_to_city = {}
for city, counties in admin_divisions.items():
    for county in counties:
        county_to_city[county] = city

city_teams = list(admin_divisions.keys())

def compute_c3_pair(groups, county_to_city):
    """计算总C3冲突对数 ∑ C(n,2)"""
    total = 0
    for g in groups:
        cnt = defaultdict(int)
        for t in g:
            if t in county_to_city:
                cnt[county_to_city[t]] += 1
        for n in cnt.values():
            if n >= 2:
                total += n * (n - 1) // 2
    return total

def compute_c3_group_info(groups, county_to_city):
    """返回每个小组的C3冲突字典，以及冲突小组数"""
    group_cnts = []
    c3_group = 0
    for g in groups:
        cnt = defaultdict(int)
        for t in g:
            if t in county_to_city:
                cnt[county_to_city[t]] += 1
        group_cnts.append(cnt)
        if any(v >= 2 for v in cnt.values()):
            c3_group += 1
    return group_cnts, c3_group

def greedy_repair_c3(groups, group_forbidden, county_to_city, city_teams, max_iter=500, verbose=False):
    """
    贪婪下降修复C3冲突：反复寻找能减少总冲突对数的合法交换，直到无法改进
    返回：(groups, improved, final_pair)
    """
    if verbose:
        print("\n--- 贪婪下降修复C3冲突 ---")
    old_pair = compute_c3_pair(groups, county_to_city)
    if verbose:
        print(f"修复前C3冲突对数: {old_pair}")
    improved = False
    iteration = 0
    while iteration < max_iter:
        current_pair = compute_c3_pair(groups, county_to_city)
        if current_pair == 0:
            break
        found = False
        for g1 in range(16):
            for idx1, t1 in enumerate(groups[g1]):
                if t1 not in county_to_city:  # 只考虑县级队
                    continue
                city1 = county_to_city[t1]
                for g2 in range(16):
                    if g1 == g2:
                        continue
                    for idx2, t2 in enumerate(groups[g2]):
                        # 检查C2：t1不能进入有city1市级队的组；t2不能进入有city2市级队的组
                        city2 = county_to_city[t2] if t2 in county_to_city else None
                        if city1 in group_forbidden[g2]:
                            continue
                        if city2 is not None and city2 in group_forbidden[g1]:
                            continue
                        # C1检查：交换后每个小组市级队数量不超过1
                        other_cities_g1 = [t for t in groups[g1] if t in city_teams and t != t1]
                        other_cities_g2 = [t for t in groups[g2] if t in city_teams and t != t2]
                        if (t2 in city_teams and len(other_cities_g1) > 0) or (t1 in city_teams and len(other_cities_g2) > 0):
                            continue
                        # 模拟交换
                        groups[g1][idx1], groups[g2][idx2] = t2, t1
                        new_pair = compute_c3_pair(groups, county_to_city)
                        if new_pair < current_pair:
                            # 接受交换
                            if verbose:
                                print(f"  交换: G{g1+1}中的 {t1} 与 G{g2+1}中的 {t2}，冲突从 {current_pair} 降到 {new_pair}")
                            # 更新禁止集合（市级队可能移动）
                            for i in range(16):
                                group_forbidden[i] = {t for t in groups[i] if t in city_teams}
                            found = True
                            improved = True
                            break
                        else:
                            # 恢复
                            groups[g1][idx1], groups[g2][idx2] = t1, t2
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
        iteration += 1
    final_pair = compute_c3_pair(groups, county_to_city)
    if verbose:
        print(f"修复后C3冲突对数: {final_pair} (减少 {old_pair - final_pair})")
    return groups, improved, final_pair

def one_draw_with_full_repair(verbose=True):
    """
    一次完整的抽签：市级队随机分配 -> 县级队贪心分配（带死锁修复） -> 贪婪下降优化C3
    返回：(success, groups, final_c3_pair, final_c3_group)
    若verbose=True则打印详细过程
    """
    # 初始化
    undrawn_county_teams_by_city = defaultdict(list)
    for city, counties in admin_divisions.items():
        for county in counties:
            undrawn_county_teams_by_city[city].append({"name": county, "admin_city": city})

    groups = [[] for _ in range(16)]
    group_forbidden = [set() for _ in range(16)]

    # 阶段1：市级队随机分配到11个不同小组
    if verbose:
        print("=== 阶段1: 抽取市级队伍 ===")
    shuffled_city = random.sample(city_teams, len(city_teams))
    chosen_groups = random.sample(range(16), 11)
    for i, city in enumerate(shuffled_city):
        g = chosen_groups[i]
        groups[g].append(city)
        group_forbidden[g].add(city)
        if verbose:
            print(f"将 {city} 抽入小组 G{g+1}")

    total_county = sum(len(v) for v in admin_divisions.values())
    placed = 0

    if verbose:
        print("\n=== 阶段2: 抽取县级队伍 (大市优先, C3最低优先) ===")

    while placed < total_county:
        # 找到当前剩余县级队最多的市
        cities_with = {c: len(teams) for c, teams in undrawn_county_teams_by_city.items() if teams}
        if not cities_with:
            break
        max_cnt = max(cities_with.values())
        max_cities = [c for c, cnt in cities_with.items() if cnt == max_cnt]
        chosen_city = random.choice(max_cities)
        county_list = undrawn_county_teams_by_city.pop(chosen_city)
        if verbose:
            print(f"\n--- 优先抽取 {chosen_city} 下的 {len(county_list)} 支县级队伍 ---")
        for county_obj in county_list:
            name = county_obj["name"]
            admin = county_obj["admin_city"]
            # 找出所有合法候选组
            eligible = []
            for g in range(16):
                if len(groups[g]) < 4 and admin not in group_forbidden[g]:
                    # 计算当前小组中已有多少同市县级队
                    score = sum(1 for t in groups[g] if t in county_to_city and county_to_city[t] == admin)
                    eligible.append((score, g))
            if not eligible:
                # 死锁：尝试交换修复
                if verbose:
                    print(f"  死锁! 尝试修复 {name}...")
                repaired = False
                # 遍历所有小组，寻找可以交换的队友
                for g in range(16):
                    if len(groups[g]) < 4:
                        continue
                    for idx, other in enumerate(groups[g]):
                        if other not in county_to_city:
                            continue
                        other_city = county_to_city[other]
                        # 检查将name放入g是否合法
                        if admin in group_forbidden[g]:
                            continue
                        # 寻找other的合法新家（有空位且不违反C2）
                        for g2 in range(16):
                            if g2 == g:
                                continue
                            if len(groups[g2]) < 4 and other_city not in group_forbidden[g2]:
                                # 执行交换
                                groups[g].pop(idx)
                                groups[g].append(name)
                                groups[g2].append(other)
                                # 更新禁止集合
                                if other in city_teams:
                                    group_forbidden[g2].add(other_city)
                                    group_forbidden[g].discard(other_city)
                                if admin not in group_forbidden[g]:
                                    group_forbidden[g].add(admin)
                                if verbose:
                                    print(f"  修复成功: 将 {name} 放入 G{g+1}, 将 {other} 移至 G{g2+1}")
                                repaired = True
                                placed += 1
                                break
                        if repaired:
                            break
                    if repaired:
                        break
                if not repaired:
                    if verbose:
                        print(f"  修复失败，本次抽签失败")
                    return False, None, None, None
                continue  # 当前队伍已处理，继续下一个

            # 正常情况：选择C3冲突分最低的组
            eligible.sort(key=lambda x: x[0])
            min_score = eligible[0][0]
            best_groups = [g for s, g in eligible if s == min_score]
            chosen_g = random.choice(best_groups)
            groups[chosen_g].append(name)
            if verbose:
                print(f"  将 {name} (隶属 {admin}) 抽入小组 G{chosen_g+1} (C3冲突分: {min_score})")
            placed += 1

    # 阶段3：贪婪下降修复C3冲突
    if verbose:
        print("\n=== 阶段3: 贪婪下降优化C3冲突 ===")
    groups, improved, final_c3_pair = greedy_repair_c3(groups, group_forbidden, county_to_city, city_teams, verbose=verbose)

    # 最终统计
    group_cnts, final_c3_group = compute_c3_group_info(groups, county_to_city)
    max_same = 0
    for cnt in group_cnts:
        if cnt:
            max_same = max(max_same, max(cnt.values()))

    if verbose:
        print("\n=== 最终分组方案 ===")
        for i, group in enumerate(groups):
            cnt_dict = group_cnts[i]
            violations = {city: c for city, c in cnt_dict.items() if c > 1}
            c3_status = ""
            if violations:
                c3_status = f" (C3冲突: {', '.join(f'{city}:{c}支' for city, c in violations.items())})"
            print(f"小组 G{i+1} ({len(group)}队): {', '.join(group)}{c3_status}")
        print(f"\n最终统计：")
        print(f"  违反C3的小组数: {final_c3_group} 个")
        print(f"  C3冲突对数: {final_c3_pair}")
        print(f"  单一小组同市县级队最大数量: {max_same}")

    return True, groups, final_c3_pair, final_c3_group

if __name__ == "__main__":
    # 运行一次完整抽签
    success, groups, pair, group_cnt = one_draw_with_full_repair(verbose=True)
    if not success:
        print("抽签失败")