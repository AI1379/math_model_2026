import random
from collections import defaultdict

# 浙江省行政区划数据（保持不变）
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

def one_draw(verbose=False):
    """
    执行一次抽签，返回：
    - success (bool): 是否成功（硬约束是否满足）
    - c3_violation_groups (int): 存在C3冲突的小组数量
    - c3_violation_pairs (int): C3冲突对数（∑ C(n,2)）
    - max_same_city (int): 单一小组内同市县级队最大数量
    """
    city_teams = list(admin_divisions.keys())
    
    # 准备县级队数据
    undrawn_county_teams_by_city = defaultdict(list)
    for city, counties in admin_divisions.items():
        for county in counties:
            undrawn_county_teams_by_city[city].append({"name": county, "admin_city": city})

    groups = [[] for _ in range(16)]
    group_forbidden_admin_cities = [set() for _ in range(16)]
    group_county_admin_counts = [defaultdict(int) for _ in range(16)]

    # 阶段1：市级队随机分配到11个不同小组
    shuffled_city_teams = random.sample(city_teams, len(city_teams))
    available_indices = list(range(16))
    chosen_groups = random.sample(available_indices, 11)
    for i, city_team in enumerate(shuffled_city_teams):
        group_idx = chosen_groups[i]
        groups[group_idx].append(city_team)
        group_forbidden_admin_cities[group_idx].add(city_team)

    # 阶段2：县级队按大市优先贪心分配
    total_county = sum(len(counties) for counties in admin_divisions.values())
    placed = 0

    while placed < total_county:
        # 找出当前剩余县级队最多的市
        cities_with_remaining = {
            city: len(teams) for city, teams in undrawn_county_teams_by_city.items() if teams
        }
        if not cities_with_remaining:
            break
        max_count = max(cities_with_remaining.values())
        max_cities = [c for c, cnt in cities_with_remaining.items() if cnt == max_count]
        chosen_city = random.choice(max_cities)
        county_list = undrawn_county_teams_by_city.pop(chosen_city)

        for county_obj in county_list:
            county_name = county_obj["name"]
            admin_city = county_obj["admin_city"]

            eligible = []
            for g_idx in range(16):
                if len(groups[g_idx]) < 4:
                    if admin_city not in group_forbidden_admin_cities[g_idx]:
                        c3_score = group_county_admin_counts[g_idx][admin_city]
                        eligible.append((c3_score, g_idx))
            if not eligible:
                # 硬约束违反
                return False, 0, 0, 0

            eligible.sort(key=lambda x: x[0])
            min_score = eligible[0][0]
            best = [g for s, g in eligible if s == min_score]
            chosen_group = random.choice(best)

            groups[chosen_group].append(county_name)
            group_county_admin_counts[chosen_group][admin_city] += 1
            placed += 1

    # 统计C3冲突
    c3_group_count = 0      # 至少有一个冲突的小组数
    c3_pair_count = 0       # 总冲突对数 ∑ C(n,2)
    max_same = 0
    for i in range(16):
        cnt_dict = group_county_admin_counts[i]
        for n in cnt_dict.values():
            if n >= 2:
                c3_pair_count += n * (n - 1) // 2
                if n > max_same:
                    max_same = n
        if any(n >= 2 for n in cnt_dict.values()):
            c3_group_count += 1

    return True, c3_group_count, c3_pair_count, max_same

# ---------- 主程序：运行10000次 ----------
num_simulations = 10000
hard_failures = 0
c3_group_counts = []   # 存储成功抽签的 C3冲突小组数
c3_pair_counts = []    # 存储成功抽签的 C3冲突对数

for sim in range(1, num_simulations + 1):
    success, c3_g, c3_p, max_s = one_draw(verbose=False)
    if not success:
        hard_failures += 1
    else:
        c3_group_counts.append(c3_g)
        c3_pair_counts.append(c3_p)

# 计算统计量
successful_runs = num_simulations - hard_failures
c3_zero_count = sum(1 for x in c3_pair_counts if x == 0)
c3_nonzero_count = successful_runs - c3_zero_count

avg_c3_groups = sum(c3_group_counts) / successful_runs if successful_runs > 0 else 0
avg_c3_pairs = sum(c3_pair_counts) / successful_runs if successful_runs > 0 else 0

print("========== 10000次蒙特卡洛模拟统计结果 ==========")
print(f"总模拟次数：{num_simulations}")
print(f"硬约束违反次数：{hard_failures}（频率：{hard_failures/num_simulations:.2%}）")
print(f"成功分配次数：{successful_runs}（频率：{successful_runs/num_simulations:.2%}）")
print(f"其中，C3完全满足（冲突对数为0）的次数：{c3_zero_count}（频率：{c3_zero_count/num_simulations:.2%}）")
print(f"其中，C3有违反的次数：{c3_nonzero_count}（频率：{c3_nonzero_count/num_simulations:.2%}）")
print(f"\n--- 在成功分配中 ---")
print(f"平均C3冲突小组数：{avg_c3_groups:.2f}")
print(f"平均C3冲突对数：{avg_c3_pairs:.2f}")