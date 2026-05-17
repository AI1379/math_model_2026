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

def implement_drawing_scheme_strict_4_teams(divisions_data):
    """
    根据抽签方案生成分组结果并统计C3软约束违反情况。
    严格执行每个小组最多4支队伍的硬约束。

    方案描述:
    1. 随机抽取11个市级队，将其随机分配到16个小组中的11个不同小组（每个小组最多一个市级队）。
    2. 对于县级队，按照所属市下辖县级队数量从多到少的原则进行抽取。
       如果多个市有相同数量的县级队，则随机选择一个市。
    3. 从选中的市中，抽取其所有剩余的县级队，逐一分配到小组。
    4. 分配县级队时，必须满足C1和C2硬约束。
       C1: 每个小组最多4支队伍。
       C2: 市级队伍不能与该市所辖县级队伍同组。
    5. 尽量满足C3软约束：优先选择C3冲突分（即小组中已有同市县级队数量）最低的小组。
       如果存在多个C3冲突分最低的小组，随机选择一个。

    返回:
    - groups: 最终的分组方案 (list of lists)
    - c3_violations_count: 违反C3软约束的小组数量（即存在同市县级队同组的小组个数）
    - max_same_city_in_group: 单一小组内来自同一市的县级队最大数量
    - group_county_admin_counts: 最终的C3软约束统计数据
    """

    city_teams = list(divisions_data.keys())
    
    # 准备县级队数据，按所属市分组
    undrawn_county_teams_by_city = defaultdict(list)
    for city, counties in divisions_data.items():
        for county in counties:
            undrawn_county_teams_by_city[city].append({"name": county, "admin_city": city})

    # 初始化16个小组
    groups = [[] for _ in range(16)]
    # C2硬约束记录：每个小组禁止哪些市的县级队进入
    group_forbidden_admin_cities = [set() for _ in range(16)]
    # C3软约束记录：每个小组中来自各市的县级队数量
    group_county_admin_counts = [defaultdict(int) for _ in range(16)]

    print("--- 阶段1: 抽取市级队伍 ---")
    # 1. 抽取11个市级队，并随机分配到16个小组中的11个不同小组
    shuffled_city_teams = random.sample(city_teams, len(city_teams))  # 随机打乱市级队顺序
    # 从0~15中随机选择11个不重复的小组索引
    available_group_indices = list(range(16))
    chosen_groups = random.sample(available_group_indices, 11)  # 随机选11个不同的组
    # 将市级队依次放入这些随机选中的组
    for i, city_team in enumerate(shuffled_city_teams):
        group_idx = chosen_groups[i]
        groups[group_idx].append(city_team)
        # 记录C2限制：该市所辖县级队不能进入此小组
        group_forbidden_admin_cities[group_idx].add(city_team)
        print(f"将 {city_team} 抽入小组 G{group_idx+1}")

    print("\n--- 阶段2: 抽取县级队伍 ---")
    
    # 获取所有县级队的总数
    total_county_teams_to_place = sum(len(counties) for counties in divisions_data.values())
    placed_county_teams_count = 0

    # 主要循环：按照“大市优先”策略处理县级队
    while placed_county_teams_count < total_county_teams_to_place:
        # 找出当前剩余县级队数量最多的市
        cities_with_remaining_counties = {
            city: len(teams) for city, teams in undrawn_county_teams_by_city.items() if teams
        }
        
        if not cities_with_remaining_counties:
            # 理论上所有县级队都已分配，但以防万一
            break

        # 选择当前县级队最多的市（同数量随机选一个）
        max_count = max(cities_with_remaining_counties.values())
        max_cities = [city for city, count in cities_with_remaining_counties.items() if count == max_count]
        chosen_city = random.choice(max_cities)
        
        # 从该市的待分配列表中取出所有县级队，并从undrawn_county_teams_by_city中移除该市
        county_teams_to_process_this_round = undrawn_county_teams_by_city.pop(chosen_city)
        print(f"\n--- 优先抽取 {chosen_city} 下的 {len(county_teams_to_process_this_round)} 支县级队伍 ---")

        # 将本轮选中的县级队逐一分配
        for county_team_obj in county_teams_to_process_this_round:
            county_name = county_team_obj["name"]
            admin_city = county_team_obj["admin_city"]

            eligible_groups_with_score = []  # 存储 (c3_score, group_idx)
            for g_idx in range(16):
                # 严格检查小组是否有空位 (最多4支队伍)
                if len(groups[g_idx]) < 4:
                    # C2校验：该小组不能有此县级队所属市的市级队
                    if admin_city not in group_forbidden_admin_cities[g_idx]:
                        # C3评分：小组中已有多少同市县级队
                        c3_score = group_county_admin_counts[g_idx][admin_city]
                        eligible_groups_with_score.append((c3_score, g_idx))
            
            if not eligible_groups_with_score:
                # 发生死锁，理论上在合理数据下不应出现，此处报错以便调试
                raise Exception(f"【错误】无法为县级队伍 {county_name} 找到符合C1/C2约束的空位！")
            
            # 优先选择C3冲突分最低的小组
            eligible_groups_with_score.sort(key=lambda x: x[0])  # 按c3_score升序
            min_score = eligible_groups_with_score[0][0]
            # 找出所有达到最低分数的组，从中随机选择一个
            best_groups = [g_idx for score, g_idx in eligible_groups_with_score if score == min_score]
            chosen_group_idx = random.choice(best_groups)

            # 放置队伍
            groups[chosen_group_idx].append(county_name)
            group_county_admin_counts[chosen_group_idx][admin_city] += 1
            placed_county_teams_count += 1
            
            print(f"  将 {county_name} (隶属 {admin_city}) 抽入小组 G{chosen_group_idx+1} (C3冲突分: {min_score})")

    # 3. 统计违反C3软约束的数量
    c3_violations_count = 0
    max_same_city_in_group = 0
    for i, group in enumerate(groups):
        if any(count > 1 for count in group_county_admin_counts[i].values()):
            c3_violations_count += 1
        if group_county_admin_counts[i]:
            max_same_city_in_group = max(max_same_city_in_group, max(group_county_admin_counts[i].values()))
    
    return groups, c3_violations_count, max_same_city_in_group, group_county_admin_counts

# 执行一次抽签方案
final_groups, c3_violations, max_same_city, final_group_county_admin_counts = implement_drawing_scheme_strict_4_teams(admin_divisions)

print("\n--- 最终分组方案 ---")
for i, group in enumerate(final_groups):
    group_c3_info = final_group_county_admin_counts[i]
    violations_in_this_group = {city: count for city, count in group_c3_info.items() if count > 1}
    c3_status = ""
    if violations_in_this_group:
        c3_status = f" (C3冲突: {', '.join(f'{city}:{count}支' for city, count in violations_in_this_group.items())})"
    print(f"小组 G{i+1} ({len(group)}队): {', '.join(group)}{c3_status}")

print(f"\n--- 统计结果 ---")
print(f"违反C3软约束的小组数量: {c3_violations} 个")
print(f"单一小组内来自同一市的县级队最大数量: {max_same_city} 支")

# 验证所有队伍是否都被分配，以及小组是否都满足4队硬约束
all_teams_in_groups = set()
for group in final_groups:
    for team in group:
        all_teams_in_groups.add(team)

original_city_teams = set(admin_divisions.keys())
original_county_teams = set()
for counties in admin_divisions.values():
    original_county_teams.update(counties)

total_original_teams = len(original_city_teams) + len(original_county_teams)

print(f"\n--- 额外验证 ---")
print(f"原始队伍总数: {total_original_teams}")
print(f"分配到小组的队伍总数: {len(all_teams_in_groups)}")
print(f"所有小组都恰好有4支队伍: {all(len(g) == 4 for g in final_groups)}")