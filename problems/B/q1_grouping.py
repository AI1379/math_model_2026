#!/usr/bin/env python3
"""
Q1: 浙超分组方案设计与比较
=============================
- 方案A: 贪心优先填充（启发式）
- 方案B: 蛇形轮转分配（启发式）
- 方案C: 整数规划最优（ILP）
- 方案D: ILP兼顾实力均衡（多目标）

评价指标: F1(C3违反数) F2(实力标准差) F3(来源多样性熵)
"""

import numpy as np
from itertools import combinations
from collections import defaultdict
from dataclasses import dataclass

# ============================================================
# 1. DATA
# ============================================================

NUM_GROUPS = 16
GROUP_SIZE = 4
STRENGTH_MAP = {"municipal": 3, "city": 2, "county": 1}

CITY_DATA = {
    "杭州": {
        "municipal": "杭州市",
        "county_teams": [
            ("建德市", "city"),
            ("桐庐县", "county"),
            ("淳安县", "county"),
        ],
    },
    "宁波": {
        "municipal": "宁波市",
        "county_teams": [
            ("余姚市", "city"),
            ("慈溪市", "city"),
            ("象山县", "county"),
            ("宁海县", "county"),
        ],
    },
    "温州": {
        "municipal": "温州市",
        "county_teams": [
            ("瑞安市", "city"),
            ("乐清市", "city"),
            ("龙港市", "city"),
            ("永嘉县", "county"),
            ("平阳县", "county"),
            ("苍南县", "county"),
            ("文成县", "county"),
            ("泰顺县", "county"),
        ],
    },
    "嘉兴": {
        "municipal": "嘉兴市",
        "county_teams": [
            ("海宁市", "city"),
            ("平湖市", "city"),
            ("桐乡市", "city"),
            ("嘉善县", "county"),
            ("海盐县", "county"),
        ],
    },
    "湖州": {
        "municipal": "湖州市",
        "county_teams": [
            ("德清县", "county"),
            ("长兴县", "county"),
            ("安吉县", "county"),
        ],
    },
    "绍兴": {
        "municipal": "绍兴市",
        "county_teams": [
            ("诸暨市", "city"),
            ("嵊州市", "city"),
            ("新昌县", "county"),
        ],
    },
    "金华": {
        "municipal": "金华市",
        "county_teams": [
            ("兰溪市", "city"),
            ("义乌市", "city"),
            ("东阳市", "city"),
            ("永康市", "city"),
            ("武义县", "county"),
            ("浦江县", "county"),
            ("磐安县", "county"),
        ],
    },
    "衢州": {
        "municipal": "衢州市",
        "county_teams": [
            ("江山市", "city"),
            ("常山县", "county"),
            ("开化县", "county"),
            ("龙游县", "county"),
        ],
    },
    "舟山": {
        "municipal": "舟山市",
        "county_teams": [
            ("岱山县", "county"),
            ("嵊泗县", "county"),
        ],
    },
    "台州": {
        "municipal": "台州市",
        "county_teams": [
            ("温岭市", "city"),
            ("临海市", "city"),
            ("玉环市", "city"),
            ("三门县", "county"),
            ("天台县", "county"),
            ("仙居县", "county"),
        ],
    },
    "丽水": {
        "municipal": "丽水市",
        "county_teams": [
            ("龙泉市", "city"),
            ("青田县", "county"),
            ("缙云县", "county"),
            ("遂昌县", "county"),
            ("松阳县", "county"),
            ("云和县", "county"),
            ("庆元县", "county"),
            ("景宁县", "county"),
        ],
    },
}


@dataclass
class Team:
    name: str
    city: str
    level: str  # municipal / city / county
    strength: int


def build_teams():
    teams = []
    for city, data in CITY_DATA.items():
        teams.append(Team(data["municipal"], city, "municipal", 3))
        for name, level in data["county_teams"]:
            teams.append(Team(name, city, level, STRENGTH_MAP[level]))
    return teams


TEAMS = build_teams()
TEAM_INDEX = {t.name: i for i, t in enumerate(TEAMS)}

LEVEL_TAG = {"municipal": "市", "city": "县市级", "county": "县"}


# ============================================================
# 2. CONSTRAINT CHECKING
# ============================================================


def check_c1(groups):
    """C1: 各市级队分配在不同小组"""
    violations = []
    for j, group in enumerate(groups):
        munis = [n for n in group if TEAMS[TEAM_INDEX[n]].level == "municipal"]
        if len(munis) > 1:
            violations.append((j + 1, munis))
    return violations


def check_c2(groups):
    """C2: 市级队不与同市县级队同组"""
    violations = []
    for j, group in enumerate(groups):
        members = [TEAMS[TEAM_INDEX[n]] for n in group]
        for t in members:
            if t.level == "municipal":
                same_city = [o.name for o in members if o.city == t.city and o.name != t.name]
                if same_city:
                    violations.append((j + 1, t.name, same_city))
    return violations


def check_c3(groups):
    """C3: 同市县级队尽量不在同一组 → 返回 (违反对数, 详情)"""
    total = 0
    details = []
    for j, group in enumerate(groups):
        members = [TEAMS[TEAM_INDEX[n]] for n in group]
        county = [t for t in members if t.level != "municipal"]
        by_city = defaultdict(list)
        for t in county:
            by_city[t.city].append(t.name)
        for city, names in by_city.items():
            if len(names) > 1:
                pairs = len(names) * (len(names) - 1) // 2
                total += pairs
                details.append((j + 1, city, names))
    return total, details


# ============================================================
# 3. EVALUATION METRICS
# ============================================================


def metric_f1(groups):
    """F1: 同市县级队同组的配对数 (C(n,2) per group per city)"""
    total = 0
    for group in groups:
        members = [TEAMS[TEAM_INDEX[n]] for n in group]
        county = [t for t in members if t.level != "municipal"]
        by_city = defaultdict(int)
        for t in county:
            by_city[t.city] += 1
        for c in by_city.values():
            total += c * (c - 1) // 2
    return total


def metric_f2(groups):
    """F2: 各组实力总和的标准差"""
    strengths = [sum(TEAMS[TEAM_INDEX[n]].strength for n in g) for g in groups]
    return float(np.std(strengths))


def metric_f2_range(groups):
    """F2': 各组实力极差"""
    strengths = [sum(TEAMS[TEAM_INDEX[n]].strength for n in g) for g in groups]
    return max(strengths) - min(strengths)


def metric_f3(groups):
    """F3: 组内城市来源多样性的平均 Shannon 熵"""
    entropies = []
    for group in groups:
        members = [TEAMS[TEAM_INDEX[n]] for n in group]
        by_city = defaultdict(int)
        for t in members:
            by_city[t.city] += 1
        total = len(group)
        h = -sum((c / total) * np.log(c / total) for c in by_city.values() if c > 0)
        entropies.append(h)
    return float(np.mean(entropies))


def evaluate(groups, label):
    """返回指标字典并打印详情"""
    c1v = check_c1(groups)
    c2v = check_c2(groups)
    c3n, c3d = check_c3(groups)

    f1 = metric_f1(groups)
    f2 = metric_f2(groups)
    f2r = metric_f2_range(groups)
    f3 = metric_f3(groups)

    strengths = [sum(TEAMS[TEAM_INDEX[n]].strength for n in g) for g in groups]

    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")

    for j, group in enumerate(groups):
        parts = []
        for n in group:
            t = TEAMS[TEAM_INDEX[n]]
            parts.append(f"{n}({LEVEL_TAG[t.level]})")
        print(f"  组{j + 1:2d} [实力={strengths[j]:2d}] {', '.join(parts)}")

    print(f"\n  约束检查:")
    print(f"    C1: {'通过' if not c1v else '违反 ' + str(c1v)}")
    print(f"    C2: {'通过' if not c2v else '违反 ' + str(c2v)}")
    print(f"    C3: {'通过' if c3n == 0 else f'{c3n} 对冲突'}")
    for g, city, names in c3d:
        print(f"         组{g} {city}: {names}")

    print(f"\n  评价指标:")
    print(f"    F1  C3冲突对数 (↓) : {f1}")
    print(f"    F2  实力标准差  (↓) : {f2:.4f}")
    print(f"    F2' 实力极差    (↓) : {f2r}")
    print(f"    F3  多样性熵    (↑) : {f3:.4f}")

    return {"label": label, "F1": f1, "F2": f2, "F2_range": f2r, "F3": f3}


# ============================================================
# 4. SCHEME A — GREEDY PRIORITY FILL
# ============================================================


def scheme_a_greedy(seed=42):
    """按县级队数量从多到少处理各市, 每支县级队优先放入容量最充裕且不违反约束的组"""
    rng = np.random.RandomState(seed)
    groups = [[] for _ in range(NUM_GROUPS)]

    # 市级队 → 组 1-11
    cities_desc = sorted(CITY_DATA, key=lambda c: len(CITY_DATA[c]["county_teams"]), reverse=True)
    mun_group = {}
    for i, city in enumerate(cities_desc):
        groups[i].append(CITY_DATA[city]["municipal"])
        mun_group[city] = i

    # 县级队贪心填充
    for city in cities_desc:
        forbidden = mun_group[city]
        teams = list(CITY_DATA[city]["county_teams"])
        rng.shuffle(teams)
        used_groups = set()  # 本市已用组 (C3)

        for name, _level in teams:
            candidates = []
            for j in range(NUM_GROUPS):
                if j == forbidden or len(groups[j]) >= GROUP_SIZE:
                    continue
                # C3: 该组已有本市县级队则降优先
                has_same = any(
                    TEAMS[TEAM_INDEX[n]].city == city and TEAMS[TEAM_INDEX[n]].level != "municipal"
                    for n in groups[j]
                )
                candidates.append((j, has_same, len(groups[j])))

            # 优先: 无C3冲突 → 人最少
            candidates.sort(key=lambda x: (x[1], x[2]))
            best = candidates[0][0]
            groups[best].append(name)

    return groups


# ============================================================
# 5. SCHEME B — SERPENTINE ROUND-ROBIN
# ============================================================


def scheme_b_serpentine(seed=42):
    """蛇形轮转: 市级队入组后, 县级队按蛇形序号依次分配到可用组"""
    rng = np.random.RandomState(seed)
    groups = [[] for _ in range(NUM_GROUPS)]

    cities_desc = sorted(CITY_DATA, key=lambda c: len(CITY_DATA[c]["county_teams"]), reverse=True)
    mun_group = {}
    for i, city in enumerate(cities_desc):
        groups[i].append(CITY_DATA[city]["municipal"])
        mun_group[city] = i

    # 全局蛇形序列 (1,2,...,16,16,...,2,1,1,2,...)
    serp = list(range(NUM_GROUPS))
    for _ in range(10):
        serp += serp[-NUM_GROUPS:][::-1]

    pos = 0  # 蛇形指针
    for city in cities_desc:
        forbidden = mun_group[city]
        teams = list(CITY_DATA[city]["county_teams"])
        rng.shuffle(teams)
        used = set()

        for name, _level in teams:
            placed = False
            # 沿蛇形序列找可用组
            for _ in range(NUM_GROUPS):
                g = serp[pos % len(serp)]
                pos += 1
                if g == forbidden or len(groups[g]) >= GROUP_SIZE:
                    continue
                if g in used:
                    continue  # C3
                groups[g].append(name)
                used.add(g)
                placed = True
                break

            if not placed:
                # 放松 C3
                for g in range(NUM_GROUPS):
                    if g != forbidden and len(groups[g]) < GROUP_SIZE and g not in used:
                        groups[g].append(name)
                        used.add(g)
                        placed = True
                        break
                if not placed:
                    for g in range(NUM_GROUPS):
                        if g != forbidden and len(groups[g]) < GROUP_SIZE:
                            groups[g].append(name)
                            placed = True
                            break

    return groups


# ============================================================
# 6. SCHEME C — ILP (MINIMIZE C3 VIOLATIONS)
# ============================================================


def scheme_c_ilp():
    """整数线性规划: 硬约束 C1+C2, 目标最小化 C3 冲突对数"""
    from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary, LpStatus, value, PULP_CBC_CMD

    N = len(TEAMS)
    prob = LpProblem("ZheChao_C3opt", LpMinimize)

    x = {(i, j): LpVariable(f"x_{i}_{j}", cat=LpBinary) for i in range(N) for j in range(NUM_GROUPS)}

    # 同市县级队配对
    pairs = []
    for city, data in CITY_DATA.items():
        idxs = [TEAM_INDEX[n] for n, _ in data["county_teams"]]
        for a, b in combinations(idxs, 2):
            pairs.append((a, b))

    y = {(p, j): LpVariable(f"y_{p}_{j}", cat=LpBinary) for p in range(len(pairs)) for j in range(NUM_GROUPS)}

    # --- 约束 ---
    # 每支球队恰在一组
    for i in range(N):
        prob += lpSum(x[i, j] for j in range(NUM_GROUPS)) == 1

    # 每组恰好4队
    for j in range(NUM_GROUPS):
        prob += lpSum(x[i, j] for i in range(N)) == GROUP_SIZE

    # C1: 每组至多1支市级队
    mun_idx = [i for i, t in enumerate(TEAMS) if t.level == "municipal"]
    for j in range(NUM_GROUPS):
        prob += lpSum(x[i, j] for i in mun_idx) <= 1

    # C2: 市级队与同市县级队不同组
    for city, data in CITY_DATA.items():
        mi = TEAM_INDEX[data["municipal"]]
        for name, _ in data["county_teams"]:
            ki = TEAM_INDEX[name]
            for j in range(NUM_GROUPS):
                prob += x[mi, j] + x[ki, j] <= 1

    # y 链接
    for p, (a, b) in enumerate(pairs):
        for j in range(NUM_GROUPS):
            prob += y[p, j] >= x[a, j] + x[b, j] - 1

    # 目标: 最小化 C3 冲突
    prob += lpSum(y[p, j] for p in range(len(pairs)) for j in range(NUM_GROUPS))

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=120))

    if LpStatus[prob.status] != "Optimal":
        print(f"  [ILP] 状态: {LpStatus[prob.status]}")
        return None

    groups = [[] for _ in range(NUM_GROUPS)]
    for i in range(N):
        for j in range(NUM_GROUPS):
            if value(x[i, j]) > 0.5:
                groups[j].append(TEAMS[i].name)
                break
    return groups


# ============================================================
# 7. SCHEME D — ILP + STRENGTH BALANCE (MULTI-OBJECTIVE)
# ============================================================


def scheme_d_ilp_balanced(w_c3=10, w_str=1):
    """ILP多目标: min w_c3·C3违反 + w_str·(S_max - S_min)"""
    from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary, LpStatus, value, PULP_CBC_CMD

    N = len(TEAMS)
    prob = LpProblem("ZheChao_Balanced", LpMinimize)

    x = {(i, j): LpVariable(f"x_{i}_{j}", cat=LpBinary) for i in range(N) for j in range(NUM_GROUPS)}

    pairs = []
    for city, data in CITY_DATA.items():
        idxs = [TEAM_INDEX[n] for n, _ in data["county_teams"]]
        for a, b in combinations(idxs, 2):
            pairs.append((a, b))

    y = {(p, j): LpVariable(f"y_{p}_{j}", cat=LpBinary) for p in range(len(pairs)) for j in range(NUM_GROUPS)}

    s_max = LpVariable("S_max")
    s_min = LpVariable("S_min")

    # 约束 (同 Scheme C)
    for i in range(N):
        prob += lpSum(x[i, j] for j in range(NUM_GROUPS)) == 1
    for j in range(NUM_GROUPS):
        prob += lpSum(x[i, j] for i in range(N)) == GROUP_SIZE
    mun_idx = [i for i, t in enumerate(TEAMS) if t.level == "municipal"]
    for j in range(NUM_GROUPS):
        prob += lpSum(x[i, j] for i in mun_idx) <= 1
    for city, data in CITY_DATA.items():
        mi = TEAM_INDEX[data["municipal"]]
        for name, _ in data["county_teams"]:
            ki = TEAM_INDEX[name]
            for j in range(NUM_GROUPS):
                prob += x[mi, j] + x[ki, j] <= 1
    for p, (a, b) in enumerate(pairs):
        for j in range(NUM_GROUPS):
            prob += y[p, j] >= x[a, j] + x[b, j] - 1

    # 实力范围约束
    for j in range(NUM_GROUPS):
        gs = lpSum(TEAMS[i].strength * x[i, j] for i in range(N))
        prob += s_max >= gs
        prob += s_min <= gs

    prob += w_c3 * lpSum(y[p, j] for p in range(len(pairs)) for j in range(NUM_GROUPS)) + w_str * (s_max - s_min)

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=120))

    if LpStatus[prob.status] != "Optimal":
        print(f"  [ILP] 状态: {LpStatus[prob.status]}")
        return None

    groups = [[] for _ in range(NUM_GROUPS)]
    for i in range(N):
        for j in range(NUM_GROUPS):
            if value(x[i, j]) > 0.5:
                groups[j].append(TEAMS[i].name)
                break
    return groups


# ============================================================
# 8. COMPARISON TABLE
# ============================================================


def print_comparison(results):
    print(f"\n{'=' * 72}")
    print(f"  方案对比汇总")
    print(f"{'=' * 72}")
    header = f"  {'方案':<28} {'F1(C3,↓)':<10} {'F2(σ,↓)':<10} {'F2(极差,↓)':<12} {'F3(熵,↑)':<10}"
    print(header)
    print(f"  {'-' * 70}")
    for r in results:
        print(
            f"  {r['label']:<28} {r['F1']:<10} "
            f"{r['F2']:<10.4f} {r['F2_range']:<12} {r['F3']:<10.4f}"
        )

    # 排名 (各指标分别排名, 然后综合)
    print(f"\n  各指标排名 (1=最优):")
    for key, direction, nice in [("F1", 1, "C3违反"), ("F2", 1, "实力σ"), ("F3", -1, "多样性熵")]:
        sorted_r = sorted(results, key=lambda r: r[key] * direction)
        print(f"    {nice}: ", end="")
        print(" > ".join(f"{r['label'][:6]}({i + 1})" for i, r in enumerate(sorted_r)))


# ============================================================
# MAIN
# ============================================================


def main():
    n_mun = sum(1 for t in TEAMS if t.level == "municipal")
    n_cc = sum(1 for t in TEAMS if t.level == "city")
    n_co = sum(1 for t in TEAMS if t.level == "county")
    total_strength = sum(t.strength for t in TEAMS)

    print("=" * 72)
    print("  浙超分组方案 — 问题1: 分组方案设计与比较")
    print(f"  共 {len(TEAMS)} 支球队 | {n_mun} 市级 + {n_cc} 县级市 + {n_co} 县")
    print(f"  {NUM_GROUPS} 组 × {GROUP_SIZE} 队 | 总实力 {total_strength} | 组均 {total_strength / NUM_GROUPS:.2f}")
    print("=" * 72)

    results = []

    groups_a = scheme_a_greedy(seed=42)
    results.append(evaluate(groups_a, "方案A: 贪心优先填充"))

    groups_b = scheme_b_serpentine(seed=42)
    results.append(evaluate(groups_b, "方案B: 蛇形轮转分配"))

    groups_c = scheme_c_ilp()
    if groups_c is not None:
        results.append(evaluate(groups_c, "方案C: ILP(C3最小)"))

    groups_d = scheme_d_ilp_balanced(w_c3=10, w_str=1)
    if groups_d is not None:
        results.append(evaluate(groups_d, "方案D: ILP(均衡)"))

    print_comparison(results)


if __name__ == "__main__":
    main()
