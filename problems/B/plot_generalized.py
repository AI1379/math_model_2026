"""
广义蒙特卡洛可视化
==================
从 generalized_mc.py 的实验数据生成图表, 保存到 figures/ 目录.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
from generalized_mc import (
    generate_random_topology, build_teams_from_topology,
    greedy_on_topology, compute_f1, compute_f2, compute_f3,
    ProvinceTopology,
)

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def run_experiment(n_topologies=200, n_seeds=50):
    """复用 generalized_mc 的实验逻辑, 返回结构化数据"""
    results = []

    for ti in range(n_topologies):
        rng_topo = np.random.default_rng(ti * 1000)
        topo = generate_random_topology(rng_topo)
        teams = build_teams_from_topology(topo)

        stats = {
            "k": topo.k, "max_county": topo.max_county,
            "tightness": topo.tightness, "gini": topo.gini,
            "county_per_city": list(topo.county_per_city),
            "no_repair_fail": 0, "repair_fail": 0, "repair_used": 0,
            "repair_f1": [], "repair_f2": [], "repair_f3": [],
            "restart_f2": [], "restart_f3": [], "restart_retries": [],
        }

        for si in range(n_seeds):
            # 无修复
            rng1 = np.random.default_rng(ti * 1000 + si + 1)
            g1, ok1, _ = greedy_on_topology(topo, teams, rng1, repair=False)
            if not ok1:
                stats["no_repair_fail"] += 1

            # 有修复
            rng2 = np.random.default_rng(ti * 1000 + si + 1)
            g2, ok2, used = greedy_on_topology(topo, teams, rng2, repair=True)
            if not ok2:
                stats["repair_fail"] += 1
            else:
                stats["repair_f1"].append(compute_f1(g2, teams))
                stats["repair_f2"].append(compute_f2(g2, teams))
                stats["repair_f3"].append(compute_f3(g2, teams))
                if used:
                    stats["repair_used"] += 1

            # 重启
            for attempt in range(20):
                rng3 = np.random.default_rng(ti * 100000 + si * 20 + attempt)
                g3, ok3, _ = greedy_on_topology(topo, teams, rng3, repair=False)
                if ok3 and compute_f1(g3, teams) == 0:
                    stats["restart_f2"].append(compute_f2(g3, teams))
                    stats["restart_f3"].append(compute_f3(g3, teams))
                    stats["restart_retries"].append(attempt + 1)
                    break
            else:
                stats["restart_retries"].append(20)

        results.append(stats)
        if (ti + 1) % 50 == 0:
            print(f"  完成 {ti + 1}/{n_topologies}")

    return results, n_seeds


def plot_all(results, n_seeds):
    """生成所有图表"""

    tau = np.array([r["tightness"] for r in results])
    no_fail = np.array([r["no_repair_fail"] / n_seeds for r in results])
    rp_fail = np.array([r["repair_fail"] / n_seeds for r in results])
    rp_c3 = np.array([
        np.mean([1 if f == 0 else 0 for f in r["repair_f1"]]) if r["repair_f1"] else 0
        for r in results
    ])
    rp_f2 = np.array([np.mean(r["repair_f2"]) if r["repair_f2"] else 0 for r in results])
    rs_f2 = np.array([np.mean(r["restart_f2"]) if r["restart_f2"] else 0 for r in results])
    rs_retries = np.array([np.mean(r["restart_retries"]) for r in results])
    gini = np.array([r["gini"] for r in results])

    zj = ProvinceTopology(k=11, county_per_city=[3, 4, 8, 5, 3, 3, 7, 4, 2, 6, 8])
    zj_tau = zj.tightness

    # ── 图1: 紧度 vs 死锁率 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau, no_fail * 100, alpha=0.4, s=20, c="#e74c3c", label="无修复贪心")
    ax.scatter(tau, rp_fail * 100, alpha=0.4, s=20, c="#2ecc71", label="贪心+修复")

    # 趋势线
    z = np.polyfit(tau, no_fail * 100, 2)
    xs = np.linspace(tau.min(), tau.max(), 100)
    ax.plot(xs, np.polyval(z, xs), "--", c="#e74c3c", alpha=0.7, linewidth=1.5)

    ax.axvline(zj_tau, color="gray", linestyle=":", alpha=0.8, label=f"浙江省 τ={zj_tau:.2f}")
    ax.set_xlabel(r"约束紧度 $\tau$ = max($n_i$) / 15", fontsize=12)
    ax.set_ylabel("死锁率 (%)", fontsize=12)
    ax.set_title("图1: 约束紧度 vs 死锁率 (200个随机拓扑)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-1, 55)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_tightness_vs_deadlock.png"), dpi=150)
    plt.close(fig)
    print(f"  保存 fig1_tightness_vs_deadlock.png")

    # ── 图2: 紧度 vs C3满足率 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau, rp_c3 * 100, alpha=0.4, s=20, c="#3498db")

    z = np.polyfit(tau, rp_c3 * 100, 2)
    ax.plot(xs, np.polyval(z, xs), "--", c="#3498db", alpha=0.7, linewidth=1.5)

    ax.axvline(zj_tau, color="gray", linestyle=":", alpha=0.8, label=f"浙江省 τ={zj_tau:.2f}")
    ax.axhline(100, color="#2ecc71", linestyle="--", alpha=0.3)
    ax.set_xlabel(r"约束紧度 $\tau$ = max($n_i$) / 15", fontsize=12)
    ax.set_ylabel("C3 完全满足率 (%)", fontsize=12)
    ax.set_title("图2: 约束紧度 vs C3完全满足率", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_tightness_vs_c3.png"), dpi=150)
    plt.close(fig)
    print(f"  保存 fig2_tightness_vs_c3.png")

    # ── 图3: 修复 vs 重启 F2 对比 (分桶箱线图) ──
    bins = [0, 0.5, 0.7, 1.0]
    labels = ["τ≤0.5\n(中等)", "0.5<τ≤0.7\n(偏紧)", "τ>0.7\n(紧张)"]

    repair_by_bin = [[] for _ in range(3)]
    restart_by_bin = [[] for _ in range(3)]

    for r in results:
        t = r["tightness"]
        for b in range(3):
            if bins[b] < t <= bins[b + 1]:
                repair_by_bin[b].extend(r["repair_f2"])
                restart_by_bin[b].extend(r["restart_f2"])
                break

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(3)
    bp1 = ax.boxplot(
        [np.array(d) if d else np.array([0]) for d in repair_by_bin],
        positions=positions - 0.15, widths=0.25, patch_artist=True,
        boxprops=dict(facecolor="#3498db", alpha=0.6),
        medianprops=dict(color="#2c3e50", linewidth=2),
    )
    bp2 = ax.boxplot(
        [np.array(d) if d else np.array([0]) for d in restart_by_bin],
        positions=positions + 0.15, widths=0.25, patch_artist=True,
        boxprops=dict(facecolor="#e67e22", alpha=0.6),
        medianprops=dict(color="#2c3e50", linewidth=2),
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("F2 (实力标准差)", fontsize=12)
    ax.set_title("图3: 修复策略 vs 重启策略 — F2实力均衡对比", fontsize=13)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["贪心+修复", "重启至C3=0"], fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_f2_comparison_boxplot.png"), dpi=150)
    plt.close(fig)
    print(f"  保存 fig3_f2_comparison_boxplot.png")

    # ── 图4: 重启策略平均重试次数 vs 紧度 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau, rs_retries, alpha=0.4, s=20, c="#9b59b6")

    z = np.polyfit(tau, rs_retries, 2)
    ax.plot(xs, np.polyval(z, xs), "--", c="#9b59b6", alpha=0.7, linewidth=1.5)

    ax.axvline(zj_tau, color="gray", linestyle=":", alpha=0.8, label=f"浙江省 τ={zj_tau:.2f}")
    ax.set_xlabel("约束紧度 τ", fontsize=12)
    ax.set_ylabel("平均重试次数", fontsize=12)
    ax.set_title("图4: 重启策略的平均重试代价随紧度增长", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_restart_retries.png"), dpi=150)
    plt.close(fig)
    print(f"  保存 fig4_restart_retries.png")

    # ── 图5: τ 的定义示意 + 浙江省位置 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左: 浙江省的县级队分布
    cities = ["杭州", "宁波", "温州", "嘉兴", "湖州", "绍兴", "金华", "衢州", "舟山", "台州", "丽水"]
    counties = [3, 4, 8, 5, 3, 3, 7, 4, 2, 6, 8]
    colors = ["#e74c3c" if c == 8 else "#3498db" for c in counties]
    bars = ax1.barh(cities, counties, color=colors, alpha=0.7)
    ax1.axvline(15, color="gray", linestyle="--", alpha=0.5, label="可用组上限=15")
    ax1.axvline(8, color="#e74c3c", linestyle=":", alpha=0.8, label="max=8, tau=8/15=0.53")
    ax1.set_xlabel("县级队数量", fontsize=11)
    ax1.set_title("浙江省: 各市县级队分布", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.invert_yaxis()

    # 右: 随机拓扑的τ分布直方图, 标注浙江位置
    ax2.hist(tau, bins=30, color="#3498db", alpha=0.6, edgecolor="white")
    ax2.axvline(zj_tau, color="#e74c3c", linestyle="-", linewidth=2,
                label=f"浙江省 τ={zj_tau:.2f}")
    ax2.set_xlabel("约束紧度 τ", fontsize=11)
    ax2.set_ylabel("拓扑数量", fontsize=11)
    ax2.set_title("200个随机拓扑的紧度分布", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5_tightness_definition.png"), dpi=150)
    plt.close(fig)
    print(f"  保存 fig5_tightness_definition.png")

    # ── 图6: τ vs F2差值 (repair - restart) ──
    f2_diff = rp_f2 - rs_f2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau, f2_diff, alpha=0.4, s=20, c="#2c3e50")
    ax.axhline(0, color="#e74c3c", linestyle="-", linewidth=1.5, alpha=0.7)
    z = np.polyfit(tau, f2_diff, 2)
    ax.plot(xs, np.polyval(z, xs), "--", c="#e67e22", alpha=0.7, linewidth=1.5, label="趋势线")
    ax.axvline(zj_tau, color="gray", linestyle=":", alpha=0.8, label=f"浙江省")
    ax.set_xlabel("约束紧度 τ", fontsize=12)
    ax.set_ylabel(r"$\Delta F2$ = F2(修复) - F2(重启)", fontsize=12)
    ax.set_title("图6: 修复与重启的F2差异随紧度变化", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig6_f2_difference.png"), dpi=150)
    plt.close(fig)
    print(f"  保存 fig6_f2_difference.png")

    # ══════════════════════════════════════════════════
    # Q3+Q4 图表
    # ══════════════════════════════════════════════════
    print(f"\n  生成 Q3+Q4 泛化图表...")
    from q3_q4_generalized import run_experiment as run_q3q4
    q3q4_results, n_tournament = run_q3q4(n_topologies=150, n_seeds_venue=10, n_seeds_tournament=50)

    tau_q = np.array([r["tightness"] for r in q3q4_results])

    # ── 图7: τ vs 总旅行距离 (Q3 选址) ──
    total_dists = np.array([np.mean(r["total_dist"]) if r["total_dist"] else 0 for r in q3q4_results])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau_q, total_dists, alpha=0.4, s=20, c="#1abc9c")
    z = np.polyfit(tau_q, total_dists, 2)
    xs_q = np.linspace(tau_q.min(), tau_q.max(), 100)
    ax.plot(xs_q, np.polyval(z, xs_q), "--", c="#16a085", alpha=0.7, linewidth=1.5, label="趋势线")
    ax.axvline(zj_tau, color="gray", linestyle=":", alpha=0.8, label=f"浙江省")
    ax.set_xlabel("约束紧度 τ", fontsize=12)
    ax.set_ylabel("平均总旅行距离 (km)", fontsize=12)
    ax.set_title("图7: 选址效果 (总旅行距离) vs 约束紧度", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    corr = np.corrcoef(tau_q, total_dists)[0, 1]
    ax.text(0.05, 0.95, f"ρ = {corr:.3f}", transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig7_venue_total_dist.png"), dpi=150)
    plt.close(fig)
    print(f"  保存 fig7_venue_total_dist.png")

    # ── 图8: τ vs Spearman (Q4 赛制公平性) ──
    spearman_means = np.array([np.mean(r["spearman"]) if r["spearman"] else 0 for r in q3q4_results])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau_q, spearman_means, alpha=0.4, s=20, c="#8e44ad")
    z = np.polyfit(tau_q, spearman_means, 2)
    ax.plot(xs_q, np.polyval(z, xs_q), "--", c="#9b59b6", alpha=0.7, linewidth=1.5, label="趋势线")
    ax.axvline(zj_tau, color="gray", linestyle=":", alpha=0.8, label=f"浙江省")
    ax.set_xlabel("约束紧度 τ", fontsize=12)
    ax.set_ylabel("Spearman 相关系数 (均值)", fontsize=12)
    ax.set_title("图8: 赛制公平性 (Spearman) vs 约束紧度", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    corr_sp = np.corrcoef(tau_q, spearman_means)[0, 1]
    ax.text(0.05, 0.05, f"ρ = {corr_sp:.3f}", transform=ax.transAxes, fontsize=11, va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig8_spearman_vs_tau.png"), dpi=150)
    plt.close(fig)
    print(f"  保存 fig8_spearman_vs_tau.png")

    # ── 图9: τ vs 前32晋级率 (Q4) ──
    top32_means = np.array([np.mean(r["top32_rate"]) if r["top32_rate"] else 0 for r in q3q4_results])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau_q, top32_means, alpha=0.4, s=20, c="#e67e22")
    z = np.polyfit(tau_q, top32_means, 2)
    ax.plot(xs_q, np.polyval(z, xs_q), "--", c="#f39c12", alpha=0.7, linewidth=1.5, label="趋势线")
    ax.axvline(zj_tau, color="gray", linestyle=":", alpha=0.8, label=f"浙江省")
    ax.set_xlabel("约束紧度 τ", fontsize=12)
    ax.set_ylabel("前32名晋级率 (均值)", fontsize=12)
    ax.set_title("图9: 晋级准确率 vs 约束紧度", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    corr_t32 = np.corrcoef(tau_q, top32_means)[0, 1]
    ax.text(0.05, 0.05, f"ρ = {corr_t32:.3f}", transform=ax.transAxes, fontsize=11, va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig9_top32_vs_tau.png"), dpi=150)
    plt.close(fig)
    print(f"  保存 fig9_top32_vs_tau.png")

    print(f"\n  所有图表已保存到 {FIGDIR}")


if __name__ == "__main__":
    print("运行广义蒙特卡洛实验并生成图表...")
    results, n_seeds = run_experiment(n_topologies=2000, n_seeds=50)
    plot_all(results, n_seeds)
