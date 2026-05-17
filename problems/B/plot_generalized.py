#!/usr/bin/env python3
"""
Plot generalized Q2 Monte Carlo results.

The current generalized experiment compares only two models:
basic greedy and look-ahead network flow.  Figures are saved under figures/.
"""

from __future__ import annotations

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generalized_mc import ProvinceTopology, run_generalized_mc


matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def plot_all(results, n_seeds):
    tau = np.array([r["tightness"] for r in results])
    greedy_deadlock = np.array([r["greedy_deadlocks"] / n_seeds for r in results])
    flow_fail = np.array([r["flow_failures"] / n_seeds for r in results])
    greedy_c3_zero = np.array(
        [
            np.mean([f == 0 for f in r["greedy_f1"]]) if r["greedy_f1"] else 0.0
            for r in results
        ]
    )
    flow_c3_zero = np.array(
        [
            np.mean([f == 0 for f in r["flow_f1"]]) if r["flow_f1"] else 0.0
            for r in results
        ]
    )
    flow_f1 = np.array([_mean(r["flow_f1"]) for r in results])
    flow_checks = np.array([_mean(r["flow_checks"]) for r in results])

    zj_tau = ProvinceTopology(
        k=11, county_per_city=[3, 4, 8, 5, 3, 3, 7, 4, 2, 6, 8]
    ).tightness

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau, greedy_deadlock * 100, s=24, alpha=0.55, label="基础贪心")
    ax.scatter(tau, flow_fail * 100, s=24, alpha=0.55, label="前瞻网络流")
    ax.axvline(zj_tau, color="gray", linestyle=":", label=f"浙江 τ={zj_tau:.2f}")
    ax.set_xlabel(r"约束紧度 $\tau=\max(n_i)/15$")
    ax.set_ylabel("失败率 / 死锁率 (%)")
    ax.set_title("广义蒙特卡洛：死锁率对比")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "generalized_deadlock_compare.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau, greedy_c3_zero * 100, s=24, alpha=0.55, label="基础贪心")
    ax.scatter(tau, flow_c3_zero * 100, s=24, alpha=0.55, label="前瞻网络流")
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.5, label=r"$\tau=1$")
    ax.axvline(zj_tau, color="gray", linestyle=":", label=f"浙江 τ={zj_tau:.2f}")
    ax.set_xlabel(r"约束紧度 $\tau=\max(n_i)/15$")
    ax.set_ylabel(r"$P(F_1=0)$ (%)")
    ax.set_title("广义蒙特卡洛：C3零冲突满足率")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "generalized_c3_zero_compare.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau, flow_f1, s=24, alpha=0.6, color="#2c7fb8")
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.5, label=r"$\tau=1$")
    ax.axvline(zj_tau, color="gray", linestyle=":", label=f"浙江 τ={zj_tau:.2f}")
    ax.set_xlabel(r"约束紧度 $\tau=\max(n_i)/15$")
    ax.set_ylabel(r"前瞻网络流 $\mathbb{E}[F_1]$")
    ax.set_title("广义蒙特卡洛：不可零冲突时的最小退化")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "generalized_flow_f1.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tau, flow_checks, s=24, alpha=0.6, color="#7b3294")
    ax.axvline(zj_tau, color="gray", linestyle=":", label=f"浙江 τ={zj_tau:.2f}")
    ax.set_xlabel(r"约束紧度 $\tau=\max(n_i)/15$")
    ax.set_ylabel("平均候选组前瞻检查次数")
    ax.set_title("广义蒙特卡洛：前瞻检查计算量")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "generalized_flow_checks.png"), dpi=150)
    plt.close(fig)

    print(f"Figures saved to {FIGDIR}")


def main():
    n_topologies = 100
    n_seeds = 3
    results = run_generalized_mc(n_topologies=n_topologies, n_seeds_per_topo=n_seeds)
    plot_all(results, n_seeds)


if __name__ == "__main__":
    main()
