#!/usr/bin/env python3
"""Generate illustrative figures for the JEIT-style paper.

All numbers are representative/illustrative values that are consistent with the
real SO-ARM101 + PPO + ResNet18 + domain-randomization system in this repo.
Replace them with your own measured results before final submission.

Axis labels are in English on purpose (robust across fonts and common in
Chinese journals); figure captions are written in Chinese in the manuscript.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "legend.frameon": False,
})

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

C = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "gray": "#7f7f7f",
    "teal": "#17becf",
}


# ----------------------------------------------------------------------------
# Figure 1: System architecture (decoupled perception-control framework)
# ----------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#222222", fs=10.5, tc="black"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2.2",
                           linewidth=1.4, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, wrap=True)
        return (x, y, w, h)

    def arrow(p1, p2, text="", color="#333333", style="-|>", rad=0.0, ls="-",
              off=(0, 0), fs=9):
        a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                            linewidth=1.5, color=color,
                            connectionstyle=f"arc3,rad={rad}", linestyle=ls)
        ax.add_patch(a)
        if text:
            mx, my = (p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1]
            ax.text(mx, my, text, ha="center", va="center", fontsize=fs,
                    color=color, style="italic")

    blue, green, orange, purple, gray = "#cfe3f7", "#d6efd6", "#ffe6cc", "#ece1f5", "#eeeeee"

    # Perception path (top row)
    sim = box(2, 33, 20, 13,
              "Isaac Lab Simulation\nSO-ARM101 + Cube\n(physics, contacts)", blue)
    cam = box(30, 36, 17, 10, "Fixed Camera\nRGB 128x128", green)
    res = box(54, 36, 19, 10, "ResNet-18\nPose Estimator  $f_\\phi$", green)

    # Control path (bottom row)
    obs = box(30, 14, 19, 12,
              "Observation $o_t$\nproprio $\\oplus$ $\\hat{o}$ $\\oplus$ goal", orange)
    pol = box(54, 14, 19, 12,
              "PPO Policy $\\pi_\\theta$\nActor-Critic\n256-128-64", purple)
    act = box(80, 14, 17, 12, "Action $a_t$\n6-DoF joint\ntargets", orange)

    # Bottom training-signal boxes
    rew = box(2, 1.5, 30, 8.5,
              "Four-Stage Gated Reward $R$  (grasp / transport / descend / release)",
              "#fde2e1", fs=9.5)
    dr = box(36, 1.5, 28, 8.5,
             "Domain Randomization $\\xi$\n(friction, light, camera, clutter)",
             gray, fs=9.5)
    noise = box(68, 1.5, 29, 8.5,
                "Perception-Error Injection $\\eta$\n$\\hat{o}=o^*+\\eta,\\ \\eta\\sim N(0,\\sigma_p^2)$",
                gray, fs=9.5)

    # Arrows: perception
    arrow((22, 42), (30, 41), "render")
    arrow((47, 41), (54, 41), "image")
    arrow((63.5, 36), (45, 26), "$\\hat{o}$", color=C["green"], rad=-0.15)
    # proprioception from sim to observation
    arrow((12, 33), (33, 26), "proprio, goal", color="#555555", rad=-0.1, fs=8.5)
    # control flow
    arrow((49, 20), (54, 20))
    arrow((73, 20), (80, 20), "$a_t$")
    # feedback action -> sim
    arrow((88, 26), (12, 33.2), "control", color=C["red"], rad=-0.32, fs=9)
    # training signals
    arrow((17, 10), (60, 14), "training", color=C["red"], rad=0.12, ls="--", fs=8.5)
    arrow((50, 10), (14, 33), "$\\xi$", color="#555555", rad=0.18, ls="--", fs=9)
    arrow((82, 10), (40, 14), "$\\eta$", color="#555555", rad=0.12, ls="--", fs=9)

    ax.text(50, 53.4, "Decoupled Perception--Control Framework for Sim-to-Real Pick-and-Place",
            ha="center", va="center", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_architecture.png"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: PPO training curves (success rate + return)
# ----------------------------------------------------------------------------
def fig_training():
    rng = np.random.default_rng(7)
    it = np.linspace(0, 12000, 240)

    def logistic(L, k, x0, base=0.0):
        return base + (L - base) / (1 + np.exp(-(it - x0) / k))

    full = logistic(0.935, 1150, 5200)
    no_pen = logistic(0.785, 1500, 6200)
    no_cur = logistic(0.40, 1700, 7600)

    def noisy(y, s):
        return np.clip(y + rng.normal(0, s, size=y.shape), 0, 1)

    full_n, no_pen_n, no_cur_n = noisy(full, 0.012), noisy(no_pen, 0.018), noisy(no_cur, 0.02)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    ax.plot(it, full_n, color=C["blue"], lw=2, label="Full method")
    ax.plot(it, no_pen_n, color=C["orange"], lw=2, label="w/o anti-failure penalties")
    ax.plot(it, no_cur_n, color=C["red"], lw=2, label="w/o curriculum (sparse reward)")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Task success rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) Success rate vs. training iterations")
    ax.legend(loc="lower right", fontsize=9.5)

    ax = axes[1]
    ret_full = 165 / (1 + np.exp(-(it - 5000) / 1200)) + rng.normal(0, 3.0, size=it.shape)
    ret_nc = 58 / (1 + np.exp(-(it - 7800) / 1700)) + rng.normal(0, 3.0, size=it.shape)
    ax.plot(it, ret_full, color=C["blue"], lw=2, label="Full method")
    ax.plot(it, ret_nc, color=C["red"], lw=2, label="w/o curriculum")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Mean episode return")
    ax.set_title("(b) Episode return vs. training iterations")
    ax.legend(loc="lower right", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_training.png"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: Ablation bars (reward design + sim-to-real)
# ----------------------------------------------------------------------------
def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # (a) reward-design ablation, ground-truth state, no disturbance
    labels_a = ["Full", "w/o stage3-4\nplace shaping", "w/o anti-failure\npenalties",
                "w/o curriculum\n(sparse)"]
    vals_a = [93.5, 64.0, 78.0, 39.0]
    colors_a = [C["blue"], C["teal"], C["orange"], C["red"]]
    ax = axes[0]
    bars = ax.bar(labels_a, vals_a, color=colors_a, width=0.62, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) Reward-design ablation (GT state, no disturbance)")
    ax.tick_params(axis="x", labelsize=9)
    for b, v in zip(bars, vals_a):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=9.5)

    # (b) sim-to-real ablation, vision in the loop
    labels_b = ["GT state\n(upper bound)", "Full\n(DR+noise inj.)", "w/o noise\ninjection",
                "w/o domain\nrandomization", "w/o both"]
    vals_b = [93.5, 91.0, 72.5, 70.0, 58.0]
    colors_b = [C["gray"], C["blue"], C["orange"], C["purple"], C["red"]]
    ax = axes[1]
    bars = ax.bar(labels_b, vals_b, color=colors_b, width=0.62, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(b) Sim-to-real ablation (vision in the loop)")
    ax.tick_params(axis="x", labelsize=8.5)
    for b, v in zip(bars, vals_b):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_ablation.png"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 4: Robustness vs. disturbance level
# ----------------------------------------------------------------------------
def fig_robustness():
    levels = np.array([0.0, 0.35, 0.7, 1.0])
    xt = ["off", "low", "medium", "high"]
    data = {
        "Cube init pose": [93.5, 92.1, 89.0, 84.5],
        "Goal position": [93.5, 91.0, 86.7, 80.2],
        "Contact friction": [93.5, 90.2, 83.5, 72.0],
        "Lighting": [93.5, 91.8, 88.0, 82.5],
        "Camera pose": [93.5, 90.5, 84.0, 75.5],
        "Clutter": [93.5, 89.0, 81.0, 70.0],
    }
    markers = ["o", "s", "^", "D", "v", "P"]
    colors = [C["blue"], C["green"], C["red"], C["orange"], C["purple"], C["brown"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    for (k, v), m, c in zip(data.items(), markers, colors):
        ax.plot(levels, v, marker=m, color=c, lw=1.8, ms=6, label=k)
    allc = [93.5, 87.0, 74.0, 58.0]
    ax.plot(levels, allc, marker="*", color="black", lw=2.2, ms=10, ls="--", label="All combined")
    ax.set_xticks(levels)
    ax.set_xticklabels(xt)
    ax.set_xlabel("Disturbance level")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(50, 100)
    ax.set_title("(a) Robustness of the full method")
    ax.legend(fontsize=8.3, ncol=2, loc="lower left")

    # (b) DR benefit on a vision-sensitive factor (lighting)
    ax = axes[1]
    with_dr = [93.5, 91.8, 88.0, 82.5]
    without_dr = [93.5, 78.0, 52.0, 28.0]
    ax.plot(levels, with_dr, marker="o", color=C["blue"], lw=2, ms=6, label="With DR + noise inj.")
    ax.plot(levels, without_dr, marker="s", color=C["red"], lw=2, ms=6, ls="--", label="Without DR")
    ax.fill_between(levels, without_dr, with_dr, color=C["blue"], alpha=0.12)
    ax.set_xticks(levels)
    ax.set_xticklabels(xt)
    ax.set_xlabel("Lighting disturbance level")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(20, 100)
    ax.set_title("(b) Effect of domain randomization (lighting)")
    ax.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_robustness.png"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 5: Vision pose-estimation accuracy
# ----------------------------------------------------------------------------
def fig_vision():
    rng = np.random.default_rng(3)
    n = 400
    # workspace (robot-frame) in metres
    true_x = rng.uniform(0.12, 0.32, n)
    true_y = rng.uniform(-0.20, 0.20, n)
    sx, sy = 0.0072, 0.0085  # per-axis error std (m)
    pred_x = true_x + rng.normal(0, sx, n)
    pred_y = true_y + rng.normal(0, sy, n)

    err = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2) * 100.0  # cm
    mae = np.mean(np.abs(pred_x - true_x) + np.abs(pred_y - true_y)) / 2 * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    ax = axes[0]
    ax.scatter(true_x * 100, pred_x * 100, s=12, color=C["blue"], alpha=0.55, label="x axis")
    ax.scatter(true_y * 100, pred_y * 100, s=12, color=C["orange"], alpha=0.55, label="y axis")
    lim = [-22, 34]
    ax.plot(lim, lim, color="black", lw=1.2, ls="--", label="ideal $y=x$")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Ground-truth position (cm)")
    ax.set_ylabel("Predicted position (cm)")
    ax.set_title("(a) Predicted vs. ground-truth cube position")
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[1]
    ax.hist(err, bins=26, color=C["teal"], edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axvline(np.mean(err), color=C["red"], lw=2, label=f"mean = {np.mean(err):.2f} cm")
    ax.axvline(np.median(err), color=C["purple"], lw=2, ls="--",
               label=f"median = {np.median(err):.2f} cm")
    ax.set_xlabel("Euclidean position error (cm)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Distribution of position-estimation error")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_vision.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[vision] per-axis MAE approx {mae:.3f} cm")


if __name__ == "__main__":
    fig_architecture()
    fig_training()
    fig_ablation()
    fig_robustness()
    fig_vision()
    print("All figures written to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
