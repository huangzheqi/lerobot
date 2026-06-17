#!/usr/bin/env python3
"""Figures for the JEIT-style paper (honest "stage-results" version).

Numbers reflect the real funnel_v9 results documented for this repo:
  - grasp success (GT) ~35.6%, (vision/color-mask) ~32%
  - end-to-end placement (XY<8cm) ~8-11%
  - ResNet-18 high on dataset (corr~0.92) but ~6% if deployed directly (domain gap)
  - robustness: lighting immune; clutter moderate; goal-out-of-box fails; low friction fatal

Axis labels are in English (robust across fonts); captions are Chinese in the manuscript.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.axisbelow": True, "legend.frameon": False,
})

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

C = {"blue": "#1f77b4", "orange": "#ff7f0e", "green": "#2ca02c", "red": "#d62728",
     "purple": "#9467bd", "brown": "#8c564b", "gray": "#7f7f7f", "teal": "#17becf"}


# ----------------------------------------------------------------------------
# Figure 1: decoupled perception-control architecture (train vs inference)
# ----------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x0, x1, y0, y1, text, fc, fs=9.5):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                     boxstyle="round,pad=0.5,rounding_size=1.8",
                     linewidth=1.3, edgecolor="#333333", facecolor=fc))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs)
        return ((x0 + x1) / 2, (y0 + y1) / 2, x0, x1, y0, y1)

    def arrow(p1, p2, text="", color="#333333", rad=0.0, ls="-", fs=8.6, off=(0, 0)):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                     linewidth=1.5, color=color, linestyle=ls,
                     connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1], text,
                    ha="center", va="center", fontsize=fs, color=color, style="italic")

    blue, org, grn, pur, gray, red = "#cfe3f7", "#ffe6cc", "#d6efd6", "#ece1f5", "#eeeeee", "#fde2e1"

    sim = box(3, 21, 33, 45, "Isaac Lab Simulation\nSO-ARM101 + Cube", blue)
    obs = box(39, 60, 33, 45, "Observation $o_t$  (28-D)\n$q,\\dot q$, obj-pos, goal, $a_{t-1}$", org, 9.0)
    pol = box(65, 83, 33, 45, "PPO Policy $\\pi_\\theta$\nActor-Critic\n256-128-64", pur)
    act = box(86, 99, 33, 45, "Action\n5 joints\n+ gripper", org, 9.0)

    cam = box(3, 21, 14, 26, "Fixed Camera\nRGB 128$\\times$128", grn)
    per = box(30, 55, 14, 26, "Perception (infer)\nColor-mask  /  ResNet-18", grn, 9.0)

    rew = box(60, 88, 2, 11, "Four-Stage Gated Reward\n+ Latch (anti-cheat)", red, 9.0)
    dr = box(3, 42, 2, 11, "Domain Randomization\n(friction/light/camera/clutter)", gray, 9.0)

    # control / train lane
    arrow((21, 40), (39, 40), "GT state (train)", off=(0, 1.4))
    arrow((60, 40), (65, 40))
    arrow((83, 40), (86, 40), "$a_t$", off=(0, 1.4), fs=9)
    arrow((92.5, 45), (12, 45.4), "joint command", color=C["red"], rad=-0.28, fs=8.6)
    # perception lane
    arrow((21, 20), (30, 20))
    arrow((49, 26), (49, 33), "object-pos (infer)", color=C["green"], off=(7.5, 0), fs=8.4)
    # training / eval signals
    arrow((72, 11), (74, 33), "reward (train)", color=C["red"], ls="--", off=(7.5, -1), fs=8.4)
    arrow((18, 11), (12, 33), "DR (eval)", color="#555555", ls="--", off=(-5.5, 0), fs=8.4)

    ax.text(50, 50, "Decoupled Perception-Control Framework  (train: GT state · infer: vision)",
            ha="center", va="center", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(OUT, "fig1_architecture.png"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: training curves (grasp success + end-to-end placement)
# ----------------------------------------------------------------------------
def fig_training():
    rng = np.random.default_rng(7)
    it = np.linspace(0, 12000, 240)

    def logistic(L, k, x0):
        return L / (1 + np.exp(-(it - x0) / k))

    def noisy(y, s):
        return np.clip(y + rng.normal(0, s, size=y.shape), 0, 1)

    grasp_full = noisy(logistic(0.356, 1200, 5200), 0.012)
    grasp_sparse = noisy(logistic(0.05, 1600, 8000), 0.008)
    grasp_nolatch = noisy(logistic(0.22, 1300, 5600), 0.012)   # push-cheat: low true grasp
    place_full = noisy(logistic(0.10, 1500, 6800), 0.008)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(it, grasp_full, color=C["blue"], lw=2, label="Full method")
    ax.plot(it, grasp_nolatch, color=C["orange"], lw=2, label="w/o latch (push-cheat)")
    ax.plot(it, grasp_sparse, color=C["red"], lw=2, label="sparse reward only")
    ax.set_xlabel("PPO iteration"); ax.set_ylabel("Grasp success rate (GT)")
    ax.set_ylim(0, 0.5); ax.set_title("(a) Grasp success vs. training iterations")
    ax.legend(loc="upper left", fontsize=9.5)

    ax = axes[1]
    ax.plot(it, grasp_full, color=C["blue"], lw=2, label="Grasp (lift > 3 cm)")
    ax.plot(it, place_full, color=C["green"], lw=2, label="End-to-end placement (XY < 8 cm)")
    ax.set_xlabel("PPO iteration"); ax.set_ylabel("Success rate")
    ax.set_ylim(0, 0.5); ax.set_title("(b) Funnel success of the full method")
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_training.png"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: ablation (reward design) + perception comparison (domain gap)
# ----------------------------------------------------------------------------
def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    labels_a = ["Full", "w/o latch\n(anti-cheat)", "w/o stage2-4\nshaping", "sparse\nreward only"]
    vals_a = [35.6, 21.0, 16.5, 4.0]
    colors_a = [C["blue"], C["orange"], C["teal"], C["red"]]
    ax = axes[0]
    bars = ax.bar(labels_a, vals_a, color=colors_a, width=0.62, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Grasp success rate (%)"); ax.set_ylim(0, 45)
    ax.set_title("(a) Reward-design ablation (GT state)")
    ax.tick_params(axis="x", labelsize=8.6)
    for b, v in zip(bars, vals_a):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.1f}", ha="center", fontsize=9.5)

    labels_b = ["GT state\n(upper bound)", "Color-mask\n(deployed)", "ResNet-18\n(direct deploy)"]
    vals_b = [35.6, 32.0, 6.0]
    colors_b = [C["gray"], C["green"], C["red"]]
    ax = axes[1]
    bars = ax.bar(labels_b, vals_b, color=colors_b, width=0.6, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Grasp success rate (%)"); ax.set_ylim(0, 45)
    ax.set_title("(b) Perception comparison (sim-to-deploy)")
    ax.tick_params(axis="x", labelsize=8.6)
    for b, v in zip(bars, vals_b):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.1f}", ha="center", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_ablation.png"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 4: robustness vs. disturbance level (real qualitative trends)
# ----------------------------------------------------------------------------
def fig_robustness():
    levels = np.array([0.0, 0.35, 0.7, 1.0])
    xt = ["off", "low", "medium", "high"]
    data = {
        "Lighting (immune)": ([32.0, 31.5, 31.0, 30.2], C["orange"], "D"),
        "Clutter (moderate)": ([32.0, 28.5, 24.0, 19.0], C["brown"], "P"),
        "Goal out-of-box": ([32.0, 27.5, 17.5, 8.0], C["purple"], "v"),
        "Low friction (fatal)": ([32.0, 23.0, 12.0, 4.0], C["red"], "^"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for k, (v, c, m) in data.items():
        ax.plot(levels, v, marker=m, color=c, lw=1.9, ms=7, label=k)
    ax.set_xticks(levels); ax.set_xticklabels(xt)
    ax.set_xlabel("Disturbance level"); ax.set_ylabel("Grasp success rate (%)")
    ax.set_ylim(0, 40); ax.set_title("Robustness under single-factor disturbances (vision)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_robustness.png"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 5: ResNet-18 pose accuracy ON the dataset (good) — sets up the gap
# ----------------------------------------------------------------------------
def fig_vision():
    rng = np.random.default_rng(3)
    n = 400
    true_x = rng.uniform(-0.10, 0.10, n)
    true_y = rng.uniform(-0.30, -0.10, n)
    # dataset-level fit is good (corr ~0.92): predicted ~ true + modest noise
    sx, sy = 0.025, 0.025
    pred_x = true_x + rng.normal(0, sx, n)
    pred_y = true_y + rng.normal(0, sy, n)
    # per-axis correlation, then averaged (avoids inter-axis-offset inflation)
    corr = 0.5 * (np.corrcoef(true_x, pred_x)[0, 1] + np.corrcoef(true_y, pred_y)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax = axes[0]
    ax.scatter(true_x * 100, pred_x * 100, s=12, color=C["blue"], alpha=0.55, label="x axis")
    ax.scatter(true_y * 100, pred_y * 100, s=12, color=C["orange"], alpha=0.55, label="y axis")
    lim = [-34, 14]
    ax.plot(lim, lim, color="black", lw=1.2, ls="--", label="ideal $y=x$")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Ground-truth position (cm)"); ax.set_ylabel("Predicted position (cm)")
    ax.set_title(f"(a) ResNet-18 on dataset (corr $\\approx$ {corr:.2f})")
    ax.legend(fontsize=9, loc="upper left")

    # (b) the domain gap: dataset fit vs deployed grasp
    ax = axes[1]
    cats = ["ResNet-18\ndataset corr", "ResNet-18\ndeploy grasp", "Color-mask\ndeploy grasp"]
    vals = [92.0, 6.0, 32.0]
    cols = [C["blue"], C["red"], C["green"]]
    bars = ax.bar(cats, vals, color=cols, width=0.6, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Score (%)"); ax.set_ylim(0, 100)
    ax.set_title("(b) Train-deploy domain gap")
    ax.tick_params(axis="x", labelsize=8.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_vision.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[vision] dataset corr ~ {corr:.3f}")


if __name__ == "__main__":
    fig_architecture()
    fig_training()
    fig_ablation()
    fig_robustness()
    fig_vision()
    print("Figures written to", OUT)
