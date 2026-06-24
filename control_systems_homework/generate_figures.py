# -*- coding: utf-8 -*-
"""
一阶 & 二阶系统时域响应分析 —— 图像生成脚本
作业：参考视频，修改参数，设计自己的一阶/二阶系统，观察响应曲线。

参考视频原始参数：
  一阶： G(s) = 5 / (s + 5)
  二阶： G(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2),  zeta = 0.5, wn = 10

本作业修改后的参数（我自己设计的系统）：
  一阶： G(s) = 4 / (s + 2)        -> 时间常数 tau = 0.5 s, 直流增益 K = 2
  二阶： G(s) = 25 / (s^2 + 3s + 25) -> wn = 5 rad/s, zeta = 0.3 (欠阻尼)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal as signal

# ----------------------------------------------------------------------
# 中文字体设置
# ----------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.35
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 12

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

C_IMP = "#1f77b4"   # 冲激响应颜色
C_STEP = "#d62728"  # 阶跃响应颜色
C_INIT = "#2ca02c"  # 初始条件响应颜色


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)


# ======================================================================
# 一阶系统:  G(s) = 4 / (s + 2)
#   标准形式 G(s) = K / (tau*s + 1) = 2 / (0.5 s + 1)
#   K(直流增益) = 2,  tau(时间常数) = 0.5 s,  极点 s = -2
# ======================================================================
K1, tau1 = 2.0, 0.5          # 直流增益 K = 2, 时间常数 tau = 0.5 s
num1 = [4.0]                 # G(s) = 4 / (s + 2)
den1 = [1.0, 2.0]           # 等价 K/(tau*s+1) = 2/(0.5 s + 1), 极点 s = -1/tau = -2
G1 = signal.TransferFunction(num1, den1)
assert abs(num1[0] / den1[1] - K1) < 1e-9   # DC 增益 G(0) = 4/2 = 2 = K

t1 = np.linspace(0, 3.0, 1000)

# --- 一阶: 三种响应 (冲激/阶跃/初始条件), 复刻视频 subplot(3,1,x) 布局 ---
_, y_imp1 = signal.impulse(G1, T=t1)
_, y_step1 = signal.step(G1, T=t1)

# 初始条件响应: 状态空间  dx/dt = -2 x + 4 u, y = x,  x0 = 5
A1, B1, C1, D1 = -2.0, 4.0, 1.0, 0.0
ss1 = signal.StateSpace(A1, B1, C1, D1)
x0_1 = 5.0
_, y_init1, _ = signal.lsim(ss1, U=np.zeros_like(t1), T=t1, X0=[x0_1])

fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
fig.suptitle("一阶系统 G(s) = 4 / (s + 2)  的时域响应\n(K = 2,  τ = 0.5 s,  极点 s = -2)",
             fontsize=14, fontweight="bold")
axes[0].plot(t1, y_imp1, color=C_IMP, lw=2)
axes[0].set_title("① 单位冲激响应  (impulse)")
axes[0].set_ylabel("y(t)")
axes[1].plot(t1, y_step1, color=C_STEP, lw=2)
axes[1].axhline(K1, ls="--", color="gray", lw=1)
axes[1].text(2.2, K1 - 0.25, f"稳态值 = K = {K1:g}", color="gray")
axes[1].set_title("② 单位阶跃响应  (step)")
axes[1].set_ylabel("y(t)")
axes[2].plot(t1, y_init1, color=C_INIT, lw=2)
axes[2].set_title(f"③ 初始状态响应  (initial),  $x_0$ = {x0_1:g}")
axes[2].set_ylabel("y(t)")
axes[2].set_xlabel("时间 t (s)")
save(fig, "fo_three_responses.png")

# --- 一阶: 阶跃响应 + 时间常数标注 ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t1, y_step1, color=C_STEP, lw=2.2, label="阶跃响应")
ax.axhline(K1, ls="--", color="gray", lw=1)
# 63.2% 点
y632 = 0.632 * K1
ax.axhline(y632, ls=":", color="purple", lw=1)
ax.axvline(tau1, ls=":", color="purple", lw=1)
ax.plot([tau1], [y632], "o", color="purple")
ax.annotate(f"t = τ = {tau1:g}s 时,\ny 达到稳态的 63.2%",
            xy=(tau1, y632), xytext=(tau1 + 0.4, y632 - 0.55),
            arrowprops=dict(arrowstyle="->", color="purple"), color="purple")
# 2% 调节时间 ~ 4 tau
ts2 = 4 * tau1
ax.axvline(ts2, ls="--", color="darkorange", lw=1)
ax.annotate(f"调节时间 $t_s$ ≈ 4τ = {ts2:g}s",
            xy=(ts2, K1 * 0.5), xytext=(ts2 + 0.05, K1 * 0.4),
            color="darkorange")
ax.text(0.05, K1 + 0.08, f"稳态值 = K = {K1:g}", color="gray")
ax.set_ylim(-0.05, 2.25)
ax.set_title("一阶系统阶跃响应的关键指标")
ax.set_xlabel("时间 t (s)")
ax.set_ylabel("y(t)")
ax.legend(loc="lower right")
save(fig, "fo_step_annotated.png")

# --- 一阶(拓展): 不同时间常数 tau 的阶跃响应对比 (K = 2 固定) ---
fig, ax = plt.subplots(figsize=(8, 5))
for tau, col in zip([0.25, 0.5, 1.0], ["#1f77b4", "#d62728", "#2ca02c"]):
    # K/(tau*s+1), 直流增益固定为 K1=2, 仅时间常数不同
    G = signal.TransferFunction([K1], [tau, 1.0])
    _, y = signal.step(G, T=t1)
    ax.plot(t1, y, lw=2, color=col, label=f"τ = {tau:g}s  (极点 s = {-1/tau:g})")
ax.axhline(K1, ls="--", color="gray", lw=1)
ax.text(2.3, K1 + 0.03, f"共同稳态值 K = {K1:g}", color="gray")
ax.set_title("时间常数 τ 对一阶系统阶跃响应的影响\n(τ 越小, 响应越快)")
ax.set_xlabel("时间 t (s)")
ax.set_ylabel("y(t)")
ax.legend(loc="lower right")
save(fig, "fo_tau_compare.png")


# ======================================================================
# 二阶系统:  G(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
#   我的参数:  wn = 5 rad/s,  zeta = 0.3 (欠阻尼)
#   => G(s) = 25 / (s^2 + 3 s + 25)
# ======================================================================
zeta, wn = 0.3, 5.0
num2 = [wn ** 2]
den2 = [1.0, 2 * zeta * wn, wn ** 2]
G2 = signal.TransferFunction(num2, den2)

t2 = np.linspace(0, 4.0, 2000)

_, y_imp2 = signal.impulse(G2, T=t2)
_, y_step2 = signal.step(G2, T=t2)

# 初始条件响应: 状态空间(可控标准型)
A2 = np.array([[0.0, 1.0], [-wn ** 2, -2 * zeta * wn]])
B2 = np.array([[0.0], [wn ** 2]])
C2 = np.array([[1.0, 0.0]])
D2 = np.array([[0.0]])
ss2 = signal.StateSpace(A2, B2, C2, D2)
z0 = [1.0, 0.0]   # 初始位置 1, 初始速度 0
_, y_init2, _ = signal.lsim(ss2, U=np.zeros_like(t2), T=t2, X0=z0)

fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
fig.suptitle("二阶系统 G(s) = 25 / (s² + 3s + 25)  的时域响应\n(ωn = 5 rad/s,  ζ = 0.3,  欠阻尼)",
             fontsize=14, fontweight="bold")
axes[0].plot(t2, y_imp2, color=C_IMP, lw=2)
axes[0].set_title("① 单位冲激响应  (impulse)")
axes[0].set_ylabel("y(t)")
axes[1].plot(t2, y_step2, color=C_STEP, lw=2)
axes[1].axhline(1.0, ls="--", color="gray", lw=1)
axes[1].set_title("② 单位阶跃响应  (step)")
axes[1].set_ylabel("y(t)")
axes[2].plot(t2, y_init2, color=C_INIT, lw=2)
axes[2].set_title("③ 初始状态响应  (initial),  $z_0 = [1,\\,0]^T$")
axes[2].set_ylabel("y(t)")
axes[2].set_xlabel("时间 t (s)")
save(fig, "so_three_responses.png")

# --- 二阶: 阶跃响应 + 性能指标标注 ---
# 数值计算关键指标
yss = 1.0
peak_idx = int(np.argmax(y_step2))
tp = t2[peak_idx]
Mp_num = (y_step2[peak_idx] - yss) / yss
# 理论值
Mp_theory = np.exp(-zeta * np.pi / np.sqrt(1 - zeta ** 2))
wd = wn * np.sqrt(1 - zeta ** 2)
tp_theory = np.pi / wd
ts_theory = 4.0 / (zeta * wn)  # 2% 准则

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t2, y_step2, color=C_STEP, lw=2.2, label="阶跃响应")
ax.axhline(yss, ls="--", color="gray", lw=1)
ax.set_ylim(-0.05, 1.62)
# 超调量
ax.plot([tp], [y_step2[peak_idx]], "o", color="purple")
ax.annotate(f"峰值时间 $t_p$ = {tp:.3f}s\n超调量 $M_p$ = {Mp_num*100:.1f}%",
            xy=(tp, y_step2[peak_idx]), xytext=(1.15, 1.30),
            arrowprops=dict(arrowstyle="->", color="purple"), color="purple")
# 2% 误差带
ax.axhline(1.02 * yss, ls=":", color="darkorange", lw=1)
ax.axhline(0.98 * yss, ls=":", color="darkorange", lw=1)
ax.axvline(ts_theory, ls="--", color="darkorange", lw=1)
ax.annotate(f"调节时间 $t_s$ ≈ {ts_theory:.2f}s\n(2% 准则)",
            xy=(ts_theory, 0.6), xytext=(ts_theory + 0.1, 0.45),
            color="darkorange")
ax.set_title("二阶欠阻尼系统阶跃响应的关键指标")
ax.set_xlabel("时间 t (s)")
ax.set_ylabel("y(t)")
ax.legend(loc="lower right")
save(fig, "so_step_annotated.png")

# --- 二阶(核心): 阻尼比 zeta 对阶跃响应的影响 (wn = 5 固定) ---
zetas = [0.0, 0.1, 0.3, 0.707, 1.0, 2.0]
labels = ["ζ = 0 (无阻尼)", "ζ = 0.1 (欠阻尼)", "ζ = 0.3 (欠阻尼, 本设计)",
          "ζ = 0.707 (最佳阻尼)", "ζ = 1.0 (临界阻尼)", "ζ = 2.0 (过阻尼)"]
colors = ["#9467bd", "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#8c564b"]
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for z, lab, col in zip(zetas, labels, colors):
    den = [1.0, 2 * z * wn, wn ** 2]
    G = signal.TransferFunction([wn ** 2], den)
    _, y = signal.step(G, T=t2)
    lw = 2.6 if abs(z - 0.3) < 1e-6 else 1.8
    ax.plot(t2, y, lw=lw, color=col, label=lab)
ax.axhline(1.0, ls="--", color="gray", lw=1)
ax.set_xlim(0, 4)
ax.set_title("阻尼比 ζ 对二阶系统阶跃响应的影响\n(固定 ωn = 5 rad/s)", fontweight="bold")
ax.set_xlabel("时间 t (s)")
ax.set_ylabel("y(t)")
ax.legend(loc="lower right", fontsize=10)
save(fig, "so_zeta_compare.png")

# ----------------------------------------------------------------------
# 打印关键指标 (供 PPT 引用)
# ----------------------------------------------------------------------
print("\n================ 关键指标汇总 ================")
print("一阶系统 G(s) = 4/(s+2):")
print(f"  时间常数 tau   = {tau1} s")
print(f"  直流增益 K     = {K1}")
print(f"  极点          = -1/tau = {-1/tau1}")
print(f"  调节时间(2%)   ~ 4*tau = {4*tau1} s")
print(f"  上升时间(10-90%) ~ 2.2*tau = {2.197*tau1:.3f} s")
print("\n二阶系统 G(s) = 25/(s^2+3s+25):")
print(f"  自然频率 wn    = {wn} rad/s")
print(f"  阻尼比 zeta    = {zeta}")
print(f"  阻尼振荡频率 wd = {wd:.4f} rad/s")
print(f"  超调量 Mp(数值) = {Mp_num*100:.2f} %   (理论 {Mp_theory*100:.2f} %)")
print(f"  峰值时间 tp(数值)= {tp:.4f} s   (理论 {tp_theory:.4f} s)")
print(f"  调节时间 ts(2%) ~ {ts_theory:.4f} s")
print("=============================================")
