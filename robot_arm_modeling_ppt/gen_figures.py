# -*- coding: utf-8 -*-
"""Render equation PNGs and the 2-link arm diagram for the PPT."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT, exist_ok=True)
DPI = 220
INK = "#1F2A44"   # dark navy ink for equations


def save_eq(name, tex, fontsize=23, color=INK):
    fig = plt.figure()
    t = fig.text(0.5, 0.5, tex, fontsize=fontsize, ha="center", va="center", color=color)
    fig.canvas.draw()
    bb = t.get_window_extent()
    w, h = bb.width / fig.dpi, bb.height / fig.dpi
    fig.set_size_inches(w + 0.08, h + 0.08)
    fig.savefig(os.path.join(OUT, name), dpi=DPI, transparent=True,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", name)


# ---------------- simple one-line equations ----------------
save_eq("eq_lagrange.png",
        r"$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right)-\frac{\partial L}{\partial q_i}=\tau_i,\quad L=T-V$")

save_eq("eq_dynamics.png",
        r"$M(q)\,\ddot{q}+C(q,\dot{q})\,\dot{q}+G(q)+F_v\,\dot{q}=\tau$")

save_eq("eq_linear.png",
        r"$M_0\,\delta\ddot{q}+F_v\,\delta\dot{q}+K_g\,\delta q=\delta\tau,\qquad M_0=M(q_0),\;\;K_g=\left.\frac{\partial G}{\partial q}\right|_{q_0}$",
        fontsize=21)

save_eq("eq_ss.png",
        r"$\dot{x}=A\,x+B\,u,\qquad y=C\,x+D\,u$")

save_eq("eq_state.png",
        r"$x=\left[\delta q^{\top}\;\;\delta\dot{q}^{\top}\right]^{\top}\in\mathbb{R}^{4},\quad u=\delta\tau,\quad y=\delta q$",
        fontsize=20)

save_eq("eq_tf.png",
        r"$G(s)=C\,(sI-A)^{-1}B+D=\left[M_0 s^2+F_v s+K_g\right]^{-1}$",
        fontsize=22)

save_eq("eq_g11.png",
        r"$G_{11}(s)=\dfrac{\delta q_1(s)}{\delta\tau_1(s)}=\dfrac{3.784\,s^{2}+8.108\,s+106.05}{s^{4}+10.73\,s^{3}+108.8\,s^{2}+184.3\,s+1300.5}$",
        fontsize=21)


# ---------------- block matrices with drawn brackets ----------------
def bracket(ax, x, y0, y1, w=0.10, side="left", lw=2.0):
    xs = [x + w, x, x, x + w] if side == "left" else [x - w, x, x, x - w]
    ax.plot(xs, [y0, y0, y1, y1], color=INK, lw=lw, solid_capstyle="round")


def block_matrix(ax, x, yc, rows, col_w, row_h=0.85, fontsize=21, pad=0.16):
    """Draw [rows] starting at x, vertically centered at yc. Returns right edge x."""
    nr = len(rows)
    h = nr * row_h
    y0, y1 = yc - h / 2, yc + h / 2
    bracket(ax, x, y0, y1, side="left")
    cx = x + pad
    for j in range(len(rows[0])):
        for i, row in enumerate(rows):
            yy = y1 - (i + 0.5) * row_h
            ax.text(cx + col_w[j] / 2, yy, row[j], fontsize=fontsize,
                    ha="center", va="center", color=INK)
        cx += col_w[j]
    bracket(ax, cx + pad, y0, y1, side="right")
    return cx + pad + 0.10


fig, ax = plt.subplots(figsize=(13.2, 2.1))
ax.set_xlim(0, 14.8); ax.set_ylim(0, 2.1)
ax.axis("off"); ax.set_aspect("auto")
yc = 1.05
x = 0.0
ax.text(x, yc, r"$A=$", fontsize=22, ha="left", va="center", color=INK); x += 0.98
x = block_matrix(ax, x, yc, [[r"$\mathbf{0}_{2}$", r"$I_{2}$"],
                             [r"$-M_0^{-1}K_g$", r"$-M_0^{-1}F_v$"]],
                 col_w=[1.85, 1.95])
x += 0.55
ax.text(x, yc, r"$B=$", fontsize=22, ha="left", va="center", color=INK); x += 0.98
x = block_matrix(ax, x, yc, [[r"$\mathbf{0}_{2}$"], [r"$M_0^{-1}$"]], col_w=[1.15])
x += 0.55
ax.text(x, yc, r"$C=$", fontsize=22, ha="left", va="center", color=INK); x += 1.00
x = block_matrix(ax, x, yc, [[r"$I_{2}$", r"$\mathbf{0}_{2}$"]], col_w=[0.85, 0.85], row_h=1.0)
x += 0.55
ax.text(x, yc, r"$D=\mathbf{0}_{2\times 2}$", fontsize=22, ha="left", va="center", color=INK)
fig.savefig(os.path.join(OUT, "eq_blocks.png"), dpi=DPI, transparent=True,
            bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print("saved eq_blocks.png")


# ---------------- 2-link arm diagram ----------------
LINK = "#3A7CA5"; LINK2 = "#5B9BBF"; JOINT = "#2B2B2B"; PAY = "#E8833A"; ACC = "#C0504D"

fig, ax = plt.subplots(figsize=(5.6, 6.0))
ax.set_aspect("equal"); ax.axis("off")

l1, l2 = 1.0, 0.8
q1, q2 = np.deg2rad(-62), np.deg2rad(30)
p0 = np.array([0.0, 0.0])
p1 = p0 + l1 * np.array([np.cos(q1), np.sin(q1)])
p2 = p1 + l2 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])

# base (ceiling-style mount at top)
ax.plot([-0.42, 0.42], [0.02, 0.02], color=JOINT, lw=2.5)
for xx in np.linspace(-0.40, 0.40, 9):
    ax.plot([xx, xx + 0.10], [0.02, 0.14], color=JOINT, lw=1.2)
ax.text(0.50, 0.10, "基座", fontsize=13, color=JOINT)

# links
ax.plot(*zip(p0, p1), color=LINK, lw=9, solid_capstyle="round", zorder=2)
ax.plot(*zip(p1, p2), color=LINK2, lw=7, solid_capstyle="round", zorder=2)

# joints
for p, r in [(p0, 0.055), (p1, 0.048)]:
    ax.add_patch(plt.Circle(p, r, fc="white", ec=JOINT, lw=2.2, zorder=3))

# payload: gripper jaws + ball
tip_dir = np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
ball = p2 + 0.13 * tip_dir
ax.add_patch(plt.Circle(ball, 0.085, fc=PAY, ec="#B05A1A", lw=1.6, zorder=4))
nrm = np.array([-tip_dir[1], tip_dir[0]])
for sgn in (+1, -1):
    a = p2 + sgn * 0.085 * nrm
    b = p2 + sgn * 0.085 * nrm + 0.16 * tip_dir
    ax.plot(*zip(a, b), color=JOINT, lw=2.6, zorder=4, solid_capstyle="round")
ax.text(ball[0] + 0.13, ball[1] - 0.10, r"负载 $m_p$", fontsize=13, color="#B05A1A")

# dashed references + angle arcs
from matplotlib.patches import Arc
ax.plot([0, 0.62], [0, 0], ls="--", color="gray", lw=1.2)
ax.add_patch(Arc(p0, 0.85, 0.85, angle=0, theta1=np.rad2deg(q1), theta2=0,
                 color=ACC, lw=1.6))
ax.text(0.50, -0.30, r"$q_1$", fontsize=15, color=ACC)
ext = p1 + 0.45 * np.array([np.cos(q1), np.sin(q1)])
ax.plot(*zip(p1, ext), ls="--", color="gray", lw=1.2)
ax.add_patch(Arc(p1, 0.78, 0.78, angle=0, theta1=np.rad2deg(q1),
                 theta2=np.rad2deg(q1 + q2), color=ACC, lw=1.6))
amid = q1 + q2 / 2
ax.text(p1[0] + 0.50 * np.cos(amid), p1[1] + 0.50 * np.sin(amid), r"$q_2$",
        fontsize=15, color=ACC)

# torque arrows (curved)
for p, lab, off in [(p0, r"$\tau_1$", (-0.33, 0.16)), (p1, r"$\tau_2$", (-0.36, 0.10))]:
    ax.annotate("", xy=(p[0] - 0.16, p[1] + 0.12), xytext=(p[0] + 0.16, p[1] + 0.12),
                arrowprops=dict(arrowstyle="->", color=ACC, lw=1.7,
                                connectionstyle="arc3,rad=0.7"))
    ax.text(p[0] + off[0], p[1] + off[1], lab, fontsize=15, color=ACC)

# link labels
mid1 = (p0 + p1) / 2
ax.text(mid1[0] - 0.42, mid1[1] - 0.05, r"$m_1,\,l_1,\,I_1$", fontsize=13, color=LINK)
mid2 = (p1 + p2) / 2
ax.text(mid2[0] + 0.10, mid2[1] + 0.12, r"$m_2,\,l_2,\,I_2$", fontsize=13, color=LINK)

# gravity arrow
gx, gy = 0.95, -0.15
ax.annotate("", xy=(gx, gy - 0.40), xytext=(gx, gy),
            arrowprops=dict(arrowstyle="-|>", color=JOINT, lw=1.8))
ax.text(gx + 0.07, gy - 0.30, r"$g$", fontsize=15, color=JOINT)

ax.set_xlim(-0.75, 1.45); ax.set_ylim(-1.85, 0.35)
fig.savefig(os.path.join(OUT, "arm_diagram.png"), dpi=DPI, transparent=True,
            bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print("saved arm_diagram.png")
