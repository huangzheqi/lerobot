#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成学术风格的数据流框图(PNG 300dpi + PDF 矢量)。

布线规则:所有连线只走横平竖直(正交布线 / Manhattan routing),仅允许 90°
直角拐弯,不出现斜线、曲线,且不穿过任何盒子(走盒子之间的空白通道)。

纯 CPU、不依赖 Isaac。用法:
    CUDA_VISIBLE_DEVICES="" uv run python docs/make_figures.py
输出到 docs/figures/。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- 中文字体(显式按文件加载) ----
_NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_NOTO_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
F = fm.FontProperties(fname=_NOTO)
FB = fm.FontProperties(fname=_NOTO_B)
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ---- 学术配色(柔和、低饱和) ----
C = {
    "net":    ("#D6E4F2", "#3B6EA5"),
    "data":   ("#ECECEC", "#7A7A7A"),
    "proc":   ("#DCEAD6", "#5A8C4E"),
    "reward": ("#F8E3C8", "#C8843C"),
    "art":    ("#FBF1C7", "#B59A2E"),
    "vision": ("#E7DCEF", "#7E5CA8"),
}


def box(ax, x, y, w, h, text, kind="data", fs=11, bold=False, lw=1.3):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.6",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x, y, text, ha="center", va="center",
            fontproperties=FB if bold else F, fontsize=fs, color="#1A1A1A",
            zorder=3, linespacing=1.45)
    return (x, y, w, h)


def P(b, side, off=0.0):
    """盒子某条边上的锚点;off 沿边偏移(top/bot 改 x,left/right 改 y)。"""
    x, y, w, h = b
    return {
        "top": (x + off, y + h / 2),
        "bot": (x + off, y - h / 2),
        "left": (x - w / 2, y + off),
        "right": (x + w / 2, y + off),
    }[side]


def orth(ax, pts, color="#333333", lw=1.4, label=None, lpos=None, lcolor=None, fs=9):
    """正交折线箭头:pts 为路径点列表,相邻点必须共 x 或共 y(横平竖直)。"""
    pts = [(float(x), float(y)) for x, y in pts]
    for i in range(len(pts) - 2):
        ax.add_patch(FancyArrowPatch(pts[i], pts[i + 1], arrowstyle="-",
                     lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=1))
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=14,
                 lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=1))
    if label:
        lx, ly = lpos if lpos else ((pts[0][0] + pts[-1][0]) / 2, (pts[0][1] + pts[-1][1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontproperties=F, fontsize=fs,
                color=lcolor or "#333333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.92), zorder=4)


def new_ax(w=12, h=8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def title(ax, t, sub=None):
    ax.text(50, 97, t, ha="center", va="top", fontproperties=FB, fontsize=15, color="#111")
    if sub:
        ax.text(50, 92.5, sub, ha="center", va="top", fontproperties=F, fontsize=10, color="#666")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {name}.png / .pdf")


# =====================================================================
# 图1:总览
# =====================================================================
def fig1():
    fig, ax = new_ax(12, 8)
    title(ax, "图1  总体架构:训练流与推理流共用同一策略网络",
          "两条流的唯一本质区别 = 方块坐标的来源")
    net = box(ax, 50, 80, 46, 11,
              "策略网络  Actor MLP  [256,128,64]\n输入:观测(含方块 xyz) → 输出:6 维动作", "net", 11.5, True)
    train = box(ax, 24, 50, 38, 20,
                "训练流  train.py\n\n• 4096 并行仿真环境\n• 方块坐标 = 仿真真值(无相机)\n• PPO 反复试错、更新网络权重", "proc", 11)
    infer = box(ax, 76, 50, 38, 20,
                "推理流  play.py\n\n• 加载已训练权重\n• 方块坐标 = 真值 / 视觉\n• 50Hz 实时闭环执行", "vision", 11)
    ckpt = box(ax, 50, 16, 34, 8.5, "模型文件  model_26994.pt\n(网络权重 + 观测归一化统计量)", "art", 10.5)
    orth(ax, [P(net, "bot", -8), (42, 66), (24, 66), P(train, "top")], label="训练:更新权重", lpos=(33, 67))
    orth(ax, [P(net, "bot", 8), (58, 66), (76, 66), P(infer, "top")], label="推理:加载权重", lpos=(67, 67))
    orth(ax, [P(train, "bot"), (24, 16), P(ckpt, "left")], label="产出", lpos=(26.5, 23))
    orth(ax, [P(ckpt, "right"), (76, 16), P(infer, "bot")], label="加载", lpos=(78, 23))
    save(fig, "fig1_overview")


# =====================================================================
# 图2:训练流
# =====================================================================
def fig2():
    fig, ax = new_ax(11, 13)
    title(ax, "图2  训练数据流(train.py):无相机 · 用真值 · PPO 更新",
          "4096 个并行环境同时采样,逐步收集 (观测,动作,奖励) 供 PPO 学习")
    ax.add_patch(FancyBboxPatch((6, 8), 88, 80, boxstyle="round,pad=0.5,rounding_size=1.2",
                 linewidth=1.2, edgecolor="#9AA7B0", facecolor="#FAFBFC", zorder=0))
    ax.text(10, 85.5, "Isaac Lab · 4096 并行仿真环境", fontproperties=FB, fontsize=10.5, color="#52606D")

    phys = box(ax, 30, 78, 30, 8, "物理引擎 PhysX 5\n(每回合随机摆方块/目标)", "proc", 10)
    obs = box(ax, 30, 64, 40, 9, "拼装观测 obs ≈ 28 维\n[关节角6|关节速6|★方块xyz真值3★|目标7|上步动作6]", "data", 9.5)
    actor = box(ax, 30, 50, 30, 8, "Actor 策略网络\n动作 = 均值 + 探索噪声", "net", 10)
    act = box(ax, 30, 38, 26, 7, "6 维动作\n[5关节角 + 夹爪开合]", "data", 9.5)
    step = box(ax, 30, 26, 24, 7, "物理推进一步", "proc", 10)
    rew = box(ax, 72, 38, 34, 15, "奖励 = Σ(各项 × 权重)\n\nreaching / lifting /\nstage1~4 / 平滑惩罚\n(共 15 项,见第5章)", "reward", 10)
    ppo = box(ax, 72, 64, 34, 14, "PPO 更新 (RSL-RL)\n\n• Critic 估状态价值 V\n• 优势 A = 回报 − V\n• A>0 概率↑ / A<0 ↓\n• clip 限幅,小步走", "net", 10)
    ckpt = box(ax, 72, 16, 30, 7, "每 100 轮存\nmodel_xxxx.pt", "art", 10)

    orth(ax, [P(phys, "bot"), P(obs, "top")], label="读真值", lpos=(34, 71))
    orth(ax, [P(obs, "bot"), P(actor, "top")])
    orth(ax, [P(actor, "bot"), P(act, "top")])
    orth(ax, [P(act, "bot"), P(step, "top")])
    orth(ax, [P(step, "right"), (72, 26), P(rew, "bot")], label="算奖励", lpos=(60, 27.5))
    orth(ax, [P(rew, "top"), P(ppo, "bot")], label="收集一批\n4096×24 步", lpos=(80, 51))
    orth(ax, [P(ppo, "left"), (52, 64), (52, 50), P(actor, "right")], label="更新权重", lpos=(49, 57))
    orth(ax, [P(ppo, "right"), (92, 64), (92, 16), P(ckpt, "right")])
    orth(ax, [P(step, "left"), (7, 26), (7, 64), P(obs, "left")], color="#888888", label="下一步", lpos=(7, 45))
    save(fig, "fig2_training")


# =====================================================================
# 图3:推理流(三种坐标来源 + 实时闭环)
# =====================================================================
def fig3():
    fig, ax = new_ax(12.5, 11)
    title(ax, "图3  推理数据流(play.py):三种方块坐标来源 · 50Hz 实时闭环",
          "三种来源最终都写入同一观测槽位,策略网络完全不变")
    ax.text(50, 88.5, "开头一次性:runner.load(model_26994.pt) → policy = 策略函数",
            ha="center", fontproperties=FB, fontsize=10, color="#3B6EA5",
            bbox=dict(boxstyle="round,pad=0.3", fc="#EAF1F8", ec="#3B6EA5", lw=1))

    gt = box(ax, 57, 79, 22, 6.5, "gt:读仿真真值", "data", 10)
    cam = box(ax, 18, 66, 20, 7, "固定相机拍图\n256×256", "vision", 9.5)
    mask = box(ax, 44, 66, 22, 7, "红色掩码\n取最大连通域质心", "vision", 9.5)
    proj = box(ax, 70, 66, 20, 7, "针孔反投影\n像素 → 机器人 xy", "vision", 9.5)
    res = box(ax, 18, 52, 28, 7, "resnet:ResNet18\n回归 xy", "data", 9.5)
    obs = box(ax, 50, 40, 44, 7.5, "写入观测:obs[object_position 槽] = 方块 xyz  (play.py:822)", "data", 10.5, True)
    pol = box(ax, 50, 28, 40, 7.5, "actions = policy(obs)   ★调用模型★  (play.py:965)\n归一化 → Actor → 动作均值(无噪声)", "net", 10)
    stp = box(ax, 50, 16, 40, 7.5, "env.step(actions)  (play.py:968)\n关节 PD 控制 + 夹爪开合 → 机械臂动一步", "proc", 10)

    orth(ax, [P(cam, "right"), P(mask, "left")])
    orth(ax, [P(mask, "right"), P(proj, "left")])
    orth(ax, [P(gt, "bot"), (57, 48), (60, 48), (60, P(obs, "top")[1])], label="真值", lpos=(53, 49.5))
    orth(ax, [P(proj, "bot"), (70, 46), (54, 46), (54, P(obs, "top")[1])], label="方块xyz", lpos=(74, 53))
    orth(ax, [P(res, "right"), (42, 52), (42, P(obs, "top")[1])])
    orth(ax, [P(obs, "bot"), P(pol, "top")])
    orth(ax, [P(pol, "bot"), P(stp, "top")])
    # 红色实时闭环:全程横平竖直,最后一段沿 cam 中心 x=18 竖直进入 cam.top
    cx_cam = cam[0]
    orth(ax, [P(stp, "right"), (88, 16), (88, 85), (cx_cam, 85), P(cam, "top")],
         color="#C0392B", lw=1.6)
    ax.text(95.5, 50, "实时闭环\n每步重拍\n(掉落可重抓)", fontproperties=FB, fontsize=9,
            color="#C0392B", ha="center", va="center")
    save(fig, "fig3_inference")


# =====================================================================
# 图4:ResNet 离线监督学习
# =====================================================================
def fig4():
    fig, ax = new_ax(12.5, 8.5)
    title(ax, "图4  ResNet 方案:离线监督学习(标签由仿真器自动生成)",
          "训练成功(数据集 corr≈0.92)但部署存在域差 → 已被颜色掩码方案取代")
    for cx, t in [(18, "第1步  采集数据"), (50, "第2步  训练"), (82, "第3步  部署")]:
        ax.text(cx, 84, t, ha="center", fontproperties=FB, fontsize=11, color="#5A4A7A")

    sim = box(ax, 18, 72, 26, 7, "仿真器随机摆方块", "proc", 10)
    img1 = box(ax, 11, 58, 15, 6.5, "固定相机\nfixed_rgb", "vision", 9)
    lbl = box(ax, 28, 58, 18, 6.5, "★真值坐标★\nobject_position", "reward", 9)
    npz = box(ax, 18, 44, 28, 7, "sample_xxxx.npz\n= 图 + 标签(约5000张)", "data", 9.5)
    tr_in = box(ax, 50, 66, 26, 6.5, "图 → ResNet18\n→ 预测 xy", "net", 9.5)
    loss = box(ax, 50, 53, 26, 7, "损失 = MSE(预测, 真值)\n反向传播调权重", "reward", 9.5)
    val = box(ax, 50, 40, 26, 6.5, "验证 corr≈0.92\n(>0.8=真在定位)", "data", 9.5)
    mdl = box(ax, 50, 28, 26, 6, "resnet18_cube_pose.pt", "art", 9.5)
    dep = box(ax, 82, 60, 26, 7, "播放:相机图\n→ ResNet → xy", "net", 9.5)
    warn = box(ax, 82, 44, 28, 9, "⚠ 退化为猜均值(~6%)\n训练图 ≠ 播放图(域差)\n→ 改用颜色掩码(32%)", "reward", 9)

    orth(ax, [P(sim, "bot"), (18, 64), (11, 64), P(img1, "top")])
    orth(ax, [P(sim, "bot"), (18, 64), (28, 64), P(lbl, "top")])
    orth(ax, [P(img1, "bot"), (11, P(npz, "top")[1])])
    orth(ax, [P(lbl, "bot"), (28, P(npz, "top")[1])])
    orth(ax, [P(npz, "right"), (34, 44), (34, 66), P(tr_in, "left")], label="喂入训练", lpos=(34, 50))
    orth(ax, [P(tr_in, "bot"), P(loss, "top")])
    orth(ax, [P(loss, "bot"), P(val, "top")])
    orth(ax, [P(val, "bot"), P(mdl, "top")])
    orth(ax, [P(mdl, "right"), (99, 28), (99, 60), P(dep, "right")], label="部署", lpos=(80, 30))
    orth(ax, [P(dep, "bot"), P(warn, "top")])
    save(fig, "fig4_resnet")


# =====================================================================
# 图5:颜色掩码识别原理(合成示例,逐阶段演示)
# =====================================================================
def _largest_cc(mask):
    """最大连通域(4-邻接 BFS),返回布尔图。"""
    import numpy as np
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    best = np.zeros_like(mask, bool)
    best_n = 0
    for sy in range(H):
        for sx in range(W):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) > best_n:
                best_n = len(comp)
                best = np.zeros_like(mask, bool)
                ys, xs = zip(*comp)
                best[np.array(ys), np.array(xs)] = True
    return best


def fig5():
    import numpy as np
    rng = np.random.default_rng(1)
    H = W = 64
    # 合成一帧:灰色桌面 + 红色方块 + 黄褐色机械臂 + 几个红色噪点
    img = np.full((H, W, 3), 175, np.int16)
    img += rng.integers(-6, 7, (H, W, 3))                       # 桌面噪声
    img[6:26, 40:60] = np.array([208, 196, 120])                # 机械臂(黄褐)
    img[28:40, 22:36] = np.array([226, 44, 40])                 # 红色方块
    img[28:40, 22:36] += rng.integers(-10, 10, (12, 14, 3))     # 方块噪声
    for (yy, xx) in [(12, 12), (50, 52), (46, 14)]:             # 零星红噪点
        img[yy:yy+2, xx:xx+2] = np.array([200, 60, 55])
    img = img.clip(0, 255).astype(np.uint8)

    r = img[..., 0].astype(np.int16); g = img[..., 1].astype(np.int16); b = img[..., 2].astype(np.int16)
    mask = (r > 110) & (r > g + 35) & (r > b + 35)
    comp = _largest_cc(mask)
    ys, xs = np.nonzero(comp)
    cx, cy = xs.mean(), ys.mean()

    fig, axs = plt.subplots(1, 4, figsize=(14, 4.1))
    fig.suptitle("图5  颜色掩码识别原理:逐像素判定 → 取最大连通域 → 求质心",
                 fontproperties=FB, fontsize=15, y=1.02)
    titles = ["① 原始 RGB 图\n(红方块 + 灰桌 + 黄褐臂 + 红噪点)",
              "② 红色判定 → 二值掩码\nR>110 且 R>G+35 且 R>B+35",
              "③ 取最大连通域\n(滤掉零星红噪点)",
              "④ 质心(像素坐标取平均)\n→ 送入针孔反投影"]
    panels = [img, mask, comp, img]
    for ax, p, t in zip(axs, panels, titles):
        if p.ndim == 3:
            ax.imshow(p, origin="upper")
        else:
            ax.imshow(p, origin="upper", cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontproperties=F, fontsize=10.5, color="#222")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#888"); s.set_linewidth(1)
    # 第4幅标出质心十字
    axs[3].plot([cx], [cy], marker="+", ms=18, mew=2.5, color="#0B6CC0")
    axs[3].plot([cx], [cy], marker="o", ms=10, mfc="none", mec="#0B6CC0", mew=2)
    axs[3].text(cx, cy + 9, f"({cx:.0f}, {cy:.0f}) 像素", ha="center", fontproperties=F,
                fontsize=9, color="#0B6CC0")
    fig.text(0.5, -0.04,
             "关键:纯红方块是画面中唯一“明显偏红”的物体 → 灰桌(R≈G≈B)、黄褐臂(R不比G大35)都被排除;"
             "连通域滤除零星红噪点;质心即方块像素坐标。色相对光照变化稳健,故光照鲁棒性测试零衰减。",
             ha="center", fontproperties=F, fontsize=10, color="#444")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig5_colormask.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig5_colormask.png / .pdf")


# =====================================================================
# 图6:分阶段奖励结构
# =====================================================================
def fig6():
    fig, ax = new_ax(13, 9.2)
    title(ax, "图6  分阶段奖励结构:四个阶段由「方块高度 + 到目标XY距离」门控",
          "每步总奖励 = 所有项 × 权重 之和(并行打分、加权求和)→ 作为 PPO 的标量回报")
    ax.text(8, 86, "＋绿 = 奖励    －橙 = 惩罚    数字 = 权重", fontproperties=F, fontsize=9.5, color="#555")

    cols = [19, 41, 63, 85]
    heads = [
        ("阶段1  接近 + 抓取", "门控 s1:方块未抬起"),
        ("阶段2  搬运", "门控 s2:已抬起\n且未到目标上方"),
        ("阶段3  下降", "门控 s3:已抬起·到目标\n上方·尚未降下"),
        ("阶段4  释放", "门控 s4:已抬起·到目标\n·已降到桌面"),
    ]
    terms = [
        [("+", "接近 reaching", "1.0"), ("+", "抬起 lifting", "5.0")],
        [("+", "搬运 XY 跟踪", "10"), ("-", "过早松爪", "5")],
        [("+", "柔和下降", "12"), ("-", "硬砸惩罚", "8"), ("+", "末端降低", "14"),
         ("+", "方块近桌面", "14"), ("+", "手腕释放姿态", "2")],
        [("+", "释放张爪", "12"), ("-", "握太久惩罚", "10"), ("+", "近桌张爪", "11"),
         ("+", "稳定放置", "16"), ("+", "放后撤离", "2")],
    ]
    for cx, (hn, hg), tl in zip(cols, heads, terms):
        head = box(ax, cx, 73, 21, 11, hn + "\n" + hg, "net", 9.5, True)
        lst = box(ax, cx, 50, 21, 28, "", "data", 9)
        orth(ax, [P(head, "bot"), P(lst, "top")])
        ax.text(cx, 61.5, "奖励项 (权重)", ha="center", fontproperties=FB, fontsize=8.5, color="#555")
        yy = 57
        for sign, name, w in tl:
            col = "#2E7D32" if sign == "+" else "#C8843C"
            ax.text(cx, yy, f"{'＋' if sign=='+' else '－'} {name}  {w}", ha="center", va="center",
                    fontproperties=F, fontsize=8.6, color=col, zorder=3)
            yy -= 4.4

    note1 = box(ax, 50, 24, 84, 8.5,
                "锁存器 was_lifted:本回合「真正抬起过」才发放 stage3/4 奖励\n"
                "→ 堵死「不抬方块、直接贴桌面拖到目标」的作弊捷径", "art", 10)
    note2 = box(ax, 50, 12.5, 84, 8.5,
                "另有全程惩罚:action_rate 与 joint_vel(各 −0.1,惩罚动作突变 / 关节乱晃)\n"
                "门控变量只有两个:方块高度 z(判断 抬起 / 降到桌面)+ 方块到目标 XY 距离(判断 到没到目标上方)", "reward", 9)
    _ = note1, note2
    save(fig, "fig6_rewards")


# =====================================================================
# 图7:PPO 训练循环与超参
# =====================================================================
def fig7():
    fig, ax = new_ax(12.5, 9)
    title(ax, "图7  PPO 训练循环:超参数标注在它们各自起作用的位置",
          "一轮迭代 = 采样 → 算优势 → 多轮更新 → 调LR;训练日志每一行就是一轮(见第7章)")

    a = box(ax, 32, 72, 34, 12, "① 并行采样 (rollout)\nnum_envs=4096 × num_steps=24\n≈ 9.8 万 样本 / 批", "proc", 9.8)
    b = box(ax, 32, 55, 34, 12, "② 计算优势 A (GAE)\nγ=0.98 折扣未来回报\nλ=0.95 偏差/方差权衡", "data", 9.8)
    c = box(ax, 32, 38, 34, 13, "③ 多轮小批更新\nepochs=5 × minibatch=4\nclip=0.2(限幅·小步走)\n熵系数 0.0025 · 价值系数 1.0", "net", 9.8)
    d = box(ax, 32, 20, 34, 12, "④ 自适应学习率 + 存档\nlr=5e-5,目标 KL=0.005\nmax_grad_norm=0.5 · 每100轮存档", "reward", 9.8)
    panel = box(ax, 76, 46, 40, 52, "", "data", 9)
    ax.text(76, 70.5, "被训练的「网络」(全程结构固定)", ha="center", va="top",
            fontproperties=FB, fontsize=10.5, color="#333")
    ax.text(76, 65.5,
            "• Actor / Critic = MLP [256,128,64]\n"
            "  (多层感知机,属 DNN 一种)\n"
            "• 激活函数 ELU · 噪声 std=0.28\n"
            "• 左侧①~④是 PPO「训练法」,\n"
            "  改的是这些网络的权重\n"
            "• max_grad_norm=0.5 · max_iter=12000",
            ha="center", va="top", fontproperties=F, fontsize=9.3, color="#333", linespacing=1.7)
    ax.text(76, 38,
            "网络 = 被训练的「大脑」(学生)\n"
            "PPO = 更新权重的「教学法」(强化学习)\n"
            "clip 把更新限在 ±20%(小步走),\n"
            "避免一步跳太远把抓取学崩。",
            ha="center", va="top", fontproperties=F, fontsize=9.3, color="#555", linespacing=1.6)
    _ = panel

    orth(ax, [P(a, "bot"), P(b, "top")])
    orth(ax, [P(b, "bot"), P(c, "top")])
    orth(ax, [P(c, "bot"), P(d, "top")])
    orth(ax, [P(d, "left"), (10, 20), (10, 72), P(a, "left")],
         color="#3B6EA5", label="更新后的策略\n→ 下一轮采样", lpos=(10, 46))
    save(fig, "fig7_ppo")


# =====================================================================
# 图8:接近奖励的两个坐标系(末端 TCP 偏移 vs 方块质心)
# =====================================================================
def _frame(ax, x, y, s=4.2, label=None, lx=0, ly=0):
    """画一个小坐标系:原点 + X(红,右) + Z(蓝,上);Y 入纸面用点表示。"""
    ax.add_patch(FancyArrowPatch((x, y), (x + s, y), arrowstyle="-|>", mutation_scale=9,
                 lw=1.6, color="#C0392B", shrinkA=0, shrinkB=0, zorder=5))   # X 红
    ax.add_patch(FancyArrowPatch((x, y), (x, y + s), arrowstyle="-|>", mutation_scale=9,
                 lw=1.6, color="#2C6FAD", shrinkA=0, shrinkB=0, zorder=5))   # Z 蓝
    ax.plot([x], [y], "o", ms=5.5, color="#1A1A1A", zorder=6)
    ax.text(x + s + 0.6, y, "X", ha="left", va="center", fontproperties=FB, fontsize=8, color="#C0392B", zorder=6)
    ax.text(x, y + s + 0.8, "Z", ha="center", va="bottom", fontproperties=FB, fontsize=8, color="#2C6FAD", zorder=6)
    if label:
        ax.text(x + lx, y + ly, label, ha="center", va="center", fontproperties=F,
                fontsize=9, color="#1A1A1A",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.95), zorder=7)


def fig8():
    fig, ax = new_ax(12, 8.5)
    title(ax, "图8  接近奖励的两个坐标系:末端 TCP 偏移 vs 方块质心",
          "object_ee_distance 只用两个坐标系『原点』的距离,不用朝向")

    # ---- 夹爪(手掌横杆 + 两根手指) ----
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((22, 66), 16, 4, lw=1.3, edgecolor="#5A8C4E", facecolor="#DCEAD6", zorder=2))  # palm
    ax.add_patch(Rectangle((23.5, 48), 3, 18, lw=1.3, edgecolor="#5A8C4E", facecolor="#DCEAD6", zorder=2))  # finger L
    ax.add_patch(Rectangle((33.5, 48), 3, 18, lw=1.3, edgecolor="#5A8C4E", facecolor="#DCEAD6", zorder=2))  # finger R
    ax.text(30, 72.5, "gripper_link(夹爪连杆)", ha="center", va="bottom", fontproperties=FB, fontsize=9.5, color="#3A5A30")

    # gripper_link 原点(手腕根部,画在手掌中心)
    _frame(ax, 30, 68, label="gripper_link 原点\n(手腕根部)", lx=17.5, ly=2)
    # TCP / 末端坐标系原点(两指之间)
    _frame(ax, 30, 50, label="末端坐标系原点 = TCP\n(两指之间抓取中心)", lx=-15.5, ly=-3)

    # 偏移箭头:沿局部 -Z 从 gripper_link 原点下移到 TCP(竖直,正交)
    orth(ax, [(44, 68), (44, 50)], color="#B59A2E", lw=1.8,
         label="固定偏移\noffset=[0.01, 0, -0.09]\n沿局部 -Z 下移 9cm",
         lpos=(63, 59), lcolor="#7A6A14", fs=9)
    orth(ax, [(30, 68), (44, 68)], color="#B59A2E", lw=1.0)   # 引线到偏移标注
    orth(ax, [(44, 50), (30, 50)], color="#B59A2E", lw=1.0)

    # ---- 方块 + 质心坐标系 ----
    ax.add_patch(Rectangle((66, 44), 12, 12, lw=1.4, edgecolor="#962A2A", facecolor="#E8B4B4", zorder=2))
    _frame(ax, 72, 50, label="方块本体坐标系原点\n= 质心", lx=0, ly=-10)

    # ---- 接近距离 d:TCP(30,50) 与 质心(72,50) 同高 → 水平正交 ----
    orth(ax, [(30, 38), (30, 50)], color="#7A7A7A", lw=1.0)   # tick 到 TCP
    orth(ax, [(72, 38), (72, 50)], color="#7A7A7A", lw=1.0)   # tick 到 质心
    orth(ax, [(30, 38), (72, 38)], color="#333333", lw=1.6,
         label="d = 接近距离\n(两原点的直线距离)", lpos=(51, 34))

    # ---- 公式 ----
    box(ax, 50, 16, 76, 11,
        "reaching_object = 1 − tanh( d / 0.05 ),   d = || 质心 − TCP ||\n"
        "距离越近分越高:d=0 → 1.0 ;  d=5cm → 0.24 。只看两原点位置,与坐标轴朝向无关。",
        "reward", 9.8)

    save(fig, "fig8_ee_frame")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print(f"\n全部输出到 {OUT}/")
