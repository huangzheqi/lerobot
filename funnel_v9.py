#!/usr/bin/env python3
"""v9 成功率/失败漏斗统计。

只统计带 gripper_joint_pos 列的 CSV（= 加了该列之后的 v9 play run），
按以下漏斗逐环报数量与占比：
  1. 总 episode
  2. 抓起来（lift = peak_z - init_z > GRASP_LIFT_M）
  3. 举着搬到 goal XY 上方（抓起后 XY 距 goal 最小值 < 阈值）
  4. 放置成功（结束时方块落回桌面且 XY 距 goal < 阈值）

重要：goal command 的 z 是空中点 (0.2~0.35m)，obj_box_dist 的 3D 距离永远
残留 20~35cm 的 z 分量 —— 用它判"搬到位"会得到假 0%。本脚本只用 XY 距离
+ 落桌(lift 回落)判定放置；play.py 内置的 success 列仅作脚注参考。

用法: python3 funnel_v9.py [gt_dir]   (默认 logs/gt_debug)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

GRASP_LIFT_M = 0.03            # 抬升 >3cm 视为真抓起
XY_THRESHOLDS = [0.05, 0.08]   # XY 距 goal 低于此视为"到位"
ON_TABLE_M = 0.015             # 结束时抬升回落到 <1.5cm 视为已落桌
SETTLE_STEPS = 5               # 用最后 N 步均值作为"结束状态"（消除弹跳）


def parse_vec(s):
    return np.array([float(x) for x in str(s).strip().strip("[]()").replace(";", ",").split(",")])


def analyze_one(path):
    df = pd.read_csv(path)
    if len(df) < SETTLE_STEPS or "gripper_joint_pos" not in df.columns:
        return None
    obj = np.vstack(df["object_position"].apply(parse_vec).values)
    obj_z = obj[:, 2]
    init_z, peak_z = obj_z[0], obj_z.max()
    lift = peak_z - init_z
    lift_series = obj_z - init_z
    succ_mask = df["success"].astype(bool).values if "success" in df.columns else np.zeros(len(df), bool)

    # XY 距 goal（box_position = object_pose command 前 3 维）
    xy_init = xy_final = xy_min_lifted = float("nan")
    final_lift = float(lift_series[-SETTLE_STEPS:].mean())
    if "box_position" in df.columns:
        box = np.vstack(df["box_position"].apply(parse_vec).values)
        xy = np.linalg.norm(obj[:, :2] - box[:, :2], axis=1)
        xy_init = float(xy[0])
        xy_final = float(np.linalg.norm(obj[-SETTLE_STEPS:, :2].mean(0) - box[-SETTLE_STEPS:, :2].mean(0)))
        lifted_mask = lift_series > GRASP_LIFT_M
        if lifted_mask.any():
            xy_min_lifted = float(xy[lifted_mask].min())

    return {
        "file": os.path.basename(path),
        "init_z": init_z,
        "lift": lift,
        "final_lift": final_lift,
        "xy_init": xy_init,
        "xy_final": xy_final,
        "xy_min_lifted": xy_min_lifted,
        "raw_success": bool(succ_mask.any()),
        "steps": len(df),
    }


def pct(n, d):
    return f"{n}  ({n / d * 100:.1f}%)" if d else f"{n}  (n/a)"


def main():
    gt_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/gt_debug"
    files = sorted(glob.glob(os.path.join(gt_dir, "gt_seed_*_env_*_episode_*.csv")))
    rows = []
    for f in files:
        try:
            r = analyze_one(f)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"[skip] {os.path.basename(f)}: {e}")
    if not rows:
        print("没有带 gripper_joint_pos 列的 episode CSV。")
        return
    df = pd.DataFrame(rows)

    total = len(df)
    grasped = df[df["lift"] > GRASP_LIFT_M]
    n_grasp = len(grasped)

    print(f"\n===== v9 漏斗 (gt_dir={gt_dir}) =====")
    print(f"判定: 抓起 lift>{GRASP_LIFT_M*100:.0f}cm ; 放置=结束落桌(lift<{ON_TABLE_M*100:.1f}cm)且 XY 距 goal<阈值\n")
    print(f"1. 总 episode               : {total}")
    print(f"2. 抓起来 (lift>3cm)         : {pct(n_grasp, total)}   <- 抓取成功率")

    has_xy = grasped["xy_final"].notna()
    g = grasped[has_xy]
    if len(g) == 0:
        print("   (CSV 无 box_position 列，无法做 XY 放置判定)")
        return

    for thr in XY_THRESHOLDS:
        n_carry = int((g["xy_min_lifted"] < thr).sum())
        print(f"3. └ 举着到过 goal 上方(XY<{thr*100:.0f}cm): {pct(n_carry, n_grasp)} (占全部 {n_carry/total*100:.1f}%)")
    for thr in XY_THRESHOLDS:
        placed = g[(g["final_lift"] < ON_TABLE_M) & (g["xy_final"] < thr)]
        n_pl = len(placed)
        print(f"4. └ 放置成功(落桌且XY<{thr*100:.0f}cm) : {pct(n_pl, n_grasp)} (占全部 {n_pl/total*100:.1f}%)   <- 真·端到端成功率")
    n_raw = int(df["raw_success"].sum())
    print(f"   (脚注: play.py 内置 3D success={n_raw} — 空中 goal 下无意义)")

    # ---- 抓起但没放进 area 的失败画像 ----
    thr_fail = XY_THRESHOLDS[-1]
    failed = g[~((g["final_lift"] < ON_TABLE_M) & (g["xy_final"] < thr_fail))]
    print(f"\n--- 抓起但没放进 area(XY<{thr_fail*100:.0f}cm) : {len(failed)} 条 ---")
    if len(failed):
        still_air = failed[failed["final_lift"] > GRASP_LIFT_M]
        on_table_far = failed[failed["final_lift"] < ON_TABLE_M]
        q = failed["xy_final"].quantile([0.25, 0.5, 0.75]).values
        print(f"  结束 XY 距 goal      : p25={q[0]*100:.1f}cm  中位={q[1]*100:.1f}cm  p75={q[2]*100:.1f}cm")
        print(f"  结束仍举在空中(>3cm) : {len(still_air)} | 已落桌但偏远: {len(on_table_far)} | 其余(半空): {len(failed)-len(still_air)-len(on_table_far)}")
        progress = 1.0 - failed["xy_final"] / failed["xy_init"]
        print(f"  XY 搬运进度 1-final/init: 中位={progress.median()*100:.0f}%  p25={progress.quantile(.25)*100:.0f}%  p75={progress.quantile(.75)*100:.0f}%")

    print(f"\n--- 参考 ---")
    print(f"  XY 起始距 goal 中位 = {df['xy_init'].median()*100:.1f}cm")
    print(f"  抬升 lift 分布: 中位={df['lift'].median()*1000:.1f}mm  最大={df['lift'].max()*1000:.1f}mm")
    if n_grasp:
        print(f"  抓起组结束 XY 距 goal: 中位={g['xy_final'].median()*100:.1f}cm")


if __name__ == "__main__":
    main()
