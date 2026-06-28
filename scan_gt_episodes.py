#!/usr/bin/env python3
"""
扫描 gt_debug 目录下的所有 episode CSV，自动汇总每条 episode 的关键指标，
并筛出 success=True 或抬起高度最高的样本，方便快速定位"真抓取"案例。

适配 play.py --save_gt_debug 导出的 CSV，列包含:
    seed, step, reward, object_position, ee_position, box_position,
    ee_obj_dist, obj_box_dist, gripper_joint_pos, ..., success, done

用法:
    python3 scan_gt_episodes.py <gt_debug_dir>
    python3 scan_gt_episodes.py <gt_debug_dir> --top 10 --save summary.csv
    # 在你的 uv 环境里:  uv run python scan_gt_episodes.py <gt_debug_dir>

判定阈值(可按需改):
    REAL_GRASP_M = 0.03   # 抬起 >3cm 视为真抓起
    MARGINAL_M   = 0.01   # 抬起 1~3cm 视为临界
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

REAL_GRASP_M = 0.03
MARGINAL_M = 0.01


def parse_vec(s):
    """把 '[0.22, -0.10, 0.015]' 这类字符串解析为 np.array。"""
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=float)
    return np.array([float(x) for x in str(s).strip().strip("[]").split(",")])


def analyze_one(path):
    df = pd.read_csv(path)
    if len(df) == 0:
        return None

    obj_z = np.vstack(df["object_position"].apply(parse_vec).values)[:, 2]
    init_z, peak_z = obj_z[0], obj_z.max()
    lift = peak_z - init_z
    peak_i = int(obj_z.argmax())

    grip = df["gripper_joint_pos"].astype(float).values
    eod = df["ee_obj_dist"].astype(float).values
    obd = df["obj_box_dist"].astype(float).values

    success = bool(df["success"].astype(bool).any()) if "success" in df.columns else False

    if success:
        verdict = "SUCCESS"
    elif lift > REAL_GRASP_M:
        verdict = "GRASPED(>3cm)"
    elif lift > MARGINAL_M:
        verdict = "MARGINAL(1-3cm)"
    else:
        verdict = "PUSH/NO-GRASP"

    return {
        "file": os.path.basename(path),
        "steps": len(df),
        "success": success,
        "verdict": verdict,
        "lift_mm": round(lift * 1000, 1),
        "peak_z_mm": round(peak_z * 1000, 1),
        "peak_step": int(df["step"].iloc[peak_i]),
        "theta_peak": round(float(grip[peak_i]), 3),      # 抬起峰值时刻的夹爪角度(最代表"真抓时的角度")
        "theta_closest": round(float(grip[eod.argmin()]), 3),  # 离方块最近时刻的夹爪角度
        "ee_obj_min_mm": round(float(eod.min()) * 1000, 1),
        "obj_box_min_mm": round(float(obd.min()) * 1000, 1),
        "obj_box_final_mm": round(float(obd[-1]) * 1000, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="批量扫描 gt_debug episode CSV")
    ap.add_argument("gt_dir", help="gt_debug CSV 所在目录")
    ap.add_argument("--pattern", default="gt_*.csv", help="文件名匹配模式 (默认 gt_*.csv)")
    ap.add_argument("--top", type=int, default=15, help="打印前 N 条 (默认 15)")
    ap.add_argument("--save", default=None, help="可选：把完整汇总另存为 CSV")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.gt_dir, args.pattern)))
    if not files:
        print(f"在 {args.gt_dir} 下没找到匹配 {args.pattern} 的文件")
        return

    rows = []
    for f in files:
        try:
            r = analyze_one(f)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"[跳过] {os.path.basename(f)}: {e}")

    if not rows:
        print("没有可解析的 episode。")
        return

    res = pd.DataFrame(rows)
    # 排序: 成功优先，其次按抬起高度降序
    res = res.sort_values(["success", "lift_mm"], ascending=[False, False]).reset_index(drop=True)

    n_succ = int(res["success"].sum())
    n_grasp = int((res["verdict"] == "GRASPED(>3cm)").sum())
    print(f"\n共扫描 {len(res)} 条 episode")
    print(f"  success=True : {n_succ} 条 ({n_succ / len(res) * 100:.1f}%)")
    print(f"  抬起>3cm     : {n_grasp} 条 ({n_grasp / len(res) * 100:.1f}%)\n")

    cols = ["file", "verdict", "success", "lift_mm", "peak_z_mm",
            "theta_peak", "theta_closest", "ee_obj_min_mm", "obj_box_final_mm", "steps"]
    print(res[cols].head(args.top).to_string(index=False))

    print("\n--- 推荐拿去深挖的样本 ---")
    succ = res[res["success"]]
    if len(succ):
        print(f"成功样本(取第一条): {succ.iloc[0]['file']}")
    grasped = res[res["verdict"] == "GRASPED(>3cm)"]
    if len(grasped):
        print(f"真抓取样本(取抬起最高): {grasped.iloc[0]['file']}")
    best = res.iloc[0]
    print(f"全场抬起最高: {best['file']}  (lift={best['lift_mm']}mm, θ_peak={best['theta_peak']}rad)")

    if n_succ == 0 and n_grasp == 0:
        print("\n[提示] 这批样本里没有成功/真抓取的 episode。")
        print("       想拿到成功样本，可多录几条，或把调试 env 指到视频里成功的那个 env。")

    if args.save:
        res.to_csv(args.save, index=False)
        print(f"\n完整汇总已保存: {args.save}")


if __name__ == "__main__":
    main()
