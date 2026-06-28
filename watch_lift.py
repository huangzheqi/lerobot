#!/usr/bin/env python3
"""命令行速览 v13 训练的关键曲线(抬起 bootstrap 监控)，不用开浏览器。

自动定位 logs/rsl_rl/<exp>/ 下最新 run 的 tfevents，打印与 lift/reward 相关的标量最近若干点，
用来判断"前 ~1500 iter 抬起有没有从 0 爬起来"，决定是否早停。

用法:
    uv run python watch_lift.py                 # 默认 exp=pick_place_v13_place_scratch
    uv run python watch_lift.py <exp_name>      # 指定实验名
    uv run python watch_lift.py <exp_name> 20   # 末尾打印 20 个点
    watch -n 30 'uv run python watch_lift.py'   # 每 30s 刷新(配合训练实时看)
"""
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYWORDS = ("lift", "grasp", "stage2", "stage4_stable", "joint_vel", "curriculum")  # 抬起 + 搬运 + 放置 + 惩罚/curriculum(确认无中途冲击)


def latest_run_dir(exp):
    base = os.path.join("logs", "rsl_rl", exp)
    runs = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)]
    runs = [d for d in runs if glob.glob(os.path.join(d, "events.out.tfevents.*"))]
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "pick_place_v16b_obsnoise_currfix"
    tail = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    run = latest_run_dir(exp)
    if run is None:
        print(f"[等待] logs/rsl_rl/{exp}/ 下还没有 tfevents (训练刚开始?)")
        return
    print(f"run: {run}")

    ea = EventAccumulator(run, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        print("[等待] 还没有标量写入。")
        return

    # 选: 含关键词的 + 总 reward(mean) 那条
    picked = [t for t in tags if any(k in t.lower() for k in KEYWORDS)]
    rew = [t for t in tags if "rew" in t.lower() and "Episode_Reward" not in t]
    picked = sorted(set(picked)) + sorted(set(rew) - set(picked))
    if not picked:
        print("没匹配到 lift/grasp/reward 相关 tag。现有 tag:")
        for t in tags:
            print("   ", t)
        return

    for t in picked:
        ev = ea.Scalars(t)
        last = ev[-tail:]
        first_v = ev[0].value
        last_v = ev[-1].value
        spark = "  ".join(f"{e.value:.3f}" for e in last)
        trend = "↑爬升" if last_v > first_v + 1e-4 else ("≈0/平" if abs(last_v) < 1e-3 else "持平")
        print(f"\n[{t}]  step {ev[0].step}->{ev[-1].step}  首={first_v:.3f} 末={last_v:.3f}  {trend}")
        print(f"    末{len(last)}点: {spark}")


if __name__ == "__main__":
    main()
