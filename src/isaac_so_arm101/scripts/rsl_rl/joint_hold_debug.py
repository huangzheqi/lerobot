# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Hold the SO101 ACT-V5 reset pose without loading ACT, PPO, or a checkpoint."""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

from isaac_so_arm101.scripts.rsl_rl.joint_hold_metrics import (
    classify_hold,
    validate_hold_thresholds,
)


parser = argparse.ArgumentParser(description="Diagnose SO101 reset-pose tracking without a learned model.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V34-v0",
    help="Factory-zero task whose robot and ActionManager configuration should be tested.",
)
parser.add_argument("--num-steps", type=int, default=600, help="Number of control steps to hold the reset pose.")
parser.add_argument(
    "--print-interval",
    type=int,
    default=20,
    help="Print one diagnostic row every N steps; 0 disables it.",
)
parser.add_argument("--stable-threshold-deg", type=float, default=5.0)
parser.add_argument("--collapse-threshold-deg", type=float, default=20.0)
parser.add_argument(
    "--arm-effort-limit",
    type=float,
    default=None,
    help="Override the arm actuator effort_limit_sim; by default the task value is preserved.",
)
parser.add_argument(
    "--disable-self-collisions",
    action="store_true",
    default=False,
    help="Disable articulation self-collisions for a controlled comparison.",
)
parser.add_argument("--output-dir", type=Path, default=Path("logs/joint_hold_debug"))
parser.add_argument(
    "--realtime",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Throttle to simulation time. Defaults to enabled for GUI and disabled headless.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _validate_args() -> None:
    if args_cli.num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if args_cli.print_interval < 0:
        raise ValueError("--print-interval must be non-negative")
    if args_cli.arm_effort_limit is not None and args_cli.arm_effort_limit <= 0.0:
        raise ValueError("--arm-effort-limit must be positive")
    validate_hold_thresholds(args_cli.stable_threshold_deg, args_cli.collapse_threshold_deg)


_validate_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim has launched."""

import gymnasium as gym
import torch

import isaac_so_arm101.tasks  # noqa: F401
from isaac_so_arm101.policies.act_contract import ACT_JOINT_NAMES
from isaaclab_tasks.utils import parse_env_cfg


ARM_JOINT_NAMES = tuple(ACT_JOINT_NAMES[:-1])
GRIPPER_JOINT_NAME = ACT_JOINT_NAMES[-1]


def _fieldnames() -> list[str]:
    fields = [
        "step",
        "time_s",
        "arm_action",
        "gripper_action",
        "arm_effort_limit_sim",
        "self_collisions_enabled",
        "max_arm_error_deg",
        "max_arm_drift_deg",
    ]
    for joint_name in ARM_JOINT_NAMES:
        fields.extend(
            [
                f"target_{joint_name}_deg",
                f"actual_{joint_name}_deg",
                f"error_{joint_name}_deg",
                f"drift_{joint_name}_deg",
                f"velocity_{joint_name}_deg_s",
            ]
        )
    fields.extend(
        [
            f"target_{GRIPPER_JOINT_NAME}_rad",
            f"actual_{GRIPPER_JOINT_NAME}_rad",
            f"error_{GRIPPER_JOINT_NAME}_rad",
            f"drift_{GRIPPER_JOINT_NAME}_rad",
            f"velocity_{GRIPPER_JOINT_NAME}_rad_s",
        ]
    )
    return fields


def _configure_env_cfg(env_cfg) -> float:
    env_cfg.scene.num_envs = 1
    if args_cli.arm_effort_limit is not None:
        env_cfg.scene.robot.actuators["arm"].effort_limit_sim = args_cli.arm_effort_limit
    if args_cli.disable_self_collisions:
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False

    control_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    duration = (args_cli.num_steps + 100) * control_dt
    env_cfg.episode_length_s = duration
    env_cfg.commands.object_pose.resampling_time_range = (duration, duration)

    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False
    for group_name in ("act_fixed", "act_handeye", "act_state"):
        if hasattr(env_cfg.observations, group_name):
            setattr(env_cfg.observations, group_name, None)
    for sensor_name in ("fixed_camera", "handeye_camera"):
        if hasattr(env_cfg.scene, sensor_name):
            setattr(env_cfg.scene, sensor_name, None)
    if hasattr(env_cfg.events, "reset_object_position"):
        env_cfg.events.reset_object_position = None
    if hasattr(env_cfg.terminations, "object_dropping"):
        env_cfg.terminations.object_dropping = None
    return control_dt


def _build_row(
    step: int,
    control_dt: float,
    target_rad: torch.Tensor,
    actual_rad: torch.Tensor,
    velocity_rad_s: torch.Tensor,
    baseline_rad: torch.Tensor,
    arm_effort_limit_sim: float,
    self_collisions_enabled: bool,
) -> dict[str, float | int]:
    error_rad = actual_rad - target_rad
    drift_rad = actual_rad - baseline_rad
    arm_error_deg = torch.rad2deg(error_rad[:-1])
    arm_drift_deg = torch.rad2deg(drift_rad[:-1])
    row: dict[str, float | int] = {
        "step": step,
        "time_s": (step + 1) * control_dt,
        "arm_action": 0.0,
        "gripper_action": 1.0,
        "arm_effort_limit_sim": arm_effort_limit_sim,
        "self_collisions_enabled": self_collisions_enabled,
        "max_arm_error_deg": float(arm_error_deg.abs().max().item()),
        "max_arm_drift_deg": float(arm_drift_deg.abs().max().item()),
    }
    target_deg = torch.rad2deg(target_rad[:-1])
    actual_deg = torch.rad2deg(actual_rad[:-1])
    velocity_deg_s = torch.rad2deg(velocity_rad_s[:-1])
    for index, joint_name in enumerate(ARM_JOINT_NAMES):
        row[f"target_{joint_name}_deg"] = float(target_deg[index].item())
        row[f"actual_{joint_name}_deg"] = float(actual_deg[index].item())
        row[f"error_{joint_name}_deg"] = float(arm_error_deg[index].item())
        row[f"drift_{joint_name}_deg"] = float(arm_drift_deg[index].item())
        row[f"velocity_{joint_name}_deg_s"] = float(velocity_deg_s[index].item())

    gripper_index = len(ACT_JOINT_NAMES) - 1
    row[f"target_{GRIPPER_JOINT_NAME}_rad"] = float(target_rad[gripper_index].item())
    row[f"actual_{GRIPPER_JOINT_NAME}_rad"] = float(actual_rad[gripper_index].item())
    row[f"error_{GRIPPER_JOINT_NAME}_rad"] = float(error_rad[gripper_index].item())
    row[f"drift_{GRIPPER_JOINT_NAME}_rad"] = float(drift_rad[gripper_index].item())
    row[f"velocity_{GRIPPER_JOINT_NAME}_rad_s"] = float(velocity_rad_s[gripper_index].item())
    return row


def _print_step(row: dict[str, float | int]) -> None:
    print(
        f"[step {int(row['step']):04d}] "
        f"shoulder_lift={float(row['actual_shoulder_lift_deg']):8.2f}deg "
        f"target={float(row['target_shoulder_lift_deg']):8.2f}deg "
        f"max_error={float(row['max_arm_error_deg']):7.2f}deg "
        f"max_drift={float(row['max_arm_drift_deg']):7.2f}deg"
    )


def main() -> int:
    env = None
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=1,
            use_fabric=not args_cli.disable_fabric,
        )
        control_dt = _configure_env_cfg(env_cfg)
        arm_effort_limit_sim = float(env_cfg.scene.robot.actuators["arm"].effort_limit_sim)
        self_collisions_enabled = bool(env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions)
        realtime = (not args_cli.headless) if args_cli.realtime is None else args_cli.realtime

        print(f"[INFO] Task: {args_cli.task}")
        print("[INFO] Policy: none; arm action=0.0, gripper action=1.0")
        print(f"[INFO] Hold: {args_cli.num_steps} steps, control_dt={control_dt:.6f}s")
        print(f"[INFO] Arm effort_limit_sim: {arm_effort_limit_sim:.6f}")
        print(f"[INFO] Self collisions enabled: {self_collisions_enabled}")
        print("[INFO] Cameras and ACT observation groups: disabled")

        env = gym.make(args_cli.task, cfg=env_cfg)
        args_cli.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = args_cli.output_dir / f"joint_hold_debug_{stamp}.csv"

        with torch.inference_mode():
            env.reset()
            robot = env.unwrapped.scene["robot"]
            joint_ids = robot.find_joints(list(ACT_JOINT_NAMES), preserve_order=True)[0]
            target_rad = robot.data.default_joint_pos[0, joint_ids].clone()
            baseline_rad = robot.data.joint_pos[0, joint_ids].clone()
            action = torch.zeros((1, len(ACT_JOINT_NAMES)), device=env.unwrapped.device, dtype=torch.float32)
            action[:, -1] = 1.0

            max_error_by_joint = torch.zeros_like(target_rad)
            max_drift_by_joint = torch.zeros_like(target_rad)
            final_actual_rad = baseline_rad.clone()
            completed_steps = 0

            print(
                "[INFO] Initial shoulder_lift: "
                f"target={torch.rad2deg(target_rad[1]).item():.6f}deg, "
                f"actual={torch.rad2deg(baseline_rad[1]).item():.6f}deg"
            )

            with out_path.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=_fieldnames())
                writer.writeheader()
                for step in range(args_cli.num_steps):
                    if not simulation_app.is_running():
                        print("[WARN] Isaac application stopped before the requested hold duration")
                        break
                    started = time.monotonic()
                    env.step(action)
                    actual_rad = robot.data.joint_pos[0, joint_ids].clone()
                    velocity_rad_s = robot.data.joint_vel[0, joint_ids].clone()
                    row = _build_row(
                        step,
                        control_dt,
                        target_rad,
                        actual_rad,
                        velocity_rad_s,
                        baseline_rad,
                        arm_effort_limit_sim,
                        self_collisions_enabled,
                    )
                    writer.writerow(row)

                    error_abs = (actual_rad - target_rad).abs()
                    drift_abs = (actual_rad - baseline_rad).abs()
                    max_error_by_joint = torch.maximum(max_error_by_joint, error_abs)
                    max_drift_by_joint = torch.maximum(max_drift_by_joint, drift_abs)
                    final_actual_rad = actual_rad
                    completed_steps += 1

                    if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
                        _print_step(row)
                    if realtime:
                        remaining = control_dt - (time.monotonic() - started)
                        if remaining > 0.0:
                            time.sleep(remaining)

        if completed_steps == 0:
            raise RuntimeError("No hold steps completed")

        max_arm_drift_deg = float(torch.rad2deg(max_drift_by_joint[:-1]).max().item())
        classification = classify_hold(
            max_arm_drift_deg,
            args_cli.stable_threshold_deg,
            args_cli.collapse_threshold_deg,
        )
        print(f"[RESULT] classification={classification}")
        print(f"[RESULT] completed_steps={completed_steps}")
        print(f"[RESULT] arm_effort_limit_sim={arm_effort_limit_sim:.6f}")
        print(f"[RESULT] self_collisions_enabled={self_collisions_enabled}")
        print(f"[RESULT] max_arm_drift_deg={max_arm_drift_deg:.6f}")
        for index, joint_name in enumerate(ARM_JOINT_NAMES):
            print(
                f"[RESULT] {joint_name}: "
                f"target_deg={torch.rad2deg(target_rad[index]).item():.6f} "
                f"final_deg={torch.rad2deg(final_actual_rad[index]).item():.6f} "
                f"max_error_deg={torch.rad2deg(max_error_by_joint[index]).item():.6f} "
                f"max_drift_deg={torch.rad2deg(max_drift_by_joint[index]).item():.6f}"
            )
        print(f"[RESULT] CSV: {out_path}")
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
