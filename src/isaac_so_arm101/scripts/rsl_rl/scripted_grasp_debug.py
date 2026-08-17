# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Model-free SO101 grasp diagnostic using a recorded LeRobot demonstration.

The script does not load ACT, PPO, RSL-RL, or a checkpoint. It maps one recorded 6-DoF
LeRobot action sequence into Isaac joint-position actions. A first open-gripper pass records the
expected grasp XY position; a second pass places the cube there and replays the same trajectory.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_DATASET_DIR = Path(
    "/home/hzq/rl_disk/so101_clean_project/data/so101_pick_place_6dof_clean_v5_factory_zero"
)

parser = argparse.ArgumentParser(description="Replay a recorded SO101 demonstration without a learned model.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V5-v0",
    help="Factory-zero ACT-v5 task to instantiate.",
)
parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="LeRobot dataset directory.")
parser.add_argument("--episode-index", type=int, default=0, help="Recorded episode to replay.")
parser.add_argument("--settle-steps", type=int, default=50, help="Settling steps before each replay pass.")
parser.add_argument("--hold-steps", type=int, default=100, help="Closed-gripper hold steps after lift detection.")
parser.add_argument(
    "--close-consecutive-frames",
    type=int,
    default=5,
    help="Consecutive dataset frames required to detect grasp onset and diagnostic threshold crossings.",
)
parser.add_argument(
    "--lift-consecutive-steps",
    type=int,
    default=5,
    help="Consecutive simulation steps above the lift threshold before the hold phase.",
)
parser.add_argument("--gripper-open-threshold", type=float, default=15.0)
parser.add_argument("--gripper-effort-limit", type=float, default=2.5)
parser.add_argument("--gripper-stiffness", type=float, default=60.0)
parser.add_argument("--gripper-damping", type=float, default=20.0)
parser.add_argument("--gripper-closed-joint-target", type=float, default=0.0)
parser.add_argument("--replay-time-scale", type=float, default=3.0)
parser.add_argument("--rise-threshold", type=float, default=0.005, help="Minimum real cube rise in m.")
parser.add_argument("--lift-threshold", type=float, default=0.045, help="Successful lift threshold in m.")
parser.add_argument("--hold-threshold", type=float, default=0.035, help="Minimum retained hold height in m.")
parser.add_argument(
    "--ejection-xy-threshold",
    type=float,
    default=0.08,
    help="Classify the cube as ejected after this much unheld XY travel in m.",
)
parser.add_argument(
    "--ejection-ee-distance",
    type=float,
    default=0.08,
    help="Require this EE/cube separation together with XY travel for ejection classification.",
)
parser.add_argument("--cube-x-offset", type=float, default=0.0, help="Cube X offset from calibrated grasp center in m.")
parser.add_argument("--cube-y-offset", type=float, default=0.0, help="Cube Y offset from calibrated grasp center in m.")
parser.add_argument(
    "--robot-base-z-offset",
    type=float,
    default=0.0,
    help="Vertical robot-base offset relative to the table in m; negative values lower the replayed gripper.",
)
parser.add_argument(
    "--reset-from-episode-state",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Reset the robot from the selected episode's first observation.state.",
)
parser.add_argument(
    "--realtime",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Throttle GUI stepping to simulation time. Defaults to enabled for GUI and disabled headless.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--output-dir", type=Path, default=Path("logs/scripted_grasp_debug"))
parser.add_argument("--print-interval", type=int, default=20, help="Print one line every N simulation steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _validate_args() -> None:
    for name in ("settle_steps", "hold_steps", "close_consecutive_frames", "lift_consecutive_steps"):
        if getattr(args_cli, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "gripper_effort_limit",
        "gripper_stiffness",
        "gripper_damping",
        "replay_time_scale",
        "rise_threshold",
        "lift_threshold",
        "hold_threshold",
        "ejection_xy_threshold",
        "ejection_ee_distance",
    ):
        if getattr(args_cli, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not args_cli.rise_threshold < args_cli.hold_threshold < args_cli.lift_threshold:
        raise ValueError("Expected rise_threshold < hold_threshold < lift_threshold")
    if not 0.0 <= args_cli.gripper_closed_joint_target < 0.5:
        raise ValueError("--gripper-closed-joint-target must be in [0.0, 0.5)")
    if not -0.03 <= args_cli.robot_base_z_offset <= 0.03:
        raise ValueError("--robot-base-z-offset must be in [-0.03, 0.03] m")
    if args_cli.episode_index < 0:
        raise ValueError("--episode-index must be non-negative")
    if args_cli.print_interval < 0:
        raise ValueError("--print-interval must be non-negative")


_validate_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

import isaac_so_arm101.tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaac_so_arm101.policies.act_contract import ACT_ACTION_KEY, ACT_JOINT_NAMES, ACT_STATE_KEY
from isaac_so_arm101.policies.act_joint_mapping import gripper_percent_to_joint
from isaac_so_arm101.tasks.pick_place.joint_pos_env_cfg import (
    ACT_V5_GRIPPER_OPEN_JOINT_POS,
)
from isaaclab_tasks.utils import parse_env_cfg


FIELDNAMES = [
    "step",
    "pass_name",
    "phase",
    "source_frame",
    "dataset_time_s",
    *[f"dataset_action_{index}" for index in range(6)],
    *[f"env_action_{index}" for index in range(6)],
    *[f"joint_{name}" for name in ACT_JOINT_NAMES],
    "object_x",
    "object_y",
    "object_z",
    "height_gain_m",
    "object_xy_move_m",
    "ee_x",
    "ee_y",
    "ee_z",
    "ee_object_dist_m",
]


@dataclass(frozen=True)
class ReplayTrajectory:
    initial_dataset_state: np.ndarray
    dataset_actions: np.ndarray
    env_actions: np.ndarray
    source_frames: np.ndarray
    sim_times: np.ndarray
    dataset_fps: float
    gripper_open_value: float
    gripper_closed_value: float
    close_dataset_row: int
    reopen_dataset_row: int | None
    replay_end_dataset_row: int
    close_sim_step: int


def _load_dataset_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset info not found: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _episode_path(dataset_dir: Path, info: dict, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    return dataset_dir / info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )


def _sustained_start(mask: np.ndarray, count: int, start: int = 0) -> int | None:
    run = 0
    for index in range(start, len(mask)):
        run = run + 1 if bool(mask[index]) else 0
        if run >= count:
            return index - count + 1
    return None


def _dataset_state_to_joint_pos_dict(dataset_state: np.ndarray) -> dict[str, float]:
    if dataset_state.shape != (6,) or not np.isfinite(dataset_state).all():
        raise ValueError(f"Expected a finite 6D dataset state, got shape={dataset_state.shape}")
    joint_pos = {
        name: float(np.deg2rad(dataset_state[index]))
        for index, name in enumerate(ACT_JOINT_NAMES[:-1])
    }
    gripper_joint = gripper_percent_to_joint(
        torch.as_tensor(dataset_state[5:6], dtype=torch.float64),
        open_joint_pos=ACT_V5_GRIPPER_OPEN_JOINT_POS,
        closed_joint_pos=args_cli.gripper_closed_joint_target,
    )
    joint_pos["gripper"] = float(gripper_joint[0].item())
    return joint_pos


def _map_dataset_actions(
    dataset_actions: np.ndarray,
    initial_dataset_state: np.ndarray,
    gripper_open_value: float,
    gripper_closed_value: float,
    close_sim_step: int,
) -> np.ndarray:
    default_arm_rad = np.deg2rad(initial_dataset_state[:5])
    arm_target_rad = np.deg2rad(dataset_actions[:, :5])
    arm_raw = (arm_target_rad - default_arm_rad) / 0.5

    gripper_range = gripper_closed_value - gripper_open_value
    gripper_close_ratio = np.clip(
        (dataset_actions[:, 5] - gripper_open_value) / gripper_range,
        0.0,
        1.0,
    )
    gripper_raw = (1.0 - gripper_close_ratio).reshape(-1, 1)
    gripper_raw[close_sim_step:, 0] = np.minimum.accumulate(
        gripper_raw[close_sim_step:, 0]
    )
    return np.concatenate((arm_raw, gripper_raw), axis=1).astype(np.float32)


def _load_replay_trajectory(dataset_dir: Path, episode_index: int, control_dt: float) -> ReplayTrajectory:
    info = _load_dataset_info(dataset_dir)
    total_episodes = int(info.get("total_episodes", 0))
    if episode_index >= total_episodes:
        raise IndexError(f"episode_index={episode_index} is outside total_episodes={total_episodes}")
    data_path = _episode_path(dataset_dir, info, episode_index)
    if not data_path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {data_path}")

    df = pd.read_parquet(data_path)
    required_columns = {ACT_ACTION_KEY, ACT_STATE_KEY, "frame_index", "timestamp"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise KeyError(f"Episode parquet is missing columns: {missing}")

    actions = np.stack(df[ACT_ACTION_KEY].to_numpy()).astype(np.float64)
    if actions.ndim != 2 or actions.shape[1] != 6 or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite dataset actions with shape (T, 6), got {actions.shape}")
    initial_dataset_state = np.asarray(df[ACT_STATE_KEY].iloc[0], dtype=np.float64)
    if initial_dataset_state.shape != (6,) or not np.isfinite(initial_dataset_state).all():
        raise ValueError(
            f"Expected a finite initial observation.state with shape (6,), got {initial_dataset_state.shape}"
        )
    frame_indices = np.asarray(df["frame_index"].to_numpy(), dtype=np.int64).reshape(-1)
    timestamps = np.asarray(df["timestamp"].to_numpy(), dtype=np.float64).reshape(-1)
    timestamps = timestamps - timestamps[0]
    dataset_fps = float(info["fps"])
    if len(timestamps) != len(actions) or np.any(np.diff(timestamps) <= 0.0):
        timestamps = np.arange(len(actions), dtype=np.float64) / dataset_fps

    closed = actions[:, 5] >= float(args_cli.gripper_open_threshold)
    close_row = _sustained_start(closed, args_cli.close_consecutive_frames)
    if close_row is None:
        raise RuntimeError(
            f"Episode {episode_index} has no sustained gripper close transition at threshold "
            f"{args_cli.gripper_open_threshold}"
        )
    first_below_threshold_row = _sustained_start(
        ~closed,
        args_cli.close_consecutive_frames,
        close_row + args_cli.close_consecutive_frames,
    )
    replay_end_row = len(actions) - 1

    open_reference_end = max(1, close_row // 2)
    gripper_open_value = float(np.median(actions[:open_reference_end, 5]))
    closed_reference_end = first_below_threshold_row if first_below_threshold_row is not None else len(actions)
    gripper_closed_value = float(np.quantile(actions[close_row:closed_reference_end, 5], 0.9))
    if gripper_closed_value - gripper_open_value < 1.0:
        raise RuntimeError(
            "Dataset gripper range is too small for continuous mapping: "
            f"open={gripper_open_value:.4f}, closed={gripper_closed_value:.4f}"
        )

    final_control_time = timestamps[replay_end_row] * args_cli.replay_time_scale
    control_times = np.arange(
        0.0,
        final_control_time,
        control_dt,
        dtype=np.float64,
    )
    control_times = np.append(control_times, final_control_time)
    sim_times = control_times / args_cli.replay_time_scale
    sim_times[-1] = timestamps[replay_end_row]
    source_rows = np.searchsorted(timestamps, sim_times, side="right") - 1
    source_rows = np.clip(source_rows, 0, replay_end_row)

    resampled_actions = np.empty((len(sim_times), 6), dtype=np.float64)
    for joint_index in range(5):
        resampled_actions[:, joint_index] = np.interp(
            sim_times,
            timestamps[: replay_end_row + 1],
            actions[: replay_end_row + 1, joint_index],
        )
    resampled_actions[:, 5] = actions[source_rows, 5]
    close_candidates = np.flatnonzero(source_rows >= close_row)
    if len(close_candidates) == 0:
        raise RuntimeError("Resampled trajectory never reaches the detected close frame")
    close_sim_step = int(close_candidates[0])
    env_actions = _map_dataset_actions(
        resampled_actions,
        initial_dataset_state,
        gripper_open_value,
        gripper_closed_value,
        close_sim_step,
    )

    return ReplayTrajectory(
        initial_dataset_state=initial_dataset_state,
        dataset_actions=resampled_actions.astype(np.float32),
        env_actions=env_actions,
        source_frames=frame_indices[source_rows],
        sim_times=sim_times,
        dataset_fps=dataset_fps,
        gripper_open_value=gripper_open_value,
        gripper_closed_value=gripper_closed_value,
        close_dataset_row=int(close_row),
        reopen_dataset_row=(
            None if first_below_threshold_row is None else int(first_below_threshold_row)
        ),
        replay_end_dataset_row=int(replay_end_row),
        close_sim_step=close_sim_step,
    )


def _configure_env_cfg(env_cfg, total_steps: int, initial_dataset_state: np.ndarray) -> None:
    env_cfg.scene.num_envs = 1
    robot_root_pos = list(env_cfg.scene.robot.init_state.pos)
    robot_root_pos[2] += args_cli.robot_base_z_offset
    env_cfg.scene.robot.init_state.pos = tuple(robot_root_pos)
    if args_cli.reset_from_episode_state:
        env_cfg.scene.robot.init_state.joint_pos.update(
            _dataset_state_to_joint_pos_dict(initial_dataset_state)
        )
    env_cfg.events.reset_object_position.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    control_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    duration = (total_steps + 100) * control_dt
    env_cfg.episode_length_s = duration
    env_cfg.commands.object_pose.resampling_time_range = (duration, duration)
    if hasattr(env_cfg.terminations, "object_dropping"):
        env_cfg.terminations.object_dropping = None
    env_cfg.actions.gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=ACT_V5_GRIPPER_OPEN_JOINT_POS - args_cli.gripper_closed_joint_target,
        offset=args_cli.gripper_closed_joint_target,
        use_default_offset=False,
        preserve_order=True,
    )
    env_cfg.scene.robot.actuators["gripper"].effort_limit_sim = args_cli.gripper_effort_limit
    env_cfg.scene.robot.actuators["gripper"].stiffness = args_cli.gripper_stiffness
    env_cfg.scene.robot.actuators["gripper"].damping = args_cli.gripper_damping
    for group_name in ("act_fixed", "act_handeye", "act_state"):
        if hasattr(env_cfg.observations, group_name):
            setattr(env_cfg.observations, group_name, None)
    for sensor_name in ("fixed_camera", "handeye_camera"):
        if hasattr(env_cfg.scene, sensor_name):
            setattr(env_cfg.scene, sensor_name, None)


def _step_with_throttle(env, action: torch.Tensor, control_dt: float, realtime: bool) -> None:
    started = time.monotonic()
    env.step(action)
    if realtime:
        remaining = control_dt - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)


def _collect_metrics(
    scene,
    initial_obj_pos: torch.Tensor,
    dataset_action: np.ndarray,
    env_action: np.ndarray,
    source_frame: int,
    dataset_time_s: float,
    step: int,
    pass_name: str,
    phase: str,
) -> dict[str, float | str]:
    robot = scene["robot"]
    object_asset = scene["object"]
    ee_frame = scene["ee_frame"]
    arm_joint_ids = robot.find_joints(list(ACT_JOINT_NAMES[:-1]), preserve_order=True)[0]
    gripper_joint_id = robot.find_joints("gripper")[0][0]

    joint_pos = robot.data.joint_pos[0, arm_joint_ids]
    gripper_pos = robot.data.joint_pos[0, gripper_joint_id]
    obj_pos = object_asset.data.root_pos_w[0, :3]
    ee_pos = ee_frame.data.target_pos_w[0, 0, :3]
    height_gain = obj_pos[2] - initial_obj_pos[2]
    object_xy_move = torch.linalg.norm(obj_pos[:2] - initial_obj_pos[:2])
    ee_object_dist = torch.linalg.norm(ee_pos - obj_pos)

    row: dict[str, float | str] = {
        "step": float(step),
        "pass_name": pass_name,
        "phase": phase,
        "source_frame": float(source_frame),
        "dataset_time_s": float(dataset_time_s),
        "object_x": float(obj_pos[0].item()),
        "object_y": float(obj_pos[1].item()),
        "object_z": float(obj_pos[2].item()),
        "height_gain_m": float(height_gain.item()),
        "object_xy_move_m": float(object_xy_move.item()),
        "ee_x": float(ee_pos[0].item()),
        "ee_y": float(ee_pos[1].item()),
        "ee_z": float(ee_pos[2].item()),
        "ee_object_dist_m": float(ee_object_dist.item()),
    }
    for index, name in enumerate(ACT_JOINT_NAMES[:-1]):
        row[f"joint_{name}"] = float(joint_pos[index].item())
    row["joint_gripper"] = float(gripper_pos.item())
    row.update({f"dataset_action_{index}": float(dataset_action[index]) for index in range(6)})
    row.update({f"env_action_{index}": float(env_action[index]) for index in range(6)})
    return row


def _print_row(row: dict[str, float | str]) -> None:
    print(
        f"[step {int(float(row['step'])):04d}] {str(row['pass_name']):>11s}/{str(row['phase']):<13s} "
        f"frame={int(float(row['source_frame'])):04d} "
        f"ee=({float(row['ee_x']):.3f},{float(row['ee_y']):.3f},{float(row['ee_z']):.3f}) "
        f"obj=({float(row['object_x']):.3f},{float(row['object_y']):.3f},{float(row['object_z']):.3f}) "
        f"rise={float(row['height_gain_m']):.4f}m "
        f"dist={float(row['ee_object_dist_m']):.4f}m "
        f"gripper={float(row['joint_gripper']):.3f}"
    )


def _execute_step(
    env,
    writer: csv.DictWriter,
    trajectory: ReplayTrajectory,
    replay_index: int,
    initial_obj_pos: torch.Tensor,
    global_step: int,
    pass_name: str,
    phase: str,
    control_dt: float,
    realtime: bool,
    force_gripper: float | None = None,
) -> tuple[int, dict[str, float | str], torch.Tensor]:
    env_action = trajectory.env_actions[replay_index].copy()
    if force_gripper is not None:
        env_action[5] = force_gripper
    action_tensor = torch.as_tensor(env_action, device=env.unwrapped.device, dtype=torch.float32).view(1, 6)
    _step_with_throttle(env, action_tensor, control_dt, realtime)
    row = _collect_metrics(
        env.unwrapped.scene,
        initial_obj_pos,
        trajectory.dataset_actions[replay_index],
        env_action,
        int(trajectory.source_frames[replay_index]),
        float(trajectory.sim_times[replay_index]),
        global_step,
        pass_name,
        phase,
    )
    writer.writerow(row)
    if args_cli.print_interval > 0 and global_step % args_cli.print_interval == 0:
        _print_row(row)
    return global_step + 1, row, action_tensor


def _run_settle(
    env,
    writer: csv.DictWriter,
    trajectory: ReplayTrajectory,
    initial_obj_pos: torch.Tensor,
    global_step: int,
    pass_name: str,
    control_dt: float,
    realtime: bool,
) -> int:
    for _ in range(args_cli.settle_steps):
        global_step, _, _ = _execute_step(
            env,
            writer,
            trajectory,
            replay_index=0,
            initial_obj_pos=initial_obj_pos,
            global_step=global_step,
            pass_name=pass_name,
            phase="settle_open",
            control_dt=control_dt,
            realtime=realtime,
            force_gripper=1.0,
        )
    return global_step


def _park_cube_for_calibration(scene) -> torch.Tensor:
    object_asset = scene["object"]
    root_pose = object_asset.data.root_pose_w.clone()
    root_pose[0, :3] = root_pose.new_tensor((2.0, 2.0, 1.0))
    object_asset.write_root_pose_to_sim(root_pose)
    object_asset.write_root_velocity_to_sim(
        torch.zeros((1, 6), device=root_pose.device, dtype=root_pose.dtype)
    )
    return root_pose[0, :3].clone()


def _place_cube_at_calibrated_grasp(scene, calibrated_ee_pos: torch.Tensor) -> torch.Tensor:
    object_asset = scene["object"]
    root_pose = object_asset.data.root_pose_w.clone()
    root_pose[0, 0] = calibrated_ee_pos[0] + args_cli.cube_x_offset
    root_pose[0, 1] = calibrated_ee_pos[1] + args_cli.cube_y_offset
    object_asset.write_root_pose_to_sim(root_pose)
    object_asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=root_pose.device, dtype=root_pose.dtype))
    return root_pose[0, :3].clone()


def _classify_result(rows: list[dict[str, float | str]], hold_rows: list[dict[str, float | str]]) -> tuple[str, dict[str, float]]:
    max_height_gain = max((float(row["height_gain_m"]) for row in rows), default=float("nan"))
    min_hold_height = min((float(row["height_gain_m"]) for row in hold_rows), default=float("nan"))
    max_xy_move = max((float(row["object_xy_move_m"]) for row in rows), default=float("nan"))
    min_ee_distance = min((float(row["ee_object_dist_m"]) for row in rows), default=float("nan"))
    max_ee_distance = max((float(row["ee_object_dist_m"]) for row in rows), default=float("nan"))
    summary = {
        "max_height_gain_m": max_height_gain,
        "min_hold_height_gain_m": min_hold_height,
        "max_object_xy_move_m": max_xy_move,
        "min_ee_object_dist_m": min_ee_distance,
        "max_ee_object_dist_m": max_ee_distance,
    }
    if min_ee_distance > 0.06:
        return "trajectory_missed_cube", summary
    if max_xy_move >= args_cli.ejection_xy_threshold and max_ee_distance >= args_cli.ejection_ee_distance:
        return "object_ejected", summary
    if max_height_gain < args_cli.rise_threshold:
        return "no_object_rise", summary
    if max_height_gain < args_cli.lift_threshold:
        return "partial_lift", summary
    if not hold_rows:
        return "unstable_lift", summary
    if min_hold_height < args_cli.hold_threshold:
        return "slip_or_drop", summary
    return "success", summary


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    control_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    trajectory = _load_replay_trajectory(args_cli.dataset_dir, args_cli.episode_index, control_dt)
    total_steps = args_cli.settle_steps * 2 + len(trajectory.env_actions) * 2 + args_cli.hold_steps
    _configure_env_cfg(env_cfg, total_steps, trajectory.initial_dataset_state)
    realtime = (not args_cli.headless) if args_cli.realtime is None else args_cli.realtime

    print(f"[INFO] Dataset: {args_cli.dataset_dir}")
    print(f"[INFO] Episode: {args_cli.episode_index}, fps={trajectory.dataset_fps:.1f}")
    print(
        "[INFO] Initial dataset state [5 arm degrees, gripper percent]: "
        + np.array2string(trajectory.initial_dataset_state, precision=4, separator=", ")
    )
    print(f"[INFO] Reset from episode state: {args_cli.reset_from_episode_state}")
    print(f"[INFO] Robot base Z offset: {args_cli.robot_base_z_offset:+.4f} m")
    print(
        f"[INFO] Full-episode replay: resampled_steps={len(trajectory.env_actions)}, "
        f"close_dataset_row={trajectory.close_dataset_row}, "
        f"first_below_threshold_row={trajectory.reopen_dataset_row}, "
        f"replay_end_dataset_row={trajectory.replay_end_dataset_row}, "
        f"final_source_frame={int(trajectory.source_frames[-1])}"
    )
    print(
        "[INFO] Episode-relative gripper mapping: "
        f"{trajectory.gripper_open_value:.3f}->{trajectory.gripper_closed_value:.3f}% maps to "
        f"joint {ACT_V5_GRIPPER_OPEN_JOINT_POS:.3f}->{args_cli.gripper_closed_joint_target:.3f} rad; "
        "closure latched after grasp onset"
    )
    print(f"[INFO] Replay time scale: {args_cli.replay_time_scale:.2f}x slower than the dataset")
    print(
        "[INFO] Debug gripper actuator: "
        f"effort={args_cli.gripper_effort_limit:.2f}, stiffness={args_cli.gripper_stiffness:.1f}, "
        f"damping={args_cli.gripper_damping:.1f}"
    )
    print("[INFO] Policy: none; replaying recorded dataset actions")
    print("[INFO] Pass 1: force gripper open and calibrate grasp XY")
    print("[INFO] Pass 2: place cube at calibrated XY and replay the same actions")

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        args_cli.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = args_cli.output_dir / f"dataset_replay_grasp_{stamp}.csv"
        global_step = 0

        with out_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            with torch.inference_mode():
                env.reset()
                scene = env.unwrapped.scene
                calibration_obj_pos = _park_cube_for_calibration(scene)
                global_step = _run_settle(
                    env,
                    writer,
                    trajectory,
                    calibration_obj_pos,
                    global_step,
                    "calibration",
                    control_dt,
                    realtime,
                )
                calibrated_ee_pos: torch.Tensor | None = None
                calibration_post_close_max_ee_rise = 0.0
                for replay_index in range(len(trajectory.env_actions)):
                    calibration_phase = (
                        "approach_open" if replay_index <= trajectory.close_sim_step else "post_close_open"
                    )
                    global_step, row, _ = _execute_step(
                        env,
                        writer,
                        trajectory,
                        replay_index,
                        calibration_obj_pos,
                        global_step,
                        "calibration",
                        calibration_phase,
                        control_dt,
                        realtime,
                        force_gripper=1.0,
                    )
                    if replay_index == trajectory.close_sim_step:
                        calibrated_ee_pos = scene["ee_frame"].data.target_pos_w[0, 0, :3].clone()
                    elif replay_index > trajectory.close_sim_step and calibrated_ee_pos is not None:
                        ee_rise = float(row["ee_z"]) - float(calibrated_ee_pos[2].item())
                        calibration_post_close_max_ee_rise = max(calibration_post_close_max_ee_rise, ee_rise)
                if calibrated_ee_pos is None:
                    raise RuntimeError("Calibration replay never reached the detected close step")
                print(
                    "[CALIBRATION] grasp_center_w="
                    f"({calibrated_ee_pos[0].item():.4f}, {calibrated_ee_pos[1].item():.4f}, "
                    f"{calibrated_ee_pos[2].item():.4f})"
                )
                print(
                    "[CALIBRATION] post_close_max_ee_rise_m="
                    f"{calibration_post_close_max_ee_rise:.6f}"
                )

                env.reset()
                scene = env.unwrapped.scene
                requested_cube_pos = _place_cube_at_calibrated_grasp(scene, calibrated_ee_pos)
                replay_reset_pos = scene["object"].data.root_pos_w[0, :3].clone()
                global_step = _run_settle(
                    env,
                    writer,
                    trajectory,
                    replay_reset_pos,
                    global_step,
                    "replay",
                    control_dt,
                    realtime,
                )
                initial_obj_pos = scene["object"].data.root_pos_w[0, :3].clone()
                print(
                    "[REPLAY] cube_position_w="
                    f"({initial_obj_pos[0].item():.4f}, {initial_obj_pos[1].item():.4f}, "
                    f"{initial_obj_pos[2].item():.4f}), requested_xy="
                    f"({requested_cube_pos[0].item():.4f}, {requested_cube_pos[1].item():.4f})"
                )

                replay_rows: list[dict[str, float | str]] = []
                hold_rows: list[dict[str, float | str]] = []
                stable_lift_count = 0
                hold_action: torch.Tensor | None = None
                hold_replay_index = 0
                for replay_index in range(len(trajectory.env_actions)):
                    gripper_action = float(trajectory.env_actions[replay_index, 5])
                    if replay_index >= trajectory.close_sim_step:
                        phase = "grasp_closed"
                    elif gripper_action < 0.95:
                        phase = "grasp_closing"
                    else:
                        phase = "approach_open"
                    global_step, row, action_tensor = _execute_step(
                        env,
                        writer,
                        trajectory,
                        replay_index,
                        initial_obj_pos,
                        global_step,
                        "replay",
                        phase,
                        control_dt,
                        realtime,
                    )
                    replay_rows.append(row)
                    if replay_index >= trajectory.close_sim_step and float(row["height_gain_m"]) >= args_cli.lift_threshold:
                        stable_lift_count += 1
                    else:
                        stable_lift_count = 0
                    if stable_lift_count >= args_cli.lift_consecutive_steps:
                        hold_action = action_tensor.clone()
                        hold_replay_index = replay_index
                        print(
                            f"[REPLAY] Lift detected at source_frame={int(row['source_frame'])}, "
                            f"height_gain={float(row['height_gain_m']):.4f}m; starting hold"
                        )
                        break

                if hold_action is not None:
                    hold_env_action = hold_action[0].detach().cpu().numpy()
                    for _ in range(args_cli.hold_steps):
                        _step_with_throttle(env, hold_action, control_dt, realtime)
                        row = _collect_metrics(
                            scene,
                            initial_obj_pos,
                            trajectory.dataset_actions[hold_replay_index],
                            hold_env_action,
                            int(trajectory.source_frames[hold_replay_index]),
                            float(trajectory.sim_times[hold_replay_index]),
                            global_step,
                            "replay",
                            "hold_closed",
                        )
                        writer.writerow(row)
                        hold_rows.append(row)
                        replay_rows.append(row)
                        if args_cli.print_interval > 0 and global_step % args_cli.print_interval == 0:
                            _print_row(row)
                        global_step += 1

        classification, summary = _classify_result(replay_rows, hold_rows)
        print(f"[RESULT] classification={classification}")
        print(f"[RESULT] max_height_gain_m={summary['max_height_gain_m']:.6f}")
        print(f"[RESULT] min_hold_height_gain_m={summary['min_hold_height_gain_m']:.6f}")
        print(f"[RESULT] max_object_xy_move_m={summary['max_object_xy_move_m']:.6f}")
        print(f"[RESULT] min_ee_object_dist_m={summary['min_ee_object_dist_m']:.6f}")
        print(f"[RESULT] max_ee_object_dist_m={summary['max_ee_object_dist_m']:.6f}")
        print(f"[RESULT] CSV: {out_path}")
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
