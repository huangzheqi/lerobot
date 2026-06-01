# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate GT-vs-ResNet object-pose policies on Vision-Play pick-place tasks.

The script runs one episode per seed for both object pose sources:
``gt`` and ``resnet``. It writes per-episode success and vision-error
statistics to a CSV file for quantitative comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from isaaclab.app import AppLauncher

# local imports
import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate Vision-Play GT vs ResNet object pose performance.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-SO-ARM101-Pick-Place-Cube-Vision-Play-v0",
    help="Vision-Play task name.",
)
parser.add_argument(
    "--resnet_model_path",
    type=str,
    default="selected_models/resnet18_cube_pose.pt",
    help="Path to trained ResNet18 cube pose model.",
)
parser.add_argument("--num_seeds", type=int, default=20, help="Number of consecutive seeds to evaluate.")
parser.add_argument("--seed_start", type=int, default=1, help="First seed to evaluate.")
parser.add_argument(
    "--output_csv",
    type=str,
    default="logs/eval_resnet_vs_gt.csv",
    help="CSV output path.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Evaluation expects 1.")
parser.add_argument(
    "--max_episode_steps",
    type=int,
    default=300,
    help="Hard maximum number of steps per evaluation episode.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()
args_cli.num_envs = 1
args_cli.enable_cameras = True
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required for evaluation.")
if args_cli.max_episode_steps < 1:
    parser.error("--max_episode_steps must be >= 1.")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac Lab task extensions only after SimulationApp is created.
import isaac_so_arm101.tasks  # noqa: F401,E402
from isaac_so_arm101.tasks.pick_place.robust_eval_cfg import apply_pick_place_disturbance  # noqa: F401,E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaac_so_arm101.scripts.rsl_rl.vision_pose_resnet import ResnetCubePoseEstimator  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


OBJECT_POSE_SOURCES = ("gt", "resnet")


@dataclass
class ResnetCacheState:
    """State for the ResNet object-position cache used by ``play.py``."""

    cached_object_pos: torch.Tensor
    sum_xy: torch.Tensor
    count: torch.Tensor
    prev_episode_steps: torch.Tensor
    first_cached_object_position: list[float] | None = None
    fallback_gt_triggered: bool = False
    fallback_last_valid_triggered: bool = False
    last_valid_object_pos: torch.Tensor | None = None


@dataclass
class EpisodeStats:
    """Per-episode metrics collected for CSV output."""

    seed: int
    object_pose_source: str
    success: bool = False
    final_reward: float = 0.0
    episode_length: int = 0
    vision_errors_xy: list[float] = field(default_factory=list)
    first_cached_resnet_object_position: list[float] | None = None
    gt_object_position: list[float] | None = None
    final_object_position: list[float] | None = None
    fallback_gt: bool = False
    grasp_success: bool = False
    place_success: bool = False
    final_cube_target_dist: float = float("nan")


def _make_json_list(value: list[float] | None) -> str:
    """Serialize optional numeric vectors in a CSV-friendly representation."""
    if value is None:
        return ""
    return json.dumps([round(float(item), 6) for item in value])


def _first_env_bool(value: Any) -> bool:
    """Return env-0 boolean for torch tensors, numpy scalars, or Python bools."""
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        value = value[0]
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _first_env_float(value: Any) -> float:
    """Return env-0 float for tensors or scalars."""
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        value = value[0]
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _or_done_flags(terminated: Any, truncated: Any) -> Any:
    """Combine terminated/truncated flags while preserving common vector formats."""
    if torch.is_tensor(terminated) or torch.is_tensor(truncated):
        ref = terminated if torch.is_tensor(terminated) else truncated
        terminated_tensor = torch.as_tensor(terminated, device=ref.device, dtype=torch.bool)
        truncated_tensor = torch.as_tensor(truncated, device=ref.device, dtype=torch.bool)
        return torch.logical_or(terminated_tensor, truncated_tensor)
    try:
        return terminated | truncated
    except TypeError:
        if isinstance(terminated, (list, tuple)) and isinstance(truncated, (list, tuple)):
            return [bool(term) or bool(trunc) for term, trunc in zip(terminated, truncated)]
        return bool(terminated) or bool(truncated)


def _step_env(env: RslRlVecEnvWrapper, actions: torch.Tensor):
    """Step env and normalize common Gymnasium/Isaac Lab API variants."""
    step_out = env.step(actions)
    if not isinstance(step_out, tuple):
        raise RuntimeError(f"Unexpected env.step output type: {type(step_out).__name__}")

    if len(step_out) == 5:
        obs, reward, terminated, truncated, extras = step_out
        done = _or_done_flags(terminated, truncated)
    elif len(step_out) == 4:
        obs, reward, done, extras = step_out
    elif len(step_out) > 5:
        obs, reward = step_out[0], step_out[1]
        terminated, truncated = step_out[2], step_out[3]
        extras = step_out[-1]
        done = _or_done_flags(terminated, truncated)
    else:
        raise RuntimeError(f"Unexpected env.step output length: {len(step_out)}")
    return obs, reward, done, extras


def _reset_env(env: RslRlVecEnvWrapper) -> torch.Tensor:
    """Reset env and normalize Gymnasium/RSL-RL reset return variants."""
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def _obs_term_size(term_dim: Any) -> int:
    """Compute flattened observation term size from Isaac Lab dim metadata."""
    if isinstance(term_dim, int):
        return term_dim
    if hasattr(term_dim, "numel") and not isinstance(term_dim, (list, tuple)):
        values = list(term_dim)
    else:
        values = list(term_dim)
    size = 1
    for dim in values:
        size *= int(dim)
    return size


def _get_policy_obs_offsets(base_env: Any) -> dict[str, tuple[int, int]]:
    """Return observation term offsets, compatible with multiple Isaac Lab versions."""
    policy_group_name = "policy"
    observation_manager = base_env.observation_manager
    obs_term_dims = observation_manager.group_obs_term_dim[policy_group_name]

    if hasattr(observation_manager, "active_terms"):
        obs_term_names = observation_manager.active_terms[policy_group_name]
    elif hasattr(observation_manager, "_group_obs_term_names"):
        obs_term_names = observation_manager._group_obs_term_names[policy_group_name]
    else:
        raise RuntimeError("Observation manager has neither active_terms nor _group_obs_term_names.")

    obs_offsets: dict[str, tuple[int, int]] = {}
    start_idx = 0
    for term_name, term_dim in zip(obs_term_names, obs_term_dims):
        end_idx = start_idx + _obs_term_size(term_dim)
        obs_offsets[term_name] = (start_idx, end_idx)
        start_idx = end_idx
    return obs_offsets


def _build_runner(env: RslRlVecEnvWrapper, agent_cfg: RslRlBaseRunnerCfg, resume_path: str):
    """Create and load the configured RSL-RL runner."""
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    return runner


def _init_resnet_cache(num_envs: int, device: str | torch.device) -> ResnetCacheState:
    return ResnetCacheState(
        cached_object_pos=torch.zeros((num_envs, 3), device=device, dtype=torch.float32),
        sum_xy=torch.zeros((num_envs, 2), device=device, dtype=torch.float32),
        count=torch.zeros((num_envs,), device=device, dtype=torch.int64),
        prev_episode_steps=torch.full((num_envs,), -1, device=device, dtype=torch.int64),
    )


def _estimate_object_position(
    *,
    object_pose_source: str,
    base_env: Any,
    fixed_camera: Any,
    resnet_estimator: ResnetCubePoseEstimator | None,
    cache_state: ResnetCacheState | None,
    gt_object_pos_robot: torch.Tensor,
) -> torch.Tensor:
    """Apply GT or the ResNet cache logic used by ``play.py`` to estimate object position."""
    estimated_object_pos = gt_object_pos_robot.clone()
    if object_pose_source == "gt":
        return estimated_object_pos

    if cache_state is None:
        raise RuntimeError("cache_state is required when object_pose_source=resnet.")

    if fixed_camera is None or resnet_estimator is None:
        cache_state.fallback_gt_triggered = True
        return estimated_object_pos

    # Keep this cache update aligned with play.py's ResNet object-pose path:
    # reset per-episode accumulators, average warm-up predictions, then freeze the
    # cached XY estimate while reusing a fixed Z value for policy observations.
    resnet_warmup_steps = 10
    resnet_fixed_z = 0.012
    try:
        episode_steps = base_env.episode_length_buf.to(torch.int64)
        reset_env_mask = (episode_steps <= 1) & (cache_state.prev_episode_steps >= 2) & (
            episode_steps < cache_state.prev_episode_steps
        )
        if torch.any(reset_env_mask):
            cache_state.sum_xy[reset_env_mask] = 0.0
            cache_state.count[reset_env_mask] = 0
            cache_state.cached_object_pos[reset_env_mask] = 0.0
        cache_state.prev_episode_steps = episode_steps.clone()

        raw_resnet_object_pos = resnet_estimator.estimate(fixed_camera.data.output["rgb"], gt_object_pos_robot[:, 2])

        warmup_mask = episode_steps < resnet_warmup_steps
        if torch.any(warmup_mask):
            cache_state.sum_xy[warmup_mask] += raw_resnet_object_pos[warmup_mask, :2]
            cache_state.count[warmup_mask] += 1
            count_f = cache_state.count[warmup_mask].unsqueeze(1).to(torch.float32)
            cache_state.cached_object_pos[warmup_mask, :2] = cache_state.sum_xy[warmup_mask] / count_f
            cache_state.cached_object_pos[warmup_mask, 2] = resnet_fixed_z

        no_cache_mask = cache_state.count == 0
        if torch.any(no_cache_mask):
            cache_state.cached_object_pos[no_cache_mask, :2] = raw_resnet_object_pos[no_cache_mask, :2]
            cache_state.cached_object_pos[no_cache_mask, 2] = resnet_fixed_z
            cache_state.sum_xy[no_cache_mask] = raw_resnet_object_pos[no_cache_mask, :2]
            cache_state.count[no_cache_mask] = 1

        estimated_object_pos = cache_state.cached_object_pos.clone()
        cache_state.last_valid_object_pos = estimated_object_pos.clone()
        if cache_state.first_cached_object_position is None:
            cache_state.first_cached_object_position = estimated_object_pos[0].detach().cpu().tolist()
    except Exception as exc:
        if cache_state.last_valid_object_pos is not None:
            estimated_object_pos = cache_state.last_valid_object_pos.clone()
            cache_state.fallback_last_valid_triggered = True
            print(f"[WARNING] ResNet18 inference failed, fallback to last valid estimate: {exc}")
        else:
            cache_state.fallback_gt_triggered = True
            print(f"[WARNING] ResNet18 inference failed, fallback to gt: {exc}")

    return estimated_object_pos


def _update_success_metrics(stats: EpisodeStats, base_env: Any) -> None:
    """Update grasp/place success metrics using the same criteria as the existing evaluator."""
    object_asset = base_env.scene["object"]
    robot_asset = base_env.scene["robot"]
    command = base_env.command_manager.get_command("object_pose")
    robot_root = robot_asset.data.root_pos_w[:, :3]
    target_pos_w = robot_root + command[:, :3]
    cube_pos = object_asset.data.root_pos_w[:, :3]
    cube_vel = torch.norm(object_asset.data.root_lin_vel_w[:, :3], dim=1)
    cube_to_target_dist = torch.norm(target_pos_w[:, :2] - cube_pos[:, :2], dim=1)

    stats.final_cube_target_dist = float(cube_to_target_dist[0].item())
    if float(cube_pos[0, 2].item()) > 0.05:
        stats.grasp_success = True
    if (
        float(cube_to_target_dist[0].item()) < 0.05
        and abs(float(cube_pos[0, 2].item()) - 0.025) < 0.02
        and float(cube_vel[0].item()) < 0.1
    ):
        stats.place_success = True
    stats.success = stats.grasp_success and stats.place_success


def _run_one_episode(
    *,
    env: RslRlVecEnvWrapper,
    policy: Any,
    object_pose_source: str,
    resnet_estimator: ResnetCubePoseEstimator | None,
    seed: int,
    max_episode_steps: int,
) -> EpisodeStats:
    """Run one evaluation episode and collect metrics."""
    base_env = env.unwrapped
    object_asset = base_env.scene["object"]
    robot_asset = base_env.scene["robot"]
    fixed_camera = base_env.scene["fixed_camera"] if "fixed_camera" in base_env.scene.keys() else None
    obs_offsets = _get_policy_obs_offsets(base_env)
    if "object_position" not in obs_offsets:
        raise RuntimeError("Could not find 'object_position' term in policy observation.")
    object_slice = obs_offsets["object_position"]

    obs = _reset_env(env)
    cache_state = _init_resnet_cache(env.unwrapped.num_envs, env.unwrapped.device) if object_pose_source == "resnet" else None
    stats = EpisodeStats(seed=seed, object_pose_source=object_pose_source)

    ended_by_max_episode_steps = False

    while simulation_app.is_running():
        with torch.inference_mode():
            gt_object_pos_world = object_asset.data.root_pos_w[:, :3]
            robot_root_world = robot_asset.data.root_pos_w[:, :3]
            gt_object_pos_robot = gt_object_pos_world - robot_root_world
            if stats.gt_object_position is None:
                stats.gt_object_position = gt_object_pos_robot[0].detach().cpu().tolist()

            estimated_object_pos = _estimate_object_position(
                object_pose_source=object_pose_source,
                base_env=base_env,
                fixed_camera=fixed_camera,
                resnet_estimator=resnet_estimator,
                cache_state=cache_state,
                gt_object_pos_robot=gt_object_pos_robot,
            )
            obs[:, object_slice[0] : object_slice[1]] = estimated_object_pos
            vision_error_xy = torch.norm(estimated_object_pos[:, :2] - gt_object_pos_robot[:, :2], dim=1)
            stats.vision_errors_xy.append(float(vision_error_xy[0].item()))

            actions = policy(obs)
            obs, reward, done, _ = _step_env(env, actions)

        stats.episode_length += 1
        stats.final_reward = _first_env_float(reward)
        final_gt_object_pos_robot = object_asset.data.root_pos_w[0, :3] - robot_asset.data.root_pos_w[0, :3]
        stats.final_object_position = final_gt_object_pos_robot.detach().cpu().tolist()
        _update_success_metrics(stats, base_env)
        if cache_state is not None:
            stats.first_cached_resnet_object_position = cache_state.first_cached_object_position
            stats.fallback_gt = cache_state.fallback_gt_triggered

        if stats.episode_length % 50 == 0:
            print(
                f"seed={seed} source={object_pose_source} step={stats.episode_length} "
                f"reward={stats.final_reward:.6f} vision_error_xy={stats.vision_errors_xy[-1]:.6f}"
            )

        if _first_env_bool(done):
            break
        if stats.episode_length >= max_episode_steps:
            ended_by_max_episode_steps = True
            stats.success = False
            break

    if ended_by_max_episode_steps:
        stats.success = False

    return stats


def _stats_to_row(stats: EpisodeStats) -> dict[str, Any]:
    mean_error = sum(stats.vision_errors_xy) / max(len(stats.vision_errors_xy), 1)
    max_error = max(stats.vision_errors_xy) if stats.vision_errors_xy else 0.0
    return {
        "seed": stats.seed,
        "object_pose_source": stats.object_pose_source,
        "success": int(stats.success),
        "final_reward": f"{stats.final_reward:.8f}",
        "episode_length": stats.episode_length,
        "mean_vision_error_xy": f"{mean_error:.8f}",
        "max_vision_error_xy": f"{max_error:.8f}",
        "first_cached_resnet_object_position": _make_json_list(stats.first_cached_resnet_object_position),
        "gt_object_position": _make_json_list(stats.gt_object_position),
        "final_object_position": _make_json_list(stats.final_object_position),
        "fallback_gt": int(stats.fallback_gt),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run GT and ResNet evaluation episodes and write a CSV report."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    resume_path = retrieve_file_path(args_cli.checkpoint)

    output_dir = os.path.dirname(os.path.abspath(args_cli.output_csv))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "seed",
        "object_pose_source",
        "success",
        "final_reward",
        "episode_length",
        "mean_vision_error_xy",
        "max_vision_error_xy",
        "first_cached_resnet_object_position",
        "gt_object_position",
        "final_object_position",
        "fallback_gt",
    ]

    num_rows = 0
    with open(args_cli.output_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        fp.flush()

        for seed in range(args_cli.seed_start, args_cli.seed_start + args_cli.num_seeds):
            for object_pose_source in OBJECT_POSE_SOURCES:
                env_cfg.seed = seed
                agent_cfg.seed = seed
                print(f"[INFO] Evaluating seed={seed}, object_pose_source={object_pose_source}")
                env = gym.make(args_cli.task, cfg=env_cfg)
                try:
                    if isinstance(env.unwrapped, DirectMARLEnv):
                        env = multi_agent_to_single_agent(env)
                    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

                    runner = _build_runner(env, agent_cfg, resume_path)
                    policy = runner.get_inference_policy(device=env.unwrapped.device)

                    resnet_estimator = None
                    if object_pose_source == "resnet":
                        try:
                            resnet_estimator = ResnetCubePoseEstimator(
                                args_cli.resnet_model_path, device=env.unwrapped.device
                            )
                            print(f"[INFO] Loaded ResNet18 cube pose model: {args_cli.resnet_model_path}")
                        except Exception as exc:
                            print(f"[WARNING] Failed loading ResNet18 model: {exc}; fallback to gt.")

                    stats = _run_one_episode(
                        env=env,
                        policy=policy,
                        object_pose_source=object_pose_source,
                        resnet_estimator=resnet_estimator,
                        seed=seed,
                        max_episode_steps=args_cli.max_episode_steps,
                    )
                    writer.writerow(_stats_to_row(stats))
                    fp.flush()
                    num_rows += 1
                finally:
                    env.close()

    print(f"[INFO] Wrote {num_rows} evaluation rows to: {args_cli.output_csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()
