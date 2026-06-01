from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _goal_metrics(env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3])
    delta = des_pos_w - obj.data.root_pos_w[:, :3]
    xy_dist = torch.norm(delta[:, :2], dim=1)
    z_err = delta[:, 2]
    return xy_dist, z_err


def _get_env_origins(env: ManagerBasedRLEnv, like: torch.Tensor) -> torch.Tensor:
    origins = getattr(env.scene, "env_origins", None)
    if origins is None:
        return torch.zeros((like.shape[0], 3), device=like.device, dtype=like.dtype)
    return origins.to(device=like.device, dtype=like.dtype)


def _object_default_root_pos_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    initial_object_pos: tuple[float, float, float] = (0.2, 0.0, 0.015),
) -> torch.Tensor:
    """Return the per-env object reset/default world position when available."""
    obj: RigidObject = env.scene[object_cfg.name]
    root_pos = obj.data.root_pos_w[:, :3]
    default_root_state = getattr(obj.data, "default_root_state", None)
    if default_root_state is not None:
        default_pos = default_root_state[:, :3].to(device=root_pos.device, dtype=root_pos.dtype)
        # Isaac Lab default root states are typically stored in env-local coordinates.
        # Convert to world coordinates if they are not already offset by env origins.
        env_origins = _get_env_origins(env, root_pos)
        default_pos_with_origins = default_pos + env_origins
        current_env_dist = torch.norm(root_pos[:, :2] - default_pos[:, :2], dim=1).mean()
        current_world_dist = torch.norm(root_pos[:, :2] - default_pos_with_origins[:, :2], dim=1).mean()
        return torch.where(
            (current_env_dist < current_world_dist).view(1, 1),
            default_pos,
            default_pos_with_origins,
        )

    env_origins = _get_env_origins(env, root_pos)
    initial_pos = torch.tensor(initial_object_pos, device=root_pos.device, dtype=root_pos.dtype).view(1, 3)
    return initial_pos + env_origins


def _object_episode_initial_root_pos_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Cache the object's actual per-episode reset position for height gain and push diagnostics."""
    obj: RigidObject = env.scene[object_cfg.name]
    root_pos = obj.data.root_pos_w[:, :3]
    cache_name = "_so101_pick_place_initial_object_pos_w"
    cached_pos = getattr(env, cache_name, None)
    if cached_pos is None or cached_pos.shape != root_pos.shape or cached_pos.device != root_pos.device:
        cached_pos = _object_default_root_pos_w(env, object_cfg).clone()

    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is None:
        reset_mask = torch.zeros((root_pos.shape[0], 1), device=root_pos.device, dtype=torch.bool)
    else:
        reset_mask = (episode_length_buf.to(device=root_pos.device) <= 1).view(-1, 1)
    cached_pos = torch.where(reset_mask, root_pos.detach(), cached_pos.to(device=root_pos.device, dtype=root_pos.dtype))
    setattr(env, cache_name, cached_pos)
    return cached_pos


def _object_height_gain(
    env: ManagerBasedRLEnv,
    initial_object_z: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    initial_pos_w = _object_episode_initial_root_pos_w(env, object_cfg)
    fallback_initial_z = _object_default_root_pos_w(env, object_cfg, (0.2, 0.0, initial_object_z))[:, 2]
    initial_z = torch.where(torch.isfinite(initial_pos_w[:, 2]), initial_pos_w[:, 2], fallback_initial_z)
    return obj.data.root_pos_w[:, 2] - initial_z


def _ee_object_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :3]
    return torch.norm(obj.data.root_pos_w[:, :3] - ee_w, dim=1)


def _get_gripper_joint_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_idx = robot.find_joints("gripper")[0][0]
    return robot.data.joint_pos[:, gripper_idx]


def _gripper_open_ratio(
    env: ManagerBasedRLEnv,
    open_joint_pos: float,
    close_joint_pos: float,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Estimate gripper opening ratio in [0, 1] with a configurable joint range.

    Assumption for SO101 in this repo: larger joint_pos means more open.
    If hardware mapping differs, only this helper needs to be adjusted.
    """
    joint_pos = _get_gripper_joint_pos(env, robot_cfg)
    denom = max(open_joint_pos - close_joint_pos, 1e-3)
    return torch.clamp((joint_pos - close_joint_pos) / denom, 0.0, 1.0)


def _get_wrist_flex_joint_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return wrist flex joint position.

    Priority: `wrist_flex`, then `wrist_.*` fallback for compatibility.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    try:
        wrist_idx = robot.find_joints("wrist_flex")[0][0]
    except Exception:
        # Fallback if the exact joint name differs in some robot variants.
        wrist_idx = robot.find_joints("wrist_.*")[0][0]
    return robot.data.joint_pos[:, wrist_idx]


def _gates(env: ManagerBasedRLEnv, lift_height: float, near_goal_xy: float, release_height: float, command_name: str,
           robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    obj: RigidObject = env.scene[object_cfg.name]
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    z = obj.data.root_pos_w[:, 2]
    is_lifted = z > lift_height
    near_goal = xy_dist < near_goal_xy
    low_height = z < release_height
    s1 = (~is_lifted).float()
    s2 = (is_lifted & ~near_goal).float()
    s3 = (is_lifted & near_goal & ~low_height).float()
    s4 = (is_lifted & near_goal & low_height).float()
    return s1, s2, s3, s4


def stage1_close_when_near_object_reward(
    env: ManagerBasedRLEnv,
    near_distance: float,
    open_joint_pos: float,
    close_joint_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closing the gripper only after the end-effector reaches the object."""
    ee_obj_dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    near_object = (ee_obj_dist < near_distance).float()
    gripper_closed = 1.0 - _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)
    return near_object * gripper_closed


def stage1_open_when_near_object_penalty(
    env: ManagerBasedRLEnv,
    near_distance: float,
    open_joint_pos: float,
    close_joint_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalize staying open at grasp distance, which usually leads to pushing."""
    ee_obj_dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    near_object = (ee_obj_dist < near_distance).float()
    gripper_open = _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)
    return near_object * gripper_open


def object_lifted_from_initial_reward(
    env: ManagerBasedRLEnv,
    min_height_gain: float,
    initial_object_z: float = 0.015,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward true lift based on height gain relative to the reset/default object height."""
    height_gain = _object_height_gain(env, initial_object_z, object_cfg)
    return (height_gain > min_height_gain).float()


def lifted_close_hold_reward(
    env: ManagerBasedRLEnv,
    min_height_gain: float,
    near_distance: float,
    open_joint_pos: float,
    close_joint_pos: float,
    initial_object_z: float = 0.015,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward maintaining a close, closed grasp after the cube leaves the table."""
    height_gain = _object_height_gain(env, initial_object_z, object_cfg)
    lifted = (height_gain > min_height_gain).float()
    ee_obj_dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    near_object = (ee_obj_dist < near_distance).float()
    gripper_closed = 1.0 - _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)
    return lifted * near_object * gripper_closed


def push_without_lift_penalty(
    env: ManagerBasedRLEnv,
    move_xy_threshold: float,
    min_height_gain: float,
    initial_object_z: float = 0.015,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize large horizontal cube displacement when the cube has not been lifted."""
    obj: RigidObject = env.scene[object_cfg.name]
    initial_pos_w = _object_episode_initial_root_pos_w(env, object_cfg)
    move_xy = torch.norm(obj.data.root_pos_w[:, :2] - initial_pos_w[:, :2], dim=1)
    height_gain = obj.data.root_pos_w[:, 2] - initial_pos_w[:, 2]
    pushed = move_xy > move_xy_threshold
    not_lifted = height_gain < min_height_gain
    return (pushed & not_lifted).float()


def stage2_goal_xy_tracking_gated(env: ManagerBasedRLEnv, std: float, lift_height: float, near_goal_xy: float,
                                  release_height: float, command_name: str,
                                  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                  object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    _, s2, _, _ = _gates(env, lift_height, near_goal_xy, release_height, command_name, robot_cfg, object_cfg)
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    return s2 * (1.0 - torch.tanh(xy_dist / std))


def stage2_early_open_penalty_gated(
    env: ManagerBasedRLEnv,
    open_joint_pos: float,
    close_joint_pos: float,
    lift_height: float,
    near_goal_xy: float,
    release_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    _, s2, _, _ = _gates(
        env,
        lift_height,
        near_goal_xy,
        release_height,
        command_name,
        SceneEntityCfg("robot"),
        SceneEntityCfg("object"),
    )
    gripper_open = _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)
    return s2 * gripper_open


def stage3_soft_descent_reward_gated(env: ManagerBasedRLEnv, target_speed: float, lift_height: float, near_goal_xy: float,
                                     release_height: float, command_name: str,
                                     object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    _, _, s3, _ = _gates(env, lift_height, near_goal_xy, release_height, command_name, SceneEntityCfg("robot"), object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    vz = obj.data.root_lin_vel_w[:, 2]
    return s3 * torch.exp(-torch.square(vz + target_speed) / (2 * target_speed * target_speed + 1e-6))


def stage3_hard_drop_penalty_gated(env: ManagerBasedRLEnv, max_down_speed: float, lift_height: float, near_goal_xy: float,
                                   release_height: float, command_name: str,
                                   object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    _, _, s3, _ = _gates(env, lift_height, near_goal_xy, release_height, command_name, SceneEntityCfg("robot"), object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    vz = obj.data.root_lin_vel_w[:, 2]
    return s3 * torch.clamp(-(vz + max_down_speed), min=0.0)


def stage3_ee_low_near_goal_gated(
    env: ManagerBasedRLEnv, near_goal_xy: float, target_ee_height: float, ee_height_std: float, command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    near_goal = (xy_dist < near_goal_xy).float()
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_height = ee_frame.data.target_pos_w[..., 0, 2]
    return near_goal * torch.exp(-torch.square(ee_height - target_ee_height) / (2 * ee_height_std * ee_height_std + 1e-6))


def stage3_object_height_near_table_gated(
    env: ManagerBasedRLEnv, near_goal_xy: float, table_height: float, table_margin: float, command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    near_goal = (xy_dist < near_goal_xy).float()
    obj: RigidObject = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    return near_goal * torch.exp(-torch.square(z - table_height) / (2 * table_margin * table_margin + 1e-6))


def stage3_wrist_flex_release_pose_gated(
    env: ManagerBasedRLEnv, near_goal_xy: float, wrist_target_pos: float, wrist_std: float, command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    near_goal = (xy_dist < near_goal_xy).float()
    wrist_pos = _get_wrist_flex_joint_pos(env, robot_cfg)
    return near_goal * torch.exp(-torch.square(wrist_pos - wrist_target_pos) / (2 * wrist_std * wrist_std + 1e-6))


def stage4_release_reward_gated(
    env: ManagerBasedRLEnv,
    open_joint_pos: float,
    close_joint_pos: float,
    lift_height: float,
    near_goal_xy: float,
    release_height: float,
    table_height: float,
    table_margin: float,
    ee_low_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    near_goal = xy_dist < near_goal_xy
    obj_low = torch.abs(obj.data.root_pos_w[:, 2] - table_height) < table_margin
    ee_low = ee_frame.data.target_pos_w[..., 0, 2] < ee_low_height
    lifted_enough = obj.data.root_pos_w[:, 2] > lift_height
    release_band = obj.data.root_pos_w[:, 2] < (release_height + table_margin)
    gate = (near_goal & obj_low & ee_low & lifted_enough & release_band).float()
    return gate * _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)


def stage4_hold_too_long_penalty_gated(
    env: ManagerBasedRLEnv,
    open_joint_pos: float,
    close_joint_pos: float,
    lift_height: float,
    near_goal_xy: float,
    release_height: float,
    table_height: float,
    table_margin: float,
    ee_low_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    near_goal = xy_dist < near_goal_xy
    obj_low = torch.abs(obj.data.root_pos_w[:, 2] - table_height) < table_margin
    ee_low = ee_frame.data.target_pos_w[..., 0, 2] < ee_low_height
    lifted_enough = obj.data.root_pos_w[:, 2] > lift_height
    release_band = obj.data.root_pos_w[:, 2] < (release_height + table_margin)
    s4 = (near_goal & obj_low & ee_low & lifted_enough & release_band).float()
    hold_close = 1.0 - _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)
    return s4 * hold_close


def stage4_gripper_open_near_table_gated(env: ManagerBasedRLEnv, open_joint_pos: float, close_joint_pos: float,
                                         near_goal_xy: float, table_height: float,
                                         table_margin: float, ee_low_height: float, command_name: str,
                                         robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                         object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
                                         ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    near_goal = xy_dist < near_goal_xy
    near_table = torch.abs(obj.data.root_pos_w[:, 2] - table_height) < table_margin
    ee_low = ee_frame.data.target_pos_w[..., 0, 2] < ee_low_height

    gripper_open = _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)

    return (near_goal & near_table & ee_low).float() * gripper_open


def stage4_stable_placed_reward_gated(env: ManagerBasedRLEnv, xy_threshold: float, table_height: float, speed_threshold: float,
                                      command_name: str,
                                      robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                      object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    speed = torch.norm(obj.data.root_lin_vel_w[:, :3], dim=1)
    good_xy = xy_dist < xy_threshold
    near_table = torch.abs(z - table_height) < 0.015
    low_speed = speed < speed_threshold
    return (good_xy & near_table & low_speed).float()


def stage4_ee_away_after_place_gated(env: ManagerBasedRLEnv, ee_min_distance: float, xy_threshold: float, table_height: float,
                                     speed_threshold: float, command_name: str,
                                     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                                     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
                                     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    placed = stage4_stable_placed_reward_gated(env, xy_threshold, table_height, speed_threshold, command_name, robot_cfg, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(obj.data.root_pos_w[:, :3] - ee_w, dim=1)
    return placed * torch.clamp((dist - ee_min_distance) / max(ee_min_distance, 1e-3), min=0.0)
