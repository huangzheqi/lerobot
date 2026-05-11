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


def stage2_goal_xy_tracking_gated(env: ManagerBasedRLEnv, std: float, lift_height: float, near_goal_xy: float,
                                  release_height: float, command_name: str,
                                  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                  object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    _, s2, _, _ = _gates(env, lift_height, near_goal_xy, release_height, command_name, robot_cfg, object_cfg)
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    return s2 * (1.0 - torch.tanh(xy_dist / std))


def stage2_early_open_penalty_gated(env: ManagerBasedRLEnv, open_joint_pos: float, lift_height: float, near_goal_xy: float,
                                    release_height: float, command_name: str,
                                    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    _, s2, _, _ = _gates(env, lift_height, near_goal_xy, release_height, command_name, SceneEntityCfg("robot"), SceneEntityCfg("object"))
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_idx = robot.find_joints("gripper")[0][0]
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    return s2 * torch.clamp((gripper_pos - open_joint_pos) / max(open_joint_pos, 1e-3), min=0.0)


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


def stage4_release_reward_gated(env: ManagerBasedRLEnv, open_joint_pos: float, lift_height: float, near_goal_xy: float,
                                release_height: float, command_name: str,
                                robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    _, _, _, s4 = _gates(env, lift_height, near_goal_xy, release_height, command_name, SceneEntityCfg("robot"), SceneEntityCfg("object"))
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_idx = robot.find_joints("gripper")[0][0]
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    return s4 * torch.clamp(gripper_pos / max(open_joint_pos, 1e-3), 0.0, 1.0)


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
