"""Observation helpers for ACT-conditioned pick-place policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaac_so_arm101.policies.act_contract import ACT_JOINT_NAMES, EXPECTED_ACT_IMAGE_SHAPE

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def act_joint_pos_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=list(ACT_JOINT_NAMES),
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Return SO101 joint positions in the 6D order expected by the ACT checkpoint."""

    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.joint_pos[:, robot_cfg.joint_ids]


def _camera_rgb_chw(
    env: ManagerBasedRLEnv,
    camera_name: str,
    image_shape: tuple[int, int, int] = EXPECTED_ACT_IMAGE_SHAPE,
) -> torch.Tensor:
    """Return a camera RGB image as float NCHW in [0, 1]."""

    camera = env.scene[camera_name]
    rgb = camera.data.output["rgb"]
    if rgb.ndim != 4:
        raise ValueError(f"Expected {camera_name!r} RGB to be 4D, got shape={tuple(rgb.shape)}")

    if rgb.shape[-1] in (3, 4):
        rgb = rgb[..., :3].permute(0, 3, 1, 2)
    elif rgb.shape[1] in (3, 4):
        rgb = rgb[:, :3]
    else:
        raise ValueError(f"Expected {camera_name!r} RGB to have 3 or 4 channels, got shape={tuple(rgb.shape)}")

    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0
    else:
        rgb = rgb.float()

    expected_channels, expected_height, expected_width = image_shape
    if rgb.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {camera_name!r} RGB to have {expected_channels} channels, got shape={tuple(rgb.shape)}"
        )
    if rgb.shape[-2:] != (expected_height, expected_width):
        rgb = F.interpolate(rgb, size=(expected_height, expected_width), mode="bilinear", align_corners=False)

    return rgb.contiguous()


def act_fixed_camera_rgb(
    env: ManagerBasedRLEnv,
    camera_name: str = "fixed_camera",
    image_shape: tuple[int, int, int] = EXPECTED_ACT_IMAGE_SHAPE,
) -> torch.Tensor:
    """Return fixed-camera RGB formatted for the ACT checkpoint."""

    return _camera_rgb_chw(env, camera_name=camera_name, image_shape=image_shape)


def act_handeye_camera_rgb(
    env: ManagerBasedRLEnv,
    camera_name: str = "handeye_camera",
    image_shape: tuple[int, int, int] = EXPECTED_ACT_IMAGE_SHAPE,
) -> torch.Tensor:
    """Return wrist hand-eye RGB formatted for the ACT checkpoint."""

    return _camera_rgb_chw(env, camera_name=camera_name, image_shape=image_shape)
