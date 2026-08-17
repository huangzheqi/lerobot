"""Pure SO101 joint-unit conversions shared by ACT adapters and replay tools."""

from __future__ import annotations

import torch


def arm_radians_to_degrees(joint_pos: torch.Tensor) -> torch.Tensor:
    """Convert SO101 arm joint positions from Isaac radians to ACT degrees."""

    return torch.rad2deg(joint_pos)


def arm_degrees_to_radians(joint_pos: torch.Tensor) -> torch.Tensor:
    """Convert SO101 arm joint positions from ACT degrees to Isaac radians."""

    return torch.deg2rad(joint_pos)


def _gripper_span(open_joint_pos: float, closed_joint_pos: float) -> float:
    span = float(open_joint_pos) - float(closed_joint_pos)
    if span <= 0.0:
        raise ValueError(
            "open_joint_pos must be greater than closed_joint_pos, "
            f"got open={open_joint_pos}, closed={closed_joint_pos}"
        )
    return span


def gripper_joint_to_percent(
    joint_pos: torch.Tensor,
    *,
    open_joint_pos: float,
    closed_joint_pos: float,
) -> torch.Tensor:
    """Map an Isaac gripper joint position to calibrated ACT percentage.

    ACT/LeRobot uses 0% for fully open and 100% for fully closed. Values are
    clamped because simulation contact can push the measured joint slightly
    outside the configured command endpoints.
    """

    span = _gripper_span(open_joint_pos, closed_joint_pos)
    progress = (float(open_joint_pos) - joint_pos) / span
    return progress.clamp(0.0, 1.0) * 100.0


def gripper_percent_to_joint(
    percent: torch.Tensor,
    *,
    open_joint_pos: float,
    closed_joint_pos: float,
) -> torch.Tensor:
    """Map calibrated ACT percentage to an Isaac gripper joint target."""

    span = _gripper_span(open_joint_pos, closed_joint_pos)
    progress = (percent / 100.0).clamp(0.0, 1.0)
    return float(open_joint_pos) - progress * span


def gripper_action_to_close_progress(
    action: torch.Tensor,
    *,
    open_action: float,
    close_action: float,
) -> torch.Tensor:
    """Convert a continuous gripper action into closure progress in [0, 1].

    V34+ commands the gripper with 1=open and 0=closed. Keeping this
    conversion separate from joint-position mapping makes the reward contract
    explicit and avoids sending negative targets into the physical joint limit.
    """

    span = float(open_action) - float(close_action)
    if span <= 0.0:
        raise ValueError(
            "open_action must be greater than close_action, "
            f"got open={open_action}, close={close_action}"
        )
    return ((float(open_action) - action) / span).clamp(0.0, 1.0)
