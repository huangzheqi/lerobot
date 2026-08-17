import pytest
import torch

from isaac_so_arm101.policies.act_joint_mapping import (
    arm_degrees_to_radians,
    arm_radians_to_degrees,
    gripper_action_to_close_progress,
    gripper_joint_to_percent,
    gripper_percent_to_joint,
)


def test_arm_angle_round_trip() -> None:
    joint_pos_rad = torch.tensor([[-1.5, -0.25, 0.0, 0.75, 1.25]])

    result = arm_degrees_to_radians(arm_radians_to_degrees(joint_pos_rad))

    torch.testing.assert_close(result, joint_pos_rad)


def test_gripper_mapping_endpoints_and_midpoint() -> None:
    joint_pos = torch.tensor([[0.5], [0.25], [0.0]])

    percent = gripper_joint_to_percent(
        joint_pos,
        open_joint_pos=0.5,
        closed_joint_pos=0.0,
    )

    torch.testing.assert_close(percent, torch.tensor([[0.0], [50.0], [100.0]]))
    torch.testing.assert_close(
        gripper_percent_to_joint(
            percent,
            open_joint_pos=0.5,
            closed_joint_pos=0.0,
        ),
        joint_pos,
    )


def test_gripper_mapping_clamps_out_of_range_values() -> None:
    percent = torch.tensor([[-10.0], [110.0]])

    result = gripper_percent_to_joint(
        percent,
        open_joint_pos=0.5,
        closed_joint_pos=0.0,
    )

    torch.testing.assert_close(result, torch.tensor([[0.5], [0.0]]))


def test_gripper_mapping_rejects_reversed_endpoints() -> None:
    with pytest.raises(ValueError, match="open_joint_pos must be greater"):
        gripper_joint_to_percent(
            torch.tensor([[0.25]]),
            open_joint_pos=0.0,
            closed_joint_pos=0.5,
        )


def test_gripper_action_to_close_progress() -> None:
    action = torch.tensor([[1.2], [1.0], [0.75], [0.25], [0.0], [-0.2]])

    result = gripper_action_to_close_progress(
        action,
        open_action=1.0,
        close_action=0.0,
    )

    torch.testing.assert_close(
        result,
        torch.tensor([[0.0], [0.0], [0.25], [0.75], [1.0], [1.0]]),
    )


def test_gripper_action_to_close_progress_rejects_reversed_endpoints() -> None:
    with pytest.raises(ValueError, match="open_action must be greater"):
        gripper_action_to_close_progress(
            torch.tensor([[0.5]]),
            open_action=0.0,
            close_action=1.0,
        )
