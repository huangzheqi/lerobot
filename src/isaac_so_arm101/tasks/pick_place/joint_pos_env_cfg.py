import isaac_so_arm101.tasks.pick_place.mdp as mdp
from isaac_so_arm101.tasks.lift.joint_pos_env_cfg import SoArm101LiftCubeEnvCfg, SoArm101LiftCubeEnvCfg_PLAY
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class SoArm101PickPlaceCubeEnvCfg(SoArm101LiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=1.0)
        self.rewards.lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.025}, weight=5.0)

        gate_params = {
            "command_name": "object_pose",
            "lift_height": 0.045,
            "near_goal_xy": 0.06,
            "release_height": 0.045,
        }
        self.rewards.stage2_goal_xy_tracking_gated = RewTerm(
            func=mdp.stage2_goal_xy_tracking_gated, params={**gate_params, "std": 0.12}, weight=10.0
        )
        self.rewards.stage2_early_open_penalty_gated = RewTerm(
            func=mdp.stage2_early_open_penalty_gated, params={**gate_params, "open_joint_pos": 0.45, "close_joint_pos": 0.12}, weight=-5.0
        )
        self.rewards.stage3_soft_descent_reward_gated = RewTerm(
            func=mdp.stage3_soft_descent_reward_gated, params={**gate_params, "target_speed": 0.04}, weight=12.0
        )
        self.rewards.stage3_hard_drop_penalty_gated = RewTerm(
            func=mdp.stage3_hard_drop_penalty_gated, params={**gate_params, "max_down_speed": 0.08}, weight=-8.0
        )
        self.rewards.stage4_release_reward_gated = RewTerm(
            func=mdp.stage4_release_reward_gated, params={**gate_params, "open_joint_pos": 0.45, "close_joint_pos": 0.12}, weight=10.0
        )
        self.rewards.stage4_hold_too_long_penalty_gated = RewTerm(
            func=mdp.stage4_hold_too_long_penalty_gated, params={**gate_params, "open_joint_pos": 0.45, "close_joint_pos": 0.12}, weight=-8.0
        )
        self.rewards.stage4_gripper_open_near_table_gated = RewTerm(
            func=mdp.stage4_gripper_open_near_table_gated,
            params={
                "command_name": "object_pose",
                "open_joint_pos": 0.45,
                "close_joint_pos": 0.12,
                "near_goal_xy": 0.06,
                "table_height": 0.025,
                "table_margin": 0.02,
            },
            weight=10.0,
        )
        self.rewards.stage4_stable_placed_reward_gated = RewTerm(
            func=mdp.stage4_stable_placed_reward_gated,
            params={"command_name": "object_pose", "xy_threshold": 0.05, "table_height": 0.025, "speed_threshold": 0.08},
            weight=15.0,
        )
        self.rewards.stage4_ee_away_after_place_gated = RewTerm(
            func=mdp.stage4_ee_away_after_place_gated,
            params={
                "command_name": "object_pose",
                "ee_min_distance": 0.08,
                "xy_threshold": 0.05,
                "table_height": 0.025,
                "speed_threshold": 0.08,
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            weight=2.0,
        )

        self.rewards.object_goal_tracking.weight = 0.0
        self.rewards.object_goal_tracking_fine_grained.weight = 0.0


@configclass
class SoArm101PickPlaceCubeEnvCfg_PLAY(SoArm101PickPlaceCubeEnvCfg, SoArm101LiftCubeEnvCfg_PLAY):
    pass
