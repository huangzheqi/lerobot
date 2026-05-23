import isaac_so_arm101.tasks.pick_place.mdp as mdp
from isaac_so_arm101.tasks.lift.joint_pos_env_cfg import SoArm101LiftCubeEnvCfg, SoArm101LiftCubeEnvCfg_PLAY
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass


@configclass
class SoArm101PickPlaceCubeEnvCfg(SoArm101LiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=1.2)
        self.rewards.lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.025}, weight=5.5)

        gate_params = {
            "command_name": "object_pose",
            "lift_height": 0.045,
            "near_goal_xy": 0.06,
            "release_height": 0.045,
        }
        self.rewards.stage2_goal_xy_tracking_gated = RewTerm(
            func=mdp.stage2_goal_xy_tracking_gated, params={**gate_params, "std": 0.12}, weight=11.0
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
        self.rewards.stage3_ee_low_near_goal_gated = RewTerm(
            func=mdp.stage3_ee_low_near_goal_gated,
            params={
                "command_name": "object_pose",
                "near_goal_xy": 0.07,
                "target_ee_height": 0.060,
                "ee_height_std": 0.020,
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            weight=14.0,
        )
        self.rewards.stage3_object_height_near_table_gated = RewTerm(
            func=mdp.stage3_object_height_near_table_gated,
            params={
                "command_name": "object_pose",
                "near_goal_xy": 0.07,
                "table_height": 0.025,
                "table_margin": 0.018,
            },
            weight=14.0,
        )
        self.rewards.stage3_wrist_flex_release_pose_gated = RewTerm(
            func=mdp.stage3_wrist_flex_release_pose_gated,
            params={
                "command_name": "object_pose",
                "near_goal_xy": 0.07,
                # Assumption: `wrist_flex` exists and lower/negative position bends down for release.
                "wrist_target_pos": -0.50,
                "wrist_std": 0.40,
            },
            weight=2.0,
        )
        self.rewards.stage4_release_reward_gated = RewTerm(
            func=mdp.stage4_release_reward_gated,
            params={
                **gate_params,
                "open_joint_pos": 0.45,
                "close_joint_pos": 0.12,
                "table_height": 0.025,
                "table_margin": 0.025,
                "ee_low_height": 0.070,
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            weight=12.0,
        )
        self.rewards.stage4_hold_too_long_penalty_gated = RewTerm(
            func=mdp.stage4_hold_too_long_penalty_gated,
            params={
                **gate_params,
                "open_joint_pos": 0.45,
                "close_joint_pos": 0.12,
                "table_height": 0.025,
                "table_margin": 0.025,
                "ee_low_height": 0.070,
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            weight=-10.0,
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
                "ee_low_height": 0.07,
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            weight=11.0,
        )
        self.rewards.stage4_stable_placed_reward_gated = RewTerm(
            func=mdp.stage4_stable_placed_reward_gated,
            params={"command_name": "object_pose", "xy_threshold": 0.05, "table_height": 0.025, "speed_threshold": 0.08},
            weight=16.0,
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


@configclass
class SoArm101PickPlaceCubeVisionEnvCfg_PLAY(SoArm101PickPlaceCubeEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # Camera extrinsics for quick manual tuning:
        # - fixed_camera: z controls camera height, x/y controls the table observation position, rot controls viewing direction.
        # - handeye_camera: pos controls camera mount position relative to gripper, rot controls lens direction.
        fixed_camera_pos = (0.85, -0.90, 0.90)
        fixed_camera_rot = (0.9009, 0.3898, 0.1213, 0.1472)
        handeye_camera_pos = (0.06, -0.12, 0.04)
        # Fallback option if handeye_camera shifts to the wrong side or the gripper is not visible:
        # handeye_camera_pos = (0.04, 0.14, 0.045)
        # If the camera is still too close to the robot body, try:
        # handeye_camera_pos = (0.02, -0.18, 0.055)
        # If the tabletop is visible but the gripper is still missing, try:
        # handeye_camera_pos = (0.02, -0.10, 0.04)
        # hand-eye camera rotation presets (w, x, y, z) for quick manual switching:
        # preset_a = (0.5, 0.5, -0.5, -0.5)
        # preset_b = (0.5, -0.5, 0.5, -0.5)
        # preset_c = (0.2706, -0.6533, 0.2706, -0.6533)
        # preset_d = (0.6533, -0.2706, 0.6533, -0.2706)
        handeye_camera_rot = (0.5, 0.5, -0.5, -0.5)

        self.scene.fixed_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/fixed_camera",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=fixed_camera_pos, rot=fixed_camera_rot, convention="opengl"),
        )

        self.scene.handeye_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_link/handeye_camera",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=6.0,
                focus_distance=200.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=handeye_camera_pos, rot=handeye_camera_rot, convention="ros"),
        )

        self.scene.handeye_debug_a = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_link/handeye_debug_a",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=6.0,
                focus_distance=200.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.06, -0.12, 0.04), rot=(0.5, 0.5, -0.5, -0.5), convention="ros"),
        )

        self.scene.handeye_debug_b = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_link/handeye_debug_b",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=6.0,
                focus_distance=200.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.06, -0.12, 0.04), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )

        self.scene.handeye_debug_c = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_link/handeye_debug_c",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=6.0,
                focus_distance=200.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.06, 0.12, 0.04), rot=(0.5, 0.5, -0.5, -0.5), convention="ros"),
        )

        self.scene.handeye_debug_d = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_link/handeye_debug_d",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=6.0,
                focus_distance=200.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.06, 0.12, 0.04), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )

        self.scene.handeye_debug_e = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_link/handeye_debug_e",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=6.0,
                focus_distance=200.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.03, -0.16, 0.05), rot=(0.5, 0.5, -0.5, -0.5), convention="ros"),
        )
