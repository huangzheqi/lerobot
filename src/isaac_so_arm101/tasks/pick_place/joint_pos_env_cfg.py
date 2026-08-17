import math

import isaac_so_arm101.tasks.pick_place.mdp as mdp
from isaac_so_arm101.policies.act_contract import ACT_JOINT_NAMES, EXPECTED_ACT_IMAGE_SHAPE
from isaac_so_arm101.robots import SO_ARM101_ACT_V5_CFG
from isaac_so_arm101.tasks.lift.joint_pos_env_cfg import SoArm101LiftCubeEnvCfg, SoArm101LiftCubeEnvCfg_PLAY
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg, PreviewSurfaceCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.utils import configclass

ACT_V5_GRIPPER_OPEN_JOINT_POS = 0.5
ACT_V5_GRIPPER_CLOSED_JOINT_POS = 0.0
ACT_V5_RESET_STATE_DEG_PERCENT = {
    "shoulder_pan": 1.1208792,
    "shoulder_lift": -103.27911,
    "elbow_flex": 96.08353,
    "wrist_flex": 55.89451,
    "wrist_roll": -2.3912086,
    "gripper": 5.4248366,
}


def _act_v5_reset_joint_pos_rad() -> dict[str, float]:
    """Return the 20-episode v5 first-frame mean in Isaac joint units."""

    reset_joint_pos = {name: math.radians(ACT_V5_RESET_STATE_DEG_PERCENT[name]) for name in ACT_JOINT_NAMES[:-1]}
    gripper_progress = ACT_V5_RESET_STATE_DEG_PERCENT["gripper"] / 100.0
    reset_joint_pos["gripper"] = ACT_V5_GRIPPER_OPEN_JOINT_POS - gripper_progress * (
        ACT_V5_GRIPPER_OPEN_JOINT_POS - ACT_V5_GRIPPER_CLOSED_JOINT_POS
    )
    return reset_joint_pos


@configclass
class ActStateObservationsCfg(ObsGroup):
    """ACT state input group: 6D SO101 joint positions."""

    joint_pos = ObsTerm(func=mdp.act_joint_pos_state)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ActFixedCameraObservationsCfg(ObsGroup):
    """ACT fixed-camera RGB input group."""

    rgb = ObsTerm(func=mdp.act_fixed_camera_rgb)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ActHandeyeCameraObservationsCfg(ObsGroup):
    """ACT hand-eye camera RGB input group."""

    rgb = ObsTerm(func=mdp.act_handeye_camera_rgb)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


def _add_act_observation_groups(env_cfg) -> None:
    """Attach ACT-specific observation groups without changing the existing PPO policy group."""

    env_cfg.observations.act_state = ActStateObservationsCfg()
    env_cfg.observations.act_fixed = ActFixedCameraObservationsCfg()
    env_cfg.observations.act_handeye = ActHandeyeCameraObservationsCfg()


def _configure_act_action_order(env_cfg) -> None:
    """Make ACT env action order match the checkpoint joint order."""

    env_cfg.actions.arm_action.joint_names = list(ACT_JOINT_NAMES[:-1])
    env_cfg.actions.arm_action.preserve_order = True


def _configure_act_input_cameras(env_cfg) -> None:
    """Configure the two RGB cameras expected by the ACT checkpoint."""

    env_cfg.commands.object_pose.debug_vis = False
    env_cfg.scene.ee_frame.debug_vis = False

    fixed_camera_pos = (0.85, -0.90, 0.90)
    fixed_camera_rot = (0.9009, 0.3898, 0.1213, 0.1472)
    handeye_camera_pos = (-0.013708, -0.045951, 0.082730)
    handeye_camera_rot = (0.521917, -0.477076, 0.453010, 0.542939)
    _, act_image_height, act_image_width = EXPECTED_ACT_IMAGE_SHAPE

    env_cfg.scene.fixed_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/fixed_camera",
        update_period=0.0,
        height=act_image_height,
        width=act_image_width,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=fixed_camera_pos, rot=fixed_camera_rot, convention="opengl"),
    )

    env_cfg.scene.light.spawn.intensity = 4500.0

    env_cfg.scene.handeye_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_link/handeye_camera",
        update_period=0.0,
        height=act_image_height,
        width=act_image_width,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=18.1475620269775,
            focus_distance=400.0,
            horizontal_aperture=20.954999923706055,
            clipping_range=(0.01, 10000000.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=handeye_camera_pos, rot=handeye_camera_rot, convention="opengl"),
    )


@configclass
class SoArm101PickPlaceCubeEnvCfg(SoArm101LiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # v9-faithful structure (experiment pick_place_v9_wrist_height_fix, commit 87d7dca):
        # the provably-working 44% grasp policy. Restored as the fallback after the place-on-table
        # experiments (low table-height goal + dense reward shaping) were abandoned: training to place
        # the cube ON the table via PPO shaping does not converge from these inits. Both reward
        # configurations were tried and both failed:
        #   - v14 (continuous lift reward): the policy farmed the lift term by lifting the cube and
        #     hovering at the pickup point -- grasp dropped 44.7%->26.5%, transport progress 56%->9%,
        #     placement ~0.
        #   - v15 (one-time lift crossing bonus + transport-dominant weights, warm-started from v9):
        #     removing the continuous lift payment caused a death spiral -- the one-shot bonus could
        #     not sustain lifting, the lifted-gated transport reward then never fired, and grasp
        #     collapsed entirely (lift bonus -> 0, mean_reward -> reach-only ~1.5) within 500 iters.
        # Conclusion: keep the high air goal (inherited 0.2-0.35) that made v9 grasp, and switch the
        # research direction to perception (ResNet cube-pose) on top of this working policy instead of
        # re-shaping the reward. NOTE: reward terms below are inert during play (they do not affect
        # the rollout); they are kept v9-faithful so a future v9 + ResNet fine-tune starts from the
        # exact landscape that produced the 44% grasp. Action/observation dims unchanged.
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
            func=mdp.stage2_early_open_penalty_gated,
            params={**gate_params, "open_joint_pos": 0.45, "close_joint_pos": 0.12},
            weight=-5.0,
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

        # v17 POST-MORTEM (2026-06-11): a push_never_lifted_penalty (-2.0, threshold 5cm) was
        # registered here and collapsed the fine-tune within ~150 iters (lifting_object 0.8 -> 0.05
        # over 960 iters). Root cause measured afterwards in the GT funnel: v9's WORKING grasp
        # strategy is push-and-chase -- 67% of SUCCESSFUL grasps displace the cube >5cm before the
        # lift (median max 7.6cm, median 12 / p90 81 steps beyond 5cm = -24..-160 per episode), so
        # the penalty taxed the main income path, not just the failures. The "clean grasps only
        # nudge 2-3cm" assumption was wrong. Do NOT re-add a displacement-based push penalty; the
        # function remains in mdp/rewards.py for reference. v18 = the latched stage3/4 landscape
        # (was_lifted latches, already in rewards.py) + 1cm obs noise, with no extra penalty.

        # Pose-noise robustness (domain randomization). The grasp is hyper-sensitive to error in the
        # observed object position -- a sensitivity sweep with a per-episode fixed XY offset gave
        # grasp 40%@0cm, 17%@2cm, 7%@4cm, 3%@6cm. The v9 policy was trained on EXACT GT, so any
        # perception error (the ResNet gives ~6cm at play) collapses it. Injecting per-step Gaussian
        # noise (std 2cm) into the object_position observation during TRAINING forces the policy to
        # stop over-trusting the exact position and to widen its grasp-approach basin, so it can still
        # grasp when the position is off by a few cm -- i.e. it shifts that sensitivity curve right.
        # Obs noise on object_position: DISABLED for v19. Every fine-tune that carried it collapsed
        # with the same slow-bleed lifting_object curve (v16/v16b @2cm, v17/v18 @1cm), while v9
        # itself trained noise-free. The robustness it was buying is no longer needed: the live
        # colour-mask pipeline delivers 0.2-0.9cm median / 2cm p90 unbiased per-step error, well
        # inside the policy's tolerance. (GaussianNoiseCfg(std=...) was here; see git history.)
        self.observations.policy.object_position.noise = None

        # Curriculum shock fix (root cause of the v16 collapse). Fine-tuning a converged v9 starts a
        # NEW run, so common_step_counter resets to 0 and the inherited Lift curriculum re-fires at
        # ~iter 417 (common_step_counter==10000 / 24 per iter), slamming the joint_vel & action_rate
        # penalties from -1e-4 to -1e-1 mid-run. Evidence: in the v16 run mean_reward was a healthy
        # 5.46 WITH the obs noise at fine-tune iter ~250, then crashed to -1.95 exactly when the
        # curriculum jumped (joint_vel Episode_Reward 0 -> -0.48), oscillating without re-converging
        # in the remaining ~580 iters -> grasp collapsed to 2%. The obs noise was NOT the cause (the
        # policy tolerated it fine before the jump). v9 already CONVERGED under the final -1e-1, so
        # apply that level from step 0 (num_steps=0): no mid-run shock, in-distribution for the loaded
        # policy. This unblocks an actual test of obs-noise robustness.
        self.curriculum.joint_vel.params["num_steps"] = 0
        self.curriculum.action_rate.params["num_steps"] = 0

        # Recolour the cube uniformly RED via a visual-material override on the spawned DexCube USD.
        # Appearance-only: mass/friction/collision are untouched, so v9 (trained on this exact
        # physics) is unaffected and no grasp re-verification is needed. Why: the stock DexCube is
        # white-ish, and in the handeye view during the approach it sits inside the gripper's
        # dome-light-occlusion shadow rendering as a black blob (FOV sweeps check3-7). Shadowed
        # pixels keep their HUE (the shadow still receives ~13% ambient), so a saturated red cube
        # stays red-detectable even in shadow -- and it matches the planned real-world red cube.
        self.scene.object.spawn.visual_material = PreviewSurfaceCfg(diffuse_color=(0.9, 0.08, 0.08))

        # Heavier cube (user-requested, 2026-06-10). PHYSICS change -- unlike the recolour above this
        # can shift v9's behaviour, so the GT funnel must be re-verified after it. Why: the stock
        # DexCube is so light that v9's closing fingers skate it across the table 2-6cm before
        # securing a grip; every observed vision-mode failure involved such a push, and the grasp is
        # hyper-sensitive to the resulting chase. More inertia + more friction force (mu*m*g) makes
        # the cube stay put under finger contact, and better matches a real graspable object.
        # 0.10 kg for a 3.25cm cube is between wood (~20g) and solid aluminium (~90g) -- dense but
        # easily within the gripper's lifting capability.
        self.scene.object.spawn.mass_props = MassPropertiesCfg(mass=0.10)


@configclass
class SoArm101PickPlaceCubeEnvCfg_PLAY(SoArm101PickPlaceCubeEnvCfg, SoArm101LiftCubeEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        # Evaluate on the clean observation: the per-episode gt_pose_noise diagnostic (play.py) is the
        # only controlled noise source at play time, so disable the training-time obs noise here.
        self.observations.policy.object_position.noise = None

        # Longer evaluation horizon (PLAY-ONLY): v9 trained at 5s, but a drop + re-grasp + place can
        # exceed 5s and then time out as a "failure" even though the policy could finish. Extending
        # the episode lets those runs complete. MUST extend the goal resampling window too --
        # object_pose resamples every 5s by default, so an 8s episode would otherwise re-randomise the
        # target at t=5s mid-episode. Training (Isaac-SO-ARM101-Pick-Place-Cube-v0) is unaffected; v9
        # weights untouched. Note: behaviour past 5s is out-of-distribution for v9 (it never saw t>5s),
        # so read long-horizon runs as "can it finish given more time", not as native competence.
        self.episode_length_s = 8.0
        self.commands.object_pose.resampling_time_range = (8.0, 8.0)


@configclass
class SoArm101PickPlaceCubeActObsEnvCfg(SoArm101PickPlaceCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ACT consumes two high-resolution camera streams, so keep the default ACT env count small.
        # Users can still override this with --num_envs from the train script.
        self.scene.num_envs = 4

        # v32 ACT-only contact stress test: v31 showed sub-millimetre cube rise but never opened the
        # real lift gate. Keep base pick-place unchanged and make this ACT run intentionally easier
        # physically so we can tell whether the remaining blocker is contact geometry/action pose.
        self.scene.object.spawn.mass_props = MassPropertiesCfg(mass=0.015)
        act_contact_material = RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.0,
            friction_combine_mode="max",
        )
        self.sim.physics_material = act_contact_material
        self.scene.object.spawn.physics_material = RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.0,
            friction_combine_mode="max",
        )
        self.scene.table.spawn.physics_material = RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.0,
            friction_combine_mode="max",
        )
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 24
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 12
        self.scene.object.spawn.rigid_props.solver_position_iteration_count = 32
        self.scene.object.spawn.rigid_props.solver_velocity_iteration_count = 12
        self.scene.robot.actuators["gripper"].effort_limit_sim = 10.0
        self.scene.robot.actuators["gripper"].stiffness = 160.0
        self.scene.robot.actuators["gripper"].damping = 50.0

        # ACT starts out-of-distribution in sim, so do not apply the final v9 smoothness penalties
        # from step 0. The residual actor needs early freedom to correct the real-to-sim prior.
        self.rewards.action_rate.weight = -1.0e-4
        self.rewards.joint_vel.weight = -1.0e-4
        self.curriculum.action_rate.params["num_steps"] = 1_000_000_000
        self.curriculum.joint_vel.params["num_steps"] = 1_000_000_000

        # ACT-only bootstrap: v23 reached the cube only in sparse spikes and never opened the
        # lift/transport gates. Widen the early reward basin and add command-timing hints so PPO can
        # first learn a stable approach-grasp-lift entry before optimizing the later placement stages.
        self.rewards.reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.12}, weight=3.0)
        self.rewards.act_approach_gripper_open = RewTerm(
            func=mdp.approach_gripper_open_reward,
            params={"contact_distance": 0.10, "approach_outer": 0.18, "lift_height": 0.045},
            weight=0.2,
        )
        self.rewards.act_grasp_close_at_contact = RewTerm(
            func=mdp.grasp_close_at_contact_reward, params={"contact_distance": 0.10}, weight=1.0
        )
        self.rewards.act_grasp_close_true_contact = RewTerm(
            func=mdp.grasp_close_at_contact_reward, params={"contact_distance": 0.06}, weight=5.0
        )
        self.rewards.act_open_at_contact_penalty = RewTerm(
            func=mdp.grasp_open_at_contact_penalty, params={"contact_distance": 0.10}, weight=-1.5
        )
        self.rewards.act_closed_near_upward_motion = RewTerm(
            func=mdp.closed_near_upward_motion_reward,
            params={"contact_distance": 0.09, "target_up_delta": 0.0015, "lift_height": 0.055},
            weight=6.0,
        )
        self.rewards.act_closed_contact_lift_pose = RewTerm(
            func=mdp.closed_contact_lift_pose_reward,
            params={"contact_distance": 0.09, "target_ee_above_object": 0.035, "lift_height": 0.055},
            weight=1.0,
        )
        self.rewards.act_closed_contact_object_rise = RewTerm(
            func=mdp.closed_contact_object_rise_reward,
            params={
                "contact_distance": 0.09,
                "lift_cap": 0.025,
                "lift_height": 0.055,
                "min_height_gain": 0.002,
                "initial_object_z": 0.015,
            },
            weight=30.0,
        )
        self.rewards.act_dense_lift_from_table = RewTerm(
            func=mdp.object_lifted_from_initial_reward,
            params={"lift_cap": 0.025, "min_height_gain": 0.002, "initial_object_z": 0.015},
            weight=40.0,
        )
        self.rewards.act_lifted_close_hold_cmd = RewTerm(
            func=mdp.lifted_close_hold_cmd_reward,
            params={"min_height_gain": 0.003, "near_distance": 0.10, "initial_object_z": 0.015},
            weight=10.0,
        )
        self.rewards.act_diag_object_height_gain_m = RewTerm(
            func=mdp.diag_object_height_gain_m,
            params={"initial_object_z": 0.015, "height_cap": 0.08},
            weight=1.0,
        )
        self.rewards.act_diag_gripper_closed_joint = RewTerm(
            func=mdp.diag_gripper_closed_joint_ratio,
            params={"open_joint_pos": 0.5, "close_joint_pos": 0.0},
            weight=0.01,
        )
        self.rewards.act_diag_near_object_06 = RewTerm(
            func=mdp.diag_near_object,
            params={"contact_distance": 0.06},
            weight=0.01,
        )
        _configure_act_action_order(self)
        _configure_act_input_cameras(self)
        _add_act_observation_groups(self)


@configclass
class SoArm101PickPlaceCubeActObsEnvCfg_PLAY(SoArm101PickPlaceCubeActObsEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.object_position.noise = None
        self.episode_length_s = 8.0
        self.commands.object_pose.resampling_time_range = (8.0, 8.0)


@configclass
class SoArm101PickPlaceCubeActV5EnvCfg(SoArm101PickPlaceCubeEnvCfg):
    """Factory-zero ACT-v5 environment isolated from the legacy v4/v32 task."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.robot = SO_ARM101_ACT_V5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos.update(_act_v5_reset_joint_pos_rad())
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=ACT_V5_GRIPPER_OPEN_JOINT_POS,
            offset=0.0,
            use_default_offset=False,
            preserve_order=True,
        )
        _configure_act_action_order(self)
        _configure_act_input_cameras(self)
        _add_act_observation_groups(self)


@configclass
class SoArm101PickPlaceCubeActV5EnvCfg_PLAY(SoArm101PickPlaceCubeActV5EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.object_position.noise = None
        self.episode_length_s = 8.0
        self.commands.object_pose.resampling_time_range = (8.0, 8.0)


@configclass
class SoArm101PickPlaceCubeActV34EnvCfg(SoArm101PickPlaceCubeActObsEnvCfg):
    """Factory-zero ACT-v5 task with safe prior motion and ACT bootstrap rewards."""

    def __post_init__(self):
        super().__post_init__()

        # Reuse the v32 ACT reward/contact bootstrap, then replace its legacy robot
        # with the factory-zero V5 articulation and action contract.
        self.scene.robot = SO_ARM101_ACT_V5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos.update(_act_v5_reset_joint_pos_rad())
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 24
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 12
        self.scene.robot.actuators["gripper"].effort_limit_sim = 10.0
        self.scene.robot.actuators["gripper"].stiffness = 160.0
        self.scene.robot.actuators["gripper"].damping = 50.0
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=ACT_V5_GRIPPER_OPEN_JOINT_POS,
            offset=0.0,
            use_default_offset=False,
            preserve_order=True,
        )
        _configure_act_action_order(self)

        # The real demonstrations last about 20 seconds. Keep the command fixed for
        # the same horizon so it cannot jump to a new goal midway through a rollout.
        self.episode_length_s = 20.0
        self.commands.object_pose.resampling_time_range = (20.0, 20.0)


@configclass
class SoArm101PickPlaceCubeActV34EnvCfg_PLAY(SoArm101PickPlaceCubeActV34EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.object_position.noise = None


@configclass
class SoArm101PickPlaceCubeActV35EnvCfg(SoArm101PickPlaceCubeActV34EnvCfg):
    """V34 ACT-v5 task with self-collisions disabled for a stable reset pose."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False


@configclass
class SoArm101PickPlaceCubeActV35EnvCfg_PLAY(SoArm101PickPlaceCubeActV35EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.object_position.noise = None


@configclass
class SoArm101PickPlaceCubeActV36EnvCfg(SoArm101PickPlaceCubeActV35EnvCfg):
    """V35 dynamics with rewards aligned to the 1=open, 0=closed action contract."""

    def __post_init__(self):
        super().__post_init__()

        gripper_action_contract = {"open_action": 1.0, "close_action": 0.0}
        close_aware_terms = (
            "act_approach_gripper_open",
            "act_grasp_close_at_contact",
            "act_grasp_close_true_contact",
            "act_open_at_contact_penalty",
            "act_closed_near_upward_motion",
            "act_closed_contact_lift_pose",
            "act_closed_contact_object_rise",
            "act_lifted_close_hold_cmd",
        )
        for term_name in close_aware_terms:
            getattr(self.rewards, term_name).params.update(gripper_action_contract)

        # Low-weight diagnostics expose the sampled PPO action contract in TensorBoard.
        # TensorBoard reports episode-integrated values, so use them for trend comparison
        # rather than interpreting them as an instantaneous [0, 1] ratio.
        self.rewards.act_diag_gripper_close_action_progress = RewTerm(
            func=mdp.diag_gripper_close_action_progress,
            params=gripper_action_contract,
            weight=0.01,
        )
        self.rewards.act_diag_near_object_10 = RewTerm(
            func=mdp.diag_near_object,
            params={"contact_distance": 0.10},
            weight=0.01,
        )


@configclass
class SoArm101PickPlaceCubeActV36EnvCfg_PLAY(SoArm101PickPlaceCubeActV36EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.object_position.noise = None


@configclass
class SoArm101PickPlaceCubeActV37EnvCfg(SoArm101PickPlaceCubeActV36EnvCfg):
    """V36 environment paired with the new factory-zero ACT checkpoint."""


@configclass
class SoArm101PickPlaceCubeActV37EnvCfg_PLAY(SoArm101PickPlaceCubeActV37EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.object_position.noise = None


@configclass
class SoArm101PickPlaceCubeVisionEnvCfg_PLAY(SoArm101PickPlaceCubeEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # The camera images are model INPUT in this variant (resnet/colour-mask cube localisation and
        # dataset collection), so the scene must be visually clean. The inherited debug markers render
        # INTO the cameras: the object_pose command draws an RGB axis gizmo at the cube's current pose
        # (its red X-arrow cone is easily mistaken for - and occludes - the red cube in the handeye
        # view) plus one at the goal, and ee_frame draws another at the gripper.
        self.commands.object_pose.debug_vis = False
        self.scene.ee_frame.debug_vis = False

        # Camera extrinsics:
        # - fixed_camera: z controls camera height, x/y controls the table observation position, rot controls viewing direction.
        # - handeye_camera: pose is calibrated in Isaac Sim "Create from View" and set as wrist_link local offset.
        fixed_camera_pos = (0.85, -0.90, 0.90)
        fixed_camera_rot = (0.9009, 0.3898, 0.1213, 0.1472)
        # AUTHORITATIVE handeye calibration (user's Isaac Sim "Create from View" session, 2026-05-25,
        # restored 2026-06-12 from the original calibration printout). The values that previously
        # lived here -- pos (0.051129, 0.120000, 0.078330) rot (0.571647, -0.416195, 0.505849,
        # 0.490483) focal 22 -- were NOT that calibration and produced a visibly wrong wrist view
        # (user caught it in the GUI). These extrinsics match the REAL SO101 wrist camera mount and
        # MUST NOT be tuned to fix sim-side symptoms; recalibrate from real captures if they ever
        # seem off. NOTE: the calibration printout's rot 3rd component is 0.453010 (the unit
        # quaternion); a restatement elsewhere in that session had a 0.430100 transposition typo.
        handeye_camera_pos = (-0.013708, -0.045951, 0.082730)
        handeye_camera_rot = (0.521917, -0.477076, 0.453010, 0.542939)

        self.scene.fixed_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/fixed_camera",
            update_period=0.0,
            # 256x256 (was 128): the ResNet localises the cube far better on sharper images -- the
            # cube is small in this far fixed view, and 128 blurs it. The original well-trained
            # dataset (corr 0.92) was 256; matching that resolution for BOTH collection and play
            # keeps train==play while making the cube learnable. Higher render cost is acceptable.
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=fixed_camera_pos, rot=fixed_camera_rot, convention="opengl"),
        )

        # NOTE on lighting: a wrist-mounted fill light was tried (SphereLightCfg under
        # Robot/wrist_link, FOV sweeps check4-7) and abandoned -- the renderer does not move lights
        # parented to physics-driven links (light xforms are not updated from physics the way camera
        # sensors track their pose), so the lamp stayed at its spawn pose as a static light and never
        # illuminated the grasp zone. Shadow robustness comes from the red cube recolour (hue
        # survives shadow) + a dome intensity bump instead.
        self.scene.light.spawn.intensity = 4500.0

        self.scene.handeye_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_link/handeye_camera",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb"],
            # Intrinsics from the same 2026-05-25 calibration printout as the extrinsics above.
            spawn=PinholeCameraCfg(
                focal_length=18.1475620269775,
                focus_distance=400.0,
                horizontal_aperture=20.954999923706055,
                clipping_range=(0.01, 10000000.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=handeye_camera_pos, rot=handeye_camera_rot, convention="opengl"),
        )
