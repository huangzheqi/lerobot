import isaac_so_arm101.tasks.pick_place.mdp as mdp
from isaac_so_arm101.tasks.lift.joint_pos_env_cfg import SoArm101LiftCubeEnvCfg, SoArm101LiftCubeEnvCfg_PLAY
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg, PreviewSurfaceCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.utils import configclass


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
