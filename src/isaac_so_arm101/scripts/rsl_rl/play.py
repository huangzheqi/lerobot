# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import sys

from isaaclab.app import AppLauncher

# local imports
import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--disturbance_type",
    type=str,
    default="off",
    choices=["off", "cube_init", "goal", "table_friction", "clutter", "lighting", "camera", "all"],
    help="Robustness disturbance type for pick-place eval.",
)
parser.add_argument(
    "--disturbance_level",
    type=str,
    default="off",
    choices=["off", "low", "medium", "high"],
    help="Robustness disturbance level.",
)
parser.add_argument(
    "--object_pose_source",
    type=str,
    default="gt",
    choices=["gt", "vision", "resnet"],
    help="Source of object pose used in policy observation.",
)
parser.add_argument(
    "--resnet_model_path",
    type=str,
    default="selected_models/resnet18_cube_pose.pt",
    help="Path to trained ResNet18 cube pose model.",
)

parser.add_argument(
    "--save_gt_debug",
    action="store_true",
    default=False,
    help="Save per-episode CSV diagnostics when object_pose_source=gt.",
)
parser.add_argument(
    "--gt_debug_dir",
    type=str,
    default="logs/gt_debug",
    help="Directory to save GT object-pose diagnostic CSV files.",
)
parser.add_argument(
    "--gt_pose_noise_xy",
    type=float,
    default=0.0,
    help=(
        "Diagnostic only (object_pose_source=gt): inject a per-episode FIXED Gaussian XY offset "
        "(std in metres) into the object position the policy observes, to measure how grasp "
        "degrades vs pose error. Mimics a ResNet that freezes one biased estimate per episode. "
        "0 = exact GT (default)."
    ),
)
parser.add_argument(
    "--save_camera_debug",
    action="store_true",
    help="Save camera debug images when using vision-based object pose.",
)
parser.add_argument(
    "--camera_debug_dir",
    type=str,
    default="logs/vision_debug",
    help="Directory to save camera debug images.",
)
parser.add_argument(
    "--camera_debug_interval",
    type=int,
    default=100,
    help="Save camera debug images every N steps.",
)
parser.add_argument(
    "--save_resnet_hard_samples",
    action="store_true",
    default=False,
    help="Save hard samples when ResNet object-pose error is above threshold.",
)
parser.add_argument(
    "--hard_sample_dir",
    type=str,
    default="data/cube_pose_hard_samples",
    help="Directory to save ResNet hard samples.",
)
parser.add_argument(
    "--hard_sample_error_threshold",
    type=float,
    default=0.08,
    help="Save sample when vision_error_xy exceeds this threshold.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# Enable rendering whenever cameras will be spawned: for video recording, for the vision/resnet pose
# sources (they read fixed_camera), or for any Vision-Play task (which spawns cameras regardless of
# the pose source). Without --enable_cameras the env raises "A camera was spawned without the
# --enable_cameras flag" at camera init.
if (
    args_cli.video
    or args_cli.object_pose_source in ("vision", "resnet")
    or (args_cli.task and "Vision" in args_cli.task)
):
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Import Isaac Lab task extensions only after SimulationApp is created.
import isaac_so_arm101.tasks  # noqa: F401
from isaac_so_arm101.tasks.pick_place.robust_eval_cfg import apply_pick_place_disturbance # isort: skip

import gymnasium as gym
import numpy as np
import os
import time
import torch
from PIL import Image, ImageDraw
from isaac_so_arm101.scripts.rsl_rl.vision_pose_resnet import ResnetCubePoseEstimator

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def _log_gripper_action_definition(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    actions_cfg = getattr(env_cfg, "actions", None)
    gripper_cfg = getattr(actions_cfg, "gripper_action", None)
    if gripper_cfg is None:
        return
    open_expr = getattr(gripper_cfg, "open_command_expr", {})
    close_expr = getattr(gripper_cfg, "close_command_expr", {})
    open_value = open_expr.get("gripper", open_expr) if isinstance(open_expr, dict) else open_expr
    close_value = close_expr.get("gripper", close_expr) if isinstance(close_expr, dict) else close_expr
    print(
        "[INFO] Gripper action definition: BinaryJointPositionAction uses positive policy action "
        "for OPEN and non-positive policy action for CLOSE; "
        f"gripper joint command {close_value}=CLOSED, {open_value}=OPEN. "
        "GT debug gripper_open ratio is 0.0=CLOSED and 1.0=OPEN."
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    _log_gripper_action_definition(env_cfg)

    if args_cli.disturbance_type != "off" and args_cli.disturbance_level != "off":
        env_cfg = apply_pick_place_disturbance(
            env_cfg,
            disturbance_type=args_cli.disturbance_type,
            level=args_cli.disturbance_level,
        )
        print(
            f"[INFO] Applied disturbance: type={args_cli.disturbance_type}, level={args_cli.disturbance_level}"
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if "Vision-Play" in task_name:
        scene = env.unwrapped.scene
        fixed_rgb = scene["fixed_camera"].data.output["rgb"]
        handeye_rgb = scene["handeye_camera"].data.output["rgb"]
        print(f"[INFO] fixed_camera rgb shape: {tuple(fixed_rgb.shape)}")
        print(f"[INFO] handeye_camera rgb shape: {tuple(handeye_rgb.shape)}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    policy_group_name = "policy"
    obs_term_dims = base_env.observation_manager.group_obs_term_dim[policy_group_name]
    if hasattr(base_env.observation_manager, "active_terms"):
        obs_term_names = base_env.observation_manager.active_terms[policy_group_name]
    elif hasattr(base_env.observation_manager, "_group_obs_term_names"):
        obs_term_names = base_env.observation_manager._group_obs_term_names[policy_group_name]
    else:
        raise RuntimeError("Observation manager does not expose active_terms or _group_obs_term_names.")
    obs_offsets = {}
    start_idx = 0
    for term_name, term_dim in zip(obs_term_names, obs_term_dims):
        end_idx = start_idx + int(term_dim[0])
        obs_offsets[term_name] = (start_idx, end_idx)
        start_idx = end_idx
    if "object_position" not in obs_offsets:
        raise RuntimeError("Could not find 'object_position' term in policy observation.")

    object_slice = obs_offsets["object_position"]
    object_asset = base_env.scene["object"]
    robot_asset = base_env.scene["robot"]
    fixed_camera = base_env.scene["fixed_camera"] if "Vision-Play" in task_name else None
    resnet_estimator = None
    if args_cli.object_pose_source == "resnet":
        if fixed_camera is None:
            print("[WARNING] object_pose_source=resnet but fixed_camera unavailable; fallback to gt.")
        else:
            try:
                resnet_estimator = ResnetCubePoseEstimator(args_cli.resnet_model_path, device=env.unwrapped.device)
                print(f"[INFO] Loaded ResNet18 cube pose model: {args_cli.resnet_model_path}")
            except Exception as exc:
                print(f"[WARNING] Failed loading ResNet18 model: {exc}; fallback to gt.")
    last_valid_object_pos = None
    debug_interval = 50

    # Analytic pixel->robot mapping for the colour-mask vision mode: known fixed-camera extrinsics
    # and pinhole intrinsics, ray intersected with the cube-centre table plane. Replaces the old
    # whole-image linear x_range/y_range interpolation, which ignored perspective (several cm of
    # error across the oblique view). Camera params are read from the env cfg so cfg tweaks stay in
    # sync. Sanity-checked offline: the optical axis lands at robot-frame (0.408, -0.015), centred
    # in the cube workspace.
    vision_cam_pos = np.array([0.85, -0.90, 0.90])
    vision_cam_quat = (0.9009, 0.3898, 0.1213, 0.1472)  # (w,x,y,z), opengl convention
    vision_cam_focal = 18.0
    vision_cam_aperture = 20.955
    fixed_cam_cfg = getattr(getattr(base_env.cfg, "scene", None), "fixed_camera", None)
    if fixed_cam_cfg is not None:
        vision_cam_pos = np.array(fixed_cam_cfg.offset.pos, dtype=np.float64)
        vision_cam_quat = tuple(fixed_cam_cfg.offset.rot)
        vision_cam_focal = float(fixed_cam_cfg.spawn.focal_length)
        vision_cam_aperture = float(fixed_cam_cfg.spawn.horizontal_aperture)
    _qw, _qx, _qy, _qz = vision_cam_quat
    vision_cam_R = np.array([
        [1 - 2 * (_qy * _qy + _qz * _qz), 2 * (_qx * _qy - _qw * _qz), 2 * (_qx * _qz + _qw * _qy)],
        [2 * (_qx * _qy + _qw * _qz), 1 - 2 * (_qx * _qx + _qz * _qz), 2 * (_qy * _qz - _qw * _qx)],
        [2 * (_qx * _qz - _qw * _qy), 2 * (_qy * _qz + _qw * _qx), 1 - 2 * (_qx * _qx + _qy * _qy)],
    ])
    vision_cube_plane_z = 0.015  # cube-centre height at rest (robot root frame)
    camera_debug_dir = os.path.abspath(args_cli.camera_debug_dir)
    if args_cli.save_camera_debug:
        os.makedirs(camera_debug_dir, exist_ok=True)
        print(f"[INFO] Camera debug images will be saved to: {camera_debug_dir}")
    hard_sample_dir = os.path.abspath(args_cli.hard_sample_dir)
    save_resnet_hard_samples = args_cli.object_pose_source == "resnet" and args_cli.save_resnet_hard_samples
    if save_resnet_hard_samples:
        os.makedirs(hard_sample_dir, exist_ok=True)
        print(f"[INFO] ResNet hard samples will be saved to: {hard_sample_dir}")

    for manager_name in ("command_manager", "action_manager"):
        manager = getattr(base_env, manager_name, None)
        if manager is not None and hasattr(manager, "set_debug_vis"):
            manager.set_debug_vis(False)
            print(f"[INFO] Disabled {manager_name} debug visualization for cleaner camera images.")

    def _extract_rgb_uint8(camera_rgb: torch.Tensor) -> np.ndarray:
        frame = camera_rgb[0, ..., :3].detach().cpu().numpy()
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        return frame

    # The red recoloured cube spans only ~25-35 px in the 256x256 fixed view (~5-6 px across), so
    # the old threshold of 50 would reject EVERY genuine detection and silently fall back to GT.
    min_cube_pixel_area = 10

    def _raw_red_mask(rgb: np.ndarray) -> np.ndarray:
        r = rgb[..., 0].astype(np.int16)
        g = rgb[..., 1].astype(np.int16)
        b = rgb[..., 2].astype(np.int16)
        return (r > 110) & (r > g + 35) & (r > b + 35)

    def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        best_component = np.zeros_like(mask, dtype=bool)
        best_area = 0

        for y in range(height):
            for x in range(width):
                if not mask[y, x] or visited[y, x]:
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                component_pixels = []
                while stack:
                    cy, cx = stack.pop()
                    component_pixels.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                if len(component_pixels) > best_area:
                    best_area = len(component_pixels)
                    best_component.fill(False)
                    ys, xs = zip(*component_pixels)
                    best_component[np.array(ys), np.array(xs)] = True

        return best_component

    def _detect_red_cube(rgb: np.ndarray) -> tuple[bool, tuple[float, float], int, np.ndarray]:
        largest_component = _largest_connected_component(_raw_red_mask(rgb))
        area = int(largest_component.sum())
        if area < min_cube_pixel_area:
            return False, (0.0, 0.0), area, largest_component
        ys, xs = np.nonzero(largest_component)
        return True, (float(xs.mean()), float(ys.mean())), area, largest_component

    def _red_mask_u8(component_mask: np.ndarray) -> np.ndarray:
        return (component_mask.astype(np.uint8) * 255)

    def _save_vision_debug_images(
        step: int,
        fixed_rgb: np.ndarray,
        fixed_mask_u8: np.ndarray,
        handeye_rgb: np.ndarray | None,
        detected: bool,
        pixel_center: tuple[float, float],
        pixel_area: int,
        est_pos_xyz: list[float],
        gt_pos_xyz: list[float],
        vision_err_xy: float,
    ):
        step_tag = f"{step:06d}"
        Image.fromarray(fixed_rgb).save(os.path.join(camera_debug_dir, f"fixed_rgb_step_{step_tag}.png"))
        Image.fromarray(fixed_mask_u8, mode="L").save(os.path.join(camera_debug_dir, f"fixed_mask_step_{step_tag}.png"))

        Image.fromarray(fixed_rgb).resize((512, 512), Image.Resampling.NEAREST).save(
            os.path.join(camera_debug_dir, f"fixed_rgb_step_{step_tag}_x4.png")
        )
        Image.fromarray(fixed_mask_u8, mode="L").resize((512, 512), Image.Resampling.NEAREST).save(
            os.path.join(camera_debug_dir, f"fixed_mask_step_{step_tag}_x4.png")
        )

        overlay = Image.fromarray(fixed_rgb.copy())
        draw = ImageDraw.Draw(overlay)
        mask_pixels = np.argwhere(fixed_mask_u8 > 0)
        if mask_pixels.shape[0] > 0:
            y_min, x_min = mask_pixels.min(axis=0)
            y_max, x_max = mask_pixels.max(axis=0)
            draw.rectangle([(int(x_min), int(y_min)), (int(x_max), int(y_max))], outline=(255, 0, 0), width=2)
        cx, cy = int(round(pixel_center[0])), int(round(pixel_center[1]))
        draw.ellipse([(cx - 3, cy - 3), (cx + 3, cy + 3)], fill=(255, 255, 0), outline=(0, 0, 0), width=1)

        text_lines = [
            f"detected={detected}",
            f"cube_pixel_center=({pixel_center[0]:.1f},{pixel_center[1]:.1f})",
            f"cube_pixel_area={pixel_area}",
            f"estimated_object_position={est_pos_xyz}",
            f"gt_object_position={gt_pos_xyz}",
            f"vision_error_xy={vision_err_xy:.4f}",
        ]
        panel_width = 380
        overlay_with_panel = Image.new("RGB", (overlay.width + panel_width, overlay.height), (0, 0, 0))
        overlay_with_panel.paste(overlay, (0, 0))
        panel_draw = ImageDraw.Draw(overlay_with_panel)
        text_y = 8
        for line in text_lines:
            panel_draw.text((overlay.width + 8, text_y), line, fill=(255, 255, 255))
            text_y += 16
        overlay_with_panel.save(os.path.join(camera_debug_dir, f"fixed_overlay_step_{step_tag}.png"))
        overlay_with_panel.resize((512 + panel_width, 512), Image.Resampling.NEAREST).save(
            os.path.join(camera_debug_dir, f"fixed_overlay_step_{step_tag}_x4.png")
        )

        if handeye_rgb is not None:
            Image.fromarray(handeye_rgb).save(os.path.join(camera_debug_dir, f"handeye_rgb_step_{step_tag}.png"))

    def _pixel_to_robot_xy(
        center_xy: tuple[float, float], width: int, height: int, plane_z: float | None = None
    ) -> tuple[float, float]:
        # Pinhole back-projection: pixel -> OpenGL camera ray (+X right, +Y up, looking along -Z),
        # rotate into the robot frame, intersect with the horizontal plane at the cube-centre height
        # (defaults to the at-rest height; pass the live height while the cube is carried).
        px, py = center_xy
        fx = width * vision_cam_focal / vision_cam_aperture
        cx_ = (width - 1) / 2.0
        cy_ = (height - 1) / 2.0
        d_cam = np.array([(px - cx_) / fx, -(py - cy_) / fx, -1.0])
        d_robot = vision_cam_R @ d_cam
        if abs(d_robot[2]) < 1e-6:
            return float(vision_cam_pos[0]), float(vision_cam_pos[1])
        z_plane = vision_cube_plane_z if plane_z is None else plane_z
        t = (z_plane - vision_cam_pos[2]) / d_robot[2]
        hit = vision_cam_pos + t * d_robot
        return float(hit[0]), float(hit[1])

    timestep = 0
    num_envs = env.unwrapped.num_envs
    # Skip the first few post-reset frames before trusting the ResNet estimate: the RTX render is not
    # converged there and the estimate is far off (measured raw XY error ~14cm at step 0, ~10cm at
    # step 1, settling to ~6cm by step ~5). The cube is static, so we average the estimate over the
    # settled window [settle, warmup) and then freeze it (after that the reaching arm starts to
    # occlude the cube). Averaging the bad early frames was the bug behind the cached 12.5cm error.
    resnet_settle_steps = 5
    resnet_warmup_steps = 12
    resnet_fixed_z = 0.012
    cached_resnet_object_pos = torch.zeros((num_envs, 3), device=env.unwrapped.device, dtype=torch.float32)
    resnet_sum_xy = torch.zeros((num_envs, 2), device=env.unwrapped.device, dtype=torch.float32)
    resnet_count = torch.zeros((num_envs,), device=env.unwrapped.device, dtype=torch.int64)
    prev_episode_steps = torch.full((num_envs,), -1, device=env.unwrapped.device, dtype=torch.int64)

    # Diagnostic GT-noise state: one fixed XY offset per env, resampled at each episode reset.
    gt_noise_xy = torch.zeros((num_envs, 2), device=env.unwrapped.device, dtype=torch.float32)
    gt_noise_prev_steps = torch.full((num_envs,), -1, device=env.unwrapped.device, dtype=torch.int64)

    # Colour-mask vision: LIVE tracking every step (median-of-3 smoothed). A freeze-after-settle
    # design was tried first and produced only 8.9% grasp (45 episodes) despite a ~4mm-accurate
    # initial estimate: the EE arrived on target (closest-approach offset median 2.1cm, unbiased)
    # but v9's closing fingers NUDGE the cube 2-6cm (31/42 failures pushed >2cm, median 3.0cm) and
    # the frozen obs kept pointing at the old spot, so the policy grasped air. In GT mode the obs
    # tracks the nudge and v9 chases it -- that behaviour needs live updates. Live tracking also
    # recovers drops during transport for free (no re-acquisition heuristics needed). The projection
    # plane height follows the cube's live height (same GT-z scope as the obs z slot this mode
    # already fills; real-robot port: gripper kinematics when held), which keeps the back-projection
    # valid while the cube is carried. Env-0 semantics: evaluate with --num_envs 1.
    # Smoothing window 1 = raw per-step detection. median-of-3 was tried first and its ~1-frame lag
    # is the prime suspect for the contact-phase error (cube moves ~1cm/step while being pushed by
    # the closing fingers; the lagged estimate trails the chase by 1-1.5cm, right at the 2cm->17%
    # sensitivity cliff). Static jitter is sub-mm anyway (median 0.2cm pre-contact), so smoothing
    # buys nothing where it is safe and costs accuracy exactly where it hurts.
    vision_smooth_window = 1
    vision_det_history: list[tuple[float, float]] = []
    vision_prev_ep0 = -1

    # The debug CSV logs ground-truth diagnostics (object/box positions from physics, lift,
    # obj_box_dist, success = obj_box_dist < 0.05). These are GT-derived and therefore valid for
    # ANY pose source -- the pose source only changes what the POLICY observes, not the GT outcome.
    # So enabling this for resnet/vision lets us measure the true grasp/transport/success of the
    # policy when it is driven by the estimated (resnet/vision) cube pose, which the funnel then
    # quantifies. (Previously this was restricted to gt only.)
    gt_debug_enabled = args_cli.save_gt_debug
    if args_cli.save_gt_debug and args_cli.object_pose_source != "gt":
        print(
            f"[INFO] --save_gt_debug with object_pose_source={args_cli.object_pose_source}: the CSV "
            "logs ground-truth diagnostics (success/obj_box_dist/lift from physics) while the policy "
            "is driven by the estimated pose, so the funnel measures the true outcome under that pose source."
        )
    gt_debug_dir = os.path.abspath(args_cli.gt_debug_dir)
    gt_debug_seed = int(env_cfg.seed if env_cfg.seed is not None else 0)
    gt_episode_indices = [0 for _ in range(num_envs)]
    gt_debug_files = [None for _ in range(num_envs)]
    gt_debug_writers = [None for _ in range(num_envs)]
    gt_debug_fieldnames = [
        "seed",
        "step",
        "reward",
        "object_position",
        "obs_object_position",
        "ee_position",
        "box_position",
        "ee_obj_dist",
        "obj_box_dist",
        "gripper_joint_pos",
        "gripper_open",
        "gripper_closed",
        "gripper_action",
        "action",
        "success",
        "done",
    ]
    if gt_debug_enabled:
        os.makedirs(gt_debug_dir, exist_ok=True)
        print(f"[INFO] GT debug CSV files will be saved to: {gt_debug_dir}")

    def _format_debug_vector(values: torch.Tensor | np.ndarray | list[float]) -> str:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().flatten().tolist()
        elif isinstance(values, np.ndarray):
            values = values.flatten().tolist()
        return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"

    def _gt_debug_csv_path(env_id: int, episode_index: int) -> str:
        seed_tag = f"{gt_debug_seed:03d}"
        if env_id == 0 and episode_index == 0:
            filename = f"gt_seed_{seed_tag}.csv"
        else:
            filename = f"gt_seed_{seed_tag}_env_{env_id:02d}_episode_{episode_index:03d}.csv"
        return os.path.join(gt_debug_dir, filename)

    def _open_gt_debug_csv(env_id: int) -> None:
        episode_index = gt_episode_indices[env_id]
        csv_path = _gt_debug_csv_path(env_id, episode_index)
        gt_debug_files[env_id] = open(csv_path, "w", newline="")
        gt_debug_writers[env_id] = csv.DictWriter(gt_debug_files[env_id], fieldnames=gt_debug_fieldnames)
        gt_debug_writers[env_id].writeheader()
        print(f"[INFO] Writing GT debug episode CSV: {csv_path}")

    def _close_gt_debug_csv(env_id: int) -> None:
        if gt_debug_files[env_id] is not None:
            gt_debug_files[env_id].flush()
            gt_debug_files[env_id].close()
            gt_debug_files[env_id] = None
            gt_debug_writers[env_id] = None

    def _get_ee_position_robot_frame() -> torch.Tensor:
        if "ee_frame" not in base_env.scene.keys():
            return torch.full((num_envs, 3), float("nan"), device=env.unwrapped.device)
        ee_frame = base_env.scene["ee_frame"]
        return ee_frame.data.target_pos_w[..., 0, :3] - robot_asset.data.root_pos_w[:, :3]

    def _get_box_position_robot_frame() -> torch.Tensor:
        command_manager = getattr(base_env, "command_manager", None)
        if command_manager is None:
            return torch.full((num_envs, 3), float("nan"), device=env.unwrapped.device)
        return command_manager.get_command("object_pose")[:, :3]

    def _get_gripper_open_ratio() -> torch.Tensor:
        try:
            gripper_idx = robot_asset.find_joints("gripper")[0][0]
            gripper_joint_pos = robot_asset.data.joint_pos[:, gripper_idx]
            return torch.clamp((gripper_joint_pos - 0.12) / (0.45 - 0.12), min=0.0, max=1.0)
        except Exception:
            return torch.full((num_envs,), float("nan"), device=env.unwrapped.device)

    def _get_gripper_joint_pos() -> torch.Tensor:
        """Raw gripper joint angle (radians); unlike gripper_open it does not saturate at 0.45."""
        try:
            gripper_idx = robot_asset.find_joints("gripper")[0][0]
            return robot_asset.data.joint_pos[:, gripper_idx]
        except Exception:
            return torch.full((num_envs,), float("nan"), device=env.unwrapped.device)

    def _collect_gt_debug_rows(actions: torch.Tensor) -> list[dict[str, object]]:
        object_position = object_asset.data.root_pos_w[:, :3] - robot_asset.data.root_pos_w[:, :3]
        ee_position = _get_ee_position_robot_frame()
        box_position = _get_box_position_robot_frame()
        ee_obj_dist = torch.norm(ee_position - object_position, dim=1)
        obj_box_dist = torch.norm(object_position - box_position, dim=1)
        gripper_open = _get_gripper_open_ratio()
        gripper_joint_pos = _get_gripper_joint_pos()
        gripper_closed = 1.0 - gripper_open
        gripper_action = actions[:, -1] if actions.ndim == 2 and actions.shape[1] > 0 else torch.full_like(gripper_open, float("nan"))
        episode_steps = base_env.episode_length_buf.to(torch.int64) + 1
        success = obj_box_dist < 0.05

        rows = []
        for env_id in range(num_envs):
            rows.append(
                {
                    "seed": gt_debug_seed,
                    "step": int(episode_steps[env_id].item()),
                    "reward": float("nan"),
                    "object_position": _format_debug_vector(object_position[env_id]),
                    "obs_object_position": _format_debug_vector(policy_obs_object[env_id]),
                    "ee_position": _format_debug_vector(ee_position[env_id]),
                    "box_position": _format_debug_vector(box_position[env_id]),
                    "ee_obj_dist": float(ee_obj_dist[env_id].item()),
                    "obj_box_dist": float(obj_box_dist[env_id].item()),
                    "gripper_joint_pos": float(gripper_joint_pos[env_id].item()),
                    "gripper_open": float(gripper_open[env_id].item()),
                    "gripper_closed": float(gripper_closed[env_id].item()),
                    "gripper_action": float(gripper_action[env_id].item()),
                    "action": _format_debug_vector(actions[env_id]),
                    "success": bool(success[env_id].item()),
                    "done": False,
                }
            )
        return rows

    def _write_gt_debug_rows(rows: list[dict[str, object]], rewards: torch.Tensor, dones: torch.Tensor) -> None:
        rewards = rewards.detach().flatten()
        dones = dones.detach().flatten().to(torch.bool)
        for env_id, row in enumerate(rows):
            if gt_debug_writers[env_id] is None:
                _open_gt_debug_csv(env_id)
            row["reward"] = float(rewards[env_id].item())
            row["done"] = bool(dones[env_id].item())
            gt_debug_writers[env_id].writerow(row)
            if env_id == 0 and int(row["step"]) % 50 == 0:
                print(
                    f"seed={row['seed']} "
                    f"step={row['step']} "
                    f"reward={row['reward']:.4f} "
                    f"ee_obj_dist={row['ee_obj_dist']:.4f} "
                    f"obj_box_dist={row['obj_box_dist']:.4f} "
                    f"gripper_joint_pos={row['gripper_joint_pos']:.4f} "
                    f"gripper_open={row['gripper_open']:.4f} "
                    f"gripper_closed={row['gripper_closed']:.4f} "
                    f"gripper_action={row['gripper_action']:.4f}"
                )
            if dones[env_id]:
                _close_gt_debug_csv(env_id)
                gt_episode_indices[env_id] += 1

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            gt_object_pos_world = object_asset.data.root_pos_w[:, :3]
            robot_root_world = robot_asset.data.root_pos_w[:, :3]
            gt_object_pos_robot = gt_object_pos_world - robot_root_world

            detected = False
            pixel_center = (0.0, 0.0)
            pixel_area = 0
            estimated_object_pos = gt_object_pos_robot.clone()
            raw_resnet_object_pos = torch.full_like(gt_object_pos_robot, float("nan"))
            fallback_last_valid = False

            # Diagnostic: inject a per-episode fixed XY offset into the observed object position
            # (GT mode only). Resample the offset on each episode reset so it stays constant within
            # an episode, then write GT+offset into the obs object slice.
            if args_cli.object_pose_source == "gt" and args_cli.gt_pose_noise_xy > 0.0:
                ep_steps = base_env.episode_length_buf.to(torch.int64)
                reset_mask = (ep_steps <= 1) & (gt_noise_prev_steps >= 2) & (ep_steps < gt_noise_prev_steps)
                first_mask = gt_noise_prev_steps < 0
                resample = reset_mask | first_mask
                if torch.any(resample):
                    gt_noise_xy[resample] = torch.randn((int(resample.sum()), 2), device=gt_noise_xy.device) * args_cli.gt_pose_noise_xy
                gt_noise_prev_steps = ep_steps.clone()
                noisy = gt_object_pos_robot.clone()
                noisy[:, :2] = noisy[:, :2] + gt_noise_xy
                obs[:, object_slice[0]:object_slice[1]] = noisy
            fallback_gt = False

            # Camera debug for non-vision pose sources (gt/resnet). The vision branch below saves its
            # own annotated overlays; this saves the RAW fixed/handeye frames so the camera FOV can be
            # inspected under any source (e.g. whether the cube enters the handeye view during the
            # approach). Filename carries the env-0 EE-to-object distance to locate near-grasp frames.
            if (
                args_cli.save_camera_debug
                and args_cli.object_pose_source != "vision"
                and (timestep % max(1, args_cli.camera_debug_interval) == 0)
            ):
                ee_obj_d = torch.norm(_get_ee_position_robot_frame()[0] - gt_object_pos_robot[0]).item()
                raw_step_tag = f"{timestep:06d}_d{ee_obj_d:.3f}"
                if "fixed_camera" in base_env.scene.keys():
                    raw_fixed = _extract_rgb_uint8(base_env.scene["fixed_camera"].data.output["rgb"])
                    Image.fromarray(raw_fixed).save(
                        os.path.join(camera_debug_dir, f"fixed_rgb_step_{raw_step_tag}.png")
                    )
                if "handeye_camera" in base_env.scene.keys():
                    raw_handeye = _extract_rgb_uint8(base_env.scene["handeye_camera"].data.output["rgb"])
                    Image.fromarray(raw_handeye).save(
                        os.path.join(camera_debug_dir, f"handeye_rgb_step_{raw_step_tag}.png")
                    )

            if args_cli.object_pose_source == "vision":
                fixed_rgb = None
                fixed_mask_u8 = None
                handeye_rgb = None
                if fixed_camera is None:
                    fallback_gt = True
                    print("[WARNING] object_pose_source=vision but fixed_camera is unavailable; fallback to gt.")
                else:
                    fixed_rgb = _extract_rgb_uint8(fixed_camera.data.output["rgb"])
                    detected, pixel_center, pixel_area, component_mask = _detect_red_cube(fixed_rgb)
                    fixed_mask_u8 = _red_mask_u8(component_mask)
                    ep0 = int(base_env.episode_length_buf[0].item())
                    if ep0 < vision_prev_ep0:
                        # episode reset: clear the smoothing history and the stale fallback
                        vision_det_history.clear()
                        last_valid_object_pos = None
                    vision_prev_ep0 = ep0
                    est_xy = None
                    if detected:
                        # Track the cube's live height so the back-projection stays valid while the
                        # cube is lifted/carried (at-rest plane otherwise).
                        cube_z_live = float(gt_object_pos_robot[0, 2].item())
                        plane_z = min(max(cube_z_live, 0.005), 0.40)
                        raw_xy = _pixel_to_robot_xy(
                            pixel_center, fixed_rgb.shape[1], fixed_rgb.shape[0], plane_z=plane_z
                        )
                        vision_det_history.append(raw_xy)
                        if len(vision_det_history) > vision_smooth_window:
                            vision_det_history.pop(0)
                        est_xy = (
                            float(np.median([p[0] for p in vision_det_history])),
                            float(np.median([p[1] for p in vision_det_history])),
                        )
                    else:
                        vision_det_history.clear()
                    if est_xy is not None:
                        estimated_object_pos[:, 0] = est_xy[0]
                        estimated_object_pos[:, 1] = est_xy[1]
                        last_valid_object_pos = estimated_object_pos.clone()
                    elif last_valid_object_pos is not None:
                        estimated_object_pos = last_valid_object_pos.clone()
                        fallback_last_valid = True
                        print(f"[WARNING] fixed_camera red-cube detection failed (area={pixel_area}); fallback to last valid vision estimate.")
                    else:
                        fallback_gt = True
                        print(
                            "[WARNING] fixed_camera red-cube detection failed (area "
                            f"{pixel_area} < {min_cube_pixel_area}) and no last valid estimate; fallback to gt object position."
                        )
                obs[:, object_slice[0]:object_slice[1]] = estimated_object_pos

                if args_cli.save_camera_debug and (timestep % max(1, args_cli.camera_debug_interval) == 0):
                    if "handeye_camera" in base_env.scene.keys():
                        handeye_rgb = _extract_rgb_uint8(base_env.scene["handeye_camera"].data.output["rgb"])
                    if fixed_rgb is not None and fixed_mask_u8 is not None:
                        _save_vision_debug_images(
                            step=timestep,
                            fixed_rgb=fixed_rgb,
                            fixed_mask_u8=fixed_mask_u8,
                            handeye_rgb=handeye_rgb,
                            detected=detected,
                            pixel_center=pixel_center,
                            pixel_area=pixel_area,
                            est_pos_xyz=estimated_object_pos[0].detach().cpu().tolist(),
                            gt_pos_xyz=gt_object_pos_robot[0].detach().cpu().tolist(),
                            vision_err_xy=torch.norm(
                                estimated_object_pos[0, :2] - gt_object_pos_robot[0, :2], dim=0
                            ).item(),
                        )
            elif args_cli.object_pose_source == "resnet":
                if fixed_camera is None or resnet_estimator is None:
                    fallback_gt = True
                else:
                    try:
                        episode_steps = base_env.episode_length_buf.to(torch.int64)
                        reset_env_mask = (
                            (episode_steps <= 1)
                            & (prev_episode_steps >= 2)
                            & (episode_steps < prev_episode_steps)
                        )
                        if torch.any(reset_env_mask):
                            resnet_sum_xy[reset_env_mask] = 0.0
                            resnet_count[reset_env_mask] = 0
                            cached_resnet_object_pos[reset_env_mask] = 0.0
                        prev_episode_steps = episode_steps.clone()

                        est_pos = resnet_estimator.estimate(fixed_camera.data.output["rgb"], gt_object_pos_robot[:, 2])
                        raw_resnet_object_pos = est_pos.clone()

                        # Accumulate only over the settled window [settle, warmup); the running
                        # average is frozen once episode_step reaches warmup.
                        warmup_mask = (episode_steps >= resnet_settle_steps) & (episode_steps < resnet_warmup_steps)
                        if torch.any(warmup_mask):
                            resnet_sum_xy[warmup_mask] += raw_resnet_object_pos[warmup_mask, :2]
                            resnet_count[warmup_mask] += 1
                            count_f = resnet_count[warmup_mask].unsqueeze(1).to(torch.float32)
                            cached_resnet_object_pos[warmup_mask, :2] = resnet_sum_xy[warmup_mask] / count_f
                            cached_resnet_object_pos[warmup_mask, 2] = resnet_fixed_z

                        # Before any settled frame has been averaged (the settle period, or as a
                        # fallback), use the current raw estimate as a TEMPORARY value but do NOT
                        # persist it into the running sum/count -- otherwise the bad early frames would
                        # pollute the settled average (the original bug).
                        no_cache_mask = resnet_count == 0
                        if torch.any(no_cache_mask):
                            cached_resnet_object_pos[no_cache_mask, :2] = raw_resnet_object_pos[no_cache_mask, :2]
                            cached_resnet_object_pos[no_cache_mask, 2] = resnet_fixed_z

                        estimated_object_pos = cached_resnet_object_pos.clone()
                        last_valid_object_pos = estimated_object_pos.clone()
                    except Exception as exc:
                        if last_valid_object_pos is not None:
                            estimated_object_pos = last_valid_object_pos.clone()
                            fallback_last_valid = True
                            print(f"[WARNING] ResNet18 inference failed, fallback to last valid estimate: {exc}")
                        else:
                            fallback_gt = True
                            print(f"[WARNING] ResNet18 inference failed, fallback to gt: {exc}")
                obs[:, object_slice[0]:object_slice[1]] = estimated_object_pos

            vision_err = torch.norm(estimated_object_pos[:, :2] - gt_object_pos_robot[:, :2], dim=1).mean().item()

            if save_resnet_hard_samples and fixed_camera is not None and resnet_estimator is not None:
                episode_step_debug = int(base_env.episode_length_buf[0].item())
                vision_error_xy_env0 = torch.norm(
                    estimated_object_pos[0, :2] - gt_object_pos_robot[0, :2], dim=0
                ).item()
                if vision_error_xy_env0 > args_cli.hard_sample_error_threshold:
                    fixed_rgb_np = _extract_rgb_uint8(fixed_camera.data.output["rgb"])
                    handeye_rgb_np = None
                    if "handeye_camera" in base_env.scene.keys():
                        handeye_rgb_np = _extract_rgb_uint8(base_env.scene["handeye_camera"].data.output["rgb"])
                    if handeye_rgb_np is None:
                        handeye_rgb_np = np.zeros_like(fixed_rgb_np)

                    hard_sample_filename = f"hard_sample_step_{timestep:06d}_error_{vision_error_xy_env0:.3f}.npz"
                    hard_sample_path = os.path.join(hard_sample_dir, hard_sample_filename)
                    np.savez_compressed(
                        hard_sample_path,
                        fixed_rgb=fixed_rgb_np,
                        handeye_rgb=handeye_rgb_np,
                        object_position=gt_object_pos_robot[0].detach().cpu().numpy().astype(np.float32),
                        raw_resnet_object_position=raw_resnet_object_pos[0].detach().cpu().numpy().astype(np.float32),
                        cached_resnet_object_position=estimated_object_pos[0].detach().cpu().numpy().astype(np.float32),
                        vision_error_xy=np.array(vision_error_xy_env0, dtype=np.float32),
                        step=np.array(timestep, dtype=np.int64),
                        episode_step=np.array(episode_step_debug, dtype=np.int64),
                    )

            if timestep % debug_interval == 0:
                episode_step_debug = int(base_env.episode_length_buf[0].item())
                cache_reset_debug = False
                resnet_count_debug = 0
                raw_resnet_pos_debug = (
                    raw_resnet_object_pos[0].tolist()
                    if args_cli.object_pose_source == "resnet"
                    else [float("nan"), float("nan"), float("nan")]
                )
                cached_resnet_pos_debug = (
                    cached_resnet_object_pos[0].tolist()
                    if args_cli.object_pose_source == "resnet"
                    else [float("nan"), float("nan"), float("nan")]
                )
                if args_cli.object_pose_source == "resnet":
                    cache_reset_debug = bool(reset_env_mask[0].item()) if "reset_env_mask" in locals() else False
                    resnet_count_debug = int(resnet_count[0].item())
                print(
                    "[DEBUG] "
                    f"step={timestep} "
                    f"episode_step={episode_step_debug} "
                    f"cache_reset={cache_reset_debug} "
                    f"object_pose_source={args_cli.object_pose_source} "
                    f"fixed_camera_detected={detected} "
                    f"cube_pixel_center=({pixel_center[0]:.1f},{pixel_center[1]:.1f}) "
                    f"cube_pixel_area={pixel_area} "
                    f"estimated_object_position={estimated_object_pos[0].tolist()} "
                    f"raw_resnet_object_position={raw_resnet_pos_debug} "
                    f"resnet_count={resnet_count_debug} "
                    f"cached_resnet_object_position={cached_resnet_pos_debug} "
                    f"gt_object_position={gt_object_pos_robot[0].tolist()} "
                    f"vision_error_xy={vision_err:.4f} "
                    f"fallback_last_valid={fallback_last_valid} "
                    f"fallback_gt={fallback_gt}"
                )
            # agent stepping
            # Snapshot the object slice the policy ACTUALLY sees this step (GT, GT+noise, vision or
            # resnet estimate) so the debug CSV can quantify estimate-vs-GT error offline. obs is a
            # TensorDict here: writing obs[:, a:b] = ... broadcasts INTO the "policy" key (which is
            # why the injections above work) but READING obs[:, a:b] raises IndexError on the 1-dim
            # batch, so the read must go through the "policy" entry explicitly.
            obs_policy_tensor = obs["policy"] if hasattr(obs, "keys") and "policy" in obs.keys() else obs
            policy_obs_object = obs_policy_tensor[:, object_slice[0]:object_slice[1]].clone()
            actions = policy(obs)
            gt_debug_rows = _collect_gt_debug_rows(actions) if gt_debug_enabled else None
            # env stepping
            obs, rewards, dones, _ = env.step(actions)
            if gt_debug_enabled:
                _write_gt_debug_rows(gt_debug_rows, rewards, dones)
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if gt_debug_enabled:
        for env_id in range(num_envs):
            _close_gt_debug_csv(env_id)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
