# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args # isort: skip
from isaac_so_arm101.tasks.pick_place.robust_eval_cfg import apply_pick_place_disturbance # isort: skip

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
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

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
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


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
    obs_term_names = base_env.observation_manager.group_obs_term_names[policy_group_name]
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
    x_range = (-0.15, 0.20)
    y_range = (-0.30, 0.30)
    camera_debug_dir = os.path.abspath(args_cli.camera_debug_dir)
    if args_cli.save_camera_debug:
        os.makedirs(camera_debug_dir, exist_ok=True)
        print(f"[INFO] Camera debug images will be saved to: {camera_debug_dir}")

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

    min_cube_pixel_area = 50

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

    def _pixel_to_robot_xy(center_xy: tuple[float, float], width: int, height: int) -> tuple[float, float]:
        px, py = center_xy
        nx = px / max(width - 1, 1)
        ny = py / max(height - 1, 1)
        est_x = x_range[0] + nx * (x_range[1] - x_range[0])
        est_y = y_range[1] - ny * (y_range[1] - y_range[0])
        return est_x, est_y

    timestep = 0
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
            fallback_last_valid = False
            fallback_gt = False

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
                    if detected:
                        est_x, est_y = _pixel_to_robot_xy(pixel_center, fixed_rgb.shape[1], fixed_rgb.shape[0])
                        estimated_object_pos[:, 0] = est_x
                        estimated_object_pos[:, 1] = est_y
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
                        est_pos = resnet_estimator.estimate(
                            fixed_camera.data.output["rgb"],
                            gt_object_pos_robot[:, 2],
                        )
                        estimated_object_pos = est_pos
                        last_valid_object_pos = est_pos.clone()
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
            if timestep % debug_interval == 0:
                print(
                    "[DEBUG] "
                    f"step={timestep} "
                    f"object_pose_source={args_cli.object_pose_source} "
                    f"fixed_camera_detected={detected} "
                    f"cube_pixel_center=({pixel_center[0]:.1f},{pixel_center[1]:.1f}) "
                    f"cube_pixel_area={pixel_area} "
                    f"estimated_object_position={estimated_object_pos[0].tolist()} "
                    f"gt_object_position={gt_object_pos_robot[0].tolist()} "
                    f"vision_error_xy={vision_err:.4f} "
                    f"fallback_last_valid={fallback_last_valid} "
                    f"fallback_gt={fallback_gt}"
                )
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
