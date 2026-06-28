"""Collect a fixed_camera cube-pose dataset for ResNet18 training.

IMPORTANT (Isaac Lab import-order constraint): the Omniverse app must be launched via AppLauncher
BEFORE importing anything that pulls in isaaclab (gymnasium task registration, isaac_so_arm101.tasks,
the scene/camera assets). So argparse + AppLauncher live at module top, and every heavy import is
deferred until after `simulation_app` exists -- mirroring play.py / train.py.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# --- CLI (parsed before launching the app) ---
parser = argparse.ArgumentParser(description="Collect fixed_camera cube-pose dataset.")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Pick-Place-Cube-Vision-Play-v0")
parser.add_argument("--num_samples", type=int, default=5000)
parser.add_argument("--output_dir", type=str, default="data/cube_pose_dataset")
# Number of parallel envs to render. Lower = faster collection and less neighbour clutter in the
# fixed_camera frame; None falls back to the task cfg default. Passed to parse_env_cfg below.
parser.add_argument("--num_envs", type=int, default=4)
# AppLauncher contributes --enable_cameras / --headless / --device etc. (do NOT add --enable_cameras
# here too -- it would duplicate AppLauncher's option string).
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Rendering the dataset images requires cameras, headless.
args_cli.enable_cameras = True
args_cli.headless = True

# --- launch Omniverse app BEFORE importing isaaclab / gym / tasks ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- now it is safe to import everything that touches isaaclab ---
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import isaac_so_arm101.tasks  # noqa: F401


def _to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    rgb = t[0, ..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def main():
    # Parse the env cfg (overriding num_envs) and pass it into gym.make so the camera/scene actually
    # match the deployment env -- without cfg= the env falls back to defaults.
    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base_env = env.unwrapped
    scene = base_env.scene
    fixed_camera = scene["fixed_camera"]
    handeye_camera = scene["handeye_camera"] if "handeye_camera" in scene.keys() else None
    object_asset = scene["object"]
    robot_asset = scene["robot"]

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ManagerBasedRLEnv.step calls action.to(device), so the action must be a torch tensor on the
    # env device (not a numpy array). Shape is constant across steps, so build it once.
    zero_action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=env.unwrapped.device)
    obs, _ = env.reset()
    for idx in range(args_cli.num_samples):
        env.reset()
        env.step(zero_action)

        fixed_rgb = _to_uint8_rgb(fixed_camera.data.output["rgb"])
        handeye_rgb = _to_uint8_rgb(handeye_camera.data.output["rgb"]) if handeye_camera is not None else np.zeros((1, 1, 3), dtype=np.uint8)

        obj_pos = object_asset.data.root_pos_w[:, :3] - robot_asset.data.root_pos_w[:, :3]
        obj_pos_np = obj_pos[0].detach().cpu().numpy().astype(np.float32)

        np.savez_compressed(
            out_dir / f"sample_{idx:06d}.npz",
            fixed_rgb=fixed_rgb,
            handeye_rgb=handeye_rgb,
            object_position=obj_pos_np,
        )
        if idx % 200 == 0:
            print(f"collected {idx}/{args_cli.num_samples}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
