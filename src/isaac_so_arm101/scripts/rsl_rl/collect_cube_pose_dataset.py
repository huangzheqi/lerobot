from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import AppLauncher

import isaac_so_arm101.tasks  # noqa: F401


def _to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    rgb = t[0, ..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def main():
    parser = argparse.ArgumentParser(description="Collect fixed_camera cube-pose dataset.")
    parser.add_argument("--task", type=str, default="Isaac-SO101-PickPlace-Cube-Vision-Play-v0")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="data/cube_pose_dataset")
    parser.add_argument("--enable_cameras", action="store_true", default=True)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    args_cli.headless = True
    args_cli.enable_cameras = True if args_cli.enable_cameras else False
    app = AppLauncher(args_cli).app

    env = gym.make(args_cli.task, render_mode=None)
    base_env = env.unwrapped
    scene = base_env.scene
    fixed_camera = scene["fixed_camera"]
    handeye_camera = scene["handeye_camera"] if "handeye_camera" in scene.keys() else None
    object_asset = scene["object"]
    robot_asset = scene["robot"]

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset()
    for idx in range(args_cli.num_samples):
        env.reset()
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))

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
    app.close()


if __name__ == "__main__":
    main()
