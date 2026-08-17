#!/usr/bin/env python3
"""Offline ACT replay check against a LeRobot dataset episode.

This does not start Isaac Sim. It feeds recorded dataset state/images into the
ACT checkpoint and compares the predicted first action with the recorded action.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from isaac_so_arm101.policies.act_actor_critic import _load_lerobot_act_policy
from isaac_so_arm101.policies.act_contract import (
    ACT_ACTION_KEY,
    ACT_FIXED_IMAGE_KEY,
    ACT_HANDEYE_IMAGE_KEY,
    ACT_JOINT_NAMES,
    ACT_STATE_KEY,
    ACT_V5_MODEL_DIR,
)
from isaac_so_arm101.policies.act_normalization import load_act_normalization_stats

DEFAULT_ACT_DATASET_DIR = Path(
    "../so101_clean_project/data/so101_pick_place_6dof_clean_v5_factory_zero"
)
DEFAULT_OUTPUT_DIR = Path("logs/act_debug")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay recorded SO101 ACT dataset frames through the checkpoint and compare actions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_ACT_DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=ACT_V5_MODEL_DIR)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--use-action-queue",
        action="store_true",
        help="Do not reset ACT between frames. This approximates native sequential ACT chunk execution.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _load_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset info not found: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _episode_paths(dataset_dir: Path, info: dict, episode_index: int) -> tuple[Path, Path, Path]:
    chunk = _episode_chunk(episode_index, int(info.get("chunks_size", 1000)))
    data_path = dataset_dir / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    fixed_path = dataset_dir / info["video_path"].format(
        episode_chunk=chunk, episode_index=episode_index, video_key=ACT_FIXED_IMAGE_KEY
    )
    handeye_path = dataset_dir / info["video_path"].format(
        episode_chunk=chunk, episode_index=episode_index, video_key=ACT_HANDEYE_IMAGE_KEY
    )
    return data_path, fixed_path, handeye_path


def _read_video_frames(video_path: Path, frame_indices: list[int]) -> dict[int, torch.Tensor]:
    wanted = set(frame_indices)
    frames: dict[int, torch.Tensor] = {}
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with av.open(str(video_path)) as container:
        for decoded_index, frame in enumerate(container.decode(video=0)):
            if decoded_index in wanted:
                array = frame.to_ndarray(format="rgb24")
                tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
                frames[decoded_index] = tensor.contiguous()
                if len(frames) == len(wanted):
                    break

    missing = sorted(wanted - set(frames))
    if missing:
        raise IndexError(f"Could not decode frames {missing} from {video_path}")
    return frames


def _select_rows(df: pd.DataFrame, frame_index: int, num_frames: int, stride: int) -> pd.DataFrame:
    selected = []
    for offset in range(num_frames):
        wanted_frame = frame_index + offset * stride
        row = df.loc[df["frame_index"] == wanted_frame]
        if row.empty:
            break
        selected.append(row.iloc[0])
    if not selected:
        raise IndexError(f"No rows selected from frame_index={frame_index}, num_frames={num_frames}, stride={stride}")
    return pd.DataFrame(selected)


def _prepare_policy_batch(
    states: torch.Tensor,
    fixed: torch.Tensor,
    handeye: torch.Tensor,
    model_dir: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    stats = load_act_normalization_stats(model_dir)
    state_mean = stats.state_mean.to(device)
    state_std = stats.state_std.to(device)
    fixed_mean = stats.fixed_image_mean.to(device)
    fixed_std = stats.fixed_image_std.to(device)
    handeye_mean = stats.handeye_image_mean.to(device)
    handeye_std = stats.handeye_image_std.to(device)

    return {
        ACT_STATE_KEY: (states.to(device) - state_mean) / (state_std + 1.0e-8),
        ACT_FIXED_IMAGE_KEY: (fixed.to(device) - fixed_mean) / (fixed_std + 1.0e-8),
        ACT_HANDEYE_IMAGE_KEY: (handeye.to(device) - handeye_mean) / (handeye_std + 1.0e-8),
    }


def _predict_actions(
    policy: torch.nn.Module,
    policy_batch: dict[str, torch.Tensor],
    model_dir: Path,
    use_action_queue: bool,
) -> torch.Tensor:
    stats = load_act_normalization_stats(model_dir)
    action_mean = stats.action_mean.to(policy_batch[ACT_STATE_KEY].device)
    action_std = stats.action_std.to(policy_batch[ACT_STATE_KEY].device)

    predictions = []
    if hasattr(policy, "reset"):
        policy.reset()
    with torch.no_grad():
        for row_idx in range(policy_batch[ACT_STATE_KEY].shape[0]):
            if not use_action_queue and hasattr(policy, "reset"):
                policy.reset()
            single_batch = {key: value[row_idx : row_idx + 1] for key, value in policy_batch.items()}
            action = policy.select_action(single_batch)
            if isinstance(action, dict):
                action = action[ACT_ACTION_KEY]
            if action.ndim == 3:
                action = action[:, 0, :]
            predictions.append(action.squeeze(0))
    normalized = torch.stack(predictions, dim=0)
    return normalized * action_std + action_mean


def _write_csv(output_path: Path, frame_indices: list[int], targets: torch.Tensor, predictions: torch.Tensor) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_index"]
    for name in ACT_JOINT_NAMES:
        fieldnames.extend([f"target_{name}", f"pred_{name}", f"error_{name}"])
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index, target, pred in zip(frame_indices, targets, predictions, strict=True):
            row: dict[str, float | int] = {"frame_index": int(frame_index)}
            for joint_idx, name in enumerate(ACT_JOINT_NAMES):
                row[f"target_{name}"] = float(target[joint_idx])
                row[f"pred_{name}"] = float(pred[joint_idx])
                row[f"error_{name}"] = float(pred[joint_idx] - target[joint_idx])
            writer.writerow(row)


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if args.use_action_queue and args.stride != 1:
        print("[WARN] --use-action-queue is only temporally faithful with --stride 1")

    info = _load_info(args.dataset_dir)
    data_path, fixed_path, handeye_path = _episode_paths(args.dataset_dir, info, args.episode_index)
    if not data_path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {data_path}")

    df = pd.read_parquet(data_path)
    selected = _select_rows(df, args.frame_index, args.num_frames, args.stride)
    frame_indices = [int(value) for value in selected["frame_index"].tolist()]

    fixed_frames = _read_video_frames(fixed_path, frame_indices)
    handeye_frames = _read_video_frames(handeye_path, frame_indices)
    states = torch.as_tensor(np.stack(selected[ACT_STATE_KEY].to_numpy()), dtype=torch.float32)
    targets = torch.as_tensor(np.stack(selected[ACT_ACTION_KEY].to_numpy()), dtype=torch.float32)
    fixed = torch.stack([fixed_frames[index] for index in frame_indices], dim=0)
    handeye = torch.stack([handeye_frames[index] for index in frame_indices], dim=0)

    device = torch.device(args.device)
    policy = _load_lerobot_act_policy(args.model_dir, device=args.device)
    policy.eval()
    policy.to(device)

    policy_batch = _prepare_policy_batch(states, fixed, handeye, args.model_dir, device)
    predictions = _predict_actions(policy, policy_batch, args.model_dir, args.use_action_queue).cpu()

    errors = predictions - targets
    mae = errors.abs().mean(dim=0)
    rmse = torch.sqrt((errors * errors).mean(dim=0))
    mode = "native_queue" if args.use_action_queue else "reset_each_frame_first_action"
    output_path = args.output_dir / f"offline_act_episode_{args.episode_index:06d}_{mode}.csv"
    _write_csv(output_path, frame_indices, targets, predictions)

    print(f"Dataset: {args.dataset_dir}")
    print(f"Model: {args.model_dir}")
    print(f"Episode: {args.episode_index}, frames: {frame_indices}")
    print(f"Mode: {mode}")
    print(f"CSV: {output_path}")
    print("Per-joint error in ACT action units, usually degrees / gripper position:")
    for idx, name in enumerate(ACT_JOINT_NAMES):
        print(f"  {name:14s} MAE={mae[idx].item():8.4f}  RMSE={rmse[idx].item():8.4f}")
    print(f"Overall MAE={errors.abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
