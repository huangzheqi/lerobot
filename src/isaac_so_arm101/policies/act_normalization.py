"""Normalization stats loader for legacy LeRobot ACT checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class ActNormalizationStats:
    """Mean/std tensors used by the ACT checkpoint's old built-in normalizers."""

    state_mean: torch.Tensor
    state_std: torch.Tensor
    fixed_image_mean: torch.Tensor
    fixed_image_std: torch.Tensor
    handeye_image_mean: torch.Tensor
    handeye_image_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor


def _read_tensor(tensors, key: str) -> torch.Tensor:
    if key not in tensors.keys():
        raise KeyError(f"ACT checkpoint is missing normalization tensor {key!r}")
    return tensors.get_tensor(key).float()


def load_act_normalization_stats(model_dir: str | Path) -> ActNormalizationStats:
    """Load normalization buffers embedded in an old LeRobot ACT `model.safetensors` file."""

    model_path = Path(model_dir) / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"ACT model weights not found: {model_path}")

    with safe_open(model_path, framework="pt", device="cpu") as tensors:
        return ActNormalizationStats(
            state_mean=_read_tensor(tensors, "normalize_inputs.buffer_observation_state.mean"),
            state_std=_read_tensor(tensors, "normalize_inputs.buffer_observation_state.std"),
            fixed_image_mean=_read_tensor(tensors, "normalize_inputs.buffer_observation_images_fixed.mean"),
            fixed_image_std=_read_tensor(tensors, "normalize_inputs.buffer_observation_images_fixed.std"),
            handeye_image_mean=_read_tensor(tensors, "normalize_inputs.buffer_observation_images_handeye.mean"),
            handeye_image_std=_read_tensor(tensors, "normalize_inputs.buffer_observation_images_handeye.std"),
            action_mean=_read_tensor(tensors, "unnormalize_outputs.buffer_action.mean"),
            action_std=_read_tensor(tensors, "unnormalize_outputs.buffer_action.std"),
        )
