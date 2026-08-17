"""ACT model input/output contract used by the SO101 pick-place adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ACT_MODEL_DIR = Path("data/act_model/train_act_6dof_clean_v4/checkpoints/100000/pretrained_model")
ACT_V5_MODEL_DIR = Path(
    "data/act_model/train_act_6dof_clean_v5_factory_zero/checkpoints/100000/pretrained_model"
)
ACT_RED_CUBE_TARGET_MODEL_DIR = Path(
    "data/act_model/train_act_red_cube_to_target_88_clean_terminal_1s/"
    "checkpoints/100000/pretrained_model"
)

ACT_STATE_KEY = "observation.state"
ACT_FIXED_IMAGE_KEY = "observation.images.fixed"
ACT_HANDEYE_IMAGE_KEY = "observation.images.handeye"
ACT_ACTION_KEY = "action"

ACT_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

EXPECTED_ACT_STATE_SHAPE = (6,)
EXPECTED_ACT_IMAGE_SHAPE = (3, 480, 640)
EXPECTED_ACT_ACTION_SHAPE = (6,)
EXPECTED_ACT_CHUNK_SIZE = 100
EXPECTED_ACT_N_ACTION_STEPS = 100
EXPECTED_ACT_VISION_BACKBONE = "resnet18"


@dataclass(frozen=True)
class ActModelContract:
    """Minimal ACT checkpoint contract needed before wiring it into PPO."""

    model_dir: Path
    state_shape: tuple[int, ...]
    fixed_image_shape: tuple[int, ...]
    handeye_image_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    chunk_size: int
    n_action_steps: int
    vision_backbone: str


def _feature_shape(features: dict, key: str) -> tuple[int, ...]:
    if key not in features:
        raise KeyError(f"ACT config is missing feature {key!r}")
    shape = features[key].get("shape")
    if shape is None:
        raise KeyError(f"ACT config feature {key!r} is missing a shape")
    return tuple(int(dim) for dim in shape)


def load_act_model_contract(model_dir: str | Path = DEFAULT_ACT_MODEL_DIR) -> ActModelContract:
    """Read the LeRobot ACT checkpoint config and return its IO contract."""

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    input_features = config["input_features"]
    output_features = config["output_features"]

    return ActModelContract(
        model_dir=model_dir,
        state_shape=_feature_shape(input_features, ACT_STATE_KEY),
        fixed_image_shape=_feature_shape(input_features, ACT_FIXED_IMAGE_KEY),
        handeye_image_shape=_feature_shape(input_features, ACT_HANDEYE_IMAGE_KEY),
        action_shape=_feature_shape(output_features, ACT_ACTION_KEY),
        chunk_size=int(config["chunk_size"]),
        n_action_steps=int(config["n_action_steps"]),
        vision_backbone=str(config["vision_backbone"]),
    )


def validate_expected_act_contract(
    model_dir: str | Path = DEFAULT_ACT_MODEL_DIR,
    *,
    expected_chunk_size: int = EXPECTED_ACT_CHUNK_SIZE,
    expected_n_action_steps: int = EXPECTED_ACT_N_ACTION_STEPS,
) -> ActModelContract:
    """Validate the checkpoint matches the SO101 ACT-PPO adapter assumptions."""

    contract = load_act_model_contract(model_dir)
    mismatches: list[str] = []

    expected = {
        "state_shape": EXPECTED_ACT_STATE_SHAPE,
        "fixed_image_shape": EXPECTED_ACT_IMAGE_SHAPE,
        "handeye_image_shape": EXPECTED_ACT_IMAGE_SHAPE,
        "action_shape": EXPECTED_ACT_ACTION_SHAPE,
        "chunk_size": int(expected_chunk_size),
        "n_action_steps": int(expected_n_action_steps),
        "vision_backbone": EXPECTED_ACT_VISION_BACKBONE,
    }
    for field_name, expected_value in expected.items():
        actual_value = getattr(contract, field_name)
        if actual_value != expected_value:
            mismatches.append(f"{field_name}: expected {expected_value!r}, got {actual_value!r}")

    if mismatches:
        mismatch_text = "\n".join(f"- {item}" for item in mismatches)
        raise ValueError(f"ACT checkpoint contract does not match this adapter:\n{mismatch_text}")

    return contract
