import json
from pathlib import Path

import pytest

from isaac_so_arm101.policies.act_contract import (
    ACT_RED_CUBE_TARGET_MODEL_DIR,
    validate_expected_act_contract,
)


def _write_act_config(model_dir: Path, *, chunk_size: int, n_action_steps: int) -> None:
    model_dir.mkdir(parents=True)
    config = {
        "input_features": {
            "observation.state": {"shape": [6]},
            "observation.images.fixed": {"shape": [3, 480, 640]},
            "observation.images.handeye": {"shape": [3, 480, 640]},
        },
        "output_features": {"action": {"shape": [6]}},
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
        "vision_backbone": "resnet18",
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_default_contract_keeps_legacy_temporal_shape(tmp_path: Path) -> None:
    model_dir = tmp_path / "legacy"
    _write_act_config(model_dir, chunk_size=100, n_action_steps=100)

    contract = validate_expected_act_contract(model_dir)

    assert contract.chunk_size == 100
    assert contract.n_action_steps == 100


def test_contract_accepts_v37_temporal_shape(tmp_path: Path) -> None:
    model_dir = tmp_path / "v37"
    _write_act_config(model_dir, chunk_size=50, n_action_steps=25)

    contract = validate_expected_act_contract(
        model_dir,
        expected_chunk_size=50,
        expected_n_action_steps=25,
    )

    assert contract.chunk_size == 50
    assert contract.n_action_steps == 25


def test_contract_reports_temporal_shape_mismatch(tmp_path: Path) -> None:
    model_dir = tmp_path / "wrong-temporal-shape"
    _write_act_config(model_dir, chunk_size=50, n_action_steps=25)

    with pytest.raises(ValueError) as exc_info:
        validate_expected_act_contract(model_dir)

    message = str(exc_info.value)
    assert "chunk_size: expected 100, got 50" in message
    assert "n_action_steps: expected 100, got 25" in message


def test_v37_model_path_is_project_relative() -> None:
    assert ACT_RED_CUBE_TARGET_MODEL_DIR == Path(
        "data/act_model/train_act_red_cube_to_target_88_clean_terminal_1s/"
        "checkpoints/100000/pretrained_model"
    )
