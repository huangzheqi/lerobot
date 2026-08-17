from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "src/isaac_so_arm101/policies/act_contract.py"
ACT_CFG = ROOT / "src/isaac_so_arm101/tasks/pick_place/agents/rsl_rl_act_ppo_cfg.py"
ENV_CFG = ROOT / "src/isaac_so_arm101/tasks/pick_place/joint_pos_env_cfg.py"
TASK_INIT = ROOT / "src/isaac_so_arm101/tasks/pick_place/__init__.py"


def test_v37_actor_uses_new_checkpoint_and_temporal_contract() -> None:
    contract_source = CONTRACT.read_text(encoding="utf-8")
    cfg_source = ACT_CFG.read_text(encoding="utf-8")

    assert "ACT_RED_CUBE_TARGET_MODEL_DIR" in contract_source
    assert "class RslRlActV37ActorCriticCfg(RslRlActV36ActorCriticCfg)" in cfg_source
    assert "act_model_dir = str(ACT_RED_CUBE_TARGET_MODEL_DIR)" in cfg_source
    assert "expected_act_chunk_size = 50" in cfg_source
    assert "expected_act_n_action_steps = 25" in cfg_source
    assert 'debug_act_dir = "logs/act_debug_v37"' in cfg_source


def test_v37_runner_and_environment_are_isolated_from_v36() -> None:
    cfg_source = ACT_CFG.read_text(encoding="utf-8")
    env_source = ENV_CFG.read_text(encoding="utf-8")

    assert "class PickPlaceCubeActV37PPORunnerCfg(PickPlaceCubeActV36PPORunnerCfg)" in cfg_source
    assert 'experiment_name = "pick_place_v37_act_red_cube_target_terminal_1s"' in cfg_source
    assert "class SoArm101PickPlaceCubeActV37EnvCfg(SoArm101PickPlaceCubeActV36EnvCfg)" in env_source
    assert "class SoArm101PickPlaceCubeActV37EnvCfg_PLAY(SoArm101PickPlaceCubeActV37EnvCfg)" in env_source


def test_v37_train_and_play_tasks_are_registered() -> None:
    registration_source = TASK_INIT.read_text(encoding="utf-8")

    assert 'id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V37-v0"' in registration_source
    assert 'id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V37-Play-v0"' in registration_source
    assert "SoArm101PickPlaceCubeActV37EnvCfg" in registration_source
    assert "PickPlaceCubeActV37PPORunnerCfg" in registration_source
