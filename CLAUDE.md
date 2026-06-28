# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Isaac Lab extension implementing reinforcement learning tasks for the SO-ARM100 and SO-ARM101 robot arms. Environments are registered as Gymnasium tasks and trained with PPO via RSL-RL. Python 3.11, managed by `uv`.

## DO NOT RUN

**Never execute Isaac Sim / training / GPU simulation commands.** These require an NVIDIA GPU with Vulkan/RTX and cannot run headlessly in a coding environment.

Forbidden:
- `uv run train ...`
- `uv run play ...`
- `python -m isaacsim ...`

**Safe** lightweight checks:
```bash
python -m py_compile src/isaac_so_arm101/tasks/pick_place/mdp/rewards.py
python -m compileall src/isaac_so_arm101
```

## Key commands (run manually on local machine)

```bash
uv run list_envs                                      # list registered environments
uv run zero_agent --task SO-ARM100-Reach-Play-v0      # sanity-check with zero actions
uv run train --task Isaac-SO-ARM101-Pick-Place-Cube-v0 --headless
uv run play  --task Isaac-SO-ARM101-Pick-Place-Cube-Play-v0
```

Play options:
- `--object_pose_source gt|vision|resnet` — switch between ground-truth, colour-mask, and ResNet18 cube detection
- `--disturbance_type cube_init|goal|table_friction|clutter|lighting|all` + `--disturbance_level low|medium|high`
- `--save_gt_debug` / `--gt_debug_dir` — dump per-step CSV diagnostics

Checkpoints are stored under `logs/rsl_rl/<experiment_name>/`.

## Code architecture

### Package layout
```
src/isaac_so_arm101/
  robots/                    # ArticulationCfg for SO-ARM100 and SO-ARM101
  tasks/
    reach/                   # Reach task (IK policy baseline)
    lift/                    # Lift-Cube task — BASE, do not modify unless asked
      lift_env_cfg.py        # Abstract LiftEnvCfg (scene, MDP, rewards)
      joint_pos_env_cfg.py   # Concrete SO-ARM100/101 bindings (robot, cube, ee_frame)
      mdp/                   # rewards, observations, terminations re-exported from isaaclab_tasks
    pick_place/              # Pick-and-Place — primary development target
      joint_pos_env_cfg.py   # Overrides lift rewards with staged pick-place reward suite
      robust_eval_cfg.py     # apply_pick_place_disturbance() for stress tests
      agents/rsl_rl_ppo_cfg.py  # PPO hyperparameters and experiment name
      mdp/rewards.py         # All pick-place reward functions
  scripts/rsl_rl/
    train.py                 # RSL-RL training entry point
    play.py                  # Evaluation with vision/ResNet/GT pose + GT debug CSV
    vision_pose_resnet.py    # ResNet18 wrapper for cube pose estimation
    collect_cube_pose_dataset.py
    train_resnet18_cube_pose.py
    eval_resnet_vs_gt.py
```

### Inheritance chain (pick-place)
```
LiftEnvCfg (abstract)
  └─ SoArm101LiftCubeEnvCfg          (lift/joint_pos_env_cfg.py)
       └─ SoArm101PickPlaceCubeEnvCfg (pick_place/joint_pos_env_cfg.py)
            └─ SoArm101PickPlaceCubeEnvCfg_PLAY
                 └─ SoArm101PickPlaceCubeVisionEnvCfg_PLAY  (adds two cameras)
                 └─ SoArm101PickPlaceCubeEnvCfg_RobustPlay  (robust_eval_cfg.py)
```
`SoArm101PickPlaceCubeEnvCfg.__post_init__` disables the inherited lift goal-tracking rewards and replaces them with the full staged reward suite.

### Staged reward design (pick_place/mdp/rewards.py)
The pick-place task uses four explicit stages gated by object height and XY proximity to goal:
- **Stage 1**: approach + grasp (close gripper near object, penalise open gripper at contact distance, lift from initial position)
- **Stage 2**: transport (XY tracking toward goal while lifted; penalise early gripper open)
- **Stage 3**: descent (soft downward speed near goal, EE height and object height near table)
- **Stage 4**: release (open gripper near table; stable placed reward; penalty for holding too long)

`_gates()` returns four scalar masks `(s1, s2, s3, s4)` used to zero-out irrelevant reward terms per stage. Private helpers (`_goal_metrics`, `_ee_object_distance`, `_gripper_open_ratio`, `_object_height_gain`, `_object_episode_initial_root_pos_w`) are shared across reward functions and should not be duplicated.

### Experiment naming
New pick-place experiments must use a new `experiment_name` in `agents/rsl_rl_ppo_cfg.py` (e.g. `pick_place_v11_...`). Never overwrite an existing `logs/rsl_rl/` directory or delete previous checkpoints.

### Observation space
Policy observation (concatenated): `joint_pos_rel`, `joint_vel_rel`, `object_position_in_robot_root_frame`, `generated_commands` (goal pose in robot frame), `last_action`.

## Coding rules

- Keep `tasks/lift/` unchanged unless explicitly asked.
- Prefer adding new reward functions over modifying existing ones.
- Do not touch robot asset definitions (`robots/`) unless asked.
- Keep diffs focused; avoid unrelated refactors.
- When summarising changes for the user, write the summary in Chinese.
