# AGENTS.md

## Project

This repository is an Isaac Lab extension for SO-ARM100/SO-ARM101 reinforcement learning tasks.

The main objective is to develop and refine SO101 reinforcement learning tasks in Isaac Lab, especially:
- Reach
- Lift-Cube
- Pick-and-Place-Cube
- Post-processing play scripts for release and home return behavior

## Important paths

Core task files:
- `src/isaac_so_arm101/tasks/lift/`
- `src/isaac_so_arm101/tasks/pick_place/`
- `src/isaac_so_arm101/tasks/lift/mdp/rewards.py`
- `src/isaac_so_arm101/tasks/pick_place/joint_pos_env_cfg.py`
- `src/isaac_so_arm101/tasks/pick_place/agents/rsl_rl_ppo_cfg.py`

Core scripts:
- `src/isaac_so_arm101/scripts/rsl_rl/train.py`
- `src/isaac_so_arm101/scripts/rsl_rl/play.py`

Utility scripts:
- `tools/plot_rl_curves.py`

## Do not run

Do not run Isaac Sim, Isaac Lab training, or GPU simulation commands inside Codex.

Do not run:
- `uv run train ...`
- `uv run play ...`
- `python isaacsim ...`
- commands requiring NVIDIA GPU, Vulkan, RTX, GUI rendering, or Isaac Sim window startup

These commands must be run manually on the user's local Ubuntu machine.

## Safe commands

Codex may run lightweight code checks only:
- `python -m py_compile <file>`
- `python -m compileall src/isaac_so_arm101`
- `git diff`
- `grep`
- `find`
- `sed`
- `cat`

## Coding rules

- Keep the original `Lift-Cube` task unchanged unless explicitly asked.
- New experiments should use new experiment names such as:
  - `pick_place_v3`
  - `pick_place_v4_release`
  - `pick_place_v5_smooth`
  - `pick_place_v6_gated`
- Do not overwrite trained checkpoints under `logs/rsl_rl/`.
- Do not delete previous experiment directories.
- Prefer adding new reward functions instead of deleting existing ones.
- Avoid modifying robot asset definitions unless explicitly asked.
- Avoid making large unrelated refactors.
- Keep changes focused and easy to diff.

## SO101 task design notes

The current development direction is:

1. Use Lift-Cube as the base manipulation ability.
2. Extend it into Pick-and-Place.
3. Use reinforcement learning for:
   - grasping
   - lifting
   - transporting the cube to the target area
   - releasing the cube near the target
4. Use scripted post-processing for:
   - opening the gripper after successful placement
   - returning the robot to the home pose

Do not force PPO to learn the full home-return behavior unless explicitly requested. Prefer a deterministic post-policy home motion in the play script.

## Reporting

When summarizing changes for the user, use Chinese.
Explain:
- what files were changed
- what each change does
- what command the user should run locally
- what metrics to observe during training
