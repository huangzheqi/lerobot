# AGENTS.md

## Project

This repository is an Isaac Lab extension for SO-ARM100/SO-ARM101 reinforcement learning tasks.
Environments are registered as Gymnasium tasks and trained with PPO via RSL-RL.
The project uses Python 3.11 and is managed with `uv`.

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
- `src/isaac_so_arm101/tasks/pick_place/mdp/rewards.py`
- `src/isaac_so_arm101/tasks/pick_place/robust_eval_cfg.py`

Core scripts:
- `src/isaac_so_arm101/scripts/rsl_rl/train.py`
- `src/isaac_so_arm101/scripts/rsl_rl/play.py`
- `src/isaac_so_arm101/scripts/rsl_rl/vision_pose_resnet.py`
- `src/isaac_so_arm101/scripts/rsl_rl/collect_cube_pose_dataset.py`
- `src/isaac_so_arm101/scripts/rsl_rl/train_resnet18_cube_pose.py`
- `src/isaac_so_arm101/scripts/rsl_rl/eval_resnet_vs_gt.py`

Utility scripts:
- `tools/plot_rl_curves.py`

Generated documents:
- Save document deliverables and their source projects under `/home/hzq/rl_disk/docs/` by default.
- This includes PPTX, PDF, DOCX, Markdown notes, reports, and document-related exported images.
- Create a dedicated subdirectory for each document project instead of writing generated documents into the repository.
- Use another destination only when the user explicitly requests it.

## Code architecture

Package layout:
- `src/isaac_so_arm101/robots/`: SO-ARM100 and SO-ARM101 `ArticulationCfg` definitions.
- `src/isaac_so_arm101/tasks/reach/`: Reach task and IK policy baseline.
- `src/isaac_so_arm101/tasks/lift/`: Lift-Cube base task. Keep unchanged unless explicitly asked.
- `src/isaac_so_arm101/tasks/pick_place/`: Primary Pick-and-Place development target.
- `src/isaac_so_arm101/scripts/rsl_rl/`: RSL-RL training, play, vision, and dataset scripts.

Pick-and-place inheritance chain:

```text
LiftEnvCfg
  -> SoArm101LiftCubeEnvCfg
       -> SoArm101PickPlaceCubeEnvCfg
            -> SoArm101PickPlaceCubeEnvCfg_PLAY
                 -> SoArm101PickPlaceCubeVisionEnvCfg_PLAY
                 -> SoArm101PickPlaceCubeEnvCfg_RobustPlay
```

`SoArm101PickPlaceCubeEnvCfg.__post_init__` disables inherited lift goal-tracking rewards
and replaces them with the staged pick-place reward suite.

## Do not run

Do not run Isaac Sim, Isaac Lab training, or GPU simulation commands inside Codex.

Do not run:
- `uv run train ...`
- `uv run play ...`
- `python isaacsim ...`
- `python -m isaacsim ...`
- commands requiring NVIDIA GPU, Vulkan, RTX, GUI rendering, or Isaac Sim window startup

These commands must be run manually on the user's local Ubuntu machine.

## Manual commands for the user

The following commands are useful, but Codex must not run them:

```bash
uv run list_envs
uv run zero_agent --task SO-ARM100-Reach-Play-v0
uv run train --task Isaac-SO-ARM101-Pick-Place-Cube-v0 --headless
uv run play --task Isaac-SO-ARM101-Pick-Place-Cube-Play-v0
```

Useful `play.py` options:
- `--object_pose_source gt|vision|resnet`: switch between ground-truth, colour-mask, and ResNet18 cube detection.
- `--disturbance_type cube_init|goal|table_friction|clutter|lighting|all` with `--disturbance_level low|medium|high`: run robustness checks.
- `--save_gt_debug` and `--gt_debug_dir`: dump per-step CSV diagnostics.

Checkpoints are stored under `logs/rsl_rl/<experiment_name>/`.

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
- For new pick-place experiments, set a fresh `experiment_name` in `src/isaac_so_arm101/tasks/pick_place/agents/rsl_rl_ppo_cfg.py`, for example `pick_place_vN_description`.
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

## Pick-and-place reward design

The pick-place task uses staged rewards gated by object height and XY proximity to the goal:
- Stage 1: approach and grasp. Encourage closing the gripper near the object, discourage opening at contact distance, and lift from the initial position.
- Stage 2: transport. Track XY goal position while lifted and penalize early gripper opening.
- Stage 3: descent. Encourage soft downward motion near the goal, appropriate end-effector height, and object height near the table.
- Stage 4: release. Encourage opening the gripper near the table, stable placement, and avoid holding too long.

`_gates()` returns scalar masks `(s1, s2, s3, s4)` to zero out irrelevant reward terms per stage.
Shared private helpers such as `_goal_metrics`, `_ee_object_distance`, `_gripper_open_ratio`,
`_object_height_gain`, and `_object_episode_initial_root_pos_w` should be reused instead of duplicated.

## Observation space

Policy observations are concatenated from:
- `joint_pos_rel`
- `joint_vel_rel`
- `object_position_in_robot_root_frame`
- `generated_commands`
- `last_action`

## Reporting

When summarizing changes for the user, use Chinese.
Explain:
- what files were changed
- what each change does
- what command the user should run locally
- what metrics to observe during training
