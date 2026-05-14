import argparse
import csv
import os
import sys
from collections import Counter

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate SO101 pick-place model with cube position perturbation.")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Pick-Place-Cube-Play-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments for evaluation.")
parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes for each test.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
parser.add_argument("--output_csv", type=str, default="outputs/eval/final_pick_place_v9_cube_position_eval.csv", help="CSV output path.")
parser.add_argument("--test_name", action="append", choices=["baseline", "cube_pos_small"], help="Test(s) to run.")
parser.add_argument("--run_all", action="store_true", help="Run both baseline and cube_pos_small.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401


TEST_POSE_RANGES = {
    "baseline": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
    "cube_pos_small": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    tests = ["baseline", "cube_pos_small"] if args_cli.run_all else (args_cli.test_name or ["baseline"])

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    resume_path = retrieve_file_path(args_cli.checkpoint)

    os.makedirs(os.path.dirname(args_cli.output_csv), exist_ok=True)
    fieldnames = [
        "episode_id",
        "test_name",
        "seed",
        "cube_init_x",
        "cube_init_y",
        "target_x",
        "target_y",
        "grasp_success",
        "place_success",
        "final_success",
        "final_cube_target_dist",
        "fail_stage",
    ]
    with open(args_cli.output_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        summaries = {}
        for test_name in tests:
            env_cfg.events.reset_object_position.params["pose_range"] = TEST_POSE_RANGES[test_name]
            env = gym.make(args_cli.task, cfg=env_cfg)
            if isinstance(env.unwrapped, DirectMARLEnv):
                env = multi_agent_to_single_agent(env)

            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            rows = run_eval_episodes(env, policy, args_cli.num_episodes, test_name, args_cli.seed)
            for row in rows:
                writer.writerow(row)
            summaries[test_name] = summarize(rows)
            env.close()

    print_summary(summaries)


def run_eval_episodes(env, policy, num_episodes: int, test_name: str, seed: int):
    rows = []
    obs, _ = env.reset()
    episode_id = 0

    object_asset = env.unwrapped.scene["object"]
    robot_asset = env.unwrapped.scene["robot"]

    current = init_episode_stats()

    while simulation_app.is_running() and episode_id < num_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            step_out = env.step(actions)
            if len(step_out) == 5:
                obs, _, terminated, truncated, _ = step_out
                done = torch.logical_or(terminated, truncated)
            elif len(step_out) == 4:
                obs, _, done, _ = step_out
            else:
                raise RuntimeError(f"Unexpected env.step output length: {len(step_out)}")

        command = env.unwrapped.command_manager.get_command("object_pose")
        robot_root = robot_asset.data.root_pos_w[:, :3]
        target_pos_w = robot_root + command[:, :3]
        cube_pos = object_asset.data.root_pos_w[:, :3]
        cube_vel = torch.norm(object_asset.data.root_lin_vel_w[:, :3], dim=1)
        cube_to_target_dist = torch.norm(target_pos_w[:, :2] - cube_pos[:, :2], dim=1)

        if current["cube_init_x"] is None:
            current["cube_init_x"] = float(cube_pos[0, 0].item())
            current["cube_init_y"] = float(cube_pos[0, 1].item())

        current["target_x"] = float(target_pos_w[0, 0].item())
        current["target_y"] = float(target_pos_w[0, 1].item())
        current["final_cube_target_dist"] = float(cube_to_target_dist[0].item())

        if float(cube_pos[0, 2].item()) > 0.05:
            current["grasp_success"] = True
        if float(cube_to_target_dist[0].item()) < 0.05 and abs(float(cube_pos[0, 2].item()) - 0.025) < 0.02 and float(cube_vel[0].item()) < 0.1:
            current["place_success"] = True

        done_flag = bool(done[0].item()) if hasattr(done[0], "item") else bool(done[0])
        if done_flag:
            current["final_success"] = current["grasp_success"] and current["place_success"]
            current["fail_stage"] = compute_fail_stage(current)
            current["episode_id"] = episode_id
            current["test_name"] = test_name
            current["seed"] = seed + episode_id
            rows.append(current)

            episode_id += 1
            current = init_episode_stats()

    return rows


def init_episode_stats():
    return {
        "episode_id": -1,
        "test_name": "",
        "seed": 0,
        "cube_init_x": None,
        "cube_init_y": None,
        "target_x": 0.0,
        "target_y": 0.0,
        "grasp_success": False,
        "place_success": False,
        "final_success": False,
        "final_cube_target_dist": 0.0,
        "fail_stage": "",
    }


def compute_fail_stage(row):
    if row["final_success"]:
        return "success"
    if not row["grasp_success"]:
        return "grasp_failed"
    if row["grasp_success"] and not row["place_success"]:
        if row["final_cube_target_dist"] > 0.08:
            return "move_failed"
        return "place_failed"
    return "place_failed"


def summarize(rows):
    total = max(len(rows), 1)
    success_rate = sum(1 for r in rows if r["final_success"]) / total
    mean_dist = sum(r["final_cube_target_dist"] for r in rows) / total
    fail_counter = Counter(r["fail_stage"] for r in rows)
    return {"success_rate": success_rate, "mean_dist": mean_dist, "fail_counter": fail_counter}


def print_summary(summaries):
    base = summaries.get("baseline", {"success_rate": 0.0, "mean_dist": 0.0, "fail_counter": {}})
    pert = summaries.get("cube_pos_small", {"success_rate": 0.0, "mean_dist": 0.0, "fail_counter": {}})

    print("\n===== Evaluation Summary =====")
    print(f"baseline success rate: {base['success_rate']:.2%}")
    print(f"cube_pos_small success rate: {pert['success_rate']:.2%}")
    print(f"success rate drop: {(base['success_rate'] - pert['success_rate']):.2%}")

    if "baseline" in summaries:
        print(f"baseline mean final_cube_target_dist: {base['mean_dist']:.4f}")
        print(f"baseline common fail stages: {dict(base['fail_counter'])}")
    if "cube_pos_small" in summaries:
        print(f"cube_pos_small mean final_cube_target_dist: {pert['mean_dist']:.4f}")
        print(f"cube_pos_small common fail stages: {dict(pert['fail_counter'])}")


if __name__ == "__main__":
    main()
    simulation_app.close()
