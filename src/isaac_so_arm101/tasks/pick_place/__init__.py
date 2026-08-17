import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubePPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeActObsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_act_ppo_cfg:PickPlaceCubeActPPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeActObsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_act_ppo_cfg:PickPlaceCubeActPPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeActV5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_act_ppo_cfg:PickPlaceCubeActV5PPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V5-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeActV5EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_act_ppo_cfg:PickPlaceCubeActV5PPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V34-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeActV34EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_act_ppo_cfg:PickPlaceCubeActV34PPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V34-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeActV34EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_act_ppo_cfg:PickPlaceCubeActV34PPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V35-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeActV35EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_act_ppo_cfg:PickPlaceCubeActV35PPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V35-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeActV35EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_act_ppo_cfg:PickPlaceCubeActV35PPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V36-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeActV36EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_act_ppo_cfg:PickPlaceCubeActV36PPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V36-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeActV36EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_act_ppo_cfg:PickPlaceCubeActV36PPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V37-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeActV37EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_act_ppo_cfg:PickPlaceCubeActV37PPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-ACT-V37-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeActV37EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_act_ppo_cfg:PickPlaceCubeActV37PPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-Robust-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robust_eval_cfg:SoArm101PickPlaceCubeEnvCfg_RobustPlay",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubePPORunnerCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Cube-Vision-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeVisionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
