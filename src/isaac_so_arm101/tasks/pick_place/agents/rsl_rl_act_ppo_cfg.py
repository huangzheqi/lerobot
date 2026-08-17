import math

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaac_so_arm101.policies.act_contract import (
    ACT_JOINT_NAMES,
    ACT_RED_CUBE_TARGET_MODEL_DIR,
    ACT_V5_MODEL_DIR,
    DEFAULT_ACT_MODEL_DIR,
)
from isaac_so_arm101.tasks.pick_place.joint_pos_env_cfg import (
    ACT_V5_RESET_STATE_DEG_PERCENT,
)


@configclass
class RslRlActActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "ActActorCritic"
    act_model_dir = str(DEFAULT_ACT_MODEL_DIR)
    expected_act_chunk_size = 100
    expected_act_n_action_steps = 100
    freeze_act = True
    # Sim images are still outside the real ACT training distribution. Keep ACT as a weak prior and
    # let PPO learn a residual large enough to cancel a wrong prior direction.
    act_prior_scale = 0.5
    use_residual = True
    residual_hidden_dims = [128, 64]
    residual_scale = 1.0
    residual_action_scale = [1.0, 1.5, 0.75, 1.5, 1.5, 2.0]
    residual_tanh = True
    act_state_group = "act_state"
    act_fixed_group = "act_fixed"
    act_handeye_group = "act_handeye"
    normalize_act_inputs = True
    unnormalize_act_actions = True
    convert_act_action_to_env_action = True
    act_joint_position_unit = "degrees"
    # Align Isaac's URDF joint zero with the hardware joint coordinates used by the ACT dataset.
    # Computed from dataset episode_000000 frame 0 minus the Isaac default arm pose in degrees.
    act_arm_joint_position_offset = [1.643193, -98.247078, -87.614273, -32.524305, -0.182911]
    env_arm_action_scale = 0.5
    env_arm_default_joint_pos = [0.0, 0.0, 0.0, 1.57, 0.0]
    # The real ACT dataset starts grasp episodes with low gripper values (~0.5-1.3) while the jaw is
    # open, then rises to 30+ during closing. Map low ACT values to Isaac's positive OPEN command.
    act_gripper_open_threshold = 15.0
    act_gripper_open_below_threshold = True
    env_gripper_open_action = 1.0
    env_gripper_close_action = -1.0
    debug_act = True
    debug_act_dir = "logs/act_debug"
    debug_act_max_calls = 64
    debug_act_env_index = 0
    debug_act_save_images = True
    debug_act_save_image_calls = 4


@configclass
class RslRlActV5ActorCriticCfg(RslRlActActorCriticCfg):
    """Factory-zero ACT-v5 prior with PPO residual correction."""

    act_model_dir = str(ACT_V5_MODEL_DIR)
    act_prior_scale = 1.0
    act_arm_joint_position_offset = [0.0] * len(ACT_JOINT_NAMES[:-1])
    env_arm_action_scale = 0.5
    env_arm_default_joint_pos = [
        math.radians(ACT_V5_RESET_STATE_DEG_PERCENT[name])
        for name in ACT_JOINT_NAMES[:-1]
    ]
    convert_act_gripper_state_to_percent = True
    act_gripper_mapping = "binary"
    act_gripper_open_threshold = 15.0
    act_gripper_open_below_threshold = True
    env_gripper_open_action = 1.0
    env_gripper_close_action = 0.0
    debug_act_dir = "logs/act_debug_v5"


@configclass
class RslRlActV34ActorCriticCfg(RslRlActV5ActorCriticCfg):
    """V5 prior constrained to locally reachable arm targets in sim."""

    act_arm_target_delta_limit = 5.0
    debug_act_dir = "logs/act_debug_v34"
    debug_act_max_calls = 256


@configclass
class RslRlActV35ActorCriticCfg(RslRlActV34ActorCriticCfg):
    """V34 ACT-v5 policy with an isolated V35 debug directory."""

    debug_act_dir = "logs/act_debug_v35"


@configclass
class RslRlActV36ActorCriticCfg(RslRlActV35ActorCriticCfg):
    """V35 policy with an isolated trace for continuous gripper rewards."""

    debug_act_dir = "logs/act_debug_v36"
    debug_act_max_calls = 512


@configclass
class RslRlActV37ActorCriticCfg(RslRlActV36ActorCriticCfg):
    """Factory-zero red-cube-to-target ACT prior with V36 residual PPO behavior."""

    act_model_dir = str(ACT_RED_CUBE_TARGET_MODEL_DIR)
    expected_act_chunk_size = 50
    expected_act_n_action_steps = 25
    debug_act_dir = "logs/act_debug_v37"


@configclass
class PickPlaceCubeActPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # High-resolution ACT image observations are stored in RSL-RL rollout buffers. Keep
    # rollout length short enough for 8 GB GPUs; increase only after checking nvidia-smi.
    num_steps_per_env = 8
    max_iterations = 12000
    save_interval = 500
    experiment_name = "pick_place_v32_act_v4_contact_stress_debug"
    empirical_normalization = False
    obs_groups = {
        "policy": ["act_state", "act_fixed", "act_handeye"],
        # Keep value learning on the original flat PPO observation group. The ACT actor consumes
        # structured image/state groups, which are intentionally not flattened into the critic.
        "critic": ["policy"],
    }
    policy = RslRlActActorCriticCfg(
        init_noise_std=0.15,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.5,
    )


@configclass
class PickPlaceCubeActV34PPORunnerCfg(PickPlaceCubeActPPORunnerCfg):
    """Safe ACT-v5 factory-zero prior with restored bootstrap rewards."""

    experiment_name = "pick_place_v34_act_v5_safe_bootstrap"
    policy = RslRlActV34ActorCriticCfg(
        init_noise_std=0.10,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class PickPlaceCubeActV35PPORunnerCfg(PickPlaceCubeActV34PPORunnerCfg):
    """V34 ACT+PPO settings with isolated V35 experiment outputs."""

    experiment_name = "pick_place_v35_act_v5_no_self_collision"
    policy = RslRlActV35ActorCriticCfg(
        init_noise_std=0.10,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class PickPlaceCubeActV36PPORunnerCfg(PickPlaceCubeActV35PPORunnerCfg):
    """V35 dynamics with the corrected continuous gripper reward contract."""

    experiment_name = "pick_place_v36_act_v5_continuous_gripper_close"
    policy = RslRlActV36ActorCriticCfg(
        init_noise_std=0.10,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class PickPlaceCubeActV37PPORunnerCfg(PickPlaceCubeActV36PPORunnerCfg):
    """V36 simulation contract with the new terminal-hold ACT checkpoint."""

    experiment_name = "pick_place_v37_act_red_cube_target_terminal_1s"
    policy = RslRlActV37ActorCriticCfg(
        init_noise_std=0.10,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class PickPlaceCubeActV5PPORunnerCfg(PickPlaceCubeActPPORunnerCfg):
    """ACT-v5 factory-zero prior with a trainable PPO residual actor."""

    experiment_name = "pick_place_v33_act_v5_factory_zero_residual"
    policy = RslRlActV5ActorCriticCfg(
        init_noise_std=0.10,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
