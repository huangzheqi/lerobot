from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PickPlaceCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 12000
    save_interval = 100
    experiment_name = "pick_place_v19_latch_fixedlr"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.28,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        # v9-standard LR restored after the v16-v19 fine-tune campaign was closed (2026-06-12).
        # Finding: EVERY warm-start fine-tune of converged v9 bled grasp away with the same gradual
        # lifting_object decline, across obs noise on/off, push penalty on/off, and adaptive 5e-5 vs
        # fixed 1e-5 (v19 isolated the stage3/4 was_lifted latch landscape as sufficient to cause
        # it). Conclusion: v9 is not safely fine-tunable on a modified reward landscape; improvements
        # must come from from-scratch runs or non-RL avenues. Active policy = v9 model_26994 frozen.
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.5,
    )
