"""RSL-RL actor-critic adapter that uses a LeRobot ACT policy as the actor prior."""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.networks import EmpiricalNormalization, MLP
from torch.distributions import Normal

from isaac_so_arm101.policies.act_joint_mapping import (
    gripper_joint_to_percent,
    gripper_percent_to_joint,
)
from isaac_so_arm101.policies.act_contract import (
    ACT_FIXED_IMAGE_KEY,
    ACT_HANDEYE_IMAGE_KEY,
    ACT_STATE_KEY,
    DEFAULT_ACT_MODEL_DIR,
    validate_expected_act_contract,
)
from isaac_so_arm101.policies.act_normalization import load_act_normalization_stats


def _ensure_lerobot_policies_stub() -> None:
    """Avoid importing every LeRobot policy when only ACT is needed."""

    if "lerobot.policies" in sys.modules:
        return

    import lerobot

    policies_path = Path(lerobot.__file__).parent / "policies"
    policies_pkg = types.ModuleType("lerobot.policies")
    policies_pkg.__path__ = [str(policies_path)]
    sys.modules["lerobot.policies"] = policies_pkg


def _load_compatible_lerobot_act_config(config_module_name: str, model_dir: Path, device: str | None = None):
    """Load old LeRobot ACT config files with fields unsupported by the installed LeRobot."""

    import draccus

    config_module = __import__(config_module_name, fromlist=["ACTConfig"])
    act_config_cls = getattr(config_module, "ACTConfig")
    valid_fields = {field.name for field in dataclasses.fields(act_config_cls)}

    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)

    clean_config = {key: value for key, value in raw_config.items() if key in valid_fields}
    if device is not None and "device" in valid_fields:
        clean_config["device"] = device
    if device == "cpu" and "use_amp" in valid_fields:
        clean_config["use_amp"] = False
    dropped_fields = sorted(set(raw_config) - valid_fields - {"type"})
    if dropped_fields:
        print(
            "[INFO] Ignoring LeRobot ACT config fields not supported by the installed ACTConfig: "
            + str(dropped_fields)
        )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump(clean_config, f)
            temp_path = Path(f.name)
        with draccus.config_type("json"):
            return draccus.parse(act_config_cls, str(temp_path), args=[])
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _load_lerobot_act_policy(model_dir: Path, device: str | None = None) -> nn.Module:
    """Load ACT from LeRobot without making LeRobot a hard import-time dependency."""

    import_errors: list[Exception] = []
    for module_name in (
        "lerobot.common.policies.act.modeling_act",
        "lerobot.policies.act.modeling_act",
    ):
        try:
            if module_name.startswith("lerobot.policies."):
                _ensure_lerobot_policies_stub()
            module = __import__(module_name, fromlist=["ACTPolicy"])
            act_policy_cls = getattr(module, "ACTPolicy")
            config_module_name = module_name.replace("modeling_act", "configuration_act")
            act_config = _load_compatible_lerobot_act_config(config_module_name, model_dir, device=device)
            return act_policy_cls.from_pretrained(str(model_dir), config=act_config)
        except ModuleNotFoundError as exc:
            import_errors.append(exc)
        except AttributeError as exc:
            import_errors.append(exc)

    details = "; ".join(str(exc) for exc in import_errors)
    raise ImportError(
        "Unable to import LeRobot ACTPolicy. Install a LeRobot version that provides "
        "`ACTPolicy.from_pretrained(...)` before running the ACT-PPO task. "
        f"Tried checkpoint directory: {model_dir}. Import errors: {details}"
    )


class ActActorCritic(nn.Module):
    """PPO-compatible actor-critic with ACT as a frozen or trainable actor backbone."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=None,
        critic_hidden_dims=(256, 128, 64),
        activation="elu",
        init_noise_std=0.1,
        noise_std_type: str = "scalar",
        act_model_dir: str | Path = DEFAULT_ACT_MODEL_DIR,
        expected_act_chunk_size: int = 100,
        expected_act_n_action_steps: int = 100,
        freeze_act: bool = True,
        act_prior_scale: float = 1.0,
        use_residual: bool = False,
        residual_hidden_dims=(128, 64),
        residual_scale: float = 0.05,
        residual_action_scale=None,
        residual_tanh: bool = True,
        act_state_group: str = "act_state",
        act_fixed_group: str = "act_fixed",
        act_handeye_group: str = "act_handeye",
        normalize_act_inputs: bool = True,
        unnormalize_act_actions: bool = True,
        convert_act_action_to_env_action: bool = True,
        act_joint_position_unit: str = "degrees",
        act_arm_joint_position_offset=None,
        act_arm_target_delta_limit: float | None = None,
        env_arm_action_scale: float = 0.5,
        env_arm_default_joint_pos=None,
        convert_act_gripper_state_to_percent: bool = False,
        act_gripper_mapping: str = "binary",
        env_gripper_open_joint_pos: float = 0.5,
        env_gripper_closed_joint_pos: float = 0.0,
        env_gripper_action_scale: float = 0.5,
        env_gripper_action_offset: float = 0.0,
        act_gripper_open_threshold: float = 15.0,
        act_gripper_open_below_threshold: bool = False,
        env_gripper_open_action: float = 1.0,
        env_gripper_close_action: float = -1.0,
        debug_act: bool = False,
        debug_act_dir: str = "logs/act_debug",
        debug_act_max_calls: int = 64,
        debug_act_env_index: int = 0,
        debug_act_save_images: bool = True,
        debug_act_save_image_calls: int = 4,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        if actor_obs_normalization:
            print("ActActorCritic ignores actor_obs_normalization because ACT owns actor-side normalization.")
        if actor_hidden_dims is not None:
            print("ActActorCritic ignores actor_hidden_dims; use residual_hidden_dims for the optional residual actor.")

        super().__init__()

        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.act_model_dir = Path(act_model_dir)
        self.contract = validate_expected_act_contract(
            self.act_model_dir,
            expected_chunk_size=expected_act_chunk_size,
            expected_n_action_steps=expected_act_n_action_steps,
        )
        if self.contract.action_shape != (num_actions,):
            raise ValueError(
                f"ACT action shape {self.contract.action_shape} does not match env num_actions={num_actions}"
            )

        self.act_state_group = act_state_group
        self.act_fixed_group = act_fixed_group
        self.act_handeye_group = act_handeye_group
        self.normalize_act_inputs = normalize_act_inputs
        self.unnormalize_act_actions = unnormalize_act_actions
        self.convert_act_action_to_env_action = convert_act_action_to_env_action
        self.act_joint_position_unit = act_joint_position_unit
        self.env_arm_action_scale = float(env_arm_action_scale)
        self.convert_act_gripper_state_to_percent = bool(convert_act_gripper_state_to_percent)
        self.act_arm_target_delta_limit = (
            None if act_arm_target_delta_limit is None else float(act_arm_target_delta_limit)
        )
        self.act_gripper_mapping = str(act_gripper_mapping)
        self.env_gripper_open_joint_pos = float(env_gripper_open_joint_pos)
        self.env_gripper_closed_joint_pos = float(env_gripper_closed_joint_pos)
        self.env_gripper_action_scale = float(env_gripper_action_scale)
        self.env_gripper_action_offset = float(env_gripper_action_offset)
        self.act_gripper_open_threshold = float(act_gripper_open_threshold)
        self.act_gripper_open_below_threshold = bool(act_gripper_open_below_threshold)
        self.env_gripper_open_action = float(env_gripper_open_action)
        self.env_gripper_close_action = float(env_gripper_close_action)
        if self.act_joint_position_unit not in ("degrees", "radians"):
            raise ValueError(
                "act_joint_position_unit must be either 'degrees' or 'radians', "
                f"got {self.act_joint_position_unit!r}"
            )
        if self.env_arm_action_scale == 0.0:
            raise ValueError("env_arm_action_scale must be non-zero")
        if self.act_arm_target_delta_limit is not None and self.act_arm_target_delta_limit <= 0.0:
            raise ValueError("act_arm_target_delta_limit must be positive or None")
        if self.act_gripper_mapping not in ("binary", "continuous_percent"):
            raise ValueError(
                "act_gripper_mapping must be 'binary' or 'continuous_percent', "
                f"got {self.act_gripper_mapping!r}"
            )
        if self.env_gripper_open_joint_pos <= self.env_gripper_closed_joint_pos:
            raise ValueError(
                "env_gripper_open_joint_pos must be greater than env_gripper_closed_joint_pos"
            )
        if self.act_gripper_mapping == "continuous_percent" and self.env_gripper_action_scale == 0.0:
            raise ValueError(
                "env_gripper_action_scale must be non-zero for continuous_percent mapping"
            )
        if env_arm_default_joint_pos is None:
            env_arm_default_joint_pos = [0.0, 0.0, 0.0, 1.57, 0.0]
        if len(env_arm_default_joint_pos) != num_actions - 1:
            raise ValueError(
                "env_arm_default_joint_pos must contain one value per non-gripper action "
                f"({num_actions - 1}), got {len(env_arm_default_joint_pos)}"
            )
        if act_arm_joint_position_offset is None:
            act_arm_joint_position_offset = [0.0] * (num_actions - 1)
        if len(act_arm_joint_position_offset) != num_actions - 1:
            raise ValueError(
                "act_arm_joint_position_offset must contain one value per non-gripper action "
                f"({num_actions - 1}), got {len(act_arm_joint_position_offset)}"
            )
        self.freeze_act = freeze_act
        self.act_prior_scale = float(act_prior_scale)
        self.use_residual = use_residual
        self.residual_scale = float(residual_scale)
        self.residual_tanh = residual_tanh
        self.debug_act = debug_act
        self.debug_act_dir = Path(debug_act_dir)
        self.debug_act_max_calls = int(debug_act_max_calls)
        self.debug_act_env_index = int(debug_act_env_index)
        self.debug_act_save_images = debug_act_save_images
        self.debug_act_save_image_calls = int(debug_act_save_image_calls)
        self._debug_act_calls = 0
        self._debug_act_csv_path: Path | None = None
        self._debug_act_payload: dict[str, torch.Tensor] | None = None
        if not self.freeze_act:
            raise NotImplementedError(
                "ActActorCritic currently supports freeze_act=True only. "
                "End-to-end ACT PPO training needs a differentiable LeRobot ACT forward path, "
                "not select_action()."
            )

        self.act_policy = _load_lerobot_act_policy(self.act_model_dir)
        if self.freeze_act:
            self.act_policy.eval()
            for param in self.act_policy.parameters():
                param.requires_grad_(False)

        act_stats = load_act_normalization_stats(self.act_model_dir)
        self.register_buffer("act_state_mean", act_stats.state_mean)
        self.register_buffer("act_state_std", act_stats.state_std)
        self.register_buffer("act_fixed_image_mean", act_stats.fixed_image_mean)
        self.register_buffer("act_fixed_image_std", act_stats.fixed_image_std)
        self.register_buffer("act_handeye_image_mean", act_stats.handeye_image_mean)
        self.register_buffer("act_handeye_image_std", act_stats.handeye_image_std)
        self.register_buffer("act_action_mean", act_stats.action_mean)
        self.register_buffer("act_action_std", act_stats.action_std)
        self.register_buffer("env_arm_default_joint_pos", torch.tensor(env_arm_default_joint_pos, dtype=torch.float32))
        self.register_buffer(
            "act_arm_joint_position_offset",
            torch.tensor(act_arm_joint_position_offset, dtype=torch.float32),
        )
        if residual_action_scale is None:
            residual_action_scale = [self.residual_scale] * num_actions
        if len(residual_action_scale) != num_actions:
            raise ValueError(
                f"residual_action_scale must contain {num_actions} values, got {len(residual_action_scale)}"
            )
        self.register_buffer("residual_action_scale", torch.tensor(residual_action_scale, dtype=torch.float32))

        # Actor observation normalization is intentionally an identity: LeRobot stores ACT normalization.
        self.actor_obs_normalization = False
        self.actor_obs_normalizer = torch.nn.Identity()

        num_critic_obs = self._num_flat_obs(obs, obs_groups["critic"])
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        if self.use_residual:
            num_residual_obs = self._num_flat_obs(obs, obs_groups["critic"])
            self.residual_actor = MLP(num_residual_obs, num_actions, list(residual_hidden_dims), activation)
            self._zero_residual_output_layer()
            print(f"ACT residual MLP: {self.residual_actor}")
        else:
            self.residual_actor = None

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        if hasattr(self.act_policy, "reset"):
            self.act_policy.reset()

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.act_inference(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        act_prior = self._act_mean(obs)
        mean = self.act_prior_scale * act_prior
        residual = None
        if self.residual_actor is not None:
            residual_obs = self.get_critic_obs(obs)
            residual = self.residual_actor(self.critic_obs_normalizer(residual_obs))
            if self.residual_tanh:
                residual = torch.tanh(residual)
            residual = residual * self.residual_action_scale
            mean = mean + residual
        self._write_act_debug(mean, residual)
        return mean

    def evaluate(self, obs, **kwargs):
        critic_obs = self.get_critic_obs(obs)
        return self.critic(self.critic_obs_normalizer(critic_obs))

    def get_actor_obs(self, obs):
        return {
            ACT_STATE_KEY: self._get_tensor_obs(obs, self.act_state_group),
            ACT_FIXED_IMAGE_KEY: self._get_tensor_obs(obs, self.act_fixed_group),
            ACT_HANDEYE_IMAGE_KEY: self._get_tensor_obs(obs, self.act_handeye_group),
        }

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            group_obs = self._get_tensor_obs(obs, obs_group)
            obs_list.append(group_obs)
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    @staticmethod
    def _get_tensor_obs(obs: Any, group_name: str) -> torch.Tensor:
        group_obs = obs[group_name]
        if isinstance(group_obs, torch.Tensor):
            return group_obs
        if hasattr(group_obs, "keys") and "rgb" in group_obs.keys():
            return group_obs["rgb"]
        if hasattr(group_obs, "keys") and "joint_pos" in group_obs.keys():
            return group_obs["joint_pos"]
        raise TypeError(f"Observation group {group_name!r} must be a tensor or contain a known ACT term.")

    @staticmethod
    def _num_flat_obs(obs, obs_group_names: list[str]) -> int:
        total = 0
        for obs_group in obs_group_names:
            group_obs = obs[obs_group]
            if not isinstance(group_obs, torch.Tensor):
                raise TypeError(f"Critic observation group {obs_group!r} must be a flat tensor, got {type(group_obs)}")
            if len(group_obs.shape) != 2:
                raise ValueError(
                    f"Critic observation group {obs_group!r} must be 2D [num_envs, dim], got {tuple(group_obs.shape)}"
                )
            total += group_obs.shape[-1]
        return total

    def _act_mean(self, obs) -> torch.Tensor:
        batch = self.get_actor_obs(obs)
        state = batch[ACT_STATE_KEY]
        fixed = batch[ACT_FIXED_IMAGE_KEY]
        handeye = batch[ACT_HANDEYE_IMAGE_KEY]

        if state.shape[-1:] != self.contract.state_shape:
            raise ValueError(f"ACT state shape mismatch: expected (*,{self.contract.state_shape}), got {tuple(state.shape)}")
        if fixed.shape[1:] != self.contract.fixed_image_shape:
            raise ValueError(
                f"ACT fixed image shape mismatch: expected (*,{self.contract.fixed_image_shape}), got {tuple(fixed.shape)}"
            )
        if handeye.shape[1:] != self.contract.handeye_image_shape:
            raise ValueError(
                "ACT handeye image shape mismatch: "
                f"expected (*,{self.contract.handeye_image_shape}), got {tuple(handeye.shape)}"
            )

        if hasattr(self.act_policy, "reset"):
            self.act_policy.reset()

        state_act_units = self._state_to_act_units(state)
        state_prepared = self._normalize_act_state(state_act_units)
        fixed_prepared = self._prepare_act_image(fixed, self.act_fixed_image_mean, self.act_fixed_image_std)
        handeye_prepared = self._prepare_act_image(handeye, self.act_handeye_image_mean, self.act_handeye_image_std)
        policy_batch = {
            ACT_STATE_KEY: state_prepared,
            ACT_FIXED_IMAGE_KEY: fixed_prepared,
            ACT_HANDEYE_IMAGE_KEY: handeye_prepared,
        }
        with torch.inference_mode():
            action = self.act_policy.select_action(policy_batch)
        if isinstance(action, dict):
            action = action.get("action")
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"ACT policy returned unsupported action type: {type(action)}")
        if action.ndim == 3:
            action = action[:, 0, :]
        if action.shape[-1] != self.num_actions:
            raise ValueError(f"ACT action dim mismatch: expected {self.num_actions}, got shape={tuple(action.shape)}")
        act_action = action
        if self.unnormalize_act_actions:
            act_action = act_action * self.act_action_std + self.act_action_mean
        act_action_raw = act_action
        if self.act_arm_target_delta_limit is not None:
            arm_state = state_act_units[:, : self.num_actions - 1]
            arm_target = act_action[:, : self.num_actions - 1]
            arm_delta = torch.clamp(
                arm_target - arm_state,
                min=-self.act_arm_target_delta_limit,
                max=self.act_arm_target_delta_limit,
            )
            act_action = torch.cat(
                [arm_state + arm_delta, act_action[:, self.num_actions - 1 :]],
                dim=-1,
            )
        if self.convert_act_action_to_env_action:
            env_action = self._act_action_to_env_action(act_action)
        else:
            env_action = act_action
        self._debug_act_payload = {
            "state": state,
            "state_act_units": state_act_units,
            "state_prepared": state_prepared,
            "fixed": fixed,
            "handeye": handeye,
            "act_action_raw": act_action_raw,
            "act_action": act_action,
            "env_action_prior": env_action,
        }
        return env_action

    def _state_to_act_units(self, state: torch.Tensor) -> torch.Tensor:
        arm_state = state[:, : self.num_actions - 1]
        gripper_state = state[:, self.num_actions - 1 :]
        if self.act_joint_position_unit == "degrees":
            arm_state = torch.rad2deg(arm_state)
            if not self.convert_act_gripper_state_to_percent:
                # Preserve the legacy v4 behavior, which interpreted the gripper
                # joint radians through the same radian-to-degree conversion.
                gripper_state = torch.rad2deg(gripper_state)
        arm_state = arm_state + self.act_arm_joint_position_offset
        if self.convert_act_gripper_state_to_percent:
            gripper_state = gripper_joint_to_percent(
                gripper_state,
                open_joint_pos=self.env_gripper_open_joint_pos,
                closed_joint_pos=self.env_gripper_closed_joint_pos,
            )
        return torch.cat([arm_state, gripper_state], dim=-1)

    def _normalize_act_state(self, state: torch.Tensor) -> torch.Tensor:
        if not self.normalize_act_inputs:
            return state
        return (state - self.act_state_mean) / (self.act_state_std + 1.0e-8)

    def _prepare_act_state(self, state: torch.Tensor) -> torch.Tensor:
        return self._normalize_act_state(self._state_to_act_units(state))

    def _prepare_act_image(self, image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.normalize_act_inputs:
            image = (image - mean) / (std + 1.0e-8)
        return image

    def _act_action_to_env_action(self, act_action: torch.Tensor) -> torch.Tensor:
        arm_target = act_action[:, : self.num_actions - 1] - self.act_arm_joint_position_offset
        gripper_target = act_action[:, self.num_actions - 1 :]
        if self.act_joint_position_unit == "degrees":
            arm_target = torch.deg2rad(arm_target)

        arm_raw_action = (arm_target - self.env_arm_default_joint_pos) / self.env_arm_action_scale
        if self.act_gripper_mapping == "continuous_percent":
            gripper_target_joint = gripper_percent_to_joint(
                gripper_target,
                open_joint_pos=self.env_gripper_open_joint_pos,
                closed_joint_pos=self.env_gripper_closed_joint_pos,
            )
            gripper_raw_action = (
                gripper_target_joint - self.env_gripper_action_offset
            ) / self.env_gripper_action_scale
        else:
            if self.act_gripper_open_below_threshold:
                gripper_is_open = gripper_target < self.act_gripper_open_threshold
            else:
                gripper_is_open = gripper_target >= self.act_gripper_open_threshold
            gripper_raw_action = torch.where(
                gripper_is_open,
                torch.full_like(gripper_target, self.env_gripper_open_action),
                torch.full_like(gripper_target, self.env_gripper_close_action),
            )
        return torch.cat([arm_raw_action, gripper_raw_action], dim=-1)

    def _zero_residual_output_layer(self) -> None:
        if self.residual_actor is None:
            return
        for module in reversed(self.residual_actor):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
                return

    def _write_act_debug(self, action_mean: torch.Tensor, residual: torch.Tensor | None) -> None:
        if not self.debug_act or self._debug_act_calls >= self.debug_act_max_calls:
            return
        payload = self._debug_act_payload
        if payload is None:
            return

        env_index = min(max(self.debug_act_env_index, 0), action_mean.shape[0] - 1)
        self.debug_act_dir.mkdir(parents=True, exist_ok=True)
        if self._debug_act_csv_path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self._debug_act_csv_path = self.debug_act_dir / f"act_io_{stamp}_pid{os.getpid()}.csv"

        row = self._build_debug_row(payload, action_mean, residual, env_index)
        write_header = not self._debug_act_csv_path.exists()
        with self._debug_act_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        if self.debug_act_save_images and self._debug_act_calls < self.debug_act_save_image_calls:
            self._save_debug_images(payload, env_index)

        if self._debug_act_calls == 0:
            print(f"[INFO] ACT debug logging to {self._debug_act_csv_path}")
        self._debug_act_calls += 1

    def _build_debug_row(
        self,
        payload: dict[str, torch.Tensor],
        action_mean: torch.Tensor,
        residual: torch.Tensor | None,
        env_index: int,
    ) -> dict[str, float | int]:
        state = payload["state"].detach()[env_index].float().cpu()
        state_act_units = payload["state_act_units"].detach()[env_index].float().cpu()
        state_prepared = payload["state_prepared"].detach()[env_index].float().cpu()
        fixed = payload["fixed"].detach()[env_index].float().cpu()
        handeye = payload["handeye"].detach()[env_index].float().cpu()
        act_action_raw = payload["act_action_raw"].detach()[env_index].float().cpu()
        act_action = payload["act_action"].detach()[env_index].float().cpu()
        env_action_prior = payload["env_action_prior"].detach()[env_index].float().cpu()
        final_action = action_mean.detach()[env_index].float().cpu()
        if residual is None:
            residual_values = torch.zeros_like(final_action)
        else:
            residual_values = residual.detach()[env_index].float().cpu()

        row: dict[str, float | int] = {
            "call": self._debug_act_calls,
            "batch_size": int(action_mean.shape[0]),
            "env_index": int(env_index),
            "fixed_min": float(fixed.min()),
            "fixed_mean": float(fixed.mean()),
            "fixed_max": float(fixed.max()),
            "handeye_min": float(handeye.min()),
            "handeye_mean": float(handeye.mean()),
            "handeye_max": float(handeye.max()),
        }
        for i, value in enumerate(state.tolist()):
            row[f"state_rad_{i}"] = float(value)
        for i, value in enumerate(state_act_units.tolist()):
            row[f"state_act_units_{i}"] = float(value)
        for i, value in enumerate(state_prepared.tolist()):
            row[f"state_norm_{i}"] = float(value)
        for i, value in enumerate(act_action_raw.tolist()):
            row[f"act_action_raw_{i}"] = float(value)
        for i, value in enumerate(act_action.tolist()):
            row[f"act_action_{i}"] = float(value)
        for i, value in enumerate(env_action_prior.tolist()):
            row[f"env_action_prior_{i}"] = float(value)
        for i, value in enumerate(residual_values.tolist()):
            row[f"residual_{i}"] = float(value)
        for i, value in enumerate(final_action.tolist()):
            row[f"action_mean_{i}"] = float(value)
        return row

    def _save_debug_images(self, payload: dict[str, torch.Tensor], env_index: int) -> None:
        try:
            from torchvision.utils import save_image
        except Exception as exc:
            if self._debug_act_calls == 0:
                print(f"[WARN] Could not import torchvision.utils.save_image for ACT debug images: {exc}")
            return

        for name in ("fixed", "handeye"):
            image = payload[name].detach()[env_index].float().cpu().clamp(0.0, 1.0)
            image_path = self.debug_act_dir / f"act_{name}_call{self._debug_act_calls:04d}_env{env_index}.png"
            save_image(image, image_path)
