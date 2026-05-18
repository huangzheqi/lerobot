from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaac_so_arm101.tasks.lift.mdp as lift_mdp
from isaac_so_arm101.tasks.pick_place.joint_pos_env_cfg import SoArm101PickPlaceCubeEnvCfg_PLAY


@configclass
class SoArm101PickPlaceCubeEnvCfg_RobustPlay(SoArm101PickPlaceCubeEnvCfg_PLAY):
    """Pick-place play config for robustness stress tests.

    This config is opt-in and does not touch existing training/play configs.
    """

    def __post_init__(self):
        super().__post_init__()
        # Keep default values close to original play cfg.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5


def _level_scale(level: str) -> float:
    level_map = {
        "off": 0.0,
        "low": 0.35,
        "medium": 0.7,
        "high": 1.0,
    }
    if level not in level_map:
        raise ValueError(f"Unsupported disturbance level: {level}")
    return level_map[level]


def _friction_by_level(level: str) -> tuple[float, float]:
    """Return (static_friction, dynamic_friction) for low/medium/high.

    Current values:
    - low:    static=0.6375, dynamic=0.5725
    - medium: static=0.3750, dynamic=0.3450
    - high:   static=0.1500, dynamic=0.1500
    """
    scale = _level_scale(level)
    dynamic = max(0.05, 0.8 - 0.65 * scale)
    static = max(0.05, 0.9 - 0.75 * scale)
    return static, dynamic


def apply_pick_place_disturbance(env_cfg, disturbance_type: str, level: str):
    """Apply disturbances for robustness evaluation.

    disturbance_type: one of
    - cube_init
    - goal
    - table_friction (legacy alias; global contact friction disturbance)
    - global_contact_friction (explicit name for global contact friction disturbance)
    - table_surface_friction (table-only material override for tabletop-contact tests)
    - clutter
    - lighting
    - camera
    - all
    """
    scale = _level_scale(level)
    if scale == 0.0:
        return env_cfg

    if disturbance_type in ("cube_init", "all"):
        amp_x = 0.05 * scale
        amp_y = 0.07 * scale
        env_cfg.events.reset_object_position = EventTerm(
            func=lift_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-amp_x, amp_x), "y": (-amp_y, amp_y), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )

    if disturbance_type in ("goal", "all"):
        # Goal sampling is constrained to a reachable tabletop workspace for robust-play only.
        # Design notes:
        # - Keep x positive to avoid sampling behind the robot base.
        # - Use symmetric y ranges so both left/right directions are evaluated.
        # - Leave margins from table edges and a base exclusion zone near y ~= 0.
        # - Expand low->medium->high monotonically, while keeping high reachable.
        goal_ranges_by_level = {
            "low": {
                "pos_x": (0.14, 0.20),
                "pos_y": (-0.15, 0.15),
                "pos_z": (0.20, 0.29),
            },
            "medium": {
                "pos_x": (0.13, 0.22),
                "pos_y": (-0.19, 0.19),
                "pos_z": (0.20, 0.31),
            },
            "high": {
                "pos_x": (0.12, 0.24),
                "pos_y": (-0.22, 0.22),
                "pos_z": (0.20, 0.33),
            },
        }
        ranges = goal_ranges_by_level[level]
        env_cfg.commands.object_pose.ranges.pos_x = ranges["pos_x"]
        env_cfg.commands.object_pose.ranges.pos_y = ranges["pos_y"]
        env_cfg.commands.object_pose.ranges.pos_z = ranges["pos_z"]

    if disturbance_type in ("table_friction", "global_contact_friction", "all"):
        # NOTE: This is a GLOBAL contact friction disturbance.
        # It overrides env_cfg.sim.physics_material (global default), therefore it can affect
        # cube-table, gripper-cube and gripper-table contacts simultaneously.
        static, dynamic = _friction_by_level(level)
        env_cfg.sim.physics_material = RigidBodyMaterialCfg(
            static_friction=static,
            dynamic_friction=dynamic,
            restitution=0.0,
        )

    if disturbance_type == "table_surface_friction":
        # NOTE: We only override table material here so gripper-cube contact is not directly changed.
        # API limitation: if other assets do not provide explicit per-body material, physics engines can
        # still combine table material with counterpart/default material, so full isolation is not guaranteed.
        static, dynamic = _friction_by_level(level)
        env_cfg.scene.table.spawn.physics_material = RigidBodyMaterialCfg(
            static_friction=static,
            dynamic_friction=dynamic,
            restitution=0.0,
        )

    if disturbance_type in ("clutter", "all"):
        offset = 0.03 * scale
        env_cfg.scene.clutter_block_1 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ClutterBlock1",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.36, 0.14 + offset, 0.02], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.25, 0.25, 0.25),
            ),
        )
        env_cfg.scene.clutter_block_2 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ClutterBlock2",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.52, -0.15 - offset, 0.02], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.2, 0.2, 0.2),
            ),
        )

    if disturbance_type in ("lighting", "all"):
        intensity = 2500.0 + 3500.0 * scale
        env_cfg.scene.light.spawn = sim_utils.DistantLightCfg(intensity=intensity, color=(0.85, 0.85, 0.9), angle=8.0)

    if disturbance_type in ("camera", "all"):
        eye_offset = 0.8 * scale
        env_cfg.viewer.eye = (2.5 + eye_offset, 2.5 - eye_offset, 1.5 + 0.4 * scale)
        env_cfg.viewer.lookat = (0.4, 0.0, 0.2)

    return env_cfg
