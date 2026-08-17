from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

from isaac_so_arm101.policies.act_joint_mapping import gripper_action_to_close_progress

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _goal_metrics(env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3])
    delta = des_pos_w - obj.data.root_pos_w[:, :3]
    xy_dist = torch.norm(delta[:, :2], dim=1)
    z_err = delta[:, 2]
    return xy_dist, z_err


def _get_env_origins(env: ManagerBasedRLEnv, like: torch.Tensor) -> torch.Tensor:
    origins = getattr(env.scene, "env_origins", None)
    if origins is None:
        return torch.zeros((like.shape[0], 3), device=like.device, dtype=like.dtype)
    return origins.to(device=like.device, dtype=like.dtype)


def _object_default_root_pos_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    initial_object_pos: tuple[float, float, float] = (0.2, 0.0, 0.015),
) -> torch.Tensor:
    """Return the per-env object reset/default world position when available."""
    obj: RigidObject = env.scene[object_cfg.name]
    root_pos = obj.data.root_pos_w[:, :3]
    default_root_state = getattr(obj.data, "default_root_state", None)
    if default_root_state is not None:
        default_pos = default_root_state[:, :3].to(device=root_pos.device, dtype=root_pos.dtype)
        # Isaac Lab default root states are typically stored in env-local coordinates.
        # Convert to world coordinates if they are not already offset by env origins.
        env_origins = _get_env_origins(env, root_pos)
        default_pos_with_origins = default_pos + env_origins
        current_env_dist = torch.norm(root_pos[:, :2] - default_pos[:, :2], dim=1).mean()
        current_world_dist = torch.norm(root_pos[:, :2] - default_pos_with_origins[:, :2], dim=1).mean()
        return torch.where(
            (current_env_dist < current_world_dist).view(1, 1),
            default_pos,
            default_pos_with_origins,
        )

    env_origins = _get_env_origins(env, root_pos)
    initial_pos = torch.tensor(initial_object_pos, device=root_pos.device, dtype=root_pos.dtype).view(1, 3)
    return initial_pos + env_origins


def _object_episode_initial_root_pos_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Cache the object's actual per-episode reset position for height gain and push diagnostics."""
    obj: RigidObject = env.scene[object_cfg.name]
    root_pos = obj.data.root_pos_w[:, :3]
    cache_name = "_so101_pick_place_initial_object_pos_w"
    cached_pos = getattr(env, cache_name, None)
    if cached_pos is None or cached_pos.shape != root_pos.shape or cached_pos.device != root_pos.device:
        cached_pos = _object_default_root_pos_w(env, object_cfg).clone()

    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is None:
        reset_mask = torch.zeros((root_pos.shape[0], 1), device=root_pos.device, dtype=torch.bool)
    else:
        reset_mask = (episode_length_buf.to(device=root_pos.device) <= 1).view(-1, 1)
    cached_pos = torch.where(reset_mask, root_pos.detach(), cached_pos.to(device=root_pos.device, dtype=root_pos.dtype))
    setattr(env, cache_name, cached_pos)
    return cached_pos


def _cube_was_lifted_this_episode(
    env: ManagerBasedRLEnv,
    lift_threshold: float = 0.045,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Per-env latch: 1.0 once the cube has risen above `lift_threshold` at any step this episode.

    Reset to 0 at episode start (episode_length_buf <= 1). Used to gate the stage-4 placement
    rewards: with the goal lowered to table height, a cube merely dragged/pushed along the table to
    the goal XY (never lifted) could otherwise collect the placement rewards, competing with the
    grasp-lift-carry path. Multiplying these rewards by this latch makes "place" payable only after a
    genuine lift, so dragging the cube to the goal earns nothing.

    The latch only grows within a step, so it is safe to call from several stage-4 terms per step.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    cache_name = "_so101_pick_place_was_lifted"
    latch = getattr(env, cache_name, None)
    if latch is None or latch.shape[0] != z.shape[0] or latch.device != z.device:
        latch = torch.zeros_like(z, dtype=torch.bool)
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is not None:
        reset_mask = episode_length_buf.to(device=z.device) <= 1
        latch = latch & ~reset_mask
    latch = latch | (z > lift_threshold)
    setattr(env, cache_name, latch)
    return latch.float()


def _object_height_gain(
    env: ManagerBasedRLEnv,
    initial_object_z: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    initial_pos_w = _object_episode_initial_root_pos_w(env, object_cfg)
    fallback_initial_z = _object_default_root_pos_w(env, object_cfg, (0.2, 0.0, initial_object_z))[:, 2]
    initial_z = torch.where(torch.isfinite(initial_pos_w[:, 2]), initial_pos_w[:, 2], fallback_initial_z)
    return obj.data.root_pos_w[:, 2] - initial_z


def _ee_object_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :3]
    return torch.norm(obj.data.root_pos_w[:, :3] - ee_w, dim=1)


def _get_gripper_joint_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_idx = robot.find_joints("gripper")[0][0]
    return robot.data.joint_pos[:, gripper_idx]


def _gripper_open_ratio(
    env: ManagerBasedRLEnv,
    open_joint_pos: float,
    close_joint_pos: float,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Estimate gripper opening ratio in [0, 1] with a configurable joint range.

    Assumption for SO101 in this repo: larger joint_pos means more open.
    If hardware mapping differs, only this helper needs to be adjusted.
    """
    joint_pos = _get_gripper_joint_pos(env, robot_cfg)
    denom = max(open_joint_pos - close_joint_pos, 1e-3)
    return torch.clamp((joint_pos - close_joint_pos) / denom, 0.0, 1.0)


def _gripper_close_command(
    env: ManagerBasedRLEnv,
    open_action: float | None = None,
    close_action: float | None = None,
) -> torch.Tensor:
    """Return gripper command closure progress.

    Without explicit endpoints this preserves the legacy binary contract where
    negative means close. V36 passes 1=open and 0=closed so intermediate PPO
    actions provide a dense closure signal instead of being misread as open.
    """
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None or action_manager.action is None or action_manager.action.shape[1] == 0:
        obj_like = env.scene["object"].data.root_pos_w[:, 0]
        return torch.zeros_like(obj_like)
    gripper_cmd = action_manager.action[:, -1]
    if open_action is None and close_action is None:
        return (gripper_cmd < 0.0).float()
    if open_action is None or close_action is None:
        raise ValueError("open_action and close_action must be provided together")
    return gripper_action_to_close_progress(
        gripper_cmd,
        open_action=open_action,
        close_action=close_action,
    )


def _get_wrist_flex_joint_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return wrist flex joint position.

    Priority: `wrist_flex`, then `wrist_.*` fallback for compatibility.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    try:
        wrist_idx = robot.find_joints("wrist_flex")[0][0]
    except Exception:
        # Fallback if the exact joint name differs in some robot variants.
        wrist_idx = robot.find_joints("wrist_.*")[0][0]
    return robot.data.joint_pos[:, wrist_idx]


def _gates(env: ManagerBasedRLEnv, lift_height: float, near_goal_xy: float, release_height: float, command_name: str,
           robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg):
    obj: RigidObject = env.scene[object_cfg.name]
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    z = obj.data.root_pos_w[:, 2]
    is_lifted = z > lift_height
    near_goal = xy_dist < near_goal_xy
    low_height = z < release_height
    s1 = (~is_lifted).float()
    s2 = (is_lifted & ~near_goal).float()
    s3 = (is_lifted & near_goal & ~low_height).float()
    s4 = (is_lifted & near_goal & low_height).float()
    return s1, s2, s3, s4


def stage1_close_when_near_object_reward(
    env: ManagerBasedRLEnv,
    near_distance: float,
    open_joint_pos: float,
    close_joint_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closing the gripper only after the end-effector reaches the object."""
    ee_obj_dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    near_object = (ee_obj_dist < near_distance).float()
    gripper_closed = 1.0 - _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)
    return near_object * gripper_closed


def stage1_open_when_near_object_penalty(
    env: ManagerBasedRLEnv,
    near_distance: float,
    open_joint_pos: float,
    close_joint_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalize staying open at grasp distance, which usually leads to pushing."""
    ee_obj_dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    near_object = (ee_obj_dist < near_distance).float()
    gripper_open = _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)
    return near_object * gripper_open


def object_lifted_from_initial_reward(
    env: ManagerBasedRLEnv,
    lift_cap: float,
    min_height_gain: float = 0.0,
    initial_object_z: float = 0.012,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense lift reward: proportional to upward displacement from the true resting height.

    Uses a FIXED reference height (`initial_object_z`, the cube's deterministic settled z ~0.012)
    instead of the per-episode cached snapshot used by `_object_height_gain`. The snapshot is
    captured at episode_length_buf <= 1, i.e. before the cube finishes settling, so it sat ~3mm
    above the true rest height and created a dead zone where the cube had to climb back ~3mm
    before any reward appeared. The cube's rest z is the same every episode (flat table, z not
    randomised), so a fixed reference is both correct and free of that transient.

    Dense (not a sparse gate): rewards any upward displacement from the first millimetre, so the
    policy can climb out of the "false grasp" plateau by gradient instead of crossing a discontinuity.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    height_gain = obj.data.root_pos_w[:, 2] - initial_object_z
    shaped = torch.clamp(height_gain - min_height_gain, min=0.0)
    return torch.clamp(shaped / lift_cap, 0.0, 1.0)


def lift_crossing_bonus(
    env: ManagerBasedRLEnv,
    lift_height: float = 0.045,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """One-time, non-farmable lift bonus: fires 1.0 only on the single step the cube first rises
    above `lift_height` this episode, and 0.0 on every other step.

    Replaces the dense `object_lifted_from_initial_reward` and the continuous `object_is_lifted`
    term, both of which paid every step the cube was airborne and were therefore farmable by
    lifting-and-hovering at the pickup point (the failure seen in the v14 warm-start: cube lifted
    but never transported, grasp degraded, placement ~0). By paying only on the rising edge,
    holding the cube up earns nothing more -- the only way to keep earning is to carry it toward the
    goal (stage2 transport) and place it (stage3/4). The 4.5cm threshold matches the is_lifted gate
    used everywhere else.

    Uses its own cache key (separate from the `_was_lifted` gating latch) so the two never interfere.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    cache_name = "_so101_pick_place_lift_bonus_given"
    given = getattr(env, cache_name, None)
    if given is None or given.shape[0] != z.shape[0] or given.device != z.device:
        given = torch.zeros_like(z, dtype=torch.bool)
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is not None:
        reset_mask = episode_length_buf.to(device=z.device) <= 1
        given = given & ~reset_mask
    is_above = z > lift_height
    newly = is_above & ~given  # rising edge: above the threshold now, bonus not yet paid this episode
    given = given | is_above
    setattr(env, cache_name, given)
    return newly.float()


def lifted_close_hold_reward(
    env: ManagerBasedRLEnv,
    min_height_gain: float,
    near_distance: float,
    open_joint_pos: float,
    close_joint_pos: float,
    initial_object_z: float = 0.015,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward maintaining a close, closed grasp after the cube leaves the table."""
    height_gain = _object_height_gain(env, initial_object_z, object_cfg)
    lifted = (height_gain > min_height_gain).float()
    ee_obj_dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    near_object = (ee_obj_dist < near_distance).float()
    gripper_closed = 1.0 - _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)
    return lifted * near_object * gripper_closed


def push_without_lift_penalty(
    env: ManagerBasedRLEnv,
    move_xy_threshold: float,
    min_height_gain: float,
    initial_object_z: float = 0.015,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize large horizontal cube displacement when the cube has not been lifted."""
    obj: RigidObject = env.scene[object_cfg.name]
    initial_pos_w = _object_episode_initial_root_pos_w(env, object_cfg)
    move_xy = torch.norm(obj.data.root_pos_w[:, :2] - initial_pos_w[:, :2], dim=1)
    height_gain = obj.data.root_pos_w[:, 2] - initial_pos_w[:, 2]
    pushed = move_xy > move_xy_threshold
    not_lifted = height_gain < min_height_gain
    return (pushed & not_lifted).float()


def push_never_lifted_penalty(
    env: ManagerBasedRLEnv,
    move_xy_threshold: float,
    lift_height: float = 0.045,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize horizontal cube displacement while the cube has NEVER been lifted this episode.

    Targets the two dominant v9 failure modes measured in the GT-heavy funnel (599 eps):
    (a) the push-chase death spiral -- 80% of grasp failures close the gripper at the cube but skate
    it a median 9.8cm across the table and run out of episode time chasing it; (b) the right-side
    drag-to-goal shortcut -- cubes spawning at y<-0.07 (the goal side) grasp at only 3-9% because the
    pre-latch v9 reward landscape paid for dragging the cube toward the goal without lifting.
    Both behaviours displace the never-lifted cube far beyond the ~2-3cm nudge of a clean grasp, so a
    displacement penalty prices them without touching normal contact.

    Unlike `push_without_lift_penalty` (instantaneous height check), this gates on the
    `_cube_was_lifted_this_episode` latch: after the first genuine lift the penalty is dead for the
    rest of the episode, so a cube carried to the goal and PLACED back on the table (large
    displacement, low height) is not penalized -- otherwise this term would fight the stage-4
    placement rewards every step after release.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    initial_pos_w = _object_episode_initial_root_pos_w(env, object_cfg)
    move_xy = torch.norm(obj.data.root_pos_w[:, :2] - initial_pos_w[:, :2], dim=1)
    pushed = (move_xy > move_xy_threshold).float()
    never_lifted = 1.0 - _cube_was_lifted_this_episode(env, lift_height, object_cfg)
    return pushed * never_lifted


def approach_gripper_open_reward(
    env: ManagerBasedRLEnv,
    contact_distance: float,
    approach_outer: float,
    lift_height: float,
    open_action: float | None = None,
    close_action: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward commanding the gripper OPEN while approaching the not-yet-lifted cube.

    Teaches the correct grasp timing: approach with an open jaw so the cube can enter between
    the fingers. Active only in the approach band (contact_distance <= ee-object distance <
    approach_outer) and while the cube is still on the table. Mutually exclusive with
    `grasp_close_at_contact_reward`, which takes over once the fingertips reach the cube.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_obj_dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    not_lifted = (obj.data.root_pos_w[:, 2] < lift_height).float()
    approaching = ((ee_obj_dist >= contact_distance) & (ee_obj_dist < approach_outer)).float()
    open_command = 1.0 - _gripper_close_command(env, open_action, close_action)
    return not_lifted * approaching * open_command


def grasp_close_at_contact_reward(
    env: ManagerBasedRLEnv,
    contact_distance: float,
    open_action: float | None = None,
    close_action: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward commanding the gripper CLOSE only once the fingertips reach the cube.

    The contact condition is geometric (ee-object distance < contact_distance) because contact
    sensors are disabled on this asset. Rewards the close *command* (not the joint angle), so a
    fat cube that holds the jaw partly open while grasped does not zero the signal.
    """
    ee_obj_dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    at_contact = (ee_obj_dist < contact_distance).float()
    close_command = _gripper_close_command(env, open_action, close_action)
    return at_contact * close_command


def grasp_open_at_contact_penalty(
    env: ManagerBasedRLEnv,
    contact_distance: float,
    open_action: float | None = None,
    close_action: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalize keeping the gripper OPEN once the end-effector reaches the grasp zone."""
    ee_obj_dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    at_contact = (ee_obj_dist < contact_distance).float()
    open_command = 1.0 - _gripper_close_command(env, open_action, close_action)
    return at_contact * open_command


def closed_near_upward_motion_reward(
    env: ManagerBasedRLEnv,
    contact_distance: float,
    target_up_delta: float,
    lift_height: float,
    open_action: float | None = None,
    close_action: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward lifting intent after a close command near the cube, before the cube is airborne."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = ee_frame.data.target_pos_w[..., 0, 2]

    cache_name = "_so101_pick_place_prev_ee_z"
    prev_ee_z = getattr(env, cache_name, None)
    if prev_ee_z is None or prev_ee_z.shape != ee_z.shape or prev_ee_z.device != ee_z.device:
        prev_ee_z = ee_z.detach()
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is not None:
        reset_mask = episode_length_buf.to(device=ee_z.device) <= 1
        prev_ee_z = torch.where(reset_mask, ee_z.detach(), prev_ee_z.to(device=ee_z.device, dtype=ee_z.dtype))

    up_delta = ee_z - prev_ee_z
    setattr(env, cache_name, ee_z.detach())

    ee_obj_dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    near_object = (ee_obj_dist < contact_distance).float()
    close_command = _gripper_close_command(env, open_action, close_action)
    not_lifted = (obj.data.root_pos_w[:, 2] < lift_height).float()
    upward = torch.clamp(up_delta / max(target_up_delta, 1.0e-6), 0.0, 1.0)
    return near_object * close_command * not_lifted * upward


def closed_contact_lift_pose_reward(
    env: ManagerBasedRLEnv,
    contact_distance: float,
    target_ee_above_object: float,
    lift_height: float,
    open_action: float | None = None,
    close_action: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward a closed contact pose that raises the gripper above the cube before lift.

    The v28 policy learned contact and close commands but often lost the lift before the cube moved
    enough for object-height rewards to activate. This term gives a continuous bridge: while the
    cube is still on the table, reward close commands near the cube as the end-effector rises above
    the cube center.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = ee_frame.data.target_pos_w[..., 0, 2]
    obj_z = obj.data.root_pos_w[:, 2]

    ee_obj_dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    near_object = (ee_obj_dist < contact_distance).float()
    close_command = _gripper_close_command(env, open_action, close_action)
    not_lifted = (obj_z < lift_height).float()
    ee_above_object = torch.clamp((ee_z - obj_z) / max(target_ee_above_object, 1.0e-6), 0.0, 1.0)
    return near_object * close_command * not_lifted * ee_above_object


def closed_contact_object_rise_reward(
    env: ManagerBasedRLEnv,
    contact_distance: float,
    lift_cap: float,
    lift_height: float,
    min_height_gain: float = 0.0,
    initial_object_z: float = 0.015,
    open_action: float | None = None,
    close_action: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward real object rise while the gripper is commanded closed at the grasp zone.

    v30 could farm a closed, above-cube end-effector pose without moving the cube. This term only
    pays when the cube center actually rises from the table, and gates that rise by the same
    close-at-contact condition used by the ACT bootstrap rewards.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_obj_dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    near_object = (ee_obj_dist < contact_distance).float()
    close_command = _gripper_close_command(env, open_action, close_action)
    not_lifted = (obj.data.root_pos_w[:, 2] < lift_height).float()
    height_gain = obj.data.root_pos_w[:, 2] - initial_object_z
    shaped_rise = torch.clamp((height_gain - min_height_gain) / max(lift_cap, 1.0e-6), 0.0, 1.0)
    return near_object * close_command * not_lifted * shaped_rise


def diag_object_height_gain_m(
    env: ManagerBasedRLEnv,
    initial_object_z: float = 0.015,
    height_cap: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Diagnostic: cube center height gain in meters, clipped to a useful TensorBoard range."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.clamp(obj.data.root_pos_w[:, 2] - initial_object_z, 0.0, height_cap)


def diag_gripper_closed_joint_ratio(
    env: ManagerBasedRLEnv,
    open_joint_pos: float,
    close_joint_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Diagnostic: achieved gripper closure from joint position, 1.0=closed and 0.0=open."""
    return 1.0 - _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)


def diag_gripper_close_action_progress(
    env: ManagerBasedRLEnv,
    open_action: float,
    close_action: float,
) -> torch.Tensor:
    """Diagnostic: commanded closure progress, 1.0=closed and 0.0=open."""
    return _gripper_close_command(env, open_action, close_action)


def diag_near_object(
    env: ManagerBasedRLEnv,
    contact_distance: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Diagnostic: 1.0 when the end-effector frame is inside the geometric grasp zone."""
    return (_ee_object_distance(env, object_cfg, ee_frame_cfg) < contact_distance).float()


def lifted_close_hold_cmd_reward(
    env: ManagerBasedRLEnv,
    min_height_gain: float,
    near_distance: float,
    initial_object_z: float = 0.015,
    open_action: float | None = None,
    close_action: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Command-based replacement for `lifted_close_hold_reward`.

    Rewards keeping the close *command* while the lifted cube stays near the end-effector. Robust
    to the gripped joint angle (a fat cube holds the jaw at ~0.85 rad, which the joint-angle metric
    would misread as open and zero this term).
    """
    height_gain = _object_height_gain(env, initial_object_z, object_cfg)
    lifted = (height_gain > min_height_gain).float()
    ee_obj_dist = _ee_object_distance(env, object_cfg, ee_frame_cfg)
    near_object = (ee_obj_dist < near_distance).float()
    close_command = _gripper_close_command(env, open_action, close_action)
    return lifted * near_object * close_command


def stage2_goal_xy_tracking_gated(env: ManagerBasedRLEnv, std: float, lift_height: float, near_goal_xy: float,
                                  release_height: float, command_name: str,
                                  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                  object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    _, s2, _, _ = _gates(env, lift_height, near_goal_xy, release_height, command_name, robot_cfg, object_cfg)
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    return s2 * (1.0 - torch.tanh(xy_dist / std))


def stage2_early_open_penalty_gated(
    env: ManagerBasedRLEnv,
    open_joint_pos: float,
    close_joint_pos: float,
    lift_height: float,
    near_goal_xy: float,
    release_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    _, s2, _, _ = _gates(
        env,
        lift_height,
        near_goal_xy,
        release_height,
        command_name,
        SceneEntityCfg("robot"),
        SceneEntityCfg("object"),
    )
    gripper_open = _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)
    return s2 * gripper_open


def stage2_early_open_penalty_cmd_gated(
    env: ManagerBasedRLEnv,
    lift_height: float,
    near_goal_xy: float,
    release_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Command-based stage-2 early-open penalty: penalize commanding OPEN during transport.

    Robust to the gripped joint angle, so a correctly-held fat cube (jaw at ~0.85 rad) is not
    falsely penalized as "open" while being carried.
    """
    _, s2, _, _ = _gates(env, lift_height, near_goal_xy, release_height, command_name, robot_cfg, object_cfg)
    open_command = 1.0 - _gripper_close_command(env)
    return s2 * open_command


def stage3_soft_descent_reward_gated(env: ManagerBasedRLEnv, target_speed: float, lift_height: float, near_goal_xy: float,
                                     release_height: float, command_name: str,
                                     object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    _, _, s3, _ = _gates(env, lift_height, near_goal_xy, release_height, command_name, SceneEntityCfg("robot"), object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    vz = obj.data.root_lin_vel_w[:, 2]
    return s3 * torch.exp(-torch.square(vz + target_speed) / (2 * target_speed * target_speed + 1e-6))


def stage3_hard_drop_penalty_gated(env: ManagerBasedRLEnv, max_down_speed: float, lift_height: float, near_goal_xy: float,
                                   release_height: float, command_name: str,
                                   object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    _, _, s3, _ = _gates(env, lift_height, near_goal_xy, release_height, command_name, SceneEntityCfg("robot"), object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    vz = obj.data.root_lin_vel_w[:, 2]
    return s3 * torch.clamp(-(vz + max_down_speed), min=0.0)


def stage3_ee_low_near_goal_gated(
    env: ManagerBasedRLEnv, near_goal_xy: float, target_ee_height: float, ee_height_std: float, command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    near_goal = (xy_dist < near_goal_xy).float()
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_height = ee_frame.data.target_pos_w[..., 0, 2]
    # Lock behind a real lift this episode: otherwise this near-goal EE-low reward is obtainable by
    # dragging the cube to the goal XY on the table (no lift), draining grasp-lift bootstrap pressure.
    was_lifted = _cube_was_lifted_this_episode(env, 0.045, object_cfg)
    return near_goal * torch.exp(-torch.square(ee_height - target_ee_height) / (2 * ee_height_std * ee_height_std + 1e-6)) * was_lifted


def stage3_object_height_near_table_gated(
    env: ManagerBasedRLEnv, near_goal_xy: float, table_height: float, table_margin: float, command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    near_goal = (xy_dist < near_goal_xy).float()
    obj: RigidObject = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    # Lock behind a real lift: this term rewards the cube AT table height near the goal, i.e. it
    # would directly reward a cube dragged along the table to the goal. Require a genuine lift first.
    was_lifted = _cube_was_lifted_this_episode(env, 0.045, object_cfg)
    return near_goal * torch.exp(-torch.square(z - table_height) / (2 * table_margin * table_margin + 1e-6)) * was_lifted


def stage3_wrist_flex_release_pose_gated(
    env: ManagerBasedRLEnv, near_goal_xy: float, wrist_target_pos: float, wrist_std: float, command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    near_goal = (xy_dist < near_goal_xy).float()
    wrist_pos = _get_wrist_flex_joint_pos(env, robot_cfg)
    # Lock behind a real lift (consistent with the other stage-3 near-goal terms).
    was_lifted = _cube_was_lifted_this_episode(env, 0.045, object_cfg)
    return near_goal * torch.exp(-torch.square(wrist_pos - wrist_target_pos) / (2 * wrist_std * wrist_std + 1e-6)) * was_lifted


def stage4_release_reward_gated(
    env: ManagerBasedRLEnv,
    open_joint_pos: float,
    close_joint_pos: float,
    lift_height: float,
    near_goal_xy: float,
    release_height: float,
    table_height: float,
    table_margin: float,
    ee_low_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    near_goal = xy_dist < near_goal_xy
    obj_low = torch.abs(obj.data.root_pos_w[:, 2] - table_height) < table_margin
    ee_low = ee_frame.data.target_pos_w[..., 0, 2] < ee_low_height
    # Release gate: cube settled near the table (obj_low) at the goal XY with the arm lowered.
    # The old gate also required `lifted_enough (z>lift_height=0.045)` AND `release_band`, which
    # contradicts obj_low (z<table_height+table_margin=0.05): a placed cube rests at z~0.02, so it
    # is obj_low but NOT lifted_enough. The intersection was a ~5mm band [0.045,0.05] -> this reward
    # essentially never fired. Dropped both so release can be rewarded once the cube is on the table.
    gate = (near_goal & obj_low & ee_low).float()
    # Only payable if the cube was genuinely lifted earlier this episode (anti-drag latch).
    was_lifted = _cube_was_lifted_this_episode(env, lift_height, object_cfg)
    return gate * _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg) * was_lifted


def stage4_hold_too_long_penalty_gated(
    env: ManagerBasedRLEnv,
    open_joint_pos: float,
    close_joint_pos: float,
    lift_height: float,
    near_goal_xy: float,
    release_height: float,
    table_height: float,
    table_margin: float,
    ee_low_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    near_goal = xy_dist < near_goal_xy
    obj_low = torch.abs(obj.data.root_pos_w[:, 2] - table_height) < table_margin
    ee_low = ee_frame.data.target_pos_w[..., 0, 2] < ee_low_height
    # See stage4_release_reward_gated: the old lifted_enough & release_band made this ~5mm-band gate
    # essentially never fire. Use the same non-degenerate "settled on table at goal" gate.
    s4 = (near_goal & obj_low & ee_low).float()
    hold_close = 1.0 - _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)
    return s4 * hold_close


def stage4_gripper_open_near_table_gated(env: ManagerBasedRLEnv, open_joint_pos: float, close_joint_pos: float,
                                         near_goal_xy: float, table_height: float,
                                         table_margin: float, ee_low_height: float, command_name: str,
                                         robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                         object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
                                         ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    near_goal = xy_dist < near_goal_xy
    near_table = torch.abs(obj.data.root_pos_w[:, 2] - table_height) < table_margin
    ee_low = ee_frame.data.target_pos_w[..., 0, 2] < ee_low_height

    gripper_open = _gripper_open_ratio(env, open_joint_pos, close_joint_pos, robot_cfg)

    was_lifted = _cube_was_lifted_this_episode(env, 0.045, object_cfg)
    return (near_goal & near_table & ee_low).float() * gripper_open * was_lifted


def stage4_stable_placed_reward_gated(env: ManagerBasedRLEnv, xy_threshold: float, table_height: float, speed_threshold: float,
                                      command_name: str,
                                      robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                      object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    speed = torch.norm(obj.data.root_lin_vel_w[:, :3], dim=1)
    good_xy = xy_dist < xy_threshold
    near_table = torch.abs(z - table_height) < 0.015
    low_speed = speed < speed_threshold
    was_lifted = _cube_was_lifted_this_episode(env, 0.045, object_cfg)
    return (good_xy & near_table & low_speed).float() * was_lifted


def stage4_ee_away_after_place_gated(env: ManagerBasedRLEnv, ee_min_distance: float, xy_threshold: float, table_height: float,
                                     speed_threshold: float, command_name: str,
                                     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                                     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
                                     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    placed = stage4_stable_placed_reward_gated(env, xy_threshold, table_height, speed_threshold, command_name, robot_cfg, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(obj.data.root_pos_w[:, :3] - ee_w, dim=1)
    return placed * torch.clamp((dist - ee_min_distance) / max(ee_min_distance, 1e-3), min=0.0)


def stage4_release_reward_cmd_gated(
    env: ManagerBasedRLEnv,
    lift_height: float,
    near_goal_xy: float,
    release_height: float,
    table_height: float,
    table_margin: float,
    ee_low_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Command-based stage-4 release reward: reward commanding OPEN at the release pose.

    Same geometric gate as `stage4_release_reward_gated`, but rewards the open *command* instead
    of the joint-angle open ratio, so a still-gripped fat cube is not falsely read as released.
    """
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    near_goal = xy_dist < near_goal_xy
    obj_low = torch.abs(obj.data.root_pos_w[:, 2] - table_height) < table_margin
    ee_low = ee_frame.data.target_pos_w[..., 0, 2] < ee_low_height
    # See stage4_release_reward_gated: drop the degenerate lifted_enough & release_band band.
    gate = (near_goal & obj_low & ee_low).float()
    open_command = 1.0 - _gripper_close_command(env)
    return gate * open_command


def stage4_hold_too_long_penalty_cmd_gated(
    env: ManagerBasedRLEnv,
    lift_height: float,
    near_goal_xy: float,
    release_height: float,
    table_height: float,
    table_margin: float,
    ee_low_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Command-based stage-4 hold-too-long penalty: penalize still commanding CLOSE at release pose."""
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    near_goal = xy_dist < near_goal_xy
    obj_low = torch.abs(obj.data.root_pos_w[:, 2] - table_height) < table_margin
    ee_low = ee_frame.data.target_pos_w[..., 0, 2] < ee_low_height
    # See stage4_release_reward_gated: drop the degenerate lifted_enough & release_band band.
    s4 = (near_goal & obj_low & ee_low).float()
    close_command = _gripper_close_command(env)
    return s4 * close_command


def stage4_gripper_open_near_table_cmd_gated(
    env: ManagerBasedRLEnv,
    near_goal_xy: float,
    table_height: float,
    table_margin: float,
    ee_low_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Command-based stage-4 open-near-table reward: reward commanding OPEN once low near the goal."""
    xy_dist, _ = _goal_metrics(env, command_name, robot_cfg, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    near_goal = xy_dist < near_goal_xy
    near_table = torch.abs(obj.data.root_pos_w[:, 2] - table_height) < table_margin
    ee_low = ee_frame.data.target_pos_w[..., 0, 2] < ee_low_height
    open_command = 1.0 - _gripper_close_command(env)
    return (near_goal & near_table & ee_low).float() * open_command
