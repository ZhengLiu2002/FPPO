# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common constraint terms for CMDP-style training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _warn_once(env: ManagerBasedRLEnv, flag_name: str, message: str) -> None:
    if getattr(env, flag_name, False):
        return
    setattr(env, flag_name, True)
    print(f"[WARN] {message}")


def _get_joint_slice(asset_cfg: SceneEntityCfg | None) -> slice | list[int]:
    if asset_cfg is None or asset_cfg.joint_ids is None:
        return slice(None)
    return asset_cfg.joint_ids


def _zeros_like_env(env: ManagerBasedRLEnv, dtype: torch.dtype | None = None) -> torch.Tensor:
    device = getattr(env, "device", torch.device("cpu"))
    return torch.zeros(env.num_envs, device=device, dtype=dtype or torch.float32)


def _normalize_cost(cost: torch.Tensor, limit: float | None) -> torch.Tensor:
    if limit is None:
        return cost
    limit_t = torch.as_tensor(limit, device=cost.device, dtype=cost.dtype)
    limit_t = torch.abs(limit_t)
    eps = torch.finfo(cost.dtype).eps if torch.is_floating_point(cost) else 1e-6
    limit_t = torch.clamp(limit_t, min=eps)
    return cost / limit_t


def _resolve_gait_frequency(
    env: ManagerBasedRLEnv,
    command_name: str | None,
    base_frequency: float,
    min_frequency: float | None,
    max_frequency: float | None,
    max_command_speed: float | None,
    frequency_scale: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    base_t = torch.as_tensor(base_frequency, device=device, dtype=dtype)
    if command_name is None or not hasattr(env, "command_manager"):
        return base_t.expand(env.num_envs)
    commands = env.command_manager.get_command(command_name)
    if commands is None:
        return base_t.expand(env.num_envs)
    speed = torch.linalg.norm(commands[:, :2], dim=1)

    if max_command_speed is not None and max_command_speed > 0.0:
        min_f = base_t if min_frequency is None else torch.as_tensor(min_frequency, device=device, dtype=dtype)
        max_f = base_t if max_frequency is None else torch.as_tensor(max_frequency, device=device, dtype=dtype)
        max_speed_t = torch.as_tensor(max_command_speed, device=device, dtype=dtype)
        ratio = torch.clamp(speed / max_speed_t, 0.0, 1.0)
        freq = min_f + (max_f - min_f) * ratio
    else:
        scale_t = torch.as_tensor(frequency_scale, device=device, dtype=dtype)
        freq = base_t + speed * scale_t

    if min_frequency is not None or max_frequency is not None:
        min_f = base_t if min_frequency is None else torch.as_tensor(min_frequency, device=device, dtype=dtype)
        max_f = base_t if max_frequency is None else torch.as_tensor(max_frequency, device=device, dtype=dtype)
        freq = torch.clamp(freq, min=min_f, max=max_f)

    return freq


def _resolve_command_speed(
    env: ManagerBasedRLEnv,
    command_name: str | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if command_name is None or not hasattr(env, "command_manager"):
        return None
    commands = env.command_manager.get_command(command_name)
    if commands is None:
        return None
    lin = torch.linalg.norm(commands[:, :2], dim=1)
    yaw = torch.abs(commands[:, 2]) if commands.shape[1] > 2 else torch.zeros_like(lin)
    return lin + yaw


def _resolve_base_speed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    if not hasattr(env, "scene"):
        return None
    if not hasattr(asset_cfg, "name"):
        return None
    scene_keys = env.scene.keys() if hasattr(env.scene, "keys") else []
    if asset_cfg.name not in scene_keys:
        return None
    try:
        asset: Articulation = env.scene[asset_cfg.name]
    except KeyError:
        return None
    lin_vel = getattr(asset.data, "root_lin_vel_w", None)
    ang_vel = getattr(asset.data, "root_ang_vel_w", None)
    if lin_vel is None or ang_vel is None:
        return None
    lin = torch.linalg.norm(lin_vel[:, :2], dim=1)
    yaw = torch.abs(ang_vel[:, 2])
    return lin + yaw


def _terrain_height_at_points(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    points_w: torch.Tensor,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    ray_hits = getattr(sensor.data, "ray_hits_w", None)
    if ray_hits is None:
        _warn_once(env, "_warn_missing_ray_hits", "Ray hits not available; using zero terrain height.")
        return torch.zeros(points_w.shape[:2], device=points_w.device, dtype=points_w.dtype)

    invalid = torch.isinf(ray_hits).any(dim=-1) | torch.isnan(ray_hits).any(dim=-1)
    ray_xy = ray_hits[..., :2]
    if invalid.any():
        far = torch.full_like(ray_xy, 1.0e6)
        ray_xy = torch.where(invalid.unsqueeze(-1), far, ray_xy)
    foot_xy = points_w[..., :2]
    diff = foot_xy.unsqueeze(2) - ray_xy.unsqueeze(1)
    dist2 = torch.sum(diff * diff, dim=-1)
    idx = torch.argmin(dist2, dim=2)
    ray_z = ray_hits[..., 2]
    if invalid.any():
        ray_z = torch.where(invalid, torch.zeros_like(ray_z), ray_z)
    ray_z_exp = ray_z.unsqueeze(1).expand(-1, foot_xy.shape[1], -1)
    return torch.gather(ray_z_exp, 2, idx.unsqueeze(-1)).squeeze(-1)


def _foot_heights_relative(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    foot_ids: list[int],
    terrain_sensor_cfg: SceneEntityCfg | None,
    height_offset: float,
) -> torch.Tensor:
    foot_pos_w = asset.data.body_pos_w[:, foot_ids]
    foot_heights = foot_pos_w[:, :, 2]
    if terrain_sensor_cfg is not None:
        terrain_heights = _terrain_height_at_points(env, terrain_sensor_cfg, foot_pos_w)
        return foot_heights - terrain_heights
    if height_offset != 0.0:
        return foot_heights - height_offset
    return foot_heights


def constraint_joint_pos(
    env: ManagerBasedRLEnv,
    margin: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fraction of joints violating soft position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _get_joint_slice(asset_cfg)
    joint_pos = asset.data.joint_pos
    if joint_pos is None:
        _warn_once(env, "_warn_missing_joint_pos", "Joint positions not available; joint_pos constraint disabled.")
        return _zeros_like_env(env)
    if not isinstance(joint_ids, slice):
        joint_pos = joint_pos[:, joint_ids]

    limits = getattr(asset.data, "soft_joint_pos_limits", None)
    if limits is None:
        _warn_once(
            env, "_warn_missing_soft_joint_limits", "soft_joint_pos_limits not found; joint_pos constraint disabled."
        )
        return _zeros_like_env(env, dtype=joint_pos.dtype)

    limits = limits.to(device=joint_pos.device, dtype=joint_pos.dtype)
    if limits.ndim == 2:
        if not isinstance(joint_ids, slice):
            limits = limits[joint_ids]
        limits = limits.unsqueeze(0).expand(joint_pos.shape[0], -1, -1)
    elif limits.ndim == 3:
        if not isinstance(joint_ids, slice):
            limits = limits[:, joint_ids]
        if limits.shape[0] == 1 and joint_pos.shape[0] > 1:
            limits = limits.expand(joint_pos.shape[0], -1, -1)
        elif limits.shape[0] != joint_pos.shape[0]:
            _warn_once(
                env,
                "_warn_soft_joint_limits_shape",
                "soft_joint_pos_limits shape mismatch; joint_pos constraint disabled.",
            )
            return _zeros_like_env(env, dtype=joint_pos.dtype)
    else:
        _warn_once(
            env,
            "_warn_soft_joint_limits_ndim",
            "soft_joint_pos_limits has unexpected shape; joint_pos constraint disabled.",
        )
        return _zeros_like_env(env, dtype=joint_pos.dtype)

    margin_t = torch.as_tensor(margin, device=joint_pos.device, dtype=joint_pos.dtype)
    lower = limits[..., 0] - margin_t
    upper = limits[..., 1] + margin_t
    violation = (joint_pos < lower) | (joint_pos > upper)
    return violation.float().mean(dim=1)


def joint_pos_prob_constraint(
    env: ManagerBasedRLEnv,
    margin: float = 0.0,
    limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Probabilistic joint position constraint (fraction of joints violating limits)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _get_joint_slice(asset_cfg)
    joint_pos = getattr(asset.data, "joint_pos", None)
    if joint_pos is None:
        _warn_once(env, "_warn_missing_joint_pos_prob", "Joint positions not available; joint_pos constraint disabled.")
        return _zeros_like_env(env)
    if not isinstance(joint_ids, slice):
        joint_pos = joint_pos[:, joint_ids]

    limits = getattr(asset.data, "soft_joint_pos_limits", None)
    if limits is None:
        limits = getattr(asset.data, "joint_pos_limits", None)
    if limits is None:
        _warn_once(
            env, "_warn_missing_joint_limits_prob", "joint_pos_limits not found; joint_pos constraint disabled."
        )
        return _zeros_like_env(env, dtype=joint_pos.dtype)

    limits = limits.to(device=joint_pos.device, dtype=joint_pos.dtype)
    if limits.ndim == 2:
        if not isinstance(joint_ids, slice):
            limits = limits[joint_ids]
        limits = limits.unsqueeze(0).expand(joint_pos.shape[0], -1, -1)
    elif limits.ndim == 3:
        if not isinstance(joint_ids, slice):
            limits = limits[:, joint_ids]
        if limits.shape[0] == 1 and joint_pos.shape[0] > 1:
            limits = limits.expand(joint_pos.shape[0], -1, -1)
        elif limits.shape[0] != joint_pos.shape[0]:
            _warn_once(
                env,
                "_warn_joint_limits_shape_prob",
                "joint_pos_limits shape mismatch; joint_pos constraint disabled.",
            )
            return _zeros_like_env(env, dtype=joint_pos.dtype)
    else:
        _warn_once(
            env,
            "_warn_joint_limits_ndim_prob",
            "joint_pos_limits has unexpected shape; joint_pos constraint disabled.",
        )
        return _zeros_like_env(env, dtype=joint_pos.dtype)

    margin_t = torch.as_tensor(margin, device=joint_pos.device, dtype=joint_pos.dtype)
    lower = limits[..., 0] + margin_t
    upper = limits[..., 1] - margin_t
    violation = (joint_pos < lower) | (joint_pos > upper)
    cost = violation.float().mean(dim=1)
    return _normalize_cost(cost, limit)


def joint_vel_prob_constraint(
    env: ManagerBasedRLEnv,
    limit: float = 50.0,
    cost_limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Probabilistic joint velocity constraint (fraction of joints violating limits)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _get_joint_slice(asset_cfg)
    joint_vel = getattr(asset.data, "joint_vel", None)
    if joint_vel is None:
        _warn_once(env, "_warn_missing_joint_vel_prob", "Joint velocities not available; constraint disabled.")
        return _zeros_like_env(env)
    if not isinstance(joint_ids, slice):
        joint_vel = joint_vel[:, joint_ids]
    limit_t = torch.as_tensor(limit, device=joint_vel.device, dtype=joint_vel.dtype)
    violation = torch.abs(joint_vel) > limit_t
    cost = violation.float().mean(dim=1)
    return _normalize_cost(cost, cost_limit)


def joint_torque_prob_constraint(
    env: ManagerBasedRLEnv,
    limit: float = 100.0,
    cost_limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Probabilistic joint torque constraint (fraction of joints violating limits)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _get_joint_slice(asset_cfg)
    torque = getattr(asset.data, "applied_torque", None)
    if torque is None:
        _warn_once(env, "_warn_missing_applied_torque_prob", "Applied torque not available; constraint disabled.")
        return _zeros_like_env(env)
    if not isinstance(joint_ids, slice):
        torque = torque[:, joint_ids]
    limit_t = torch.as_tensor(limit, device=torque.device, dtype=torque.dtype)
    violation = torch.abs(torque) > limit_t
    cost = violation.float().mean(dim=1)
    return _normalize_cost(cost, cost_limit)


def body_contact_prob_constraint(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    foot_body_names: list[str] | None = None,
    threshold: float = 1.0,
    limit: float | None = None,
) -> torch.Tensor:
    """Probabilistic body contact constraint (any non-foot contact)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    force_mag = torch.norm(net_forces, dim=-1).max(dim=1)[0]

    cache_attr = "_body_contact_non_foot_ids"
    non_foot_ids = getattr(env, cache_attr, None)
    if non_foot_ids is None:
        foot_body_names = foot_body_names or []
        if foot_body_names:
            foot_ids, _ = contact_sensor.find_bodies(foot_body_names, preserve_order=True)
        else:
            foot_ids = []
        mask = torch.ones(contact_sensor.num_bodies, dtype=torch.bool, device=force_mag.device)
        if foot_ids:
            mask[torch.as_tensor(foot_ids, device=force_mag.device)] = False
        non_foot_ids = torch.where(mask)[0]
        setattr(env, cache_attr, non_foot_ids)

    if non_foot_ids.numel() == 0:
        return _zeros_like_env(env, dtype=force_mag.dtype)

    violation = force_mag[:, non_foot_ids] > threshold
    cost = violation.any(dim=1).float()
    return _normalize_cost(cost, limit)


def com_frame_prob_constraint(
    env: ManagerBasedRLEnv,
    height_range: tuple[float, float],
    max_angle_rad: float = 0.35,
    cost_limit: float | None = None,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    height_offset: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Probabilistic COM frame constraint on height and tilt."""
    asset: Articulation = env.scene[asset_cfg.name]
    com_pos = getattr(asset.data, "root_com_pos_w", None)
    if com_pos is None:
        com_pos = getattr(asset.data, "root_pos_w", None)
    if com_pos is None:
        _warn_once(env, "_warn_missing_root_pos", "Root position not available; COM constraint disabled.")
        return _zeros_like_env(env)
    height = com_pos[:, 2]
    if terrain_sensor_cfg is not None:
        terrain_h = _terrain_height_at_points(env, terrain_sensor_cfg, com_pos.unsqueeze(1)).squeeze(1)
        height = height - terrain_h
    elif height_offset != 0.0:
        height = height - height_offset
    min_h, max_h = height_range
    height_violation = (height < min_h) | (height > max_h)

    projected_gravity = getattr(asset.data, "projected_gravity_b", None)
    if projected_gravity is None:
        _warn_once(env, "_warn_missing_projected_gravity_com", "Projected gravity not available; COM constraint disabled.")
        return _zeros_like_env(env, dtype=height.dtype)
    grav_xy = torch.linalg.norm(projected_gravity[:, :2], dim=1)
    angle_limit = math.sin(max_angle_rad)
    angle_violation = grav_xy > torch.as_tensor(angle_limit, device=grav_xy.device, dtype=grav_xy.dtype)

    cost = (height_violation | angle_violation).float()
    return _normalize_cost(cost, cost_limit)


def gait_pattern_prob_constraint(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    foot_body_names: list[str],
    gait_frequency: float,
    phase_offsets: list[float],
    stance_ratio: float = 0.5,
    contact_threshold: float = 1.0,
    command_name: str | None = None,
    min_frequency: float | None = None,
    max_frequency: float | None = None,
    max_command_speed: float | None = None,
    frequency_scale: float = 0.0,
    min_command_speed: float | None = None,
    min_base_speed: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit: float | None = None,
) -> torch.Tensor:
    """Probabilistic gait constraint based on phase vs. contact state."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_ids, _ = contact_sensor.find_bodies(foot_body_names, preserve_order=True)
    if not foot_ids:
        return _zeros_like_env(env)

    force_mag = torch.norm(contact_sensor.data.net_forces_w_history, dim=-1).max(dim=1)[0]
    contact = force_mag[:, foot_ids] > contact_threshold

    t = env.episode_length_buf.to(force_mag.dtype) * env.step_dt
    offsets = torch.as_tensor(phase_offsets, device=force_mag.device, dtype=force_mag.dtype)
    freq = torch.as_tensor(gait_frequency, device=force_mag.device, dtype=force_mag.dtype).expand(env.num_envs)
    phase = torch.remainder(t.unsqueeze(-1) * freq.unsqueeze(-1) + offsets, 1.0)
    expected_contact = phase < stance_ratio
    mismatch = contact != expected_contact
    cost = mismatch.float().mean(dim=1)
    return _normalize_cost(cost, limit)


def orthogonal_velocity_constraint(
    env: ManagerBasedRLEnv,
    limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Average constraint on non-commanded velocity components."""
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_w
    ang_vel = asset.data.root_ang_vel_w
    cost = (torch.abs(lin_vel[:, 2]) + torch.abs(ang_vel[:, 0]) + torch.abs(ang_vel[:, 1])) / 3.0
    return _normalize_cost(cost, limit)


def contact_velocity_constraint(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    foot_body_names: list[str],
    contact_threshold: float = 1.0,
    limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Average constraint on horizontal contact velocity of feet."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_foot_ids, _ = contact_sensor.find_bodies(foot_body_names, preserve_order=True)
    if not contact_foot_ids:
        return _zeros_like_env(env)

    asset: Articulation = env.scene[asset_cfg.name]
    asset_foot_ids, _ = asset.find_bodies(foot_body_names, preserve_order=True)
    if not asset_foot_ids:
        return _zeros_like_env(env)
    contact_pos = getattr(contact_sensor.data, "contact_pos_w", None)
    if contact_pos is None or contact_pos.numel() == 0:
        force_mag = torch.norm(contact_sensor.data.net_forces_w_history, dim=-1).max(dim=1)[0]
        contact = force_mag[:, contact_foot_ids] > contact_threshold
        foot_vel_xy = asset.data.body_lin_vel_w[:, asset_foot_ids, :2]
        speed = torch.linalg.norm(foot_vel_xy, dim=-1)
        cost = (speed * contact.float()).sum(dim=1) / max(1, len(asset_foot_ids))
        return _normalize_cost(cost, limit)

    contact_pos = contact_pos[:, contact_foot_ids]
    valid = torch.isfinite(contact_pos).all(dim=-1)

    force_mag = torch.norm(contact_sensor.data.net_forces_w_history, dim=-1).max(dim=1)[0]
    contact_body = force_mag[:, contact_foot_ids] > contact_threshold
    contact_mask = valid & contact_body.unsqueeze(-1)

    foot_pos = asset.data.body_pos_w[:, asset_foot_ids]
    foot_lin = asset.data.body_lin_vel_w[:, asset_foot_ids]
    foot_ang = asset.data.body_ang_vel_w[:, asset_foot_ids]
    r = contact_pos - foot_pos.unsqueeze(2)
    v_point = foot_lin.unsqueeze(2) + torch.linalg.cross(foot_ang.unsqueeze(2), r)
    speed = torch.linalg.norm(v_point[..., :2], dim=-1)
    speed = torch.where(contact_mask, speed, torch.zeros_like(speed))
    denom = contact_mask.sum(dim=(1, 2)).clamp(min=1)
    cost = speed.sum(dim=(1, 2)) / denom
    return _normalize_cost(cost, limit)


def foot_clearance_constraint(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    foot_body_names: list[str],
    min_height: float | None = None,
    height_offset: float = 0.0,
    contact_threshold: float = 1.0,
    gait_frequency: float = 1.0,
    phase_offsets: list[float] | None = None,
    stance_ratio: float = 0.5,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Average constraint on swing foot clearance (negative peak swing height)."""
    asset: Articulation = env.scene[asset_cfg.name]
    asset_foot_ids, _ = asset.find_bodies(foot_body_names, preserve_order=True)
    if not asset_foot_ids:
        return _zeros_like_env(env)
    foot_heights = _foot_heights_relative(env, asset, asset_foot_ids, terrain_sensor_cfg, height_offset)

    num_legs = len(asset_foot_ids)
    offsets = phase_offsets if phase_offsets is not None else [0.0] * num_legs
    offsets_t = torch.as_tensor(offsets, device=foot_heights.device, dtype=foot_heights.dtype)
    if offsets_t.numel() != num_legs:
        _warn_once(
            env,
            "_warn_fc_phase_offsets",
            "phase_offsets size mismatch for foot_clearance; truncating or padding with zeros.",
        )
        if offsets_t.numel() < num_legs:
            pad = torch.zeros(num_legs - offsets_t.numel(), device=foot_heights.device, dtype=foot_heights.dtype)
            offsets_t = torch.cat([offsets_t, pad], dim=0)
        else:
            offsets_t = offsets_t[:num_legs]

    t = env.episode_length_buf.to(foot_heights.dtype) * env.step_dt
    freq = torch.as_tensor(gait_frequency, device=foot_heights.device, dtype=foot_heights.dtype).expand(env.num_envs)
    phase = torch.remainder(t.unsqueeze(-1) * freq.unsqueeze(-1) + offsets_t, 1.0)
    swing = phase >= stance_ratio

    buf_shape = (env.num_envs, num_legs)
    if not hasattr(env, "_foot_clearance_max") or env._foot_clearance_max.shape != buf_shape:
        env._foot_clearance_max = torch.zeros(buf_shape, device=foot_heights.device, dtype=foot_heights.dtype)
        env._foot_clearance_last = torch.zeros_like(env._foot_clearance_max)
        env._foot_clearance_prev_swing = swing.clone()

    max_buf = env._foot_clearance_max
    last_buf = env._foot_clearance_last
    prev_swing = env._foot_clearance_prev_swing

    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        reset_ids = torch.nonzero(reset_buf, as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            max_buf[reset_ids] = 0.0
            last_buf[reset_ids] = 0.0
            prev_swing[reset_ids] = swing[reset_ids]

    max_buf = torch.where(swing, torch.maximum(max_buf, foot_heights), max_buf)
    swing_end = prev_swing & ~swing
    last_buf = torch.where(swing_end, max_buf, last_buf)
    max_buf = torch.where(swing, max_buf, torch.zeros_like(max_buf))

    env._foot_clearance_max = max_buf
    env._foot_clearance_last = last_buf
    env._foot_clearance_prev_swing = swing

    cost = -torch.mean(last_buf, dim=1)
    return _normalize_cost(cost, limit)


def foot_height_limit_constraint(
    env: ManagerBasedRLEnv,
    foot_body_names: list[str],
    height_offset: float = 0.0,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    limit: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Average constraint on maximum foot height."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_ids, _ = asset.find_bodies(foot_body_names, preserve_order=True)
    if not foot_ids:
        return _zeros_like_env(env)
    foot_heights = _foot_heights_relative(env, asset, foot_ids, terrain_sensor_cfg, height_offset)
    cost = torch.max(foot_heights, dim=1).values
    return _normalize_cost(cost, limit)


def symmetric_constraint(
    env: ManagerBasedRLEnv,
    joint_pair_indices: list[tuple[int, int]],
    action_pair_indices: list[tuple[int, int]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    include_actions: bool = True,
    command_name: str | None = None,
    min_command_speed: float | None = None,
    min_base_speed: float | None = None,
    limit: float | None = None,
) -> torch.Tensor:
    """Average action symmetry constraint using L1 distance on mirrored joints."""
    if not include_actions or not hasattr(env, "action_manager"):
        return _zeros_like_env(env)
    action_pairs = action_pair_indices or joint_pair_indices
    if not action_pairs:
        return _zeros_like_env(env)
    actions = env.action_manager.action
    sym = torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)
    for left_idx, right_idx in action_pairs:
        sym += torch.abs(actions[:, left_idx] - actions[:, right_idx])
    sym /= len(action_pairs)
    return _normalize_cost(sym, limit)


def constraint_joint_vel(
    env: ManagerBasedRLEnv,
    limit: float = 50.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fraction of joints violating velocity limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _get_joint_slice(asset_cfg)
    joint_vel = getattr(asset.data, "joint_vel", None)
    if joint_vel is None:
        _warn_once(env, "_warn_missing_joint_vel", "Joint velocities not available; joint_vel constraint disabled.")
        return _zeros_like_env(env)
    if not isinstance(joint_ids, slice):
        joint_vel = joint_vel[:, joint_ids]
    limit_t = torch.as_tensor(limit, device=joint_vel.device, dtype=joint_vel.dtype)
    violation = torch.abs(joint_vel) > limit_t
    return violation.float().mean(dim=1)


def constraint_joint_torque(
    env: ManagerBasedRLEnv,
    limit: float = 100.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fraction of joints violating torque limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _get_joint_slice(asset_cfg)
    torque = getattr(asset.data, "applied_torque", None)
    if torque is None:
        _warn_once(env, "_warn_missing_applied_torque", "Applied torque not available; torque constraint disabled.")
        return _zeros_like_env(env)
    if not isinstance(joint_ids, slice):
        torque = torque[:, joint_ids]
    limit_t = torch.as_tensor(limit, device=torque.device, dtype=torque.dtype)
    violation = torch.abs(torque) > limit_t
    return violation.float().mean(dim=1)


def constraint_com_orientation(
    env: ManagerBasedRLEnv,
    max_angle_rad: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Binary constraint for excessive base tilt."""
    projected_gravity = None
    try:
        from isaaclab.envs import mdp as isaaclab_mdp
    except ImportError:
        isaaclab_mdp = None
    if isaaclab_mdp is not None and hasattr(isaaclab_mdp, "projected_gravity"):
        projected_gravity = isaaclab_mdp.projected_gravity(env, asset_cfg)
    if projected_gravity is None:
        asset: Articulation = env.scene[asset_cfg.name]
        projected_gravity = getattr(asset.data, "projected_gravity_b", None)
    if projected_gravity is None:
        _warn_once(env, "_warn_missing_projected_gravity", "Projected gravity not available; tilt constraint disabled.")
        return _zeros_like_env(env)
    grav_xy = torch.linalg.norm(projected_gravity[:, :2], dim=1)
    limit = math.sin(max_angle_rad)
    limit_t = torch.as_tensor(limit, device=grav_xy.device, dtype=grav_xy.dtype)
    return (grav_xy > limit_t).float()


def stability_constraint(
    env: ManagerBasedRLEnv,
    max_angle_rad: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Binary constraint for excessive base tilt (stability)."""
    return constraint_com_orientation(env, max_angle_rad=max_angle_rad, asset_cfg=asset_cfg)


def torque_constraint(
    env: ManagerBasedRLEnv,
    limit: float = 100.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fraction of joints violating torque limits."""
    return constraint_joint_torque(env, limit=limit, asset_cfg=asset_cfg)


def compute_constraint_info(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    config: object | None = None,
) -> dict[str, torch.Tensor]:
    cfg = config
    if cfg is None and hasattr(env, "cfg"):
        cfg = getattr(env.cfg.config_summary, "constraint", None)

    margin = getattr(cfg, "joint_pos_margin", 0.3) if cfg is not None else 0.3
    vel_limit = getattr(cfg, "joint_vel_limit", 50.0) if cfg is not None else 50.0
    torque_limit = getattr(cfg, "joint_torque_limit", 100.0) if cfg is not None else 100.0
    max_angle = getattr(cfg, "com_max_angle_rad", 0.35) if cfg is not None else 0.35

    info = {
        "constraint_joint_pos": constraint_joint_pos(env, margin=margin, asset_cfg=asset_cfg),
        "constraint_joint_vel": constraint_joint_vel(env, limit=vel_limit, asset_cfg=asset_cfg),
        "constraint_joint_torque": constraint_joint_torque(env, limit=torque_limit, asset_cfg=asset_cfg),
        "constraint_com_orientation": constraint_com_orientation(env, max_angle_rad=max_angle, asset_cfg=asset_cfg),
        "stability_constraint": stability_constraint(env, max_angle_rad=max_angle, asset_cfg=asset_cfg),
        "torque_constraint": torque_constraint(env, limit=torque_limit, asset_cfg=asset_cfg),
    }
    return info


def aggregate_constraint_cost(
    constraint_info: dict[str, torch.Tensor],
    cost_keys: list[str] | None = None,
) -> torch.Tensor:
    if not constraint_info:
        return torch.tensor(0.0)
    values: list[torch.Tensor] = []
    if cost_keys:
        values = [constraint_info[key].view(-1) for key in cost_keys if key in constraint_info]
    if not values:
        values = [value.view(-1) for value in constraint_info.values()]
    cost = torch.zeros_like(values[0])
    for value in values:
        cost = cost + torch.abs(value)
    return cost
