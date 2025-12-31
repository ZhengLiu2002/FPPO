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
