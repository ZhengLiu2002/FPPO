# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def command_tracking_quadratic(
    env: ManagerBasedRLEnv,
    command_name: str,
    kappa_lin: float,
    kappa_ang: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Command tracking reward using a negative quadratic penalty."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    cmd = env.command_manager.get_command(command_name)
    lin_error = torch.sum(torch.square(cmd[:, :2] - vel_yaw[:, :2]), dim=1)
    ang_error = torch.square(cmd[:, 2] - asset.data.root_ang_vel_w[:, 2])
    kappa_ang = kappa_lin if kappa_ang is None else kappa_ang
    return -(kappa_lin * lin_error + kappa_ang * ang_error)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint power."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )


def joint_power_distribution(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint power distribution."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.var(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )


def joint_torque_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ref_mass: float | None = None,
    ref_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize joint torque using L2 norm squared."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.square(torque), dim=1)

    if ref_mass is None:
        return penalty

    scale_attr = "_torque_mass_scale"
    if not hasattr(env, scale_attr):
        masses = getattr(asset.data, "default_mass", None)
        if masses is None:
            setattr(env, scale_attr, ref_weight)
        else:
            robot_mass = masses[0].sum().item()
            scale = ref_weight * (ref_mass / max(robot_mass, 1.0e-6))
            setattr(env, scale_attr, scale)
    scale = getattr(env, scale_attr)
    return penalty * scale


def amp_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward AMP."""
    # extract the used quantities (to enable type-hinting)
    reward = torch.clamp(1 - (1 / 4) * torch.square(env.amp_out - 1), min=0)
    return reward.squeeze()


def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward action smoothness."""
    actions = getattr(env, "actions_history", None)
    if actions is not None:
        diff = torch.square(actions.get_data_vec([0]) - 2 * actions.get_data_vec([1]) + actions.get_data_vec([2]))
        return torch.sum(diff, dim=1)
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    prev_prev = getattr(env, "_prev_prev_action", prev_action)
    diff2 = action - 2 * prev_action + prev_prev
    return torch.sum(torch.square(diff2), dim=1)


def action_smoothness_penalty(
    env: ManagerBasedRLEnv,
    action_diff_weight: float = 1.0,
    action_diff2_weight: float = 1.0,
    joint_vel_weight: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize action smoothness using first- and second-order differences."""
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    prev_prev = getattr(env, "_prev_prev_action", prev_action)

    diff1 = action - prev_action
    diff2 = action - 2 * prev_action + prev_prev
    penalty = action_diff_weight * torch.sum(torch.square(diff1), dim=1)
    penalty += action_diff2_weight * torch.linalg.norm(diff2, dim=1)

    if joint_vel_weight > 0.0:
        asset: Articulation = env.scene[asset_cfg.name]
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
        penalty += joint_vel_weight * torch.sum(torch.square(joint_vel), dim=1)

    return penalty


def stand_joint_deviation_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(angle), dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) < 0.1
    return reward


def base_height_l2_fix(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        # 检查sensor数据是否包含inf或nan
        ray_hits = sensor.data.ray_hits_w[..., 2]
        ray_hits = torch.where(torch.isinf(ray_hits), 0.0, ray_hits)
        ray_hits = torch.where(torch.isnan(ray_hits), 0.0, ray_hits)
        adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
