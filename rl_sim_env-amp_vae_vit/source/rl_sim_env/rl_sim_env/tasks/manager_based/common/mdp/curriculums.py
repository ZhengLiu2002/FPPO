# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_x_command_threshold(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    command = env.command_manager.get_term("base_velocity")
    max_episode_length = env.max_episode_length

    lin_x_level = command.cfg.lin_x_level
    max_lin_x_level = command.cfg.max_lin_x_level
    if (env.common_step_counter > ((lin_x_level + 1) * max_episode_length * 8)) and (lin_x_level < max_lin_x_level):
        if lin_x_level < max_lin_x_level:
            lin_x_level += 1.0
            command.cfg.lin_x_level = lin_x_level
        for key, range_cfg in command.cfg.ranges.items():
            step0 = (range_cfg.max_curriculum_lin_x[0] - range_cfg.start_curriculum_lin_x[0]) / max_lin_x_level
            step1 = (range_cfg.max_curriculum_lin_x[1] - range_cfg.start_curriculum_lin_x[1]) / max_lin_x_level
            new_min = range_cfg.start_curriculum_lin_x[0] + step0 * lin_x_level
            new_max = range_cfg.start_curriculum_lin_x[1] + step1 * lin_x_level
            range_cfg.lin_vel_x = (new_min, new_max)

    return torch.tensor(lin_x_level, device=env.device)


def ang_vel_z_command_threshold(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    command = env.command_manager.get_term("base_velocity")
    max_episode_length = env.max_episode_length

    ang_z_level = command.cfg.ang_z_level
    max_ang_z_level = command.cfg.max_ang_z_level
    if (env.common_step_counter > ((ang_z_level + 1) * max_episode_length * 8)) and (ang_z_level < max_ang_z_level):
        if ang_z_level < max_ang_z_level:
            ang_z_level += 1.0
            command.cfg.ang_z_level = ang_z_level
        for key, range_cfg in command.cfg.ranges.items():
            step0 = (range_cfg.max_curriculum_ang_z[0] - range_cfg.start_curriculum_ang_z[0]) / max_ang_z_level
            step1 = (range_cfg.max_curriculum_ang_z[1] - range_cfg.start_curriculum_ang_z[1]) / max_ang_z_level
            new_min = range_cfg.start_curriculum_ang_z[0] + step0 * ang_z_level
            new_max = range_cfg.start_curriculum_ang_z[1] + step1 * ang_z_level
            range_cfg.ang_vel_z = (new_min, new_max)

    return torch.tensor(ang_z_level, device=env.device)
