# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import omni.log
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformVelocityCommandTerrainCfg


class UniformVelocityCommandTerrain(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: UniformVelocityCommandTerrainCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandTerrainCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # check key set consistency
        keys_cmd = set(self.cfg.command_ids.keys())
        keys_ranges = set(self.cfg.ranges.keys())
        if keys_cmd != keys_ranges:
            missing_in_ranges = keys_cmd - keys_ranges
            missing_in_cmd = keys_ranges - keys_cmd
            raise ValueError(
                f"UniformVelocityCommandTerrainCfg configuration error:\n"
                f" keys in command_ids but not in ranges: {missing_in_ranges}\n"
                f" keys in ranges but not in command_ids: {missing_in_cmd}"
            )
        # check if command_ids covers 0~num_envs-1 and no duplicates
        all_ids = [i for ids in self.cfg.command_ids.values() for i in ids]
        # check number
        if len(all_ids) != self.num_envs:
            raise ValueError(
                f"UniformVelocityCommandTerrain configuration error:\n"
                f" command ID number should be {self.num_envs}, but got {len(all_ids)}"
            )
        # check if covers 0~num_envs-1 and no duplicates
        expected = set(range(self.num_envs))
        actual = set(all_ids)
        if actual != expected:
            missing = expected - actual
            extra = actual - expected
            msgs = []
            if missing:
                msgs.append(f"missing env id: {sorted(missing)}")
            if extra:
                msgs.append(f"extra env id: {sorted(extra)}")
            raise ValueError(
                "UniformVelocityCommandTerrain configuration error:\n" + "\n".join(msgs)
            )

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.ang_vel_z_limit_low = torch.zeros(self.num_envs, device=self.device)
        self.ang_vel_z_limit_high = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_yaw_env = torch.zeros_like(self.is_heading_env)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        for key, id_list in self.cfg.command_ids.items():
            id_list_torch = torch.tensor(id_list, device=self.device, dtype=torch.long)
            self.ang_vel_z_limit_low[id_list_torch] = self.cfg.ranges[key].ang_vel_z[0]
            self.ang_vel_z_limit_high[id_list_torch] = self.cfg.ranges[key].ang_vel_z[1]

        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformVelocityCommandTerrain:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        # 显示每个 terrain 的 heading_command_prob
        probs = [r.heading_command_prob for r in self.cfg.ranges.values()]
        msg += f"\tHeading command probs: {probs}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    # def _resample_command(self, env_ids: Sequence[int]):
    #     """
    #     For each terrain, sample velocity commands using the corresponding ranges and write to self.vel_command_b and self.heading_target.
    #     """
    #     if isinstance(env_ids, torch.Tensor):
    #         env_ids = env_ids.tolist()
    #     # loop over each terrain
    #     for key, id_list in self.cfg.command_ids.items():
    #         # find env ids that belong to current id_list
    #         overlap = [eid for eid in env_ids if eid in id_list]
    #         if not overlap:
    #             continue
    #         overlap_ids = torch.tensor(overlap, device=self.device, dtype=torch.long)

    #         # get the corresponding range configuration
    #         range_cfg = self.cfg.ranges[key]

    #         # sample in x, y, yaw direction
    #         # lin_vel_x
    #         r = torch.empty(len(overlap_ids), device=self.device)
    #         self.vel_command_b[overlap_ids, 0] = r.uniform_(*range_cfg.lin_vel_x)
    #         # lin_vel_y
    #         self.vel_command_b[overlap_ids, 1] = r.uniform_(*range_cfg.lin_vel_y)
    #         # ang_vel_z
    #         self.vel_command_b[overlap_ids, 2] = r.uniform_(*range_cfg.ang_vel_z)
    #         # heading
    #         self.heading_target[overlap_ids] = r.uniform_(*range_cfg.heading)

    # def _update_command(self):
    #     for key, id_list in self.cfg.command_ids.items():
    #         prob = self.cfg.ranges[key].heading_command_prob
    #         if prob > 0.0:
    #             # resolve indices of heading envs
    #             if prob >= 1.0:
    #                 env_ids = self._id_tensors[key]
    #             else:
    #                 mask = torch.rand(len(id_list), device=self.device) < prob
    #                 env_ids = self._id_tensors[key][mask]

    #             if env_ids.numel() == 0:
    #                 continue
    #             # compute angular velocity
    #             heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
    #             self.vel_command_b[env_ids, 2] = torch.clip(
    #                 self.cfg.ranges[key].heading_control_stiffness * heading_error,
    #                 min=self.cfg.ranges[key].ang_vel_z[0],
    #                 max=self.cfg.ranges[key].ang_vel_z[1],
    #             )
    #     mask = torch.norm(self.vel_command_b[:, :3], dim=1) <= 0.1
    #     self.vel_command_b[mask] = 0.0
    #     # print("self.vel_command_b:", self.vel_command_b)
    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        # —— 第一步：对每个 terrain 抽线速度 & heading_target —— #
        for key, id_list in self.cfg.command_ids.items():
            id_list_torch = torch.tensor(id_list, device=self.device, dtype=torch.long)
            self.ang_vel_z_limit_low[id_list_torch] = self.cfg.ranges[key].ang_vel_z[0]
            self.ang_vel_z_limit_high[id_list_torch] = self.cfg.ranges[key].ang_vel_z[1]
            # 只对本次要 resample 的 env_ids 做交集
            overlap = [eid for eid in env_ids if eid in id_list]
            if not overlap:
                continue
            overlap_ids = torch.tensor(overlap, device=self.device, dtype=torch.long)

            range_cfg = self.cfg.ranges[key]
            r = torch.empty(len(overlap_ids), device=self.device)
            self.vel_command_b[overlap_ids, 0] = r.uniform_(*range_cfg.lin_vel_x)
            self.vel_command_b[overlap_ids, 1] = r.uniform_(*range_cfg.lin_vel_y)
            self.vel_command_b[overlap_ids, 2] = r.uniform_(*range_cfg.ang_vel_z)
            self.heading_target[overlap_ids] = r.uniform_(*range_cfg.heading)

            heading_prob = self.cfg.ranges[key].heading_command_prob
            yaw_prob = self.cfg.ranges[key].yaw_command_prob
            standing_prob = self.cfg.ranges[key].standing_command_prob

            if heading_prob > 0.0:
                self.is_heading_env[overlap_ids] = r.uniform_(0.0, 1.0) <= heading_prob
            if yaw_prob > 0.0:
                self.is_yaw_env[overlap_ids] = r.uniform_(0.0, 1.0) <= yaw_prob
            if standing_prob > 0.0:
                self.is_standing_env[overlap_ids] = r.uniform_(0.0, 1.0) <= standing_prob

    def _update_command(self):
        heading_env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
        yaw_env_ids = self.is_yaw_env.nonzero(as_tuple=False).flatten()
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if heading_env_ids.numel() > 0:
            heading_error = math_utils.wrap_to_pi(self.heading_target[heading_env_ids] - self.robot.data.heading_w[heading_env_ids])
            self.vel_command_b[heading_env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.ang_vel_z_limit_low[heading_env_ids],
                max=self.ang_vel_z_limit_high[heading_env_ids],
            )
        if yaw_env_ids.numel() > 0:
            self.vel_command_b[yaw_env_ids, :2] = 0.0
        if standing_env_ids.numel() > 0:
            self.vel_command_b[standing_env_ids, :] = 0.0

        mask = torch.norm(self.vel_command_b[:, :3], dim=1) <= 0.1
        self.vel_command_b[mask] = 0.0
        # print("vel_command_b:", self.vel_command_b)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
