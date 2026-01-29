# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp                       # 导入标准 MDP 库
from isaaclab.managers import EventTermCfg as EventTerm # 导入事件项配置
from isaaclab.managers import SceneEntityCfg            # 导入场景实体配置
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from rl_sim_env.envs.fppo_env import FPPOEnv
from rl_sim_env.tasks.manager_based.fppo.fppo_base_env_cfg import CostsCfg, FPPOEnvCfg
from rl_sim_env.tasks.manager_based.fppo.config.grq20_v2d3.config_summary import (
    ROBOT_BASE_LINK,
    ROBOT_CFG,
    ROBOT_FOOT_NAMES,
    ROBOT_LEG_JOINT_NAMES,
    ROBOT_THIGH_BODY_NAMES,
    ConfigSummary,
)
from rl_sim_env.tasks.manager_based.common.command.config import (
    create_uniform_velocity_command_terrain_cfg,
    create_uniform_velocity_command_cfg,
)
from rl_sim_env.tasks.manager_based.common.sensor.frame_transform_config import (
    create_body_frame_transform_cfg,
)
from rl_sim_env.tasks.manager_based.common.sensor.ray_caster_config import (
    CRITIC_HEIGHT_SCANNER_CFG,
)
from rl_sim_env.tasks.manager_based.common.terrain.config import AMP_VAE_TERRAIN_CFG
from typing import Dict, Tuple

@configclass
class Grq20V2d3FPPOEnvCfg(FPPOEnvCfg):
    class_type = FPPOEnv

    def __post_init__(self):
        # config summary
        self.config_summary = ConfigSummary
        self.costs = CostsCfg()

        # general settings
        self.decimation = self.config_summary.general.decimation
        self.episode_length_s = self.config_summary.general.episode_length_s

        # robot settings
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain settings
        # default num_envs comes from summary; may be overridden later (CLI).
        if self.scene.num_envs is None:
            self.scene.num_envs = self.config_summary.env.num_envs
        self.scene.terrain = AMP_VAE_TERRAIN_CFG
        # num_terrains = int(20.0 / self.config_summary.env.num_terrains_percent)
        # self.scene.terrain.terrain_generator.num_cols = num_terrains
        # self.scene.terrain.terrain_generator.sub_terrains['pyramid_stairs'].proportion = 4.0 / float(num_terrains)
        # self.scene.terrain.terrain_generator.sub_terrains['pyramid_stairs_inv'].proportion = 4.0 / float(num_terrains)
        # self.scene.terrain.terrain_generator.sub_terrains['boxes'].proportion = 4.0 / float(num_terrains)
        # self.scene.terrain.terrain_generator.sub_terrains['random_rough'].proportion = 4.0 / float(num_terrains)
        # self.scene.terrain.terrain_generator.sub_terrains['hf_pyramid_slope'].proportion = 2.0 / float(num_terrains)
        # self.scene.terrain.terrain_generator.sub_terrains['hf_pyramid_slope_inv'].proportion = 2.0 / float(num_terrains)
        # self.scene.terrain.terrain_generator.sub_terrains['plane'].proportion = 1.0 - 20.0 / float(num_terrains)

        # simulation settings
        self.sim.dt = self.config_summary.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # height scanner settings
        self.scene.height_scanner = CRITIC_HEIGHT_SCANNER_CFG
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + ROBOT_BASE_LINK
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # contact forces settings
        self.scene.contact_forces.update_period = self.sim.dt

        # frame transform settings
        self.scene.frame_transform = create_body_frame_transform_cfg(ROBOT_BASE_LINK, ROBOT_FOOT_NAMES)

        # # command settings
        self._rebuild_command_cfg(self.scene.num_envs)

        leg_joint_cfg = SceneEntityCfg("robot", joint_names=ROBOT_LEG_JOINT_NAMES, preserve_order=True)

        # reduce action scale & 仅控制腿部关节（机械臂从动作向量中移除，实现“锁死”）
        self.actions.joint_pos.scale = self.config_summary.action.scale
        self.actions.joint_pos.joint_names = ROBOT_LEG_JOINT_NAMES

        # observations
        # scale
        # critic obs
        self.observations.critic_obs.base_lin_vel.scale = self.config_summary.observation.scale.base_lin_vel
        self.observations.critic_obs.base_ang_vel.scale = self.config_summary.observation.scale.base_ang_vel
        self.observations.critic_obs.projected_gravity.scale = self.config_summary.observation.scale.projected_gravity
        self.observations.critic_obs.velocity_commands.params["scale"] = (
            self.config_summary.observation.scale.vel_command
        )
        self.observations.critic_obs.joint_pos.scale = self.config_summary.observation.scale.joint_pos
        self.observations.critic_obs.joint_vel.scale = self.config_summary.observation.scale.joint_vel
        self.observations.critic_obs.height_scan.scale = self.config_summary.observation.scale.height_measurements
        # optional critic-only aux terms (may be disabled in _rebuild_command_cfg)
        if self.observations.critic_obs.random_mass is not None:
            self.observations.critic_obs.random_mass.scale = self.config_summary.observation.scale.random_mass
        if self.observations.critic_obs.random_com is not None:
            self.observations.critic_obs.random_com.scale = self.config_summary.observation.scale.random_com
        if self.observations.critic_obs.random_material is not None:
            self.observations.critic_obs.random_material.scale = self.config_summary.observation.scale.random_material
        # actor obs
        self.observations.actor_obs.base_lin_vel.scale = self.config_summary.observation.scale.base_lin_vel
        self.observations.actor_obs.base_ang_vel.scale = self.config_summary.observation.scale.base_ang_vel
        self.observations.actor_obs.projected_gravity.scale = self.config_summary.observation.scale.projected_gravity
        self.observations.actor_obs.velocity_commands.params["scale"] = (
            self.config_summary.observation.scale.vel_command
        )
        if self.observations.actor_obs.height_scan is not None:
            self.observations.actor_obs.height_scan.scale = self.config_summary.observation.scale.height_measurements
        if self.observations.actor_obs.random_mass is not None:
            self.observations.actor_obs.random_mass.scale = self.config_summary.observation.scale.random_mass
        if self.observations.actor_obs.random_com is not None:
            self.observations.actor_obs.random_com.scale = self.config_summary.observation.scale.random_com
        # align joint observation ordering with action joints for symmetry
        self.observations.actor_obs.joint_pos.params["asset_cfg"] = leg_joint_cfg
        self.observations.actor_obs.joint_vel.params["asset_cfg"] = leg_joint_cfg
        if getattr(self.observations.actor_obs, "joint_torques", None) is not None:
            self.observations.actor_obs.joint_torques.params["asset_cfg"] = leg_joint_cfg
        self.observations.critic_obs.joint_pos.params["asset_cfg"] = leg_joint_cfg
        self.observations.critic_obs.joint_vel.params["asset_cfg"] = leg_joint_cfg


    def _compute_command_ids_and_ranges(self, num_envs: int) -> Tuple[Dict[str, list[int]], Dict[str, object]]:
        """Split env ids across terrains, ensuring full coverage even when num_envs is overridden."""
        command_ids: Dict[str, list[int]] = {}
        command_ranges: Dict[str, object] = {}
        env_start = 0

        sub_terrain_keys = list(self.scene.terrain.terrain_generator.sub_terrains.keys())

        for i, key in enumerate(sub_terrain_keys):
            item = self.scene.terrain.terrain_generator.sub_terrains[key]
            if i == len(sub_terrain_keys) - 1:
                count = max(0, num_envs - env_start)
            else:
                count = int(item.proportion * num_envs)
            command_ids[key] = list(range(env_start, env_start + count))
            env_start += count
            command_ranges[key] = self.config_summary.command.ranges[key]

        # If due to rounding we still miss some envs, append them to the last key
        total_assigned = sum(len(v) for v in command_ids.values())
        if total_assigned < num_envs and sub_terrain_keys:
            missing = num_envs - total_assigned
            last_key = sub_terrain_keys[-1]
            start = env_start
            command_ids[last_key].extend(range(start, start + missing))
        return command_ids, command_ranges

    def _rebuild_command_cfg(self, num_envs: int):
        """Recreate command config to match current num_envs (used when CLI overrides num_envs)."""
        self.scene.num_envs = num_envs
        command_ids, command_ranges = self._compute_command_ids_and_ranges(num_envs)
        self.commands.base_velocity = create_uniform_velocity_command_terrain_cfg(
            command_ids=command_ids,
            ranges=command_ranges,
            lin_x_level=self.config_summary.command.lin_x_level,
            ang_z_level=self.config_summary.command.ang_z_level,
            max_lin_x_level=self.config_summary.command.max_lin_x_level,
            max_ang_z_level=self.config_summary.command.max_ang_z_level,
            heading_control_stiffness=self.config_summary.command.heading_control_stiffness,
        )
        leg_joint_cfg = SceneEntityCfg("robot", joint_names=ROBOT_LEG_JOINT_NAMES, preserve_order=True)
        self.observations.actor_obs.joint_pos.scale = self.config_summary.observation.scale.joint_pos
        self.observations.actor_obs.joint_vel.scale = self.config_summary.observation.scale.joint_vel
        if getattr(self.observations.actor_obs, "joint_torques", None) is not None:
            self.observations.actor_obs.joint_torques.params["asset_cfg"] = leg_joint_cfg
        # Drop auxiliary critic terms (push_vel, random material/com) to match the 250-dim critic_obs used in training;
        # keep random_mass so VAE can learn mass inference
        self.observations.critic_obs.push_vel = None
        self.observations.critic_obs.random_material = None
        self.observations.critic_obs.random_com = None
        # noise
        base_lin_vel_noise = self.config_summary.observation.noise.base_lin_vel
        base_ang_vel_noise = self.config_summary.observation.noise.base_ang_vel
        gravity_noise = self.config_summary.observation.noise.projected_gravity
        joint_pos_noise = self.config_summary.observation.noise.joint_pos
        joint_vel_noise = self.config_summary.observation.noise.joint_vel

        # actor obs
        self.observations.actor_obs.base_lin_vel.noise = Unoise(
            n_min=-base_lin_vel_noise,
            n_max=base_lin_vel_noise,
        )
        self.observations.actor_obs.base_ang_vel.noise = Unoise(
            n_min=-base_ang_vel_noise,
            n_max=base_ang_vel_noise,
        )
        self.observations.actor_obs.projected_gravity.noise = Unoise(n_min=-gravity_noise, n_max=gravity_noise)
        self.observations.actor_obs.joint_pos.noise = Unoise(
            n_min=-joint_pos_noise,
            n_max=joint_pos_noise,
        )
        self.observations.actor_obs.joint_vel.noise = Unoise(
            n_min=-joint_vel_noise,
            n_max=joint_vel_noise,
        )

        if self.observations.actor_obs.height_scan is not None:
            self.observations.actor_obs.height_scan.clip = self.config_summary.observation.clip.height_measurements

        # clip
        # critic obs
        self.observations.critic_obs.height_scan.clip = self.config_summary.observation.clip.height_measurements

        # event
        self.events.add_base_mass.params["mass_distribution_params"] = self.config_summary.event.randomize_base_mass
        self.events.add_base_mass.params["asset_cfg"].body_names = ROBOT_BASE_LINK

        self.events.base_com_randomization.params["asset_cfg"].body_names = ROBOT_BASE_LINK
        self.events.base_com_randomization.params["com_range"] = self.config_summary.event.randomize_base_com

        self.events.physics_material.params["asset_cfg"].body_names = ".*"
        self.events.physics_material.params["static_friction_range"] = (
            self.config_summary.event.randomize_static_friction
        )
        self.events.physics_material.params["dynamic_friction_range"] = (
            self.config_summary.event.randomize_dynamic_friction
        )
        self.events.physics_material.params["restitution_range"] = self.config_summary.event.randomize_restitution

        self.events.reset_base.params["pose_range"] = self.config_summary.event.reset_base_pose
        self.events.reset_base.params["velocity_range"] = self.config_summary.event.reset_base_velocity

        # Only randomize leg joints.
        self.events.reset_robot_joints.params["position_range"] = self.config_summary.event.reset_robot_joints
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ROBOT_LEG_JOINT_NAMES)

        self.events.reset_actuator_gains.params["stiffness_distribution_params"] = (
            self.config_summary.event.randomize_actuator_kp_gains
        )
        self.events.reset_actuator_gains.params["damping_distribution_params"] = (
            self.config_summary.event.randomize_actuator_kd_gains
        )
        self.events.reset_actuator_gains.params["kt_distribution_params"] = (
            self.config_summary.event.randomize_actuator_kt_gains
        )

        self.events.push_robot.params["velocity_range"] = self.config_summary.event.push_robot_vel
        self.events.push_robot.params["velocity_schedule"] = self.config_summary.event.push_robot_schedule

        # rewards
        self.rewards.track_lin_vel_xy_exp.weight = self.config_summary.reward.track_lin_vel_xy_exp.weight
        self.rewards.track_lin_vel_xy_exp.params["std"] = self.config_summary.reward.track_lin_vel_xy_exp.std
        self.rewards.track_lin_vel_xy_exp.params["command_name"] = self.config_summary.reward.track_lin_vel_xy_exp.command_name

        self.rewards.track_ang_vel_z_exp.weight = self.config_summary.reward.track_ang_vel_z_exp.weight
        self.rewards.track_ang_vel_z_exp.params["std"] = self.config_summary.reward.track_ang_vel_z_exp.std
        self.rewards.track_ang_vel_z_exp.params["command_name"] = self.config_summary.reward.track_ang_vel_z_exp.command_name

        self.rewards.flat_orientation_l2.weight = self.config_summary.reward.flat_orientation_l2.weight
        self.rewards.base_height_l2_fix.weight = self.config_summary.reward.base_height_l2_fix.weight
        self.rewards.base_height_l2_fix.params["target_height"] = self.config_summary.reward.base_height_l2_fix.target_height
        self.rewards.base_height_l2_fix.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.rewards.base_height_l2_fix.params["asset_cfg"] = SceneEntityCfg("robot")

        self.rewards.joint_torques_l2.weight = self.config_summary.reward.joint_torques_l2.weight
        self.rewards.joint_torques_l2.params["asset_cfg"] = leg_joint_cfg

        self.rewards.joint_vel_l2.weight = self.config_summary.reward.joint_vel_l2.weight
        self.rewards.joint_vel_l2.params["asset_cfg"] = leg_joint_cfg

        self.rewards.joint_acc_l2.weight = self.config_summary.reward.joint_acc_l2.weight
        self.rewards.joint_acc_l2.params["asset_cfg"] = leg_joint_cfg

        self.rewards.dof_error_l2.weight = self.config_summary.reward.dof_error_l2.weight
        self.rewards.dof_error_l2.params["asset_cfg"] = leg_joint_cfg

        # self.rewards.stand_joint_deviation_l1.weight = self.config_summary.reward.stand_joint_deviation_l1.weight
        # self.rewards.stand_joint_deviation_l1.params["command_name"] = self.config_summary.reward.stand_joint_deviation_l1.command_name
        # self.rewards.stand_joint_deviation_l1.params["asset_cfg"] = leg_joint_cfg

        self.rewards.action_rate_l2.weight = self.config_summary.reward.action_rate_l2.weight

        self.rewards.action_smoothness_l2.weight = self.config_summary.reward.action_smoothness_l2.weight

        self.rewards.lin_vel_z_l2.weight = self.config_summary.reward.lin_vel_z_l2.weight

        self.rewards.ang_vel_xy_l2.weight = self.config_summary.reward.ang_vel_xy_l2.weight

        self.rewards.feet_air_time.weight = self.config_summary.reward.feet_air_time.weight
        self.rewards.feet_air_time.params["threshold"] = self.config_summary.reward.feet_air_time.threshold
        self.rewards.feet_air_time.params["command_name"] = self.config_summary.reward.feet_air_time.command_name

        self.rewards.feet_slide.weight = self.config_summary.reward.feet_slide.weight
        self.rewards.feet_slide.params["asset_cfg"] = SceneEntityCfg("robot") # Use default robot cfg

        self.rewards.load_sharing.weight = self.config_summary.reward.load_sharing.weight

        self.rewards.undesired_contacts.weight = self.config_summary.reward.undesired_contacts.weight
        self.rewards.undesired_contacts.params["threshold"] = self.config_summary.reward.undesired_contacts.threshold

        # costs
        self.costs.prob_joint_pos.weight = self.config_summary.cost.prob_joint_pos.weight
        self.costs.prob_joint_pos.params["margin"] = self.config_summary.cost.prob_joint_pos.margin
        self.costs.prob_joint_pos.params["limit"] = self.config_summary.cost.prob_joint_pos.limit
        self.costs.prob_joint_pos.params["asset_cfg"] = leg_joint_cfg

        self.costs.prob_joint_vel.weight = self.config_summary.cost.prob_joint_vel.weight
        self.costs.prob_joint_vel.params["limit"] = self.config_summary.cost.prob_joint_vel.velocity_limit
        self.costs.prob_joint_vel.params["cost_limit"] = self.config_summary.cost.prob_joint_vel.limit
        self.costs.prob_joint_vel.params["asset_cfg"] = leg_joint_cfg

        self.costs.prob_joint_torque.weight = self.config_summary.cost.prob_joint_torque.weight
        self.costs.prob_joint_torque.params["limit"] = self.config_summary.cost.prob_joint_torque.torque_limit
        self.costs.prob_joint_torque.params["cost_limit"] = self.config_summary.cost.prob_joint_torque.limit
        self.costs.prob_joint_torque.params["asset_cfg"] = leg_joint_cfg

        self.costs.prob_body_contact.weight = self.config_summary.cost.prob_body_contact.weight
        self.costs.prob_body_contact.params["foot_body_names"] = ROBOT_FOOT_NAMES
        self.costs.prob_body_contact.params["threshold"] = self.config_summary.cost.prob_body_contact.contact_force_threshold
        self.costs.prob_body_contact.params["limit"] = self.config_summary.cost.prob_body_contact.limit

        self.costs.prob_com_frame.weight = self.config_summary.cost.prob_com_frame.weight
        self.costs.prob_com_frame.params["height_range"] = self.config_summary.cost.prob_com_frame.height_range
        self.costs.prob_com_frame.params["max_angle_rad"] = self.config_summary.cost.prob_com_frame.max_angle_rad
        self.costs.prob_com_frame.params["cost_limit"] = self.config_summary.cost.prob_com_frame.limit
        self.costs.prob_com_frame.params["terrain_sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.costs.prob_com_frame.params["height_offset"] = 0.0

        self.costs.prob_gait_pattern.weight = self.config_summary.cost.prob_gait_pattern.weight
        self.costs.prob_gait_pattern.params["foot_body_names"] = ROBOT_FOOT_NAMES
        self.costs.prob_gait_pattern.params["gait_frequency"] = self.config_summary.cost.prob_gait_pattern.gait_frequency
        self.costs.prob_gait_pattern.params["min_frequency"] = self.config_summary.cost.prob_gait_pattern.min_frequency
        self.costs.prob_gait_pattern.params["max_frequency"] = self.config_summary.cost.prob_gait_pattern.max_frequency
        self.costs.prob_gait_pattern.params["max_command_speed"] = (
            self.config_summary.cost.prob_gait_pattern.max_command_speed
        )
        self.costs.prob_gait_pattern.params["frequency_scale"] = (
            self.config_summary.cost.prob_gait_pattern.frequency_scale
        )
        self.costs.prob_gait_pattern.params["command_name"] = "base_velocity"
        self.costs.prob_gait_pattern.params["min_command_speed"] = (
            self.config_summary.cost.prob_gait_pattern.min_command_speed
        )
        self.costs.prob_gait_pattern.params["min_base_speed"] = (
            self.config_summary.cost.prob_gait_pattern.min_base_speed
        )
        self.costs.prob_gait_pattern.params["asset_cfg"] = SceneEntityCfg("robot")
        self.costs.prob_gait_pattern.params["phase_offsets"] = self.config_summary.cost.prob_gait_pattern.phase_offsets
        self.costs.prob_gait_pattern.params["stance_ratio"] = self.config_summary.cost.prob_gait_pattern.stance_ratio
        self.costs.prob_gait_pattern.params["contact_threshold"] = (
            self.config_summary.cost.prob_gait_pattern.contact_force_threshold
        )
        self.costs.prob_gait_pattern.params["limit"] = self.config_summary.cost.prob_gait_pattern.limit

        self.costs.orthogonal_velocity.weight = self.config_summary.cost.orthogonal_velocity.weight
        self.costs.orthogonal_velocity.params["limit"] = self.config_summary.cost.orthogonal_velocity.limit

        self.costs.contact_velocity.weight = self.config_summary.cost.contact_velocity.weight
        self.costs.contact_velocity.params["foot_body_names"] = ROBOT_FOOT_NAMES
        self.costs.contact_velocity.params["contact_threshold"] = self.config_summary.cost.contact_velocity.contact_force_threshold
        self.costs.contact_velocity.params["limit"] = self.config_summary.cost.contact_velocity.limit

        self.costs.foot_clearance.weight = self.config_summary.cost.foot_clearance.weight
        self.costs.foot_clearance.params["foot_body_names"] = ROBOT_FOOT_NAMES
        self.costs.foot_clearance.params["min_height"] = self.config_summary.cost.foot_clearance.min_height
        self.costs.foot_clearance.params["height_offset"] = 0.0
        self.costs.foot_clearance.params["contact_threshold"] = self.config_summary.cost.contact_velocity.contact_force_threshold
        self.costs.foot_clearance.params["gait_frequency"] = self.config_summary.cost.prob_gait_pattern.gait_frequency
        self.costs.foot_clearance.params["phase_offsets"] = self.config_summary.cost.prob_gait_pattern.phase_offsets
        self.costs.foot_clearance.params["stance_ratio"] = self.config_summary.cost.prob_gait_pattern.stance_ratio
        self.costs.foot_clearance.params["command_name"] = "base_velocity"
        self.costs.foot_clearance.params["min_command_speed"] = self.config_summary.cost.foot_clearance.min_command_speed
        self.costs.foot_clearance.params["min_base_speed"] = self.config_summary.cost.foot_clearance.min_base_speed
        self.costs.foot_clearance.params["terrain_sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.costs.foot_clearance.params["limit"] = self.config_summary.cost.foot_clearance.limit

        self.costs.foot_height_limit.weight = self.config_summary.cost.foot_height_limit.weight
        self.costs.foot_height_limit.params["foot_body_names"] = ROBOT_FOOT_NAMES
        self.costs.foot_height_limit.params["height_offset"] = 0.0
        self.costs.foot_height_limit.params["terrain_sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.costs.foot_height_limit.params["limit"] = self.config_summary.cost.foot_height_limit.limit

        self.costs.symmetric.weight = self.config_summary.cost.symmetric.weight
        self.costs.symmetric.params["joint_pair_indices"] = self.config_summary.cost.symmetric.joint_pairs
        self.costs.symmetric.params["action_pair_indices"] = self.config_summary.cost.symmetric.joint_pairs
        self.costs.symmetric.params["asset_cfg"] = leg_joint_cfg
        self.costs.symmetric.params["command_name"] = "base_velocity"
        self.costs.symmetric.params["min_command_speed"] = self.config_summary.cost.symmetric.min_command_speed
        self.costs.symmetric.params["min_base_speed"] = self.config_summary.cost.symmetric.min_base_speed
        self.costs.symmetric.params["limit"] = self.config_summary.cost.symmetric.limit
        # Symmetry cost is computed on-policy using mirrored observations; disable env-side proxy.
        self.costs.symmetric.weight = 0.0

        self.costs.base_contact_force.weight = self.config_summary.cost.base_contact_force.weight
        self.costs.base_contact_force.params["sensor_cfg"] = SceneEntityCfg("contact_forces")
        self.costs.base_contact_force.params["body_names"] = ROBOT_THIGH_BODY_NAMES
        self.costs.base_contact_force.params["threshold"] = (
            self.config_summary.cost.base_contact_force.contact_force_threshold
        )
        self.costs.base_contact_force.params["limit"] = self.config_summary.cost.base_contact_force.limit

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ROBOT_BASE_LINK
        # 提高接触终止阈值，避免轻微触地直接终止训练
        self.terminations.base_contact.params["threshold"] = 50.0

@configclass
class Grq20V2d3FPPOEnvCfg_PLAY(Grq20V2d3FPPOEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        command_ids = dict()
        command_ranges = dict()
        env_start = 0
        for key, item in self.scene.terrain.terrain_generator.sub_terrains.items():
            command_ids[key] = list(range(env_start, env_start + int(item.proportion * self.scene.num_envs)))
            env_start += int(item.proportion * self.scene.num_envs)
            command_ranges[key] = self.config_summary.command.ranges[key]

        self.commands.base_velocity = create_uniform_velocity_command_terrain_cfg(
            command_ids=command_ids,
            ranges=command_ranges,
            lin_x_level=self.config_summary.command.lin_x_level,
            ang_z_level=self.config_summary.command.ang_z_level,
            max_lin_x_level=self.config_summary.command.max_lin_x_level,
            max_ang_z_level=self.config_summary.command.max_ang_z_level,
            heading_control_stiffness=self.config_summary.command.heading_control_stiffness,
        )

        # disable randomization for play
        self.observations.critic_obs.enable_corruption = False
        self.observations.actor_obs.enable_corruption = False
        leg_joint_cfg = SceneEntityCfg("robot", joint_names=ROBOT_LEG_JOINT_NAMES, preserve_order=True)
        # remove random pushing event
        self.events.add_base_mass = None
        self.events.base_com_randomization = None
        self.events.physics_material = None
        self.events.reset_actuator_gains = None
        self.events.reset_robot_joints = None
        self.events.push_robot = None
        self.costs.prob_joint_torque.params["asset_cfg"] = leg_joint_cfg
        self.costs.symmetric.params["asset_cfg"] = leg_joint_cfg
