# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp                       # 导入标准 MDP 库
from isaaclab.managers import EventTermCfg as EventTerm # 导入事件项配置
from isaaclab.managers import SceneEntityCfg            # 导入场景实体配置
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from rl_sim_env.tasks.manager_based.amp_vae.amp_vae_base_env_cfg import AmpVaeEnvCfg
from rl_sim_env.tasks.manager_based.amp_vae.config.grq20_v2d3.config_summary import (
    ROBOT_BASE_LINK,
    ROBOT_CFG,
    ROBOT_FOOT_NAMES,
    ROBOT_LEG_JOINT_NAMES,
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

# Helper: instantiate randomize_rigid_body_mass class lazily to avoid passing class directly to EventTerm.
def randomize_payload_mass_once(
    env,
    env_ids,
    asset_cfg,
    mass_distribution_params,
    operation,
    distribution="uniform",
    recompute_inertia=True,
):
    term = getattr(env, "_randomize_payload_mass_term", None)
    if term is None:
        cfg = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": asset_cfg,
                "mass_distribution_params": mass_distribution_params,
                "operation": operation,
                "distribution": distribution,
                "recompute_inertia": recompute_inertia,
            },
        )
        term = mdp.randomize_rigid_body_mass(cfg=cfg, env=env)
        env._randomize_payload_mass_term = term
    term(env, env_ids, asset_cfg=asset_cfg, mass_distribution_params=mass_distribution_params, operation=operation, distribution=distribution, recompute_inertia=recompute_inertia)

# Backward-compat: hydra configs saved earlier may reference this name.
def _randomize_payload_mass(env, env_ids, **kwargs):
    return randomize_payload_mass_once(env, env_ids, **kwargs)

@configclass
class Grq20V2d3AmpVaeEnvCfg(AmpVaeEnvCfg):
    def __post_init__(self):
        # config summary
        self.config_summary = ConfigSummary

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
        self.observations.actor_obs.base_ang_vel.scale = self.config_summary.observation.scale.base_ang_vel
        self.observations.actor_obs.projected_gravity.scale = self.config_summary.observation.scale.projected_gravity
        self.observations.actor_obs.velocity_commands.params["scale"] = (
            self.config_summary.observation.scale.vel_command
        )

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
        self.observations.actor_obs.joint_pos.scale = self.config_summary.observation.scale.joint_pos
        self.observations.actor_obs.joint_vel.scale = self.config_summary.observation.scale.joint_vel
        # Drop auxiliary critic terms (push_vel, random material/mass/com) to match the 250-dim critic_obs used in training
        self.observations.critic_obs.push_vel = None
        self.observations.critic_obs.random_material = None
        self.observations.critic_obs.random_com = None
        self.observations.critic_obs.random_mass = None
        # Only feed the 12 leg joints into AMP (motion files do not contain arm data)
        leg_joint_cfg = SceneEntityCfg("robot", joint_names=ROBOT_LEG_JOINT_NAMES, preserve_order=True)
        self.observations.amp_obs.joint_pos.params = {"asset_cfg": leg_joint_cfg}
        self.observations.amp_obs.joint_vel.params = {"asset_cfg": leg_joint_cfg}

        # noise
        base_ang_vel_noise = self.config_summary.observation.noise.base_ang_vel
        gravity_noise = self.config_summary.observation.noise.projected_gravity
        joint_pos_noise = self.config_summary.observation.noise.joint_pos
        joint_vel_noise = self.config_summary.observation.noise.joint_vel

        # actor obs
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

        # 只随机腿部关节，机械臂由专门的随机化事件控制
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

        # rewards
        self.rewards.track_lin_vel_xy_exp.weight = self.config_summary.reward.track_lin_vel_xy_exp.weight
        self.rewards.track_lin_vel_xy_exp.params["std"] = self.config_summary.reward.track_lin_vel_xy_exp.std

        self.rewards.track_ang_vel_z_exp.weight = self.config_summary.reward.track_ang_vel_z_exp.weight
        self.rewards.track_ang_vel_z_exp.params["std"] = self.config_summary.reward.track_ang_vel_z_exp.std

        # self.rewards.base_height_l2.weight = self.config_summary.reward.base_height_l2.weight
        # self.rewards.base_height_l2.params["target_height"] = self.config_summary.reward.base_height_l2.target_height
        # self.rewards.base_height_l2.params["asset_cfg"].body_names = ROBOT_BASE_LINK

        self.rewards.orientation_l2.weight = self.config_summary.reward.orientation_l2.weight
        self.rewards.orientation_l2.params["asset_cfg"].body_names = ROBOT_BASE_LINK

        self.rewards.lin_vel_z_l2.weight = self.config_summary.reward.lin_vel_z_l2.weight

        self.rewards.ang_vel_xy_l2.weight = self.config_summary.reward.ang_vel_xy_l2.weight

        self.rewards.dof_torques_l2.weight = self.config_summary.reward.dof_torques_l2.weight

        self.rewards.dof_vel_l2.weight = self.config_summary.reward.dof_vel_l2.weight

        self.rewards.dof_acc_l2.weight = self.config_summary.reward.dof_acc_l2.weight

        self.rewards.action_rate_l2.weight = self.config_summary.reward.action_rate_l2.weight

        self.rewards.action_smoothness_l2.weight = self.config_summary.reward.action_smoothness_l2.weight

        self.rewards.joint_power.weight = self.config_summary.reward.joint_power.weight
        self.rewards.joint_power.params["asset_cfg"] = leg_joint_cfg

        self.rewards.joint_power_distribution.weight = self.config_summary.reward.joint_power_distribution.weight
        self.rewards.joint_power_distribution.params["asset_cfg"] = leg_joint_cfg
        # limit torque/velocity/acc penalties to leg joints
        self.rewards.dof_torques_l2.params = {"asset_cfg": leg_joint_cfg}
        self.rewards.dof_vel_l2.params = {"asset_cfg": leg_joint_cfg}
        self.rewards.dof_acc_l2.params = {"asset_cfg": leg_joint_cfg}

        self.rewards.feet_air_time.weight = self.config_summary.reward.feet_air_time.weight
        self.rewards.feet_air_time.params["threshold"] = self.config_summary.reward.feet_air_time.threshold
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"

        self.rewards.undesired_contacts.weight = self.config_summary.reward.undesired_contacts.weight
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*thigh", ".*calf"]

        self.rewards.feet_slide.weight = self.config_summary.reward.feet_slide.weight
        self.rewards.feet_slide.params["asset_cfg"].body_names = ".*_foot"
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*_foot"

        # self.rewards.stand_joint_deviation_l1.weight = self.config_summary.reward.stand_joint_deviation_l1.weight
        # self.rewards.stand_joint_deviation_l1.params["asset_cfg"].joint_names = ".*_joint"

        # self.rewards.dof_pos_limits.weight = self.config_summary.reward.dof_pos_limits.weight

        # self.rewards.dof_vel_limits.weight = self.config_summary.reward.dof_vel_limits.weight
        # self.rewards.dof_vel_limits.params["soft_ratio"] = self.config_summary.reward.dof_vel_limits.soft_ratio

        # self.rewards.applied_torque_limits.weight = self.config_summary.reward.applied_torque_limits.weight

        self.rewards.amp_reward.weight = self.config_summary.reward.amp_reward.weight

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ROBOT_BASE_LINK

        # === 新增：训练时的机械臂随机化配置 ===
        
        # 1. 随机旋转角度 (-180度 到 180度)
        # 解释：Reset时，机械臂会在圆周上随机选一个角度。
        # 由于 stiffness=0，它初始化在哪里，就会停在哪里。
        self.events.reset_arm_angle = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-3.14, 3.14),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_base_joint"]),
            },
        )

        # 2. 随机杆长 (0.6m 到 0.8m)
        # 解释：Reset时，机械臂长度随机伸缩。
        self.events.randomize_arm_length = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                # 这里用“相对偏移”到默认值 0.6，所以偏移范围设置为 -0.2~0.2
                "position_range": (-0.2, 0.2),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_length_joint"]),
            },
        )

        # 3. 随机负载质量 (5kg 到 10kg)
        # 解释：Reset时，改变末端负载的物理属性。
        self.events.randomize_payload_mass = EventTerm(
            func=randomize_payload_mass_once,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["arm_load_link"]),
                "mass_distribution_params": (5.0, 10.0),
                # use "abs" to set absolute mass uniformly in range
                "operation": "abs",
            },
        )


@configclass
class Grq20V2d3AmpVaeEnvCfg_PLAY(Grq20V2d3AmpVaeEnvCfg):
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

        # Only feed the leg joints into AMP (motion files do not contain arm data)
        leg_joint_cfg = SceneEntityCfg("robot", joint_names=ROBOT_LEG_JOINT_NAMES, preserve_order=True)
        self.observations.amp_obs.joint_pos.params = {"asset_cfg": leg_joint_cfg}
        self.observations.amp_obs.joint_vel.params = {"asset_cfg": leg_joint_cfg}

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
        self.observations.amp_obs.enable_corruption = False
        # remove random pushing event
        self.events.add_base_mass = None
        self.events.base_com_randomization = None
        self.events.physics_material = None
        self.events.reset_actuator_gains = None
        self.events.reset_robot_joints = None
        self.events.push_robot = None
        self.rewards.amp_reward = None

        # 1. 随机旋转角度 (-180度 到 180度)
        self.events.reset_arm_angle = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-3.14, 3.14),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_base_joint"]),
            },
        )

        # 2. 随机杆长 (60cm 到 80cm)
        # 注意：这里我们修改的是 Prismatic 关节的位置
        self.events.randomize_arm_length = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                # 偏移到默认值 0.6 上，得到 0.6~0.8 的绝对长度
                "position_range": (0.0, 0.2), 
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_length_joint"]),
            },
        )

        # 3. 随机负载质量 (5kg 到 10kg)
        self.events.randomize_payload_mass = EventTerm(
            func=randomize_payload_mass_once,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["arm_load_link"]),
                "mass_distribution_params": (5.0, 10.0),
                "operation": "abs",
            },
        )

class Grq20V2d3AmpVaeEnvCfg_REPLAY_AMPDATA(Grq20V2d3AmpVaeEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.rewards = None
        self.scene.height_scanner = None
        self.observations.critic_obs.height_scan = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.critic_obs.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
