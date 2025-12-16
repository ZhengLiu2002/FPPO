# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import rl_sim_env.tasks.manager_based.common.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, RayCasterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # robots
    robot: ArticulationCfg = MISSING

    # ground terrain
    terrain: TerrainImporterCfg = MISSING

    # sensors
    height_scanner: RayCasterCfg = MISSING
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1, track_air_time=True)

    # frame transform
    frame_transform: FrameTransformerCfg = FrameTransformerCfg(prim_path="{ENV_REGEX_NS}/Robot/base")

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity: mdp.UniformVelocityCommandTerrainCfg = MISSING


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class CriticObsCfg(ObsGroup):
        """Observations for critic."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands_scale, params={"command_name": "base_velocity", "scale": (2.0, 2.0, 0.25)}
        )
        height_scan = ObsTerm(func=mdp.height_scan_fix, params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        push_vel = ObsTerm(func=mdp.push_vel)
        random_material = ObsTerm(
            func=mdp.random_material, params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")}
        )
        random_com = ObsTerm(func=mdp.random_com, params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")})
        random_mass = ObsTerm(func=mdp.random_mass, params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ActorObsCfg(ObsGroup):
        """Observations for actor."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands_scale, params={"command_name": "base_velocity", "scale": (2.0, 2.0, 0.25)}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class AmpObsCfg(ObsGroup):
        """Observations for amp."""

        # observation terms (order preserved)
        base_lin_xy_vel = ObsTerm(func=mdp.base_lin_xy_vel)
        base_ang_yaw_vel = ObsTerm(func=mdp.base_ang_yaw_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        foot_positions = ObsTerm(func=mdp.foot_positions, params={"sensor_cfg": SceneEntityCfg("frame_transform")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    critic_obs: CriticObsCfg = CriticObsCfg()
    actor_obs: ActorObsCfg = ActorObsCfg()
    amp_obs: AmpObsCfg = AmpObsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass_plus,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    base_com_randomization = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.25, 1.75),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.1, 0.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains_plus,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "kt_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_obs_xy,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # base_height_l2 = RewTerm(
    #     func=mdp.base_height_l2_fix,
    #     weight=-0.1,
    #     params={
    #         "target_height": 0.426,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #     },
    # )
    orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-7)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    joint_power = RewTerm(
        func=mdp.joint_power, weight=-2.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")}
    )
    joint_power_distribution = RewTerm(
        func=mdp.joint_power_distribution,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.35,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 0.1},
    )
    # stand_joint_deviation_l1 = RewTerm(
    #     func=mdp.stand_joint_deviation_l1,
    #     weight=-1.0,
    #     params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    # )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    # -- optional penalties
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    # dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=0.0)
    # applied_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=0.0)
    # -- amp
    amp_reward = RewTerm(func=mdp.amp_reward, weight=0.5)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_x_command_threshold = CurrTerm(func=mdp.lin_vel_x_command_threshold)
    ang_vel_z_command_threshold = CurrTerm(func=mdp.ang_vel_z_command_threshold)


##
# Environment configuration
##


@configclass
class AmpVaeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the AMP-VAE environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1500, env_spacing=0.1)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
