# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import rl_sim_env.tasks.manager_based.common.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as CostTerm
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
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=1,
        track_air_time=True,
        track_contact_points=False,
        filter_prim_paths_expr=[],
    )

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
            self.concatenate_terms = True

    # observation groups
    critic_obs: CriticObsCfg = CriticObsCfg()
    actor_obs: ActorObsCfg = ActorObsCfg()


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

    command_tracking = RewTerm(
        func=mdp.command_tracking_quadratic,
        weight=10.0,
        params={"command_name": "base_velocity", "kappa_lin": 0.005, "kappa_ang": 0.005},
    )
    joint_torque_l2 = RewTerm(
        func=mdp.joint_torque_l2,
        weight=-0.003,
        params={"asset_cfg": SceneEntityCfg("robot"), "ref_mass": None, "ref_weight": 1.0},
    )
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-0.06,
        params={
            "action_diff_weight": 1.0,
            "action_diff2_weight": 1.0,
            "joint_vel_weight": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class CostsCfg:
    """Cost terms for CMDP-style training."""

    prob_joint_pos = CostTerm(
        func=mdp.joint_pos_prob_constraint,
        weight=1.0,
        params={"margin": 0.0, "limit": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    prob_joint_vel = CostTerm(
        func=mdp.joint_vel_prob_constraint,
        weight=1.0,
        params={"limit": 50.0, "cost_limit": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    prob_joint_torque = CostTerm(
        func=mdp.joint_torque_prob_constraint,
        weight=1.0,
        params={"limit": 100.0, "cost_limit": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    prob_body_contact = CostTerm(
        func=mdp.body_contact_prob_constraint,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "foot_body_names": [],
            "threshold": 1.0,
            "limit": 1.0,
        },
    )
    prob_com_frame = CostTerm(
        func=mdp.com_frame_prob_constraint,
        weight=1.0,
        params={
            "height_range": (0.2, 0.8),
            "max_angle_rad": 0.35,
            "cost_limit": 1.0,
            "terrain_sensor_cfg": None,
            "height_offset": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    prob_gait_pattern = CostTerm(
        func=mdp.gait_pattern_prob_constraint,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "foot_body_names": [],
            "gait_frequency": 1.0,
            "phase_offsets": [0.0, 0.5, 0.5, 0.0],
            "stance_ratio": 0.5,
            "contact_threshold": 1.0,
            "command_name": None,
            "min_frequency": None,
            "max_frequency": None,
            "max_command_speed": None,
            "frequency_scale": 0.0,
            "min_command_speed": None,
            "min_base_speed": None,
            "asset_cfg": SceneEntityCfg("robot"),
            "limit": 1.0,
        },
    )
    orthogonal_velocity = CostTerm(
        func=mdp.orthogonal_velocity_constraint,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit": 1.0},
    )
    contact_velocity = CostTerm(
        func=mdp.contact_velocity_constraint,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "foot_body_names": [],
            "contact_threshold": 1.0,
            "limit": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    foot_clearance = CostTerm(
        func=mdp.foot_clearance_constraint,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "foot_body_names": [],
            "min_height": None,
            "height_offset": 0.0,
            "contact_threshold": 1.0,
            "gait_frequency": 1.0,
            "phase_offsets": None,
            "stance_ratio": 0.5,
            "terrain_sensor_cfg": None,
            "limit": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    foot_height_limit = CostTerm(
        func=mdp.foot_height_limit_constraint,
        weight=1.0,
        params={
            "foot_body_names": [],
            "height_offset": 0.0,
            "terrain_sensor_cfg": None,
            "limit": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    symmetric = CostTerm(
        func=mdp.symmetric_constraint,
        weight=1.0,
        params={
            "joint_pair_indices": [],
            "action_pair_indices": None,
            "asset_cfg": SceneEntityCfg("robot"),
            "include_actions": True,
            "command_name": None,
            "min_command_speed": None,
            "min_base_speed": None,
            "limit": 1.0,
        },
    )


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
class FPPOEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the FPPO environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1500, env_spacing=0.1)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    costs: CostsCfg = CostsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
