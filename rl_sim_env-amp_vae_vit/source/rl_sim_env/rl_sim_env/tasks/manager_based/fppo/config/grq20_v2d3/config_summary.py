import math
import os

from rl_sim_env import RL_SIM_ENV_ROOT_DIR
from rl_sim_env.tasks.manager_based.common.mdp import UniformVelocityCommandTerrainCfg

RL_SIM_ENV_ASSETS_DIR = os.path.join(RL_SIM_ENV_ROOT_DIR, "assets")

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
from rl_algorithms.rsl_rl_wrapper import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

ROBOT_BASE_LINK = "base_link"
ROBOT_FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
ROBOT_LEG_JOINT_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(
            RL_SIM_ENV_ASSETS_DIR, "robots", "galileo_grq20_v2d3", "grq20_v2d3_default.urdf"
        ),
        usd_dir=os.path.join(RL_SIM_ENV_ASSETS_DIR, "robots", "galileo_grq20_v2d3"),
        usd_file_name="grq20_v2d3_default.usd",
        force_usd_conversion=True,
        make_instanceable=True,
        fix_base=False,
        root_link_name=None,
        link_density=0.0,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0),
        ),
        collision_from_visuals=False,
        collider_type="convex_hull",
        self_collision=False,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "FL_hip_joint": -0.05,
            "FL_thigh_joint": 0.795,
            "FL_calf_joint": -1.61,
            "FR_hip_joint": 0.05,
            "FR_thigh_joint": 0.795,
            "FR_calf_joint": -1.61,
            "RL_hip_joint": -0.05,
            "RL_thigh_joint": 0.795,
            "RL_calf_joint": -1.61,
            "RR_hip_joint": 0.05,
            "RR_thigh_joint": 0.795,
            "RR_calf_joint": -1.61,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    # todo_liz: friction=0.05, armature=0.01
    actuators={
        "base_legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=120.0,
            velocity_limit=16.0,
            stiffness=70.0,
            damping=2.0,
            friction=0.05,
            armature=0.01,
            min_delay=0,
            max_delay=6,
        ),
    },
)


@configclass
class ConfigSummary:

    class general:
        decimation = 4
        episode_length_s = 20.0
        render_interval = 4

    class sim:
        dt = 0.0025

    class env:
        num_envs = 4096
        # Policy controls the 12 leg joints only.
        num_actions = 12
        # base_ang_vel*3 + projected_gravity*3 + joint_pos*14 + joint_vel*14 + actions*12 + commands*3
        num_actor_obs = 49
        # base_lin_vel*3 + base_ang_vel*3 + projected_gravity*3 + commands*3 + joint_pos*14 + joint_vel*14 + actions*12 + height_scan*187
        num_critic_obs = 239
        clip_actions = 1.0
        clip_obs = 100.0

    class cost:
        class prob_joint_pos:
            weight = 1.0
            margin = -0.05
            limit = 1.0

        class prob_joint_vel:
            weight = 1.0
            velocity_limit = 16.0
            limit = 1.0

        class prob_joint_torque:
            weight = 1.0
            torque_limit = 120.0
            limit = 1.0

        class prob_body_contact:
            weight = 0.1
            contact_force_threshold = 10.0
            limit = 5.0

        class prob_com_frame:
            weight = 0.1
            height_range = (0.1, 1.1)
            max_angle_rad = 1.0
            limit = 5.0

        class prob_gait_pattern:
            weight = 0.1
            gait_frequency = 1.5
            min_frequency = None
            max_frequency = None
            max_command_speed = None
            frequency_scale = 0.0
            min_command_speed = None
            min_base_speed = None
            stance_ratio = 0.5
            phase_offsets = [0.0, 0.5, 0.5, 0.0]
            contact_force_threshold = 1.0
            limit = 5.0

        class orthogonal_velocity:
            weight = 0.3
            limit = 3.0

        class contact_velocity:
            weight = 1.0
            contact_force_threshold = 1.0
            limit = 0.8

        class foot_clearance:
            weight = 1.0
            min_height = None
            limit = 0.08

        class foot_height_limit:
            weight = 0.3
            limit = 0.4

        class symmetric:
            weight = 0.1
            limit = 5.0
            min_command_speed = None
            min_base_speed = None
            joint_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]

        total_limit = 25.0

    class command:
        lin_x_level: float = 0.0
        max_lin_x_level: float = 5.0
        ang_z_level: float = 0.0
        max_ang_z_level: float = 5.0

        heading_control_stiffness = 0.5

        ranges = {
            "pyramid_stairs": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(-0.5, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(-0.8, 0.8),
                max_curriculum_ang_z=(-1.0, 1.0),
            ),
            "pyramid_stairs_inv": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(0.0, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(0.0, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(0.0, 0.8),
                max_curriculum_ang_z=(-1.0, 1.0),
            ),
            "boxes": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(0.0, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(0.0, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(0.0, 0.8),
                max_curriculum_ang_z=(-1.0, 1.0),
            ),
            "random_rough": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(-0.5, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(-1.0, 1.0),
                max_curriculum_ang_z=(-1.0, 1.0),
            ),
            "hf_pyramid_slope": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(-0.5, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(-1.0, 1.0),
                max_curriculum_ang_z=(-1.0, 1.0),
            ),
            "hf_pyramid_slope_inv": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(-0.5, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(-1.0, 1.0),
                max_curriculum_ang_z=(-1.0, 1.0),
            ),
            "plane_run": UniformVelocityCommandTerrainCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.25, 0.25),
                heading=(-math.pi / 2, math.pi / 2),
                heading_command_prob=1.0,
                yaw_command_prob=0.0,
                standing_command_prob=0.0,
                start_curriculum_lin_x=(-0.5, 0.5),
                start_curriculum_ang_z=(-0.25, 0.25),
                max_curriculum_lin_x=(-1.2, 1.2),
                max_curriculum_ang_z=(-1.5, 1.5),
            ),
            # "plane_yaw": UniformVelocityCommandTerrainCfg.Ranges(
            #     lin_vel_x=(0.0, 0.0),
            #     lin_vel_y=(0.0, 0.0),
            #     ang_vel_z=(-0.25, 0.25),
            #     heading=(-math.pi / 2, math.pi / 2),
            #     heading_command_prob=0.0,
            #     yaw_command_prob=0.05,
            #     standing_command_prob=0.0,
            #     start_curriculum_lin_x=(0.0, 0.0),
            #     start_curriculum_ang_z=(-0.25, 0.25),
            #     max_curriculum_lin_x=(0.0, 0.0),
            #     max_curriculum_ang_z=(-1.5, 1.5),
            # ),
            # "plane_stand": UniformVelocityCommandTerrainCfg.Ranges(
            #     lin_vel_x=(0.0, 0.0),
            #     lin_vel_y=(0.0, 0.0),
            #     ang_vel_z=(0.0, 0.0),
            #     heading=(-math.pi / 2, math.pi / 2),
            #     heading_command_prob=0.0,
            #     yaw_command_prob=0.05,
            #     standing_command_prob=0.0,
            #     start_curriculum_lin_x=(0.0, 0.0),
            #     start_curriculum_ang_z=(0.0, 0.0),
            #     max_curriculum_lin_x=(0.0, 0.0),
            #     max_curriculum_ang_z=(0.0, 0.0),
            # ),
        }

    class action:
        scale = 0.4

    class observation:
        class delay:
            min_delay = 0
            max_delay = 3

        class scale:
            base_lin_vel = 2.0
            base_ang_vel = 0.25
            projected_gravity = 1.0
            vel_command = (2.0, 2.0, 0.25)
            joint_pos = 1.0
            joint_vel = 0.05
            height_measurements = 5.0
            random_mass = 0.2
            random_material = 1.0
            random_com = 5.0

        class noise:
            base_lin_vel = 0.1
            base_ang_vel = 0.3
            projected_gravity = 0.05
            joint_pos = 0.03
            joint_vel = 1.5

        class clip:
            height_measurements = (-1.0, 1.0)

    class event:
        randomize_base_mass = (-3.0, 5.0)
        randomize_base_com = {"x": (-0.05, 0.05), "y": (-0.03, 0.03), "z": (-0.03, 0.05)}
        randomize_static_friction = (0.25, 1.2)
        randomize_dynamic_friction = (0.25, 1.2)
        randomize_restitution = (0.0, 1.0)
        reset_base_pose = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}
        reset_base_velocity = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        reset_robot_joints = (-0.2, 0.2)
        randomize_actuator_kp_gains = (0.8, 1.2)
        randomize_actuator_kd_gains = (0.8, 1.2)
        randomize_actuator_kt_gains = (0.8, 1.2)
        push_robot_vel = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
        # staged curriculum for push velocity (common_step_counter based)
        push_robot_schedule = [
            {"steps": 0, "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
            {"steps": 100000, "velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}},
            {"steps": 300000, "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        ]

    class reward:
        class command_tracking:
            weight = 10
            kappa_lin = 1.5
            kappa_ang = 1.5

        class joint_torque:
            weight = 1e-6
            ref_mass = None
            ref_weight = 1.0

        class action_smoothness:
            weight = 5e-4
            action_diff_weight = 1.0
            action_diff2_weight = 1.0
            joint_vel_weight = 0.0


@configclass
class FPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "grq20_v2d3_fppo"
    wandb_project = "isaaclab-fppo"
    clip_actions = 1.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.4,
        noise_std_type="scalar",
        min_std=0.05,
        max_std=1.0,
        action_mean_clip=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        name="fppo",
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        cost_limit=ConfigSummary.cost.total_limit,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
