import math
import os

from rl_sim_env import RL_SIM_ENV_ROOT_DIR
from rl_sim_env.tasks.manager_based.common.mdp import UniformVelocityCommandTerrainCfg

RL_SIM_ENV_ASSETS_DIR = os.path.join(RL_SIM_ENV_ROOT_DIR, "assets")
RL_SIM_ENV_DATASETS_DIR = os.path.join(RL_SIM_ENV_ROOT_DIR, "datasets")

import glob

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
from rl_algorithms.rsl_rl_wrapper import (
    AmpVaeOnPolicyRunnerCfg,
    AmpVaePpoActorCriticCfg,
    AmpVaePpoAlgorithmCfg,
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
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            RL_SIM_ENV_ASSETS_DIR, "robots", "galileo_grq20_v2d3", "grq20_v2d3_with_arm.usd"
        ),
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
            # === 新增：机械臂关节 ===
            "arm_base_joint": 0.0,   # 旋转角度 0
            "arm_length_joint": 0.6, # 初始杆长 60cm 
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
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
        "manipulator": IdealPDActuatorCfg(
            joint_names_expr=["arm_base_joint", "arm_length_joint"],
            effort_limit=5000.0, # 给大一点，防止阻尼力不够
            velocity_limit=100.0,
            stiffness=0.0,       # <--- 关键：设为0，没有回弹力
            damping=1000.0,      # <--- 关键：极大阻尼，像是在胶水里，很难动
            friction=10.0,       # 辅助一点摩擦
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

    class amp:
        motion_files = glob.glob(f"{RL_SIM_ENV_DATASETS_DIR}/grq20_v1d6/0417/npz/*")
        num_preload_transitions = 2000000
        discr_hidden_dims = [1024, 512]

    class env:
        num_envs = 4096
        # 仅让策略控制腿部 12 个关节，机械臂关节锁死（仍在仿真中，但不在动作向量里）
        num_actions = 12
        # base_ang_vel*3 + projected_gravity*3 + joint_pos*14 + joint_vel*14 + actions*12 + commands*3
        num_actor_obs = 49
        # base_lin_vel*3 + base_ang_vel*3 + projected_gravity*3 + commands*3 + joint_pos*14 + joint_vel*14 + actions*12 + height_scan*187
        num_critic_obs = 239
        num_amp_obs = 39  # joint_pos*12 + foot_pos*12 + base_lin_vel*2 + base_ang_vel_yaw*1 + joint_vel*12
        num_vae_obs = 49  # matches actor_obs length
        obs_history_length = 5
        num_vae_out = 20  # code_vel*3 + mass*1 + latent*16
        action_history_length = 3
        clip_actions = 1.0
        clip_obs = 100.0

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
        scale = 0.25

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
        reset_robot_joints = (-0.6, 0.6)
        randomize_actuator_kp_gains = (0.8, 1.2)
        randomize_actuator_kd_gains = (0.8, 1.2)
        randomize_actuator_kt_gains = (0.8, 1.2)
        push_robot_vel = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}

    class reward:
        only_positive_reward = True

        class track_lin_vel_xy_exp:
            weight = 1.5
            std = math.sqrt(0.25)

        class track_ang_vel_z_exp:
            weight = 0.5
            std = math.sqrt(0.25)

        # class base_height_l2:
        #     weight = 0.0
        #     target_height = 0.426

        class orientation_l2:
            weight = -0.1

        class lin_vel_z_l2:
            weight = -2.0

        class ang_vel_xy_l2:
            weight = -0.05

        class dof_torques_l2:
            weight = -2.0e-6

        class dof_vel_l2:
            weight = -2.5e-4

        class dof_acc_l2:
            weight = -1.0e-8

        class action_rate_l2:
            weight = -0.002

        class action_smoothness_l2:
            weight = -0.002

        class joint_power:
            weight = -2.0e-5

        class joint_power_distribution:
            weight = -1.0e-5

        class feet_air_time:
            weight = 1.0
            threshold = 0.35

        class undesired_contacts:
            weight = -0.1

        class stand_joint_deviation_l1:
            weight = -0.5

        class feet_slide:
            weight = -0.1

        # class dof_pos_limits:
        #     weight = 0.0

        # class dof_vel_limits:
        #     weight = 0.0
        #     soft_ratio = 0.9

        # class applied_torque_limits:
        #     weight = 0.0

        class amp_reward:
            weight = 0.5


@configclass
class AmpVaePPORunnerCfg(AmpVaeOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "grq20_v2d3_amp_vae"

    policy = AmpVaePpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # min_normalized_std=[0.01, 0.01, 0.01] * 4,
        min_normalized_std=[0.01] * 14,
    )
    algorithm = AmpVaePpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_replay_buffer_size=1000000,
        amp_disc_grad_penalty=5.0,
        learning_rate_vae=1.0e-3,
        vae_beta=0.01,
        vae_beta_min=1.0e-4,
        vae_beta_max=0.1,
        vae_desired_recon_loss=0.1,
    )
