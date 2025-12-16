from rl_sim_env.tasks.manager_based.common.mdp import (
    UniformVelocityCommandCfg,
    UniformVelocityCommandTerrainCfg,
)


def create_uniform_velocity_command_terrain_cfg(
    command_ids: dict[str, list[int]],
    ranges: dict[str, UniformVelocityCommandTerrainCfg.Ranges],
    lin_x_level: float,
    ang_z_level: float,
    max_lin_x_level: float,
    max_ang_z_level: float,
    heading_control_stiffness: float,
) -> UniformVelocityCommandTerrainCfg:
    base_velocity = UniformVelocityCommandTerrainCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        command_ids=command_ids,
        ranges=ranges,
        lin_x_level=lin_x_level,
        ang_z_level=ang_z_level,
        max_lin_x_level=max_lin_x_level,
        max_ang_z_level=max_ang_z_level,
        heading_control_stiffness=heading_control_stiffness,
    )

    return base_velocity


def create_uniform_velocity_command_cfg(
    rel_standing_envs: float,
    rel_heading_envs: float,
    heading_command: bool,
    heading_control_stiffness: float,
    lin_vel_x: tuple[float, float],
    lin_vel_y: tuple[float, float],
    ang_vel_z: tuple[float, float],
    heading: tuple[float, float],
) -> UniformVelocityCommandCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=rel_standing_envs,
        rel_heading_envs=rel_heading_envs,
        heading_command=heading_command,
        heading_control_stiffness=heading_control_stiffness,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=lin_vel_x, lin_vel_y=lin_vel_y, ang_vel_z=ang_vel_z, heading=heading
        ),
    )

    return base_velocity
