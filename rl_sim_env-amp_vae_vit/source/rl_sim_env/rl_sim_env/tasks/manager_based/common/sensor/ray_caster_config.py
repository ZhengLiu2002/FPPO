from isaaclab.sensors import RayCasterCfg, patterns

CRITIC_HEIGHT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    attach_yaw_only=True,
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)

VOXEL_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    attach_yaw_only=False,
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)

FOOT_SCANNER_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    attach_yaw_only=False,
    pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[0.0, 0.0]),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)

E1R_FRONT_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.41178, 0.0013, 0.02815), rot=(0.953717, 0, 0.3007058, 0)),
    attach_yaw_only=False,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=32, vertical_fov_range=(-45.0, 45.0), horizontal_fov_range=(-60.0, 60.0), horizontal_res=1.0
    ),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)

E1R_BACK_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(-0.4078, 0.0013, 0.022), rot=(0, -0.3007058, 0, 0.953717)),
    attach_yaw_only=False,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=32, vertical_fov_range=(-45.0, 45.0), horizontal_fov_range=(-60.0, 60.0), horizontal_res=1.0
    ),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)

MID360_UP_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.3138, 0.0, 0.125)),
    attach_yaw_only=False,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=32, vertical_fov_range=(-7.0, 52.0), horizontal_fov_range=(-180, 180.0), horizontal_res=1.3
    ),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)
