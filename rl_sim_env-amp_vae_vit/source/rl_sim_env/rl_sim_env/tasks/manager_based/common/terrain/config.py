# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import numpy as np
import scipy.interpolate as interpolate
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.height_field import hf_terrains_cfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@height_field_to_mesh
def random_uniform_terrain_difficulty(difficulty: float, cfg: hf_terrains_cfg.HfRandomUniformTerrainCfg) -> np.ndarray:
    """Generate a terrain whose噪声幅度随 difficulty 变化，且始终保持上下起伏，不会直接成凸台。

    逻辑：
      - difficulty=0 → 完全平坦，所有高度都为 0。
      - 0<difficulty≤1 → 在 [-max_h, +max_h] 区间里做均匀随机采样，其中 max_h = cfg.noise_range[1] * difficulty。
        这样即便 difficulty 很小，也会有正负两个方向的起伏。
      - 如果计算后离散化得到的 height_min == height_max == 0（即幅度太小），则强制让 height_min=-1，height_max=+1，
        保证至少产生一点正负波动，而不是全部为 0。
    """

    # 1. 校验 downsampled_scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            f"Downsampled scale must be ≥ horizontal scale: {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # 2. 强制把 difficulty 截断在 [0,1]
    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    # 3. 当 difficulty = 0 时，直接返回全 0（完全平坦）
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    if difficulty <= 0.0:
        return np.zeros((width_pixels, length_pixels), dtype=np.int16)

    # 4. 计算“连续高度上限” max_h = cfg.noise_range[1] * difficulty
    #    只使用 noise_range[1] 作为最大正向高度；负向高度对称
    real_max = cfg.noise_range[1]
    max_h_continuous = real_max * difficulty
    min_h_continuous = -max_h_continuous  # 对称取反

    # 5. 离散化到“高度索引”空间
    #    vertical_scale = 几米 对应 1 个离散层
    height_min = int(np.floor(min_h_continuous / cfg.vertical_scale))
    height_max = int(np.ceil(max_h_continuous / cfg.vertical_scale))
    # 如果两者计算后反过来了，就交换
    if height_min > height_max:
        height_min, height_max = height_max, height_min

    # 6. 计算“离散步长”——高度索引之间的差距
    height_step = int(np.round(cfg.noise_step / cfg.vertical_scale))
    if height_step < 1:
        raise ValueError(f"noise_step ({cfg.noise_step}) must be at least vertical_scale ({cfg.vertical_scale}).")

    # 7. 如果离散化后幅度太小（height_min == height_max == 0），强制扩展为 [-1, +1]
    #    这样就不会全部采到 0，起码会有 -1, +1 两个可能
    if height_min == 0 and height_max == 0:
        height_min = -1
        height_max = +1

    # 8. 计算横向下采样与全分辨率网格大小
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)

    # 9. 构造离散索引区间 [height_min, height_max]，步长为 height_step
    #    例如 height_min=-3, height_max=3, height_step=2 → [-3, -1, 1, 3]
    height_range = np.arange(height_min, height_max + 1, height_step, dtype=np.int32)
    if height_range.size == 0:
        # 极端保护：如果运算出现意外，比如 height_step 非常大，就至少保留 [-1, 1]
        height_range = np.array([-1, 1], dtype=np.int32)

    # 10. 在下采样网格上做均匀随机采样（产生离散高度索引）
    #     形状是 (width_downsampled, length_downsampled)
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))

    # 11. 插值：把下采样网格扩展到全分辨率
    x_ds = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y_ds = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    interp_func = interpolate.RectBivariateSpline(x_ds, y_ds, height_field_downsampled)

    x_full = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_full = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_full = interp_func(x_full, y_full)

    # 12. 四舍五入并转为 int16
    z_int = np.rint(z_full).astype(np.int16)
    return z_int


@configclass
class HfRandomUniformTerrainDifficultyCfg(terrain_gen.HfTerrainBaseCfg):
    """Configuration for a random uniform height field terrain."""

    function = random_uniform_terrain_difficulty

    noise_range: tuple[float, float] = MISSING
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""
    noise_step: float = MISSING
    """The minimum height (in m) change between two points."""
    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """


PLANE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(100.0, 100.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0,
        ),
    },
)


ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": HfRandomUniformTerrainDifficultyCfg(
            proportion=0.15, noise_range=(0.01, 0.06), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        "plane_run": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.15,
        ),
        # "plane_yaw": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.05,
        # ),
        # "plane_stand": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.05,
        # ),
    },
)
"""Rough terrains configuration."""

ROUGH_TERRAINS_CFG_ORIGINAL = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.025, 0.1), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""

AMP_VAE_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)

AMP_VAE_VIT_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)

AMP_VAE_PERCEPTION_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=1,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)
