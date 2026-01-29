# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, RayCaster
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply_yaw, quat_rotate_inverse, yaw_quat
from rl_sim_env.tasks.manager_based.common.mdp.utils.warp import rasterize_voxels

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


"""
Root state.
"""


def base_lin_xy_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, :2]


def base_ang_yaw_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b[:, 2].unsqueeze(-1)


def foot_positions(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Foot positions from the base link frame."""
    sensor: FrameTransformer = env.scene.sensors[sensor_cfg.name]
    return sensor.data.target_pos_source.flatten(start_dim=1)


def push_vel(env: ManagerBasedEnv) -> torch.Tensor:
    if not hasattr(env, "event_push_vel_buf"):
        num_envs = env.scene.num_envs
        device = getattr(env, "device", torch.device("cpu"))
        env.event_push_vel_buf = torch.zeros((num_envs, 2), device=device, dtype=torch.float32)
    return env.event_push_vel_buf


def random_com(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    coms = asset.root_physx_view.get_coms().clone()
    coms = coms[:, asset_cfg.body_ids, :3].squeeze(1)
    # print("coms", coms)
    return coms.to(env.device)


def random_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()
    mass_obs = masses[:, asset_cfg.body_ids] - asset.data.default_mass[:, asset_cfg.body_ids]
    # print("mass_obs", mass_obs)
    return mass_obs.to(env.device)


def random_material(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    material_obs = asset.root_physx_view.get_material_properties()[:, asset_cfg.body_ids, :]
    material_obs = material_obs.reshape(material_obs.shape[0], -1)
    # print("material_obs", material_obs)
    return material_obs.to(env.device)


def generated_commands_scale(env: ManagerBasedRLEnv, command_name: str, scale: tuple[float, ...]) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    command = env.command_manager.get_command(command_name)
    scale_tensor = torch.tensor(scale, dtype=command.dtype, device=command.device)
    return command * scale_tensor


def heading_error(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Heading error between target heading and current base heading."""
    command_term = env.command_manager.get_term(command_name)
    heading_target = getattr(command_term, "heading_target", None)
    robot = getattr(command_term, "robot", None)
    if heading_target is None or robot is None:
        return torch.zeros((env.num_envs, 1), device=env.device)
    error = math_utils.wrap_to_pi(heading_target - robot.data.heading_w)
    return error.unsqueeze(-1)


def action_history(
    env: ManagerBasedEnv, history_length: int = 2, action_name: str | None = None
) -> torch.Tensor:
    """Concatenate the most recent actions into a single vector."""
    if action_name is None:
        action = env.action_manager.action
    else:
        action = env.action_manager.get_term(action_name).raw_actions
    if action is None:
        return torch.zeros((env.num_envs, 0), device=env.device)

    hist = getattr(env, "_action_hist_buffer", None)
    if hist is None or hist.shape[1] != history_length or hist.shape[2] != action.shape[1]:
        hist = torch.zeros(
            (env.num_envs, history_length, action.shape[1]), device=action.device, dtype=action.dtype
        )
        env._action_hist_buffer = hist

    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        reset_ids = reset_buf.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            hist[reset_ids] = 0.0

    if history_length > 1:
        hist[:, 1:] = hist[:, :-1].clone()
    hist[:, 0] = action
    return hist.reshape(env.num_envs, -1)


def voxel_occupancy(env: ManagerBasedEnv) -> torch.Tensor:
    asset: RigidObject = env.scene["robot"]
    e1r_front: RayCaster = env.scene.sensors["e1r_front"]
    e1r_back: RayCaster = env.scene.sensors["e1r_back"]
    mid360_up: RayCaster = env.scene.sensors["mid360_up"]
    lidar_data = torch.cat([e1r_front.data.ray_hits_w, e1r_back.data.ray_hits_w, mid360_up.data.ray_hits_w], dim=1)

    root_pos = asset.data.root_state_w[:, 0:3]
    root_quat = asset.data.root_state_w[:, 3:7]
    test_data = crop_and_collect_points(lidar_data, root_pos, root_quat)
    print(test_data[0].shape)
    # 定义体素的起始和结束位置
    x_start, x_end = -1.0, 1.0
    y_start, y_end = -0.5, 0.5
    z_start, z_end = 0.0, 2.0

    # 计算每个维度上的体素数量
    x_num = int((x_end - x_start) / 0.05) + 1
    y_num = int((y_end - y_start) / 0.05) + 1
    z_num = int((z_end - z_start) / 0.05) + 1
    voxel_num = x_num * y_num * z_num

    # 生成网格点坐标
    x = torch.linspace(x_start, x_end, x_num, device=env.device)
    y = torch.linspace(y_start, y_end, y_num, device=env.device)
    z = torch.linspace(z_start, z_end, z_num, device=env.device)

    # 使用meshgrid生成所有组合
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")

    # 将坐标展平并组合成(3, voxel_num)的形状
    voxel_pos = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=0)

    # 扩展为(env.num_envs, voxel_num, 3)的形状
    voxel_pos = voxel_pos.T.unsqueeze(0).repeat(env.num_envs, 1, 1)
    voxel_pos_w = quat_apply_yaw(root_quat.repeat(1, voxel_num), voxel_pos)
    voxel_pos_w += root_pos.unsqueeze(1)
    # print(voxel_pos_w.shape)

    return test_data[0].unsqueeze(0)


def crop_and_collect_points(
    points: torch.Tensor,  # (N, M, 3)
    pos: torch.Tensor,  # (N, 3)
    quat: torch.Tensor,  # (N, 4)
    x_lower: float = -1.0,
    x_upper: float = 1.0,
    y_lower: float = -0.5,
    y_upper: float = 0.5,
    z_lower: float = -1.0,
    z_upper: float = 1.0,
) -> list[torch.Tensor]:
    """
    对形状 (N, M, 3) 的点云进行有向包围盒裁切，直接返回裁切好的点云列表。

    Args:
        points: (N, M, 3) 大小的张量，表示 N 个环境并行，每个环境 M 个三维点的世界坐标 (x,y,z)。
        O:      (N, 3) 大小的张量，表示每个环境包围盒中心的世界坐标 (x,y,z)。
        quat:   (N, 4) 大小的张量，表示每个环境包围盒或传感器的朝向四元数 (w, x, y, z)。
                其中只会提取 yaw 分量绕 Z 轴做旋转。roll/pitch 会被忽略。
        length: 包围盒在自身局部 X 轴方向的总长度（米）。
        width:  包围盒在自身局部 Y 轴方向的总宽度（米）。
        height: 包围盒在自身局部 Z 轴方向的总高度（米）。

    Returns:
        cropped_pts_list: 长度为 N 的 Python 列表，其中第 i 项是一个形状 (M_i, 3) 的
                          张量，包含第 i 个环境里所有落在有向包围盒内部的点。M_i ≤ M。
    """
    N, M, _ = points.shape

    # 1) 平移：先把 points 从世界坐标系移动到以 O 为原点
    P_rel = points - pos.unsqueeze(1)  # (N, M, 3)

    # 2) 旋转：只绕 Z 轴旋转，利用 quat_apply_yaw
    #    把 (N, M, 3) 展平成 (N*M, 3)，同时把 quat 扩展到 (N*M, 4)
    quat_flat = quat.unsqueeze(1).expand(-1, M, -1).reshape(-1, 4)  # (N*M, 4)
    P_rel_flat = P_rel.reshape(-1, 3)  # (N*M, 3)
    #    调用外部已定义的 quat_apply_yaw (假设已 import)，得到 (N*M, 3)
    P_local_flat = quat_rotate_inverse(yaw_quat(quat_flat), P_rel_flat)  # (N*M, 3)
    #    还原成 (N, M, 3)
    P_local = P_local_flat.view(N, M, 3)  # (N, M, 3)

    # 3) 计算局部坐标范围掩码
    x_loc = P_local[..., 0]  # (N, M)
    y_loc = P_local[..., 1]  # (N, M)
    z_loc = P_local[..., 2]  # (N, M)

    mask_x = (x_loc >= x_lower) & (x_loc <= x_upper)
    mask_y = (y_loc >= y_lower) & (y_loc <= y_upper)
    mask_z = (z_loc >= z_lower) & (z_loc <= z_upper)

    cropped_mask = mask_x & mask_y & mask_z  # (N, M) 的布尔张量

    # 4) 根据掩码逐环境收集点
    cropped_pts_list: list[torch.Tensor] = []
    for i in range(N):
        # points[i]: (M, 3)，cropped_mask[i]: (M,)
        pts_in_box = points[i][cropped_mask[i]]  # 形状 (M_i, 3)
        cropped_pts_list.append(pts_in_box)

    return cropped_pts_list


def height_scan_fix(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # 原始的射线击中 Z 值
    ray_z = sensor.data.ray_hits_w[..., 2]
    # 把 inf 和 nan 全部替换成 0.0
    ray_z = torch.where(torch.isinf(ray_z), 0.0, ray_z)
    ray_z = torch.where(torch.isnan(ray_z), 0.0, ray_z)

    # 计算 height = sensor_height_z - ray_z - offset
    # pos_w[:,2] 的形状是 [batch]，先 unsqueeze 成 [batch,1] 以便广播
    sensor_z = sensor.data.pos_w[:, 2].unsqueeze(1)
    return sensor_z - ray_z - offset


def generate_batched_occupancy(
    pc_list: list[torch.Tensor],  # 长度为 N，每个元素 shape == (m_i, 3)
    nx: int,
    ny: int,
    nz: int,  # 体素网格分辨率
    lx: float,
    ly: float,
    lz: float,  # 立方体物理尺寸（米）
    device: torch.device,
) -> torch.Tensor:
    """
    Args:
        pc_list: 长度为 N 的 Python 列表, pc_list[i].shape == (m_i, 3), dtype=torch.float32.
        nx, ny, nz: 每个环境体素网格的分辨率.
        lx, ly, lz: 立方体在 x, y, z 方向的真实物理长度（单位 米）.
        device:     torch.device (通常是 torch.device("cuda")).

    Returns:
        occupancy: 大小为 (N, nx, ny, nz) 的 torch.uint8 张量.
                   如果第 i 个环境的第 (x,y,z) 体素被占据, 则 occupancy[i, x, y, z] = 1, 否则 0.
    """

    N = len(pc_list)

    # ----------------------------
    # 1) 构造 offsets 长度为 N+1 的 int32 列表/张量
    #    offsets[i] 记录第 i 个环境在“扁平化点数组”中的起始 idx
    #    offsets[N] = sum_i(m_i) = total_points
    # ----------------------------
    offsets = [0] * (N + 1)
    for i in range(N):
        offsets[i + 1] = offsets[i] + int(pc_list[i].shape[0])
    total_points = offsets[-1]  # 所有环境点的总数

    # 2) 把 pc_list 中每个环境 (m_i,3) 拼到 pts_flat (total_points, 3)，再 reshape(-1) -> (total_points*3,)
    pts_flat = torch.empty((total_points, 3), dtype=torch.float32, device=device)
    for i in range(N):
        m_i = pc_list[i].shape[0]
        # 如果 pc_list[i] 不在 GPU 或 dtype 不是 float32，都要先转一下
        pts_flat[offsets[i] : offsets[i] + m_i, :] = pc_list[i].to(device=device, dtype=torch.float32)
    pts_flat = pts_flat.reshape(-1)  # (total_points * 3,)

    # 3) 把 offsets 列表转成 GPU 上的 int32 张量
    offsets_tensor = torch.tensor(offsets, dtype=torch.int32, device=device)  # (N+1,)

    # ----------------------------
    # 4) 分配输出占据图：voxels_flat，shape = (N * nx * ny * nz,), dtype=int32，初始全 0
    # ----------------------------
    voxels_flat = torch.zeros((N * nx * ny * nz,), dtype=torch.int32, device=device)

    # 5) 计算每个体素对应的物理大小：sx = lx/nx, sy = ly/ny, sz = lz/nz
    sx = lx / nx
    sy = ly / ny
    sz = lz / nz

    # 6) 计算包围盒最小角坐标 (ox, oy, oz)
    #    为了让立方体中心对齐到世界原点, 就设 ox = -lx/2, oy = -ly/2, oz = -lz/2
    ox = -lx * 0.5
    oy = -ly * 0.5
    oz = -lz * 0.5

    # ----------------------------
    # 7) 将 PyTorch 张量包装成 Warp Array，并 launch kernel
    # ----------------------------
    with wp.ScopedDevice(device=device):
        # 把 torch.Tensor 转给 Warp
        wp_pts = wp.from_dlpack(pts_flat)  # float32 (total_points*3,)
        wp_offsets = wp.from_dlpack(offsets_tensor)  # int32  (N+1,)
        wp_voxels = wp.from_dlpack(voxels_flat)  # int32  (N*nx*ny*nz,)

        # Launch kernel：并行线程数 = total_points
        wp.launch(
            kernel=rasterize_voxels,
            dim=total_points,  # 每个线程处理一个“全局点编号 idx”
            inputs=[wp_voxels, wp_pts, wp_offsets, nx, ny, nz, ox, oy, oz, sx, sy, sz],
            device=device,
        )

        # kernel 完成后，wp_voxels 已被写入 0/1
        voxels_flat = wp_voxels.to_torch()  # 同步回 PyTorch

    # ----------------------------
    # 8) 把 (N*nx*ny*nz,) reshape 成 (N, nx, ny, nz)，并 cast 为 uint8
    # ----------------------------
    occupancy = voxels_flat.view(N, nx, ny, nz).to(torch.uint8)
    return occupancy
