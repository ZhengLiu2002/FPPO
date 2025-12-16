import warp as wp

# -----------------------------------------------------------------------------
# 1) 定义一个 Warp kernel，把所有点"打包"到体素格子里。
#    输入：
#       voxels     : wp.array(dtype=wp.int32)   —— 大小 = N * (nx*ny*nz)，
#                     按照 batch-major 以 (voxel_z * ny + voxel_y) * nx + voxel_x 的顺序存储。
#       pts        : wp.array(dtype=wp.float32) —— 所有环境点云扁平化后的一维数组，
#                     长度 = sum_i(m_i)，其中我们要配合 offsets[] 里记录每个环境点的起始/结束索引。
#       offsets    : wp.array(dtype=wp.int32)   —— 大小 = N+1，offsets[i] 表示第 i 个环境
#                     点在 pts[] 里的起始下标，offsets[i+1]-offsets[i] = m_i。
#       nx, ny, nz : int32，表示体素网格在 x,y,z 方向的分辨率。
#       ox, oy, oz : float32，表示"立方体网格"在世界坐标系的最小角坐标 (也就是包围盒原点)。
#       sx, sy, sz : float32，表示每个体素在 x/y/z 方向的体素大小，sx = lx / nx，等等。
#
#    输出：
#       voxels[...] 在相应的 index 上原子性写 1，表示该 voxel 被占据。
# -----------------------------------------------------------------------------


@wp.kernel
def rasterize_voxels(
    voxels: wp.array(dtype=wp.int32),
    pts: wp.array(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    ox: float,
    oy: float,
    oz: float,
    sx: float,
    sy: float,
    sz: float,
):
    idx = wp.tid()

    # 1) 找到当前线程 idx 属于哪个环境 env
    env = 0
    while idx >= offsets[env + 1]:
        env += 1

    # 2) 读取第 idx 个点 (x, y, z)
    px = pts[3 * idx + 0]
    py = pts[3 * idx + 1]
    pz = pts[3 * idx + 2]

    # 3) 转到体素索引
    ix = int((px - ox) / sx)
    iy = int((py - oy) / sy)
    iz = int((pz - oz) / sz)

    # 4) 如果在范围内，就标记占据
    if (ix >= 0) and (ix < nx) and (iy >= 0) and (iy < ny) and (iz >= 0) and (iz < nz):
        base = env * (nx * ny * nz)
        flat = base + iz * (nx * ny) + iy * nx + ix
        wp.atomic_max(voxels, flat, 1)
