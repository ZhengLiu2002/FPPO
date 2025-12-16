import glob
from pathlib import Path

import numpy as np
import torch
from rl_algorithms.rsl_rl.utils.log_print import (
    print_placeholder_end,
    print_placeholder_start,
)


class AMPLoader:
    ROOT_POS_SIZE = 3
    ROOT_ROT_SIZE = 4
    ROOT_LINEAR_VEL_SIZE = 3
    ROOT_ANGULAR_VEL_SIZE = 3
    FOOT_POS_SIZE = 12
    FOOT_VEL_SIZE = 12
    JOINT_POS_SIZE = 12
    JOINT_VEL_SIZE = 12

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + ROOT_POS_SIZE
    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROOT_ROT_SIZE
    ROOT_LINEAR_VEL_START_IDX = ROOT_ROT_END_IDX
    ROOT_LINEAR_VEL_END_IDX = ROOT_LINEAR_VEL_START_IDX + ROOT_LINEAR_VEL_SIZE
    ROOT_ANGULAR_VEL_START_IDX = ROOT_LINEAR_VEL_END_IDX
    ROOT_ANGULAR_VEL_END_IDX = ROOT_ANGULAR_VEL_START_IDX + ROOT_ANGULAR_VEL_SIZE
    FOOT_POS_START_IDX = ROOT_ANGULAR_VEL_END_IDX
    FOOT_POS_END_IDX = FOOT_POS_START_IDX + FOOT_POS_SIZE
    FOOT_VEL_START_IDX = FOOT_POS_END_IDX
    FOOT_VEL_END_IDX = FOOT_VEL_START_IDX + FOOT_VEL_SIZE
    JOINT_POS_START_IDX = FOOT_VEL_END_IDX
    JOINT_POS_END_IDX = JOINT_POS_START_IDX + JOINT_POS_SIZE
    JOINT_VEL_START_IDX = JOINT_POS_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    TOTAL_SIZE = (
        ROOT_POS_SIZE
        + ROOT_ROT_SIZE
        + ROOT_LINEAR_VEL_SIZE
        + ROOT_ANGULAR_VEL_SIZE
        + FOOT_POS_SIZE
        + FOOT_VEL_SIZE
        + JOINT_POS_SIZE
        + JOINT_VEL_SIZE
    )

    root_keys = [
        "root_position_world",
        "root_quaternion_wxyz",
        "root_linear_velocity_base",
        "root_angular_velocity_base",
    ]
    foot_keys = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    joint_keys = [
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
    foot_pos_keys = [foot + "_position_base" for foot in foot_keys]
    foot_vel_keys = [foot + "_velocity_base" for foot in foot_keys]
    joint_pos_keys = [joint + "_q" for joint in joint_keys]
    joint_vel_keys = [joint + "_dq" for joint in joint_keys]
    all_keys = root_keys + foot_pos_keys + foot_vel_keys + joint_pos_keys + joint_vel_keys

    root_position_world_indices = list(range(ROOT_POS_START_IDX, ROOT_POS_END_IDX))
    root_quaternion_indices = list(range(ROOT_ROT_START_IDX, ROOT_ROT_END_IDX))
    root_linear_velocity_base_indices = list(range(ROOT_LINEAR_VEL_START_IDX, ROOT_LINEAR_VEL_END_IDX))
    root_linear_velocity_xy_indices = list(range(ROOT_LINEAR_VEL_START_IDX, ROOT_LINEAR_VEL_END_IDX - 1))
    root_angular_velocity_base_indices = list(range(ROOT_ANGULAR_VEL_START_IDX, ROOT_ANGULAR_VEL_END_IDX))
    root_angular_velocity_yaw_indices = list(range(ROOT_ANGULAR_VEL_START_IDX + 2, ROOT_ANGULAR_VEL_END_IDX))
    foot_position_base_indices = list(range(FOOT_POS_START_IDX, FOOT_POS_END_IDX))
    foot_velocity_base_indices = list(range(FOOT_VEL_START_IDX, FOOT_VEL_END_IDX))
    joint_q_indices = list(range(JOINT_POS_START_IDX, JOINT_POS_END_IDX))
    joint_qd_indices = list(range(JOINT_VEL_START_IDX, JOINT_VEL_END_IDX))
    combined_indices = (
        root_linear_velocity_xy_indices
        + root_angular_velocity_yaw_indices
        + joint_q_indices
        + joint_qd_indices
        + foot_position_base_indices
    )
    # combined_indices = (
    #     root_linear_velocity_xy_indices
    #     + root_angular_velocity_yaw_indices
    #     + joint_q_indices
    #     + joint_qd_indices
    # )

    def __init__(
        self,
        device,
        time_between_frames,
        num_preload_transitions=1000000,
        motion_files=glob.glob("datasets/motion_files2/*"),
    ):
        """
        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames

        # Values to store for each trajectory.
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_duration = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        print_placeholder_start("Loading motions")
        print(f"Loading {len(motion_files)} motions")
        print_placeholder_end()
        self.traj_length = len(motion_files)

        for i, motion_file in enumerate(motion_files):
            # 1) 名称改为直接取 npz 文件名（去掉后缀）
            name = Path(motion_file).stem
            self.trajectory_names.append(name)

            # 2) 打开 npz 而不是 yaml
            npz_data = np.load(motion_file)

            # 3) 依然按 all_keys 构造 motion_data
            motion_data = {}
            for key in AMPLoader.all_keys:
                if key in npz_data:
                    arr = npz_data[key]
                    # 保证一维数组变成 (N,1)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    motion_data[key] = arr
                else:
                    print(f"Warning: key '{key}' not found in {motion_file}")

            # 4) length/weight/dt 也从 npz 中读取
            data_length = int(npz_data["length"])
            weight = float(npz_data["weight"])
            frame_duration = float(npz_data["dt"])

            # 5) 和之前完全一样的位移／旋转处理
            first_root_pos = motion_data["root_position_world"][0].copy()
            for f_i in range(data_length):
                root_pos = motion_data["root_position_world"][f_i]
                root_pos[:2] -= first_root_pos[:2]
                root_pos[2] += 0.026
                motion_data["root_position_world"][f_i] = root_pos

                root_rot = motion_data["root_quaternion_wxyz"][f_i]
                root_rot = self.QuaternionNormalize(root_rot)
                root_rot = self.standardize_quaternion(root_rot)
                motion_data["root_quaternion_wxyz"][f_i] = root_rot

            # 6) 拼接、存入 trajectories_full、idx、weights、durations、num_frames
            concatenated_data_all = np.concatenate(
                [motion_data[key] for key in AMPLoader.all_keys if key in motion_data], axis=1
            )
            self.trajectories_full.append(torch.tensor(concatenated_data_all, dtype=torch.float32, device=device))
            self.trajectory_idxs.append(i)
            self.trajectory_weights.append(weight)
            self.trajectory_frame_durations.append(frame_duration)
            traj_len = (data_length - 1) * frame_duration
            self.trajectory_duration.append(traj_len)
            self.trajectory_num_frames.append(float(data_length))

            # print(f"Loaded {traj_len}s. motion from {motion_file}.")

        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_duration = np.array(self.trajectory_duration)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        print(f"Preloading {num_preload_transitions} transitions")
        traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
        times = self.traj_time_sample_batch(traj_idxs)
        self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
        self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)

        print("Finished preloading")

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(0, (self.trajectory_duration[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_duration[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_duration[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), AMPLoader.ROOT_POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), AMPLoader.ROOT_POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), AMPLoader.ROOT_ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), AMPLoader.ROOT_ROT_SIZE, device=self.device)

        all_frame_amp_starts = torch.zeros(
            len(traj_idxs), AMPLoader.TOTAL_SIZE - AMPLoader.ROOT_POS_SIZE - AMPLoader.ROOT_ROT_SIZE, device=self.device
        )
        all_frame_amp_ends = torch.zeros(
            len(traj_idxs), AMPLoader.TOTAL_SIZE - AMPLoader.ROOT_POS_SIZE - AMPLoader.ROOT_ROT_SIZE, device=self.device
        )

        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][
                :, AMPLoader.ROOT_POS_SIZE + AMPLoader.ROOT_ROT_SIZE : AMPLoader.TOTAL_SIZE
            ]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][
                :, AMPLoader.ROOT_POS_SIZE + AMPLoader.ROOT_ROT_SIZE : AMPLoader.TOTAL_SIZE
            ]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = self.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        # print("combined_indices:",combined_indices)
        for _ in range(num_mini_batch):
            idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
            s = self.preloaded_s[idxs][:, AMPLoader.combined_indices]
            s_next = self.preloaded_s_next[idxs][:, AMPLoader.combined_indices]

            yield s, s_next

    @property
    def observation_dim(self):
        return len(AMPLoader.combined_indices)

    @staticmethod
    def get_root_pos(pose):
        return pose[AMPLoader.ROOT_POS_START_IDX : AMPLoader.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_pos_batch(poses):
        return poses[:, AMPLoader.ROOT_POS_START_IDX : AMPLoader.ROOT_POS_END_IDX]

    @staticmethod
    def get_root_rot(pose):
        return pose[AMPLoader.ROOT_ROT_START_IDX : AMPLoader.ROOT_ROT_END_IDX]

    @staticmethod
    def get_linear_vel_batch(poses):
        return poses[:, AMPLoader.ROOT_LINEAR_VEL_START_IDX : AMPLoader.ROOT_LINEAR_VEL_END_IDX]

    @staticmethod
    def get_angular_vel_batch(poses):
        return poses[:, AMPLoader.ROOT_ANGULAR_VEL_START_IDX : AMPLoader.ROOT_ANGULAR_VEL_END_IDX]

    @staticmethod
    def get_root_rot_batch(poses):
        return poses[:, AMPLoader.ROOT_ROT_START_IDX : AMPLoader.ROOT_ROT_END_IDX]

    @staticmethod
    def get_joint_pos_batch(poses):
        return poses[:, AMPLoader.JOINT_POS_START_IDX : AMPLoader.JOINT_POS_END_IDX]

    @staticmethod
    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX : AMPLoader.JOINT_VEL_END_IDX]

    @staticmethod
    def get_foot_pos_batch(poses):
        return poses[:, AMPLoader.FOOT_POS_START_IDX : AMPLoader.FOOT_POS_END_IDX]

    def QuaternionNormalize(self, q):
        """Normalizes the quaternion to length 1.

        Divides the quaternion by its magnitude.  If the magnitude is too
        small, returns the quaternion identity value (1.0).

        Args:
        q: A quaternion to be normalized.

        Raises:
        ValueError: If input quaternion has length near zero.

        Returns:
        A quaternion with magnitude 1 in a numpy array [x, y, z, w].

        """
        q_norm = np.linalg.norm(q)
        if np.isclose(q_norm, 0.0):
            raise ValueError(f"Quaternion may not be zero in QuaternionNormalize: |q| = {q_norm:f}, q = {q}")
        return q / q_norm

    def standardize_quaternion(self, q):
        """Returns a quaternion where q.w >= 0 to remove redundancy due to q = -q.

        Args:
        q: A quaternion to be standardized.

        Returns:
        A quaternion with q.w >= 0.

        """
        if q[-1] < 0:
            q = -q
        return q

    def quaternion_slerp(self, q0, q1, fraction, spin=0, shortestpath=True):
        """Batch quaternion spherical linear interpolation."""
        _EPS = np.finfo(float).eps * 4.0
        out = torch.zeros_like(q0)

        zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
        ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
        out[zero_mask] = q0[zero_mask]
        out[ones_mask] = q1[ones_mask]

        d = torch.sum(q0 * q1, dim=-1, keepdim=True)
        dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
        out[dist_mask] = q0[dist_mask]

        if shortestpath:
            d_old = torch.clone(d)
            d = torch.where(d_old < 0, -d, d)
            q1 = torch.where(d_old < 0, -q1, q1)

        angle = torch.acos(d) + spin * torch.pi
        angle_mask = (torch.abs(angle) < _EPS).squeeze() | torch.isnan(angle.squeeze())
        out[angle_mask] = q0[angle_mask]

        final_mask = torch.logical_or(zero_mask, ones_mask)
        final_mask = torch.logical_or(final_mask, dist_mask)
        final_mask = torch.logical_or(final_mask, angle_mask)
        final_mask = torch.logical_not(final_mask)

        isin = 1.0 / angle
        q0 *= torch.sin((1.0 - fraction) * angle) * isin
        q1 *= torch.sin(fraction * angle) * isin
        q0 += q1
        out[final_mask] = q0[final_mask]
        return out
