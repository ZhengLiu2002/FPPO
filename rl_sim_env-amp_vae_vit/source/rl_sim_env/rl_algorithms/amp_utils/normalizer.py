from typing import Tuple

import numpy as np
import torch


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class Normalizer(RunningMeanStd):
    def __init__(self, input_dim, epsilon=1e-4, clip_obs=10.0):
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input):
        return np.clip((input - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def normalize_torch(self, input, device):
        mean_torch = torch.tensor(self.mean, device=device, dtype=torch.float32)
        std_torch = torch.sqrt(torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp((input - mean_torch) / std_torch, -self.clip_obs, self.clip_obs)

    def update_normalizer(self, rollouts, expert_loader):
        policy_data_generator = rollouts.feed_forward_generator_amp(None, mini_batch_size=expert_loader.batch_size)
        expert_data_generator = expert_loader.dataset.feed_forward_generator_amp(expert_loader.batch_size)

        for expert_batch, policy_batch in zip(expert_data_generator, policy_data_generator):
            self.update(torch.vstack(tuple(policy_batch) + tuple(expert_batch)).cpu().numpy())


def sync_normalizer(normalizer: Normalizer, device: torch.device):
    """
    正确地将各卡的 RunningMeanStd 统计合并到一起（按一阶矩、二阶矩公式）。
    不再需要传入 world_size，因为 count_t 本身在 all_reduce 后就代表了全局样本数。
    """
    # 把 numpy 数组转换为 tensor
    mean_t = torch.tensor(normalizer.mean, device=device, dtype=torch.float32)
    var_t = torch.tensor(normalizer.var, device=device, dtype=torch.float32)
    count_t = torch.tensor(normalizer.count, device=device, dtype=torch.float32)  # 标量

    # 计算局部 sum 与 sum_sq
    sum_t = mean_t * count_t
    sum_sq_t = (var_t + mean_t.pow(2)) * count_t

    # all_reduce 累加
    torch.distributed.all_reduce(sum_t, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(sum_sq_t, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(count_t, op=torch.distributed.ReduceOp.SUM)

    # 转 numpy 计算全局 mean, var
    count_glob = count_t.cpu().numpy()  # 标量
    sum_glob = sum_t.cpu().numpy()  # 数组，形状 = normalizer.mean.shape
    sum_sq_glob = sum_sq_t.cpu().numpy()  # 数组

    mean_glob = sum_glob / count_glob
    var_glob = sum_sq_glob / count_glob - np.square(mean_glob)

    # 写回 normalizer
    normalizer.mean = mean_glob
    normalizer.var = var_glob
    normalizer.count = count_glob
