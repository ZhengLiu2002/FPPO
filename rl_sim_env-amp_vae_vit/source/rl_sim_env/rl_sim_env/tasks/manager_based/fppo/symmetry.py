from __future__ import annotations

import torch


def mirror_policy_obs_actions(env, obs: torch.Tensor | None = None, actions: torch.Tensor | None = None):
    """Apply left-right symmetry to policy observations/actions for GRQ20-style quadrupeds."""
    obs_mirror = None
    actions_mirror = None

    action_dim = _get_action_dim(env)
    action_pairs = _get_action_pairs(env)
    hip_indices = _get_hip_indices(env)

    if obs is not None:
        obs_mirror = _mirror_policy_obs(obs, action_dim, action_pairs, hip_indices, env)
    if actions is not None:
        actions_mirror = _mirror_joint_data(actions, action_pairs, hip_indices)
    return obs_mirror, actions_mirror


def _get_action_dim(env) -> int:
    if hasattr(env, "action_manager"):
        return env.action_manager.total_action_dim
    if hasattr(env, "num_actions"):
        return int(env.num_actions)
    return 0


def _get_action_pairs(env) -> list[tuple[int, int]]:
    pairs = []
    if hasattr(env, "cfg") and hasattr(env.cfg, "config_summary"):
        pairs = getattr(env.cfg.config_summary.cost.symmetric, "joint_pairs", [])
    return list(pairs) if pairs is not None else []


def _get_hip_indices(env) -> list[int]:
    if not hasattr(env, "cfg"):
        return []
    actions_cfg = getattr(env.cfg, "actions", None)
    if actions_cfg is None or not hasattr(actions_cfg, "joint_pos"):
        return []
    joint_names = getattr(actions_cfg.joint_pos, "joint_names", None)
    if not joint_names:
        return []
    return [idx for idx, name in enumerate(joint_names) if "hip" in name]


def _mirror_policy_obs(
    obs: torch.Tensor, action_dim: int, action_pairs: list[tuple[int, int]], hip_indices: list[int], env
) -> torch.Tensor:
    obs = obs.clone()
    obs_dim = obs.shape[1]
    base_dim = 9
    cmd_dim = 3
    joint_dim = (obs_dim - base_dim - cmd_dim - action_dim) // 2
    if joint_dim <= 0 or (base_dim + cmd_dim + action_dim + 2 * joint_dim) != obs_dim:
        if not getattr(env, "_warn_symmetry_layout", False):
            print("[WARN] Symmetry obs layout mismatch; skipping symmetry transform.")
            setattr(env, "_warn_symmetry_layout", True)
        return obs

    # base_lin_vel
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([1.0, -1.0, 1.0], device=obs.device, dtype=obs.dtype)
    # base_ang_vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1.0, 1.0, -1.0], device=obs.device, dtype=obs.dtype)
    # projected_gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1.0, -1.0, 1.0], device=obs.device, dtype=obs.dtype)

    pos_start = 9
    vel_start = pos_start + joint_dim
    act_start = vel_start + joint_dim
    cmd_start = act_start + action_dim

    obs[:, pos_start:vel_start] = _mirror_joint_data(obs[:, pos_start:vel_start], action_pairs, hip_indices)
    obs[:, vel_start:act_start] = _mirror_joint_data(obs[:, vel_start:act_start], action_pairs, hip_indices)
    obs[:, act_start:cmd_start] = _mirror_joint_data(obs[:, act_start:cmd_start], action_pairs, hip_indices)
    obs[:, cmd_start : cmd_start + 3] = obs[:, cmd_start : cmd_start + 3] * torch.tensor(
        [1.0, -1.0, -1.0], device=obs.device, dtype=obs.dtype
    )

    return obs


def _mirror_joint_data(
    joint_data: torch.Tensor, action_pairs: list[tuple[int, int]], hip_indices: list[int]
) -> torch.Tensor:
    if not action_pairs:
        return joint_data
    joint_data_swapped = joint_data.clone()
    for left_idx, right_idx in action_pairs:
        if left_idx >= joint_data.shape[1] or right_idx >= joint_data.shape[1]:
            continue
        joint_data_swapped[:, left_idx] = joint_data[:, right_idx]
        joint_data_swapped[:, right_idx] = joint_data[:, left_idx]
    for idx in hip_indices:
        if idx < joint_data_swapped.shape[1]:
            joint_data_swapped[:, idx] *= -1.0
    return joint_data_swapped
