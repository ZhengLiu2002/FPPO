# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.managers import RewardManager
from rl_sim_env.tasks.manager_based.fppo.fppo_base_env_cfg import FPPOEnvCfg
from rl_sim_env.tasks.manager_based.fppo.symmetry import mirror_policy_obs_actions


class FPPOEnv(ManagerBasedRLEnv):
    """Manager-based RL environment with cost tracking for CMDP training."""

    cfg: FPPOEnvCfg
    cost_manager: RewardManager | None
    is_vector_env: ClassVar[bool] = True

    def __init__(self, cfg: FPPOEnvCfg, render_mode: str | None = None, **kwargs):
        self.cost_manager = None
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        if hasattr(self, "action_manager"):
            self._prev_prev_action = torch.zeros(
                (self.num_envs, self.action_manager.total_action_dim),
                device=self.device,
                dtype=torch.float32,
            )

    def load_managers(self):
        super().load_managers()
        if getattr(self.cfg, "costs", None) is not None:
            self.cost_manager = RewardManager(self.cfg.costs, self)
            print("[INFO] Cost Manager: ", self.cost_manager)

    def _collect_cost_terms(self) -> dict[str, torch.Tensor] | None:
        if self.cost_manager is None:
            return None
        step_reward = getattr(self.cost_manager, "_step_reward", None)
        term_names = getattr(self.cost_manager, "active_terms", None)
        if step_reward is None or term_names is None:
            return None
        if not torch.is_tensor(step_reward):
            return None
        if step_reward.ndim == 1:
            step_reward = step_reward.unsqueeze(0)
        if step_reward.ndim != 2 or len(term_names) != step_reward.shape[1]:
            return None
        cost_terms: dict[str, torch.Tensor] = {}
        for idx, name in enumerate(term_names):
            cost_terms[str(name)] = step_reward[:, idx]
        return cost_terms

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        if hasattr(self, "_prev_prev_action") and hasattr(self, "action_manager"):
            self._prev_prev_action.copy_(self.action_manager.prev_action)
        obs, rewards, terminated, truncated, extras = super().step(action)
        if extras is None:
            extras = {}
        if self.cost_manager is not None:
            cost = self.cost_manager.compute(dt=self.step_dt)
            if isinstance(cost, dict):
                extras["cost"] = cost
            else:
                cost_terms = self._collect_cost_terms()
                extras["cost"] = cost_terms if cost_terms is not None else cost
        return obs, rewards, terminated, truncated, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if hasattr(self, "_prev_prev_action"):
            self._prev_prev_action[env_ids] = 0.0
        if self.cost_manager is None:
            return
        cost_info = self.cost_manager.reset(env_ids)
        if not cost_info:
            return
        log = self.extras.setdefault("log", {})
        for key, value in cost_info.items():
            if key.startswith("Episode_Reward/"):
                log["Episode_Cost/" + key.split("/", 1)[1]] = value
            else:
                log["Episode_Cost/" + key] = value

    def compute_symmetry_cost(
        self, policy, obs: torch.Tensor, obs_history: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Compute strict symmetry constraint using policy means and mirrored observations."""
        if obs is None:
            return None
        obs_mirror, _ = mirror_policy_obs_actions(self, obs=obs, actions=None)
        if obs_mirror is None:
            return None
        if hasattr(policy, "get_policy_obs") and obs_history is not None:
            obs_full = policy.get_policy_obs(obs, obs_history, deterministic=True)
            obs_mirror_full = policy.get_policy_obs(obs_mirror, obs_history, deterministic=True)
            mu_orig = policy.act_inference(obs_full)
            mu_mirror = policy.act_inference(obs_mirror_full)
        else:
            mu_orig = policy.act_inference(obs)
            mu_mirror = policy.act_inference(obs_mirror)
        _, mu_mirror_mapped = mirror_policy_obs_actions(self, obs=None, actions=mu_mirror)
        if mu_mirror_mapped is None:
            return None
        cost = torch.mean(torch.abs(mu_orig - mu_mirror_mapped), dim=1)
        if hasattr(self.cfg, "config_summary"):
            sym_cfg = getattr(self.cfg.config_summary, "cost", None)
            if sym_cfg is not None:
                sym_cfg = getattr(sym_cfg, "symmetric", None)
            if sym_cfg is not None:
                command_name = getattr(sym_cfg, "command_name", None)
                min_command_speed = getattr(sym_cfg, "min_command_speed", None)
                min_base_speed = getattr(sym_cfg, "min_base_speed", None)
                gate = None
                if command_name is not None and hasattr(self, "command_manager"):
                    cmd = self.command_manager.get_command(command_name)
                    if cmd is not None:
                        cmd_speed = torch.linalg.norm(cmd[:, :2], dim=1)
                        if min_command_speed is not None:
                            gate = cmd_speed >= min_command_speed
                        else:
                            gate = torch.ones_like(cmd_speed, dtype=torch.bool)
                if min_base_speed is not None and hasattr(self, "scene"):
                    asset = None
                    try:
                        asset = self.scene["robot"]
                    except Exception:
                        asset = None
                    if asset is not None:
                        base_speed = torch.linalg.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
                        base_gate = base_speed >= min_base_speed
                        gate = base_gate if gate is None else (gate & base_gate)
                if gate is not None:
                    cost = cost * gate.to(device=cost.device, dtype=cost.dtype)
                weight = getattr(sym_cfg, "weight", 1.0)
                limit = getattr(sym_cfg, "limit", None)
                if limit is not None:
                    limit_t = torch.as_tensor(limit, device=cost.device, dtype=cost.dtype).abs()
                    eps = torch.finfo(cost.dtype).eps
                    limit_t = torch.clamp(limit_t, min=eps)
                    cost = cost / limit_t
                cost = cost * torch.as_tensor(weight, device=cost.device, dtype=cost.dtype)
        return cost
