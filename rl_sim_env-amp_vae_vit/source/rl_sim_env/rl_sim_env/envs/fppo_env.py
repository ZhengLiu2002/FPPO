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

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        if hasattr(self, "_prev_prev_action") and hasattr(self, "action_manager"):
            self._prev_prev_action.copy_(self.action_manager.prev_action)
        obs, rewards, terminated, truncated, extras = super().step(action)
        if extras is None:
            extras = {}
        if self.cost_manager is not None:
            extras["cost"] = self.cost_manager.compute(dt=self.step_dt)
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

    def compute_symmetry_cost(self, policy, obs: torch.Tensor) -> torch.Tensor | None:
        """Compute strict symmetry constraint using policy means and mirrored observations."""
        if obs is None:
            return None
        obs_mirror, _ = mirror_policy_obs_actions(self, obs=obs, actions=None)
        if obs_mirror is None:
            return None
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
                weight = getattr(sym_cfg, "weight", 1.0)
                limit = getattr(sym_cfg, "limit", None)
                if limit is not None:
                    limit_t = torch.as_tensor(limit, device=cost.device, dtype=cost.dtype).abs()
                    eps = torch.finfo(cost.dtype).eps
                    limit_t = torch.clamp(limit_t, min=eps)
                    cost = cost / limit_t
                cost = cost * torch.as_tensor(weight, device=cost.device, dtype=cost.dtype)
        return cost
