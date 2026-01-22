# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose NaN/Inf in FPPO training pipeline (rollout -> returns -> update)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import gymnasium as gym

from isaaclab.app import AppLauncher


def _scan_tensor(name: str, tensor: torch.Tensor, warn_only: bool = False) -> bool:
    if not torch.is_tensor(tensor):
        return False
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    if nan_count or inf_count:
        min_val = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).min().item()
        max_val = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).max().item()
        level = "[WARN]" if warn_only else "[ERROR]"
        print(f"{level} {name}: nan={nan_count} inf={inf_count} min={min_val:.4g} max={max_val:.4g}")
        return True
    return False


def _scan_params(name: str, params) -> bool:
    has_bad = False
    for idx, param in enumerate(params):
        if not torch.is_tensor(param):
            continue
        has_bad |= _scan_tensor(f"{name}[{idx}]", param)
    return has_bad


def _scan_storage(storage) -> bool:
    has_bad = False
    for key in (
        "observations",
        "privileged_observations",
        "actions",
        "rewards",
        "cost_rewards",
        "values",
        "cost_values",
        "actions_log_prob",
        "mu",
        "sigma",
        "returns",
        "advantages",
        "cost_returns",
        "cost_advantages",
    ):
        value = getattr(storage, key, None)
        if value is not None:
            has_bad |= _scan_tensor(f"storage.{key}", value)
    if hasattr(storage, "sigma"):
        sigma = storage.sigma
        if torch.is_tensor(sigma) and (sigma <= 0).any():
            min_val = sigma.min().item()
            print(f"[WARN] storage.sigma has non-positive values, min={min_val:.4g}")
    return has_bad


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose NaN/Inf in FPPO rollout/update.")
    parser.add_argument("--task", type=str, required=True, help="Gym task name.")
    parser.add_argument("--num_envs", type=int, default=256, help="Number of environments.")
    parser.add_argument("--steps", type=int, default=24, help="Steps per environment per rollout.")
    parser.add_argument("--iterations", type=int, default=2, help="Number of rollout/update iterations.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the environment.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    # clear out sys.argv for AppLauncher/Hydra
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Ensure local source tree is on sys.path
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    workspace_root = project_root.parent
    source_root = project_root / "source"
    for path in (source_root, source_root / "rl_sim_env"):
        if path.exists():
            sys.path.insert(0, str(path))
    os.environ.setdefault("RL_SIM_ENV_ROOT_DIR", str(workspace_root))

    # Import IsaacLab modules after SimulationApp is instantiated.
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
    import rl_sim_env.tasks  # noqa: F401

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
        rebuild_fn = getattr(env_cfg, "_rebuild_command_cfg", None)
        if callable(rebuild_fn):
            rebuild_fn(env_cfg.scene.num_envs)
    agent_cfg.num_steps_per_env = args_cli.steps
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = env.unwrapped

    # local imports after sys.path is set
    from rl_algorithms.rsl_rl_wrapper import RslRlVecEnvWrapper
    from rl_algorithms.rsl_rl.runners import OnPolicyRunner

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.train_mode()

    obs, extras = env.get_observations()
    privileged_obs = extras["observations"].get(runner.privileged_obs_type, obs)
    obs, privileged_obs = obs.to(runner.device), privileged_obs.to(runner.device)

    for iteration in range(1, args_cli.iterations + 1):
        print(f"[INFO] Iteration {iteration}/{args_cli.iterations}")
        with torch.inference_mode():
            for step in range(1, args_cli.steps + 1):
                actions = runner.alg.act(obs, privileged_obs)
                transition = runner.alg.transition
                _scan_tensor(f"iter{iteration}.step{step}.actions", actions)
                _scan_tensor(f"iter{iteration}.step{step}.action_mean", transition.action_mean)
                _scan_tensor(f"iter{iteration}.step{step}.action_sigma", transition.action_sigma)
                _scan_tensor(f"iter{iteration}.step{step}.actions_log_prob", transition.actions_log_prob)
                _scan_tensor(f"iter{iteration}.step{step}.values", transition.values)
                _scan_tensor(f"iter{iteration}.step{step}.cost_values", transition.cost_values)

                obs, rewards, dones, infos = env.step(actions.to(env.device))
                obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                obs = runner.obs_normalizer(obs)
                if runner.privileged_obs_type is not None:
                    privileged_obs = runner.privileged_obs_normalizer(
                        infos["observations"][runner.privileged_obs_type].to(runner.device)
                    )
                else:
                    privileged_obs = obs

                _scan_tensor(f"iter{iteration}.step{step}.obs", obs)
                _scan_tensor(f"iter{iteration}.step{step}.priv_obs", privileged_obs)
                _scan_tensor(f"iter{iteration}.step{step}.rewards", rewards)
                costs = runner._extract_costs(infos, rewards)
                _scan_tensor(f"iter{iteration}.step{step}.costs", costs)

                runner.alg.process_env_step(rewards, dones, infos, costs)

        # compute_returns should run under no_grad/inference to match training and
        # to avoid autograd on inference-mode tensors.
        with torch.inference_mode():
            runner.alg.compute_returns(privileged_obs)
        if _scan_storage(runner.alg.storage):
            print("[ERROR] NaN/Inf detected in rollout storage.")
            break

        loss_dict = runner.alg.update()
        for key, value in loss_dict.items():
            if not torch.isfinite(torch.tensor(value)):
                print(f"[ERROR] Loss {key} is non-finite: {value}")
        _scan_params("policy.actor", runner.alg.policy.actor.parameters())
        _scan_params("policy.critic", runner.alg.policy.critic.parameters())
        if hasattr(runner.alg.policy, "cost_critic"):
            _scan_params("policy.cost_critic", runner.alg.policy.cost_critic.parameters())

        print(
            "[INFO] Losses: "
            + ", ".join([f"{k}={v:.4g}" for k, v in loss_dict.items() if isinstance(v, (int, float))])
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
