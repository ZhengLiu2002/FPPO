# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose contact/cost stats for FPPO tasks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher


def _format_top_contacts(body_names: list[str], rates: torch.Tensor, top_k: int = 5) -> str:
    if rates.numel() == 0:
        return "none"
    k = min(top_k, rates.numel())
    vals, idx = torch.topk(rates, k)
    items = [f"{body_names[i]}:{vals[j]:.3f}" for j, i in enumerate(idx.tolist())]
    return ", ".join(items)


def _format_top_costs(cost_manager) -> str:
    if cost_manager is None:
        return "none"
    step_reward = getattr(cost_manager, "_step_reward", None)
    term_names = getattr(cost_manager, "active_terms", [])
    if step_reward is None or not term_names:
        return "none"
    means = step_reward.mean(dim=0)
    order = torch.argsort(torch.abs(means), descending=True)
    items = []
    for idx in order[: min(6, means.numel())]:
        items.append(f"{term_names[idx]}:{means[idx].item():.4g}")
    return ", ".join(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose contact and cost statistics for FPPO.")
    parser.add_argument("--task", type=str, required=True, help="Gym task name.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps to run.")
    parser.add_argument(
        "--action_mode",
        type=str,
        choices={"random", "zero", "policy"},
        default="zero",
        help="Action source: random, zero, or policy (requires --policy_checkpoint).",
    )
    parser.add_argument("--policy_checkpoint", type=str, default=None, help="Path to checkpoint when action_mode=policy.")
    parser.add_argument("--print_every", type=int, default=20, help="Steps between prints.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the environment.")
    parser.add_argument("--contact_threshold", type=float, default=1.0, help="Contact force threshold.")
    # add rsl-rl args so we can reuse parse_rsl_rl_cfg for policy mode
    here = Path(__file__).resolve()
    scripts_root = here.parents[1]
    rsl_rl_dir = scripts_root / "rsl_rl"
    if rsl_rl_dir.exists():
        sys.path.insert(0, str(rsl_rl_dir))
    import cli_args  # isort: skip

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    # clear out sys.argv for AppLauncher/Hydra
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # ensure local source tree is on sys.path
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    workspace_root = project_root.parent
    source_root = project_root / "source"
    for path in (source_root, source_root / "rl_sim_env"):
        if path.exists():
            sys.path.insert(0, str(path))
    os.environ.setdefault("RL_SIM_ENV_ROOT_DIR", str(workspace_root))

    # import IsaacLab after SimulationApp is instantiated
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    import rl_sim_env.tasks  # noqa: F401

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    policy = None
    if args_cli.action_mode == "policy":
        if not args_cli.policy_checkpoint:
            raise ValueError("--policy_checkpoint is required when action_mode=policy.")
        try:
            from rl_algorithms.rsl_rl_wrapper import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
            from rl_algorithms.rsl_rl.runners import OnPolicyRunner
        except ImportError:
            from rl_sim_env.rl_algorithms.rsl_rl_wrapper import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
            from rl_sim_env.rl_algorithms.rsl_rl.runners import OnPolicyRunner

        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.policy_checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped

    obs, _ = env.reset()
    print("[INFO] Reset complete.")

    contact_sensor = raw_env.scene.sensors["contact_forces"]
    body_names = contact_sensor.body_names
    foot_names = []
    if getattr(raw_env.cfg, "costs", None) is not None and raw_env.cfg.costs.prob_body_contact is not None:
        foot_names = raw_env.cfg.costs.prob_body_contact.params.get("foot_body_names", [])

    foot_ids, resolved_foot = contact_sensor.find_bodies(foot_names, preserve_order=True)
    print(f"[INFO] Contact sensor bodies: {len(body_names)}")
    print(f"[INFO] Foot names cfg: {foot_names}")
    print(f"[INFO] Foot names resolved: {resolved_foot}")
    if not foot_ids:
        print("[WARN] No foot bodies matched in contact sensor. Body-contact cost may be incorrect.")

    mask = torch.ones(len(body_names), dtype=torch.bool, device=raw_env.device)
    if foot_ids:
        mask[torch.tensor(foot_ids, device=raw_env.device)] = False
    non_foot_ids = torch.where(mask)[0]

    action_dim = (
        raw_env.action_manager.total_action_dim if hasattr(raw_env, "action_manager") else raw_env.action_space.shape[0]
    )
    device = raw_env.device
    num_envs = raw_env.num_envs

    for step in range(1, args_cli.steps + 1):
        if args_cli.action_mode == "zero":
            actions = torch.zeros(num_envs, action_dim, device=device)
        elif args_cli.action_mode == "random":
            actions = torch.rand(num_envs, action_dim, device=device) * 2.0 - 1.0
        else:
            with torch.inference_mode():
                actions = policy(obs)

        step_out = env.step(actions)
        if len(step_out) == 5:
            obs, rewards, terminated, truncated, extras = step_out
        else:
            obs, rewards, dones, extras = step_out

        if step % args_cli.print_every != 0:
            continue

        force_mag = torch.norm(contact_sensor.data.net_forces_w_history, dim=-1).max(dim=1)[0]
        threshold = args_cli.contact_threshold

        if foot_ids:
            foot_contact = force_mag[:, foot_ids] > threshold
            foot_rate = foot_contact.any(dim=1).float().mean().item()
        else:
            foot_rate = 0.0

        if non_foot_ids.numel() > 0:
            non_foot_contact = force_mag[:, non_foot_ids] > threshold
            non_foot_rate = non_foot_contact.any(dim=1).float().mean().item()
        else:
            non_foot_rate = 0.0

        total_cost = extras.get("cost") if isinstance(extras, dict) else None
        cost_mean = total_cost.mean().item() if torch.is_tensor(total_cost) else 0.0
        top_contacts = _format_top_contacts(body_names, (force_mag > threshold).float().mean(dim=0))
        top_costs = _format_top_costs(getattr(raw_env, "cost_manager", None))

        print(
            f"[INFO] step={step} cost_mean={cost_mean:.4g} "
            f"foot_contact_rate={foot_rate:.3f} non_foot_contact_rate={non_foot_rate:.3f}"
        )
        print(f"[INFO] top_body_contacts: {top_contacts}")
        print(f"[INFO] top_cost_terms: {top_costs}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
