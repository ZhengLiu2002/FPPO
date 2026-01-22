# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose command/velocity/action stats for FPPO tasks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher


def _safe_mean(tensor: torch.Tensor) -> float:
    return tensor.mean().item() if torch.is_tensor(tensor) else 0.0


def _safe_max(tensor: torch.Tensor) -> float:
    return tensor.max().item() if torch.is_tensor(tensor) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose command/motion stats for FPPO.")
    parser.add_argument("--task", type=str, required=True, help="Gym task name.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps to run.")
    parser.add_argument("--print_every", type=int, default=20, help="Steps between prints.")
    parser.add_argument(
        "--action_mode",
        type=str,
        choices={"random", "zero", "policy"},
        default="policy",
        help="Action source: random, zero, or policy (requires --checkpoint).",
    )
    parser.add_argument("--policy_checkpoint", type=str, default=None, help="Path to checkpoint when action_mode=policy.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the environment.")

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

        cmd = None
        if hasattr(raw_env, "command_manager"):
            cmd = raw_env.command_manager.get_command("base_velocity")

        robot = raw_env.scene["robot"]
        lin_vel_b = robot.data.root_lin_vel_b
        ang_vel_w = robot.data.root_ang_vel_w

        cmd_lin = torch.linalg.norm(cmd[:, :2], dim=1) if cmd is not None else torch.zeros(num_envs, device=device)
        cmd_yaw = torch.abs(cmd[:, 2]) if cmd is not None else torch.zeros(num_envs, device=device)
        vel_lin = torch.linalg.norm(lin_vel_b[:, :2], dim=1)
        vel_yaw = torch.abs(ang_vel_w[:, 2])

        err_lin = torch.abs(cmd_lin - vel_lin)
        err_yaw = torch.abs(cmd_yaw - vel_yaw)

        act_abs = actions.abs()
        act_sat = (act_abs > 0.95).float().mean(dim=1)

        cost = extras.get("cost") if isinstance(extras, dict) else None
        cost_mean = cost.mean().item() if torch.is_tensor(cost) else 0.0

        print(
            f"[INFO] step={step} cmd_lin={_safe_mean(cmd_lin):.3f} cmd_yaw={_safe_mean(cmd_yaw):.3f} "
            f"vel_lin={_safe_mean(vel_lin):.3f} vel_yaw={_safe_mean(vel_yaw):.3f} "
            f"err_lin={_safe_mean(err_lin):.3f} err_yaw={_safe_mean(err_yaw):.3f}"
        )
        print(
            f"[INFO] action_mean_abs={_safe_mean(act_abs):.3f} action_max_abs={_safe_max(act_abs):.3f} "
            f"action_sat_rate={_safe_mean(act_sat):.3f} cost_mean={cost_mean:.3f}"
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
