# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose NaN/Inf in FPPO env outputs (obs/reward/cost/state)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import gymnasium as gym

from isaaclab.app import AppLauncher


def _scan_tensor(name: str, tensor: torch.Tensor) -> bool:
    if not torch.is_tensor(tensor):
        return False
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    if nan_count or inf_count:
        min_val = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).min().item()
        max_val = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).max().item()
        print(f"[ERROR] {name}: nan={nan_count} inf={inf_count} min={min_val:.4g} max={max_val:.4g}")
        return True
    return False


def _scan_obs(prefix: str, obs) -> bool:
    has_bad = False
    if isinstance(obs, dict):
        for key, value in obs.items():
            has_bad |= _scan_obs(f"{prefix}.{key}", value)
    else:
        has_bad |= _scan_tensor(prefix, obs)
    return has_bad


def _scan_robot_state(env) -> bool:
    has_bad = False
    scene = getattr(env, "scene", None)
    if scene is None:
        return False

    robot = None
    try:
        robot = scene["robot"]
    except Exception:
        pass
    if robot is None and hasattr(scene, "articulations"):
        if "robot" in scene.articulations:
            robot = scene.articulations["robot"]
        elif len(scene.articulations) == 1:
            robot = next(iter(scene.articulations.values()))

    if robot is None:
        return False

    data = getattr(robot, "data", None)
    if data is None:
        return False
    for name in ("root_pos_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel", "applied_torque"):
        value = getattr(data, name, None)
        if value is not None:
            has_bad |= _scan_tensor(f"robot.{name}", value)
    return has_bad


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose NaN/Inf in FPPO env outputs.")
    parser.add_argument("--task", type=str, required=True, help="Gym task name.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps to run.")
    parser.add_argument(
        "--action_mode",
        type=str,
        choices={"random", "zero"},
        default="random",
        help="Action source: random or zero.",
    )
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
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    import rl_sim_env.tasks  # noqa: F401

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = env.unwrapped

    obs, extras = env.reset(seed=args_cli.seed)
    print("[INFO] Reset complete.")
    if _scan_obs("reset_obs", obs) or _scan_robot_state(env):
        print("[ERROR] NaN/Inf detected at reset.")
        env.close()
        simulation_app.close()
        return

    action_dim = env.action_manager.total_action_dim if hasattr(env, "action_manager") else env.action_space.shape[0]
    device = env.device
    num_envs = env.num_envs

    for step in range(1, args_cli.steps + 1):
        if args_cli.action_mode == "zero":
            actions = torch.zeros(num_envs, action_dim, device=device)
        else:
            actions = torch.rand(num_envs, action_dim, device=device) * 2.0 - 1.0

        obs, rewards, terminated, truncated, extras = env.step(actions)
        has_bad = False
        has_bad |= _scan_obs(f"step{step}.obs", obs)
        has_bad |= _scan_tensor(f"step{step}.rewards", rewards)
        cost = extras.get("cost") if isinstance(extras, dict) else None
        if cost is not None:
            has_bad |= _scan_tensor(f"step{step}.cost", cost)
        has_bad |= _scan_robot_state(env)

        if step % 20 == 0:
            reward_mean = rewards.mean().item()
            cost_mean = cost.mean().item() if torch.is_tensor(cost) else 0.0
            max_act = actions.abs().max().item()
            print(f"[INFO] step={step} reward_mean={reward_mean:.4g} cost_mean={cost_mean:.4g} max|a|={max_act:.3f}")

        if has_bad:
            print(f"[ERROR] NaN/Inf detected at step {step}.")
            break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
