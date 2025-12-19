# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained AMP-VAE policy with optional visualization/video.

This script is optimized for *inference*. It loads only the policy + VAE weights from the checkpoint and
does NOT preload AMP motion transitions (which are only needed for training the discriminator).
"""

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


def _ensure_local_paths_on_sys_path() -> None:
    """Ensure local source tree is on sys.path (helps when running from different workdirs)."""
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    workspace_root = project_root.parent
    source_root = project_root / "source"
    for p in (source_root, source_root / "rl_sim_env"):
        p_str = str(p)
        if p.exists() and p_str not in sys.path:
            sys.path.insert(0, p_str)
    os.environ.setdefault("RL_SIM_ENV_ROOT_DIR", str(workspace_root))


def main() -> None:
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Play an AMP-VAE policy checkpoint.")
    parser.add_argument("--video", action="store_true", default=False, help="Record a video during play.")
    parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (per process).")
    parser.add_argument("--task", type=str, required=True, help="Name of the task.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
    # append RSL-RL cli arguments (load_run/checkpoint/etc.)
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args (headless/livestream/device/etc.)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # always enable cameras when recording video
    if args_cli.video:
        args_cli.enable_cameras = True

    _ensure_local_paths_on_sys_path()

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

    import rl_sim_env.tasks  # noqa: F401

    # Prefer importing from local "rl_algorithms" if present (mirrors train.py robustness)
    try:
        from rl_algorithms.rsl_rl_wrapper import AmpVaeVecEnvWrapper
        from rl_algorithms.rsl_rl.modules import ActorCritic, VAE
    except ImportError:
        from rl_sim_env.rl_algorithms.rsl_rl_wrapper import AmpVaeVecEnvWrapper
        from rl_sim_env.rl_algorithms.rsl_rl.modules import ActorCritic, VAE

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # locate checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    print(f"[INFO] Loading checkpoint: {resume_path}")

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = env.unwrapped
    if isinstance(env, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording (records only one clip)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for AMP-VAE runner
    env = AmpVaeVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---------------------------------------------------------------------
    # Inference-only model construction (skip AMP motion preload)
    # ---------------------------------------------------------------------
    cfg_env = env.unwrapped.cfg.config_summary.env
    cfg_policy = agent_cfg.to_dict()["policy"]

    num_actor_obs = int(cfg_env.num_actor_obs)
    num_critic_obs = int(cfg_env.num_critic_obs)
    num_vae_obs = int(cfg_env.num_vae_obs)
    obs_history_length = int(cfg_env.obs_history_length)
    cenet_in_dim = num_vae_obs * obs_history_length
    cenet_out_dim = int(cfg_env.num_vae_out)
    num_actions = int(env.num_actions)

    # min_std is only used for stochastic sampling; we still compute it to keep the module consistent
    dof_range = (
        env.unwrapped.scene["robot"].data.default_joint_pos_limits[0][:, 1]
        - env.unwrapped.scene["robot"].data.default_joint_pos_limits[0][:, 0]
    )
    dof_range = torch.as_tensor(dof_range, device=agent_cfg.device, dtype=torch.float32)[:num_actions]
    base_min_std = torch.tensor(cfg_policy["min_normalized_std"], device=agent_cfg.device, dtype=torch.float32)
    if base_min_std.numel() < num_actions:
        pad = base_min_std.new_full((num_actions - base_min_std.numel(),), base_min_std[-1])
        base_min_std = torch.cat([base_min_std, pad])
    elif base_min_std.numel() > num_actions:
        base_min_std = base_min_std[:num_actions]
    min_std = base_min_std * torch.abs(dof_range)

    actor_critic = ActorCritic(
        num_actor_obs=num_actor_obs + cenet_out_dim,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        min_std=min_std,
        **cfg_policy,
    ).to(agent_cfg.device)
    vae = VAE(cenet_in_dim, cenet_out_dim).to(agent_cfg.device)

    # PyTorch >= 2.6 defaults `weights_only=True`, which can fail for our checkpoints because they may
    # contain non-tensor objects (e.g., Normalizer). For local checkpoints you trust, fall back to
    # `weights_only=False` to load the full dict.
    try:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    except TypeError:
        # older torch without `weights_only` kwarg
        ckpt = torch.load(resume_path, map_location="cpu")
    actor_critic.load_state_dict(ckpt["model_state_dict"], strict=True)
    vae.load_state_dict(ckpt["vae_state_dict"], strict=True)
    actor_critic.eval()
    vae.eval()

    dt = env.unwrapped.step_dt

    # initial observations
    obs_dict = env.get_observations()
    actor_obs = obs_dict["actor_obs"].to(agent_cfg.device)
    critic_obs = obs_dict["critic_obs"].to(agent_cfg.device)
    amp_obs = obs_dict["amp_obs"].to(agent_cfg.device)
    next_amp_obs = amp_obs.clone()
    vae_obs = obs_dict["vae_obs"].to(agent_cfg.device)

    timestep = 0
    # Note: p_boot_mean is a training-time scalar. For inference/visualization, keep it at 0.0 unless you
    # decide to load a saved value explicitly.
    p_boot_mean = 0.0
    # AMP reward term expects env.amp_out to exist. For pure visualization we can feed a constant
    # tensor (amp_out==1 -> maximal amp_reward) to avoid needing the discriminator + motion preload.
    dummy_amp_out = torch.ones((env.num_envs, 1), device=agent_cfg.device, dtype=torch.float32)

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # deterministic VAE code
            (
                _code,
                code_vel,
                code_mass,
                code_latent,
                *_,
            ) = vae.cenet_forward(vae_obs, deterministic=True)
            obs_full = torch.cat(
                (
                    p_boot_mean * code_vel + (1.0 - p_boot_mean) * critic_obs[:, 0:3],
                    p_boot_mean * code_mass + (1.0 - p_boot_mean) * critic_obs[:, -1:],
                    code_latent,
                    actor_obs,
                ),
                dim=-1,
            )
            actions = actor_critic.act_inference(obs_full)
            amp_out = dummy_amp_out
            amp_obs = next_amp_obs.clone()
            obs_buf, _, _, _, reset_env_ids, terminal_amp_states, _ = env.step(actions.to(agent_cfg.device), amp_out)
            actor_obs = obs_buf["actor_obs"].to(agent_cfg.device)
            critic_obs = obs_buf["critic_obs"].to(agent_cfg.device)
            next_amp_obs = obs_buf["amp_obs"].to(agent_cfg.device)
            vae_obs = obs_buf["vae_obs"].to(agent_cfg.device)
            # keep terminal AMP states consistent with training loop
            if reset_env_ids.numel() > 0:
                next_amp_obs_with_term = next_amp_obs.clone()
                next_amp_obs_with_term[reset_env_ids] = terminal_amp_states.to(agent_cfg.device)
                next_amp_obs = next_amp_obs_with_term

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
