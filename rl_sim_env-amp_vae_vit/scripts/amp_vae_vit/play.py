# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from ruamel.yaml import YAML

yaml = YAML()
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--onnx", action="store_true", default=False, help="Run in onnx mode.")
# append AMP-VAE-VIT cli arguments
cli_args.add_amp_vae_vit_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import numpy as np
import rl_sim_env.tasks
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rl_algorithms.rsl_rl.runners import AMPVAEVITOnPolicyRunner
from rl_algorithms.rsl_rl_wrapper import (
    AmpVaeVITOnPolicyRunnerCfg,
    AmpVaeVITVecEnvWrapper,
    export_amp_vae_vit_policy_as_onnx,
    load_onnx_model,
    onnx_run_inference,
    verify_onnx_model,
)

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: AmpVaeVITOnPolicyRunnerCfg = cli_args.parse_amp_vae_vit_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "amp_vae_vit", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("amp_vae_vit", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    # elif args_cli.checkpoint:
    #     resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    policy_yaml_path = os.path.join(log_dir, "policy.yaml")
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = AmpVaeVitVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = AMPVAEVITOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy, vae = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    dir_path, filename = os.path.split(resume_path)
    base_dir, exp_dir = os.path.split(dir_path)
    base_dir = base_dir + os.sep
    name_no_ext, _ = os.path.splitext(filename)
    print("base_dir:", base_dir)
    print("exp_dir:", exp_dir)
    print("name_no_ext:", name_no_ext)
    export_model_dir = os.path.join(os.path.dirname(base_dir), "exported", exp_dir, name_no_ext)
    export_amp_vae_vit_policy_as_onnx(
        policy,
        vae,
        path=export_model_dir,
        filename="policy.onnx",
    )
    verify_onnx_model(os.path.join(export_model_dir, "policy.onnx"), "policy")
    with open(policy_yaml_path, encoding="utf-8") as f:
        policy_yaml = yaml.load(f)
    policy_yaml["load_run"] = agent_cfg.load_run
    policy_yaml["checkpoint"] = agent_cfg.load_checkpoint
    policy_yaml_export_path = os.path.join(export_model_dir, "policy.yaml")
    with open(policy_yaml_export_path, "w", encoding="utf-8") as f:
        yaml.dump(policy_yaml, f)

    if args_cli.onnx:
        print("[INFO] Running in onnx mode.")
        onnx_session = load_onnx_model(os.path.join(export_model_dir, "policy.onnx"))
        print("[INFO] ONNX model loaded successfully.")

    dt = env.unwrapped.physics_dt

    # reset environment
    obs_dict = env.get_observations()
    actor_obs = obs_dict["actor_obs"]
    vae_obs = obs_dict["vae_obs"]
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.onnx:
                actor_obs_onnx = actor_obs.cpu().numpy().astype(np.float32)
                vae_obs_onnx = vae_obs.cpu().numpy().astype(np.float32)
                actions = torch.from_numpy(onnx_run_inference(onnx_session, actor_obs_onnx, vae_obs_onnx)[0]).to(
                    env.device
                )
            else:
                code = vae.act_inference(vae_obs)
                full_obs = torch.cat((code, actor_obs), dim=-1)
                actions = policy.act_inference(full_obs)

            # env stepping
            (
                obs_buf,
                rewards,
                dones,
                infos,
                reset_env_ids,
                terminal_amp_states,
                episode_reward,
            ) = env.step(actions)
            actor_obs = obs_buf["actor_obs"]
            vae_obs = obs_buf["vae_obs"]
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
