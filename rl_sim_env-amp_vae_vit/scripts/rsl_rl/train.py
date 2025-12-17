# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path
import os


from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.io import dump_pickle, dump_yaml
import pickle
import yaml
import os
# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# from isaaclab_tasks.utils import get_checkpoint_path
# =================================================================================
# [Fix] 增强后的导入模块 - 自动处理路径差异
# =================================================================================
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
_WORKSPACE_ROOT = _PROJECT_ROOT.parent
_SOURCE_ROOT = _PROJECT_ROOT / "source"
# Ensure local source tree is on sys.path so distributed runs can import modules
for _p in (_SOURCE_ROOT, _SOURCE_ROOT / "rl_sim_env"):
    _p_str = str(_p)
    if _p.exists() and _p_str not in sys.path:
        sys.path.insert(0, _p_str)
# Ensure RL_SIM_ENV resolves assets relative to workspace root unless user overrides
os.environ.setdefault("RL_SIM_ENV_ROOT_DIR", str(_WORKSPACE_ROOT))

import rl_sim_env.tasks  # noqa: F401

# 1. 尝试导入配置类 (RslRlOnPolicyRunnerCfg) 和环境包装器 (AmpVaeVecEnvWrapper)
try:
    # 优先尝试直接从 rl_algorithms 导入
    from rl_algorithms.rsl_rl_wrapper import RslRlOnPolicyRunnerCfg, AmpVaeVecEnvWrapper
except ImportError:
    try:
        # 备选：从 rl_sim_env 包内部导入
        from rl_sim_env.rl_algorithms.rsl_rl_wrapper import RslRlOnPolicyRunnerCfg, AmpVaeVecEnvWrapper
    except ImportError as e:
        # 如果都失败，打印详细路径并抛出致命错误
        print(f"[ERROR] Python Path: {sys.path}")
        raise ImportError(f"CRITICAL: 无法导入 RslRlOnPolicyRunnerCfg 或 AmpVaeVecEnvWrapper。请检查 rl_algorithms 路径。\n详细错误: {e}")

# 2. 尝试导入自定义运行器 (AMPVAEOnPolicyRunner) - 注意大写
try:
    # 修改：将 AMPVAEOnPolicyRunner 导入并重命名为 AmpVaeOnPolicyRunner，这样下面的代码不用改
    from rl_algorithms.rsl_rl.runners.amp_vae_on_policy_runner import AMPVAEOnPolicyRunner as AmpVaeOnPolicyRunner
except ImportError:
    try:
        # 备选路径同理
        from rl_sim_env.rl_algorithms.rsl_rl.runners.amp_vae_on_policy_runner import AMPVAEOnPolicyRunner as AmpVaeOnPolicyRunner
    except ImportError as e:
        raise ImportError(f"CRITICAL: 无法导入 AMPVAEOnPolicyRunner (请检查类名大小写)。\n详细错误: {e}")

print(f"[INFO] Successfully imported: Runner={AmpVaeOnPolicyRunner.__name__}, Wrapper={AmpVaeVecEnvWrapper.__name__}")
# =================================================================================

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # number of envs: when distributed, interpret CLI num_envs as total and split per rank
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args_cli.num_envs is not None:
        requested_envs = args_cli.num_envs
        if args_cli.distributed and world_size > 1:
            per_rank_envs = max(1, (requested_envs + world_size - 1) // world_size)
            print(
                f"[INFO] Distributed run detected (world_size={world_size}); "
                f"using {per_rank_envs} envs per rank to approximate total {requested_envs}."
            )
            env_cfg.scene.num_envs = per_rank_envs
        else:
            env_cfg.scene.num_envs = requested_envs
        # rebuild command ids if env cfg provides helper (ensures coverage matches num_envs)
        rebuild_fn = getattr(env_cfg, "_rebuild_command_cfg", None)
        if callable(rebuild_fn):
            rebuild_fn(env_cfg.scene.num_envs)
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # === 新增：解包环境，移除 Gymnasium 的默认 Wrapper (如 OrderEnforcing) ===
    # 这是为了通过 RslRlVecEnvWrapper 的严格类型检查
    env = env.unwrapped

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    # 修改：安全访问 class_name
    algo_class_name = getattr(agent_cfg.algorithm, "class_name", None)
    if agent_cfg.resume or algo_class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    # env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env = AmpVaeVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    # runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # === 修复：适配新版 rsl_rl 并注入自定义配置 ===
    agent_cfg_dict = agent_cfg.to_dict()
    
    # 1. 注入 obs_groups (映射逻辑分组)
    if "obs_groups" not in agent_cfg_dict:
        agent_cfg_dict["obs_groups"] = {
            "policy": ["actor_obs"],    
            "critic": ["critic_obs"],   
            "amp": ["amp_obs"],         
            "vae": ["vae_obs"],         
        }

    # 2. 注入 class_name (解决 KeyError)
    # 即使使用自定义 Runner，底层父类可能仍会检查这些字段
    if "policy" in agent_cfg_dict and "class_name" not in agent_cfg_dict["policy"]:
        agent_cfg_dict["policy"]["class_name"] = "ActorCritic"
    
    if "algorithm" in agent_cfg_dict and "class_name" not in agent_cfg_dict["algorithm"]:
        # 注意：这里名字必须对应自定义 Runner 内部能够识别或导入的类
        # 通常自定义 Runner 会覆盖构建逻辑，但为了防止父类检查，我们填一个占位符或真实名
        agent_cfg_dict["algorithm"]["class_name"] = "AmpVaePPO" 

    # 3. 使用自定义的 AmpVaeOnPolicyRunner
    # 注意：这里不再使用标准的 OnPolicyRunner
    runner = AmpVaeOnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    # 修改：安全访问 class_name
    algo_class_name = getattr(agent_cfg.algorithm, "class_name", None)
    if agent_cfg.resume or algo_class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    # dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    # dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    # dump the configuration into log-directory
    
    # 1. 确保 params 目录存在
    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    # 2. 使用标准库保存 YAML (需要转换成字典，通常 config 对象有 to_dict() 方法，或者是 dataclass)
    # 如果 env_cfg 没有 to_dict()，可以尝试跳过保存 yaml 或仅保存 pkl
    try:
        env_dict = env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else env_cfg.__dict__
        with open(os.path.join(params_dir, "env.yaml"), "w") as f:
            yaml.dump(env_dict, f, default_flow_style=False)
    except Exception as e:
        print(f"[WARN] Could not save env.yaml: {e}")

    try:
        agent_dict = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else agent_cfg.__dict__
        with open(os.path.join(params_dir, "agent.yaml"), "w") as f:
            yaml.dump(agent_dict, f, default_flow_style=False)
    except Exception as e:
        print(f"[WARN] Could not save agent.yaml: {e}")

    # 3. 使用标准库保存 Pickle (最重要，用于复现)
    with open(os.path.join(params_dir, "env.pkl"), "wb") as f:
        pickle.dump(env_cfg, f)
    with open(os.path.join(params_dir, "agent.pkl"), "wb") as f:
        pickle.dump(agent_cfg, f)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
