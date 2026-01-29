# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect
import math
import os
import statistics
import time
from collections import deque

import rsl_rl
import torch
# from rsl_rl.algorithms import PPO, Distillation
# from rsl_rl.env import VecEnv
# from rsl_rl.modules import (
#     ActorCritic,
#     ActorCriticRecurrent,
#     EmpiricalNormalization,
#     StudentTeacher,
#     StudentTeacherRecurrent,
# )
# from rsl_rl.utils import store_code_state
from .. import algorithms as algs
from ..env import VecEnv
from ..modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from ..utils import ConstraintNormalizer, store_code_state
from ..storage.data_buffer import DataBuffer

class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm
        alg_id = self.alg_cfg.get("name", self.alg_cfg.get("class_name"))
        if alg_id is None:
            raise ValueError("Algorithm name not found. Please set 'algorithm.name' or 'algorithm.class_name'.")
        if alg_id.lower() == "distillation":
            self.training_type = "distillation"
        else:
            self.training_type = "rl"

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        self._actor_obs_dim = num_obs
        self._vae_enabled = False
        self._vae_num_obs = None
        self._vae_history_length = None
        self._vae_out_dim = None
        self._vae_obs_buffer = None
        self._critic_obs_slices = {}
        self._amp_obs_slices = {}
        self._amp_obs_dim = None

        if hasattr(self.env.unwrapped, "observation_manager"):
            group_obs_dim = self.env.unwrapped.observation_manager.group_obs_dim
            critic_group = "critic" if "critic" in group_obs_dim else ("critic_obs" if "critic_obs" in group_obs_dim else None)
            if critic_group is not None:
                self._critic_obs_slices = self._build_obs_term_slices(critic_group)
            amp_group = None
            if "amp_obs" in group_obs_dim:
                amp_group = "amp_obs"
            elif "amp" in group_obs_dim:
                amp_group = "amp"
            if amp_group is not None:
                self._amp_obs_slices = self._build_obs_term_slices(amp_group)
                self._amp_obs_dim = group_obs_dim[amp_group][0]

        # resolve type of privileged observations
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learnig, e.g., PPO
            elif "critic_obs" in extras["observations"]:
                self.privileged_obs_type = "critic_obs"
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # evaluate the policy class
        policy_cfg = dict(self.policy_cfg)
        vae_cfg = policy_cfg.get("vae", None)
        if vae_cfg is not None:
            if hasattr(vae_cfg, "to_dict"):
                vae_cfg = vae_cfg.to_dict()
            elif not isinstance(vae_cfg, dict):
                vae_cfg = vars(vae_cfg)
            if vae_cfg.get("enabled", False):
                self._vae_enabled = True
                self._vae_num_obs = int(vae_cfg.get("num_vae_obs", self._actor_obs_dim))
                if self._vae_num_obs != self._actor_obs_dim:
                    self._vae_num_obs = self._actor_obs_dim
                self._vae_history_length = int(vae_cfg.get("obs_history_length", 1))
                self._vae_out_dim = vae_cfg.get("cenet_out_dim", vae_cfg.get("num_vae_out"))
                if self._vae_out_dim is None:
                    raise ValueError("VAE enabled but 'cenet_out_dim'/'num_vae_out' is not set.")
                self._vae_out_dim = int(self._vae_out_dim)
                if vae_cfg.get("cenet_in_dim") is None:
                    vae_cfg["cenet_in_dim"] = self._vae_num_obs * self._vae_history_length
                if vae_cfg.get("cenet_out_dim") is None:
                    vae_cfg["cenet_out_dim"] = self._vae_out_dim
                if vae_cfg.get("cenet_recon_dim") is None:
                    vae_cfg["cenet_recon_dim"] = self._actor_obs_dim
                policy_cfg["vae"] = vae_cfg
                num_obs = num_obs + self._vae_out_dim
        policy_class_name = policy_cfg.pop("class_name")
        policy_class = eval(policy_class_name)
        policy: ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **policy_cfg
        ).to(self.device)
        if self._vae_enabled:
            self._vae_obs_buffer = DataBuffer(
                num_envs=self.env.num_envs,
                num_data=self._vae_num_obs,
                data_history_length=self._vae_history_length,
                device=self.device,
            )

        # initialize algorithm
        alg_cfg = dict(self.alg_cfg)
        alg_name = alg_cfg.pop("name", None)
        alg_class_name = alg_cfg.pop("class_name", None)
        if alg_name is not None:
            alg_class = algs.ALGORITHM_REGISTRY.get(alg_name.lower())
            if alg_class is None:
                raise ValueError(f"Algorithm '{alg_name}' not found in registry.")
        elif alg_class_name is not None:
            alg_class = getattr(algs, alg_class_name, None)
            if alg_class is None:
                raise ValueError(f"Algorithm class '{alg_class_name}' not found.")
        else:
            raise ValueError("Algorithm name not found. Please set 'algorithm.name' or 'algorithm.class_name'.")
        alg_cfg = self._filter_kwargs(alg_class, alg_cfg)
        self.alg = alg_class(policy, device=self.device, **alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        if hasattr(self.alg, "set_vae_obs_slices"):
            self.alg.set_vae_obs_slices(self._critic_obs_slices, self._amp_obs_slices)

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.early_save_interval = self.cfg.get("early_save_interval")
        self.early_save_iterations = self.cfg.get("early_save_iterations")
        self.empirical_normalization = self.cfg["empirical_normalization"]
        self._constraint_normalizer = ConstraintNormalizer.from_cfg(self.alg_cfg, device=self.device)
        if not self._constraint_normalizer.enabled:
            self._constraint_normalizer = None
        self._constraint_scale = self.alg_cfg.get("constraint_cost_scale")
        self._constraint_scale_by_gamma = self.alg_cfg.get("constraint_scale_by_gamma", False)
        self._constraint_gamma = self.alg_cfg.get("cost_gamma")
        if self._constraint_gamma is None:
            self._constraint_gamma = self.alg_cfg.get("gamma")
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
            vae_actor_obs_shape=[self._actor_obs_dim] if self._vae_enabled else None,
            vae_obs_history_shape=[self._vae_num_obs * self._vae_history_length] if self._vae_enabled else None,
            next_actor_obs_shape=[self._actor_obs_dim] if self._vae_enabled else None,
            amp_obs_shape=[self._amp_obs_dim] if self._amp_obs_dim is not None else None,
        )

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                # from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                from ..utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                # from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                from ..utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # check if teacher is loaded
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        actor_obs = obs.to(self.device)
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        privileged_obs = privileged_obs.to(self.device)
        amp_obs_current = None
        if isinstance(extras, dict):
            obs_payload = extras.get("observations", {})
            if isinstance(obs_payload, dict):
                amp_obs_current = obs_payload.get("amp_obs")
                if amp_obs_current is None:
                    amp_obs_current = obs_payload.get("amp")
                if amp_obs_current is not None:
                    amp_obs_current = amp_obs_current.to(self.device)
        if self._vae_enabled and self._vae_obs_buffer is not None:
            # initialize VAE history with first observation
            all_ids = torch.arange(self.env.num_envs, device=self.device)
            self._vae_obs_buffer.reset(all_ids, new_data=actor_obs[:, : self._vae_num_obs])
            obs_history = self._vae_obs_buffer.get_all_data()
            policy_obs = self.alg.policy.get_policy_obs(actor_obs, obs_history, deterministic=True)
        else:
            obs_history = None
            policy_obs = actor_obs
        # perform normalization
        policy_obs = self.obs_normalizer(policy_obs)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(policy_obs, privileged_obs, actor_obs=actor_obs, vae_obs_history=obs_history)
                    # Optional symmetry cost from policy (strict mirror constraint)
                    sym_cost = None
                    base_env = getattr(self.env, "unwrapped", None)
                    if base_env is None and hasattr(self.env, "env"):
                        base_env = self.env.env
                    if base_env is not None and hasattr(base_env, "compute_symmetry_cost"):
                        sym_cost = base_env.compute_symmetry_cost(
                            self.alg.policy, actor_obs, obs_history=obs_history
                        )
                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # Move to device
                    actor_obs = obs.to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    # perform normalization
                    if self._vae_enabled and self._vae_obs_buffer is not None:
                        # reset history on episode boundaries
                        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                        if done_ids.numel() > 0:
                            self._vae_obs_buffer.reset(done_ids, new_data=actor_obs[done_ids, : self._vae_num_obs])
                        self._vae_obs_buffer.insert(actor_obs[:, : self._vae_num_obs])
                        obs_history = self._vae_obs_buffer.get_all_data()
                        policy_obs = self.alg.policy.get_policy_obs(actor_obs, obs_history, deterministic=True)
                    else:
                        obs_history = None
                        policy_obs = actor_obs
                    policy_obs = self.obs_normalizer(policy_obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = policy_obs

                    # process the step
                    if sym_cost is not None and isinstance(infos, dict):
                        log = infos.setdefault("log", {})
                        log["Episode_Cost/symmetric"] = sym_cost.to(self.device)
                        cost_payload = infos.setdefault("cost", {})
                        if isinstance(cost_payload, dict):
                            cost_payload["symmetric"] = sym_cost
                    costs = self._extract_costs(infos, rewards)
                    amp_obs_next = None
                    if isinstance(infos, dict):
                        obs_payload = infos.get("observations", {})
                        if isinstance(obs_payload, dict):
                            amp_obs_next = obs_payload.get("amp_obs")
                            if amp_obs_next is None:
                                amp_obs_next = obs_payload.get("amp")
                            if amp_obs_next is not None:
                                amp_obs_next = amp_obs_next.to(self.device)
                    self.alg.process_env_step(
                        rewards,
                        dones,
                        infos,
                        costs,
                        next_actor_obs=actor_obs,
                        next_amp_obs=amp_obs_current,
                    )
                    amp_obs_current = amp_obs_next

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if self._should_save_checkpoint(it):
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _should_save_checkpoint(self, iteration: int) -> bool:
        if iteration <= 0:
            return False
        if self.early_save_interval and self.early_save_iterations:
            if iteration <= self.early_save_iterations and iteration % self.early_save_interval == 0:
                return True
        if self.save_interval and iteration % self.save_interval == 0:
            return True
        return False

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Training metrics
        train_metrics = getattr(self.alg, "train_metrics", None)
        extra_metrics_string = ""
        if train_metrics:
            for key, value in train_metrics.items():
                if value is None:
                    continue
                if key == "cost_limit_margin":
                    self.writer.add_scalar("Train/cost_limit_margin", value, locs["it"])
                elif key == "mean_cost_return":
                    self.writer.add_scalar("Cost/mean", value, locs["it"])
                elif key == "cost_violation_rate":
                    self.writer.add_scalar("Cost/violation", value, locs["it"])
                elif key == "kl":
                    self.writer.add_scalar("Policy/kl", value, locs["it"])
                else:
                    self.writer.add_scalar(f"Train/{key}", value, locs["it"])
            # terminal extras for quick diagnosis
            mean_cost_return = train_metrics.get("mean_cost_return")
            if mean_cost_return is not None:
                extra_metrics_string += f"""{'Mean cost return:':>{pad}} {mean_cost_return:.4f}\n"""
            if hasattr(self.alg, "cost_limit"):
                extra_metrics_string += f"""{'Cost limit:':>{pad}} {self.alg.cost_limit:.4f}\n"""
            cost_limit_margin = train_metrics.get("cost_limit_margin")
            if cost_limit_margin is not None:
                extra_metrics_string += f"""{'Cost limit margin:':>{pad}} {cost_limit_margin:.4f}\n"""
            cost_violation_rate = train_metrics.get("cost_violation_rate")
            if cost_violation_rate is not None:
                extra_metrics_string += f"""{'Cost violation rate:':>{pad}} {cost_violation_rate:.4f}\n"""
            kl_value = train_metrics.get("kl")
            if kl_value is not None:
                extra_metrics_string += f"""{'Mean KL:':>{pad}} {kl_value:.6f}\n"""
            step_size = train_metrics.get("step_size")
            if step_size is not None:
                extra_metrics_string += f"""{'Step size:':>{pad}} {step_size:.6f}\n"""
        cost_scale = self._get_constraint_scale()
        if cost_scale is not None:
            extra_metrics_string += f"""{'Cost scale (1-gamma):':>{pad}} {cost_scale:.6f}\n"""

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            log_string += extra_metrics_string
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
            log_string += extra_metrics_string

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        optimizer_state = None
        if hasattr(self.alg, "optimizer") and self.alg.optimizer is not None:
            if isinstance(self.alg.optimizer, dict):
                optimizer_state = {key: opt.state_dict() for key, opt in self.alg.optimizer.items()}
            else:
                optimizer_state = self.alg.optimizer.state_dict()
        vae_optimizer_state = None
        if hasattr(self.alg, "vae_optimizer") and self.alg.vae_optimizer is not None:
            vae_optimizer_state = self.alg.vae_optimizer.state_dict()
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "vae_optimizer_state_dict": vae_optimizer_state,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            if resumed_training:
                # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
                # and the critic/teacher normalizer is loaded for the critic/teacher
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # if the training is not resumed but a model is loaded, this run must be distillation training following
                # an rl training. Thus the actor normalizer is loaded for the teacher model. The student's normalizer
                # is not loaded, as the observation space could differ from the previous rl training.
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            optimizer_state = loaded_dict.get("optimizer_state_dict")
            if optimizer_state is not None and hasattr(self.alg, "optimizer") and self.alg.optimizer is not None:
                if isinstance(self.alg.optimizer, dict):
                    for key, opt in self.alg.optimizer.items():
                        if key in optimizer_state:
                            opt.load_state_dict(optimizer_state[key])
                else:
                    self.alg.optimizer.load_state_dict(optimizer_state)
            # -- VAE optimizer
            vae_optimizer_state = loaded_dict.get("vae_optimizer_state_dict")
            if (
                vae_optimizer_state is not None
                and hasattr(self.alg, "vae_optimizer")
                and self.alg.vae_optimizer is not None
            ):
                self.alg.vae_optimizer.load_state_dict(vae_optimizer_state)
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
        if not self._vae_enabled:
            if self.cfg["empirical_normalization"]:
                policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
            return policy

        # VAE-enabled inference: build history and concatenate code
        def _vae_policy(obs: torch.Tensor):
            actor_obs = obs.to(self.device)
            if self._vae_obs_buffer is None:
                self._vae_obs_buffer = DataBuffer(
                    num_envs=self.env.num_envs,
                    num_data=self._vae_num_obs,
                    data_history_length=self._vae_history_length,
                    device=self.device,
                )
                all_ids = torch.arange(self.env.num_envs, device=self.device)
                self._vae_obs_buffer.reset(all_ids, new_data=actor_obs[:, : self._vae_num_obs])
            self._vae_obs_buffer.insert(actor_obs[:, : self._vae_num_obs])
            obs_history = self._vae_obs_buffer.get_all_data()
            policy_obs = self.alg.policy.get_policy_obs(actor_obs, obs_history, deterministic=True)
            if self.cfg["empirical_normalization"]:
                policy_obs = self.obs_normalizer(policy_obs)
            return self.alg.policy.act_inference(policy_obs)

        return _vae_policy

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    @staticmethod
    def _filter_kwargs(cls, kwargs: dict) -> dict:
        signature = inspect.signature(cls.__init__)
        valid_keys = set(signature.parameters.keys()) - {"self"}
        return {key: value for key, value in kwargs.items() if key in valid_keys}

    def _extract_costs(self, infos: dict, rewards: torch.Tensor) -> torch.Tensor:
        cost = infos.get("cost") if isinstance(infos, dict) else None
        if cost is None:
            return torch.zeros_like(rewards)
        if isinstance(cost, dict):
            if not cost:
                return torch.zeros_like(rewards)
            if self._constraint_normalizer is not None:
                cost, _ = self._constraint_normalizer.aggregate(cost)
            else:
                cost_values = []
                for value in cost.values():
                    if not torch.is_tensor(value):
                        value = torch.as_tensor(value, device=self.device)
                    value = value.to(self.device)
                    if value.ndim > 1 and value.shape[-1] > 1:
                        value = value.sum(dim=-1)
                    cost_values.append(torch.clamp(value, min=0.0))
                cost = cost_values[0]
                for value in cost_values[1:]:
                    cost = cost + value
        else:
            if self._constraint_normalizer is not None:
                cost, _ = self._constraint_normalizer.aggregate({"total": cost})
        if not torch.is_tensor(cost):
            cost = torch.as_tensor(cost, device=self.device)
        cost = torch.clamp(cost.to(self.device), min=0.0)
        cost_scale = self._get_constraint_scale()
        if cost_scale is not None and cost_scale != 1.0:
            cost = cost * cost_scale
        if cost.ndim > 1 and cost.shape[-1] > 1:
            cost = cost.sum(dim=-1)
        if cost.ndim == 1 and rewards.ndim > 1:
            cost = cost.unsqueeze(-1)
        return cost

    def _get_constraint_scale(self) -> float | None:
        if self._constraint_scale is not None:
            return float(self._constraint_scale)
        if not self._constraint_scale_by_gamma:
            return None
        cost_gamma = self._constraint_gamma
        if cost_gamma is None:
            return None
        if not (0.0 < cost_gamma < 1.0):
            return None
        return 1.0 - float(cost_gamma)

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _build_obs_term_slices(self, group_name: str) -> dict[str, slice]:
        obs_manager = getattr(self.env.unwrapped, "observation_manager", None)
        if obs_manager is None:
            return {}
        terms = obs_manager.active_terms.get(group_name, [])
        dims = obs_manager.group_obs_term_dim.get(group_name, [])
        slices: dict[str, slice] = {}
        idx = 0
        for name, shape in zip(terms, dims):
            length = math.prod(shape)
            slices[str(name)] = slice(idx, idx + length)
            idx += length
        return slices
