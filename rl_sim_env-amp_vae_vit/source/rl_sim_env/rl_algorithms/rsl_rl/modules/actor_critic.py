# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.utils import resolve_nn_activation
from torch.distributions import Normal
from .vae import VAE


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        min_std=0.0,
        max_std=None,
        action_mean_clip=None,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        cost_critic_hidden_dims=None,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        vae: dict | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        cost_critic_hidden_dims = cost_critic_hidden_dims or critic_hidden_dims
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Cost value function (fully decoupled from reward critic)
        cost_critic_layers = []
        cost_critic_layers.append(nn.Linear(mlp_input_dim_c, cost_critic_hidden_dims[0]))
        cost_critic_layers.append(activation)
        for layer_index in range(len(cost_critic_hidden_dims)):
            if layer_index == len(cost_critic_hidden_dims) - 1:
                cost_critic_layers.append(nn.Linear(cost_critic_hidden_dims[layer_index], 1))
            else:
                cost_critic_layers.append(
                    nn.Linear(cost_critic_hidden_dims[layer_index], cost_critic_hidden_dims[layer_index + 1])
                )
                cost_critic_layers.append(activation)
        self.cost_critic = nn.Sequential(*cost_critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Cost Critic MLP: {self.cost_critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        if min_std is None:
            min_std = 0.0
        self.min_std = min_std
        self.max_std = max_std
        self.action_mean_clip = action_mean_clip

        # Optional VAE for auxiliary training (no effect unless enabled)
        self.vae_enabled = False
        self.vae = None
        self.vae_loss_coef = 0.0
        self.vae_kl_coef = 0.0
        self.vae_recon_dim = None
        self.vae_learning_rate = None
        if vae is not None:
            if hasattr(vae, "to_dict"):
                vae_cfg = vae.to_dict()
            elif isinstance(vae, dict):
                vae_cfg = dict(vae)
            else:
                vae_cfg = vars(vae)
            enabled = bool(vae_cfg.get("enabled", False))
            if enabled:
                cenet_in_dim = vae_cfg.get("cenet_in_dim")
                cenet_out_dim = vae_cfg.get("cenet_out_dim")
                cenet_recon_dim = vae_cfg.get("cenet_recon_dim", 45)
                if cenet_in_dim is None or cenet_out_dim is None:
                    raise ValueError("VAE enabled but 'cenet_in_dim' or 'cenet_out_dim' is not set.")
                self.vae = VAE(cenet_in_dim=cenet_in_dim, cenet_out_dim=cenet_out_dim, cenet_recon_dim=cenet_recon_dim)
                self.vae_enabled = True
                self.vae_loss_coef = float(vae_cfg.get("loss_coef", 1.0))
                self.vae_kl_coef = float(vae_cfg.get("kl_coef", 1.0))
                self.vae_recon_dim = int(cenet_recon_dim)
                self.vae_learning_rate = vae_cfg.get("learning_rate")

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        if self.action_mean_clip is not None:
            clip = abs(self.action_mean_clip)
            if clip > 0.0:
                mean = torch.clamp(mean, min=-clip, max=clip)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        std = torch.nan_to_num(std, nan=self.min_std, posinf=self.min_std, neginf=self.min_std)
        std = torch.clamp(std, min=self.min_std)
        if self.max_std is not None:
            max_std = abs(self.max_std)
            if max_std > 0.0:
                std = torch.clamp(std, max=max_std)
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        if self.action_mean_clip is not None:
            clip = abs(self.action_mean_clip)
            if clip > 0.0:
                actions_mean = torch.clamp(actions_mean, min=-clip, max=clip)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def evaluate_cost(self, critic_observations, **kwargs):
        cost_value = self.cost_critic(critic_observations)
        return cost_value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True

    def compute_vae_loss(self, obs_history: torch.Tensor | None, next_actor_obs: torch.Tensor | None):
        """Compute VAE reconstruction + KL loss for auxiliary training."""
        if not self.vae_enabled or self.vae is None:
            return None, {}
        if obs_history is None or next_actor_obs is None:
            return None, {}
        (
            _code,
            _code_vel,
            _code_mass,
            _code_com,
            _code_latent,
            decoded,
            mean_vel,
            logvar_vel,
            mean_latent,
            logvar_latent,
            mean_mass,
            logvar_mass,
            mean_com,
            logvar_com,
        ) = self.vae.cenet_forward(obs_history, deterministic=False)

        recon_dim = self.vae_recon_dim
        if recon_dim is None:
            recon_dim = decoded.shape[-1]
        recon_dim = min(int(recon_dim), int(decoded.shape[-1]), int(next_actor_obs.shape[-1]))
        recon_target = next_actor_obs[..., :recon_dim]
        recon_pred = decoded[..., :recon_dim]
        recon_loss = F.mse_loss(recon_pred, recon_target, reduction="mean")

        def _kl_loss(mean, logvar):
            return 0.5 * torch.mean(torch.exp(logvar) + mean.pow(2) - 1.0 - logvar)

        kl_loss = (
            _kl_loss(mean_vel, logvar_vel)
            + _kl_loss(mean_latent, logvar_latent)
            + _kl_loss(mean_mass, logvar_mass)
            + _kl_loss(mean_com, logvar_com)
        )

        total_loss = self.vae_loss_coef * recon_loss + self.vae_kl_coef * kl_loss
        metrics = {
            "vae_total": total_loss.detach(),
            "vae_recon": recon_loss.detach(),
            "vae_kl": kl_loss.detach(),
        }
        return total_loss, metrics

    def get_policy_obs(self, actor_obs: torch.Tensor, obs_history: torch.Tensor, deterministic: bool = True):
        """Build policy input by concatenating VAE code with actor observations."""
        if not self.vae_enabled or self.vae is None:
            return actor_obs
        code, *_ = self.vae.cenet_forward(obs_history, deterministic=deterministic)
        return torch.cat((code, actor_obs), dim=-1)
