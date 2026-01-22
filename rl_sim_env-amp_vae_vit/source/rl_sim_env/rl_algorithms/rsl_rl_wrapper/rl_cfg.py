# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    min_std: float = 0.0
    """Minimum standard deviation clamp for action noise."""

    max_std: float | None = None
    """Maximum standard deviation clamp for action noise. If None, no upper bound is applied."""

    action_mean_clip: float | None = None
    """Clamp value for action means before sampling/inference. If None, no clipping is applied."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    cost_critic_hidden_dims: list[int] | None = None
    """The hidden dimensions of the cost critic network. Defaults to critic_hidden_dims if None."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for CMDP-capable on-policy algorithms."""

    name: str = "fppo"
    """The algorithm name (e.g., 'fppo', 'ppo_lagrange', 'cpo', 'pcpo', 'focpo')."""

    class_name: str | None = None
    """Optional algorithm class name. Default is None (use name-based selection)."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    cost_value_loss_coef: float = 1.0
    """The coefficient for the cost value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    step_size: float = 1e-3
    """The step size for projected policy updates (FPPO/PCPO)."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    cost_gamma: float | None = None
    """The discount factor for costs. Defaults to gamma if None."""

    cost_lam: float | None = None
    """The GAE lambda for costs. Defaults to lam if None."""

    cost_limit: float = 0.0
    """The cost limit for CMDP constraints."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    delta_safe: float | None = 0.01
    """Trust region radius for FPPO step correction. If None, disables backtracking."""

    backtrack_coeff: float = 0.5
    """Step size decay factor for FPPO backtracking."""

    max_backtracks: int = 10
    """Maximum number of backtracking steps for FPPO."""

    projection_eps: float = 1e-8
    """Epsilon used in projection to avoid divide-by-zero."""

    lagrange_lr: float = 1e-2
    """Learning rate for Lagrange multiplier (PPO-Lagrange)."""

    lagrange_max: float = 100.0
    """Maximum value for Lagrange multiplier."""

    focpo_eta: float = 0.02
    """KL threshold for FOCPO-style updates."""

    focpo_lambda: float = 1.0
    """FOCPO loss scaling parameter."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the entire collected trajectories.
    Otherwise, the advantage is normalized over the mini-batches only.
    """
    normalize_cost_advantage: bool = False
    """Whether to normalize the cost advantage. Default is False."""


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    clip_actions: float | None = None
    """The clipping value for actions. If ``None``, then no clipping is done.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "wandb"
    """The logger to use. Default is wandb."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
