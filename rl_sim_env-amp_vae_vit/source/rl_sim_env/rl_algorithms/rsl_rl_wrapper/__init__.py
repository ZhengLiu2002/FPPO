# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrappers and utilities to configure an environment for RSL-RL library.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""

from .exporter import (
    export_amp_vae_perception_policy_as_onnx,
    export_amp_vae_policy_as_onnx,
    export_inference_cfg,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from .load import load_onnx_model, onnx_run_inference, verify_onnx_model
from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from .rl_cfg_amp_vae import (
    AmpVaeOnPolicyRunnerCfg,
    AmpVaePpoActorCriticCfg,
    AmpVaePpoAlgorithmCfg,
)
from .rl_cfg_amp_vae_perception import (
    AmpVaePerceptionOnPolicyRunnerCfg,
    AmpVaePerceptionPpoActorCriticCfg,
    AmpVaePerceptionPpoAlgorithmCfg,
)
from .rl_cfg_amp_vae_vit import (
    AmpVaeVitOnPolicyRunnerCfg,
    AmpVaeVitPpoActorCriticCfg,
    AmpVaeVitPpoAlgorithmCfg,
)
from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg
from .vecenv_wrapper import RslRlVecEnvWrapper
from .vecenv_wrapper_amp_vae import AmpVaeVecEnvWrapper
from .vecenv_wrapper_amp_vae_perception import AmpVaePerceptionVecEnvWrapper
from .vecenv_wrapper_amp_vae_vit import AmpVaeVitVecEnvWrapper
