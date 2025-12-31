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

from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from .vecenv_wrapper import RslRlVecEnvWrapper

__all__ = [
    "RslRlVecEnvWrapper",
    "RslRlOnPolicyRunnerCfg",
    "RslRlPpoActorCriticCfg",
    "RslRlPpoAlgorithmCfg",
]
