# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Rl-Sim-Env-AmpVaePerception-E1-V1d6-E1R-v0",
    entry_point="rl_sim_env.envs:AmpVaePerceptionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_vae_perception_env_cfg:E1V1d6E1rAmpVaePerceptionEnvCfg",
        "amp_vae_perception_cfg_entry_point": f"{__name__}.config_summary:AmpVaePerceptionPPORunnerCfg",
    },
)

gym.register(
    id="Rl-Sim-Env-AmpVaePerception-E1-V1d6-E1R-Play-v0",
    entry_point="rl_sim_env.envs:AmpVaePerceptionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_vae_perception_env_cfg:E1V1d6E1rAmpVaePerceptionEnvCfg_PLAY",
        "amp_vae_perception_cfg_entry_point": f"{__name__}.config_summary:AmpVaePerceptionPPORunnerCfg",
    },
)

gym.register(
    id="Rl-Sim-Env-AmpVaePerception-E1-V1d6-E1R-ReplayAmpData-v0",
    entry_point="rl_sim_env.envs:AmpVaePerceptionRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_vae_perception_env_cfg:E1V1d6E1rAmpVaePerceptionEnvCfg_REPLAY_AMPDATA",
        "amp_vae_perception_cfg_entry_point": f"{__name__}.config_summary:AmpVaePerceptionPPORunnerCfg",
    },
)
