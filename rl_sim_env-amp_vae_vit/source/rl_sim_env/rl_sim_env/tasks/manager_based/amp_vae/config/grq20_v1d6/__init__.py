# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Rl-Sim-Env-AmpVae-Grq20-V1d6-v0",
    entry_point="rl_sim_env.envs:AmpVaeRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_vae_env_cfg:Grq20V1d6AmpVaeEnvCfg",
        "amp_vae_cfg_entry_point": f"{__name__}.config_summary:AmpVaePPORunnerCfg",
    },
)

gym.register(
    id="Rl-Sim-Env-AmpVae-Grq20-V1d6-Play-v0",
    entry_point="rl_sim_env.envs:AmpVaeRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_vae_env_cfg:Grq20V1d6AmpVaeEnvCfg_PLAY",
        "amp_vae_cfg_entry_point": f"{__name__}.config_summary:AmpVaePPORunnerCfg",
    },
)

gym.register(
    id="Rl-Sim-Env-AmpVae-Grq20-V1d6-ReplayAmpData-v0",
    entry_point="rl_sim_env.envs:AmpVaeRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_vae_env_cfg:Grq20V1d6AmpVaeEnvCfg_REPLAY_AMPDATA",
        "amp_vae_cfg_entry_point": f"{__name__}.config_summary:AmpVaePPORunnerCfg",
    },
)
