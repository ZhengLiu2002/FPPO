# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Rl-Sim-Env-AmpVaeVit-E1-V1d6-E1R-v0",
    entry_point="rl_sim_env.envs:AmpVaeVitRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_vae_vit_env_cfg:E1V1d6E1rAmpVaeVitEnvCfg",
        "amp_vae_vit_cfg_entry_point": f"{__name__}.config_summary:AmpVaeVitPPORunnerCfg",
    },
)

gym.register(
    id="Rl-Sim-Env-AmpVaeVit-E1-V1d6-E1R-Play-v0",
    entry_point="rl_sim_env.envs:AmpVaeVitRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_vae_vit_env_cfg:E1V1d6E1rAmpVaeVitEnvCfg_PLAY",
        "amp_vae_vit_cfg_entry_point": f"{__name__}.config_summary:AmpVaeVitPPORunnerCfg",
    },
)

gym.register(
    id="Rl-Sim-Env-AmpVaeVit-E1-V1d6-E1R-ReplayAmpData-v0",
    entry_point="rl_sim_env.envs:AmpVaeVitRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_vae_vit_env_cfg:E1V1d6E1rAmpVaeVitEnvCfg_REPLAY_AMPDATA",
        "amp_vae_vit_cfg_entry_point": f"{__name__}.config_summary:AmpVaeVitPPORunnerCfg",
    },
)
