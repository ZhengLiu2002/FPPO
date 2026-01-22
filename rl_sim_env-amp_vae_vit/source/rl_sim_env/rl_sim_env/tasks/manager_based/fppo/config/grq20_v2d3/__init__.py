# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

# 1. 注册标准训练环境 (V2d3)
gym.register(
    id="Isaac-FPPO-Grq20-V2d3-v0",
    entry_point="rl_sim_env.envs:FPPOEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fppo_env_cfg:Grq20V2d3FPPOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.config_summary:FPPORunnerCfg",
    },
)
# 2. 注册 Play 环境 (用于推理/演示)
gym.register(
    id="Isaac-FPPO-Grq20-V2d3-Play-v0",
    entry_point="rl_sim_env.envs:FPPOEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fppo_env_cfg:Grq20V2d3FPPOEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.config_summary:FPPORunnerCfg",
    },
)
