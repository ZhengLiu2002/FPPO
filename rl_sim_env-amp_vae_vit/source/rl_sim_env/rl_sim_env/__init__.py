# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

import os
from pathlib import Path

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

_env_root_override = os.environ.get("RL_SIM_ENV_ROOT_DIR")
if _env_root_override:
    RL_SIM_ENV_ROOT_DIR = Path(_env_root_override).expanduser().resolve()
else:
    _here_parents = Path(__file__).resolve().parents
    _auto_root = None
    for _candidate in _here_parents:
        if (_candidate / "assets").is_dir():
            _auto_root = _candidate
            break
    # fallback to previous heuristic if nothing found
    RL_SIM_ENV_ROOT_DIR = _auto_root if _auto_root else _here_parents[4]
