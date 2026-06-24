# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments.

The package is structured as follows:

- ``direct``: These include single-file implementations of tasks.
- ``manager_based``: These include task implementations that use the manager-based API.
- ``utils``: These include utility functions for the tasks.

"""
"""包含各种机器人环境的任务实现的包。

包装结构如下:

- ``direct``:其中包括单文件执行任务。
- ``manager_based``:包括使用基于管理器的API的任务实现。
- ``utils``:包括任务的实用功能。
"""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""
"""扩展源目录的路径。"""

ISAACLAB_TASKS_METADATA = toml.load(os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""
"""从extension.toml文件中解析扩展元数据字典。"""

# Configure the module-level variables
__version__ = ISAACLAB_TASKS_METADATA["package"]["version"]

##
# Register Gym environments.
##

from .utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
# TODO(@ashwinvk): Remove pick_place from the blacklist once pinocchio from Isaac Sim is compatibility
_BLACKLIST_PKGS = ["utils", ".mdp", "pick_place", "direct.humanoid_amp.motions"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
