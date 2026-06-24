# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for wrapping spawner configurations.

Unlike the other spawner modules, this module provides a way to wrap multiple spawner configurations
into a single configuration. This is useful when the user wants to spawn multiple assets based on
different configurations.
"""
"""包装子配置子模块

与其他产物模块不同，该模块提供了将多个产物模块配置包装成一个配置的方法。
这在用户想要基于不同的配置生成多个资产时是有用的。
"""

from .wrappers import spawn_multi_asset, spawn_multi_usd_file
from .wrappers_cfg import MultiAssetSpawnerCfg, MultiUsdFileCfg
