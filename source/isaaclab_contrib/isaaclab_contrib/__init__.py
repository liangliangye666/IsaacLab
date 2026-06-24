# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package for externally contributed components for Isaac Lab.

This package provides externally contributed components for Isaac Lab, such as multirotors.
These components are not part of the core Isaac Lab framework yet, but are planned to be added
in the future. They are contributed by the community to extend the capabilities of Isaac Lab.
"""
"""对伊萨克实验室外部贡献的组件的包装。

该包为艾萨克实验室提供了外部贡献的组件，例如多旋转器。
这些组件尚未成为伊萨克实验室核心框架的一部分，但计划在未来增加。
社区为扩大艾萨克实验室的能力提供了这些资金。
"""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_CONTRIB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""
"""扩展源目录的路径。"""

ISAACLAB_CONTRIB_METADATA = toml.load(os.path.join(ISAACLAB_CONTRIB_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""
"""从extension.toml文件中解析扩展元数据字典。"""

# Configure the module-level variables
__version__ = ISAACLAB_CONTRIB_METADATA["package"]["version"]
