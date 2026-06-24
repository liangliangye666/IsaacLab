# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Package containing asset and sensor configurations."""
"""包含资产和传感器配置的包。"""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""
"""扩展源目录的路径。"""

ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""
"""扩展数据目录的路径。"""

ISAACLAB_ASSETS_METADATA = toml.load(os.path.join(ISAACLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""
"""从extension.toml文件中解析扩展元数据字典。"""

# Configure the module-level variables
__version__ = ISAACLAB_ASSETS_METADATA["package"]["version"]

from .robots import *
from .sensors import *
