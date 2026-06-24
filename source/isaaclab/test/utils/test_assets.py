# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
"""首先发射艾萨克仿真器。"""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""
"""休息，一切都跟着。"""

import isaaclab.utils.assets as assets_utils


def test_nucleus_connection():
    """Test checking the Nucleus connection."""
    """测试核连接。"""
    # check nucleus connection
    assert assets_utils.NUCLEUS_ASSET_ROOT_DIR is not None


def test_check_file_path_nucleus():
    """Test checking a file path on the Nucleus server."""
    """检查核电脑服务器的文件路径。"""
    # robot file path
    usd_path = f"{assets_utils.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    # check file path
    assert assets_utils.check_file_path(usd_path) == 2


def test_check_file_path_invalid():
    """Test checking an invalid file path."""
    """检查一个无效的文件路径。"""
    # robot file path
    usd_path = f"{assets_utils.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_xyz.usd"
    # check file path
    assert assets_utils.check_file_path(usd_path) == 0


def test_check_usd_path_with_timeout():
    """Test checking a USD path with timeout."""
    """测试检查一个USD路径，"""
    # robot file path
    usd_path = f"{assets_utils.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    # check file path
    assert assets_utils.check_usd_path_with_timeout(usd_path) is True

    # invalid file path
    usd_path = f"{assets_utils.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_xyz.usd"
    # check file path
    assert assets_utils.check_usd_path_with_timeout(usd_path) is False
