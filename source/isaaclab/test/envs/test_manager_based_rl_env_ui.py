# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
"""首先发射艾萨克仿真器。"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""
"""休息，一切都跟着。"""

import carb
import omni.usd
from isaacsim.core.utils.extensions import enable_extension

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

enable_extension("isaacsim.gui.components")


@configclass
class EmptyManagerCfg:
    """Empty manager specifications for the environment."""
    """管理器对环境的规格是空的。"""

    pass


@configclass
class EmptySceneCfg(InteractiveSceneCfg):
    """Configuration for an empty scene."""
    """设置为空场景。"""

    pass


def get_empty_base_env_cfg(device: str = "cuda:0", num_envs: int = 1, env_spacing: float = 1.0):
    """Generate base environment config based on device"""
    """根据设备生成基环境配置"""

    @configclass
    class EmptyEnvCfg(ManagerBasedRLEnvCfg):
        """Configuration for the empty test environment."""
        """对于空试环境的配置。"""

        # Scene settings
        scene: EmptySceneCfg = EmptySceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        # Basic settings
        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyManagerCfg = EmptyManagerCfg()
        rewards: EmptyManagerCfg = EmptyManagerCfg()
        terminations: EmptyManagerCfg = EmptyManagerCfg()
        # Define window
        ui_window_class_type: type[ManagerBasedRLEnvWindow] = ManagerBasedRLEnvWindow

        def __post_init__(self):
            """Post initialization."""
            """在初始化后。"""
            # step settings
            self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
            # simulation settings
            self.sim.dt = 0.005  # sim step every 5ms: 200Hz
            self.sim.render_interval = self.decimation  # render every 4 sim steps
            # pass device down from test
            self.sim.device = device
            # episode length
            self.episode_length_s = 5.0

    return EmptyEnvCfg()


def test_ui_window():
    """Test UI window of ManagerBasedRLEnv."""
    """测试ManagerBasedRLEnv的UI窗口。"""
    device = "cuda:0"
    # override sim setting to enable UI
    carb.settings.get_settings().set_bool("/app/window/enabled", True)
    # create a new stage
    omni.usd.get_context().new_stage()
    # create environment
    env = ManagerBasedRLEnv(cfg=get_empty_base_env_cfg(device=device))
    # close the environment
    env.close()
