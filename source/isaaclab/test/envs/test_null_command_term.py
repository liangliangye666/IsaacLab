# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""
"""首先发射艾萨克仿真器。"""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""
"""休息，一切都跟着。"""

from collections import namedtuple

import pytest

from isaaclab.envs.mdp import NullCommandCfg


@pytest.fixture
def env():
    """Create a dummy environment."""
    """创建一个仿真的环境。"""
    return namedtuple("ManagerBasedRLEnv", ["num_envs", "dt", "device"])(20, 0.1, "cpu")


def test_str(env):
    """Test the string representation of the command manager."""
    """测试命令管理器的字符串表示。"""
    cfg = NullCommandCfg()
    command_term = cfg.class_type(cfg, env)
    # print the expected string
    print()
    print(command_term)


def test_compute(env):
    """Test the compute function. For null command generator, it does nothing."""
    """测试计算功能。
    对于零命令生成器来说，它什么都没有。
    """
    cfg = NullCommandCfg()
    command_term = cfg.class_type(cfg, env)

    # test the reset function
    command_term.reset()
    # test the compute function
    command_term.compute(dt=env.dt)
    # expect error
    with pytest.raises(RuntimeError):
        command_term.command
