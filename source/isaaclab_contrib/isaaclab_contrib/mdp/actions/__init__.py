# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for multirotor control.

This module provides action terms specifically designed for controlling multirotor
vehicles through thrust commands. These action terms integrate with Isaac Lab's
MDP framework and :class:`~isaaclab_contrib.assets.Multirotor` assets.
"""
"""多轮机控制的动作项。

该模块提供了专门用于通过推力指令控制多轮动力车辆的操作项。
这些动作项与艾萨克实验室的MDP框架和:class:`~isaaclab_contrib.assets.Multirotor`资产集成。
"""

from .thrust_actions import *  # noqa: F401, F403
from .thrust_actions_cfg import *  # noqa: F401, F403
