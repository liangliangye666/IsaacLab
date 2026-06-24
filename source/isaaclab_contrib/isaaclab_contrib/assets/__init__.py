# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for externally contributed assets.

This package provides specialized asset classes for simulating externally contributed
robots in Isaac Lab, such as multirotors. These assets are not part of the core
Isaac Lab framework yet, but are planned to be added in the future. They are
contributed by the community to extend the capabilities of Isaac Lab.
"""
"""外资资产子包

该包提供了专门的资产类别，用于仿真艾萨克实验室的外部贡献的机器人，例如多旋转机。
这些资产尚未成为Isaac Lab核心框架的一部分，但计划在未来增加。
社区为扩大艾萨克实验室的能力提供了这些资金。
"""

from .multirotor import Multirotor, MultirotorCfg, MultirotorData
