# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment environments for manipulation tasks.

These environments are designed for real-world deployment of manipulation tasks.
They containconfigurations and implementations that have been tested
and deployed on physical robots.

The deploy module includes:
- Reach environments for end-effector pose tracking

"""
"""操纵任务的部署环境。

这些环境是为了实在部署操纵任务。
它们包含已在物理机器人上测试和部署的配置和实施。

部署模块包括:
- 终端效应者姿势跟踪的覆盖环境
"""

from .reach import *  # noqa: F401, F403
