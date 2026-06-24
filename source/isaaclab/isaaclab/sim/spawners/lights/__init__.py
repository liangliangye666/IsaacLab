# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn lights in the simulation.

There are various different kinds of lights that can be spawned into the USD stage.
Please check the Omniverse documentation for `lighting overview
<https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html>`_.
"""
"""在仿真中产生灯光的产物器子模块。

在USD阶段可以产生各种不同的灯光。
请查看全宇宙文档`lighting overview
<https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html>`_。
"""

from .lights import spawn_light
from .lights_cfg import CylinderLightCfg, DiskLightCfg, DistantLightCfg, DomeLightCfg, LightCfg, SphereLightCfg
