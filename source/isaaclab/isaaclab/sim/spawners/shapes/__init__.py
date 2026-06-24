# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawning primitive shapes in the simulation.

NVIDIA Omniverse provides various primitive shapes that can be used to create USDGeom prims. Based
on the configuration, the spawned prim can be:

* a visual mesh (no physics)
* a static collider (no rigid body)
* a rigid body (with collision and rigid body properties).

"""
"""在仿真中产生的原始形状的子模块。

NVIDIA全宇宙提供了各种原始形状，可用于创建USDGeom prims。
根据配置，产生的prim可以是:

* 视觉网 (没有物理)
* 静态碰撞机 (没有硬体)
* 硬体 (具有碰撞和硬体特性)。
"""

from .shapes import spawn_capsule, spawn_cone, spawn_cuboid, spawn_cylinder, spawn_sphere
from .shapes_cfg import CapsuleCfg, ConeCfg, CuboidCfg, CylinderCfg, ShapeCfg, SphereCfg
