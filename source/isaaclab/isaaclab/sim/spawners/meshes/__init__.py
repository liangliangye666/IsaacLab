# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawning meshes in the simulation.

NVIDIA Omniverse deals with meshes as `USDGeomMesh`_ prims. This sub-module provides various
configurations to spawn different types of meshes. Based on the configuration, the spawned prim can be:

* a visual mesh (no physics)
* a static collider (no rigid or deformable body)
* a deformable body (with deformable properties)

.. note::
    While rigid body properties can be set on a mesh, it is recommended to use the
    :mod:`isaaclab.sim.spawners.shapes` module to spawn rigid bodies. This is because USD shapes
    are more optimized for physics simulations.

.. _USDGeomMesh: https://openusd.org/release/api/class_usd_geom_mesh.html
"""
"""在仿真中，用于产卵网的子模块。

NVIDIA全宇宙处理网格作为`USDGeomMesh`_prims。
这一子模块提供了各种配置，以产生不同类型的网格。
根据配置，产生的prim可以是:

* 视觉网 (没有物理)
* 静态碰撞器 (没有硬体或可变形体)
* 可变形体 (具有变形性质)

.. 说明::
    虽然可以设置硬体特性在网格上，但建议使用:mod:`isaaclab.sim.spawners.shapes`模块产生硬体。
    这是因为USD形状更适合物理仿真。

.. _USDGeomMesh: https://openusd.org/release/api/class_usd_geom_mesh.html
"""

from .meshes import spawn_mesh_capsule, spawn_mesh_cone, spawn_mesh_cuboid, spawn_mesh_cylinder, spawn_mesh_sphere
from .meshes_cfg import MeshCapsuleCfg, MeshCfg, MeshConeCfg, MeshCuboidCfg, MeshCylinderCfg, MeshSphereCfg
