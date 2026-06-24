# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg
from isaaclab.utils import configclass

from . import meshes


@configclass
class MeshCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Configuration parameters for a USD Geometry or Geom prim.

    This class is similar to :class:`ShapeCfg` but is specifically for meshes.

    Meshes support both rigid and deformable properties. However, their schemas are applied at
    different levels in the USD hierarchy based on the type of the object. These are described below:

    - Deformable body properties: Applied to the mesh prim: ``{prim_path}/geometry/mesh``.
    - Collision properties: Applied to the mesh prim: ``{prim_path}/geometry/mesh``.
    - Rigid body properties: Applied to the parent prim: ``{prim_path}``.

    where ``{prim_path}`` is the path to the prim in the USD stage and ``{prim_path}/geometry/mesh``
    is the path to the mesh prim.

    .. note::
        There are mututally exclusive parameters for rigid and deformable properties. If both are set,
        then an error will be raised. This also holds if collision and deformable properties are set together.

    """
    """对USD几何或GEOM prim的配置参数。

    这类类似于:class:`ShapeCfg`，但专门适用于网格。

    子支持硬性和可变性。
    然而，根据对象类型，它们的方案在USD层次上应用于不同的层次。
    这些情况如下描述:

    - 可变形的车身特性:适用于网格prim:``{prim_path}/geometry/mesh``。
    - 碰撞性能:应用于网格 prim: ``{prim_path}/geometry/mesh``。
    - 固体特性:适用于母体prim:``{prim_path}``。

    在 USD 阶段 ``{prim_path}`` 是 prim 的路径，而 ``{prim_path}/geometry/mesh`` 是 prim 的网格路径。

    .. 说明::
        固体和变形性有相互排斥的参数。
        如果两者都被定制，
        碰撞和可变性质的组合也是如此。
    """

    visual_material_path: str = "material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """
    """在prim中使用的视觉材料的路径。
    "材料"的默认设置。

    如果路径是相对的，那么它将相对于prim的路径。
    如果`visual_material`不是None，则会忽略这个参数。
    """

    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """
    """视觉材料的特性。

    说明：
        如果是None，则不会添加任何视觉材料。
    """

    physics_material_path: str = "material"
    """Path to the physics material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `physics_material` is not None.
    """
    """进入prim的物理材料。
    "材料"的默认设置。

    如果路径是相对的，那么它将相对于prim的路径。
    如果`physics_material`不是None，则会忽略这个参数。
    """

    physics_material: materials.PhysicsMaterialCfg | None = None
    """Physics material properties.

    Note:
        If None, then no physics material will be added.
    """
    """物理材料的特性。

    说明：
        如果是None，则不会添加物理材料。
    """


@configclass
class MeshSphereCfg(MeshCfg):
    """Configuration parameters for a sphere mesh prim with deformable properties.

    See :meth:`spawn_mesh_sphere` for more information.
    """
    """具有可变性的球网prim的配置参数。

    See :麻:`spawn_mesh_sphere`更多信息。
    """

    func: Callable = meshes.spawn_mesh_sphere

    radius: float = MISSING
    """Radius of the sphere (in m)."""
    """球半径 (m)。"""


@configclass
class MeshCuboidCfg(MeshCfg):
    """Configuration parameters for a cuboid mesh prim with deformable properties.

    See :meth:`spawn_mesh_cuboid` for more information.
    """
    """具有可变性 prim的立方形网格配置参数。

    See :麻:`spawn_mesh_cuboid`更多信息。
    """

    func: Callable = meshes.spawn_mesh_cuboid

    size: tuple[float, float, float] = MISSING
    """Size of the cuboid (in m)."""
    """立方体的大小 (以m)。"""


@configclass
class MeshCylinderCfg(MeshCfg):
    """Configuration parameters for a cylinder mesh prim with deformable properties.

    See :meth:`spawn_cylinder` for more information.
    """
    """具有可变化性质的prim筒网的配置参数。

    See :麻:`spawn_cylinder`更多信息。
    """

    func: Callable = meshes.spawn_mesh_cylinder

    radius: float = MISSING
    """Radius of the cylinder (in m)."""
    """圆半径 (m)。"""
    height: float = MISSING
    """Height of the cylinder (in m)."""
    """的高度 (m)。"""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the cylinder. Defaults to "Z"."""
    """的轴。
    默认的"Z"。
    """


@configclass
class MeshCapsuleCfg(MeshCfg):
    """Configuration parameters for a capsule mesh prim.

    See :meth:`spawn_capsule` for more information.
    """
    """囊网 prim的配置参数。

    See :麻:`spawn_capsule`更多信息。
    """

    func: Callable = meshes.spawn_mesh_capsule

    radius: float = MISSING
    """Radius of the capsule (in m)."""
    """囊半径 (m)。"""
    height: float = MISSING
    """Height of the capsule (in m)."""
    """囊的高度 (m)。"""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the capsule. Defaults to "Z"."""
    """囊的轴。
    默认的"Z"。
    """


@configclass
class MeshConeCfg(MeshCfg):
    """Configuration parameters for a cone mesh prim.

    See :meth:`spawn_cone` for more information.
    """
    """子网 prim的配置参数

    See :麻:`spawn_cone`更多信息。
    """

    func: Callable = meshes.spawn_mesh_cone

    radius: float = MISSING
    """Radius of the cone (in m)."""
    """圆半径 (m)。"""
    height: float = MISSING
    """Height of the v (in m)."""
    """在 m 中的 v 的高度。"""
    axis: Literal["X", "Y", "Z"] = "Z"
    """Axis of the cone. Defaults to "Z"."""
    """子的轴。
    默认的"Z"。
    """
