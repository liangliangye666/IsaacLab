# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.utils import configclass

from . import shapes


@configclass
class ShapeCfg(RigidObjectSpawnerCfg):
    """Configuration parameters for a USD Geometry or Geom prim."""
    """对USD几何或GEOM prim的配置参数。"""

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
class SphereCfg(ShapeCfg):
    """Configuration parameters for a sphere prim.

    See :meth:`spawn_sphere` for more information.
    """
    """对球体 prim的配置参数。

    See :麻:`spawn_sphere`更多信息。
    """

    func: Callable = shapes.spawn_sphere

    radius: float = MISSING
    """Radius of the sphere (in m)."""
    """球半径 (m)。"""


@configclass
class CuboidCfg(ShapeCfg):
    """Configuration parameters for a cuboid prim.

    See :meth:`spawn_cuboid` for more information.
    """
    """立方体prim的配置参数

    See :麻:`spawn_cuboid`更多信息。
    """

    func: Callable = shapes.spawn_cuboid

    size: tuple[float, float, float] = MISSING
    """Size of the cuboid."""
    """立方体的大小。"""


@configclass
class CylinderCfg(ShapeCfg):
    """Configuration parameters for a cylinder prim.

    See :meth:`spawn_cylinder` for more information.
    """
    """prim的配置参数

    See :麻:`spawn_cylinder`更多信息。
    """

    func: Callable = shapes.spawn_cylinder

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
class CapsuleCfg(ShapeCfg):
    """Configuration parameters for a capsule prim.

    See :meth:`spawn_capsule` for more information.
    """
    """一个prim囊的配置参数。

    See :麻:`spawn_capsule`更多信息。
    """

    func: Callable = shapes.spawn_capsule

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
class ConeCfg(ShapeCfg):
    """Configuration parameters for a cone prim.

    See :meth:`spawn_cone` for more information.
    """
    """对于prim角的配置参数。

    See :麻:`spawn_cone`更多信息。
    """

    func: Callable = shapes.spawn_cone

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
