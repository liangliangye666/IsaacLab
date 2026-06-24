# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .terrain_importer import TerrainImporter

if TYPE_CHECKING:
    from .terrain_generator_cfg import TerrainGeneratorCfg


@configclass
class TerrainImporterCfg:
    """Configuration for the terrain manager."""
    """对于地形管理器的配置。"""

    class_type: type = TerrainImporter
    """The class to use for the terrain importer.

    Defaults to :class:`isaaclab.terrains.terrain_importer.TerrainImporter`.
    """
    """在地形进口商使用的类型。

    在:class:`isaaclab.terrains.terrain_importer.TerrainImporter`上默认。
    """

    collision_group: int = -1
    """The collision group of the terrain. Defaults to -1."""
    """土地的碰撞组。
    设置为 -1。
    """

    prim_path: str = MISSING
    """The absolute path of the USD terrain prim.

    All sub-terrains are imported relative to this prim path.
    """
    """在USD地形prim的绝对路径。

    所有地表都是对此prim路径进行进口的。
    """

    num_envs: int = 1
    """The number of environment origins to consider. Defaults to 1.

    In case, the :class:`~isaaclab.scene.InteractiveSceneCfg` is used, this parameter gets overridden by
    :attr:`isaaclab.scene.InteractiveSceneCfg.num_envs` attribute.
    """
    """需要考虑的环境来源数量。
    默认的1。

    如果使用:class:`~isaaclab.scene.InteractiveSceneCfg`，这个参数将被:attr:`isaaclab.scene.InteractiveSceneCfg.num
    _envs`属性覆盖。
    """

    terrain_type: Literal["generator", "plane", "usd"] = "generator"
    """The type of terrain to generate. Defaults to "generator".

    Available options are "plane", "usd", and "generator".
    """
    """这种地形需要产生。
    默认的发电机。

    可用的选项是"飞机"，"usd"和"发电机"。
    """

    terrain_generator: TerrainGeneratorCfg | None = None
    """The terrain generator configuration.

    Only used if ``terrain_type`` is set to "generator".
    """
    """土地发电机配置。

    只有在``terrain_type``设置为"生成器"时使用。
    """

    usd_path: str | None = None
    """The path to the USD file containing the terrain.

    Only used if ``terrain_type`` is set to "usd".
    """
    """包含地形的USD文件。

    只有当``terrain_type``设置为"usd"时使用。
    """

    env_spacing: float | None = None
    """The spacing between environment origins when defined in a grid. Defaults to None.

    Note:
      This parameter is used only when the ``terrain_type`` is "plane" or "usd" or if
      :attr:`use_terrain_origins` is False.
    """
    """在网格中定义环境起源之间的距离。
    默认为 None。

    说明：
      如果 ``terrain_type`` 是"平面"或"usd"，或者如果 :attr:`use_terrain_origins` 是 False，则使用此参数。
    """

    use_terrain_origins: bool = True
    """Whether to set the environment origins based on the terrain origins or in a grid
    according to :attr:`env_spacing`. Defaults to True.

    Note:
      This parameter is used only when the :attr:`terrain type` is "generator".
    """
    """基于地形起源或根据:attr:`env_spacing`的网格设置环境起源。
    默认为 True。

    说明：
      这一参数只有当:attr:`terrain type`是"生成器"时才使用。
    """

    visual_material: sim_utils.VisualMaterialCfg | None = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
    """The visual material of the terrain. Defaults to a dark gray color material.

    This parameter is used for both the "generator" and "plane" terrains.

    - If the ``terrain_type`` is "generator", then the material is created at the path
      ``{prim_path}/visualMaterial`` and applied to all the sub-terrains.
    - If the ``terrain_type`` is "plane", then the diffuse color of the material is set to
      to the grid color of the imported ground plane.
    """
    """视觉地形的材料。
    黑色灰色材料的默认设置。

    这一参数用于"发电机"和"平面"地形。

    - 如果``terrain_type``是"发电器"，那么该材料是在``{prim_path}/visualMaterial``路径上创建并应用于所有地下。
    - 如果``terrain_type``是"平面"，则材料的散射颜色设置为进口的地面平面的网格颜色。
    """

    physics_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg()
    """The physics material of the terrain. Defaults to a default physics material.

    The material is created at the path: ``{prim_path}/physicsMaterial``.

    .. note::
        This parameter is used only when the ``terrain_type`` is "generator" or "plane".
    """
    """在地形的物理材料。
    物理材料的默认设置。

    材料是在``{prim_path}/physicsMaterial``路径上创建的。

    .. 说明::
        这一参数只有当``terrain_type``是"生成器"或"平面"时才使用。
    """

    max_init_terrain_level: int | None = None
    """The maximum initial terrain level for defining environment origins. Defaults to None.

    The terrain levels are specified by the number of rows in the grid arrangement of
    sub-terrains. If None, then the initial terrain level is set to the maximum
    terrain level available (``num_rows - 1``).

    Note:
      This parameter is used only when sub-terrain origins are defined.
    """
    """定义环境起源的初始地形水平。
    默认为 None。

    地形层面由地形下层格格的排行数量指定。
    如果None，则初始地形水平设置为可用的最高地形水平 (``num_rows - 1``)。

    说明：
      这一参数仅用于地下起源定义时。
    """

    debug_vis: bool = False
    """Whether to enable visualization of terrain origins for the terrain. Defaults to False."""
    """能否实现地形起源的视觉化。
    默认为 False。
    """
