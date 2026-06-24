# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from pxr import Usd

from isaaclab.sim import schemas
from isaaclab.utils import configclass


@configclass
class SpawnerCfg:
    """Configuration parameters for spawning an asset.

    Spawning an asset is done by calling the :attr:`func` function. The function takes in the
    prim path to spawn the asset at, the configuration instance and transformation, and returns the
    prim path of the spawned asset.

    The function is typically decorated with :func:`isaaclab.sim.spawner.utils.clone` decorator
    that checks if input prim path is a regex expression and spawns the asset at all matching prims.
    For this, the decorator uses the Cloner API from Isaac Sim and handles the :attr:`copy_from_source`
    parameter.
    """
    """对资产产产生的配置参数

    通过调用 :attr:`func` 函数来生成资产。
    函数采用prim路径来生成产品，配置实例和转换，并返回产品的prim路径。

    函数通常用:func:`isaaclab.sim.spawner.utils.clone`装饰器进行装饰，检查输入prim路径是否是regex表达式，并产生所有匹配prims的资产。
    为此，装饰师使用了Isaac Sim的Cloner API，并处理:attr:`copy_from_source`参数。
    """

    func: Callable[..., Usd.Prim] = MISSING
    """Function to use for spawning the asset.

    The function takes in the prim path (or expression) to spawn the asset at, the configuration instance
    and transformation, and returns the source prim spawned.
    """
    """功能用于产产资产。

    函数采用prim路径 (或表达式) 来生成产品，配置实例和转换，并返回产生的源prim。
    """

    visible: bool = True
    """Whether the spawned asset should be visible. Defaults to True."""
    """产生的资产是否可见。
    默认为 True。
    """

    semantic_tags: list[tuple[str, str]] | None = None
    """List of semantic tags to add to the spawned asset. Defaults to None,
    which means no semantic tags will be added.

    The semantic tags follow the `Replicator Semantic` tagging system. Each tag is a tuple of the
    form ``(type, data)``, where ``type`` is the type of the tag and ``data`` is the semantic label
    associated with the tag. For example, to annotate a spawned asset in the class avocado, the semantic
    tag would be ``[("class", "avocado")]``.

    You can specify multiple semantic tags by passing in a list of tags. For example, to annotate a
    spawned asset in the class avocado and the color green, the semantic tags would be
    ``[("class", "avocado"), ("color", "green")]``.

    .. seealso::

        For more information on the semantics filter, see the documentation for the `semantics schema editor`_.

    .. _semantics schema editor: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html#semantics-filtering

    """
    """在产生的资产中添加的语义标签列表。
    在None中默认设置，这意味着不会添加任何语义标签。

    语义标签遵循`Replicator Semantic`标签系统。
    每个标签是 ``(type， data)`` 形式的图پل，其中 ``type`` 是标签的类型，而 ``data`` 是与标签相关的语义标签。
    例如，在 avo梨类中注释产生的资产，语义标签将是``[("class"， "avocado")]``。

    你可以通过通过标签列表来指定多个语义标签。
    例如，在 green梨类和绿色中注释产生的资产，语义标签将是``[("class"， "avocado")， ("color"， "green")]``。

    ..
    查看:

        对于语义过器的更多信息，请参见`semantics schema editor`_的文档。

    .. _semantics schema editor: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html#semantics-filtering
    """

    copy_from_source: bool = True
    """Whether to copy the asset from the source prim or inherit it. Defaults to True.

    This parameter is only used when cloning prims. If False, then the asset will be inherited from
    the source prim, i.e. all USD changes to the source prim will be reflected in the cloned prims.
    """
    """复制源prim的资产或继承。
    默认为 True。

    这一参数仅用于克隆prims时。
    如果False，那么资产将从源 prim继承，i.e.所有USD变化到源 prim将反映在克隆 prims。
    """


@configclass
class RigidObjectSpawnerCfg(SpawnerCfg):
    """Configuration parameters for spawning a rigid asset.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.
    """
    """固体产物产生的配置参数

    说明：
        默认情况下，所有属性设置为None。
        这意味着将不会添加或修改任何特性prim在产卵时默认可用的属性之外prim。
    """

    mass_props: schemas.MassPropertiesCfg | None = None
    """Mass properties."""
    """大量物质。"""

    rigid_props: schemas.RigidBodyPropertiesCfg | None = None
    """Rigid body properties.

    For making a rigid object static, set the :attr:`schemas.RigidBodyPropertiesCfg.kinematic_enabled`
    as True. This will make the object static and will not be affected by gravity or other forces.
    """
    """固体特性。

    为了使硬体静态，设置:attr:`schemas.RigidBodyPropertiesCfg.kinematic_enabled`为True。
    这将使物体静止，不会受到重力或其他力量的影响。
    """

    collision_props: schemas.CollisionPropertiesCfg | None = None
    """Properties to apply to all collision meshes."""
    """适用于所有碰撞网格的特性。"""

    activate_contact_sensors: bool = False
    """Activate contact reporting on all rigid bodies. Defaults to False.

    This adds the PhysxContactReporter API to all the rigid bodies in the given prim path and its children.
    """
    """激活所有硬体的接触报告。
    默认为 False。

    这将PhysxContactReporter API添加到给定的prim路径中的所有硬体及其子女中。
    """


@configclass
class DeformableObjectSpawnerCfg(SpawnerCfg):
    """Configuration parameters for spawning a deformable asset.

    Unlike rigid objects, deformable objects are affected by forces and can deform when subjected to
    external forces. This class is used to configure the properties of the deformable object.

    Deformable bodies don't have a separate collision mesh. The collision mesh is the same as the visual mesh.
    The collision properties such as rest and collision offsets are specified in the :attr:`deformable_props`.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.
    """
    """对可变化资产的产卵配置参数。

    与硬物体不同，可变形的物体受到外力的影响，并且在受到外力的影响时可以变形。
    这类用于配置可变形物体的属性。

    变形体没有独立的碰撞网。
    碰撞网格与视觉网格相同。
    在:attr:`deformable_props`中指定了休息和碰撞抵消等碰撞特性。

    说明：
        默认情况下，所有属性设置为None。
        这意味着将不会添加或修改任何特性prim在产卵时默认可用的属性之外prim。
    """

    mass_props: schemas.MassPropertiesCfg | None = None
    """Mass properties."""
    """大量物质。"""

    deformable_props: schemas.DeformableBodyPropertiesCfg | None = None
    """Deformable body properties."""
    """可变形的身体特性。"""
