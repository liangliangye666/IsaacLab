# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from . import physics_materials


@configclass
class PhysicsMaterialCfg:
    """Configuration parameters for creating a physics material.

    Physics material are PhysX schemas that can be applied to a USD material prim to define the
    physical properties related to the material. For example, the friction coefficient, restitution
    coefficient, etc. For more information on physics material, please refer to the
    `PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxBaseMaterial.html>`__.
    """
    """制造物理材料的配置参数。

    物理材料是PhysX方案，可以应用于USD材料prim来定义与材料相关的物理特性。
    例如摩擦系数，恢复系数等。
    更多关于物理材料的信息请参阅`PhysX documentation
    <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxBaseMaterial.html>`__。
    """

    func: Callable = MISSING
    """Function to use for creating the material."""
    """用于创建材料的功能。"""


@configclass
class RigidBodyMaterialCfg(PhysicsMaterialCfg):
    """Physics material parameters for rigid bodies.

    See :meth:`spawn_rigid_body_material` for more information.
    """
    """物理材料对硬体的参数。

    See :麻:`spawn_rigid_body_material`更多信息。
    """

    func: Callable = physics_materials.spawn_rigid_body_material

    static_friction: float = 0.5
    """The static friction coefficient. Defaults to 0.5."""
    """静态摩擦系数。
    默认为0.5。
    """

    dynamic_friction: float = 0.5
    """The dynamic friction coefficient. Defaults to 0.5."""
    """动态摩擦系数。
    默认为0.5。
    """

    restitution: float = 0.0
    """The restitution coefficient. Defaults to 0.0."""
    """退还系数。
    默认为0.0。
    """

    friction_combine_mode: Literal["average", "min", "multiply", "max"] = "average"
    """Determines the way friction will be combined during collisions. Defaults to `"average"`.

    .. attention::

        When two physics materials with different combine modes collide, the combine mode with the higher
        priority will be used. The priority order is provided `here
        <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxCombineMode.html>`__.
    """
    """在碰撞时，摩擦将如何结合。
    在`"average"`上默认。

    .. 注意::

        当两个不同组合模式的物理材料碰撞时，将使用具有更高优先级的组合模式。
        提供优先顺序`here
        <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxCombineMode.html>`__。
    """

    restitution_combine_mode: Literal["average", "min", "multiply", "max"] = "average"
    """Determines the way restitution coefficient will be combined during collisions. Defaults to `"average"`.

    .. attention::

        When two physics materials with different combine modes collide, the combine mode with the higher
        priority will be used. The priority order is provided `here
        <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxCombineMode.html>`__.
    """
    """在碰撞时，确定恢复系数的结合方式。
    在`"average"`上默认。

    .. 注意::

        当两个不同组合模式的物理材料碰撞时，将使用具有更高优先级的组合模式。
        提供优先顺序`here
        <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxCombineMode.html>`__。
    """

    compliant_contact_stiffness: float = 0.0
    """Spring stiffness for a compliant contact model using implicit springs. Defaults to 0.0.

    A higher stiffness results in behavior closer to a rigid contact. The compliant contact model is only enabled
    if the stiffness is larger than 0.
    """
    """使用隐形弹的合规接触模型的弹硬性。
    默认为0.0。

    较高的硬度会导致行为更接近硬接触。
    只有符合标准的接触模式才能实现
    if the stiffness is larger than 0.
    """

    compliant_contact_damping: float = 0.0
    """Damping coefficient for a compliant contact model using implicit springs. Defaults to 0.0.

    Irrelevant if compliant contacts are disabled when :obj:`compliant_contact_stiffness` is set to zero and
    rigid contacts are active.
    """
    """使用隐形弹的符合条件的接触模型的缩系数。
    默认为0.0。

    如果 :obj:`compliant_contact_stiffness` 设置为零，且刚性接触事件时，不重要的是是否禁用符合标准的联系方式。
    """


@configclass
class DeformableBodyMaterialCfg(PhysicsMaterialCfg):
    """Physics material parameters for deformable bodies.

    See :meth:`spawn_deformable_body_material` for more information.

    """
    """变形体的物理材料参数。

    See :麻:`spawn_deformable_body_material`更多信息。
    """

    func: Callable = physics_materials.spawn_deformable_body_material

    density: float | None = None
    """The material density. Defaults to None, in which case the simulation decides the default density."""
    """材料密度。
    默认为 None，在这种情况下，仿真决定默认密度。
    """

    dynamic_friction: float = 0.25
    """The dynamic friction. Defaults to 0.25."""
    """动态摩擦。
    默认值为0.25。
    """

    youngs_modulus: float = 50000000.0
    """The Young's modulus, which defines the body's stiffness. Defaults to 50000000.0.

    The Young's modulus is a measure of the material's ability to deform under stress. It is measured in Pascals (Pa).
    """
    """年轻人的模块，定义了身体的度。
    默认到50000000.0。

    青少年的模块是材料在压力下变形的能力的衡量。
    它以Pascals (Pa) 测量。
    """

    poissons_ratio: float = 0.45
    """The Poisson's ratio which defines the body's volume preservation. Defaults to 0.45.

    The Poisson's ratio is a measure of the material's ability to expand in the lateral direction when compressed
    in the axial direction. It is a dimensionless number between 0 and 0.5. Using a value of 0.5 will make the
    material incompressible.
    """
    """波森的比率定义了身体的体积保存。
    默认为0.45。

    波森比率是材料在轴向压缩时在侧向扩张的能力的衡量量。
    它是0到0.5之间的无维数。
    使用0.5的值将使材料不可压缩。
    """

    elasticity_damping: float = 0.005
    """The elasticity damping for the deformable material. Defaults to 0.005."""
    """可变形材料的弹性缩。
    默认为0.005。
    """

    damping_scale: float = 1.0
    """The damping scale for the deformable material. Defaults to 1.0.

    A scale of 1 corresponds to default damping. A value of 0 will only apply damping to certain motions leading
    to special effects that look similar to water filled soft bodies.
    """
    """可变形材料的缩尺度。
    默认到1.0。

    一个尺度相当于默认缩。
    0的值只会对某些运动进行缩，从而产生类似于充满水的软体的特殊效应。
    """
