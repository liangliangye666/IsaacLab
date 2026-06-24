# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class ActuatorBaseCfg:
    """Configuration for default actuators in an articulation."""
    """在关节中默认执行器的配置。"""

    class_type: type = MISSING
    """The associated actuator class.

    The class should inherit from :class:`isaaclab.actuators.ActuatorBase`.
    """
    """相关的执行器类。

    这类应该继承:class:`isaaclab.actuators.ActuatorBase`。
    """

    joint_names_expr: list[str] = MISSING
    """Articulation's joint names that are part of the group.

    Note:
        This can be a list of joint names or a list of regex expressions (e.g. ".*").
    """
    """关节的共同名字是集团的一部分。

    说明：
        这可能是共同名称列表或regex表达式列表 (e.g。 ".*")。
    """

    effort_limit: dict[str, float] | float | None = None
    """Force/Torque limit of the joints in the group. Defaults to None.

    This limit is used to clip the computed torque sent to the simulation. If None, the
    limit is set to the value specified in the USD joint prim.

    .. attention::

        The :attr:`effort_limit_sim` attribute should be used to set the effort limit for
        the simulation physics solver.

        The :attr:`effort_limit` attribute is used for clipping the effort output of the
        actuator model **only** in the case of explicit actuators, such as the
        :class:`~isaaclab.actuators.IdealPDActuator`.

    .. note::

        For implicit actuators, the attributes :attr:`effort_limit` and :attr:`effort_limit_sim`
        are equivalent. However, we suggest using the :attr:`effort_limit_sim` attribute because
        it is more intuitive.

    """
    """组中关节的力/扭矩限制。
    默认为 None。

    这一极限用于切断到仿真中发送的计算扭矩。
    如果是None，则设定为USD联合prim中指定的值。

    .. 注意::

        The :应使用 attr:`effort_limit_sim`属性来设定
        仿真物理解决器。

        The :attr:`effort_limit`属性用于裁剪输出力量的
        执行器模型 **仅**在:class:`~isaaclab.actuators.IdealPDActuator`等明确执行器的情况下。

    .. 说明::

        对于隐含执行器，:attr:`effort_limit`和:attr:`effort_limit_sim`的属性是相等的。
        我们建议使用:attr:`effort_limit_sim`属性，
    """

    velocity_limit: dict[str, float] | float | None = None
    """Velocity limit of the joints in the group. Defaults to None.

    This limit is used by the actuator model. If None, the limit is set to the value specified
    in the USD joint prim.

    .. attention::

        The :attr:`velocity_limit_sim` attribute should be used to set the velocity limit for
        the simulation physics solver.

        The :attr:`velocity_limit` attribute is used for clipping the effort output of the
        actuator model **only** in the case of explicit actuators, such as the
        :class:`~isaaclab.actuators.IdealPDActuator`.

    .. note::

        For implicit actuators, the attribute :attr:`velocity_limit` is not used. This is to stay
        backwards compatible with previous versions of the Isaac Lab, where this parameter was
        unused since PhysX did not support setting the velocity limit for the joints using the
        PhysX Tensor API.
    """
    """组的关节的速度限制。
    默认为 None。

    这一极限是动机模型使用的。
    如果是None，则设定为USD联合prim中指定的值。

    .. 注意::

        The :应使用 attr:`velocity_limit_sim`属性来设置速度限制
        仿真物理解决器。

        The :attr:`velocity_limit`属性用于裁剪输出力量的
        执行器模型 **仅**在:class:`~isaaclab.actuators.IdealPDActuator`等明确执行器的情况下。

    .. 说明::

        对于隐含执行器，不使用:attr:`velocity_limit`属性。
        这将与以往版本的艾萨克实验室保持反向兼容性，因为PhysX不支持使用PhysX Tensor API的关节设置速度限制。
    """

    effort_limit_sim: dict[str, float] | float | None = None
    """Effort limit of the joints in the group applied to the simulation physics solver. Defaults to None.

    The effort limit is used to constrain the computed joint efforts in the physics engine. If the
    computed effort exceeds this limit, the physics engine will clip the effort to this value.

    Since explicit actuators (e.g. DC motor), compute and clip the effort in the actuator model, this
    limit is by default set to a large value to prevent the physics engine from any additional clipping.
    However, at times, it may be necessary to set this limit to a smaller value as a safety measure.

    If None, the limit is resolved based on the type of actuator model:

    * For implicit actuators, the limit is set to the value specified in the USD joint prim.
    * For explicit actuators, the limit is set to 1.0e9.

    """
    """组中的关节应对仿真物理溶剂的应力限制。
    默认为 None。

    在物理引擎中的计算联合努力被限制。
    如果计算的功率超过这个限度，物理引擎将把功率缩小到这个值。

    由于明确的执行器 (e.g。 DC电机)，计算和裁剪执行器模型中的努力，因此这个限制默认设置为高值，以防止物理引擎进一步裁剪。
    然而，有时，作为安全措施，可能需要将此限值设置为较小的值。

    如果None，则根据动机模型的类型确定限度:

    * 对于隐含动力器，限制设置为USD关联prim中指定的值。
    * 对于明确的动机，限值设置为1.0e9。
    """

    velocity_limit_sim: dict[str, float] | float | None = None
    """Velocity limit of the joints in the group applied to the simulation physics solver. Defaults to None.

    The velocity limit is used to constrain the joint velocities in the physics engine. The joint will only
    be able to reach this velocity if the joint's effort limit is sufficiently large. If the joint is moving
    faster than this velocity, the physics engine will actually try to brake the joint to reach this velocity.

    If None, the limit is set to the value specified in the USD joint prim for both implicit and explicit actuators.

    .. tip::
        If the velocity limit is too tight, the physics engine may have trouble converging to a solution.
        In such cases, we recommend either keeping this value sufficiently large or tuning the stiffness and
        damping parameters of the joint to ensure the limits are not violated.

    """
    """在仿真物理解决器上应用组中的关节的速度限制。
    默认为 None。

    速度限制用于限制物理引擎中的关节速度。
    关节只能达到这个速度，如果关节的力度极限足够大。
    如果关节速度超过这个速度，物理引擎实际上会试图制关节以达到这个速度。

    如果是None，则限制为USD关联prim中既隐含又明确的执行器所指定的值。

    .. 提示::
        如果速度限制太紧，物理引擎可能会遇到难以接近解决方案。
        在这种情况下，我们建议要么保持这个值足够大，要么调整关节的硬度和缩参数，以确保没有违反限制。
    """

    stiffness: dict[str, float] | float | None = MISSING
    """Stiffness gains (also known as p-gain) of the joints in the group.

    The behavior of the stiffness is different for implicit and explicit actuators. For implicit actuators,
    the stiffness gets set into the physics engine directly. For explicit actuators, the stiffness is used
    by the actuator model to compute the joint efforts.

    If None, the stiffness is set to the value from the USD joint prim.
    """
    """组的关节的硬度增长 (也称为p-gain)。

    隐含和明确的执行器的硬度行为不同。
    对于隐含动力， 度直接进入物理引擎。
    对于明确执行器，执行器模型使用度来计算联合努力。

    如果 None，硬度设置为 USD 关节 prim 的值。
    """

    damping: dict[str, float] | float | None = MISSING
    """Damping gains (also known as d-gain) of the joints in the group.

    The behavior of the damping is different for implicit and explicit actuators. For implicit actuators,
    the damping gets set into the physics engine directly. For explicit actuators, the damping gain is used
    by the actuator model to compute the joint efforts.

    If None, the damping is set to the value from the USD joint prim.
    """
    """集团中关节的缩增长 (也称为d-gain)。

    隐含和明确的执行器的压缩行为不同。
    对于隐含动机，压缩将直接安装在物理引擎中。
    对于明确的执行器，执行器模型使用缩增长来计算联合努力。

    如果 None，缩量设置为USD关节 prim的值。
    """

    armature: dict[str, float] | float | None = None
    """Armature of the joints in the group. Defaults to None.

    The armature is directly added to the corresponding joint-space inertia. It helps improve the
    simulation stability by reducing the joint velocities.

    It is a physics engine solver parameter that gets set into the simulation.

    If None, the armature is set to the value from the USD joint prim.
    """
    """团队的关节的装备。
    默认为 None。

    机器直接加入相应的关节空间惯性。
    通过减少关节速度，它有助于提高仿真稳定性。

    它是物理引擎解析器参数，

    如果 None， armature 设置为 USD 关节 prim 的值。
    """

    friction: dict[str, float] | float | None = None
    r"""The static friction coefficient of the joints in the group. Defaults to None.

    The joint static friction is a unitless quantity. It relates the magnitude of the spatial force transmitted
    from the parent body to the child body to the maximal static friction force that may be applied by the solver
    to resist the joint motion.

    Mathematically, this means that: :math:`F_{resist} \leq \mu F_{spatial}`, where :math:`F_{resist}`
    is the resisting force applied by the solver and :math:`F_{spatial}` is the spatial force
    transmitted from the parent body to the child body. The simulated static friction effect is therefore
    similar to static and Coulomb static friction.

    If None, the joint static friction is set to the value from the USD joint prim.

    Note: In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
    it is modeled as an effort (torque or force).
    """
    """组中关节的静态摩擦系数。
    默认为 None。

    关节静态摩擦是无单位的数量。
    它与传输的空间力量的大小有关
    from the parent body to the child body to the maximal static friction force that may be applied by the solver
    为了抵制联合运动。

    数学上，这意味着:`F_{resist} \leq \mu
    F_{spatial}`在哪里:`F_{resist}`是解决器所应用的阻力和:`F_{spatial}`是从父母身体传输到儿童身体的空间力。
    因此，仿真的静态摩擦效应类似于静态和库伦布静态摩擦。

    如果 None，关节静态摩擦设置为 USD关节 prim的值。

    Note: 在Isaac Sim 4.5中，这个参数是作为一个系数的模型。
          在Isaac Sim 5.0及后版本中，
    它是以力力 (扭矩或力) 模型的。
    """

    dynamic_friction: dict[str, float] | float | None = None
    """The dynamic friction coefficient of the joints in the group. Defaults to None.

    Note: In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
    it is modeled as an effort (torque or force).
    """
    """组中关节的动态摩擦系数。
    默认为 None。

    Note: 在Isaac Sim 4.5中，这个参数是作为一个系数的模型。
          在Isaac Sim 5.0及后版本中，
    它是以力力 (扭矩或力) 模型的。
    """

    viscous_friction: dict[str, float] | float | None = None
    """The viscous friction coefficient of the joints in the group. Defaults to None.
    """
    """在组中关节的粘性摩擦系数。
    默认为 None。
    """
