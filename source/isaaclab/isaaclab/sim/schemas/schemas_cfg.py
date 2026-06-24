# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from pxr import PhysxSchema, UsdPhysics

from isaaclab.utils import configclass


@configclass
class ArticulationRootPropertiesCfg:
    """Properties to apply to the root of an articulation.

    See :meth:`modify_articulation_root_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """
    """适用于关节根的属性。

    See :麻:`modify_articulation_root_properties`更多信息。

    .. 说明::
        如果值为None，则不会修改。
        这很有用，如果你只想设置一个子集的属性，
    """

    articulation_enabled: bool | None = None
    """Whether to enable or disable articulation."""
    """是否启用或禁用关节。"""

    enabled_self_collisions: bool | None = None
    """Whether to enable or disable self-collisions."""
    """是否启动或禁用自动碰撞。"""

    solver_position_iteration_count: int | None = None
    """Solver position iteration counts for the body."""
    """解决器位置的反复数量对身体来说很重要。"""

    solver_velocity_iteration_count: int | None = None
    """Solver velocity iteration counts for the body."""
    """解决器速度的反复数量对机体来说很重要。"""

    sleep_threshold: float | None = None
    """Mass-normalized kinetic energy threshold below which an actor may go to sleep."""
    """按质量正常化运动能量门下，一个演员可以睡觉。"""

    stabilization_threshold: float | None = None
    """The mass-normalized kinetic energy threshold below which an articulation may participate in stabilization."""
    """按质量正常化的动力能量门以下，关节可以参与稳定。"""

    fix_root_link: bool | None = None
    """Whether to fix the root link of the articulation.

    * If set to None, the root link is not modified.
    * If the articulation already has a fixed root link, this flag will enable or disable the fixed joint.
    * If the articulation does not have a fixed root link, this flag will create a fixed joint between the world
      frame and the root link. The joint is created with the name "FixedJoint" under the articulation prim.

    .. note::
        This is a non-USD schema property. It is handled by the :meth:`modify_articulation_root_properties` function.

    """
    """是否修复关节的根链。

    * 如果设置为None，则根本链接不会被修改。
    * 如果关节已经有一个固定的根链，该标志将启动或禁用固定关节。
    * 如果关节没有固定根链，则这个标志将在世界框架和根链之间创建一个固定关节.关节在关节prim下以"FixedJoint"的名字创建。

    .. 说明::
        这是一个非USD方案属性。
        它由:meth:`modify_articulation_root_properties`函数来处理。
    """


@configclass
class RigidBodyPropertiesCfg:
    """Properties to apply to a rigid body.

    See :meth:`modify_rigid_body_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """
    """适用于硬体的特性。

    See :麻:`modify_rigid_body_properties`更多信息。

    .. 说明::
        如果值为None，则不会修改。
        这很有用，如果你只想设置一个子集的属性，
    """

    rigid_body_enabled: bool | None = None
    """Whether to enable or disable the rigid body."""
    """是否启动或禁用硬体。"""

    kinematic_enabled: bool | None = None
    """Determines whether the body is kinematic or not.

    A kinematic body is a body that is moved through animated poses or through user defined poses. The simulation
    still derives velocities for the kinematic body based on the external motion.

    For more information on kinematic bodies, please refer to the `documentation <https://openusd.org/release/wp_rigid_body_physics.html#kinematic-bodies>`_.
    """
    """确定身体是否动态。

    动态身体是通过动画姿势或用户定义的姿势移动的身体。
    仿真仍然基于外部运动来推出运动体的速度。

    更多关于动态体的信息请参阅`documentation
    <https://openusd.org/release/wp_rigid_body_physics.html#kinematic-bodies>`_。
    """

    disable_gravity: bool | None = None
    """Disable gravity for the actor."""
    """关闭演员的重力。"""

    linear_damping: float | None = None
    """Linear damping for the body."""
    """身体的线性缩。"""

    angular_damping: float | None = None
    """Angular damping for the body."""
    """身体的角压力。"""

    max_linear_velocity: float | None = None
    """Maximum linear velocity for rigid bodies (in m/s)."""
    """硬体的最大线性速度 (m/s)。"""

    max_angular_velocity: float | None = None
    """Maximum angular velocity for rigid bodies (in deg/s)."""
    """硬体的最大角速度 (deg/s)。"""

    max_depenetration_velocity: float | None = None
    """Maximum depenetration velocity permitted to be introduced by the solver (in m/s)."""
    """溶剂允许引入的最大 dep透速度 (以m/s)。"""

    max_contact_impulse: float | None = None
    """The limit on the impulse that may be applied at a contact."""
    """在接触时可以施加的冲动的限制。"""

    enable_gyroscopic_forces: bool | None = None
    """Enables computation of gyroscopic forces on the rigid body."""
    """能够计算硬体上的陀螺力。"""

    retain_accelerations: bool | None = None
    """Carries over forces/accelerations over sub-steps."""
    """通过子步骤传递力量/加速。"""

    solver_position_iteration_count: int | None = None
    """Solver position iteration counts for the body."""
    """解决器位置的反复数量对身体来说很重要。"""

    solver_velocity_iteration_count: int | None = None
    """Solver position iteration counts for the body."""
    """解决器位置的反复数量对身体来说很重要。"""

    sleep_threshold: float | None = None
    """Mass-normalized kinetic energy threshold below which an actor may go to sleep."""
    """按质量正常化运动能量门下，一个演员可以睡觉。"""

    stabilization_threshold: float | None = None
    """The mass-normalized kinetic energy threshold below which an actor may participate in stabilization."""
    """按质量正常化的运动能量门以下，一个参与者可以参与稳定。"""


@configclass
class CollisionPropertiesCfg:
    """Properties to apply to colliders in a rigid body.

    See :meth:`modify_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """
    """适用于硬体的碰撞机的特性。

    See :麻:`modify_collision_properties`更多信息。

    .. 说明::
        如果值为None，则不会修改。
        这很有用，如果你只想设置一个子集的属性，
    """

    collision_enabled: bool | None = None
    """Whether to enable or disable collisions."""
    """不管是启动或禁用碰撞。"""

    contact_offset: float | None = None
    """Contact offset for the collision shape (in m).

    The collision detector generates contact points as soon as two shapes get closer than the sum of their
    contact offsets. This quantity should be non-negative which means that contact generation can potentially start
    before the shapes actually penetrate.
    """
    """碰撞形状的接触偏移 (m)。

    碰撞探测器只要两个形状比它们的接触相对相近，就会产生接触点。
    这种数量应该是非负的，这意味着接触生成可能在形状实际透之前开始。
    """

    rest_offset: float | None = None
    """Rest offset for the collision shape (in m).

    The rest offset quantifies how close a shape gets to others at rest, At rest, the distance between two
    vertically stacked objects is the sum of their rest offsets. If a pair of shapes have a positive rest
    offset, the shapes will be separated at rest by an air gap.
    """
    """对碰撞形状的休息偏移 (m)。

    在休息时，两种垂直堆叠的物体之间的距离是它们的休息偏移的总和。
    如果两种形状有一个正确的休息偏移，则这些形状将通过空气空隙在休息时分开。
    """

    torsional_patch_radius: float | None = None
    """Radius of the contact patch for applying torsional friction (in m).

    It is used to approximate rotational friction introduced by the compression of contacting surfaces.
    If the radius is zero, no torsional friction is applied.
    """
    """接触贴子用于扭矩摩擦的半径 (m)。

    它用于通过接触表面的压缩来 приблизи旋转摩擦。
    如果半径为零，则不会使用扭曲摩擦。
    """

    min_torsional_patch_radius: float | None = None
    """Minimum radius of the contact patch for applying torsional friction (in m)."""
    """应用于扭曲摩擦的接触贴的最小半径 (m)。"""


@configclass
class MassPropertiesCfg:
    """Properties to define explicit mass properties of a rigid body.

    See :meth:`modify_mass_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """
    """定义硬体的明确质量特性的特性。

    See :麻:`modify_mass_properties`更多信息。

    .. 说明::
        如果值为None，则不会修改。
        这很有用，如果你只想设置一个子集的属性，
    """

    mass: float | None = None
    """The mass of the rigid body (in kg).

    Note:
        If non-zero, the mass is ignored and the density is used to compute the mass.
    """
    """硬体质量 (公斤)。

    说明：
        如果不为零，则质量被忽视，密度被用来计算质量。
    """

    density: float | None = None
    """The density of the rigid body (in kg/m^3).

    The density indirectly defines the mass of the rigid body. It is generally computed using the collision
    approximation of the body.
    """
    """硬体密度 (kg/m^3)。

    密度间接定义了硬体质量。
    它通常是使用身体的碰撞近距离计算的。
    """


@configclass
class JointDrivePropertiesCfg:
    """Properties to define the drive mechanism of a joint.

    See :meth:`modify_joint_drive_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """
    """定义关节的驱动机制的特性。

    See :麻:`modify_joint_drive_properties`更多信息。

    .. 说明::
        如果值为None，则不会修改。
        这很有用，如果你只想设置一个子集的属性，
    """

    drive_type: Literal["force", "acceleration"] | None = None
    """Joint drive type to apply.

    If the drive type is "force", then the joint is driven by a force. If the drive type is "acceleration",
    then the joint is driven by an acceleration (usually used for kinematic joints).
    """
    """应使用的联合驱动型。

    如果驱动型是"力"，则关节由力驱动。
    如果驱动型是"加速"，则关节由加速驱动 (通常用于动态关节)。
    """

    max_effort: float | None = None
    """Maximum effort that can be applied to the joint (in kg-m^2/s^2)."""
    """对关节可施加的最大力 (kg-m^2/s^2)。"""

    max_velocity: float | None = None
    """Maximum velocity of the joint.

    The unit depends on the joint model:

    * For linear joints, the unit is m/s.
    * For angular joints, the unit is rad/s.
    """
    """关节的最大速度。

    单元取决于联合模型:

    * 对于线性关节，单位是m/s。
    * 对于角关节，单位是rad/s。
    """

    stiffness: float | None = None
    """Stiffness of the joint drive.

    The unit depends on the joint model:

    * For linear joints, the unit is kg-m/s^2 (N/m).
    * For angular joints, the unit is kg-m^2/s^2/rad (N-m/rad).
    """
    """关节驱动的硬度。

    单元取决于联合模型:

    * 对于线性关节，单位是kg-m/s^2 (N/m)。
    * 对于角关节，单位是kg-m^2/s^2/rad (N-m/rad)。
    """

    damping: float | None = None
    """Damping of the joint drive.

    The unit depends on the joint model:

    * For linear joints, the unit is kg-m/s (N-s/m).
    * For angular joints, the unit is kg-m^2/s/rad (N-m-s/rad).
    """
    """joint湿的联合驱动。

    单元取决于联合模型:

    * 对于线性关节，单位是kg-m/s (N-s/m)。
    * 对于角关节，单位是kg-m^2/s/rad (N-m-s/rad)。
    """


@configclass
class FixedTendonPropertiesCfg:
    """Properties to define fixed tendons of an articulation.

    See :meth:`modify_fixed_tendon_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """
    """定义关节的固定节的特性。

    See :麻:`modify_fixed_tendon_properties`更多信息。

    .. 说明::
        如果值为None，则不会修改。
        这很有用，如果你只想设置一个子集的属性，
    """

    tendon_enabled: bool | None = None
    """Whether to enable or disable the tendon."""
    """是否启动或禁用。"""

    stiffness: float | None = None
    """Spring stiffness term acting on the tendon's length."""
    """Spring节长度影响的春节硬度项。"""

    damping: float | None = None
    """The damping term acting on both the tendon length and the tendon-length limits."""
    """适用于门长度和 limits门长度限制。"""

    limit_stiffness: float | None = None
    """Limit stiffness term acting on the tendon's length limits."""
    """限制硬度，影响长度限制。"""

    offset: float | None = None
    """Length offset term for the tendon.

    It defines an amount to be added to the accumulated length computed for the tendon. This allows the application
    to actuate the tendon by shortening or lengthening it.
    """
    """子的长度抵消项。

    它定义了应加到um子计算的积累长度的数量。
    这使应用程序可以通过缩短或延长部来激活部。
    """

    rest_length: float | None = None
    """Spring rest length of the tendon."""
    """节的春节休息长度。"""


@configclass
class SpatialTendonPropertiesCfg:
    """Properties to define spatial tendons of an articulation.

    See :meth:`modify_spatial_tendon_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """
    """定义关节的空间的特性。

    See :麻:`modify_spatial_tendon_properties`更多信息。

    .. 说明::
        如果值为None，则不会修改。
        这很有用，如果你只想设置一个子集的属性，
    """

    tendon_enabled: bool | None = None
    """Whether to enable or disable the tendon."""
    """是否启动或禁用。"""

    stiffness: float | None = None
    """Spring stiffness term acting on the tendon's length."""
    """Spring节长度影响的春节硬度项。"""

    damping: float | None = None
    """The damping term acting on both the tendon length and the tendon-length limits."""
    """适用于门长度和 limits门长度限制。"""

    limit_stiffness: float | None = None
    """Limit stiffness term acting on the tendon's length limits."""
    """限制硬度，影响长度限制。"""

    offset: float | None = None
    """Length offset term for the tendon.

    It defines an amount to be added to the accumulated length computed for the tendon. This allows the application
    to actuate the tendon by shortening or lengthening it.
    """
    """子的长度抵消项。

    它定义了应加到um子计算的积累长度的数量。
    这使应用程序可以通过缩短或延长部来激活部。
    """


@configclass
class DeformableBodyPropertiesCfg:
    """Properties to apply to a deformable body.

    A deformable body is a body that can deform under forces. The configuration allows users to specify
    the properties of the deformable body, such as the solver iteration counts, damping, and self-collision.

    An FEM-based deformable body is created by providing a collision mesh and simulation mesh. The collision mesh
    is used for collision detection and the simulation mesh is used for simulation. The collision mesh is usually
    a simplified version of the simulation mesh.

    Based on the above, the PhysX team provides APIs to either set the simulation and collision mesh directly
    (by specifying the points) or to simplify the collision mesh based on the simulation mesh. The simplification
    process involves remeshing the collision mesh and simplifying it based on the target triangle count.

    Since specifying the collision mesh points directly is not a common use case, we only expose the parameters
    to simplify the collision mesh based on the simulation mesh. If you want to provide the collision mesh points,
    please open an issue on the repository and we can add support for it.

    See :meth:`modify_deformable_body_properties` for more information.

    .. note::
        If the values are :obj:`None`, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """
    """适用于可变形体的特性。

    一个可变化的身体是可以被强势变形的身体。
    配置允许用户指定可变体的特性，例如溶剂代数，缩和自撞。

    通过提供碰撞网和仿真网来创建基于FEM的可变体。
    碰撞网用于碰撞检测，仿真网用于仿真。
    碰撞网通常是仿真网的简化版本。

    基于上述情况，PhysX团队提供APIs来直接设置仿真和碰撞网 (通过指定点) 或简化基于仿真网的碰撞网。
    简化过程包括重新结碰撞网，并根据目标三角形数量简化。

    由于直接指定碰撞网点并不是常见的应用案例，我们只将参数暴露在仿真网的基础上来简化碰撞网。
    如果您想提供碰撞网点，请在存储库上打开一个问题，我们可以添加支持。

    See :麻:`modify_deformable_body_properties`更多信息。

    .. 说明::
        如果值为:obj:`None`，则不会修改。
        这很有用，如果你只想设置一个子集的属性，
    """

    deformable_enabled: bool | None = None
    """Enables deformable body."""
    """能够使身体变形。"""

    kinematic_enabled: bool = False
    """Enables kinematic body. Defaults to False, which means that the body is not kinematic.

    Similar to rigid bodies, this allows setting user-driven motion for the deformable body. For more information,
    please refer to the `documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html#kinematic-soft-bodies>`__.
    """
    """启动动动态身体。
    默认为 False，这意味着身体不动态。

    类似于硬体，这可以为可变体设置用户驱动的运动。
    更多信息请参阅`documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html#kin
    ematic-soft-bodies>`__。
    """

    self_collision: bool | None = None
    """Whether to enable or disable self-collisions for the deformable body based on the rest position distances."""
    """根据休息位置距离，是否能够或不能根据可变体进行自撞。"""

    self_collision_filter_distance: float | None = None
    """Penetration value that needs to get exceeded before contacts for self collision are generated.

    This parameter must be greater than of equal to twice the :attr:`rest_offset` value.

    This value has an effect only if :attr:`self_collision` is enabled.
    """
    """在产生自动碰撞的接触之前需要超越的透值。

    这一参数必须超过:attr:`rest_offset`值的两倍。

    如果启用:attr:`self_collision`，该值只有效果。
    """

    settling_threshold: float | None = None
    """Threshold vertex velocity (in m/s) under which sleep damping is applied in addition to velocity damping."""
    """极顶速度 (以m/s) 在睡眠缓解除了速度缓解之外，应用于睡眠缓解。"""

    sleep_damping: float | None = None
    """Coefficient for the additional damping term if fertex velocity drops below setting threshold."""
    """如果肥料速度低于设定门，则额外缩项的系数。"""

    sleep_threshold: float | None = None
    """The velocity threshold (in m/s) under which the vertex becomes a candidate for sleeping in the next step."""
    """速度门 (以m/s) 在下一步的顶点成为睡眠候选人。"""

    solver_position_iteration_count: int | None = None
    """Number of the solver positional iterations per step. Range is [1,255]"""
    """每一步的解决器位置反转数量。
    范围为 [1，255]
    """

    vertex_velocity_damping: float | None = None
    """Coefficient for artificial damping on the vertex velocity.

    This parameter can be used to approximate the effect of air drag on the deformable body.
    """
    """在顶点速度上的人工缩系数。

    该参数可用于近似对可变体的气阻的影响。
    """

    simulation_hexahedral_resolution: int = 10
    """The target resolution for the hexahedral mesh used for simulation. Defaults to 10.

    Note:
        This value is ignored if the user provides the simulation mesh points directly. However, we assume that
        most users will not provide the simulation mesh points directly. If you want to provide the simulation mesh
        directly, please set this value to :obj:`None`.
    """
    """为仿真所使用的六网的目标分辨率。
    默认为10

    说明：
        如果用户直接提供仿真网格点，则会忽略此值。
        然而，我们认为大多数用户不会直接提供仿真网点。
        如果您想直接提供仿真网格，请设置这个值为:obj:`None`。
    """

    collision_simplification: bool = True
    """Whether or not to simplify the collision mesh before creating a soft body out of it. Defaults to True.

    Note:
        This flag is ignored if the user provides the simulation mesh points directly. However, we assume that
        most users will not provide the simulation mesh points directly. Hence, this flag is enabled by default.

        If you want to provide the simulation mesh points directly, please set this flag to False.
    """
    """在从中创建一个软体之前，是否要简化碰撞网。
    默认为 True。

    说明：
        如果用户直接提供仿真网点，则将忽略此标志。
        然而，我们认为大多数用户不会直接提供仿真网点。
        因此，这个旗默认启用。

        如果您想直接提供仿真网点，请设置这个标志为False。
    """

    collision_simplification_remeshing: bool = True
    """Whether or not the collision mesh should be remeshed before simplification. Defaults to True.

    This parameter is ignored if :attr:`collision_simplification` is False.
    """
    """在简化之前，是否应该重新结碰撞网。
    默认为 True。

    如果:attr:`collision_simplification`是False，则会忽略这个参数。
    """

    collision_simplification_remeshing_resolution: int = 0
    """The resolution used for remeshing. Defaults to 0, which means that a heuristic is used to determine the
    resolution.

    This parameter is ignored if :attr:`collision_simplification_remeshing` is False.
    """
    """用于复合的分辨率。
    默认为0，这意味着用于确定分辨率的数。

    如果:attr:`collision_simplification_remeshing`是False，则会忽略这个参数。
    """

    collision_simplification_target_triangle_count: int = 0
    """The target triangle count used for the simplification. Defaults to 0, which means that a heuristic based on
    the :attr:`simulation_hexahedral_resolution` is used to determine the target count.

    This parameter is ignored if :attr:`collision_simplification` is False.
    """
    """为简化使用的目标三角形数。
    默认为0，这意味着基于
    the :attr:`simulation_hexahedral_resolution`用于确定目标数量。

    如果:attr:`collision_simplification`是False，则会忽略这个参数。
    """

    collision_simplification_force_conforming: bool = True
    """Whether or not the simplification should force the output mesh to conform to the input mesh. Defaults to True.

    The flag indicates that the tretrahedralizer used to generate the collision mesh should produce tetrahedra
    that conform to the triangle mesh. If False, the simplifier uses the output from the tretrahedralizer used.

    This parameter is ignored if :attr:`collision_simplification` is False.
    """
    """简化是否使输出网格与输入网格相符。
    默认为 True。

    标志表明，用于产生碰撞网的三角形缩水器应该产生符合三角形网的四角形。
    如果 False，简化器使用使用的三角制解器的输出。

    如果:attr:`collision_simplification`是False，则会忽略这个参数。
    """

    contact_offset: float | None = None
    """Contact offset for the collision shape (in m).

    The collision detector generates contact points as soon as two shapes get closer than the sum of their
    contact offsets. This quantity should be non-negative which means that contact generation can potentially start
    before the shapes actually penetrate.
    """
    """碰撞形状的接触偏移 (m)。

    碰撞探测器只要两个形状比它们的接触相对相近，就会产生接触点。
    这种数量应该是非负的，这意味着接触生成可能在形状实际透之前开始。
    """

    rest_offset: float | None = None
    """Rest offset for the collision shape (in m).

    The rest offset quantifies how close a shape gets to others at rest, At rest, the distance between two
    vertically stacked objects is the sum of their rest offsets. If a pair of shapes have a positive rest
    offset, the shapes will be separated at rest by an air gap.
    """
    """对碰撞形状的休息偏移 (m)。

    在休息时，两种垂直堆叠的物体之间的距离是它们的休息偏移的总和。
    如果两种形状有一个正确的休息偏移，则这些形状将通过空气空隙在休息时分开。
    """

    max_depenetration_velocity: float | None = None
    """Maximum depenetration velocity permitted to be introduced by the solver (in m/s)."""
    """溶剂允许引入的最大 dep透速度 (以m/s)。"""


@configclass
class MeshCollisionPropertiesCfg:
    """Properties to apply to a mesh in regards to collision.
    See :meth:`set_mesh_collision_properties` for more information.

    .. note::
        If the values are MISSING, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """
    """适用于碰撞的网格的特性。
    See :麻:`set_mesh_collision_properties`更多信息。

    .. 说明::
        如果值为MISSING，则不会修改。
        这很有用，如果你只想设置一个子集的属性，
    """

    usd_func: callable = MISSING
    """USD API function for modifying mesh collision properties.
    Refer to
    `original USD Documentation <https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_usd_physics_mesh_collision_a_p_i.html>`_
    for more information.
    """
    """修改网格碰撞性能的USD API函数。
    参考`original USD Documentation <https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/la
    test/class_usd_physics_mesh_collision_a_p_i.html>`_
    for more information.
    """

    physx_func: callable = MISSING
    """PhysX API function for modifying mesh collision properties.
    Refer to
    `original PhysX Documentation <https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/annotated.html>`_
    for more information.
    """
    """修改网格碰撞性能的PhysX API函数。
    参考`original PhysX Documentation
    <https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/annotated.html>`_
    for more information.
    """

    mesh_approximation_name: str = "none"
    """Name of mesh collision approximation method. Default: "none".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """
    """网格碰撞接近方法名称
    默认:没有。
    查看:const:`schemas.MESH_APPROXIMATION_TOKENS`可用的选项。
    """


@configclass
class BoundingCubePropertiesCfg(MeshCollisionPropertiesCfg):
    usd_func: callable = UsdPhysics.MeshCollisionAPI
    """Original USD Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_usd_physics_mesh_collision_a_p_i.html
    """
    """原始USD文件:
    https://docs.omniverse.nvidia.com其他类型omni_usd_schema_physics最后一次class_usd_physics_mesh_collision_a_p
          _i.html
    """

    mesh_approximation_name: str = "boundingCube"
    """Name of mesh collision approximation method. Default: "boundingCube".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """
    """网格碰撞接近方法名称
    默认情况下: "bo关Cube"。
    查看:const:`schemas.MESH_APPROXIMATION_TOKENS`可用的选项。
    """


@configclass
class BoundingSpherePropertiesCfg(MeshCollisionPropertiesCfg):
    usd_func: callable = UsdPhysics.MeshCollisionAPI
    """Original USD Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_usd_physics_mesh_collision_a_p_i.html
    """
    """原始USD文件:
    https://docs.omniverse.nvidia.com其他类型omni_usd_schema_physics最后一次class_usd_physics_mesh_collision_a_p
          _i.html
    """

    mesh_approximation_name: str = "boundingSphere"
    """Name of mesh collision approximation method. Default: "boundingSphere".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """
    """网格碰撞接近方法名称
    默认情况下: " boundingSphere"。
    查看:const:`schemas.MESH_APPROXIMATION_TOKENS`可用的选项。
    """


@configclass
class ConvexDecompositionPropertiesCfg(MeshCollisionPropertiesCfg):
    usd_func: callable = UsdPhysics.MeshCollisionAPI
    """Original USD Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_usd_physics_mesh_collision_a_p_i.html
    """
    """原始USD文件:
    https://docs.omniverse.nvidia.com其他类型omni_usd_schema_physics最后一次class_usd_physics_mesh_collision_a_p
          _i.html
    """

    physx_func: callable = PhysxSchema.PhysxConvexDecompositionCollisionAPI
    """Original PhysX Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_convex_decomposition_collision_a_p_i.html
    """
    """原始PhysX文件:
    https://docs.omniverse.nvidia.com其他类型omni_usd_schema_physics最后一次class_physx_schema_physx_convex_deco
          mposition_collision_a_p_i.html
    """

    mesh_approximation_name: str = "convexDecomposition"
    """Name of mesh collision approximation method. Default: "convexDecomposition".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """
    """网格碰撞接近方法名称
    默认: "曲解"。
    查看:const:`schemas.MESH_APPROXIMATION_TOKENS`可用的选项。
    """

    hull_vertex_limit: int | None = None
    """Convex hull vertex limit used for convex hull cooking.

    Defaults to 64.
    """
    """limit形顶限用于形体。

    默认到64
    """
    max_convex_hulls: int | None = None
    """Maximum of convex hulls created during convex decomposition.
    Default value is 32.
    """
    """在曲分解过程中形成的imum曲体最大。
    默认值为32
    """
    min_thickness: float | None = None
    """Convex hull min thickness.

    Range: [0, inf). Units are distance. Default value is 0.001.
    """
    """形体的厚度

    Range: [0，inf)。
           单位是距离。
           默认值为0.001。
    """
    voxel_resolution: int | None = None
    """Voxel resolution used for convex decomposition.

    Defaults to 500,000 voxels.
    """
    """用于曲分解的伏素分辨率。

    默认情况下500，000个语音。
    """
    error_percentage: float | None = None
    """Convex decomposition error percentage parameter.

    Defaults to 10 percent. Units are percent.
    """
    """曲分解错误百分比参数

    默认到10%
    单位是百分比。
    """
    shrink_wrap: bool | None = None
    """Attempts to adjust the convex hull points so that they are projected onto the surface of the original graphics
    mesh.

    Defaults to False.
    """
    """试图调整凸的船体点，以便它们投射到原始图形网的表面。

    默认为 False。
    """


@configclass
class ConvexHullPropertiesCfg(MeshCollisionPropertiesCfg):
    usd_func: callable = UsdPhysics.MeshCollisionAPI
    """Original USD Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_usd_physics_mesh_collision_a_p_i.html
    """
    """原始USD文件:
    https://docs.omniverse.nvidia.com其他类型omni_usd_schema_physics最后一次class_usd_physics_mesh_collision_a_p
          _i.html
    """

    physx_func: callable = PhysxSchema.PhysxConvexHullCollisionAPI
    """Original PhysX Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_convex_hull_collision_a_p_i.html
    """
    """原始PhysX文件:
    https://docs.omniverse.nvidia.com其他类型omni_usd_schema_physics最后一次class_physx_schema_physx_convex_hull
          _collision_a_p_i.html
    """

    mesh_approximation_name: str = "convexHull"
    """Name of mesh collision approximation method. Default: "convexHull".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """
    """网格碰撞接近方法名称
    默认: "形"。
    查看:const:`schemas.MESH_APPROXIMATION_TOKENS`可用的选项。
    """

    hull_vertex_limit: int | None = None
    """Convex hull vertex limit used for convex hull cooking.

    Defaults to 64.
    """
    """limit形顶限用于形体。

    默认到64
    """
    min_thickness: float | None = None
    """Convex hull min thickness.

    Range: [0, inf). Units are distance. Default value is 0.001.
    """
    """形体的厚度

    Range: [0，inf)。
           单位是距离。
           默认值为0.001。
    """


@configclass
class TriangleMeshPropertiesCfg(MeshCollisionPropertiesCfg):
    physx_func: callable = PhysxSchema.PhysxTriangleMeshCollisionAPI
    """Triangle mesh is only supported by PhysX API.

    Original PhysX Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_triangle_mesh_collision_a_p_i.html
    """
    """只有PhysXAPI支持三角形网格。

    原始PhysX文件:
    https://docs.omniverse.nvidia.com其他类型omni_usd_schema_physics最后一次class_physx_schema_physx_triangle_me
          sh_collision_a_p_i.html
    """

    mesh_approximation_name: str = "none"
    """Name of mesh collision approximation method. Default: "none" (uses triangle mesh).
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """
    """网格碰撞接近方法名称
    默认: "没有" (使用三角网)。
    查看:const:`schemas.MESH_APPROXIMATION_TOKENS`可用的选项。
    """

    weld_tolerance: float | None = None
    """Mesh weld tolerance, controls the distance at which vertices are welded.

    Default -inf will autocompute the welding tolerance based on the mesh size. Zero value will disable welding.
    Range: [0, inf) Units: distance
    """
    """接宽度，控制顶点 distance接的距离。

    根据网格大小，默认 -inf 将自动计算接耐受性。
    零值将禁用接。
    Range: [0，inf) 单位:距离
    """


@configclass
class TriangleMeshSimplificationPropertiesCfg(MeshCollisionPropertiesCfg):
    usd_func: callable = UsdPhysics.MeshCollisionAPI
    """Original USD Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_usd_physics_mesh_collision_a_p_i.html
    """
    """原始USD文件:
    https://docs.omniverse.nvidia.com其他类型omni_usd_schema_physics最后一次class_usd_physics_mesh_collision_a_p
          _i.html
    """

    physx_func: callable = PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI
    """Original PhysX Documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_triangle_mesh_simplification_collision_a_p_i.html
    """
    """原始PhysX文件:
    https://docs.omniverse.nvidia.com其他类型omni_usd_schema_physics最后一次class_physx_schema_physx_triangle_me
          sh_simplification_collision_a_p_i.html
    """

    mesh_approximation_name: str = "meshSimplification"
    """Name of mesh collision approximation method. Default: "meshSimplification".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """
    """网格碰撞接近方法名称
    默认: "网格简化"。
    查看:const:`schemas.MESH_APPROXIMATION_TOKENS`可用的选项。
    """

    simplification_metric: float | None = None
    """Mesh simplification accuracy.

    Defaults to 0.55.
    """
    """网格简化精度

    默认为0.55。
    """
    weld_tolerance: float | None = None
    """Mesh weld tolerance, controls the distance at which vertices are welded.

    Default -inf will autocompute the welding tolerance based on the mesh size. Zero value will disable welding.
    Range: [0, inf) Units: distance
    """
    """接宽度，控制顶点 distance接的距离。

    根据网格大小，默认 -inf 将自动计算接耐受性。
    零值将禁用接。
    Range: [0，inf) 单位:距离
    """


@configclass
class SDFMeshPropertiesCfg(MeshCollisionPropertiesCfg):
    physx_func: callable = PhysxSchema.PhysxSDFMeshCollisionAPI
    """SDF mesh is only supported by PhysX API.

    Original PhysX documentation:
    https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_s_d_f_mesh_collision_a_p_i.html

    More details and steps for optimizing SDF results can be found here:
    https://nvidia-omniverse.github.io/PhysX/physx/5.2.1/docs/RigidBodyCollision.html#dynamic-triangle-meshes-with-sdfs
    """
    """只有PhysXXAPI支持SDF网格。

    原始PhysX文件:
    https://docs.omniverse.nvidia.com其他类型omni_usd_schema_physics最后一次class_physx_schema_physx_s_d_f_mesh_
          collision_a_p_i.html

    更多关于优化SDF结果的细节和步骤可以在这里找到:
    https:视频omniverse.github.io/PhysX医疗保健服务RigidBodyCollision.html#动态三角形网带sdfs
    """

    mesh_approximation_name: str = "sdf"
    """Name of mesh collision approximation method. Default: "sdf".
    Refer to :const:`schemas.MESH_APPROXIMATION_TOKENS` for available options.
    """
    """网格碰撞接近方法名称
    默认:sdf。
    查看:const:`schemas.MESH_APPROXIMATION_TOKENS`可用的选项。
    """

    sdf_margin: float | None = None
    """Margin to increase the size of the SDF relative to the bounding box diagonal length of the mesh.


    A sdf margin value of 0.01 means the sdf boundary will be enlarged in any direction by 1% of the mesh's bounding
    box diagonal length. Representing the margin relative to the bounding box diagonal length ensures that it is scale
    independent. Margins allow for precise distance queries in a region slightly outside of the mesh's bounding box.

    Default value is 0.01.
    Range: [0, inf) Units: dimensionless
    """
    """边缘以增加SDF的尺寸与网格边界框的斜面长度相比。


    一个 sdf 边界值为0.01意味着 sdf 边界将在任何方向上扩大到网格边界框对角长度的1%。
    代表边缘与边界框的横向长度相对，确保其独立于尺度。
    边缘允许在网格的边界框外略有区域进行精确的距离查询。

    默认值为0.01。
    Range: [0，inf) 单位:无维度
    """
    sdf_narrow_band_thickness: float | None = None
    """Size of the narrow band around the mesh surface where high resolution SDF samples are available.

    Outside of the narrow band, only low resolution samples are stored. Representing the narrow band thickness as a
    fraction of the mesh's bounding box diagonal length ensures that it is scale independent. A value of 0.01 is
    usually large enough. The smaller the narrow band thickness, the smaller the memory consumption of the sparse SDF.

    Default value is 0.01.
    Range: [0, 1] Units: dimensionless
    """
    """在高分辨率SDF样本可用的地方，网格表面周围的窄带的尺寸。

    在窄带外，只有低分辨率的样本才能存储。
    代表窄带厚度为网格边界框长度的微小部分，确保其独立于尺度。
    值通常是足够大的。
    窄带厚度越小，稀少SDF的内存消耗就越小。

    默认值为0.01。
    Range: [0， 1]单位:无维度
    """
    sdf_resolution: int | None = None
    """The spacing of the uniformly sampled SDF is equal to the largest AABB extent of the mesh,
    divided by the resolution.

    Choose the lowest possible resolution that provides acceptable performance; very high resolution results in large
    memory consumption, and slower cooking and simulation performance.

    Default value is 256.
    Range: (1, inf)
    """
    """均样本SDF的距离等于 largest的最大AABB范围，分为分辨率。

    选择提供可接受性能的最小分辨率；非常高分辨率导致了大量的内存消耗，以及缓慢的和仿真性能。

    默认值为256。
    Range: 1，inf)
    """
    sdf_subgrid_resolution: int | None = None
    """A positive subgrid resolution enables sparsity on signed-distance-fields (SDF) while a value of 0 leads to the
    usage of a dense SDF.

    A value in the range of 4 to 8 is a reasonable compromise between block size and the overhead introduced by block
    addressing. The smaller a block, the more memory is spent on the address table. The bigger a block, the less
    precisely the sparse SDF can adapt to the mesh's surface. In most cases sparsity reduces the memory consumption of
    a SDF significantly.

    Default value is 6.
    Range: [0, inf)
    """
    """一个正面的子网分辨率使得在签署距离领域 (SDF) 上的稀疏性能够实现，而一个值为0导致使用密集的SDF。

    区间4至8的值是区块大小和区块地址带来的总费之间的合理妥协。
    区块越小，地址表上的记忆就越多。
    一块越大，SDF就越不准确地适应网格表面。
    在大多数情况下，稀缺性显著降低SDF的记忆消耗。

    默认值为6。
    Range: [0，inf)
    """
