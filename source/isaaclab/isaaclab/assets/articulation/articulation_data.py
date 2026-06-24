# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import weakref

import torch

import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager

import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import TimestampedBuffer

# import logger
logger = logging.getLogger(__name__)


class ArticulationData:
    """Data container for an articulation.

    This class contains the data for an articulation in the simulation. The data includes the state of
    the root rigid body, the state of all the bodies in the articulation, and the joint state. The data is
    stored in the simulation world frame unless otherwise specified.

    An articulation is comprised of multiple rigid bodies or links. For a rigid body, there are two frames
    of reference that are used:

    - Actor frame: The frame of reference of the rigid body prim. This typically corresponds to the Xform prim
      with the rigid body schema.
    - Center of mass frame: The frame of reference of the center of mass of the rigid body.

    Depending on the settings, the two frames may not coincide with each other. In the robotics sense, the actor frame
    can be interpreted as the link frame.
    """
    """关节的数据容器。

    这类包含在仿真中的关节数据。
    数据包括根固体的状态，关节中的所有身体的状态和关节状态。
    除非另有说明，则数据存储在仿真世界框架中。

    一个关节由多个硬体或链接组成。
    对于硬体，使用的两个参考框架:

    - 演员框架:硬体prim的参考框架.这通常与X形式prim相符
      with the rigid body schema.
    - 质量框架的中心:是硬体质量中心的参考框架。

    根据设置，两个框架可能不匹配。
    在机器人意义上，演员框架可以被解释为链接框架。
    """

    def __init__(self, root_physx_view: physx.ArticulationView, device: str):
        """Initializes the articulation data.

        Args:
            root_physx_view: The root articulation view.
            device: The device used for processing.
        """
        """启动关节数据。

        参数：
            root_physx_view: 根关节的观点。
            device: 用于加工的装置。
        """
        # Set the parameters
        self.device = device
        # Set the root articulation view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_physx_view: physx.ArticulationView = weakref.proxy(root_physx_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        gravity = self._physics_sim_view.get_gravity()
        # Convert to direction vector
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)

        # Initialize constants
        self.GRAVITY_VEC_W = gravity_dir.repeat(self._root_physx_view.count, 1)
        self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_physx_view.count, 1)

        # Initialize history for finite differencing
        self._previous_joint_vel = self._root_physx_view.get_dof_velocities().clone()

        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._root_link_pose_w = TimestampedBuffer()
        self._root_link_vel_w = TimestampedBuffer()
        self._body_link_pose_w = TimestampedBuffer()
        self._body_link_vel_w = TimestampedBuffer()
        # -- com frame w.r.t. link frame
        self._body_com_pose_b = TimestampedBuffer()
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedBuffer()
        self._root_com_vel_w = TimestampedBuffer()
        self._body_com_pose_w = TimestampedBuffer()
        self._body_com_vel_w = TimestampedBuffer()
        self._body_com_acc_w = TimestampedBuffer()
        # -- combined state (these are cached as they concatenate)
        self._root_state_w = TimestampedBuffer()
        self._root_link_state_w = TimestampedBuffer()
        self._root_com_state_w = TimestampedBuffer()
        self._body_state_w = TimestampedBuffer()
        self._body_link_state_w = TimestampedBuffer()
        self._body_com_state_w = TimestampedBuffer()
        # -- joint state
        self._joint_pos = TimestampedBuffer()
        self._joint_vel = TimestampedBuffer()
        self._joint_acc = TimestampedBuffer()
        self._body_incoming_joint_wrench_b = TimestampedBuffer()

    def update(self, dt: float):
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the joint acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.joint_acc

    ##
    # Names.
    ##

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""
    """在仿真视图中分析的顺序中，"""

    joint_names: list[str] = None
    """Joint names in the order parsed by the simulation view."""
    """在仿真视图中解析的顺序中，"""

    fixed_tendon_names: list[str] = None
    """Fixed tendon names in the order parsed by the simulation view."""
    """按仿真视图分析的顺序确定位名字。"""

    spatial_tendon_names: list[str] = None
    """Spatial tendon names in the order parsed by the simulation view."""
    """在仿真视图中解析的顺序中，空间位名称。"""

    ##
    # Defaults - Initial state.
    ##

    default_root_state: torch.Tensor = None
    """Default root state ``[pos, quat, lin_vel, ang_vel]`` in the local environment frame.
    Shape is (num_instances, 13).

    The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
    velocities are of its center of mass frame.

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """
    """在本地环境框架中的默认根状态``[pos， quat， lin_vel， ang_vel]``。
    形状是 (num_instances， 13)。

    位置和四元数是关节根的演员框架。
    与此同时，线性和角度速度是它的质量框架中心。

    这种数量通过:attr:`isaaclab.assets.ArticulationCfg.init_state`参数进行配置。
    """

    default_joint_pos: torch.Tensor = None
    """Default joint positions of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """
    """所有关节的默认关节位置。
    形状是 (num_instances，num_joints)。

    这种数量通过:attr:`isaaclab.assets.ArticulationCfg.init_state`参数进行配置。
    """

    default_joint_vel: torch.Tensor = None
    """Default joint velocities of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """
    """所有关节的默认关节速度。
    形状是 (num_instances，num_joints)。

    这种数量通过:attr:`isaaclab.assets.ArticulationCfg.init_state`参数进行配置。
    """

    ##
    # Defaults - Physical properties.
    ##

    default_mass: torch.Tensor = None
    """Default mass for all the bodies in the articulation. Shape is (num_instances, num_bodies).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """在关节中所有身体的默认质量。
    形状是 (num_instances，num_bodies)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_inertia: torch.Tensor = None
    """Default inertia for all the bodies in the articulation. Shape is (num_instances, num_bodies, 9).

    The inertia tensor should be given with respect to the center of mass, expressed in the articulation links'
    actor frame. The values are stored in the order
    :math:`[I_{xx}, I_{yx}, I_{zx}, I_{xy}, I_{yy}, I_{zy}, I_{xz}, I_{yz}, I_{zz}]`. However, due to the
    symmetry of inertia tensors, row- and column-major orders are equivalent.

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """在关节中的所有体体的默认惯性。
    形状是 (num_instances，num_bodies， 9)。

    按 mass结链的演员框架表达的质量中心应给出惯性子。
    值按顺序存储
    :math:`[I_{xx}， I_{yx}， I_{zx}， I_{xy}， I_{yy}， I_{zy}， I_{xz}， I_{yz}， I_{zz}]`然而，由于
    惰性子，排列和列大序列的对称性等等。

    在初始化时，这个数量从USD方案中解析。
    """

    default_joint_stiffness: torch.Tensor = None
    """Default joint stiffness of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.stiffness`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.

    .. attention::
        The default stiffness is the value configured by the user or the value parsed from the USD schema.
        It should not be confused with :attr:`joint_stiffness`, which is the value set into the simulation.
    """
    """所有关节的默认关节硬度。
    形状是 (num_instances，num_joints)。

    这种数量通过执行器模型的:attr:`isaaclab.actuators.ActuatorBaseCfg.stiffness`参数进行配置。
    如果参数的值是None，则在初始化时使用从USD方案解析的值。

    .. 注意::
        默认硬度是用户配置的值或从USD方案中解析的值。
        它不应与:attr:`joint_stiffness`混，这是仿真中设置的值。
    """

    default_joint_damping: torch.Tensor = None
    """Default joint damping of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.damping`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.

    .. attention::
        The default stiffness is the value configured by the user or the value parsed from the USD schema.
        It should not be confused with :attr:`joint_damping`, which is the value set into the simulation.
    """
    """所有关节的默认关节缩。
    形状是 (num_instances，num_joints)。

    这种数量通过执行器模型的:attr:`isaaclab.actuators.ActuatorBaseCfg.damping`参数进行配置。
    如果参数的值是None，则在初始化时使用从USD方案解析的值。

    .. 注意::
        默认硬度是用户配置的值或从USD方案中解析的值。
        它不应与:attr:`joint_damping`混，这是仿真中设置的值。
    """

    default_joint_armature: torch.Tensor = None
    """Default joint armature of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.armature`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.
    """
    """所有关节的默认关节 armature。
    形状是 (num_instances，num_joints)。

    这种数量通过执行器模型的:attr:`isaaclab.actuators.ActuatorBaseCfg.armature`参数进行配置。
    如果参数的值是None，则在初始化时使用从USD方案解析的值。
    """

    default_joint_friction_coeff: torch.Tensor = None
    """Default joint static friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.friction`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.

    Note:
        In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
        it is modeled as an effort (torque or force).
    """
    """所有关节的默认静态摩擦系数。
    形状是 (num_instances，num_joints)。

    这种数量通过执行器模型的:attr:`isaaclab.actuators.ActuatorBaseCfg.friction`参数进行配置。
    如果参数的值是None，则在初始化时使用从USD方案解析的值。

    说明：
        在Isaac Sim 4.5中，这个参数是作为一个系数的模型。
        在Isaac Sim 5.0及后版本中，它被仿真为功率 (扭矩或力)。
    """

    default_joint_dynamic_friction_coeff: torch.Tensor = None
    """Default joint dynamic friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's
    :attr:`isaaclab.actuators.ActuatorBaseCfg.dynamic_friction` parameter. If the parameter's value is None,
    the value parsed from the USD schema, at the time of initialization, is used.

    Note:
        In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
        it is modeled as an effort (torque or force).
    """
    """所有关节的默认关节动态摩擦系数
    形状是 (num_instances，num_joints)。

    这种数量通过执行器模型的:attr:`isaaclab.actuators.ActuatorBaseCfg.dynamic_friction`参数进行配置。
    如果参数的值是None，则在初始化时使用从USD方案解析的值。

    说明：
        在Isaac Sim 4.5中，这个参数是作为一个系数的模型。
        在Isaac Sim 5.0及后版本中，它被仿真为功率 (扭矩或力)。
    """

    default_joint_viscous_friction_coeff: torch.Tensor = None
    """Default joint viscous friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's
    :attr:`isaaclab.actuators.ActuatorBaseCfg.viscous_friction` parameter. If the parameter's value is None,
    the value parsed from the USD schema, at the time of initialization, is used.
    """
    """所有关节的默认粘性摩擦系数。
    形状是 (num_instances，num_joints)。

    这种数量通过执行器模型的:attr:`isaaclab.actuators.ActuatorBaseCfg.viscous_friction`参数进行配置。
    如果参数的值是None，则在初始化时使用从USD方案解析的值。
    """

    default_joint_pos_limits: torch.Tensor = None
    """Default joint position limits of all joints. Shape is (num_instances, num_joints, 2).

    The limits are in the order :math:`[lower, upper]`. They are parsed from the USD schema at the
    time of initialization.
    """
    """所有关节的默认关节位置限制。
    形状是 (num_instances，num_joints，2)。

    极限是以数学为`[lower， upper]`的顺序。
    在启动时，它们从USD方案中解析。
    """

    default_fixed_tendon_stiffness: torch.Tensor = None
    """Default tendon stiffness of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """所有固定的子的默认硬性。
    形状是 (num_instances，num_fixed_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_fixed_tendon_damping: torch.Tensor = None
    """Default tendon damping of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """所有固定的fa门默认缩。
    形状是 (num_instances，num_fixed_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_fixed_tendon_limit_stiffness: torch.Tensor = None
    """Default tendon limit stiffness of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """默认的门限制了所有固定门的硬性。
    形状是 (num_instances，num_fixed_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_fixed_tendon_rest_length: torch.Tensor = None
    """Default tendon rest length of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """所有固定的 rest门休息长度是默认的。
    形状是 (num_instances，num_fixed_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_fixed_tendon_offset: torch.Tensor = None
    """Default tendon offset of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """部的缺陷，
    形状是 (num_instances，num_fixed_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_fixed_tendon_pos_limits: torch.Tensor = None
    """Default tendon position limits of all fixed tendons. Shape is (num_instances, num_fixed_tendons, 2).

    The position limits are in the order :math:`[lower, upper]`. They are parsed from the USD schema at the time of
    initialization.
    """
    """所有固定的 tend门位置限制。
    形状是 (num_instances，num_fixed_tendons，2)。

    位置限制是数学:`[lower， upper]`的顺序。
    在启动时，它们从USD方案中解析。
    """

    default_spatial_tendon_stiffness: torch.Tensor = None
    """Default tendon stiffness of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """所有空间肌的默认硬性。
    形状是 (num_instances，num_spatial_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_spatial_tendon_damping: torch.Tensor = None
    """Default tendon damping of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """所有空间肌的默认节。
    形状是 (num_instances，num_spatial_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_spatial_tendon_limit_stiffness: torch.Tensor = None
    """Default tendon limit stiffness of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """默认的门限制了所有空间门的硬性。
    形状是 (num_instances，num_spatial_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_spatial_tendon_offset: torch.Tensor = None
    """Default tendon offset of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """所有空间的默认 of位。
    形状是 (num_instances，num_spatial_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    ##
    # Joint commands -- Set into simulation.
    ##

    joint_pos_target: torch.Tensor = None
    """Joint position targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """
    """用户命令的关节位置目标。
    形状是 (num_instances，num_joints)。

    对于隐含的执行器模型，目标直接被设置在仿真中。
    对于明确的执行器模型，目标用于计算关节力矩 (见:attr:`applied_torque`)，然后设置在仿真中。
    """

    joint_vel_target: torch.Tensor = None
    """Joint velocity targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """
    """用户命令的关节速度目标。
    形状是 (num_instances，num_joints)。

    对于隐含的执行器模型，目标直接被设置在仿真中。
    对于明确的执行器模型，目标用于计算关节力矩 (见:attr:`applied_torque`)，然后设置在仿真中。
    """

    joint_effort_target: torch.Tensor = None
    """Joint effort targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """
    """用户命令的联合努力目标。
    形状是 (num_instances，num_joints)。

    对于隐含的执行器模型，目标直接被设置在仿真中。
    对于明确的执行器模型，目标用于计算关节力矩 (见:attr:`applied_torque`)，然后设置在仿真中。
    """

    ##
    # Joint commands -- Explicit actuators.
    ##

    computed_torque: torch.Tensor = None
    """Joint torques computed from the actuator model (before clipping). Shape is (num_instances, num_joints).

    This quantity is the raw torque output from the actuator mode, before any clipping is applied.
    It is exposed for users who want to inspect the computations inside the actuator model.
    For instance, to penalize the learning agent for a difference between the computed and applied torques.
    """
    """从动机模型计算的关节扭矩 (在切断之前)。
    形状是 (num_instances，num_joints)。

    在裁剪之前，该量是从动机模式中输出的原始扭矩。
    对于想要检查执行器模型内部计算的用户来说，
    例如，为计算和应用扭矩之间的差异处罚学习代理。
    """

    applied_torque: torch.Tensor = None
    """Joint torques applied from the actuator model (after clipping). Shape is (num_instances, num_joints).

    These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
    actuator model.
    """
    """从执行器模型上应用的关节扭矩 (裁剪后)。
    形状是 (num_instances，num_joints)。

    这些扭矩按动力驱动器模型切断:attr:`computed_torque`后设置在仿真中。
    """

    ##
    # Joint properties.
    ##

    joint_stiffness: torch.Tensor = None
    """Joint stiffness provided to the simulation. Shape is (num_instances, num_joints).

    In the case of explicit actuators, the value for the corresponding joints is zero.
    """
    """在仿真过程中提供关节硬度。
    形状是 (num_instances，num_joints)。

    在明确执行器的情况下，对相应的关节的值为零。
    """

    joint_damping: torch.Tensor = None
    """Joint damping provided to the simulation. Shape is (num_instances, num_joints)

    In the case of explicit actuators, the value for the corresponding joints is zero.
    """
    """在仿真中提供了关节缩。
    形状是 (num_instances，num_joints)

    在明确执行器的情况下，对相应的关节的值为零。
    """

    joint_armature: torch.Tensor = None
    """Joint armature provided to the simulation. Shape is (num_instances, num_joints)."""
    """在仿真中提供的联合 armature。
    形状是 (num_instances，num_joints)。
    """

    joint_friction_coeff: torch.Tensor = None
    """Joint static friction coefficient provided to the simulation. Shape is (num_instances, num_joints).

    Note: In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
    it is modeled as an effort (torque or force).
    """
    """为仿真提供的联合静态摩擦系数。
    形状是 (num_instances，num_joints)。

    Note: 在Isaac Sim 4.5中，这个参数是作为一个系数的模型。
          在Isaac Sim 5.0及后版本中，
    它是以力力 (扭矩或力) 模型的。
    """

    joint_dynamic_friction_coeff: torch.Tensor = None
    """Joint dynamic friction coefficient provided to the simulation. Shape is (num_instances, num_joints).

    Note: In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
    it is modeled as an effort (torque or force).
    """
    """为仿真提供联合动态摩擦系数。
    形状是 (num_instances，num_joints)。

    Note: 在Isaac Sim 4.5中，这个参数是作为一个系数的模型。
          在Isaac Sim 5.0及后版本中，
    它是以力力 (扭矩或力) 模型的。
    """

    joint_viscous_friction_coeff: torch.Tensor = None
    """Joint viscous friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
    """在仿真过程中提供的粘性摩擦系数。
    形状是 (num_instances，num_joints)。
    """

    joint_pos_limits: torch.Tensor = None
    """Joint position limits provided to the simulation. Shape is (num_instances, num_joints, 2).

    The limits are in the order :math:`[lower, upper]`.
    """
    """对仿真提供的关节位置限制。
    形状是 (num_instances，num_joints，2)。

    极限是以数学为`[lower， upper]`的顺序。
    """

    joint_vel_limits: torch.Tensor = None
    """Joint maximum velocity provided to the simulation. Shape is (num_instances, num_joints)."""
    """为仿真提供的联合最大速度。
    形状是 (num_instances，num_joints)。
    """

    joint_effort_limits: torch.Tensor = None
    """Joint maximum effort provided to the simulation. Shape is (num_instances, num_joints)."""
    """为仿真提供的最大共同努力。
    形状是 (num_instances，num_joints)。
    """

    ##
    # Joint properties - Custom.
    ##

    soft_joint_pos_limits: torch.Tensor = None
    r"""Soft joint positions limits for all joints. Shape is (num_instances, num_joints, 2).

    The limits are in the order :math:`[lower, upper]`.The soft joint position limits are computed as
    a sub-region of the :attr:`joint_pos_limits` based on the
    :attr:`~isaaclab.assets.ArticulationCfg.soft_joint_pos_limit_factor` parameter.

    Consider the joint position limits :math:`[lower, upper]` and the soft joint position limits
    :math:`[soft_lower, soft_upper]`. The soft joint position limits are computed as:

    .. math::

        soft\_lower = (lower + upper) / 2 - factor * (upper - lower) / 2
        soft\_upper = (lower + upper) / 2 + factor * (upper - lower) / 2

    The soft joint position limits help specify a safety region around the joint limits. It isn't used by the
    simulation, but is useful for learning agents to prevent the joint positions from violating the limits.
    """
    """所有关节的柔软位置限制。
    形状是 (num_instances，num_joints，2)。

    极限是以数学顺序进行的:`[lower， upper]`软结合位置限制是计算为:attr:`joint_pos_limits`基于:attr:`~isaaclab.assets.Articulatio
    nCfg.soft_joint_pos_limit_factor`参数

    考虑关节位置限制:数学:`[lower， upper]`和软关节位置限制
    :math:`[soft_lower， soft_upper]`软关节位置限制计算为:

    .. math::

        soft\_lower = (lower + upper) / 2 - factor * (upper - lower) / 2
        soft\_upper = (lower + upper) / 2 + factor * (upper - lower) / 2

    柔性关节位置限制有助于确定关节限制周围的安全区域。
    它不是在仿真中使用的，但对于学习代理来说是有用的，
    """

    soft_joint_vel_limits: torch.Tensor = None
    """Soft joint velocity limits for all joints. Shape is (num_instances, num_joints).

    These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
    has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
    """
    """所有关节的软关节速度限制。
    形状是 (num_instances，num_joints)。

    这些来自动机模型。
    如果执行器模型具有可变速度限制模型，它可能与:attr:`joint_vel_limits`不同。
    例如，在变速率驱动器模型中。
    """

    gear_ratio: torch.Tensor = None
    """Gear ratio for relating motor torques to applied Joint torques. Shape is (num_instances, num_joints)."""
    """适用于运动扭矩和应用的关节力矩的交换比。
    形状是 (num_instances，num_joints)。
    """

    ##
    # Fixed tendon properties.
    ##

    fixed_tendon_stiffness: torch.Tensor = None
    """Fixed tendon stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
    """在仿真过程中提供固定的门硬度。
    形状是 (num_instances，num_fixed_tendons)。
    """

    fixed_tendon_damping: torch.Tensor = None
    """Fixed tendon damping provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
    """在仿真过程中提供固定门。
    形状是 (num_instances，num_fixed_tendons)。
    """

    fixed_tendon_limit_stiffness: torch.Tensor = None
    """Fixed tendon limit stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
    """在仿真中提供固定门限制硬度。
    形状是 (num_instances，num_fixed_tendons)。
    """

    fixed_tendon_rest_length: torch.Tensor = None
    """Fixed tendon rest length provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
    """在仿真过程中提供固定的肌休息长度。
    形状是 (num_instances，num_fixed_tendons)。
    """

    fixed_tendon_offset: torch.Tensor = None
    """Fixed tendon offset provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
    """在仿真过程中提供固定的肌肉偏移。
    形状是 (num_instances，num_fixed_tendons)。
    """

    fixed_tendon_pos_limits: torch.Tensor = None
    """Fixed tendon position limits provided to the simulation. Shape is (num_instances, num_fixed_tendons, 2)."""
    """在仿真过程中提供固定位限制。
    形状是 (num_instances，num_fixed_tendons，2)。
    """

    ##
    # Spatial tendon properties.
    ##

    spatial_tendon_stiffness: torch.Tensor = None
    """Spatial tendon stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
    """在仿真过程中提供了空间硬性。
    形状是 (num_instances，num_spatial_tendons)。
    """

    spatial_tendon_damping: torch.Tensor = None
    """Spatial tendon damping provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
    """在仿真中提供了空间门。
    形状是 (num_instances，num_spatial_tendons)。
    """

    spatial_tendon_limit_stiffness: torch.Tensor = None
    """Spatial tendon limit stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
    """在仿真中提供空间门限制硬度。
    形状是 (num_instances，num_spatial_tendons)。
    """

    spatial_tendon_offset: torch.Tensor = None
    """Spatial tendon offset provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
    """在仿真过程中提供空间位。
    形状是 (num_instances，num_spatial_tendons)。
    """

    ##
    # Root state properties.
    ##

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        """Root link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中，根链接呈现``[pos， quat]``。
        形状是 (num_instances， 7)。

        这个数量是关节根的演员框架相对于世界的姿势。
        导向提供 (w， x， y， z) 格式。
        """
        if self._root_link_pose_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_physx_view.get_root_transforms().clone()
            pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._root_link_pose_w.data = pose
            self._root_link_pose_w.timestamp = self._sim_timestamp

        return self._root_link_pose_w.data

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        """在仿真世界框架中的根链速度``[lin_vel， ang_vel]``。
        形状是 (num_instances， 6)。

        这个数量包含关节根的演员框架相对于世界的线性和角速度。
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            # read the CoM velocity
            vel = self.root_com_vel_w.clone()
            # adjust linear velocity to link from center of mass
            vel[:, :3] += torch.linalg.cross(
                vel[:, 3:], math_utils.quat_apply(self.root_link_quat_w, -self.body_com_pos_b[:, 0]), dim=-1
            )
            # set the buffer data and timestamp
            self._root_link_vel_w.data = vel
            self._root_link_vel_w.timestamp = self._sim_timestamp

        return self._root_link_vel_w.data

    @property
    def root_com_pose_w(self) -> torch.Tensor:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中，质量中心 ``[pos， quat]``。
        形状是 (num_instances， 7)。

        这个数量是关节根的质量框架中心相对于世界的姿势。
        导向提供 (w， x， y， z) 格式。
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            pos, quat = math_utils.combine_frame_transforms(
                self.root_link_pos_w, self.root_link_quat_w, self.body_com_pos_b[:, 0], self.body_com_quat_b[:, 0]
            )
            # set the buffer data and timestamp
            self._root_com_pose_w.data = torch.cat((pos, quat), dim=-1)
            self._root_com_pose_w.timestamp = self._sim_timestamp

        return self._root_com_pose_w.data

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        """大量速度的根中心``[lin_vel， ang_vel]``在仿真世界框架中。
        形状是 (num_instances， 6)。

        这种数量包含与世界相对的关节根质量框架中心的线性和角速度。
        """
        if self._root_com_vel_w.timestamp < self._sim_timestamp:
            self._root_com_vel_w.data = self._root_physx_view.get_root_velocities()
            self._root_com_vel_w.timestamp = self._sim_timestamp

        return self._root_com_vel_w.data

    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame relative to the world. Meanwhile,
        the linear and angular velocities are of the articulation root's center of mass frame.
        """
        """在仿真世界框架中的根状态``[pos， quat， lin_vel， ang_vel]``。
        形状是 (num_instances， 13)。

        位置和四元数是关节根与世界相对的演员框架。
        与此同时，线性和角的速度是关节根的质量框架中心。
        """
        if self._root_state_w.timestamp < self._sim_timestamp:
            self._root_state_w.data = torch.cat((self.root_link_pose_w, self.root_com_vel_w), dim=-1)
            self._root_state_w.timestamp = self._sim_timestamp

        return self._root_state_w.data

    @property
    def root_link_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the articulation root's actor frame relative to the
        world.
        """
        """在仿真世界框架中的根状态``[pos， quat， lin_vel， ang_vel]``。
        形状是 (num_instances， 13)。

        位置，四元数和线性/角的速度是关节根的演员框架相对于世界。
        """
        if self._root_link_state_w.timestamp < self._sim_timestamp:
            self._root_link_state_w.data = torch.cat((self.root_link_pose_w, self.root_link_vel_w), dim=-1)
            self._root_link_state_w.timestamp = self._sim_timestamp

        return self._root_link_state_w.data

    @property
    def root_com_state_w(self):
        """Root center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the articulation root link's center of mass frame
        relative to the world. Center of mass frame is assumed to be the same orientation as the link rather than the
        orientation of the principle inertia.
        """
        """在仿真世界框架中的质量状态``[pos， quat， lin_vel， ang_vel]``的根中心。
        形状是 (num_instances， 13)。

        位置，四元数和线性/角速度是关节根链的质量框架中心相对于世界。
        质量框架的中心被认为是与链接相同的方向，而不是惯性原则的方向。
        """
        if self._root_com_state_w.timestamp < self._sim_timestamp:
            self._root_com_state_w.data = torch.cat((self.root_com_pose_w, self.root_com_vel_w), dim=-1)
            self._root_com_state_w.timestamp = self._sim_timestamp

        return self._root_com_state_w.data

    ##
    # Body state properties.
    ##

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        """Body link pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中，身体链接呈现``[pos， quat]``。
        形状是 (num_instances，num_bodies， 7)。

        这个数量是关节链的演员框架相对于世界的姿势。
        导向提供 (w， x， y， z) 格式。
        """
        if self._body_link_pose_w.timestamp < self._sim_timestamp:
            # perform forward kinematics (shouldn't cause overhead if it happened already)
            self._physics_sim_view.update_articulations_kinematic()
            # read data from simulation
            poses = self._root_physx_view.get_link_transforms().clone()
            poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._body_link_pose_w.data = poses
            self._body_link_pose_w.timestamp = self._sim_timestamp

        return self._body_link_pose_w.data

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        """在仿真世界框架中，身体链接速度``[lin_vel， ang_vel]``。
        形状是 (num_instances，num_bodies， 6)。

        这个数量包含关节链的演员框架相对于世界的线性和角速度。
        """
        if self._body_link_vel_w.timestamp < self._sim_timestamp:
            # read data from simulation
            velocities = self.body_com_vel_w.clone()
            # adjust linear velocity to link from center of mass
            velocities[..., :3] += torch.linalg.cross(
                velocities[..., 3:], math_utils.quat_apply(self.body_link_quat_w, -self.body_com_pos_b), dim=-1
            )
            # set the buffer data and timestamp
            self._body_link_vel_w.data = velocities
            self._body_link_vel_w.timestamp = self._sim_timestamp

        return self._body_link_vel_w.data

    @property
    def body_com_pose_w(self) -> torch.Tensor:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中，体质中心 ``[pos， quat]``。
        形状是 (num_instances，num_bodies， 7)。

        这个数量是对世界相对的关节链的质量框架中心的姿势。
        导向提供 (w， x， y， z) 格式。
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            pos, quat = math_utils.combine_frame_transforms(
                self.body_link_pos_w, self.body_link_quat_w, self.body_com_pos_b, self.body_com_quat_b
            )
            # set the buffer data and timestamp
            self._body_com_pose_w.data = torch.cat((pos, quat), dim=-1)
            self._body_com_pose_w.timestamp = self._sim_timestamp

        return self._body_com_pose_w.data

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        """在仿真世界框架中体积速度``[lin_vel， ang_vel]``的中心。
        形状是 (num_instances，num_bodies， 6)。

        这个数量包含与世界相对的质量框架的关节链的直线和角速度。
        """
        if self._body_com_vel_w.timestamp < self._sim_timestamp:
            self._body_com_vel_w.data = self._root_physx_view.get_link_velocities()
            self._body_com_vel_w.timestamp = self._sim_timestamp

        return self._body_com_vel_w.data

    @property
    def body_state_w(self):
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position and quaternion are of all the articulation links' actor frame. Meanwhile, the linear and angular
        velocities are of the articulation links's center of mass frame.
        """
        """在仿真世界框架中所有物体的状态 `[pos， quat， lin_vel， ang_vel]`。
        形状是 (num_instances，num_bodies， 13)。

        位置和四元数是所有关节链的演员框架。
        与此同时，线性和角的速度是关节链的质量框架中心。
        """
        if self._body_state_w.timestamp < self._sim_timestamp:
            self._body_state_w.data = torch.cat((self.body_link_pose_w, self.body_com_vel_w), dim=-1)
            self._body_state_w.timestamp = self._sim_timestamp

        return self._body_state_w.data

    @property
    def body_link_state_w(self):
        """State of all bodies' link frame`[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        """
        """在仿真世界框架中，所有机体的链接框架`[pos， quat， lin_vel， ang_vel]`状态。
        形状是 (num_instances，num_bodies， 13)。

        位置，四元数和线性/角速度是身体与世界相对的链接框架。
        """
        if self._body_link_state_w.timestamp < self._sim_timestamp:
            self._body_link_state_w.data = torch.cat((self.body_link_pose_w, self.body_link_vel_w), dim=-1)
            self._body_link_state_w.timestamp = self._sim_timestamp

        return self._body_link_state_w.data

    @property
    def body_com_state_w(self):
        """State of all bodies center of mass `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia.
        """
        """在仿真世界框架中所有物体的质量中心`[pos， quat， lin_vel， ang_vel]`的状态。
        形状是 (num_instances，num_bodies， 13)。

        位置，四元数和线性/角的速度是身体与世界相对的质量框架中心。
        质量框架的中心被认为是与链接相同的方向，而不是惯性原则的方向。
        """
        if self._body_com_state_w.timestamp < self._sim_timestamp:
            self._body_com_state_w.data = torch.cat((self.body_com_pose_w, self.body_com_vel_w), dim=-1)
            self._body_com_state_w.timestamp = self._sim_timestamp

        return self._body_com_state_w.data

    @property
    def body_com_acc_w(self):
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]``.
        Shape is (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        """所有物体的加速重量中心``[lin_acc， ang_acc]``。
        形状是 (num_instances，num_bodies， 6)。

        所有的价值观都与世界相对。
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._body_com_acc_w.data = self._root_physx_view.get_link_accelerations()
            self._body_com_acc_w.timestamp = self._sim_timestamp

        return self._body_com_acc_w.data

    @property
    def body_com_pose_b(self) -> torch.Tensor:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.
        Shape is (num_instances, 1, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (w, x, y, z) format.
        """
        """所有体体在各自体的链接框架中，质量中心 ``[pos， quat]``。
        形状是 (num_instances， 1， 7)。

        这个数量是硬体质量框架中心的姿势与身体的链接框架相比。
        导向提供 (w， x， y， z) 格式。
        """
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_physx_view.get_coms().to(self.device)
            pose[..., 3:7] = math_utils.convert_quat(pose[..., 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._body_com_pose_b.data = pose
            self._body_com_pose_b.timestamp = self._sim_timestamp

        return self._body_com_pose_b.data

    @property
    def body_incoming_joint_wrench_b(self) -> torch.Tensor:
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation.

        For more information on joint wrenches, please check the`PhysX documentation`_ and the underlying
        `PhysX Tensor API`_.

        .. _`PhysX documentation`: https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force
        .. _`PhysX Tensor API`: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force
        """
        """从父母身体到孩子身体的联合反应钥匙在父母身体框架中应用。

        形状是 (num_instances，num_bodies， 6)。
        所有的身体反应钥匙都提供，包括根体，

        有关关键的更多信息，请查看`PhysX documentation`_和底层`PhysX Tensor API`_。

        .. _`PhysX documentation`: https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force
        .. _`PhysX Tensor API`: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force
        """

        if self._body_incoming_joint_wrench_b.timestamp < self._sim_timestamp:
            self._body_incoming_joint_wrench_b.data = self._root_physx_view.get_link_incoming_joint_force()
            self._body_incoming_joint_wrench_b.time_stamp = self._sim_timestamp
        return self._body_incoming_joint_wrench_b.data

    ##
    # Joint state properties.
    ##

    @property
    def joint_pos(self):
        """Joint positions of all joints. Shape is (num_instances, num_joints)."""
        """所有关节的关节位置。
        形状是 (num_instances，num_joints)。
        """
        if self._joint_pos.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_pos.data = self._root_physx_view.get_dof_positions()
            self._joint_pos.timestamp = self._sim_timestamp
        return self._joint_pos.data

    @property
    def joint_vel(self):
        """Joint velocities of all joints. Shape is (num_instances, num_joints)."""
        """所有关节的关节速度。
        形状是 (num_instances，num_joints)。
        """
        if self._joint_vel.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_vel.data = self._root_physx_view.get_dof_velocities()
            self._joint_vel.timestamp = self._sim_timestamp
        return self._joint_vel.data

    @property
    def joint_acc(self):
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        """所有关节的联合加速。
        形状是 (num_instances，num_joints)。
        """
        if self._joint_acc.timestamp < self._sim_timestamp:
            # note: we use finite differencing to compute acceleration
            time_elapsed = self._sim_timestamp - self._joint_acc.timestamp
            self._joint_acc.data = (self.joint_vel - self._previous_joint_vel) / time_elapsed
            self._joint_acc.timestamp = self._sim_timestamp
            # update the previous joint velocity
            self._previous_joint_vel[:] = self.joint_vel
        return self._joint_acc.data

    ##
    # Derived Properties.
    ##

    @property
    def projected_gravity_b(self):
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        """在基架上投射重力方向。
        形状是 (num_instances， 3)。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        """基架的 Yaw方向 (在半径中)。
        形状是 (num_instances，)。

        说明：
            这个数量是通过假设基架的前向方向沿着x方向计算的，i.e.:数学:`(1， 0， 0)`。
        """
        forward_w = math_utils.quat_apply(self.root_link_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        """根链的线性速度在基架中。
        形状是 (num_instances， 3)。

        这个数量是关节根的演员框架与其演员框架的线性速度。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        """根链角速度在基础世界框架。
        形状是 (num_instances， 3)。

        这个数量是关节根的演员框架与其演员框架的角速度。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        """在基架中，质量线性速度的根中心。
        形状是 (num_instances， 3)。

        这个数量是关节根的质量框架中心与其演员框架的线性速度。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        """基层世界框架中的质量角速度的根中心。
        形状是 (num_instances， 3)。

        这个数量是关节根的质量框架中心与其演员框架的角速度。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_ang_vel_w)

    ##
    # Sliced properties.
    ##

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        """在仿真世界框架中的根链位置。
        形状是 (num_instances， 3)。

        这种数量是根固体与世界相对的演员框架的位置。
        """
        return self.root_link_pose_w[:, :3]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        """在仿真世界框架中，根链的导向 (w，x，y，z)。
        形状是 (num_instances， 4)。

        这种数量是根固体的演员框架的方向。
        """
        return self.root_link_pose_w[:, 3:7]

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        """在仿真世界框架中的根线性速度。
        形状是 (num_instances， 3)。

        这个数量是根固体的演员框架相对于世界的线性速度。
        """
        return self.root_link_vel_w[:, :3]

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        """在仿真世界框架中的根链角速度。
        形状是 (num_instances， 3)。

        这个数量是根固体与世界相对的演员框架的角速度。
        """
        return self.root_link_vel_w[:, 3:6]

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        """在仿真世界框架中，
        形状是 (num_instances， 3)。

        这种数量是根固体与世界相对的演员框架的位置。
        """
        return self.root_com_pose_w[:, :3]

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        """在仿真世界框架中的质量导向的根中心 (w，x，y，z)。
        形状是 (num_instances， 4)。

        这种数量是根固体与世界相对的演员框架的方向。
        """
        return self.root_com_pose_w[:, 3:7]

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        """在仿真世界框架中的质量线性速度的根中心。
        形状是 (num_instances， 3)。

        这个数量是根固体质量框架中心与世界相对的线性速度。
        """
        return self.root_com_vel_w[:, :3]

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        """在仿真世界框架中的质量角速度的根中心。
        形状是 (num_instances， 3)。

        这个数量是根固体质量框架中心与世界相对的角速度。
        """
        return self.root_com_vel_w[:, 3:6]

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        """仿真世界框架中的所有物体的位置。
        形状是 (num_instances，num_bodies， 3)。

        这种数量是关节体的演员框架与世界相对的位置。
        """
        return self.body_link_pose_w[..., :3]

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        """在仿真世界框架中的所有物体的导向 (w，x，y，z)。
        形状是 (num_instances，num_bodies， 4)。

        这种数量是关节体的演员框架与世界相对的方向。
        """
        return self.body_link_pose_w[..., 3:7]

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame relative to the world.
        """
        """在仿真世界框架中的所有物体的线性速度。
        形状是 (num_instances，num_bodies， 3)。

        这个数量是关节体质量框架中心的线性速度相对于世界。
        """
        return self.body_link_vel_w[..., :3]

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame relative to the world.
        """
        """在仿真世界框架中的所有物体的角速度。
        形状是 (num_instances，num_bodies， 3)。

        这个数量是关节体质量框架中心与世界相对的角速度。
        """
        return self.body_link_vel_w[..., 3:6]

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame.
        """
        """仿真世界框架中的所有物体的位置。
        形状是 (num_instances，num_bodies， 3)。

        这种数量是关节体的演员框架的位置。
        """
        return self.body_com_pose_w[..., :3]

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame.
        """
        """在仿真世界框架中所有物体的惯性基本轴的导向 (w，x，y，z)。
        形状是 (num_instances，num_bodies， 4)。

        这种数量是关节体的演员框架的方向。
        """
        return self.body_com_pose_w[..., 3:7]

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        """在仿真世界框架中的所有物体的线性速度。
        形状是 (num_instances，num_bodies， 3)。

        这个数量是关节体质量框架中心的线性速度。
        """
        return self.body_com_vel_w[..., :3]

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        """在仿真世界框架中的所有物体的角速度。
        形状是 (num_instances，num_bodies， 3)。

        这个数量是关节体质量框架中心的角速度。
        """
        return self.body_com_vel_w[..., 3:6]

    @property
    def body_com_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        """在仿真世界框架中的所有物体的线性加速。
        形状是 (num_instances，num_bodies， 3)。

        这种数量是关节体质量框架中心的线性加速。
        """
        return self.body_com_acc_w[..., :3]

    @property
    def body_com_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        """在仿真世界框架中的所有物体的角加速。
        形状是 (num_instances，num_bodies， 3)。

        这种数量是关节体质量框架中心的角加速。
        """
        return self.body_com_acc_w[..., 3:6]

    @property
    def body_com_pos_b(self) -> torch.Tensor:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body'slink frame.
        """
        """所有物体在各自的链接框架中的质量位置中心。
        形状是 (num_instances，num_bodies， 3)。

        这个数量是相对于其身体的斜体位置的中心。
        """
        return self.body_com_pose_b[..., :3]

    @property
    def body_com_quat_b(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        """所有物体在各自的链接框架中的惯性轴的方向 (w，x，y，z)。
        形状是 (num_instances，num_bodies， 4)。

        这种数量是对其身体的链接框架的惯性轴的方向。
        """
        return self.body_com_pose_b[..., 3:7]

    ##
    # Backward compatibility.
    ##

    @property
    def root_pose_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_pose_w`."""
        """像:attr:`root_link_pose_w`一样。"""
        return self.root_link_pose_w

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_pos_w`."""
        """像:attr:`root_link_pos_w`一样。"""
        return self.root_link_pos_w

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_quat_w`."""
        """像:attr:`root_link_quat_w`一样。"""
        return self.root_link_quat_w

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_vel_w`."""
        """像:attr:`root_com_vel_w`一样。"""
        return self.root_com_vel_w

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_lin_vel_w`."""
        """像:attr:`root_com_lin_vel_w`一样。"""
        return self.root_com_lin_vel_w

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_ang_vel_w`."""
        """像:attr:`root_com_ang_vel_w`一样。"""
        return self.root_com_ang_vel_w

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Same as :attr:`root_com_lin_vel_b`."""
        """像:attr:`root_com_lin_vel_b`一样。"""
        return self.root_com_lin_vel_b

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Same as :attr:`root_com_ang_vel_b`."""
        """像:attr:`root_com_ang_vel_b`一样。"""
        return self.root_com_ang_vel_b

    @property
    def body_pose_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_pose_w`."""
        """像:attr:`body_link_pose_w`一样。"""
        return self.body_link_pose_w

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_pos_w`."""
        """像:attr:`body_link_pos_w`一样。"""
        return self.body_link_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_quat_w`."""
        """像:attr:`body_link_quat_w`一样。"""
        return self.body_link_quat_w

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_vel_w`."""
        """像:attr:`body_com_vel_w`一样。"""
        return self.body_com_vel_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_lin_vel_w`."""
        """像:attr:`body_com_lin_vel_w`一样。"""
        return self.body_com_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_ang_vel_w`."""
        """像:attr:`body_com_ang_vel_w`一样。"""
        return self.body_com_ang_vel_w

    @property
    def body_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_acc_w`."""
        """像:attr:`body_com_acc_w`一样。"""
        return self.body_com_acc_w

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_lin_acc_w`."""
        """像:attr:`body_com_lin_acc_w`一样。"""
        return self.body_com_lin_acc_w

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_ang_acc_w`."""
        """像:attr:`body_com_ang_acc_w`一样。"""
        return self.body_com_ang_acc_w

    @property
    def com_pos_b(self) -> torch.Tensor:
        """Same as :attr:`body_com_pos_b`."""
        """像:attr:`body_com_pos_b`一样。"""
        return self.body_com_pos_b

    @property
    def com_quat_b(self) -> torch.Tensor:
        """Same as :attr:`body_com_quat_b`."""
        """像:attr:`body_com_quat_b`一样。"""
        return self.body_com_quat_b

    @property
    def joint_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_pos_limits` instead."""
        """废弃的财产。
        请使用:attr:`joint_pos_limits`。
        """
        logger.warning(
            "The `joint_limits` property will be deprecated in a future release. Please use `joint_pos_limits` instead."
        )
        return self.joint_pos_limits

    @property
    def default_joint_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_joint_pos_limits` instead."""
        """废弃的财产。
        请使用:attr:`default_joint_pos_limits`。
        """
        logger.warning(
            "The `default_joint_limits` property will be deprecated in a future release. Please use"
            " `default_joint_pos_limits` instead."
        )
        return self.default_joint_pos_limits

    @property
    def joint_velocity_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_vel_limits` instead."""
        """废弃的财产。
        请使用:attr:`joint_vel_limits`。
        """
        logger.warning(
            "The `joint_velocity_limits` property will be deprecated in a future release. Please use"
            " `joint_vel_limits` instead."
        )
        return self.joint_vel_limits

    @property
    def joint_friction(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead."""
        """废弃的财产。
        请使用:attr:`joint_friction_coeff`。
        """
        logger.warning(
            "The `joint_friction` property will be deprecated in a future release. Please use"
            " `joint_friction_coeff` instead."
        )
        return self.joint_friction_coeff

    @property
    def default_joint_friction(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_joint_friction_coeff` instead."""
        """废弃的财产。
        请使用:attr:`default_joint_friction_coeff`。
        """
        logger.warning(
            "The `default_joint_friction` property will be deprecated in a future release. Please use"
            " `default_joint_friction_coeff` instead."
        )
        return self.default_joint_friction_coeff

    @property
    def fixed_tendon_limit(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`fixed_tendon_pos_limits` instead."""
        """废弃的财产。
        请使用:attr:`fixed_tendon_pos_limits`。
        """
        logger.warning(
            "The `fixed_tendon_limit` property will be deprecated in a future release. Please use"
            " `fixed_tendon_pos_limits` instead."
        )
        return self.fixed_tendon_pos_limits

    @property
    def default_fixed_tendon_limit(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_fixed_tendon_pos_limits` instead."""
        """废弃的财产。
        请使用:attr:`default_fixed_tendon_pos_limits`。
        """
        logger.warning(
            "The `default_fixed_tendon_limit` property will be deprecated in a future release. Please use"
            " `default_fixed_tendon_pos_limits` instead."
        )
        return self.default_fixed_tendon_pos_limits
