# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.utils import configclass

from .operational_space import OperationalSpaceController


@configclass
class OperationalSpaceControllerCfg:
    """Configuration for operational-space controller."""
    """操作空间控制器的配置"""

    class_type: type = OperationalSpaceController
    """The associated controller class."""
    """相关控制器类。"""

    target_types: Sequence[str] = MISSING
    """Type of task-space targets.

    It has two sub-strings joined by underscore:
        - type of task-space target: ``"pose"``, ``"wrench"``
        - reference for the task-space targets: ``"abs"`` (absolute), ``"rel"`` (relative, only for pose)
    """
    """任务空间目标类型。

    它有两个子字符串，
        - 任务空间目标类型:``"pose"``，``"wrench"``
        - 任务空间目标的参考:``"abs"`` (绝对)，``"rel"`` (相对，仅适用于姿势)
    """

    motion_control_axes_task: Sequence[int] = (1, 1, 1, 1, 1, 1)
    """Motion direction to control in task reference frame. Mark as ``0/1`` for each axis."""
    """在任务参考框架中控制的运动方向。
    每个轴的标记为``0/1``。
    """

    contact_wrench_control_axes_task: Sequence[int] = (0, 0, 0, 0, 0, 0)
    """Contact wrench direction to control in task reference frame. Mark as 0/1 for each axis."""
    """在任务参考框架中控制的触摸钥匙方向。
    标记为每个轴的0/1。
    """

    inertial_dynamics_decoupling: bool = False
    """Whether to perform inertial dynamics decoupling for motion control (inverse dynamics)."""
    """对运动控制 (反动动态) 进行惯性动力脱是否。"""

    partial_inertial_dynamics_decoupling: bool = False
    """Whether to ignore the inertial coupling between the translational & rotational motions."""
    """转换和旋转运动之间的惯性合是否要忽视。"""

    gravity_compensation: bool = False
    """Whether to perform gravity compensation."""
    """是否进行重力补偿。"""

    impedance_mode: str = "fixed"
    """Type of gains for motion control: ``"fixed"``, ``"variable"``, ``"variable_kp"``."""
    """运动控制的收益类型:``"fixed"``，``"variable"``，``"variable_kp"``。"""

    motion_stiffness_task: float | Sequence[float] = (100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
    """The positional gain for determining operational space command forces based on task-space pose error."""
    """基于任务空间的位置获益来确定运营空间命令力量存在错误。"""

    motion_damping_ratio_task: float | Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    """The damping ratio is used in-conjunction with positional gain to compute operational space command forces
    based on task-space velocity error.

    The following math operation is performed for computing velocity gains:
        :math:`d_gains = 2 * sqrt(p_gains) * damping_ratio`.
    """
    """压缩比与定位增长结合使用，以基于任务空间速度错误计算操作空间命令力量。

    为计算速度增长执行以下数学操作:
        :math:`d_gains = 2 * sqrt(p_gains) * damping_ratio`。
    """

    motion_stiffness_limits_task: tuple[float, float] = (0, 1000)
    """Minimum and maximum values for positional gains.

    Note: Used only when :obj:`impedance_mode` is ``"variable"`` or ``"variable_kp"``.
    """
    """定位收益的最低和最高值。

    Note: 只有当:obj:`impedance_mode`是``"variable"``或``"variable_kp"``时使用。
    """

    motion_damping_ratio_limits_task: tuple[float, float] = (0, 100)
    """Minimum and maximum values for damping ratios used to compute velocity gains.

    Note: Used only when :obj:`impedance_mode` is ``"variable"``.
    """
    """用于计算速度增长的压缩比的最小和最大值。

    Note: 只有当:obj:`impedance_mode`是``"variable"``时使用。
    """

    contact_wrench_stiffness_task: float | Sequence[float] | None = None
    """The proportional gain for determining operational space command forces for closed-loop contact force control.

    If ``None``, then open-loop control of desired contact wrench is performed.

    Note: since only the linear forces could be measured at the moment,
    only the first three elements are used for the feedback loop.
    """
    """对于闭环接触力控制的操作空间命令力量的比例增长。

    如果 ``None``，则需要的接触钥匙进行开放循环控制。

    Note: 由于目前只能测量线性力量，
    只有第一三个元素用于反循环。
    """

    nullspace_control: str = "none"
    """The null space control method for redundant manipulators: ``"none"``, ``"position"``.

    Note: ``"position"`` is used to drive the redundant manipulator to zero configuration by default. If
    ``target_joint_pos`` is provided in the ``compute()`` method, it will be driven to this configuration.
    """
    """冗余操纵器的零空间控制方法:``"none"``，``"position"``。

    Note: ``"position"``用于默认地将冗余操作器驱动到零配置。
          如果
    ``target_joint_pos``在``compute()``方法中提供，它将被驱动到这个配置。
    """

    nullspace_stiffness: float = 10.0
    """The stiffness for null space control."""
    """对于零空间控制的硬度。"""

    nullspace_damping_ratio: float = 1.0
    """The damping ratio for null space control."""
    """零空间控制的缩比。"""
