# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from . import actuator_pd
from .actuator_base_cfg import ActuatorBaseCfg

"""
Implicit Actuator Models.
"""
"""隐含的执行器模型。
"""


@configclass
class ImplicitActuatorCfg(ActuatorBaseCfg):
    """Configuration for an implicit actuator.

    Note:
        The PD control is handled implicitly by the simulation.
    """
    """一个隐含动机的配置。

    说明：
        仿真器将隐含地处理PD控制器。
    """

    class_type: type = actuator_pd.ImplicitActuator


"""
Explicit Actuator Models.
"""
"""显而易见的执行器模型。
"""


@configclass
class IdealPDActuatorCfg(ActuatorBaseCfg):
    """Configuration for an ideal PD actuator."""
    """设置一个理想的PD执行器。"""

    class_type: type = actuator_pd.IdealPDActuator


@configclass
class DCMotorCfg(IdealPDActuatorCfg):
    """Configuration for direct control (DC) motor actuator model."""
    """直接控制 (DC) 发动机执行器模型的配置。"""

    class_type: type = actuator_pd.DCMotor

    saturation_effort: float = MISSING
    """Peak motor force/torque of the electric DC motor (in N-m)."""
    """电动DC电动机的峰值动力/扭矩 (N-m)。"""


@configclass
class DelayedPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for a delayed PD actuator."""
    """延迟PD执行器的配置。"""

    class_type: type = actuator_pd.DelayedPDActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""
    """执行器命令可能延迟的物理时间最小数。
    默认为0。
    """

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""
    """执行器命令可能延迟的物理时间步骤的最大数量。
    默认为0。
    """


@configclass
class RemotizedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for a remotized PD actuator.

    Note:
        The torque output limits for this actuator is derived from a linear interpolation of a lookup table
        in :attr:`joint_parameter_lookup`. This table describes the relationship between joint angles and
        the output torques.
    """
    """设置一个移动的PD执行器。

    说明：
        这种动机的扭矩输出限量来自查找表的线性插图
        in :吸引:`joint_parameter_lookup`。
            这张表描述了关节角与
        输出扭矩。
    """

    class_type: type = actuator_pd.RemotizedPDActuator

    joint_parameter_lookup: list[list[float]] = MISSING
    """Joint parameter lookup table. Shape is (num_lookup_points, 3).

    This tensor describes the relationship between the joint angle (rad), the transmission ratio (in/out),
    and the output torque (N*m). The table is used to interpolate the output torque based on the joint angle.
    """
    """共同参数查找表
    形状是 (num_lookup_points， 3)。

    这种子描述了关节角 (rad)，传输比 (入/出) 和输出扭矩 (N*m) 的关系。
    该表用于基于合角的输出扭矩进行回合。
    """
