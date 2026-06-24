# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .thruster import Thruster


@configclass
class ThrusterCfg:
    """Configuration for thruster actuator groups.

    This config defines per-actuator-group parameters used by the low-level
    thruster/motor models (time-constants, thrust ranges, integration scheme,
    and initial state specifications). Fields left as ``MISSING`` are required
    and must be provided by the user configuration.
    """
    """驱动器执行器组的配置

    该配置定义了低级推进器/发动机模型所使用的每个执行器组参数 (时间常量，推进范围，集成方案和初始状态规范)。
    要求留为``MISSING``的字段，必须由用户配置提供。
    """

    class_type: type[Thruster] = Thruster
    """Concrete Python class that consumes this config."""
    """需要使用这个配置。"""

    dt: float = MISSING
    """Simulation/integration timestep used by the thruster update [s]."""
    """推进器更新[s]所使用的仿真/集成时间步骤。"""

    thrust_range: tuple[float, float] = MISSING
    """Per-motor thrust clamp range [N]: values are clipped to this interval."""
    """每动机推力门范围 [N]:值将裁剪到此间隔。"""

    max_thrust_rate: float = 100000.0
    """Per-motor thrust slew-rate limit applied inside the first-order model [N/s]."""
    """在一级模型内 (N/s) 应用的每发动机推力杀速限制。"""

    thrust_const_range: tuple[float, float] = MISSING
    """Range for thrust coefficient :math:`k_f` [N/(rps²)]."""
    """推力系数范围:数学:`k_f` [N/(rps2)]。"""

    tau_inc_range: tuple[float, float] = MISSING
    """Range of time constants when commanded output is **increasing** (rise dynamics) [s]."""
    """命令输出时的时间常数范围是 **增加** (上升动态) [s]。"""

    tau_dec_range: tuple[float, float] = MISSING
    """Range of time constants when commanded output is **decreasing** (fall dynamics) [s]."""
    """命令输出时的时间常数范围是 **降低** (下降动态) [s]。"""

    torque_to_thrust_ratio: float = MISSING
    """Yaw-moment coefficient converting thrust to motor torque about +Z [N·m per N].
    Used as ``tau_z = torque_to_thrust_ratio * thrust_z * direction``.
    """
    """Yaw时系数将推力转换为电动扭矩约为+Z [N·m/N]。
    用于``tau_z = torque_to_thrust_ratio * thrust_z * direction``。
    """

    use_discrete_approximation: bool = True
    """
    Determines how the actuator/motor mixing factor is computed. Defaults to True.

    If True, uses the discrete-time factor ``1 / (dt + tau)``, accounting for the control loop timestep.
    If False, uses the continuous-time factor ``1 / tau``.
    """
    """确定执行器/发动机混合因素的计算方式。
    默认为 True。

    如果 True，则使用分离时间因子 ``1 / (dt + tau)``，计算控制循环时间步骤。
    如果 False，则使用连续时间系数 ``1 / tau``。
    """

    integration_scheme: Literal["rk4", "euler"] = "rk4"
    """Numerical integrator for the first-order model. Defaults to 'rk4'."""
    """第一级模型的数字集成器。
    在"rk4"中默认设置。
    """

    thruster_names_expr: list[str] = MISSING
    """Articulation's joint names that are part of the group."""
    """关节的共同名字是集团的一部分。"""
