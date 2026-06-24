# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for thruster actuator models.

This package provides actuator models specifically designed for multirotor thrusters.
The thruster actuator simulates realistic motor/propeller dynamics including asymmetric
rise and fall time constants, thrust limits, and dynamic response characteristics.
"""
"""推动器动机模型的子包

本包提供了专门为多轮驱动器设计的动力机型。
推进器执行器仿真了现实的发动机/螺旋动力，包括不对称的上降时间常量，推力限制和动态响应特性。
"""

from .thruster import Thruster
from .thruster_cfg import ThrusterCfg
