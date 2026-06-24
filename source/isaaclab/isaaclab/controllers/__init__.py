# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for different controllers and motion-generators.

Controllers or motion generators are responsible for closed-loop tracking of a given command. The
controller can be a simple PID controller or a more complex controller such as impedance control
or inverse kinematics control. The controller is responsible for generating the desired joint-level
commands to be sent to the robot.
"""
"""对于不同控制器和运动生成器的子包装。

控制器或运动生成器负责对给定的命令进行闭环跟踪。
控制器可以是简单的PID控制器或更复杂的控制器，如阻力控制或逆动力控制。
控制器负责生成想要的联合级命令，将发送给机器人。
"""

from .differential_ik import DifferentialIKController
from .differential_ik_cfg import DifferentialIKControllerCfg
from .operational_space import OperationalSpaceController
from .operational_space_cfg import OperationalSpaceControllerCfg
