# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for externally contributed sensors.

This package provides specialized sensor classes for simulating externally contributed
sensors in Isaac Lab. These sensors are not part of the core Isaac Lab framework yet,
but are planned to be added in the future. They are contributed by the community to
extend the capabilities of Isaac Lab.

Following the categorization in :mod:`isaaclab.sensors` sub-package, the prim paths passed
to the sensor's configuration class are interpreted differently based on the sensor type.
The following table summarizes the interpretation of the prim paths for different sensor types:

+---------------------+---------------------------+---------------------------------------------------------------+
| Sensor Type         | Example Prim Path         | Pre-check                                                     |
+=====================+===========================+===============================================================+
| Visuo-Tactile Sensor| /World/robot/base         | Leaf exists and is a physics body (Rigid Body)                |
+---------------------+---------------------------+---------------------------------------------------------------+

"""
"""对于外部传感器的子包装。

本包提供了专门的传感器类，用于仿真艾萨克实验室的外部贡献传感器。
这些传感器尚未成为伊萨克实验室核心框架的一部分，但计划在未来加入。
社区为扩大艾萨克实验室的能力提供了这些资金。

在:mod:`isaaclab.sensors`子包中的分类之后，通过传感器配置类的prim路径根据传感器类型被不同的解释。
下表概述了对不同传感器类型的prim路径的解释:

+----------------------+----------------------------------------------------------------------------
----------+=========================================================================================
=============================================================
"""

from .tacsl_sensor import *
