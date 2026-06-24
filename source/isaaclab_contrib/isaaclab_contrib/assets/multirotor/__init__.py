# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for multirotor assets.

This module provides specialized classes for simulating multirotor vehicles (drones,
quadcopters, hexacopters, etc.) in Isaac Lab. It extends the base articulation
framework to support thrust-based control through individual rotor/propeller actuators.

Key Components:
    - :class:`Multirotor`: Asset class for multirotor vehicles with thruster control
    - :class:`MultirotorCfg`: Configuration class for multirotors
    - :class:`MultirotorData`: Data container for multirotor state information

Example:
    .. code-block:: python

        from isaaclab_contrib.assets import Multirotor, MultirotorCfg
        from isaaclab_contrib.actuators import ThrusterCfg
        import isaaclab.sim as sim_utils

        # Configure multirotor
        cfg = MultirotorCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path="path/to/quadcopter.usd"),
            actuators={
                "thrusters": ThrusterCfg(
                    thruster_names_expr=["rotor_[0-3]"],
                    thrust_range=(0.0, 10.0),
                )
            },
        )

        # Create multirotor instance
        multirotor = Multirotor(cfg)

.. seealso::
    - :mod:`isaaclab_contrib.actuators`: Thruster actuator models
    - :mod:`isaaclab_contrib.mdp.actions`: Thrust action terms for RL
"""
"""多机动产品子模块

该模块提供了以伊萨克实验室仿真多机动车辆 (无人机，四旋翼，六旋翼等) 的专业课程。
它扩大了基关节框架，通过单个旋转/螺旋驱动器支持基于推力的控制。

主要组成部分:
    - :class:`Multirotor`:具有推进控制的多动机车辆的资产类别
    - :class:`MultirotorCfg`:多轮机配置类
    - :class:`MultirotorData`:用于多轮机状态信息的数据容器

示例：
    .. code-block:: python

        from isaaclab_contrib.assets import Multirotor, MultirotorCfg
        from isaaclab_contrib.actuators import ThrusterCfg
        import isaaclab.sim as sim_utils

        # Configure multirotor
        cfg = MultirotorCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path="path/to/quadcopter.usd"),
            actuators={
                "thrusters": ThrusterCfg(
                    thruster_names_expr=["rotor_[0-3]"],
                    thrust_range=(0.0, 10.0),
                )
            },
        )

        # Create multirotor instance
        multirotor = Multirotor(cfg)

..
查看:
    - :mod:`isaaclab_contrib.actuators`:推动器驱动器模型
    - :mod:`isaaclab_contrib.mdp.actions`:RL的推力动作项
"""

from .multirotor import Multirotor
from .multirotor_cfg import MultirotorCfg
from .multirotor_data import MultirotorData
