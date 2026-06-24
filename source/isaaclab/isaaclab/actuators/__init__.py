# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for different actuator models.

Actuator models are used to model the behavior of the actuators in an articulation. These
are usually meant to be used in simulation to model different actuator dynamics and delays.

There are two main categories of actuator models that are supported:

- **Implicit**: Motor model with ideal PD from the physics engine. This is similar to having a continuous time
  PD controller. The motor model is implicit in the sense that the motor model is not explicitly defined by the user.
- **Explicit**: Motor models based on physical drive models.

  - **Physics-based**: Derives the motor models based on first-principles.
  - **Neural Network-based**: Learned motor models from actuator data.

Every actuator model inherits from the :class:`isaaclab.actuators.ActuatorBase` class,
which defines the common interface for all actuator models. The actuator models are handled
and called by the :class:`isaaclab.assets.Articulation` class.
"""
"""对于不同动机模型的子包装。

动机模型用于仿真动机在关节中的行为。
这些通常用于仿真模型中，以模型不同的动机动态和延迟。

支持的执行器模型主要有两类:

- **隐含**:由物理引擎的理想PD的发动机模型。 这类似于具有连续时间PD控制器。
- **明确**:基于物理驱动模型的发动机模型。

  - **基于物理**:基于第一原则的运动模型。
  - **基于神经网络**:从执行器数据中学习的运动模型。

每个动机模型都继承了:class:`isaaclab.actuators.ActuatorBase`类，这定义了所有动机模型的共同界面。
驱动器模型由:class:`isaaclab.assets.Articulation`类处理和调用。
"""

from .actuator_base import ActuatorBase
from .actuator_base_cfg import ActuatorBaseCfg
from .actuator_net import ActuatorNetLSTM, ActuatorNetMLP
from .actuator_net_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg
from .actuator_pd import DCMotor, DelayedPDActuator, IdealPDActuator, ImplicitActuator, RemotizedPDActuator
from .actuator_pd_cfg import (
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuatorCfg,
    ImplicitActuatorCfg,
    RemotizedPDActuatorCfg,
)
