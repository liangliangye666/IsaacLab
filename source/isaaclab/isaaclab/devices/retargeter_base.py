# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass
class RetargeterCfg:
    """Base configuration for hand tracking retargeters."""
    """基本配置用于手动追踪回器。"""

    sim_device: str = "cpu"
    # Concrete retargeter class to construct for this config. Set by each retargeter module.
    retargeter_type: type["RetargeterBase"] | None = None


class RetargeterBase(ABC):
    """Base interface for input data retargeting.

    This abstract class defines the interface for components that transform
    raw device data into robot control commands. Implementations can handle
    various types of transformations including:
    - Hand joint data to end-effector poses
    - Input device commands to robot movements
    - Sensor data to control signals
    """
    """输入数据重定位的基接口。

    这种抽象类定义了将原始设备数据转化为机器人控制命令的组件的界面。
    实施可以处理各种类型的转型，包括:
    - 双手关节数据到末端执行器姿势
    - 输入设备对机器人运动的命令
    - 控制信号的传感器数据
    """

    def __init__(self, cfg: RetargeterCfg):
        """Initialize the retargeter.

        Args:
            cfg: Configuration for the retargeter
        """
        """启动重定位器。

        参数：
            cfg: 针对重定位器的配置
        """
        self._sim_device = cfg.sim_device

    class Requirement(Enum):
        """Features a retargeter may require from a device's raw data feed."""
        """一个重定位器可能需要从设备的原始数据源中获得的功能。"""

        HAND_TRACKING = "hand_tracking"
        HEAD_TRACKING = "head_tracking"
        MOTION_CONTROLLER = "motion_controller"

    @abstractmethod
    def retarget(self, data: Any) -> Any:
        """Retarget input data to desired output format.

        Args:
            data: Raw input data to be transformed

        Returns:
            Retargeted data in implementation-specific format
        """
        """将输入数据重定向到所需输出格式。

        参数：
            data: 要转换的原始输入数据

        返回：
            实现特定格式的重定位数据
        """
        pass

    def get_requirements(self) -> list["RetargeterBase.Requirement"]:
        """Return the list of required data features for this retargeter.

        Defaults to requesting all available features for backward compatibility.
        Implementations should override to narrow to only what they need.
        """
        """返回该重定位器所需数据功能列表。

        要求所有可用功能进行后退兼容的默认问题。
        实施措施应仅限于所需。
        """
        return [
            RetargeterBase.Requirement.HAND_TRACKING,
            RetargeterBase.Requirement.HEAD_TRACKING,
            RetargeterBase.Requirement.MOTION_CONTROLLER,
        ]
