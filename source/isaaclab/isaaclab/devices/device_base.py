# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for teleoperation interface."""
"""电操作界面的基类。"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


@dataclass
class DeviceCfg:
    """Configuration for teleoperation devices."""
    """远程操作设备的配置。"""

    # Whether teleoperation should start active by default
    teleoperation_active_default: bool = True
    # Torch device string to place output tensors on
    sim_device: str = "cpu"
    # Retargeters that transform device data into robot commands
    retargeters: list[RetargeterCfg] = field(default_factory=list)
    # Concrete device class to construct for this config. Set by each device module.
    class_type: type["DeviceBase"] | None = None


@dataclass
class DevicesCfg:
    """Configuration for all supported teleoperation devices."""
    """所有支持的远程操作设备的配置。"""

    devices: dict[str, DeviceCfg] = field(default_factory=dict)


class DeviceBase(ABC):
    """An interface class for teleoperation devices.

    Derived classes have two implementation options:

    1. Override _get_raw_data() and use the base advance() implementation:
       This approach is suitable for devices that want to leverage the built-in
       retargeting logic but only need to customize the raw data acquisition.

    2. Override advance() completely:
       This approach gives full control over the command generation process,
       and _get_raw_data() can be ignored entirely.
    """
    """一个用于远程操作设备的接口类。

    衍生类有两个实现选项:

    1. 覆盖 _get_raw_data() 并使用基础推进() 实现:这种方法适合想要利用内置的重定位逻辑但只需要定制原始数据采集的设备。

    2. 彻底覆盖前进:这种方法可以完全控制命令生成过程，并且可以完全忽略 _get_raw_data。
    """

    def __init__(self, retargeters: list[RetargeterBase] | None = None):
        """Initialize the teleoperation interface.

        Args:
            retargeters: List of components that transform device data into robot commands.
                        If None or empty list, the device will output its native data format.
        """
        """启动远程操作接口。

        参数：
            retargeters: 将设备数据转化为机器人命令的组件列表。
                         如果 None或空格列表，设备将输出其原生数据格式。
        """
        # Initialize empty list if None is provided
        self._retargeters = retargeters or []
        # Aggregate required features across all retargeters
        self._required_features = set()
        for retargeter in self._retargeters:
            self._required_features.update(retargeter.get_requirements())

    def __str__(self) -> str:
        """Returns: A string identifier for the device."""
        """Returns: 设备的字符串识别器。"""
        return f"{self.__class__.__name__}"

    """
    Operations
    """
    """运营
    """

    @abstractmethod
    def reset(self):
        """Reset the internals."""
        """重置内部。"""
        raise NotImplementedError

    @abstractmethod
    def add_callback(self, key: Any, func: Callable):
        """Add additional functions to bind keyboard.

        Args:
            key: The button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        """添加额外的功能来绑定键盘。

        参数：
            key: 按检查。
            func: 在键时调用的函数。
                  召回函数不应进行任何争论。
        """
        raise NotImplementedError

    def _get_raw_data(self) -> Any:
        """Internal method to get the raw data from the device.

        This method is intended for internal use by the advance() implementation.
        Derived classes can override this method to customize raw data acquisition
        while still using the base class's advance() implementation.

        Returns:
            Raw device data in a device-specific format

        Note:
            This is an internal implementation detail. Clients should call advance()
            instead of this method.
        """
        """内部方法从设备中获取原始数据。

        这种方法是预先实施的内部使用。
        衍生类可以取代这种方法来定制原始数据采集
        while still using the base class's advance() implementation.

        返回：
            设备特定格式的原材料数据

        说明：
            这是一个内部实施细节。
            客户应使用此方法而不是预先。
        """
        raise NotImplementedError("Derived class must implement _get_raw_data() or override advance()")

    def advance(self) -> torch.Tensor:
        """Process current device state and return control commands.

        This method retrieves raw data from the device and optionally applies
        retargeting to convert it to robot commands.

        Derived classes can either:
        1. Override _get_raw_data() and use this base implementation, or
        2. Override this method completely for custom command processing

        Returns:
            When no retargeters are configured, returns raw device data in its native format.
            When retargeters are configured, returns a torch.Tensor containing the concatenated
            outputs from all retargeters.
        """
        """处理设备现状和返回控制命令。

        这种方法从设备中获取原始数据，并可选择地应用重定向，将其转换为机器人命令。

        衍生类可以:
        1. 删除_get_raw_data() 并使用此基础实现，或
        2. 完全覆盖这个方法，以处理定制命令

        返回：
            如果没有重定位器配置，则将原始设备数据返回原始格式。
            当重定向器配置时，返回包含所有重定向器的连接输出的torch.Tensor。
        """
        raw_data = self._get_raw_data()

        # If no retargeters, return raw data directly (not as a tuple)
        if not self._retargeters:
            return raw_data

        # With multiple retargeters, return a tuple of outputs
        # Concatenate retargeted outputs into a single tensor
        return torch.cat([retargeter.retarget(raw_data) for retargeter in self._retargeters], dim=-1)

    # -----------------------------
    # Shared data layout helpers (for retargeters across devices)
    # -----------------------------
    class TrackingTarget(Enum):
        """Standard tracking targets shared across devices."""
        """在设备中共享标准追踪目标。"""

        HAND_LEFT = 0
        HAND_RIGHT = 1
        HEAD = 2
        CONTROLLER_LEFT = 3
        CONTROLLER_RIGHT = 4

    class MotionControllerDataRowIndex(Enum):
        """Rows in the motion-controller 2x7 array."""
        """运动控制器2x7阵列中的行列。"""

        POSE = 0
        INPUTS = 1

    class MotionControllerInputIndex(Enum):
        """Indices in the motion-controller input row."""
        """在运动控制器输入行中的指标。"""

        THUMBSTICK_X = 0
        THUMBSTICK_Y = 1
        TRIGGER = 2
        SQUEEZE = 3
        BUTTON_0 = 4
        BUTTON_1 = 5
        PADDING = 6
