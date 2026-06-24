# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spacemouse controller for SE(2) control."""

from __future__ import annotations
"""为SE ((2) 控制的空间鼠标控制器。"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import hid
import numpy as np
import torch

from isaaclab.utils.array import convert_to_torch

from ..device_base import DeviceBase, DeviceCfg
from .utils import convert_buffer


class Se2SpaceMouse(DeviceBase):
    r"""A space-mouse controller for sending SE(2) commands as delta poses.

    This class implements a space-mouse controller to provide commands to mobile base.
    It uses the `HID-API`_ which interfaces with USD and Bluetooth HID-class devices across multiple platforms.

    The command comprises of the base linear and angular velocity: :math:`(v_x, v_y, \omega_z)`.

    Note:
        The interface finds and uses the first supported device connected to the computer.

    Currently tested for following devices:

    - SpaceMouse Compact: https://3dconnexion.com/de/product/spacemouse-compact/

    .. _HID-API: https://github.com/libusb/hidapi

    """
    """一个空间鼠标控制器用于发送SE(2) 命令作为三角形姿势。

    这个类实现了一个空间鼠标控制器，以提供命令给移动基座。
    它使用`HID-API`_，它与USD和蓝牙HID类设备接口在多个平台上。

    命令包括基线和角速度:`(v_x， v_y， \omega_z)`。

    说明：
        接口找到和使用与计算机连接的第一个支持设备。

    目前用于以下设备进行测试:

    - SpaceMouse 紧: https://3dconnexion.com/de/product/spacemouse-compact/

    .. _HID-API: https://github.com/libusb/hidapi
    """

    def __init__(self, cfg: Se2SpaceMouseCfg):
        """Initialize the spacemouse layer.

        Args:
            cfg: Configuration for the spacemouse device.
        """
        """启动空间鼠标层。

        参数：
            cfg: 空间鼠标设备的配置。
        """
        # store inputs
        self.v_x_sensitivity = cfg.v_x_sensitivity
        self.v_y_sensitivity = cfg.v_y_sensitivity
        self.omega_z_sensitivity = cfg.omega_z_sensitivity
        self._sim_device = cfg.sim_device
        # acquire device interface
        self._device = hid.device()
        self._find_device()
        # command buffers
        self._base_command = np.zeros(3)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()
        # run a thread for listening to device updates
        self._thread = threading.Thread(target=self._run_device)
        self._thread.daemon = True
        self._thread.start()

    def __del__(self):
        """Destructor for the class."""
        """对于这个班级来说，它是降采样性的。"""
        self._thread.join()

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        """Returns: 包含玩具信息的字符串。"""
        msg = f"Spacemouse Controller for SE(2): {self.__class__.__name__}\n"
        msg += f"\tManufacturer: {self._device.get_manufacturer_string()}\n"
        msg += f"\tProduct: {self._device.get_product_string()}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tRight button: reset command\n"
        msg += "\tMove mouse laterally: move base horizontally in x-y plane\n"
        msg += "\tTwist mouse about z-axis: yaw base about a corresponding axis"
        return msg

    """
    Operations
    """
    """运营
    """

    def reset(self):
        # default flags
        self._base_command.fill(0.0)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind spacemouse.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        """增加额外的功能来绑定空间鼠标。

        参数：
            key: 键盘检查。
            func: 在键时调用的函数。
                  召回函数不应进行任何争论。
        """
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Provides the result from spacemouse event state.

        Returns:
            A 3D tensor containing the linear (x,y) and angular velocity (z).
        """
        """提供空间鼠事件状态的结果。

        返回：
            包含线性 (x，y) 和角速度 (z) 的3D子。
        """
        return convert_to_torch(self._base_command, device=self._sim_device)

    """
    Internal helpers.
    """
    """内部助理。
    """

    def _find_device(self):
        """Find the device connected to computer."""
        """找到连接到电脑的设备。"""
        found = False
        # implement a timeout for device search
        for _ in range(5):
            for device in hid.enumerate():
                if device["product_string"] == "SpaceMouse Compact":
                    # set found flag
                    found = True
                    vendor_id = device["vendor_id"]
                    product_id = device["product_id"]
                    # connect to the device
                    self._device.open(vendor_id, product_id)
            # check if device found
            if not found:
                time.sleep(1.0)
            else:
                break
        # no device found: return false
        if not found:
            raise OSError("No device found by SpaceMouse. Is the device connected?")

    def _run_device(self):
        """Listener thread that keeps pulling new messages."""
        """听者线索，不断吸引新的信息。"""
        # keep running
        while True:
            # read the device data
            data = self._device.read(13)
            if data is not None:
                # readings from 6-DoF sensor
                if data[0] == 1:
                    # along y-axis
                    self._base_command[1] = self.v_y_sensitivity * convert_buffer(data[1], data[2])
                    # along x-axis
                    self._base_command[0] = self.v_x_sensitivity * convert_buffer(data[3], data[4])
                elif data[0] == 2:
                    # along z-axis
                    self._base_command[2] = self.omega_z_sensitivity * convert_buffer(data[3], data[4])
                # readings from the side buttons
                elif data[0] == 3:
                    # press left button
                    if data[1] == 1:
                        # additional callbacks
                        if "L" in self._additional_callbacks:
                            self._additional_callbacks["L"]
                    # right button is for reset
                    if data[1] == 2:
                        # reset layer
                        self.reset()
                        # additional callbacks
                        if "R" in self._additional_callbacks:
                            self._additional_callbacks["R"]


@dataclass
class Se2SpaceMouseCfg(DeviceCfg):
    """Configuration for SE2 space mouse devices."""
    """设置SE2空间鼠标设备。"""

    v_x_sensitivity: float = 0.8
    v_y_sensitivity: float = 0.4
    omega_z_sensitivity: float = 1.0
    class_type: type[DeviceBase] = Se2SpaceMouse
