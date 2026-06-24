# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(3) control."""

from __future__ import annotations
"""控制器 SE(3) 控制器。"""

import weakref
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import carb
import omni

from ..device_base import DeviceBase, DeviceCfg


class Se3Keyboard(DeviceBase):
    """A keyboard controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a keyboard controller for a robotic arm with a gripper.
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle gripper (open/close)    K
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Move along z-axis              Q                 E
        Rotate along x-axis            Z                 X
        Rotate along y-axis            T                 G
        Rotate along z-axis            C                 V
        ============================== ================= =================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """
    """键盘控制器用于发送SE(3) 命令作为三角形姿势和二进制命令 (开/关)。

    这类课程旨在为具有抓住器的机器人手臂提供键盘控制器。
    它使用"万能键盘接口"来听到键盘事件并将它们映射到机器人的任务空间命令中。

    命令包括两个部分:

    * 角定位:在米和半径中 (x，y，z，roll，pitch， yaw) 的六维向量。
    * 抓住器:是打开或关闭抓住器的二进制命令。

    关键约束:
    ====================================================================================================
    ====================================================================================================
    ====================================================================================================
    ====================================================================================================
    ====================================================================================================
    ==========

    ..
    查看:

        键盘接口的官方文档:`Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programme
        r_ref/input-devices/keyboard.html>`__。
    """

    def __init__(self, cfg: Se3KeyboardCfg):
        """Initialize the keyboard layer.

        Args:
            cfg: Configuration object for keyboard settings.
        """
        """启动键盘层。

        参数：
            cfg: 键盘设置配置对象
        """
        # store inputs
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.gripper_term = cfg.gripper_term
        self._sim_device = cfg.sim_device
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        """释放键盘接口。"""
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        """Returns: 包含玩具信息的字符串。"""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): K\n"
        msg += "\tMove arm along x-axis: W/S\n"
        msg += "\tMove arm along y-axis: A/D\n"
        msg += "\tMove arm along z-axis: Q/E\n"
        msg += "\tRotate arm along x-axis: Z/X\n"
        msg += "\tRotate arm along y-axis: T/G\n"
        msg += "\tRotate arm along z-axis: C/V"
        return msg

    """
    Operations
    """
    """运营
    """

    def reset(self):
        # default flags
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        """添加额外的功能来绑定键盘。

        提供可用的键列表`carb documentation
        <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__。

        参数：
            key: 键盘检查。
            func: 在键时调用的函数。
                  召回函数不应进行任何争论。
        """
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Provides the result from keyboard event state.

        Returns:
            torch.Tensor: A 7-element tensor containing:
                - delta pose: First 6 elements as [x, y, z, rx, ry, rz] in meters and radians.
                - gripper command: Last element as a binary value (+1.0 for open, -1.0 for close).
        """
        """提供键盘事件状态的结果。

        返回：
            torch.Tensor: 一个含有:
                - 德尔塔姿势:第6个元素以米和半径为 [x，y，z，rx， ry，rz]。
                - 抓住器命令:最后一个元素作为二进制值 (+1.0为开放，-1.0为关闭)。
        """
        # convert to rotation vector
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        # return the command and gripper state
        command = np.concatenate([self._delta_pos, rot_vec])
        if self.gripper_term:
            gripper_value = -1.0 if self._close_gripper else 1.0
            command = np.append(command, gripper_value)

        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    """
    Internal helpers.
    """
    """内部助理。
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        """订阅者将在更新套件时回调。

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/最新programmer_ref/输入设备/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            if event.input.name == "K":
                self._close_gripper = not self._close_gripper
            elif event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot += self._INPUT_KEY_MAPPING[event.input.name]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot -= self._INPUT_KEY_MAPPING[event.input.name]
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        """创建默认键绑定。"""
        self._INPUT_KEY_MAPPING = {
            # toggle: gripper command
            "K": True,
            # x-axis (forward)
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # y-axis (left-right)
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # z-axis (up-down)
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            # roll (around x-axis)
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # pitch (around y-axis)
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # yaw (around z-axis)
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }


@dataclass
class Se3KeyboardCfg(DeviceCfg):
    """Configuration for SE3 keyboard devices."""
    """对于SE3键盘设备的配置。"""

    gripper_term: bool = True
    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8
    retargeters: None = None
    class_type: type[DeviceBase] = Se3Keyboard
