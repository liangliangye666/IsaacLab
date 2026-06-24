# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package providing interfaces to different teleoperation devices.

Currently, the following categories of devices are supported:

* **Keyboard**: Standard keyboard with WASD and arrow keys.
* **Spacemouse**: 3D mouse with 6 degrees of freedom.
* **Gamepad**: Gamepad with 2D two joysticks and buttons. Example: Xbox controller.
* **OpenXR**: Uses hand tracking of index/thumb tip avg to drive the target pose. Gripping is done with pinching.
* **Haply**: Haptic device (Inverse3 + VerseGrip) with position, orientation tracking and force feedback.

All device interfaces inherit from the :class:`DeviceBase` class, which provides a
common interface for all devices. The device interface reads the input data when
the :meth:`DeviceBase.advance` method is called. It also provides the function :meth:`DeviceBase.add_callback`
to add user-defined callback functions to be called when a particular input is pressed from
the peripheral device.
"""
"""提供各种远程操作设备的接口的子包。

目前支持以下类型的设备:

* **键盘**:具有WASD和箭头键的标准键盘。
* 3D鼠标具有6度自由度。
* **Gamepad**:有2D两个玩具和按的游戏台。
* **OpenXR**:使用指标/指尖avg的手跟踪来驱动目标姿势.抓取是通过。
* **快乐**:具有位置，方向跟踪和力反的触觉装置 (反向3 + VerseGrip)。

所有设备接口都从:class:`DeviceBase`类中继承，该类为所有设备提供了共同的接口。
设备界面读取输入数据
the :这种方法叫做Meth:`DeviceBase.advance`。
     它还提供:meth:`DeviceBase.add_callback`函数
在外围设备中按一下特定输入时，添加用户定义的回调函数。
"""

from .device_base import DeviceBase, DeviceCfg, DevicesCfg
from .gamepad import Se2Gamepad, Se2GamepadCfg, Se3Gamepad, Se3GamepadCfg
from .haply import HaplyDevice, HaplyDeviceCfg
from .keyboard import Se2Keyboard, Se2KeyboardCfg, Se3Keyboard, Se3KeyboardCfg
from .openxr import ManusVive, ManusViveCfg, OpenXRDevice, OpenXRDeviceCfg
from .retargeter_base import RetargeterBase, RetargeterCfg
from .spacemouse import Se2SpaceMouse, Se2SpaceMouseCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from .teleop_device_factory import create_teleop_device
