# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn sensors in the simulation.

Currently, the following sensors are supported:

* Camera: A USD camera prim with settings for pinhole or fisheye projections.

"""
"""在仿真中产生传感器的产物器子模块。

目前支持以下传感器:

* 摄像头:一个USD摄像头prim，设置用于孔或鱼眼投影。
"""

from .sensors import spawn_camera
from .sensors_cfg import FisheyeCameraCfg, PinholeCameraCfg
