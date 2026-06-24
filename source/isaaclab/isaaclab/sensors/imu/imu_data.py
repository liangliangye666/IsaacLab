# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ImuData:
    """Data container for the Imu sensor."""
    """为Imu传感器的数据容器。"""

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """
    """传感器在世界框架中的位置。

    形状是 (N， 3)，其中``N``是环境的数量。
    """

    quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion ``(w, x, y, z)`` in world frame.

    Shape is (N, 4), where ``N`` is the number of environments.
    """
    """传感器起源在世界框架中的四元数``(w， x， y， z)``的导向。

    形状是 (N， 4)，其中``N``是环境的数量。
    """

    projected_gravity_b: torch.Tensor = None
    """Gravity direction unit vector projected on the imu frame.

    Shape is (N,3), where ``N`` is the number of environments.
    """
    """引力方向单位向量投射在图像框架上。

    形状是 (N，3)，其中``N``是环境的数量。
    """

    lin_vel_b: torch.Tensor = None
    """IMU frame angular velocity relative to the world expressed in IMU frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """
    """IMU框架与世界相比的角速度，表达为IMU框架。

    形状是 (N， 3)，其中``N``是环境的数量。
    """

    ang_vel_b: torch.Tensor = None
    """IMU frame angular velocity relative to the world expressed in IMU frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """
    """IMU框架与世界相比的角速度，表达为IMU框架。

    形状是 (N， 3)，其中``N``是环境的数量。
    """

    lin_acc_b: torch.Tensor = None
    """IMU frame linear acceleration relative to the world expressed in IMU frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """
    """IMU框架线性加速与世界相比，表达为IMU框架。

    形状是 (N， 3)，其中``N``是环境的数量。
    """

    ang_acc_b: torch.Tensor = None
    """IMU frame angular acceleration relative to the world expressed in IMU frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """
    """IMU图形角度加速与世界相对，表达为IMU框架

    形状是 (N， 3)，其中``N``是环境的数量。
    """
