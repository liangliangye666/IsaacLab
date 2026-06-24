# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

import torch


@dataclass
class RayCasterData:
    """Data container for the ray-cast sensor."""
    """射线传感器的数据容器。"""

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where N is the number of sensors.
    """
    """传感器在世界框架中的位置。

    形状是 (N， 3)，其中N是传感器的数量。
    """
    quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame.

    Shape is (N, 4), where N is the number of sensors.
    """
    """传感器起源在世界框架中的四元数 (w， x， y， z) 的导向。

    形状是 (N， 4)，其中N是传感器的数量。
    """
    ray_hits_w: torch.Tensor = None
    """The ray hit positions in the world frame.

    Shape is (N, B, 3), where N is the number of sensors, B is the number of rays
    in the scan pattern per sensor.
    """
    """射线在世界框架中的位置。

    形状是 (N，B，3)，其中N是传感器的数量，B是每个传感器的扫描模式中的射线数量。
    """
