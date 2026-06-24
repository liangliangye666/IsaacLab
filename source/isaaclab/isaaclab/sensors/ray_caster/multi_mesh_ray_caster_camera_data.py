# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data container for the multi-mesh ray-cast camera sensor."""
"""多网射线摄像头传感器的数据容器。"""

import torch

from isaaclab.sensors.camera import CameraData

from .ray_caster_data import RayCasterData


class MultiMeshRayCasterCameraData(CameraData, RayCasterData):
    """Data container for the multi-mesh ray-cast sensor."""
    """多网射线传感器的数据容器。"""

    image_mesh_ids: torch.Tensor = None
    """The mesh ids of the image pixels.

    Shape is (N, H, W, 1), where N is the number of sensors, H and W are the height and width of the image,
    and 1 is the number of mesh ids per pixel.
    """
    """图像像像素的网格标识。

    形状是 (N，H，W，1)，其中N是传感器的数量，H和W是图像的高度和宽度，1是每像素的网格ID数量。
    """
