# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Data container for the multi-mesh ray-cast sensor."""
"""多网射线传感器的数据容器。"""

import torch

from .ray_caster_data import RayCasterData


class MultiMeshRayCasterData(RayCasterData):
    """Data container for the multi-mesh ray-cast sensor."""
    """多网射线传感器的数据容器。"""

    ray_mesh_ids: torch.Tensor = None
    """The mesh ids of the ray hits.

    Shape is (N, B, 1), where N is the number of sensors, B is the number of rays
    in the scan pattern per sensor.
    """
    """射线的网格标识。

    形状是 (N，B，1)，其中N是传感器的数量，B是每个传感器的扫描模式中的射线数量。
    """
