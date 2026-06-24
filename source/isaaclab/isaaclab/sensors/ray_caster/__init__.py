# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for Warp-based ray-cast sensor.

The sub-module contains two implementations of the ray-cast sensor:

- :class:`isaaclab.sensors.ray_caster.RayCaster`: A basic ray-cast sensor that can be used to ray-cast against a single mesh.
- :class:`isaaclab.sensors.ray_caster.MultiMeshRayCaster`: A multi-mesh ray-cast sensor that can be used to ray-cast against
  multiple meshes. For these meshes, it tracks their transformations and updates the warp meshes accordingly.

Corresponding camera implementations are also provided for each of the sensor implementations. Internally, they perform
the same ray-casting operations as the sensor implementations, but return the results as images.
"""
"""基于光射线传感器的子模块。

该子模块包含射线传感器的两个实现:

- :class:`isaaclab.sensors.ray_caster.RayCaster`:一种基本射线传感器，可用于射线与单个网格。
- :class:`isaaclab.sensors.ray_caster.MultiMeshRayCaster`:一个多网射线射线传感器，可用于射线射线与多个网.对于这些网，它跟踪他们的变化，并相应
  地更新扭曲网。

对于每个传感器实现，也提供相应的摄像头实现。
在内部，它们执行与传感器实现相同的射线运行，但返回结果作为图像。
"""

from . import patterns
from .multi_mesh_ray_caster import MultiMeshRayCaster
from .multi_mesh_ray_caster_camera import MultiMeshRayCasterCamera
from .multi_mesh_ray_caster_camera_cfg import MultiMeshRayCasterCameraCfg
from .multi_mesh_ray_caster_camera_data import MultiMeshRayCasterCameraData
from .multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg
from .multi_mesh_ray_caster_data import MultiMeshRayCasterData
from .ray_caster import RayCaster
from .ray_caster_camera import RayCasterCamera
from .ray_caster_camera_cfg import RayCasterCameraCfg
from .ray_caster_cfg import RayCasterCfg
from .ray_caster_data import RayCasterData
