# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast camera sensor."""
"""射线摄像头传感器的配置。"""

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .patterns import PinholeCameraPatternCfg
from .ray_caster_camera import RayCasterCamera
from .ray_caster_cfg import RayCasterCfg


@configclass
class RayCasterCameraCfg(RayCasterCfg):
    """Configuration for the ray-cast sensor."""
    """射线传感器的配置。"""

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""
        """传感器框架的偏移姿势与传感器的母体框架。"""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        """翻译w.r.t
        它们的母体。
        在 (0.0，0.0，0.0) 之前的默认设置。
        """

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""
        """四元数旋转 (w，x，y，z) w.r.t。
        它们的母体。
        在 (1.0，0.0，0.0，0.0，0.0的默认情况下。
        """

        convention: Literal["opengl", "ros", "world"] = "ros"
        """The convention in which the frame offset is applied. Defaults to "ros".

        - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y`` - Offset is applied in the OpenGL (Usd.Camera)
          convention.
        - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y`` - Offset is applied in the ROS convention.
        - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z`` - Offset is applied in the World Frame convention.

        """
        """框架抵消的公约
        默认的"ros"。

        - 在OpenGL (Usd.Camera) 公约中，应用``"opengl"`` - 前轴:``-Z`` - 上轴:``+Y`` - 抵消。
        - 在ROS公约中，``"ros"`` - 前轴:``+Z`` - 上轴:``-Y`` - 抵消是应用的。
        - 在"世界框架"公约中，应用``"world"`` - 前轴:``+X`` - 上轴:``+Z`` - 抵消。
        """

    class_type: type = RayCasterCamera

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""
    """传感器框架的偏移姿势与传感器的母体框架。
    默认身份。
    """

    data_types: list[str] = ["distance_to_image_plane"]
    """List of sensor names/types to enable for the camera. Defaults to ["distance_to_image_plane"]."""
    """传感器名字/类型列表
    在 ["distance_to_image_plane"上默认的设置。
    """

    depth_clipping_behavior: Literal["max", "zero", "none"] = "none"
    """Clipping behavior for the camera for values exceed the maximum value. Defaults to "none".

    - ``"max"``: Values are clipped to the maximum value.
    - ``"zero"``: Values are clipped to zero.
    - ``"none``: No clipping is applied. Values will be returned as ``inf`` for ``distance_to_camera`` and ``nan``
      for ``distance_to_image_plane`` data type.
    """
    """摄像机的裁剪行为，以查取值超过最大值。
    默认调整为"没有"。

    - ``"max"``:值被裁剪到最大值。
    - ``"zero"``:值被切断到零。
    - ``"none``:没有裁剪.为``distance_to_camera``和``nan``的值将被返回为``inf``
      for ``distance_to_image_plane`` data type.
    """

    pattern_cfg: PinholeCameraPatternCfg = MISSING
    """The pattern that defines the local ray starting positions and directions in a pinhole camera pattern."""
    """在 camera孔摄像头图案中定义了本地射线起始位置和方向的模式。"""

    def __post_init__(self):
        # for cameras, this quantity should be False always.
        self.ray_alignment = "base"
