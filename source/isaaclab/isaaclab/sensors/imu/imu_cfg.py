# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .imu import Imu


@configclass
class ImuCfg(SensorBaseCfg):
    """Configuration for an Inertial Measurement Unit (IMU) sensor."""
    """动力测量单位 (IMU) 传感器的配置。"""

    class_type: type = Imu

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

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""
    """传感器框架的偏移姿势与传感器的母体框架。
    默认身份。
    """

    visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
    """The configuration object for the visualization markers. Defaults to RED_ARROW_X_MARKER_CFG.

    This attribute is only used when debug visualization is enabled.
    """
    """视觉化标记的配置对象。
    在 RED_ARROW_X_MARKER_CFG 中默认错误。

    只有在启用调试可视化时才使用此属性。
    """
    gravity_bias: tuple[float, float, float] = (0.0, 0.0, 9.81)
    """The linear acceleration bias applied to the linear acceleration in the world frame (x,y,z).

    Imu sensors typically output a positive gravity acceleration in opposition to the direction of gravity. This
    config parameter allows users to subtract that bias if set to (0.,0.,0.). By default this is set to (0.0,0.0,9.81)
    which results in a positive acceleration reading in the world Z.
    """
    """在世界框架中的线性加速偏差应用于线性加速 (x，y，z)。

    图像传感器通常会产生正重力加速，
    如果设置为 (0.，0.，0.)，这个配置参数允许用户减去这种偏差。
    默认设置为 (0.0，0.0，9.81) 结果在世界Z中产生正加速读数。
    """
