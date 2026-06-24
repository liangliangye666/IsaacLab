# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

import torch

from isaaclab.utils.math import convert_camera_frame_orientation_convention


@dataclass
class CameraData:
    """Data container for the camera sensor."""
    """摄像头传感器的数据容器。"""

    ##
    # Frame state.
    ##

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame, following ROS convention.

    Shape is (N, 3) where N is the number of sensors.
    """
    """传感器在世界框架中的位置，按照ROS规则。

    形状是 (N，3) ，其中N是传感器的数量。
    """

    quat_w_world: torch.Tensor = None
    """Quaternion orientation `(w, x, y, z)` of the sensor origin in world frame, following the world coordinate frame

    .. note::
        World frame convention follows the camera aligned with forward axis +X and up axis +Z.

    Shape is (N, 4) where N is the number of sensors.
    """
    """在世界框架中传感器起源的四元数导向`(w， x， y， z)`，遵循世界坐标框架

    .. 说明::
        世界框架公约遵循相机的前轴+X和上轴+Z。

    形状是 (N，4) ，其中N是传感器的数量。
    """

    ##
    # Camera data
    ##

    image_shape: tuple[int, int] = None
    """A tuple containing (height, width) of the camera sensor."""
    """包含摄像头传感器 (高度，宽度) 的图布。"""

    intrinsic_matrices: torch.Tensor = None
    """The intrinsic matrices for the camera.

    Shape is (N, 3, 3) where N is the number of sensors.
    """
    """摄像机的内在矩阵。

    形状是 (N，3，3) ，其中N是传感器的数量。
    """

    output: dict[str, torch.Tensor] = None
    """The retrieved sensor data with sensor types as key.

    The format of the data is available in the `Replicator Documentation`_. For semantic-based data,
    this corresponds to the ``"data"`` key in the output of the sensor.

    .. _Replicator Documentation: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/annotators_details.html#annotator-output
    """
    """获取的传感器数据，具有传感器类型的关键。

    数据的格式可用于`Replicator Documentation`_。
    对于基于语义的数据，这与传感器输出中的``"data"``键相符。

    .. _Replicator Documentation: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/annotators_details.html#annotator-output
    """

    info: list[dict[str, Any]] = None
    """The retrieved sensor info with sensor types as key.

    This contains extra information provided by the sensor such as semantic segmentation label mapping, prim paths.
    For semantic-based data, this corresponds to the ``"info"`` key in the output of the sensor. For other sensor
    types, the info is empty.
    """
    """获取的传感器信息，

    这包含传感器提供的额外信息，如语义细分标签映射，prim路径。
    对于基于语义的数据，这与传感器输出中的``"info"``键相符。
    对于其他传感器类型，信息是空的。
    """

    ##
    # Additional Frame orientation conventions
    ##

    @property
    def quat_w_ros(self) -> torch.Tensor:
        """Quaternion orientation `(w, x, y, z)` of the sensor origin in the world frame, following ROS convention.

        .. note::
            ROS convention follows the camera aligned with forward axis +Z and up axis -Y.

        Shape is (N, 4) where N is the number of sensors.
        """
        """在世界框架中的传感器起源的四元数方向`(w， x， y， z)`，按照ROS公约。

        .. 说明::
            随着ROS规则，相机与前轴+Z和上轴 -Y相对齐。

        形状是 (N，4) ，其中N是传感器的数量。
        """
        return convert_camera_frame_orientation_convention(self.quat_w_world, origin="world", target="ros")

    @property
    def quat_w_opengl(self) -> torch.Tensor:
        """Quaternion orientation `(w, x, y, z)` of the sensor origin in the world frame, following
        Opengl / USD Camera convention.

        .. note::
            OpenGL convention follows the camera aligned with forward axis -Z and up axis +Y.

        Shape is (N, 4) where N is the number of sensors.
        """
        """在世界框架中的传感器起源的四元数方向`(w， x， y， z)`，遵循Opengl/USD摄像头公约。

        .. 说明::
            OpenGL 规则遵循相机的前轴 - Z 和上轴 + Y 的排列。

        形状是 (N，4) ，其中N是传感器的数量。
        """
        return convert_camera_frame_orientation_convention(self.quat_w_world, origin="world", target="opengl")
