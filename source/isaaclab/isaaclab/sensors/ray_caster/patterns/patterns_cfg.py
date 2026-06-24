# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""

from __future__ import annotations
"""射线传感器的配置。"""

from collections.abc import Callable, Sequence
from dataclasses import MISSING
from typing import Literal

import torch

from isaaclab.utils import configclass

from . import patterns


@configclass
class PatternBaseCfg:
    """Base configuration for a pattern."""
    """为图案的基础配置。"""

    func: Callable[[PatternBaseCfg, str], tuple[torch.Tensor, torch.Tensor]] = MISSING
    """Function to generate the pattern.

    The function should take in the configuration and the device name as arguments. It should return
    the pattern's starting positions and directions as a tuple of torch.Tensor.
    """
    """函数生成模式。

    函数应将配置和设备名称作为参数。
    它应该作为torch.Tensor的图پل返回模式的起始位置和方向。
    """


@configclass
class GridPatternCfg(PatternBaseCfg):
    """Configuration for the grid pattern for ray-casting.

    Defines a 2D grid of rays in the coordinates of the sensor.

    .. attention::
        The points are ordered based on the :attr:`ordering` attribute.

    """
    """对于射线casting的网格格格的配置。

    定义传感器坐标中的2D射线网格。

    .. 注意::
        这些点是根据:attr:`ordering`属性进行排序的。
    """

    func: Callable = patterns.grid_pattern

    resolution: float = MISSING
    """Grid resolution (in meters)."""
    """电网分辨率 (以米)。"""

    size: tuple[float, float] = MISSING
    """Grid size (length, width) (in meters)."""
    """电网尺寸 (长度，宽度) (以米)。"""

    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Ray direction. Defaults to (0.0, 0.0, -1.0)."""
    """雷的方向。
    在 (0.0， 0.0， -1.0) 之前的默认值。
    """

    ordering: Literal["xy", "yx"] = "xy"
    """Specifies the ordering of points in the generated grid. Defaults to ``"xy"``.

    Consider a grid pattern with points at :math:`(x, y)` where :math:`x` and :math:`y` are the grid indices.
    The ordering of the points can be specified as "xy" or "yx". This determines the inner and outer loop order
    when iterating over the grid points.

    * If "xy" is selected, the points are ordered with inner loop over "x" and outer loop over "y".
    * If "yx" is selected, the points are ordered with inner loop over "y" and outer loop over "x".

    For example, the grid pattern points with :math:`X = (0, 1, 2)` and :math:`Y = (3, 4)`:

    * "xy" ordering: :math:`[(0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (2, 4)]`
    * "yx" ordering: :math:`[(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)]`
    """
    """指定生成的网格中的点排序。
    在``"xy"``上默认。

    考虑一个有点的网格图案:math:`(x， y)`，其中:math:`x`和:math:`y`是网格索引。
    点的排序可以指定为"xy"或"yx"。
    这决定了在网格点上代时的内部和外部循环顺序。

    * 如果选择"xy"，则分点由"x"上内部循环和"y"上外部循环排列。
    * 如果选出"yx"，则分点由"y"上内部循环和"x"上外部循环排列。

    例如，网格图案的点是:math:`X = (0， 1， 2)`和:math:`Y = (3， 4)`:

    * "xy"序列:`[(0， 3)， (1， 3)， (2， 3)， (1， 4)， (2， 4)， (2， 4)]`
    * "yx"顺序:`[(0， 3)， (0， 4)， (1， 3)， (1， 4)， (2， 3)， (2， 4)]`
    """


@configclass
class PinholeCameraPatternCfg(PatternBaseCfg):
    """Configuration for a pinhole camera depth image pattern for ray-casting.

    .. caution::
        Focal length as well as the aperture sizes and offsets are set as a tenth of the world unit. In our case, the
        world unit is meters, so all of these values are in cm. For more information, please check:
        https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html
    """
    """为射线投射的 camera孔摄像头深度图像模式的配置。

    .. 谨慎::
        焦点长度以及开口大小和偏移设定为世界单位的十分之一。
        在我们的例子中，世界单位是米，所以所有这些值都是cm。
        更多信息请查看:
        https://docs.omniverse.nvidia.com/材料和渲染/最新cameras.html
    """

    func: Callable = patterns.pinhole_camera_pattern

    focal_length: float = 24.0
    """Perspective focal length (in cm). Defaults to 24.0cm.

    Longer lens lengths narrower FOV, shorter lens lengths wider FOV.
    """
    """视角焦距 (在cm中)。
    在24厘米之前。

    更长的镜头长度较窄FOV，更短的镜头长度较宽FOV。
    """

    horizontal_aperture: float = 20.955
    """Horizontal aperture (in cm). Defaults to 20.955 cm.

    Emulates sensor/film width on a camera.

    Note:
        The default value is the horizontal aperture of a 35 mm spherical projector.
    """
    """水平开口 (厘米)。
    默认值为20955厘米。

    在相机上仿真传感器/片幅。

    说明：
        默认值是35mm圆形投影机的水平开口。
    """
    vertical_aperture: float | None = None
    r"""Vertical aperture (in cm). Defaults to None.

    Emulates sensor/film height on a camera. If None, then the vertical aperture is calculated based on the
    horizontal aperture and the aspect ratio of the image to maintain squared pixels. In this case, the vertical
    aperture is calculated as:

    .. math::
        \text{vertical aperture} = \text{horizontal aperture} \times \frac{\text{height}}{\text{width}}
    """
    """垂直开口 (厘米)。
    默认为 None。

    在相机上仿真传感器/电影高度。
    如果是None，则垂直开口是根据水平开口和图像的视角比计算的，以保持像素的平方。
    在这种情况下，垂直开口计算为:

    .. math::
        \text{vertical aperture} = \text{horizontal aperture} \times \frac{\text{height}}{\text{width}}
    """

    horizontal_aperture_offset: float = 0.0
    """Offsets Resolution/Film gate horizontally. Defaults to 0.0."""
    """Off平地抵消分辨率/电影门。
    默认为0.0。
    """

    vertical_aperture_offset: float = 0.0
    """Offsets Resolution/Film gate vertically. Defaults to 0.0."""
    """垂直抵消分辨率/电影门。
    默认为0.0。
    """

    width: int = MISSING
    """Width of the image (in pixels)."""
    """图像宽度 (在像素中)。"""

    height: int = MISSING
    """Height of the image (in pixels)."""
    """图像的高度 (在像素中)。"""

    @classmethod
    def from_intrinsic_matrix(
        cls,
        intrinsic_matrix: list[float],
        width: int,
        height: int,
        focal_length: float = 24.0,
    ) -> PinholeCameraPatternCfg:
        r"""Create a :class:`PinholeCameraPatternCfg` class instance from an intrinsic matrix.

        The intrinsic matrix is a 3x3 matrix that defines the mapping between the 3D world coordinates and
        the 2D image. The matrix is defined as:

        .. math::
            I_{cam} = \begin{bmatrix}
            f_x & 0 & c_x \\
            0 & f_y & c_y \\
            0 & 0 & 1
            \end{bmatrix},

        where :math:`f_x` and :math:`f_y` are the focal length along x and y direction, while
        :math:`c_x` and :math:`c_y` are the principle point offsets along x and y direction, respectively.

        Args:
            intrinsic_matrix: Intrinsic matrix of the camera in row-major format.
                The matrix is defined as [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]. Shape is (9,).
            width: Width of the image (in pixels).
            height: Height of the image (in pixels).
            focal_length: Focal length of the camera (in cm). Defaults to 24.0 cm.

        Returns:
            An instance of the :class:`PinholeCameraPatternCfg` class.
        """
        """从内在矩阵创建:class:`PinholeCameraPatternCfg`类实例。

        内在矩阵是一个3x3矩阵，它定义了3D世界坐标和2D图像之间的映射。
        矩阵定义为:

        .. math::
            I_{cam} = \begin{bmatrix}
            f_x & 0 & c_x \\
            0 & f_y & c_y \\
            0 & 0 & 1
            \end{bmatrix},

        where :数学:`f_x`和:math:`f_y`是沿 x和y方向的焦距，而
        :math:`c_x`和:数学:`c_y`是分别沿 x 和 y 方向的主要点偏移。

        参数：
            intrinsic_matrix: 摄像头的内在矩阵在线大格式。
                              矩阵定义为 [f_x， 0， c_x， 0， f_y， c_y， 0， 0， 1]。
                              形状是 (9，)。
            width: 图像宽度 (在像素中)。
            height: 图像的高度 (在像素中)。
            focal_length: 摄像机的焦点长度 (厘米)。
                          默认值为24.0厘米。

        返回：
            一个:class:`PinholeCameraPatternCfg`类的例子。
        """
        # extract parameters from matrix
        f_x = intrinsic_matrix[0]
        c_x = intrinsic_matrix[2]
        f_y = intrinsic_matrix[4]
        c_y = intrinsic_matrix[5]
        # resolve parameters for usd camera
        horizontal_aperture = width * focal_length / f_x
        vertical_aperture = height * focal_length / f_y
        horizontal_aperture_offset = (c_x - width / 2) / f_x
        vertical_aperture_offset = (c_y - height / 2) / f_y

        return cls(
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            horizontal_aperture_offset=horizontal_aperture_offset,
            vertical_aperture_offset=vertical_aperture_offset,
            width=width,
            height=height,
        )


@configclass
class BpearlPatternCfg(PatternBaseCfg):
    """Configuration for the Bpearl pattern for ray-casting."""
    """对于射线casting的BPearl模式的配置。"""

    func: Callable = patterns.bpearl_pattern

    horizontal_fov: float = 360.0
    """Horizontal field of view (in degrees). Defaults to 360.0."""
    """视野水平 (在度)。
    默认为360.0。
    """

    horizontal_res: float = 10.0
    """Horizontal resolution (in degrees). Defaults to 10.0."""
    """水平分辨率 (在度)。
    默认调到10.0。
    """

    # fmt: off
    vertical_ray_angles: Sequence[float] = [
        89.5, 86.6875, 83.875, 81.0625, 78.25, 75.4375, 72.625, 69.8125, 67.0, 64.1875, 61.375,
        58.5625, 55.75, 52.9375, 50.125, 47.3125, 44.5, 41.6875, 38.875, 36.0625, 33.25, 30.4375,
        27.625, 24.8125, 22, 19.1875, 16.375, 13.5625, 10.75, 7.9375, 5.125, 2.3125
    ]
    # fmt: on
    """Vertical ray angles (in degrees). Defaults to a list of 32 angles.

    Note:
        We manually set the vertical ray angles to match the Bpearl sensor. The ray-angles
        are not evenly spaced.
    """
    """垂直射线角 (在度)。
    在32个角的列表中默认设置。

    说明：
        我们手动设置垂直射线角度，
        射线角没有均的距离。
    """


@configclass
class LidarPatternCfg(PatternBaseCfg):
    """Configuration for the LiDAR pattern for ray-casting."""
    """对于射线casting的LiDAR模式的配置。"""

    func: Callable = patterns.lidar_pattern

    channels: int = MISSING
    """Number of Channels (Beams). Determines the vertical resolution of the LiDAR sensor."""
    """频道数量 (束)。
    确定LiDAR传感器的垂直分辨率。
    """

    vertical_fov_range: tuple[float, float] = MISSING
    """Vertical field of view range in degrees."""
    """垂直视野在度范围内。"""

    horizontal_fov_range: tuple[float, float] = MISSING
    """Horizontal field of view range in degrees."""
    """水平视野在度范围内。"""

    horizontal_res: float = MISSING
    """Horizontal resolution (in degrees)."""
    """水平分辨率 (在度)。"""
