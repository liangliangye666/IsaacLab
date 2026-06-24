# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import isaaclab.utils.sensors as sensor_utils
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass

from . import sensors


@configclass
class PinholeCameraCfg(SpawnerCfg):
    """Configuration parameters for a USD camera prim with pinhole camera settings.

    For more information on the parameters, please refer to the `camera documentation <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html>`__.

    ..note ::
        Focal length as well as the aperture sizes and offsets are set as a tenth of the world unit. In our case, the
        world unit is Meter s.t. all of these values are set in cm.
    """
    """具有孔摄像头设置的USD摄像头prim的配置参数。

    更多有关参数的信息请参见`camera documentation
    <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html>`__。

    ...注:焦点长度以及开口大小和偏移设置为世界单位的十分之一。
    在我们的情况下，世界单位是Meter s.t。
    所有这些值均为cm。
    """

    func: Callable = sensors.spawn_camera

    projection_type: str = "pinhole"
    """Type of projection to use for the camera. Defaults to "pinhole".

    Note:
        Currently only "pinhole" is supported.
    """
    """用于相机的投影类型。
    默认的"pinhole"。

    说明：
        目前只支持"pin孔"。
    """

    clipping_range: tuple[float, float] = (0.01, 1e6)
    """Near and far clipping distances (in m). Defaults to (0.01, 1e6).

    The minimum clipping range will shift the camera forward by the specified distance. Don't set it too high to
    avoid issues for distance related data types (e.g., ``distance_to_image_plane``).
    """
    """接近和远的切断距离 (m)。
    在 (0.01， 1e6) 之前的默认设置。

    最低的裁剪范围将使相机以指定距离向前移动。
    不要设置太高，以避免与距离相关的数据类型 (e.g.，``distance_to_image_plane``) 的问题。
    """

    focal_length: float = 24.0
    """Perspective focal length (in cm). Defaults to 24.0cm.

    Longer lens lengths narrower FOV, shorter lens lengths wider FOV.
    """
    """视角焦距 (在cm中)。
    在24厘米之前。

    更长的镜头长度较窄FOV，更短的镜头长度较宽FOV。
    """

    focus_distance: float = 400.0
    """Distance from the camera to the focus plane (in m). Defaults to 400.0.

    The distance at which perfect sharpness is achieved.
    """
    """从相机到焦点平面的距离 (m)。
    默认为400.0。

    达到完美的度的距离。
    """

    f_stop: float = 0.0
    """Lens aperture. Defaults to 0.0, which turns off focusing.

    Controls Distance Blurring. Lower Numbers decrease focus range, larger numbers increase it.
    """
    """镜头开口。
    默认为0.0，这将关闭聚焦。

    控制距离模糊。
    较低的数字减少了焦点范围，较大的数字增加了它。
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
    r"""Vertical aperture (in mm). Defaults to None.

    Emulates sensor/film height on a camera. If None, then the vertical aperture is calculated based on the
    horizontal aperture and the aspect ratio of the image to maintain squared pixels. This is calculated as:

    .. math::
        \text{vertical aperture} = \text{horizontal aperture} \times \frac{\text{height}}{\text{width}}
    """
    """垂直开口 (mm)。
    默认为 None。

    在相机上仿真传感器/电影高度。
    如果是None，则垂直开口是根据水平开口和图像的视角比计算的，以保持像素的平方。
    这计算为:

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

    lock_camera: bool = True
    """Locks the camera in the Omniverse viewport. Defaults to True.

    If True, then the camera remains fixed at its configured transform. This is useful when wanting to view
    the camera output on the GUI and not accidentally moving the camera through the GUI interactions.
    """
    """锁定相机在全宇宙视角。
    默认为 True。

    如果是True，那么相机将保持在配置转换时固定。
    这在GUI上想要查看相机输出时有用，而不是意外地通过GUI交互移动相机。
    """

    @classmethod
    def from_intrinsic_matrix(
        cls,
        intrinsic_matrix: list[float],
        width: int,
        height: int,
        clipping_range: tuple[float, float] = (0.01, 1e6),
        focal_length: float | None = None,
        focus_distance: float = 400.0,
        f_stop: float = 0.0,
        projection_type: str = "pinhole",
        lock_camera: bool = True,
    ) -> PinholeCameraCfg:
        r"""Create a :class:`PinholeCameraCfg` class instance from an intrinsic matrix.

        The intrinsic matrix is a 3x3 matrix that defines the mapping between the 3D world coordinates and
        the 2D image. The matrix is defined as:

        .. math::
            I_{cam} = \begin{bmatrix}
            f_x & 0 & c_x \\
            0 & f_y & c_y \\
            0 & 0 & 1
            \\end{bmatrix},

        where :math:`f_x` and :math:`f_y` are the focal length along x and y direction, while :math:`c_x` and
        :math:`c_y` are the principle point offsets along x and y direction respectively.

        Args:
            intrinsic_matrix: Intrinsic matrix of the camera in row-major format.
                The matrix is defined as [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]. Shape is (9,).
            width: Width of the image (in pixels).
            height: Height of the image (in pixels).
            clipping_range: Near and far clipping distances (in m). Defaults to (0.01, 1e6).
            focal_length: Perspective focal length (in cm) used to calculate pixel size. Defaults to None. If None
                focal_length will be calculated 1 / width.
            focus_distance: Distance from the camera to the focus plane (in m). Defaults to 400.0 m.
            f_stop: Lens aperture. Defaults to 0.0, which turns off focusing.
            projection_type: Type of projection to use for the camera. Defaults to "pinhole".
            lock_camera: Locks the camera in the Omniverse viewport. Defaults to True.

        Returns:
            An instance of the :class:`PinholeCameraCfg` class.
        """
        """从内在矩阵创建:class:`PinholeCameraCfg`类实例。

        内在矩阵是一个3x3矩阵，它定义了3D世界坐标和2D图像之间的映射。
        矩阵定义为:

        .. math::
            I_{cam} = \begin{bmatrix}
            f_x & 0 & c_x \
            0 & f_y & c_y \
            0 & 0 & 1
            \end{bmatrix},

        where :数学:`f_x`和:math:`f_y`是沿 x和y方向的焦距，而:math:`c_x`和
        :math:`c_y`是分别沿 x 和 y 方向的主要点偏移。

        参数：
            intrinsic_matrix: 摄像头的内在矩阵在线大格式。
                              矩阵定义为 [f_x， 0， c_x， 0， f_y， c_y， 0， 0， 1]。
                              形状是 (9，)。
            width: 图像宽度 (在像素中)。
            height: 图像的高度 (在像素中)。
            clipping_range: 接近和远的切断距离 (m)。
                            在 (0.01， 1e6) 之前的默认设置。
            focal_length: 用于计算像素大小的视角焦距 (在cm)。
                          默认为 None。
                          如果 None focal_length将计算为 1/宽度。
            focus_distance: 从相机到焦点平面的距离 (m)。
                            默认值为400.0 m
            f_stop: 镜头开口。
                    默认为0.0，这将关闭聚焦。
            projection_type: 用于相机的投影类型。
                             默认的"pinhole"。
            lock_camera: 锁定相机在全宇宙视角。
                         默认为 True。

        返回：
            一个:class:`PinholeCameraCfg`类的例子。
        """
        # raise not implemented error is projection type is not pinhole
        if projection_type != "pinhole":
            raise NotImplementedError("Only pinhole projection type is supported.")

        usd_camera_params = sensor_utils.convert_camera_intrinsics_to_usd(
            intrinsic_matrix=intrinsic_matrix, height=height, width=width, focal_length=focal_length
        )

        return cls(
            projection_type=projection_type,
            clipping_range=clipping_range,
            focal_length=usd_camera_params["focal_length"],
            focus_distance=focus_distance,
            f_stop=f_stop,
            horizontal_aperture=usd_camera_params["horizontal_aperture"],
            vertical_aperture=usd_camera_params["vertical_aperture"],
            horizontal_aperture_offset=usd_camera_params["horizontal_aperture_offset"],
            vertical_aperture_offset=usd_camera_params["vertical_aperture_offset"],
            lock_camera=lock_camera,
        )


@configclass
class FisheyeCameraCfg(PinholeCameraCfg):
    """Configuration parameters for a USD camera prim with `fish-eye camera`_ settings.

    For more information on the parameters, please refer to the
    `camera documentation <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html#fisheye-properties>`__.

    .. note::
        The default values are taken from the `Replicator camera <https://docs.omniverse.nvidia.com/py/replicator/1.12.16/source/extensions/omni.replicator.core/docs/API.html#cameras>`__
        function.

    .. _fish-eye camera: https://en.wikipedia.org/wiki/Fisheye_lens
    """
    """设置`fish-eye camera`_设置的USD相机prim的配置参数。

    更多有关参数的信息请参见`camera documentation <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/
    cameras.html#fisheye-properties>`__。

    .. 说明::
        默认值从`Replicator camera <https://docs.omniverse.nvidia.com/py/replicator/1.12.16/source/extension
        s/omni.replicator.core/docs/API.html#cameras>`它们的功能。

    .. _fish-eye camera: https://en.wikipedia.org/wiki/Fisheye_lens
    """

    func: Callable = sensors.spawn_camera

    projection_type: Literal[
        "fisheyePolynomial",
        "fisheyeSpherical",
        "fisheyeKannalaBrandtK3",
        "fisheyeRadTanThinPrism",
        "omniDirectionalStereo",
    ] = "fisheyePolynomial"
    r"""Type of projection to use for the camera. Defaults to "fisheyePolynomial".

    Available options:

    - ``"fisheyePolynomial"``: Fisheye camera model with :math:`360^{\circ}` spherical projection.
    - ``"fisheyeSpherical"``: Fisheye camera model with :math:`360^{\circ}` full-frame projection.
    - ``"fisheyeKannalaBrandtK3"``: Fisheye camera model using the Kannala-Brandt K3 distortion model.
    - ``"fisheyeRadTanThinPrism"``: Fisheye camera model that combines radial and tangential distortions.
    - ``"omniDirectionalStereo"``: Fisheye camera model supporting :math:`360^{\circ}` stereoscopic imaging.
    """
    """用于相机的投影类型。
    默认的"鱼眼多项式"。

    可供选择:

    - ``"fisheyePolynomial"``:Fisheye摄像机模型:数学:`360^{\circ}`球状投影。
    - ``"fisheyeSpherical"``:Fisheye摄像头模型，具有:数学:`360^{\circ}`全投影。
    - ``"fisheyeKannalaBrandtK3"``:使用Kannala-Brandt K3扭曲模型的鱼眼相机模型。
    - ``"fisheyeRadTanThinPrism"``:Fisheye摄像头模型，结合射线和形扭曲。
    - ``"omniDirectionalStereo"``: 鱼眼相机模型支持:数学:`360^{\circ}stere立体成像。
    """

    fisheye_nominal_width: float = 1936.0
    """Nominal width of fisheye lens model (in pixels). Defaults to 1936.0."""
    """鱼眼镜模型的名义宽度 (在像素中)。
    默认调整到19360。
    """

    fisheye_nominal_height: float = 1216.0
    """Nominal height of fisheye lens model (in pixels). Defaults to 1216.0."""
    """鱼眼镜模型的名义高度 (在像素中)。
    在1216.0的默认状态。
    """

    fisheye_optical_centre_x: float = 970.94244
    """Horizontal optical centre position of fisheye lens model (in pixels). Defaults to 970.94244."""
    """鱼眼镜模型水平光学中心位置 (在像素中)。
    在 970.94244 上默认设置。
    """

    fisheye_optical_centre_y: float = 600.37482
    """Vertical optical centre position of fisheye lens model (in pixels). Defaults to 600.37482."""
    """鱼眼镜模型垂直光学中心位置 (在像素中)。
    在600.37482的默认状态下。
    """

    fisheye_max_fov: float = 200.0
    """Maximum field of view of fisheye lens model (in degrees). Defaults to 200.0 degrees."""
    """鱼眼镜模型的最大视野 (在度)。
    在200.0度之前。
    """

    fisheye_polynomial_a: float = 0.0
    """First component of fisheye polynomial. Defaults to 0.0."""
    """鱼眼多项数的第一个组成部分。
    默认为0.0。
    """

    fisheye_polynomial_b: float = 0.00245
    """Second component of fisheye polynomial. Defaults to 0.00245."""
    """鱼眼多项数的第二个组成部分。
    默认为0.00245。
    """

    fisheye_polynomial_c: float = 0.0
    """Third component of fisheye polynomial. Defaults to 0.0."""
    """鱼眼多项数的第三个组成部分。
    默认为0.0。
    """

    fisheye_polynomial_d: float = 0.0
    """Fourth component of fisheye polynomial. Defaults to 0.0."""
    """鱼眼多项数的第四个组成部分。
    默认为0.0。
    """

    fisheye_polynomial_e: float = 0.0
    """Fifth component of fisheye polynomial. Defaults to 0.0."""
    """鱼眼多项数的第五组件。
    默认为0.0。
    """

    fisheye_polynomial_f: float = 0.0
    """Sixth component of fisheye polynomial. Defaults to 0.0."""
    """鱼眼多项数的第六组件。
    默认为0.0。
    """
