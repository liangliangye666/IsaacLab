# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

from dataclasses import MISSING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import VISUO_TACTILE_SENSOR_MARKER_CFG
from isaaclab.sensors import SensorBaseCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from .visuotactile_sensor import VisuoTactileSensor

##
# GelSight Render Configuration
##


@configclass
class GelSightRenderCfg:
    """Configuration for GelSight sensor rendering parameters.

    This configuration defines the rendering parameters for example-based tactile image synthesis
    using the Taxim approach.

    Reference:
        Si, Z., & Yuan, W. (2022). Taxim: An example-based simulation model for GelSight
        tactile sensors. IEEE Robotics and Automation Letters, 7(2), 2361-2368.
        https://arxiv.org/abs/2109.04027

    Data Directory Structure:
        The sensor data should be organized in the following structure::

            base_data_path/
            └── sensor_data_dir_name/
                ├── bg.jpg              # Background image (required)
                ├── polycalib.npz       # Polynomial calibration data (required)
                └── real_bg.npy         # Real background data (optional)

    Example:
        Using predefined sensor configuration::

            from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

            from isaaclab_assets.sensors import GELSIGHT_R15_CFG

            sensor_cfg = VisuoTactileSensorCfg(render_cfg=GELSIGHT_R15_CFG)

        Using custom sensor data::

            custom_cfg = GelSightRenderCfg(
                base_data_path="/path/to/my/sensors",
                sensor_data_dir_name="my_custom_sensor",
                image_height=480,
                image_width=640,
                mm_per_pixel=0.05,
            )
    """
    """对GelSight传感器渲染参数的配置。

    这种配置定义了使用Taxim方法基于例子的触觉图像合成的渲染参数。

    Reference: 西，Z.，和元，W。
               (2022).
               塔克西姆:为GelSight触觉传感器的基于示例的仿真模型。
               IEEE机器人和自动化信件， 7(2)， 2361-2368。
        https://arxiv.org/abs/2109.04027

    数据目录结构:传感器数据应按照以下结构进行组织:

            base_data_path/ ── sensor_data_dir_name/ ── bg.jpg #背景图像 (必需) ── polycalib.npz #多项式校准数据 (必需)
            ── real_bg.npy #真实背景数据 (可选)

    示例：
        使用预定义传感器配置::

            from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

            from isaaclab_assets.sensors import GELSIGHT_R15_CFG

            sensor_cfg = VisuoTactileSensorCfg(render_cfg=GELSIGHT_R15_CFG)

        使用自定义传感器数据::

            custom_cfg = GelSightRenderCfg(
                base_data_path="/path/to/my/sensors",
                sensor_data_dir_name="my_custom_sensor",
                image_height=480,
                image_width=640,
                mm_per_pixel=0.05,
            )
    """

    base_data_path: str = f"{ISAACLAB_NUCLEUS_DIR}/TacSL"
    """Base path to the directory containing sensor calibration data. Defaults to
    Isaac Lab Nucleus directory at ``{ISAACLAB_NUCLEUS_DIR}/TacSL``.
    """
    """传感器校准数据的目录的基路。
    在``{ISAACLAB_NUCLEUS_DIR}/TacSL``中，Isaac Lab Nucleus目录的默认错误。
    """

    sensor_data_dir_name: str = MISSING
    """Directory name containing the sensor calibration and background data.

    This should be a relative path (directory name) inside the :attr:`base_data_path`.
    """
    """包含传感器校准和背景数据的目录名称。

    这应该是:attr:`base_data_path`内部的相对路径 (目录名称)。
    """

    background_path: str = "bg.jpg"
    """Filename of the background image within the data directory."""
    """在数据目录中的背景图像的文件名。"""

    calib_path: str = "polycalib.npz"
    """Filename of the polynomial calibration data within the data directory."""
    """数据目录中的多项定位数据的文件名。"""

    real_background: str = "real_bg.npy"
    """Filename of the real background data within the data directory."""
    """在数据目录中的实际背景数据的文件名。"""

    image_height: int = MISSING
    """Height of the tactile image in pixels."""
    """触觉图像的高度在像素中。"""

    image_width: int = MISSING
    """Width of the tactile image in pixels."""
    """触觉图像的宽度在像素中。"""

    num_bins: int = 120
    """Number of bins for gradient magnitude and direction quantization."""
    """梯度大小和方向定量化的垃圾箱数量。"""

    mm_per_pixel: float = MISSING
    """Millimeters per pixel conversion factor for reconstructing 2D tactile image from the height map."""
    """在高度地图中重建2D触觉图像的每像素转换因子的毫米。"""


##
# Visuo-Tactile Sensor Configuration
##


@configclass
class VisuoTactileSensorCfg(SensorBaseCfg):
    """Configuration for the visuo-tactile sensor.

    This sensor provides both camera-based tactile sensing and force field tactile sensing.
    It can capture tactile RGB/depth images and compute penalty-based contact forces.
    """
    """视觉触觉传感器的配置

    这种传感器提供了基于相机的触觉传感器和力场触觉传感器。
    它可以捕获触觉RGB/深度图像，并计算基于惩罚的接触力。
    """

    class_type: type = VisuoTactileSensor

    # Sensor type and capabilities
    render_cfg: GelSightRenderCfg = MISSING
    """Configuration for GelSight sensor rendering.

    This defines the rendering parameters for converting depth maps to realistic tactile images.

    For simplicity, you can use the predefined configs for standard sensor models:

    - :attr:`isaaclab_assets.sensors.GELSIGHT_R15_CFG`
    - :attr:`isaaclab_assets.sensors.GELSIGHT_MINI_CFG`

    """
    """为GelSight传感器渲染配置。

    这定义了转换深度地图为现实触觉图像的渲染参数。

    为了简单化，您可以使用标准传感器模型的预定义配置:

    - :attr:`isaaclab_assets.sensors.GELSIGHT_R15_CFG`
    - :attr:`isaaclab_assets.sensors.GELSIGHT_MINI_CFG`
    """

    enable_camera_tactile: bool = True
    """Whether to enable camera-based tactile sensing."""
    """是否启用基于摄像头的触觉传感。"""

    enable_force_field: bool = True
    """Whether to enable force field tactile sensing."""
    """能否启用动力场触觉传感。"""

    # Force field configuration
    tactile_array_size: tuple[int, int] = MISSING
    """Number of tactile points for force field sensing in (rows, cols) format."""
    """在 (行， cols) 格式中感觉力场的触觉点数。"""

    tactile_margin: float = MISSING
    """Margin for tactile point generation (in meters).

    This parameter defines the exclusion margin from the edges of the elastomer mesh when generating
    the tactile point grid. It ensures that force field points are not generated on the very edges
    of the sensor surface where geometry might be unstable or less relevant for contact.
    """
    """触觉点生成的边缘 (在米)。

    在产生触点网时，该参数定义了弹性omer网边缘的排除边缘。
    它确保在传感器表面的边缘不会产生强力场点，其中几何可能不稳定或对接触不太重要。
    """

    contact_object_prim_path_expr: str | None = None
    """Prim path expression to find the contact object for force field computation.

    This specifies the object that will make contact with the tactile sensor. The sensor will automatically
    find the SDF collision mesh within this object for optimal force field computation.

    .. note::
        The expression can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/ContactObject`` will be replaced with ``/World/envs/env_.*/ContactObject``.

    .. attention::
        For force field computation to work properly, the contact object must have an SDF collision mesh.
        The sensor will search for the first SDF mesh within the specified prim hierarchy.
    """
    """为计算力场计算的接触对象的基本路径表达。

    这指明将与触觉传感器接触的对象。
    传感器将自动找到该物体内的SDF碰撞网格，以实现最佳的力场计算。

    .. 说明::
        这个表达式可以包含环境命名空间regex ``{ENV_REGEX_NS}``，将被环境命名空间取代。

        Example: ``{ENV_REGEX_NS}/ContactObject``将被 ``/World/envs/env_.*/ContactObject`` 取代。

    .. 注意::
        要使力场计算正常运行，接触物体必须具有SDF碰撞网。
        传感器将在指定的prim层次内搜索第一个SDF网格。
    """

    # Force field physics parameters
    normal_contact_stiffness: float = 1.0
    """Normal contact stiffness for penalty-based force computation."""
    """标准的接触硬度，用于基于惩罚的力计算。"""

    friction_coefficient: float = 2.0
    """Friction coefficient for shear forces."""
    """切割力摩擦系数"""

    tangential_stiffness: float = 0.1
    """Tangential stiffness for shear forces."""
    """切割力的固性。"""

    camera_cfg: TiledCameraCfg | None = None
    """Camera configuration for tactile RGB/depth sensing.

    If None, camera-based sensing will be disabled even if :attr:`enable_camera_tactile` is True.
    """
    """触觉RGB/深度传感的摄像头配置。

    如果None，即使:attr:`enable_camera_tactile`是True，相机传感器将被禁用。
    """

    # Visualization
    visualizer_cfg: VisualizationMarkersCfg = VISUO_TACTILE_SENSOR_MARKER_CFG.replace(
        prim_path="/Visuals/TactileSensor"
    )
    """The configuration object for the visualization markers.

    .. note::
        This attribute is only used when debug visualization is enabled.
    """
    """视觉化标记的配置对象。

    .. 说明::
        只有在启用调试可视化时才使用此属性。
    """

    trimesh_vis_tactile_points: bool = False
    """Whether to visualize tactile points for debugging using trimesh. Defaults to False."""
    """如何可视化使用trimesh的触觉点进行调试。
    默认为 False。
    """

    visualize_sdf_closest_pts: bool = False
    """Whether to visualize SDF closest points for debugging. Defaults to False."""
    """是否可查看 SDF 最接近的位置进行调试。
    默认为 False。
    """
