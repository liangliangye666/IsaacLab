# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass

from . import lights


@configclass
class LightCfg(SpawnerCfg):
    """Configuration parameters for creating a light in the scene.

    Please refer to the documentation on `USD LuxLight <https://openusd.org/dev/api/class_usd_lux_light_a_p_i.html>`_
    for more information.

    .. note::
        The default values for the attributes are those specified in the their official documentation.
    """
    """在场景中创建灯光的配置参数。

    请参阅有关`USD LuxLight <https://openusd.org/dev/api/class_usd_lux_light_a_p_i.html>`_
    for more information.

    .. 说明::
        对属性的默认值是其官方文档中所指定的值。
    """

    func: Callable = lights.spawn_light

    prim_type: str = MISSING
    """The prim type name for the light prim."""
    """轻光prim的prim类型名称。"""

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """The color of emitted light, in energy-linear terms. Defaults to white."""
    """在能量线性方面，发射光的颜色。
    默认的白色。
    """

    enable_color_temperature: bool = False
    """Enables color temperature. Defaults to false."""
    """允许颜色温度。
    默认到错误。
    """

    color_temperature: float = 6500.0
    """Color temperature (in Kelvin) representing the white point. The valid range is [1000, 10000]. Defaults to 6500K.

    The `color temperature <https://en.wikipedia.org/wiki/Color_temperature>`_ corresponds to the warmth
    or coolness of light. Warmer light has a lower color temperature, while cooler light has a higher
    color temperature.

    Note:
        It only takes effect when :attr:`enable_color_temperature` is true.
    """
    """颜色温度 (Kelvin) 表示白点。
    有效范围为 [1000， 10000]。
    默认情况下为6500K。

    在`color temperature <https://en.wikipedia.org/wiki/Color_temperature>`_ 与光的热度或冷度相符。
    温暖的光具有较低的颜色温度，而冷的光具有更高的颜色温度。

    说明：
        只有当:attr:`enable_color_temperature`是真的时才会生效。
    """

    normalize: bool = False
    """Normalizes power by the surface area of the light. Defaults to false.

    This makes it easier to independently adjust the power and shape of the light, by causing the power
    to not vary with the area or angular size of the light.
    """
    """通过光的表面面积来正常化功率。
    默认到错误。

    这使得更容易独立调整光线的功率和形状，
    """

    exposure: float = 0.0
    """Scales the power of the light exponentially as a power of 2. Defaults to 0.0.

    The result is multiplied against the intensity.
    """
    """按指数量计算光的功率为2。
    默认为0.0。

    结果与强度相反乘以。
    """

    intensity: float = 1.0
    """Scales the power of the light linearly. Defaults to 1.0."""
    """通过线性测量光的功率。
    默认到1.0。
    """


@configclass
class DiskLightCfg(LightCfg):
    """Configuration parameters for creating a disk light in the scene.

    A disk light is a light source that emits light from a disk. It is useful for simulating
    fluorescent lights. For more information, please refer to the documentation on
    `USDLux DiskLight <https://openusd.org/dev/api/class_usd_lux_disk_light.html>`_.

    .. note::
        The default values for the attributes are those specified in the their official documentation.
    """
    """在场景中创建磁盘灯的配置参数。

    磁盘光是从磁盘发出的光源。
    它用于仿真光灯。
    更多信息请参阅有关`USDLux DiskLight <https://openusd.org/dev/api/class_usd_lux_disk_light.html>`_。

    .. 说明::
        对属性的默认值是其官方文档中所指定的值。
    """

    prim_type = "DiskLight"

    radius: float = 0.5
    """Radius of the disk (in m). Defaults to 0.5m."""
    """磁盘半径 (m)。
    默认到0.5m。
    """


@configclass
class DistantLightCfg(LightCfg):
    """Configuration parameters for creating a distant light in the scene.

    A distant light is a light source that is infinitely far away, and emits parallel rays of light.
    It is useful for simulating sun/moon light. For more information, please refer to the documentation on
    `USDLux DistantLight <https://openusd.org/dev/api/class_usd_lux_distant_light.html>`_.

    .. note::
        The default values for the attributes are those specified in the their official documentation.
    """
    """设置参数用于在场景中产生遥远的光线。

    一个遥远的光是无限遥远的光源，
    它用于仿真太阳/月光。
    更多信息请参阅有关`USDLux DistantLight <https://openusd.org/dev/api/class_usd_lux_distant_light.html>`_。

    .. 说明::
        对属性的默认值是其官方文档中所指定的值。
    """

    prim_type = "DistantLight"

    angle: float = 0.53
    """Angular size of the light (in degrees). Defaults to 0.53 degrees.

    As an example, the Sun is approximately 0.53 degrees as seen from Earth.
    Higher values broaden the light and therefore soften shadow edges.
    """
    """光的角大小 (在度)。
    默认状态为0.53度。

    例如，从地球看，太阳大约是0.53度。
    较高的值使光线变宽，因此使阴影边缘变柔。
    """


@configclass
class DomeLightCfg(LightCfg):
    """Configuration parameters for creating a dome light in the scene.

    A dome light is a light source that emits light inwards from all directions. It is also possible to
    attach a texture to the dome light, which will be used to emit light. For more information, please refer
    to the documentation on `USDLux DomeLight <https://openusd.org/dev/api/class_usd_lux_dome_light.html>`_.

    .. note::
        The default values for the attributes are those specified in the their official documentation.
    """
    """在场景中创建圆顶灯的配置参数。

    圆顶光是从各个方向向内发射光的光源。
    也可以将纹理粘合到顶光，用于发射光。
    更多信息请参阅有关`USDLux DomeLight <https://openusd.org/dev/api/class_usd_lux_dome_light.html>`_。

    .. 说明::
        对属性的默认值是其官方文档中所指定的值。
    """

    prim_type = "DomeLight"

    texture_file: str | None = None
    """A color texture to use on the dome, such as an HDR (high dynamic range) texture intended
    for IBL (image based lighting). Defaults to None.

    If None, the dome will emit a uniform color.
    """
    """用于顶的颜色纹理，例如HDR (高动态范围) 纹理
    for IBL (image based lighting). Defaults to None.

    如果是None，顶将发射一个均的颜色。
    """

    texture_format: Literal["automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"] = "automatic"
    """The parametrization format of the color map file. Defaults to "automatic".

    Valid values are:

    * ``"automatic"``: Tries to determine the layout from the file itself. For example, Renderman texture files
      embed an explicit parameterization.
    * ``"latlong"``: Latitude as X, longitude as Y.
    * ``"mirroredBall"``: An image of the environment reflected in a sphere, using an implicitly orthogonal projection.
    * ``"angular"``: Similar to mirroredBall but the radial dimension is mapped linearly to the angle, providing better
      sampling at the edges.
    * ``"cubeMapVerticalCross"``: A cube map with faces laid out as a vertical cross.
    """
    """颜色地图文件的参数格式。
    默认的"自动"。

    有效值为:

    * ``"automatic"``:试图从文件本身确定布局.例如，Renderman纹理文件嵌入了明确的参数化。
    * ``"latlong"``:宽度为 X，长度为 Y。
    * ``"mirroredBall"``:在球体中反映的环境图像，使用隐含的直角投影。
    * ``"angular"``:类似于镜子球，但射线尺寸是线性地映射到角落，为边缘提供更好的样本。
    * ``"cubeMapVerticalCross"``:一个立方形地图，面部设为垂直的十字。
    """

    visible_in_primary_ray: bool = True
    """Whether the dome light is visible in the primary ray. Defaults to True.

    If true, the texture in the sky is visible, otherwise the sky is black.
    """
    """primary顶光是否可见于主要射线。
    默认为 True。

    如果是真的，天空的纹理是可见的，否则天空是黑色的。
    """


@configclass
class CylinderLightCfg(LightCfg):
    """Configuration parameters for creating a cylinder light in the scene.

    A cylinder light is a light source that emits light from a cylinder. It is useful for simulating
    fluorescent lights. For more information, please refer to the documentation on
    `USDLux CylinderLight <https://openusd.org/dev/api/class_usd_lux_cylinder_light.html>`_.

    .. note::
        The default values for the attributes are those specified in the their official documentation.
    """
    """在场景中创建 light灯的配置参数。

    一个光是从中发出光的光源。
    它用于仿真光灯。
    更多信息请参阅有关`USDLux CylinderLight <https://openusd.org/dev/api/class_usd_lux_cylinder_light.html>`_。

    .. 说明::
        对属性的默认值是其官方文档中所指定的值。
    """

    prim_type = "CylinderLight"

    length: float = 1.0
    """Length of the cylinder (in m). Defaults to 1.0m."""
    """的长度 (m)。
    默认到1.0m。
    """

    radius: float = 0.5
    """Radius of the cylinder (in m). Defaults to 0.5m."""
    """圆半径 (m)。
    默认到0.5m。
    """

    treat_as_line: bool = False
    """Treats the cylinder as a line source, i.e. a zero-radius cylinder. Defaults to false."""
    """处理为线源，i.e.为零射线。
    默认到错误。
    """


@configclass
class SphereLightCfg(LightCfg):
    """Configuration parameters for creating a sphere light in the scene.

    A sphere light is a light source that emits light outward from a sphere. For more information,
    please refer to the documentation on
    `USDLux SphereLight <https://openusd.org/dev/api/class_usd_lux_sphere_light.html>`_.

    .. note::
        The default values for the attributes are those specified in the their official documentation.
    """
    """在场景中创建球光的配置参数。

    球体光是从球体中发出的光源。
    更多信息请参阅有关`USDLux SphereLight <https://openusd.org/dev/api/class_usd_lux_sphere_light.html>`_。

    .. 说明::
        对属性的默认值是其官方文档中所指定的值。
    """

    prim_type = "SphereLight"

    radius: float = 0.5
    """Radius of the sphere. Defaults to 0.5m."""
    """球的半径。
    默认到0.5m。
    """

    treat_as_point: bool = False
    """Treats the sphere as a point source, i.e. a zero-radius sphere. Defaults to false."""
    """处理球体作为一个点源，i.e.是一个零射线球体。
    默认到错误。
    """
