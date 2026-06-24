# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils import configclass

from . import visual_materials


@configclass
class VisualMaterialCfg:
    """Configuration parameters for creating a visual material."""
    """创建视觉材料的配置参数"""

    func: Callable = MISSING
    """The function to use for creating the material."""
    """用于创建材料的功能。"""


@configclass
class PreviewSurfaceCfg(VisualMaterialCfg):
    """Configuration parameters for creating a preview surface.

    See :meth:`spawn_preview_surface` for more information.
    """
    """设置设置参数

    See :麻:`spawn_preview_surface`更多信息。
    """

    func: Callable = visual_materials.spawn_preview_surface

    diffuse_color: tuple[float, float, float] = (0.18, 0.18, 0.18)
    """The RGB diffusion color. This is the base color of the surface. Defaults to a dark gray."""
    """在RGB扩散颜色。
    这就是表面的基本颜色。
    默认调整为深灰色。
    """
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The RGB emission component of the surface. Defaults to black."""
    """表面的RGB排放成分。
    默认的黑色。
    """
    roughness: float = 0.5
    """The roughness for specular lobe. Ranges from 0 (smooth) to 1 (rough). Defaults to 0.5."""
    """镜状叶片的粗度。
    范围从0 (柔顺) 到1 (粗)。
    默认为0.5。
    """
    metallic: float = 0.0
    """The metallic component. Ranges from 0 (dielectric) to 1 (metal). Defaults to 0."""
    """金属成分。
    范围从0 (不电) 到1 (金属)。
    默认为0。
    """
    opacity: float = 1.0
    """The opacity of the surface. Ranges from 0 (transparent) to 1 (opaque). Defaults to 1.

    Note:
        Opacity only affects the surface's appearance during interactive rendering.
    """
    """表面的度。
    从0 (透明) 到1 (不透明)
    默认的1。

    说明：
        在交互式渲染过程中，光率只影响表面的外观。
    """


@configclass
class MdlFileCfg(VisualMaterialCfg):
    """Configuration parameters for loading an MDL material from a file.

    See :meth:`spawn_from_mdl_file` for more information.
    """
    """从文件中载入MDL材料的配置参数。

    See :麻:`spawn_from_mdl_file`更多信息。
    """

    func: Callable = visual_materials.spawn_from_mdl_file

    mdl_path: str = MISSING
    """The path to the MDL material.

    NVIDIA Omniverse provides various MDL materials in the NVIDIA Nucleus.
    To use these materials, you can set the path of the material in the nucleus directory
    using the ``{NVIDIA_NUCLEUS_DIR}`` variable. This is internally resolved to the path of the
    NVIDIA Nucleus directory on the host machine through the attribute
    :attr:`isaaclab.utils.assets.NVIDIA_NUCLEUS_DIR`.

    For example, to use the "Aluminum_Anodized" material, you can set the path to:
    ``{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl``.
    """
    """进入MDL材料的路径。

    NVIDIA全球提供各种MDL在NVIDIA核子。
    为了使用这些材料，你可以使用``{NVIDIA_NUCLEUS_DIR}``变量设置核目录中的材料的路径。
    通过属性 :attr:`isaaclab.utils.assets.NVIDIA_NUCLEUS_DIR`，将此内部解决在主机上的NVIDIA核目录的路径。

    例如，使用"An_化"材料，可以设置路径为:``{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl``。
    """
    project_uvw: bool | None = None
    """Whether to project the UVW coordinates of the material. Defaults to None.

    If None, then the default setting in the MDL material will be used.
    """
    """是否投射材料的UVW坐标。
    默认为 None。

    如果 None，则将使用MDL材料中的默认设置。
    """
    albedo_brightness: float | None = None
    """Multiplier for the diffuse color of the material. Defaults to None.

    If None, then the default setting in the MDL material will be used.
    """
    """对材料的散颜色的乘法。
    默认为 None。

    如果 None，则将使用MDL材料中的默认设置。
    """
    texture_scale: tuple[float, float] | None = None
    """The scale of the texture. Defaults to None.

    If None, then the default setting in the MDL material will be used.
    """
    """质量的尺度。
    默认为 None。

    如果 None，则将使用MDL材料中的默认设置。
    """


@configclass
class GlassMdlCfg(VisualMaterialCfg):
    """Configuration parameters for loading a glass MDL material.

    This is a convenience class for loading a glass MDL material. For more information on
    glass materials, see the `documentation <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html#omniglass>`__.

    .. note::
        The default values are taken from the glass material in the NVIDIA Nucleus.
    """
    """装载玻璃MDL材料的配置参数

    这是一个方便的类型，用于加载玻璃MDL材料。
    更多关于玻璃材料的信息请见`documentation
    <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html#omniglass>`__。

    .. 说明::
        默认值从NVIDIA核中的玻璃材料中取出。
    """

    func: Callable = visual_materials.spawn_from_mdl_file

    mdl_path: str = "OmniGlass.mdl"
    """The path to the MDL material. Defaults to the glass material in the NVIDIA Nucleus."""
    """进入MDL材料的路径。
    在NVIDIA核中的玻璃材料的故障。
    """
    glass_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """The RGB color or tint of the glass. Defaults to white."""
    """玻璃的RGB颜色或色彩。
    默认的白色。
    """
    frosting_roughness: float = 0.0
    """The amount of reflectivity of the surface. Ranges from 0 (perfectly clear) to 1 (frosted).
    Defaults to 0."""
    """表面的反射性量
    从0 (完美清晰) 到1 (结)。
    默认为0。
    """
    thin_walled: bool = False
    """Whether to perform thin-walled refraction. Defaults to False."""
    """是否进行薄壁折断。
    默认为 False。
    """
    glass_ior: float = 1.491
    """The incidence of refraction to control how much light is bent when passing through the glass.
    Defaults to 1.491, which is the IOR of glass.
    """
    """在玻璃穿过时， 控制光线曲的折射率。
    默认情况下是1.491，这是玻璃的IOR。
    """
