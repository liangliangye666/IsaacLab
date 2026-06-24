# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import MISSING
from typing import Literal

import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from ..sub_terrain_cfg import SubTerrainBaseCfg

"""
Different trimesh terrain configurations.
"""
"""不同的地形配置。
"""


@configclass
class MeshPlaneTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a plane mesh terrain."""
    """对平面网形地形的配置。"""

    function = mesh_terrains.flat_terrain


@configclass
class MeshPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain."""
    """为金字塔楼梯网格地形的配置。"""

    function = mesh_terrains.pyramid_stairs_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """
    """地形周边边界宽度 (m)。
    默认为0.0。

    边界是平坦的地形，与地形的高度相同。
    """

    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""
    """阶梯的最低和最高高度 (以m)。"""

    step_width: float = MISSING
    """The width of the steps (in m)."""
    """步骤宽度 (m)。"""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    """在地形中心的方形平台宽度。
    默认到1.0。
    """

    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.
    """
    """如果是True，地形会有梯子里的洞。
    默认为 False。

    If :`holes`是True，地形将具有长度或宽度的金字塔楼梯
    :obj:`platform_width` (取决于方向) 在剩余区域没有步骤。
    此外，不会增加任何边界。
    """


@configclass
class MeshInvertedPyramidStairsTerrainCfg(MeshPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stair mesh terrain.

    Note:
        This is the same as :class:`MeshPyramidStairsTerrainCfg` except that the steps are inverted.
    """
    """设置反向金字塔楼梯网格地形。

    说明：
        这与:class:`MeshPyramidStairsTerrainCfg`相同，但步骤倒车。
    """

    function = mesh_terrains.inverted_pyramid_stairs_terrain


@configclass
class MeshRandomGridTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a random grid mesh terrain."""
    """设置为随机网网地形。"""

    function = mesh_terrains.random_grid_terrain

    grid_width: float = MISSING
    """The width of the grid cells (in m)."""
    """网格电池的宽度 (m)。"""

    grid_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the grid cells (in m)."""
    """电网电池的最小和最大高度 (以m)。"""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    """在地形中心的方形平台宽度。
    默认到1.0。
    """

    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have randomized grid cells only along the plane extending
    from the platform (like a plus sign). The remaining area remains empty and no border will be added.
    """
    """如果是True，地形会有梯子里的洞。
    默认为 False。

    If :`holes`是True，地形将仅沿着延伸的平面随机格式细胞
    from the platform (like a plus sign). The remaining area remains empty and no border will be added.
    """


@configclass
class MeshRailsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with box rails as extrusions."""
    """设置一个地形，用 extr道作为挤出。"""

    function = mesh_terrains.rails_terrain

    rail_thickness_range: tuple[float, float] = MISSING
    """The thickness of the inner and outer rails (in m)."""
    """内部和外部轨道厚度 (m)。"""

    rail_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the rails (in m)."""
    """轨道的最低和最高高度 (以m)。"""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    """在地形中心的方形平台宽度。
    默认到1.0。
    """


@configclass
class MeshPitTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a pit that leads out of the pit."""
    """设置地形，有坑出坑。"""

    function = mesh_terrains.pit_terrain

    pit_depth_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the pit (in m)."""
    """坑的最小和最大高度 (在m)。"""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    """在地形中心的方形平台宽度。
    默认到1.0。
    """

    double_pit: bool = False
    """If True, the pit contains two levels of stairs. Defaults to False."""
    """如果True，坑里有两个层楼梯。
    默认为 False。
    """


@configclass
class MeshBoxTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with boxes (similar to a pyramid)."""
    """设置一个带盒的地形 (类似于金字塔)。"""

    function = mesh_terrains.box_terrain

    box_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the box (in m)."""
    """盒子的最小和最大高度 (以m)。"""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    """在地形中心的方形平台宽度。
    默认到1.0。
    """

    double_box: bool = False
    """If True, the pit contains two levels of stairs/boxes. Defaults to False."""
    """如果True，坑里有两个层楼梯/盒子。
    默认为 False。
    """


@configclass
class MeshGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a gap around the platform."""
    """在平台周围有空隙的地形配置。"""

    function = mesh_terrains.gap_terrain

    gap_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the gap (in m)."""
    """差距的最小和最大宽度 (以m)。"""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    """在地形中心的方形平台宽度。
    默认到1.0。
    """


@configclass
class MeshFloatingRingTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a floating ring around the center."""
    """设置一个带着环绕中心浮动的地形。"""

    function = mesh_terrains.floating_ring_terrain

    ring_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the ring (in m)."""
    """戒指的最小和最大宽度 (m)。"""

    ring_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the ring (in m)."""
    """戒指的最小和最大高度 (m)。"""

    ring_thickness: float = MISSING
    """The thickness (along z) of the ring (in m)."""
    """环的厚度 (z) (m)。"""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    """在地形中心的方形平台宽度。
    默认到1.0。
    """


@configclass
class MeshStarTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a star pattern."""
    """设置一个带有恒星图案的地形。"""

    function = mesh_terrains.star_terrain

    num_bars: int = MISSING
    """The number of bars per-side the star. Must be greater than 2."""
    """恒星的一边的杆数。
    必须大于2。
    """

    bar_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the bars in the star (in m)."""
    """恒星中的条的最小和最大宽度 (m)。"""

    bar_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the bars in the star (in m)."""
    """恒星中的条的最小和最大高度 (以m)。"""

    platform_width: float = 1.0
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""
    """在地形中心的圆柱形平台宽度。
    默认到1.0。
    """


@configclass
class MeshRepeatedObjectsTerrainCfg(SubTerrainBaseCfg):
    """Base configuration for a terrain with repeated objects."""
    """基于重复对象的地形的基准配置。"""

    @configclass
    class ObjectCfg:
        """Configuration of repeated objects."""
        """复制物体的配置。"""

        num_objects: int = MISSING
        """The number of objects to add to the terrain."""
        """增加到地形的物体数量。"""
        height: float = MISSING
        """The height (along z) of the object (in m)."""
        """对象的高度 (z) (m)。"""

    function = mesh_terrains.repeated_objects_terrain

    object_type: Literal["cylinder", "box", "cone"] | callable = MISSING
    """The type of object to generate.

    The type can be a string or a callable. If it is a string, the function will look for a function called
    ``make_{object_type}`` in the current module scope. If it is a callable, the function will
    use the callable to generate the object.
    """
    """产生的物体类型。

    这种类型可以是字符串或电话。
    如果是字符串，函数将在当前模块范围中寻找称为``make_{object_type}``的函数。
    如果它是一个可调用函数，函数将使用可调用函数来生成对象。
    """

    object_params_start: ObjectCfg = MISSING
    """The object curriculum parameters at the start of the curriculum."""
    """课程开始时的目标课程参数。"""

    object_params_end: ObjectCfg = MISSING
    """The object curriculum parameters at the end of the curriculum."""
    """在课程结束时的对象课程参数。"""

    max_height_noise: float | None = None
    """"This parameter is deprecated, but stated here to support backward compatibility"""
    """"这个参数已过时，但在这里表示支持后退兼容性"""

    abs_height_noise: tuple[float, float] = (0.0, 0.0)
    """The minimum and maximum amount of additive noise for the height of the objects. Default is set to 0.0,
    which is no noise.
    """
    """对物体高度的添加噪音最小和最大量。
    默认设置为0.0，这是没有噪音。
    """

    rel_height_noise: tuple[float, float] = (1.0, 1.0)
    """The minimum and maximum amount of multiplicative noise for the height of the objects. Default is set to 1.0,
    which is no noise.
    """
    """对物体的高度的最小和最大乘用噪音量。
    默认设置为1.0，这是没有噪音。
    """

    platform_width: float = 1.0
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""
    """在地形中心的圆柱形平台宽度。
    默认到1.0。
    """

    platform_height: float = -1.0
    """The height of the platform. Defaults to -1.0.

    If the value is negative, the height is the same as the object height.
    """
    """平台的高度。
    设置为 -1.0。

    如果值是负值，高度与对象高度相同。
    """

    def __post_init__(self):
        if self.max_height_noise is not None:
            warnings.warn(
                "MeshRepeatedObjectsTerrainCfg: max_height_noise:float is deprecated and support will be removed in the"
                " future. Use abs_height_noise:list[float] instead."
            )
            self.abs_height_noise = (-self.max_height_noise, self.max_height_noise)


@configclass
class MeshRepeatedPyramidsTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated pyramids."""
    """设置一个具有重复金字塔的地形。"""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for a curriculum of repeated pyramids."""
        """复制金字塔的课程配置。"""

        radius: float = MISSING
        """The radius of the pyramids (in m)."""
        """金字塔半径 (m)。"""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        """沿着y和x轴的最大角。
        默认为0.0。
        """
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""
        """如果角是度。
        默认为 True。
        """

    object_type = mesh_utils_terrains.make_cone

    object_params_start: ObjectCfg = MISSING
    """The object curriculum parameters at the start of the curriculum."""
    """课程开始时的目标课程参数。"""

    object_params_end: ObjectCfg = MISSING
    """The object curriculum parameters at the end of the curriculum."""
    """在课程结束时的对象课程参数。"""


@configclass
class MeshRepeatedBoxesTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated boxes."""
    """设置为重复框的地形。"""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for repeated boxes."""
        """复制框的配置"""

        size: tuple[float, float] = MISSING
        """The width (along x) and length (along y) of the box (in m)."""
        """箱的宽度 (x) 和长度 (y) (m)。"""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        """沿着y和x轴的最大角。
        默认为0.0。
        """
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""
        """如果角是度。
        默认为 True。
        """

    object_type = mesh_utils_terrains.make_box

    object_params_start: ObjectCfg = MISSING
    """The box curriculum parameters at the start of the curriculum."""
    """在课程开始时的课程参数框。"""

    object_params_end: ObjectCfg = MISSING
    """The box curriculum parameters at the end of the curriculum."""
    """课程结束时的课程参数框。"""


@configclass
class MeshRepeatedCylindersTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated cylinders."""
    """对于重复的地形的配置。"""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for repeated cylinder."""
        """复制的配置"""

        radius: float = MISSING
        """The radius of the pyramids (in m)."""
        """金字塔半径 (m)。"""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        """沿着y和x轴的最大角。
        默认为0.0。
        """
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""
        """如果角是度。
        默认为 True。
        """

    object_type = mesh_utils_terrains.make_cylinder

    object_params_start: ObjectCfg = MISSING
    """The box curriculum parameters at the start of the curriculum."""
    """在课程开始时的课程参数框。"""

    object_params_end: ObjectCfg = MISSING
    """The box curriculum parameters at the end of the curriculum."""
    """课程结束时的课程参数框。"""
