# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from ..sub_terrain_cfg import SubTerrainBaseCfg
from . import hf_terrains


@configclass
class HfTerrainBaseCfg(SubTerrainBaseCfg):
    """The base configuration for height field terrains."""
    """高度地形的基础配置。"""

    border_width: float = 0.0
    """The width of the border/padding around the terrain (in m). Defaults to 0.0.

    The border width is subtracted from the :obj:`size` of the terrain. If non-zero, it must be
    greater than or equal to the :obj:`horizontal scale`.
    """
    """边界/地形周围接的宽度 (m)。
    默认为0.0。

    边界宽度从地形的:obj:`size`中减去。
    如果不为零，则必须大于或等于:obj:`horizontal scale`。
    """

    horizontal_scale: float = 0.1
    """The discretization of the terrain along the x and y axes (in m). Defaults to 0.1."""
    """沿 x 和 y 轴 (m) 的地形的分离。
    默认为0.1。
    """

    vertical_scale: float = 0.005
    """The discretization of the terrain along the z axis (in m). Defaults to 0.005."""
    """沿着z轴 (m) 的地形的化。
    默认为0.005。
    """

    slope_threshold: float | None = None
    """The slope threshold above which surfaces are made vertical. Defaults to None,
    in which case no correction is applied."""
    """垂直的坡门。
    在 None 中，没有修改。
    """


"""
Different height field terrain configurations.
"""
"""不同的高度地形配置。
"""


@configclass
class HfRandomUniformTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a random uniform height field terrain."""
    """为随机统一高度的场地配置。"""

    function = hf_terrains.random_uniform_terrain

    noise_range: tuple[float, float] = MISSING
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""
    """地形的最小和最大高度噪音 (i.e.沿 z) (以m)。"""

    noise_step: float = MISSING
    """The minimum height (in m) change between two points."""
    """两个点之间的最低高度 (在m) 变化。"""

    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """
    """在地形上的两个随机采样点之间的距离。
    默认为 None，在这种情况下使用:obj:`horizontal scale`。

    在此分辨率下采样高度，对中间点进行插射。
    这必须大于或等于:obj:`horizontal scale`。
    """


@configclass
class HfPyramidSlopedTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a pyramid sloped height field terrain."""
    """为金字塔倾斜的高度地形配置。"""

    function = hf_terrains.pyramid_sloped_terrain

    slope_range: tuple[float, float] = MISSING
    """The slope of the terrain (in radians)."""
    """地形的倾斜 (在半径中)。"""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    """在地形中心的方形平台宽度。
    默认到1.0。
    """

    inverted: bool = False
    """Whether the pyramid is inverted. Defaults to False.

    If True, the terrain is inverted such that the platform is at the bottom and the slopes are upwards.
    """
    """无论金字塔是否倒向。
    默认为 False。

    如果True，地形将倒向，平台在底部，斜坡上升。
    """


@configclass
class HfInvertedPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    """Configuration for an inverted pyramid sloped height field terrain.

    Note:
        This is a subclass of :class:`HfPyramidSlopedTerrainCfg` with :obj:`inverted` set to True.
        We make it as a separate class to make it easier to distinguish between the two and match
        the naming convention of the other terrains.
    """
    """向金字塔的高度地形配置。

    说明：
        这是一个:class:`HfPyramidSlopedTerrainCfg`的子类，:obj:`inverted`为True。
        我们将其作为一个单独的类别， 让它更容易区分两者，
    """

    inverted: bool = True


@configclass
class HfPyramidStairsTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a pyramid stairs height field terrain."""
    """为金字塔楼梯高度的场地配置。"""

    function = hf_terrains.pyramid_stairs_terrain

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

    inverted: bool = False
    """Whether the pyramid stairs is inverted. Defaults to False.

    If True, the terrain is inverted such that the platform is at the bottom and the stairs are upwards.
    """
    """是否是逆转的?
    默认为 False。

    如果是True，地形会倒向，平台会在底部，楼梯会上升。
    """


@configclass
class HfInvertedPyramidStairsTerrainCfg(HfPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stairs height field terrain.

    Note:
        This is a subclass of :class:`HfPyramidStairsTerrainCfg` with :obj:`inverted` set to True.
        We make it as a separate class to make it easier to distinguish between the two and match
        the naming convention of the other terrains.
    """
    """对反向金字塔楼梯高度的地形配置。

    说明：
        这是一个:class:`HfPyramidStairsTerrainCfg`的子类，:obj:`inverted`为True。
        我们将其作为一个单独的类别， 让它更容易区分两者，
    """

    inverted: bool = True


@configclass
class HfDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a discrete obstacles height field terrain."""
    """对于单独的障碍高度地形配置。"""

    function = hf_terrains.discrete_obstacles_terrain

    obstacle_height_mode: str = "choice"
    """The mode to use for the obstacle height. Defaults to "choice".

    The following modes are supported: "choice", "fixed".
    """
    """障碍高度使用的模式。
    默认的"选择"。

    支持以下模式:"选择"，"固定"。
    """

    obstacle_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the obstacles (in m)."""
    """障碍物的最小和最大宽度 (m)。"""

    obstacle_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the obstacles (in m)."""
    """障碍物的最小和最大高度 (m)。"""

    num_obstacles: int = MISSING
    """The number of obstacles to generate."""
    """需要创建的障碍。"""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    """在地形中心的方形平台宽度。
    默认到1.0。
    """


@configclass
class HfWaveTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a wave height field terrain."""
    """对波高场地进行配置。"""

    function = hf_terrains.wave_terrain

    amplitude_range: tuple[float, float] = MISSING
    """The minimum and maximum amplitude of the wave (in m)."""
    """波的最小和最大振幅 (m)。"""

    num_waves: int = 1
    """The number of waves to generate. Defaults to 1."""
    """需要生成的波数。
    默认的1。
    """


@configclass
class HfSteppingStonesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a stepping stones height field terrain."""
    """设置一个脚梯高度的场地形。"""

    function = hf_terrains.stepping_stones_terrain

    stone_height_max: float = MISSING
    """The maximum height of the stones (in m)."""
    """石头的最大高度 (m)。"""

    stone_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the stones (in m)."""
    """石头的最小和最大宽度 (m)。"""

    stone_distance_range: tuple[float, float] = MISSING
    """The minimum and maximum distance between stones (in m)."""
    """石头之间的最小和最大距离 (m)。"""

    holes_depth: float = -10.0
    """The depth of the holes (negative obstacles). Defaults to -10.0."""
    """洞的深度 (负面障碍)。
    默认到 -10.0。
    """

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    """在地形中心的方形平台宽度。
    默认到1.0。
    """
