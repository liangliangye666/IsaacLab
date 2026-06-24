# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

from __future__ import annotations
"""定义可用的不同地形的配置类。
每个配置类必须继承``isaaclab.terrains.terrains_cfg.TerrainConfig``并定义以下属性:

- ``name``:地形名称.这是USD阶段的prim名称。
- ``function``函数生成地形.该函数必须以地形难度和配置参数作为输入，并返回一个`trimesh``网格物体和地形来源。
"""

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .sub_terrain_cfg import SubTerrainBaseCfg
from .terrain_generator import TerrainGenerator


@configclass
class TerrainGeneratorCfg:
    """Configuration for the terrain generator."""
    """土地发电机的配置。"""

    class_type: type = TerrainGenerator
    """The class to use for the terrain generator.

    Defaults to :class:`isaaclab.terrains.terrain_generator.TerrainGenerator`.
    """
    """这种类型的地形生成器。

    在:class:`isaaclab.terrains.terrain_generator.TerrainGenerator`上默认。
    """

    seed: int | None = None
    """The seed for the random number generator. Defaults to None, in which case the seed from the
    current NumPy's random state is used.

    When the seed is set, the random number generator is initialized with the given seed. This ensures
    that the generated terrains are deterministic across different runs. If the seed is not set, the
    seed from the current NumPy's random state is used. This assumes that the seed is set elsewhere in
    the code.
    """
    """随机数生成器的种子。
    默认为 None，在这种情况下使用来自当前NumPy的随机状态的种子。

    当种子设置时，随机数生成器将与给定的种子初始化。
    这确保生成的地形在不同的运行中具有确定性。
    如果种子没有设置，则使用当前NumPy的随机状态的种子。
    这假设种子在代码的其他地方设置。
    """

    curriculum: bool = False
    """Whether to use the curriculum mode. Defaults to False.

    If True, the terrains are generated based on their difficulty parameter. Otherwise,
    they are randomly generated.
    """
    """是否使用课程模式。
    默认为 False。

    如果True，则根据其难度参数生成地形。
    否则，它们是随机生成的。
    """

    size: tuple[float, float] = MISSING
    """The width (along x) and length (along y) of each sub-terrain (in m).

    Note:
      This value is passed on to all the sub-terrain configurations.
    """
    """每个地下区域的宽度 (沿 x) 和长度 (沿 y)。

    说明：
      这值将传递到所有地下配置。
    """

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0."""
    """地形周边边界宽度 (m)。
    默认为0.0。
    """

    border_height: float = 1.0
    """The height of the border around the terrain (in m). Defaults to 1.0.

    .. note::
      The default border extends below the ground. If you want to make the border above the ground,
      choose a negative value.

    """
    """地形周边边界的高度 (m)。
    默认到1.0。

    .. 说明::
      默认的边界延伸到地面以下。
      如果您想使地面的边界，请选择负值。
    """

    num_rows: int = 1
    """Number of rows of sub-terrains to generate. Defaults to 1."""
    """需要生成的地下行数。
    默认的1。
    """

    num_cols: int = 1
    """Number of columns of sub-terrains to generate. Defaults to 1."""
    """需要生成的子地列数量。
    默认的1。
    """

    color_scheme: Literal["height", "random", "none"] = "none"
    """Color scheme to use for the terrain. Defaults to "none".

    The available color schemes are:

    - "height": Color based on the height of the terrain.
    - "random": Random color scheme.
    - "none": No color scheme.
    """
    """颜色方案用于地形。
    默认调整为"没有"。

    可用的颜色方案是:

    - "高度":基于地形的高度的颜色。
    - "随机":随机颜色方案。
    - 没有颜色。
    """

    horizontal_scale: float = 0.1
    """The discretization of the terrain along the x and y axes (in m). Defaults to 0.1.

    This value is passed on to all the height field sub-terrain configurations.
    """
    """沿 x 和 y 轴 (m) 的地形的分离。
    默认为0.1。

    这一值将转移到所有高度场地底配置。
    """

    vertical_scale: float = 0.005
    """The discretization of the terrain along the z axis (in m). Defaults to 0.005.

    This value is passed on to all the height field sub-terrain configurations.
    """
    """沿着z轴 (m) 的地形的化。
    默认为0.005。

    这一值将转移到所有高度场地底配置。
    """

    slope_threshold: float | None = 0.75
    """The slope threshold above which surfaces are made vertical. Defaults to 0.75.

    If None no correction is applied.

    This value is passed on to all the height field sub-terrain configurations.
    """
    """垂直的坡门。
    默认为0.75。

    如果None没有进行校正。

    这一值将转移到所有高度场地底配置。
    """

    sub_terrains: dict[str, SubTerrainBaseCfg] = MISSING
    """Dictionary of sub-terrain configurations.

    The keys correspond to the name of the sub-terrain configuration and the values are the corresponding
    configurations.
    """
    """地下配置字典。

    键对应地下配置名称，值是相应的配置。
    """

    difficulty_range: tuple[float, float] = (0.0, 1.0)
    """The range of difficulty values for the sub-terrains. Defaults to (0.0, 1.0).

    If curriculum is enabled, the terrains will be generated based on this range in ascending order
    of difficulty. Otherwise, the terrains will be generated based on this range in a random order.
    """
    """地下地区的难度值范围。
    默认值为 (0.0， 1.0)。

    如果课程启用，将根据这个范围在难度上升的顺序生成地形。
    否则，地形将以随机顺序根据这个范围生成。
    """

    use_cache: bool = False
    """Whether to load the sub-terrain from cache if it exists. Defaults to False.

    If enabled, the generated terrains are stored in the cache directory. When generating terrains, the cache
    is checked to see if the terrain already exists. If it does, the terrain is loaded from the cache. Otherwise,
    the terrain is generated and stored in the cache. Caching can be used to speed up terrain generation.
    """
    """如果存在，将地下区域从缓存中加载。
    默认为 False。

    如果启用，生成的地形将存储在缓存目录中。
    在生成地形时，检查地形是否已经存在。
    如果是这样，地形将从缓存中被加载。
    否则，地形会生成并存储在缓存中。
    缓存可以用来加速地形生成。
    """

    cache_dir: str = "/tmp/isaaclab/terrains"
    """The directory where the terrain cache is stored. Defaults to "/tmp/isaaclab/terrains"."""
    """地形缓存存储处的目录。
    "/tmp/isaaclab/terrains"的默认设置
    """
