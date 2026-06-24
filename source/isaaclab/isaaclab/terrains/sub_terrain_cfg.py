# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.utils import configclass


@configclass
class FlatPatchSamplingCfg:
    """Configuration for sampling flat patches on the sub-terrain.

    For a given sub-terrain, this configuration specifies how to sample flat patches on the terrain.
    The sampled flat patches can be used for spawning robots, targets, etc.

    Please check the function :meth:`~isaaclab.terrains.utils.find_flat_patches` for more details.
    """
    """在地下采样平板块的配置。

    对于给定的地下区域，这个配置指定了如何在地形上采样平坦的斑点。
    采用样本的平面贴片可用于产卵机器人，目标等。

    请查看函数 :meth:`~isaaclab.terrains.utils.find_flat_patches` 详细信息。
    """

    num_patches: int = MISSING
    """Number of patches to sample."""
    """检测的补丁数量。"""

    patch_radius: float | list[float] = MISSING
    """Radius of the patches.

    A list of radii can be provided to check for patches of different sizes. This is useful to deal with
    cases where the terrain may have holes or obstacles in some areas.
    """
    """斑点的半径。

    可以提供半径列表，以检查不同尺寸的补丁。
    这对于地形可能在某些地区存在洞或障碍的情况来说是有用的。
    """

    x_range: tuple[float, float] = (-1e6, 1e6)
    """The range of x-coordinates to sample from. Defaults to (-1e6, 1e6).

    This range is internally clamped to the size of the terrain mesh.
    """
    """取样的x坐标范围。
    在 (-1e6， 1e6) 中的默认值。

    这种范围是根据地形网格的尺寸进行的。
    """

    y_range: tuple[float, float] = (-1e6, 1e6)
    """The range of y-coordinates to sample from. Defaults to (-1e6, 1e6).

    This range is internally clamped to the size of the terrain mesh.
    """
    """取样的y坐标范围。
    在 (-1e6， 1e6) 中的默认值。

    这种范围是根据地形网格的尺寸进行的。
    """

    z_range: tuple[float, float] = (-1e6, 1e6)
    """Allowed range of z-coordinates for the sampled patch. Defaults to (-1e6, 1e6)."""
    """对采样补丁的z坐标允许范围。
    在 (-1e6， 1e6) 中的默认值。
    """

    max_height_diff: float = MISSING
    """Maximum allowed height difference between the highest and lowest points on the patch."""
    """补丁上最高点和最低点之间的最大允许高度差异。"""


@configclass
class SubTerrainBaseCfg:
    """Base class for terrain configurations.

    All the sub-terrain configurations must inherit from this class.

    The :attr:`size` attribute is the size of the generated sub-terrain. Based on this, the terrain must
    extend from :math:`(0, 0)` to :math:`(size[0], size[1])`.
    """
    """地形配置的基础类。

    所有地下配置都必须继承这个类。

    The :attr:`size`属性是生成的地下区域的大小。
         根据这一点，地形必须
    从数学:`(0， 0)`到数学:`(size[0]， size[1])`。
    """

    function: Callable[[float, SubTerrainBaseCfg], tuple[list[trimesh.Trimesh], np.ndarray]] = MISSING
    """Function to generate the terrain.

    This function must take as input the terrain difficulty and the configuration parameters and
    return a tuple with a list of ``trimesh`` mesh objects and the terrain origin.
    """
    """能产生地形的功能。

    这项函数必须以地形难度和配置参数为输入，
    return a tuple with a list of ``trimesh`` mesh objects and the terrain origin.
    """

    proportion: float = 1.0
    """Proportion of the terrain to generate. Defaults to 1.0.

    This is used to generate a mix of terrains. The proportion corresponds to the probability of sampling
    the particular terrain. For example, if there are two terrains, A and B, with proportions 0.3 and 0.7,
    respectively, then the probability of sampling terrain A is 0.3 and the probability of sampling terrain B
    is 0.7.
    """
    """产生的地形比例。
    默认到1.0。

    这种方法用于生成混合地形。
    比例与特定地形采样概率相符。
    例如，如果有两个地形，A和B，分别有0.3和0.7的比例，则采样地形A的概率为0.3，采样地形B的概率为0.7。
    """

    size: tuple[float, float] = (10.0, 10.0)
    """The width (along x) and length (along y) of the terrain (in m). Defaults to (10.0, 10.0).

    In case the :class:`~isaaclab.terrains.TerrainImporterCfg` is used, this parameter gets overridden by
    :attr:`isaaclab.scene.TerrainImporterCfg.size` attribute.
    """
    """地形的宽度 (沿 x) 和长度 (沿 y)。
    默认的 (10.0， 10.0)

    如果使用:class:`~isaaclab.terrains.TerrainImporterCfg`，这个参数将被:attr:`isaaclab.scene.TerrainImporterCfg.si
    ze`属性覆盖。
    """

    flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
    """Dictionary of configurations for sampling flat patches on the sub-terrain. Defaults to None,
    in which case no flat patch sampling is performed.

    The keys correspond to the name of the flat patch sampling configuration and the values are the
    corresponding configurations.
    """
    """在地下采样平板块的配置字典。
    在 None 时的默认情况上，在这种情况下，没有进行平面补丁样本。

    键与平板补丁采样配置名称相符，值是相应的配置。
    """
