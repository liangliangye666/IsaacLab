# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This sub-module provides utilities to create different terrains as height fields (HF).

Height fields are a 2.5D terrain representation that is used in robotics to obtain the
height of the terrain at a given point. This is useful for controls and planning algorithms.

Each terrain is represented as a 2D numpy array with discretized heights. The shape of the array
is (width, length), where width and length are the number of points along the x and y axis,
respectively. The height of the terrain at a given point is obtained by indexing the array with
the corresponding x and y coordinates.

.. caution::

    When working with height field terrains, it is important to remember that the terrain is generated
    from a discretized 3D representation. This means that the height of the terrain at a given point
    is only an approximation of the real height of the terrain at that point. The discretization
    error is proportional to the size of the discretization cells. Therefore, it is important to
    choose a discretization size that is small enough for the application. A larger discretization
    size will result in a faster simulation, but the terrain will be less accurate.

"""
"""这一子模块提供了用于创建不同地形的公用事项，即高度领域 (HF)。

高度字段是2.5维地形表示，用于机器人技术，以获得特定点地形的高度。
这对于控制和规划算法是有用的。

每个地形都以 2D 形阵列呈现，
阵列的形状是 (宽度，长度)，宽度和长度分别是x和y轴沿线的点数。
在给定的点上地形的高度通过对应的x和y坐标对阵列进行索引来获得。

.. 谨慎::

    在使用高地形时，重要的是要记住地形的生成
    from a discretized 3D representation. This means that the height of the terrain at a given point
    在此时，地形的实际高度仅仅是近似的。
    化错误与化细胞的大小相比例。
    因此，选择适用于应用的小尺寸的缩尺寸是重要的。
    较大的化尺寸将导致更快的仿真，
"""

from .hf_terrains_cfg import (
    HfDiscreteObstaclesTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfSteppingStonesTerrainCfg,
    HfTerrainBaseCfg,
    HfWaveTerrainCfg,
)
