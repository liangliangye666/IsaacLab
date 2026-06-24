# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This sub-module provides methods to create different terrains using the ``trimesh`` library.

In contrast to the height-field representation, the trimesh representation does not
create arbitrarily small triangles. Instead, the terrain is represented as a single
tri-mesh primitive. Thus, this representation is more computationally and memory
efficient than the height-field representation, but it is not as flexible.
"""
"""该子模块提供使用``trimesh``库创建不同地形的方法。

与高度场表示不同，三角形表示不会任意创建小三角形。
而地形则被表示为单个三网格原始。
因此，这种表示比高度场表示更具计算效率和存储效率，但它并不灵活。
"""

from .mesh_terrains_cfg import (
    MeshBoxTerrainCfg,
    MeshFloatingRingTerrainCfg,
    MeshGapTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPitTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRailsTerrainCfg,
    MeshRandomGridTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    MeshRepeatedCylindersTerrainCfg,
    MeshRepeatedPyramidsTerrainCfg,
    MeshStarTerrainCfg,
)
