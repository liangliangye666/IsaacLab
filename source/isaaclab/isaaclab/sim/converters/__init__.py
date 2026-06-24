# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing converters for converting various file types to USD.

In order to support direct loading of various file types into Omniverse, we provide a set of
converters that can convert the file into a USD file. The converters are implemented as
sub-classes of the :class:`AssetConverterBase` class.

The following converters are currently supported:

* :class:`UrdfConverter`: Converts a URDF file into a USD file.
* :class:`MeshConverter`: Converts a mesh file into a USD file. This supports OBJ, STL and FBX files.

"""
"""包含各种文件类型转换为USD的转换器的子模块。

为了支持将各种文件类型直接加载到Omniverse中，我们提供了一套可将文件转换为USD文件的转换器。
转换器作为:class:`AssetConverterBase`类的子类进行实施。

目前支持以下转换器:

* :class:`UrdfConverter`:将URDF文件转换为USD文件。
* :class:`MeshConverter`:将网格文件转换为USD文件。 这支持OBJ，STL和FBX文件。
"""

from .asset_converter_base import AssetConverterBase
from .asset_converter_base_cfg import AssetConverterBaseCfg
from .mesh_converter import MeshConverter
from .mesh_converter_cfg import MeshConverterCfg
from .mjcf_converter import MjcfConverter
from .mjcf_converter_cfg import MjcfConverterCfg
from .urdf_converter import UrdfConverter
from .urdf_converter_cfg import UrdfConverterCfg
