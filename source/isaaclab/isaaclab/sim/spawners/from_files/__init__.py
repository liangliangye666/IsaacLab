# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn assets from files.

Currently, the following spawners are supported:

* :class:`UsdFileCfg`: Spawn an asset from a USD file.
* :class:`UrdfFileCfg`: Spawn an asset from a URDF file.
* :class:`GroundPlaneCfg`: Spawn a ground plane using the grid-world USD file.

"""
"""对于从文件中产生的产物产物器的子模块。

目前，以下产品支持:

* :class:`UsdFileCfg`:从一个USD文件。
* :class:`UrdfFileCfg`:从一个URDF文件。
* :class:`GroundPlaneCfg`通过网格世界来生成地面飞机USD文件。
"""

from .from_files import (
    spawn_from_mjcf,
    spawn_from_urdf,
    spawn_from_usd,
    spawn_from_usd_with_compliant_contact_material,
    spawn_ground_plane,
)
from .from_files_cfg import GroundPlaneCfg, MjcfFileCfg, UrdfFileCfg, UsdFileCfg, UsdFileWithCompliantContactCfg
