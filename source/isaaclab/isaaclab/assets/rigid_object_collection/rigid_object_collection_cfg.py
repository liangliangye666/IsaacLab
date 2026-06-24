# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.utils import configclass

from .rigid_object_collection import RigidObjectCollection


@configclass
class RigidObjectCollectionCfg:
    """Configuration parameters for a rigid object collection."""
    """硬体集合的配置参数。"""

    class_type: type = RigidObjectCollection
    """The associated asset class.

    The class should inherit from :class:`isaaclab.assets.asset_base.AssetBase`.
    """
    """相关资产类别。

    这类应该继承:class:`isaaclab.assets.asset_base.AssetBase`。
    """

    rigid_objects: dict[str, RigidObjectCfg] = MISSING
    """Dictionary of rigid object configurations to spawn.

    The keys are the names for the objects, which are used as unique identifiers throughout the code.
    """
    """复制的硬体配置字典。

    键是对象的名称，它们在整个代码中作为独特的标识符。
    """
