# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .rigid_object import RigidObject


@configclass
class RigidObjectCfg(AssetBaseCfg):
    """Configuration parameters for a rigid object."""
    """固体的配置参数。"""

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the rigid body."""
        """硬体的初始状态。"""

        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        """在仿真世界框架中的根的线性速度。
        在 (0.0，0.0，0.0) 之前的默认设置。
        """
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        """在仿真世界框架中的根的角速度。
        在 (0.0，0.0，0.0) 之前的默认设置。
        """

    ##
    # Initialize configurations.
    ##

    class_type: type = RigidObject

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose with zero velocity."""
    """硬体的初始状态。
    在零速度下，身份默认出现。
    """
