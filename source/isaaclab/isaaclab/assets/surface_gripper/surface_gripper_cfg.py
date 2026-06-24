# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .surface_gripper import SurfaceGripper


@configclass
class SurfaceGripperCfg(AssetBaseCfg):
    """Configuration parameters for a surface gripper actuator."""
    """表面抓住器执行器的配置参数"""

    prim_path: str = MISSING
    """The expression to find the grippers in the stage."""
    """在舞台上找到抓住器的表情。"""

    max_grip_distance: float | None = None
    """The maximum grip distance of the gripper."""
    """抓住器的最大抓住距离。"""

    coaxial_force_limit: float | None = None
    """The coaxial force limit of the gripper."""
    """抓住器的同轴力限制。"""

    shear_force_limit: float | None = None
    """The shear force limit of the gripper."""
    """抓住器的切割力限制。"""

    retry_interval: float | None = None
    """The amount of time the gripper will spend trying to grasp an object."""
    """抓住器将花费的时间试图抓住物体。"""

    class_type: type = SurfaceGripper
