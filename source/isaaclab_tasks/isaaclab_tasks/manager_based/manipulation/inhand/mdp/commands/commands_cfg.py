# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .orientation_command import InHandReOrientationCommand


@configclass
class InHandReOrientationCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """
    """为统一的3D导向命令项的配置。

    详细请参阅:class:`InHandReOrientationCommand`类。
    """

    class_type: type = InHandReOrientationCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)  # no resampling based on time

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    """在环境中产生的指令的资产名称。"""

    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the asset from its default position.

    This is used to account for the offset typically present in the object's default position
    so that the object is spawned at a height above the robot's palm. When the position command
    is generated, the object's default position is used as the reference and the offset specified
    is added to it to get the desired position of the object.
    """
    """资产的负债表

    这用于解释对物体默认位置中通常存在的偏移，以便物体在机器人的手掌上高处产生。
    当生成位置命令时，对象的默认位置被使用为参考，并添加指定的偏移，以获得对象的所需位置。
    """

    make_quat_unique: bool = MISSING
    """Whether to make the quaternion unique or not.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """
    """不管要么，要么，要么。

    如果是True，则通过确保真实部分是正的，使四分之一变得独特。
    """

    orientation_success_threshold: float = MISSING
    """Threshold for the orientation error to consider the goal orientation to be reached."""
    """导向错误的门值，以考虑实现目标导向。"""

    update_goal_on_success: bool = MISSING
    """Whether to update the goal orientation when the goal orientation is reached."""
    """当目标方向达到时是否更新目标方向。"""

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the marker from the object's desired position.

    This is useful to position the marker at a height above the object's desired position.
    Otherwise, the marker may occlude the object in the visualization.
    """
    """标记的位置从对象所需的位置偏移。

    这有助于将标记定位在对象所需位置以上的高度。
    否则，标记可能会封闭视觉中的对象。
    """

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            ),
        },
    )
    """The configuration for the goal pose visualization marker. Defaults to a DexCube marker."""
    """目标位置的配置可视化标记。
    默认的DexCube标记。
    """
