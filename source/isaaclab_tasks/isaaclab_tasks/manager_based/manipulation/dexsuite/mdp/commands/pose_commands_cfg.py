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

from . import pose_commands as dex_cmd

ALIGN_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
        ),
        "position_far": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "position_near": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)


@configclass
class ObjectUniformPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""
    """配置为统一姿势命令生成器。"""

    class_type: type = dex_cmd.ObjectUniformPoseCommand

    asset_name: str = MISSING
    """Name of the coordinate referencing asset in the environment for which the commands are generated respect to."""
    """环境中坐标引用资产的名称，该指令被生成的指令。"""

    object_name: str = MISSING
    """Name of the object in the environment for which the commands are generated."""
    """命令生成的环境中的对象名称。"""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """
    """不管要么，要么，要么。
    默认为 False。

    如果是True，则通过确保真实部分是正的，使四分之一变得独特。
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""
        """对于姿势命令的均分布范围。"""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""
        """对 x 位置的范围 (在 m 中)。"""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""
        """为 y 位置的范围 (m)。"""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""
        """为z位置的范围 (m)。"""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""
        """滚动角的范围 (rAD)。"""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""
        """射程为角 (在rad)。"""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""
        """对角的范围 (在rad)。"""

    ranges: Ranges = MISSING
    """Ranges for the commands."""
    """距离是命令的。"""

    position_only: bool = True
    """Command goal position only. Command includes goal quat if False"""
    """只有命令目标位置。
    命令包括目标四分之一，如果False
    """

    # Pose Markers
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = ALIGN_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""
    """目标位置的配置可视化标记。
    在 FRAME_MARKER_CFG 设置中，
    """

    curr_pose_visualizer_cfg: VisualizationMarkersCfg = ALIGN_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""
    """目前姿势可视化标记的配置。
    在 FRAME_MARKER_CFG 设置中，
    """

    success_vis_asset_name: str = MISSING
    """Name of the asset in the environment for which the success color are indicated."""
    """在环境中的资产名称中，成功颜色表示。"""

    # success markers
    success_visualizer_cfg = VisualizationMarkersCfg(prim_path="/Visuals/SuccessMarkers", markers={})
    """The configuration for the success visualization marker. User needs to add the markers"""
    """设置成功可视化标记。
    用户需要添加标记
    """
