# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .frame_transformer import FrameTransformer


@configclass
class OffsetCfg:
    """The offset pose of one frame relative to another frame."""
    """一个框架对另一个框架的偏移姿势。"""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
    """翻译w.r.t
    它们的母体。
    在 (0.0，0.0，0.0) 之前的默认设置。
    """
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""
    """四元数旋转 (w，x，y，z) w.r.t。
    它们的母体。
    在 (1.0，0.0，0.0，0.0，0.0的默认情况下。
    """


@configclass
class FrameTransformerCfg(SensorBaseCfg):
    """Configuration for the frame transformer sensor."""
    """框架变压器传感器的配置。"""

    @configclass
    class FrameCfg:
        """Information specific to a coordinate frame."""
        """特定对坐标框架的信息。"""

        prim_path: str = MISSING
        """The prim path corresponding to a rigid body.

        This can be a regex pattern to match multiple prims. For example, "/Robot/.*"
        will match all prims under "/Robot".

        This means that if the source :attr:`FrameTransformerCfg.prim_path` is "/Robot/base",
        and the target :attr:`FrameTransformerCfg.FrameCfg.prim_path` is "/Robot/.*", then
        the frame transformer will track the poses of all the prims under "/Robot",
        including "/Robot/base" (even though this will result in an identity pose w.r.t.
        the source frame).
        """
        """体的 prim路径。

        这可能是一个符合多个prims的regex模式。
        例如"，/Robot/.*"将与"/Robot"下的所有prims匹配。

        这意味着，如果源:attr:`FrameTransformerCfg.prim_path`是"/机器人/基地"，目标:attr:`FrameTransformerCfg.FrameCfg.prim_p
        ath`是"/机器人/.*"，则框架变压器将追踪所有prims的姿势在"/机器人"下，包括"/机器人/基地" (尽管这会导致身份姿势w.r.t。
        源框架)。
        """

        name: str | None = None
        """User-defined name for the new coordinate frame. Defaults to None.

        If None, then the name is extracted from the leaf of the prim path.
        """
        """新坐标框架的用户定义名称。
        默认为 None。

        如果是None，则该名称从prim路径的叶子中提取。
        """

        offset: OffsetCfg = OffsetCfg()
        """The pose offset from the parent prim frame."""
        """parent置于母体prim框架。"""

    class_type: type = FrameTransformer

    prim_path: str = MISSING
    """The prim path of the body to transform from (source frame)."""
    """从 (源框架) 转换身体的prim路径。"""

    source_frame_offset: OffsetCfg = OffsetCfg()
    """The pose offset from the source prim frame."""
    """从源 prim 框架的姿势。"""

    target_frames: list[FrameCfg] = MISSING
    """A list of the target frames.

    This allows a single FrameTransformer to handle multiple target prims. For example, in a quadruped,
    we can use a single FrameTransformer to track each foot's position and orientation in the body
    frame using four frame offsets.
    """
    """目标框架的列表。

    这允许一个FrameTransformer处理多个目标prims。
    例如，在四足机器人中，我们可以使用单个FrameTransformer来追踪每条脚的位置和方向在车身框架中，使用四个框架抵消。
    """

    visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer")
    """The configuration object for the visualization markers. Defaults to FRAME_MARKER_CFG.

    Note:
        This attribute is only used when debug visualization is enabled.
    """
    """视觉化标记的配置对象。
    在 FRAME_MARKER_CFG 设置中，

    说明：
        只有在启用调试可视化时才使用此属性。
    """
