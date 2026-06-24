# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""
"""射线传感器的配置。"""

from dataclasses import MISSING
from typing import Literal

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .patterns.patterns_cfg import PatternBaseCfg
from .ray_caster import RayCaster


@configclass
class RayCasterCfg(SensorBaseCfg):
    """Configuration for the ray-cast sensor."""
    """射线传感器的配置。"""

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""
        """传感器框架的偏移姿势与传感器的母体框架。"""

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

    class_type: type = RayCaster

    mesh_prim_paths: list[str] = MISSING
    """The list of mesh primitive paths to ray cast against.

    Note:
        Currently, only a single static mesh is supported. We are working on supporting multiple
        static meshes and dynamic meshes.
    """
    """列表对射线的原始网路。

    说明：
        目前，仅支持单个静态网格。
        我们正在支持多个静态和动态网格。
    """

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""
    """传感器框架的偏移姿势与传感器的母体框架。
    默认身份。
    """

    attach_yaw_only: bool | None = None
    """Whether the rays' starting positions and directions only track the yaw orientation.
    Defaults to None, which doesn't raise a warning of deprecated usage.

    This is useful for ray-casting height maps, where only yaw rotation is needed.

    .. deprecated:: 2.1.1

        This attribute is deprecated and will be removed in the future. Please use
        :attr:`ray_alignment` instead.

        To get the same behavior as setting this parameter to ``True`` or ``False``, set
        :attr:`ray_alignment` to ``"yaw"`` or "base" respectively.

    """
    """射线的起始位置和方向是否仅追踪向。
    默认为 None，这没有提醒使用过时。

    这对于射线高度地图是有用的，只需要旋转。

    ..
    已过期:: 2.1.1

        这种属性已过时，将来将被删除。
        请使用:attr:`ray_alignment`。

        为了获得与设置这个参数为``True``或``False``相同的行为，分别设置:attr:`ray_alignment`为``"yaw"``或"基础"。
    """

    ray_alignment: Literal["base", "yaw", "world"] = "base"
    """Specify in what frame the rays are projected onto the ground. Default is "base".

    The options are:

    * ``base`` if the rays' starting positions and directions track the full root position and orientation.
    * ``yaw`` if the rays' starting positions and directions track root position and only yaw component of
      the orientation. This is useful for ray-casting height maps.
    * ``world`` if rays' starting positions and directions are always fixed. This is useful in combination
      with a mapping package on the robot and querying ray-casts in a global frame.
    """
    """指定射线投射到地面的框架。
    默认是"基础"。

    选择是:

    * ``base``如果射线的起始位置和方向跟踪了根的全部位置和方向。
    * 如果射线的起始位置和方向跟踪根位置和导向的只有部件，则``yaw``。 这对于射线casting高度地图是有用的。
    * ``world`` 如果射线的起始位置和方向总是固定。
      with a mapping package on the robot and querying ray-casts in a global frame.
    """

    pattern_cfg: PatternBaseCfg = MISSING
    """The pattern that defines the local ray starting positions and directions."""
    """定义本地射线起始位置和方向的模式。"""

    max_distance: float = 1e6
    """Maximum distance (in meters) from the sensor to ray cast to. Defaults to 1e6."""
    """从传感器到射线的最大距离 (以米)
    在1e6上默认设置。
    """

    drift_range: tuple[float, float] = (0.0, 0.0)
    """The range of drift (in meters) to add to the ray starting positions (xyz) in world frame. Defaults to (0.0, 0.0).

    For floating base robots, this is useful for simulating drift in the robot's pose estimation.
    """
    """漂移范围 (以米) 在世界框架中增加射线起始位置 (xyz)。
    默认值为 (0.0，0.0)。

    对于浮动基座机器人来说，这对于仿真机器人的姿势估计中的漂移有用。
    """

    ray_cast_drift_range: dict[str, tuple[float, float]] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    """The range of drift (in meters) to add to the projected ray points in local projection frame. Defaults to
    a dictionary with zero drift for each x, y and z axis.

    For floating base robots, this is useful for simulating drift in the robot's pose estimation.
    """
    """在本地投射框架中增加投射射射线点的漂移范围 (以米)。
    对于每一个x，y和z轴的零漂移字典的默认。

    对于浮动基座机器人来说，这对于仿真机器人的姿势估计中的漂移有用。
    """

    visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/RayCaster")
    """The configuration object for the visualization markers. Defaults to RAY_CASTER_MARKER_CFG.

    Note:
        This attribute is only used when debug visualization is enabled.
    """
    """视觉化标记的配置对象。
    在 RAY_CASTER_MARKER_CFG 里默认错误。

    说明：
        只有在启用调试可视化时才使用此属性。
    """
