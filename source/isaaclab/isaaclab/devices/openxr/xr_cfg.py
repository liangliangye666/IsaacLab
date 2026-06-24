# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

import enum
from collections.abc import Callable

import numpy as np

from isaaclab.utils import configclass


class XrAnchorRotationMode(enum.Enum):
    """Enumeration for XR anchor rotation modes."""
    """列表XR旋转模式。"""

    FIXED = "fixed"
    """Fixed rotation mode: sets rotation once and doesn't change it."""
    """固定旋转模式:设置一次旋转，不会改变。"""

    FOLLOW_PRIM = "follow_prim"
    """Follow prim rotation mode: rotation follows prim's rotation."""
    """按照prim旋转模式:旋转遵循prim的旋转。"""

    FOLLOW_PRIM_SMOOTHED = "follow_prim_smoothed"
    """Follow prim rotation mode with smooth interpolation: rotation smoothly follows prim's rotation using slerp."""
    """随着流的插射，遵循prim旋转模式:旋转顺利遵循prim的旋转，使用slerp。"""

    CUSTOM = "custom_rotation"
    """Custom rotation mode: user provided function to calculate the rotation."""
    """定制旋转模式:用户提供了计算旋转的功能。"""


@configclass
class XrCfg:
    """Configuration for viewing and interacting with the environment through an XR device."""
    """通过XR设备查看和与环境交互的配置。"""

    anchor_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Specifies the position (in m) of the simulation when viewed in an XR device.

    Specifically: this position will appear at the origin of the XR device's local coordinate frame.
    """
    """在XR装置中查看时，指定仿真的位置 (以m)。

    Specifically: 这个位置将出现在XR设备的本地坐标框架的源头。
    """

    anchor_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Specifies the rotation (as a quaternion) of the simulation when viewed in an XR device.

    Specifically: this rotation will determine how the simulation is rotated with respect to the
    origin of the XR device's local coordinate frame.

    This quantity is only effective if :attr:`xr_anchor_pos` is set.
    """
    """在XR装置中查看时指定仿真的旋转 (作为四元数)。

    Specifically: 这种旋转将决定仿真如何与
    XR设备的本地坐标框架的来源。

    如果设置:attr:`xr_anchor_pos`，这个数量才有效。
    """

    anchor_prim_path: str | None = None
    """Specifies the prim path to attach the XR anchor to for dynamic positioning.

    When set, the XR anchor will be attached to the specified prim (e.g., robot root prim),
    allowing the XR camera to move with the prim. This is particularly useful for locomotion
    robot teleoperation where the robot moves and the XR camera should follow it.

    If None, the anchor will use the static :attr:`anchor_pos` and :attr:`anchor_rot` values.
    """
    """指定prim路径，以连接XR为动态定位。

    在设置时，XR将被连接到指定的prim (e.g.，机器人根 prim) 上，使XR摄像头能够与prim一起移动。
    这对于机器人移动机器人远程操作特别有用，机器人移动，XR摄像头应该跟随它。

    如果None，将使用静态:attr:`anchor_pos`和:attr:`anchor_rot`值。
    """

    anchor_rotation_mode: XrAnchorRotationMode = XrAnchorRotationMode.FIXED
    """Specifies how the XR anchor rotation should behave when attached to a prim.

    The available modes are:
    - :attr:`XrAnchorRotationMode.FIXED`: Sets rotation once to anchor_rot value
    - :attr:`XrAnchorRotationMode.FOLLOW_PRIM`: Rotation follows prim's rotation
    - :attr:`XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED`: Rotation smoothly follows prim's rotation using slerp
    - :attr:`XrAnchorRotationMode.CUSTOM`: user provided function to calculate the rotation
    """
    """指定XR杆旋转应在安装到prim时如何表现。

    可用的模式是:
    - :attr:`XrAnchorRotationMode.FIXED`:设置一次旋转为anchor_rot值
    - :attr:`XrAnchorRotationMode.FOLLOW_PRIM`:旋转跟随prim的旋转
    - :attr:`XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED`:旋转顺利跟随prim的旋转，使用slerp
    - :attr:`XrAnchorRotationMode.CUSTOM`:用户提供了计算旋转的函数
    """

    anchor_rotation_smoothing_time: float = 1.0
    """Wall-clock time constant (seconds) for rotation smoothing in FOLLOW_PRIM_SMOOTHED mode.

    This time constant is applied using wall-clock delta time between frames (not physics dt).
    Smaller values (e.g., 0.1) result in faster/snappier response but less smoothing.
    Larger values (e.g., 0.75–2.0) result in slower/smoother response but more lag.
    Typical useful range: 0.3 – 1.5 seconds depending on runtime frame-rate and comfort.
    """
    """在FOLLOW_PRIM_SMOOTHED模式下进行旋转平滑的墙钟时间常量 (秒)。

    这种时间常量是使用墙钟间的直角时间 (而不是物理dt) 应用的。
    较小的值 (e.g.，0.1) 导致更快/更快的反应，但更少的平滑性。
    较大的值 (e.g.，0.752.0) 导致反应缓慢/柔软，但延迟更大。
    典型的有用范围:0.3 1.5秒，取决于运行时间框架速度和舒适性。
    """

    anchor_rotation_custom_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda headpose, primpose: np.array(
        [1, 0, 0, 0], dtype=np.float64
    )
    """Specifies the function to calculate the rotation of the XR anchor when anchor_rotation_mode is CUSTOM.

    Args:
        headpose: Previous head pose as numpy array [x, y, z, w, x, y, z] (position + quaternion)
        pose: Anchor prim pose as numpy array [x, y, z, w, x, y, z] (position + quaternion)

    Returns:
        np.ndarray: Quaternion as numpy array [w, x, y, z]
    """
    """指定anchor_rotation_mode为CUSTOM时计算XR的旋转函数。

    参数：
        headpose: 前头姿势为 numpy 阵列 [x， y， z， w， x， y， z] (位置+四角)
        pose: prim 作为 numpy array [x， y， z， w， x， y， z] (位置+四元数)

    返回：
        np.ndarray: 作为 numpy array [w， x， y， z]
    """

    near_plane: float = 0.15
    """Specifies the near plane distance for the XR device.

    This value determines the closest distance at which objects will be rendered in the XR device.
    """
    """指定XR设备的近平面距离。

    这一值确定XR设备中对象将以最接近距离的距离。
    """

    fixed_anchor_height: bool = True
    """Specifies if the anchor height should be fixed.

    If True, the anchor height will be fixed to the initial height of the anchor prim.
    """
    """指定是否应该固定杆高度。

    如果True，杆高度将固定到杆prim的初始高度。
    """


from typing import Any


def remove_camera_configs(env_cfg: Any) -> Any:
    """Removes cameras from environments when using XR devices.

    XR does not support additional cameras in the environment as they can cause
    rendering conflicts and performance issues. This function scans the environment
    configuration for camera objects and removes them, along with any associated
    observation terms that reference these cameras.

    Args:
        env_cfg: The environment configuration to modify.

    Returns:
        The modified environment configuration with cameras removed.
    """
    """在使用XR设备时将摄像头从环境中移除。

    XR不支持环境中的额外摄像头，因为它们可能导致渲染冲突和性能问题。
    这项功能扫描了摄像头对象的环境配置并删除它们，以及这些摄像头的相关观测项。

    参数：
        env_cfg: 环境配置要修改。

    返回：
        通过移除摄像头修改的环境配置。
    """

    import logging

    # import logger
    logger = logging.getLogger(__name__)

    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import CameraCfg

    for attr_name in dir(env_cfg.scene):
        attr = getattr(env_cfg.scene, attr_name)
        if isinstance(attr, CameraCfg):
            delattr(env_cfg.scene, attr_name)
            logger.info(f"Removed camera config: {attr_name}")

            # Remove any ObsTerms for the camera
            if hasattr(env_cfg.observations, "policy"):
                for obs_name in dir(env_cfg.observations.policy):
                    obsterm = getattr(env_cfg.observations.policy, obs_name)
                    if hasattr(obsterm, "params") and obsterm.params:
                        for param_value in obsterm.params.values():
                            if isinstance(param_value, SceneEntityCfg) and param_value.name == attr_name:
                                delattr(env_cfg.observations.policy, attr_name)
                                logger.info(f"Removed camera observation term: {attr_name}")
                                break
    return env_cfg
