# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from .sensor_base import SensorBase


@configclass
class SensorBaseCfg:
    """Configuration parameters for a sensor."""
    """传感器的配置参数"""

    class_type: type[SensorBase] = MISSING
    """The associated sensor class.

    The class should inherit from :class:`isaaclab.sensors.sensor_base.SensorBase`.
    """
    """相关传感器类。

    这类应该继承:class:`isaaclab.sensors.sensor_base.SensorBase`。
    """

    prim_path: str = MISSING
    """Prim path (or expression) to the sensor.

    .. note::
        The expression can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Robot/sensor`` will be replaced with ``/World/envs/env_.*/Robot/sensor``.

    """
    """传感器的基本路径 (或表达式)。

    .. 说明::
        这个表达式可以包含环境命名空间regex ``{ENV_REGEX_NS}``，将被环境命名空间取代。

        Example: ``{ENV_REGEX_NS}/Robot/sensor``将被 ``/World/envs/env_.*/Robot/sensor`` 取代。
    """

    update_period: float = 0.0
    """Update period of the sensor buffers (in seconds). Defaults to 0.0 (update every step)."""
    """传感器缓冲器的更新时间 (几秒钟)。
    默认到0.0 (更新每一步)。
    """

    history_length: int = 0
    """Number of past frames to store in the sensor buffers. Defaults to 0, which means that only
    the current data is stored (no history)."""
    """在传感器缓冲器中存储的过去框架数量。
    默认到0，这意味着只有当前数据存储 (没有历史记录)。
    """

    debug_vis: bool = False
    """Whether to visualize the sensor. Defaults to False."""
    """是否可视化传感器。
    默认为 False。
    """
