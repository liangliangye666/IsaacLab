# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ContactSensorData:
    """Data container for the contact reporting sensor."""
    """接触报告传感器的数据容器"""

    pos_w: torch.Tensor | None = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where N is the number of sensors.

    Note:
        If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.

    """
    """传感器在世界框架中的位置。

    形状是 (N， 3)，其中N是传感器的数量。

    说明：
        如果:attr:`ContactSensorCfg.track_pose`是False，那么这个数量是None。
    """

    contact_pos_w: torch.Tensor | None = None
    """Average of the positions of contact points between sensor body and filter prim in world frame.

    Shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    and M is the number of filtered bodies.

    Collision pairs not in contact will result in NaN.

    Note:

        * If the :attr:`ContactSensorCfg.track_contact_points` is False, then this quantity is None.
        * If the :attr:`ContactSensorCfg.track_contact_points` is True, a ValueError will be raised if:

          * If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
          * If the :attr:`ContactSensorCfg.max_contact_data_per_prim` is not specified or less than 1.
            will not be calculated.

    """
    """传感器体和过器 prim之间的接触点位置的平均值。

    形状是 (N， B， M， 3)，其中N是传感器的数量，B是每个传感器中的体体数，M是过体数。

    不接触的碰撞对将导致NaN。

    说明：

        * 如果:attr:`ContactSensorCfg.track_contact_points`是False，那么这个数量是None。
        * 如果:attr:`ContactSensorCfg.track_contact_points`是True， 一ValueError如果:

          * 如果:attr:`ContactSensorCfg.filter_prim_paths_expr`是空的。
          * 如果:attr:`ContactSensorCfg.max_contact_data_per_prim`未指定或小于1.则不会计算。
    """

    friction_forces_w: torch.Tensor | None = None
    """Sum of the friction forces between sensor body and filter prim in world frame.

    Shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    and M is the number of filtered bodies.

    Collision pairs not in contact will result in NaN.

    Note:

        * If the :attr:`ContactSensorCfg.track_friction_forces` is False, then this quantity is None.
        * If the :attr:`ContactSensorCfg.track_friction_forces` is True, a ValueError will be raised if:

          * The :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
          * The :attr:`ContactSensorCfg.max_contact_data_per_prim` is not specified or less than 1.

    """
    """传感器体和过器 prim之间的摩擦力总和。

    形状是 (N， B， M， 3)，其中N是传感器的数量，B是每个传感器中的体体数，M是过体数。

    不接触的碰撞对将导致NaN。

    说明：

        * 如果:attr:`ContactSensorCfg.track_friction_forces`是False，那么这个数量是None。
        * 如果:attr:`ContactSensorCfg.track_friction_forces`是True， 一ValueError如果:

          * :attr:`ContactSensorCfg.filter_prim_paths_expr`是空的。
          * :attr:`ContactSensorCfg.max_contact_data_per_prim`不指定或不小于1。
    """

    quat_w: torch.Tensor | None = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame.

    Shape is (N, 4), where N is the number of sensors.

    Note:
        If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
    """
    """传感器起源在世界框架中的四元数 (w， x， y， z) 的导向。

    形状是 (N， 4)，其中N是传感器的数量。

    说明：
        如果:attr:`ContactSensorCfg.track_pose`是False，那么这个数量是None。
    """

    net_forces_w: torch.Tensor | None = None
    """The net normal contact forces in world frame.

    Shape is (N, B, 3), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
        with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
    """
    """在世界框架中的正常接触力。

    形状是 (N，B，3) ，其中N是传感器的数量，B是每个传感器的体体数。

    说明：
        这种数量是对传感器体作用的正常接触力的总和。
        这不应该被混。
        with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
    """

    net_forces_w_history: torch.Tensor | None = None
    """The net normal contact forces in world frame.

    Shape is (N, T, B, 3), where N is the number of sensors, T is the configured history length
    and B is the number of bodies in each sensor.

    In the history dimension, the first index is the most recent and the last index is the oldest.

    Note:
        This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
        with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
    """
    """在世界框架中的正常接触力。

    形状是 (N，T，B，3)，其中N是传感器的数量，T是配置的历史长度，B是每个传感器的体体数。

    在历史层面上，第一个索引是最新的，最后一个索引是最古老的。

    说明：
        这种数量是对传感器体作用的正常接触力的总和。
        这不应该被混。
        with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
    """

    force_matrix_w: torch.Tensor | None = None
    """The normal contact forces filtered between the sensor bodies and filtered bodies in world frame.

    Shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    and M is the number of filtered bodies.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
    """
    """传感器体和世界框架中的过体之间的正常接触力。

    形状是 (N， B， M， 3)，其中N是传感器的数量，B是每个传感器中的体体数，M是过体数。

    说明：
        如果:attr:`ContactSensorCfg.filter_prim_paths_expr`是空的，那么这个数量是None。
    """

    force_matrix_w_history: torch.Tensor | None = None
    """The normal contact forces filtered between the sensor bodies and filtered bodies in world frame.

    Shape is (N, T, B, M, 3), where N is the number of sensors, T is the configured history length,
    B is number of bodies in each sensor and M is the number of filtered bodies.

    In the history dimension, the first index is the most recent and the last index is the oldest.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
    """
    """传感器体和世界框架中的过体之间的正常接触力。

    形状是 (N，T，B，M，3)，其中N是传感器的数量，T是配置的历史长度，B是每个传感器中的体体数，M是过体数。

    在历史层面上，第一个索引是最新的，最后一个索引是最古老的。

    说明：
        如果:attr:`ContactSensorCfg.filter_prim_paths_expr`是空的，那么这个数量是None。
    """

    last_air_time: torch.Tensor | None = None
    """Time spent (in s) in the air before the last contact.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """
    """在最后一次接触之前在空气中度过的时间。

    形状是 (N，B)，其中N是传感器的数量，B是每个传感器的体体数量。

    说明：
        如果:attr:`ContactSensorCfg.track_air_time`是False，那么这个数量是None。
    """

    current_air_time: torch.Tensor | None = None
    """Time spent (in s) in the air since the last detach.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """
    """自最后一次脱离以来，空中时间 (s)。

    形状是 (N，B)，其中N是传感器的数量，B是每个传感器的体体数量。

    说明：
        如果:attr:`ContactSensorCfg.track_air_time`是False，那么这个数量是None。
    """

    last_contact_time: torch.Tensor | None = None
    """Time spent (in s) in contact before the last detach.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """
    """在最后一次脱离之前接触时间 (s)。

    形状是 (N，B)，其中N是传感器的数量，B是每个传感器的体体数量。

    说明：
        如果:attr:`ContactSensorCfg.track_air_time`是False，那么这个数量是None。
    """

    current_contact_time: torch.Tensor | None = None
    """Time spent (in s) in contact since the last contact.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """
    """自最后一次接触以来，接触时间 (s)。

    形状是 (N，B)，其中N是传感器的数量，B是每个传感器的体体数量。

    说明：
        如果:attr:`ContactSensorCfg.track_air_time`是False，那么这个数量是None。
    """
