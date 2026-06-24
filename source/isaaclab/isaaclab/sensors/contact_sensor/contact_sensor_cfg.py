# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import CONTACT_SENSOR_MARKER_CFG
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .contact_sensor import ContactSensor


@configclass
class ContactSensorCfg(SensorBaseCfg):
    """Configuration for the contact sensor."""
    """接触传感器的配置。"""

    class_type: type = ContactSensor

    track_pose: bool = False
    """Whether to track the pose of the sensor's origin. Defaults to False."""
    """是否追踪传感器起源的姿势。
    默认为 False。
    """

    track_contact_points: bool = False
    """Whether to track the contact point locations. Defaults to False."""
    """是否追踪接触点的位置。
    默认为 False。
    """

    track_friction_forces: bool = False
    """Whether to track the friction forces at the contact points. Defaults to False."""
    """如何跟踪接触点的摩擦力。
    默认为 False。
    """

    max_contact_data_count_per_prim: int = 4
    """The maximum number of contacts across all batches of the sensor to keep track of. Default is 4.

    This parameter sets the total maximum counts of the simulation across all bodies and environments. The total number
    of contacts allowed is max_contact_data_count_per_prim*num_envs*num_sensor_bodies.

    .. note::

        If the environment is very contact rich it is suggested to increase this parameter to avoid out of bounds memory
        errors and loss of contact data leading to inaccurate measurements.

        """
    """传感器所有批次的最大接触数量
    默认是4。

    这一参数设定了所有体体和环境中仿真的总最大数量。
    允许的联系人总数为max_contact_data_count_per_prim*num_envs*num_sensor_bodies。

    .. 说明::

        如果环境非常接触丰富，建议增加这个参数，以避免超出界限的记忆错误和导致不准确测量的接触数据丢失。
    """

    track_air_time: bool = False
    """Whether to track the air/contact time of the bodies (time between contacts). Defaults to False."""
    """是否追踪机体的空气/接触时间 (接触之间的时间)。
    默认为 False。
    """

    force_threshold: float = 1.0
    """The threshold on the norm of the contact force that determines whether two bodies are in collision or not.

    This value is only used for tracking the mode duration (the time in contact or in air),
    if :attr:`track_air_time` is True.
    """
    """接触力标准的门值，确定两体是否碰撞。

    这一值仅用于跟踪模式持续时间 (接触或空气时间)，
    if :attr:`track_air_time` is True.
    """

    filter_prim_paths_expr: list[str] = list()
    """The list of primitive paths (or expressions) to filter contacts with. Defaults to an empty list, in which case
    no filtering is applied.

    The contact sensor allows reporting contacts between the primitive specified with :attr:`prim_path` and
    other primitives in the scene. For instance, in a scene containing a robot, a ground plane and an object,
    you can obtain individual contact reports of the base of the robot with the ground plane and the object.

    .. note::
        The expression in the list can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Object`` will be replaced with ``/World/envs/env_.*/Object``.

    .. attention::
        The reporting of filtered contacts only works when the sensor primitive :attr:`prim_path` corresponds to a
        single primitive in that environment. If the sensor primitive corresponds to multiple primitives, the
        filtering will not work as expected. Please check :class:`~isaaclab.sensors.contact_sensor.ContactSensor`
        for more details.
        If track_contact_points is true, then filter_prim_paths_expr cannot be an empty list!
    """
    """清单原始路径 (或表达式) 过联系人。
    默认的空清单，在这种情况下没有过。

    接触传感器允许在:attr:`prim_path`所指定的原始和场景中的其他原始之间的联系报告。
    例如，在包含机器人，地面平面和物体的场景中，你可以获得机器人的基础与地面平面和物体的个人接触报告。

    .. 说明::
        列表中的表达式可以包含环境命名空间regex ``{ENV_REGEX_NS}``，该表达式将被环境命名空间所取代。

        Example: ``{ENV_REGEX_NS}/Object``将被 ``/World/envs/env_.*/Object`` 取代。

    .. 注意::
        只有当传感器原始 :attr:`prim_path` 与该环境中的单个原始 :attr:`prim_path` 相符时，过接触的报告才有效。
        如果传感器原始对应多个原始， 过将不会像预期的那样工作。
        请检查:class:`~isaaclab.sensors.contact_sensor.ContactSensor`
        for more details.
        如果track_contact_points是真的，那么filter_prim_paths_expr不能是一个空清单!
    """

    visualizer_cfg: VisualizationMarkersCfg = CONTACT_SENSOR_MARKER_CFG.replace(prim_path="/Visuals/ContactSensor")
    """The configuration object for the visualization markers. Defaults to CONTACT_SENSOR_MARKER_CFG.

    .. note::
        This attribute is only used when debug visualization is enabled.
    """
    """视觉化标记的配置对象。
    在 CONTACT_SENSOR_MARKER_CFG 中默认错误。

    .. 说明::
        只有在启用调试可视化时才使用此属性。
    """
