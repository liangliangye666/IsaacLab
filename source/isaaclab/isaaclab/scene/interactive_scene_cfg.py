# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass


@configclass
class InteractiveSceneCfg:
    """Configuration for the interactive scene.

    The users can inherit from this class to add entities to their scene. This is then parsed by the
    :class:`InteractiveScene` class to create the scene.

    .. note::
        The adding of entities to the scene is sensitive to the order of the attributes in the configuration.
        Please make sure to add the entities in the order you want them to be added to the scene.
        The recommended order of specification is terrain, physics-related assets (articulations and rigid bodies),
        sensors and non-physics-related assets (lights).

    For example, to add a robot to the scene, the user can create a configuration class as follows:

    .. code-block:: python

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors.ray_caster import GridPatternCfg, RayCasterCfg
        from isaaclab.utils import configclass

        from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


        @configclass
        class MySceneCfg(InteractiveSceneCfg):
            # terrain - flat terrain plane
            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
            )

            # articulation - robot 1
            robot_1 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
            # articulation - robot 2
            robot_2 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
            robot_2.init_state.pos = (0.0, 1.0, 0.6)

            # sensor - ray caster attached to the base of robot 1 that scans the ground
            height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot_1/base",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
                debug_vis=True,
                mesh_prim_paths=["/World/ground"],
            )

            # extras - light
            light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
            )

    """
    """交互场景的配置。

    用户可以从这个类中继承到他们的场景。
    然后，:class:`InteractiveScene`类分析了这个情况。

    .. 说明::
        增加实体到场景对配置中的属性顺序敏感。
        请确保将实体添加到你想要添加到场景的顺序。
        推的规范顺序是地形，物理相关资产 (关节和硬体)，传感器和非物理相关资产 (灯)。

    例如，为了将机器人添加到场景中，用户可以创建一个配置类，如下:

    .. code-block:: python

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors.ray_caster import GridPatternCfg, RayCasterCfg
        from isaaclab.utils import configclass

        from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


        @configclass
        class MySceneCfg(InteractiveSceneCfg):
            # terrain - flat terrain plane
            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
            )

            # articulation - robot 1
            robot_1 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
            # articulation - robot 2
            robot_2 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
            robot_2.init_state.pos = (0.0, 1.0, 0.6)

            # sensor - ray caster attached to the base of robot 1 that scans the ground
            height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot_1/base",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
                debug_vis=True,
                mesh_prim_paths=["/World/ground"],
            )

            # extras - light
            light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
            )
    """

    num_envs: int = MISSING
    """Number of environment instances handled by the scene."""
    """场景处理的环境实例数。"""

    env_spacing: float = MISSING
    """Spacing between environments.

    This is the default distance between environment origins in the scene. Used only when the
    number of environments is greater than one.
    """
    """环境之间的距离。

    这就是场景环境起源之间的默认距离。
    只有在环境数量超过一个时使用。
    """

    lazy_sensor_update: bool = True
    """Whether to update sensors only when they are accessed. Default is True.

    If true, the sensor data is only updated when their attribute ``data`` is accessed. Otherwise, the sensor
    data is updated every time sensors are updated.
    """
    """仅在访问时更新传感器。
    默认是True。

    如果是正确的，传感器数据只会在访问其属性``data``时更新。
    否则，每次更新传感器，传感器数据都会更新。
    """

    replicate_physics: bool = True
    """Enable/disable replication of physics schemas when using the Cloner APIs. Default is True.

    If True, the simulation will have the same asset instances (USD prims) in all the cloned environments.
    Internally, this ensures optimization in setting up the scene and parsing it via the physics stage parser.

    If False, the simulation allows having separate asset instances (USD prims) in each environment.
    This flexibility comes at a cost of slowdowns in setting up and parsing the scene.

    .. note::
        Optimized parsing of certain prim types (such as deformable objects) is not currently supported
        by the physics engine. In these cases, this flag needs to be set to False.
    """
    """在使用Cloner APIs时，可以/不能复制物理方案。
    默认是True。

    如果 True，仿真将在所有克隆环境中具有相同的资产实例 (USD prims)。
    在内部，这确保了设置场景的优化，并通过物理阶段解析器进行分析。

    如果是False，则仿真允许每个环境中设有单独的资产实例 (USD prims)。
    这种灵活性是由于场景设置和分析的放缓。

    .. 说明::
        目前，物理引擎不支持对某些prim类型的优化解析 (如可变形物体)。
        在这些情况下，该旗必须设置为False。
    """

    filter_collisions: bool = True
    """Enable/disable collision filtering between cloned environments. Default is True.

    If True, collisions will not occur between cloned environments.

    If False, the simulation will generate collisions between environments.

    .. note::
        Collisions can only be filtered automatically in direct workflows when physics replication is enabled.
        If :attr:`replicated_physics` is ``False`` and collision filtering is desired, make sure to call
        ``scene.filter_collisions()``.
    """
    """在克隆环境之间启用/禁用碰撞过。
    默认是True。

    如果True，克隆环境之间不会发生碰撞。

    如果False，仿真将产生环境之间的碰撞。

    .. 说明::
        在直接工作流中，只有在启用物理复制时才能自动过碰撞。
        If :attr:`replicated_physics`是``False``和碰撞过是希望的，确保打电话
        ``scene.filter_collisions()``。
    """

    clone_in_fabric: bool = False
    """Enable/disable cloning in fabric. Default is False.

    Omniverse Fabric is a more optimized method for performing cloning in scene creation. This reduces the time
    taken to create the scene. However, it limits flexibility in accessing the stage through USD APIs and instead,
    the stage must be accessed through USDRT.

    .. note::
        Cloning in fabric can only be enabled if :attr:`replicated_physics` is also enabled.
        If :attr:`replicated_physics` is ``False``, cloning in Fabric will automatically
        default to ``False``.

    """
    """在布料中启用/禁用克隆。
    默认是False。

    环球织物是一种更优化的方法，用于在场景创作中进行克隆。
    这减少了制作场景的时间。
    然而，它限制了通过USD APIs进入舞台的灵活性，而不是通过USDRT进入舞台。

    .. 说明::
        只有在:attr:`replicated_physics`也被启用时才能在织物中克隆。
        If :attr:`replicated_physics`是``False``，在布料中克隆将自动
        默认为 ``False``。
    """
