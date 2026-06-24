# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing an interactive scene definition.

A scene is a collection of entities (e.g., terrain, articulations, sensors, lights, etc.) that can be added to the
simulation. However, only a subset of these entities are of direct interest for the user to interact with.
For example, the user may want to interact with a robot in the scene, but not with the terrain or the lights.
For this reason, we integrate the different entities into a single class called :class:`InteractiveScene`.

The interactive scene performs the following tasks:

1. It parses the configuration class :class:`InteractiveSceneCfg` to create the scene. This configuration class is
   inherited by the user to add entities to the scene.
2. It clones the entities based on the number of environments specified by the user.
3. It clubs the entities into different groups based on their type (e.g., articulations, sensors, etc.).
4. It provides a set of methods to unify the common operations on the entities in the scene (e.g., resetting internal
   buffers, writing buffers to simulation and updating buffers from simulation).

The interactive scene can be passed around to different modules in the framework to perform different tasks.
For instance, computing the observations based on the state of the scene, or randomizing the scene, or applying
actions to the scene. All these are handled by different "managers" in the framework. Please refer to the
:mod:`isaaclab.managers` sub-package for more details.
"""
"""包含交互场景定义的子包。

场景是可以添加到仿真中的实体 (e.g.，地形，关节，传感器，灯光等) 的集合。
然而，只有这些实体中的一小组对用户进行直接交互的利益。
例如，用户可能希望与场景机器人交互，
我们将不同的实体整合到一个叫做:class:`InteractiveScene`的类。

交互场景执行以下任务:

1. 它解析配置类:class:`InteractiveSceneCfg`来创建场景.这个配置类由用户继承，以添加实体到场景。
2. 它根据用户指定的环境数量克隆实体。
3. 它根据其类型 (e.g.，关节，传感器等) 将实体分为不同的组。
4. 它提供了一系列方法来统一场景实体的共同操作 (e.g.，重置内部缓冲器，写入缓冲器到仿真和更新缓冲器从仿真)。

交互场景可以在框架内转移到不同的模块来执行不同的任务。
例如，根据场景的状态计算观测，或随机化场景，或对场景应用动作。
所有这些都由框架内不同的"管理器"来处理。
详细请参阅:mod:`isaaclab.managers`子包。
"""

from .interactive_scene import InteractiveScene
from .interactive_scene_cfg import InteractiveSceneCfg
