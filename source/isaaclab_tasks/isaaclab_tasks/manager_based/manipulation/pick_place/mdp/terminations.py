# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations
"""常见功能，可用于激活某些升降任务的终止。

函数可以传递到:class:`isaaclab.managers.TerminationTermCfg`对象，以实现函数引入的终止。
"""

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done_pick_place(
    env: ManagerBasedRLEnv,
    task_link_name: str = "",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_wrist_max_x: float = 0.26,
    min_x: float = 0.40,
    max_x: float = 0.85,
    min_y: float = 0.35,
    max_y: float = 0.60,
    max_height: float = 1.10,
    min_vel: float = 0.20,
) -> torch.Tensor:
    """Determine if the object placement task is complete.

    This function checks whether all success conditions for the task have been met:
    1. object is within the target x/y range
    2. object is below a minimum height
    3. object velocity is below threshold
    4. Right robot wrist is retracted back towards body (past a given x pos threshold)

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object entity.
        right_wrist_max_x: Maximum x position of the right wrist for task completion.
        min_x: Minimum x position of the object for task completion.
        max_x: Maximum x position of the object for task completion.
        min_y: Minimum y position of the object for task completion.
        max_y: Maximum y position of the object for task completion.
        max_height: Maximum height (z position) of the object for task completion.
        min_vel: Minimum velocity magnitude of the object for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    """确定对象放置任务是否完成。

    该函数检查任务是否符合所有成功条件:
    1. 对象在目标 x/y 范围内
    2. 对象低于最低高度
    3. 对象的速度低于门
    4. 右机器人手腕向身体拉回 (经过给定的x后门)

    参数：
        env: 在RL环境实例。
        object_cfg: 对象实体的配置
        right_wrist_max_x: 完成任务的右手腕最大x位置。
        min_x: 任务完成时对象的最小x位置。
        max_x: 任务完成时对象的最大x位置。
        min_y: 任务完成时对象的最小y位置。
        max_y: 任务完成时对象的最大位置。
        max_height: 任务完成时对象的最大高度 (z位置)。
        min_vel: 完成任务的对象最小速度大小。

    返回：
        布尔电压器表示哪些环境完成任务。
    """
    if task_link_name == "":
        raise ValueError("task_link_name must be provided to task_done_pick_place")

    # Get object entity from the scene
    object: RigidObject = env.scene[object_cfg.name]

    # Extract wheel position relative to environment origin
    object_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    object_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    object_height = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    object_vel = torch.abs(object.data.root_vel_w)

    # Get right wrist position relative to environment origin
    robot_body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index(task_link_name)
    right_wrist_x = robot_body_pos_w[:, right_eef_idx, 0] - env.scene.env_origins[:, 0]

    # Check all success conditions and combine with logical AND
    done = object_x < max_x
    done = torch.logical_and(done, object_x > min_x)
    done = torch.logical_and(done, object_y < max_y)
    done = torch.logical_and(done, object_y > min_y)
    done = torch.logical_and(done, object_height < max_height)
    done = torch.logical_and(done, right_wrist_x < right_wrist_max_x)
    done = torch.logical_and(done, object_vel[:, 0] < min_vel)
    done = torch.logical_and(done, object_vel[:, 1] < min_vel)
    done = torch.logical_and(done, object_vel[:, 2] < min_vel)

    return done


def task_done_nut_pour(
    env: ManagerBasedRLEnv,
    sorting_scale_cfg: SceneEntityCfg = SceneEntityCfg("sorting_scale"),
    sorting_bowl_cfg: SceneEntityCfg = SceneEntityCfg("sorting_bowl"),
    sorting_beaker_cfg: SceneEntityCfg = SceneEntityCfg("sorting_beaker"),
    factory_nut_cfg: SceneEntityCfg = SceneEntityCfg("factory_nut"),
    sorting_bin_cfg: SceneEntityCfg = SceneEntityCfg("black_sorting_bin"),
    max_bowl_to_scale_x: float = 0.055,
    max_bowl_to_scale_y: float = 0.055,
    max_bowl_to_scale_z: float = 0.025,
    max_nut_to_bowl_x: float = 0.050,
    max_nut_to_bowl_y: float = 0.050,
    max_nut_to_bowl_z: float = 0.019,
    max_beaker_to_bin_x: float = 0.08,
    max_beaker_to_bin_y: float = 0.12,
    max_beaker_to_bin_z: float = 0.07,
) -> torch.Tensor:
    """Determine if the nut pouring task is complete.

    This function checks whether all success conditions for the task have been met:
    1. The factory nut is in the sorting bowl
    2. The sorting beaker is in the sorting bin
    3. The sorting bowl is placed on the sorting scale

    Args:
        env: The RL environment instance.
        sorting_scale_cfg: Configuration for the sorting scale entity.
        sorting_bowl_cfg: Configuration for the sorting bowl entity.
        sorting_beaker_cfg: Configuration for the sorting beaker entity.
        factory_nut_cfg: Configuration for the factory nut entity.
        sorting_bin_cfg: Configuration for the sorting bin entity.
        max_bowl_to_scale_x: Maximum x position of the sorting bowl relative to the sorting scale for task completion.
        max_bowl_to_scale_y: Maximum y position of the sorting bowl relative to the sorting scale for task completion.
        max_bowl_to_scale_z: Maximum z position of the sorting bowl relative to the sorting scale for task completion.
        max_nut_to_bowl_x: Maximum x position of the factory nut relative to the sorting bowl for task completion.
        max_nut_to_bowl_y: Maximum y position of the factory nut relative to the sorting bowl for task completion.
        max_nut_to_bowl_z: Maximum z position of the factory nut relative to the sorting bowl for task completion.
        max_beaker_to_bin_x: Maximum x position of the sorting beaker relative to the sorting bin for task completion.
        max_beaker_to_bin_y: Maximum y position of the sorting beaker relative to the sorting bin for task completion.
        max_beaker_to_bin_z: Maximum z position of the sorting beaker relative to the sorting bin for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    """确定坚果倒任务是否完成。

    该函数检查任务是否符合所有成功条件:
    1. 工厂的 factory子在分类碗里
    2. 排序杯在排序箱里
    3. 排序碗放在排序尺度上

    参数：
        env: 在RL环境实例。
        sorting_scale_cfg: 对分类规模实体的配置。
        sorting_bowl_cfg: 排序碗实体的配置。
        sorting_beaker_cfg: 对分类杯实体的配置。
        factory_nut_cfg: 工厂 entity蛋实体的配置。
        sorting_bin_cfg: 排序桶实体的配置。
        max_bowl_to_scale_x: 排序碗的最大x位置与任务完成的排序规模相比。
        max_bowl_to_scale_y: 排序碗的最大位置与任务完成的排序规模相比。
        max_bowl_to_scale_z: 排序碗的最大z位置与任务完成的排序规模相比。
        max_nut_to_bowl_x: 工厂坚果对分类碗进行完成任务的最大x位置。
        max_nut_to_bowl_y: 工厂核子与分类碗相比的最大位置为完成任务。
        max_nut_to_bowl_z: 工厂核子对分类碗的最大z位置，以完成任务。
        max_beaker_to_bin_x: 排序杯与排序箱相比的最大x位置，以完成任务。
        max_beaker_to_bin_y: 排序杯与排序箱相比的最大位置为完成任务。
        max_beaker_to_bin_z: 排序杯与排序箱相比的最大z位置完成任务。

    返回：
        布尔电压器表示哪些环境完成任务。
    """
    # Get object entities from the scene
    sorting_scale: RigidObject = env.scene[sorting_scale_cfg.name]
    sorting_bowl: RigidObject = env.scene[sorting_bowl_cfg.name]
    factory_nut: RigidObject = env.scene[factory_nut_cfg.name]
    sorting_beaker: RigidObject = env.scene[sorting_beaker_cfg.name]
    sorting_bin: RigidObject = env.scene[sorting_bin_cfg.name]

    # Get positions relative to environment origin
    scale_pos = sorting_scale.data.root_pos_w - env.scene.env_origins
    bowl_pos = sorting_bowl.data.root_pos_w - env.scene.env_origins
    sorting_beaker_pos = sorting_beaker.data.root_pos_w - env.scene.env_origins
    nut_pos = factory_nut.data.root_pos_w - env.scene.env_origins
    bin_pos = sorting_bin.data.root_pos_w - env.scene.env_origins

    # nut to bowl
    nut_to_bowl_x = torch.abs(nut_pos[:, 0] - bowl_pos[:, 0])
    nut_to_bowl_y = torch.abs(nut_pos[:, 1] - bowl_pos[:, 1])
    nut_to_bowl_z = nut_pos[:, 2] - bowl_pos[:, 2]

    # bowl to scale
    bowl_to_scale_x = torch.abs(bowl_pos[:, 0] - scale_pos[:, 0])
    bowl_to_scale_y = torch.abs(bowl_pos[:, 1] - scale_pos[:, 1])
    bowl_to_scale_z = bowl_pos[:, 2] - scale_pos[:, 2]

    # beaker to bin
    beaker_to_bin_x = torch.abs(sorting_beaker_pos[:, 0] - bin_pos[:, 0])
    beaker_to_bin_y = torch.abs(sorting_beaker_pos[:, 1] - bin_pos[:, 1])
    beaker_to_bin_z = sorting_beaker_pos[:, 2] - bin_pos[:, 2]

    done = nut_to_bowl_x < max_nut_to_bowl_x
    done = torch.logical_and(done, nut_to_bowl_y < max_nut_to_bowl_y)
    done = torch.logical_and(done, nut_to_bowl_z < max_nut_to_bowl_z)
    done = torch.logical_and(done, bowl_to_scale_x < max_bowl_to_scale_x)
    done = torch.logical_and(done, bowl_to_scale_y < max_bowl_to_scale_y)
    done = torch.logical_and(done, bowl_to_scale_z < max_bowl_to_scale_z)
    done = torch.logical_and(done, beaker_to_bin_x < max_beaker_to_bin_x)
    done = torch.logical_and(done, beaker_to_bin_y < max_beaker_to_bin_y)
    done = torch.logical_and(done, beaker_to_bin_z < max_beaker_to_bin_z)

    return done


def task_done_exhaust_pipe(
    env: ManagerBasedRLEnv,
    blue_exhaust_pipe_cfg: SceneEntityCfg = SceneEntityCfg("blue_exhaust_pipe"),
    blue_sorting_bin_cfg: SceneEntityCfg = SceneEntityCfg("blue_sorting_bin"),
    max_blue_exhaust_to_bin_x: float = 0.085,
    max_blue_exhaust_to_bin_y: float = 0.200,
    min_blue_exhaust_to_bin_y: float = -0.090,
    max_blue_exhaust_to_bin_z: float = 0.070,
) -> torch.Tensor:
    """Determine if the exhaust pipe task is complete.

    This function checks whether all success conditions for the task have been met:
    1. The blue exhaust pipe is placed in the correct position

    Args:
        env: The RL environment instance.
        blue_exhaust_pipe_cfg: Configuration for the blue exhaust pipe entity.
        blue_sorting_bin_cfg: Configuration for the blue sorting bin entity.
        max_blue_exhaust_to_bin_x: Maximum x position of the blue exhaust pipe
            relative to the blue sorting bin for task completion.
        max_blue_exhaust_to_bin_y: Maximum y position of the blue exhaust pipe
            relative to the blue sorting bin for task completion.
        max_blue_exhaust_to_bin_z: Maximum z position of the blue exhaust pipe
            relative to the blue sorting bin for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    """确定排气管任务是否完成。

    该函数检查任务是否符合所有成功条件:
    1. 蓝色排气管放在正确的位置

    参数：
        env: 在RL环境实例。
        blue_exhaust_pipe_cfg: 蓝色排气管的配置
        blue_sorting_bin_cfg: 蓝色分类垃圾桶实体的配置。
        max_blue_exhaust_to_bin_x: 蓝色排气管与蓝色分类箱相对的最大x位置，以便完成任务。
        max_blue_exhaust_to_bin_y: 蓝色排气管与蓝色分类箱相比的最大位置为完成任务。
        max_blue_exhaust_to_bin_z: 蓝色排气管与蓝色分类箱相比的最大z位置，以便完成任务。

    返回：
        布尔电压器表示哪些环境完成任务。
    """
    # Get object entities from the scene
    blue_exhaust_pipe: RigidObject = env.scene[blue_exhaust_pipe_cfg.name]
    blue_sorting_bin: RigidObject = env.scene[blue_sorting_bin_cfg.name]

    # Get positions relative to environment origin
    blue_exhaust_pipe_pos = blue_exhaust_pipe.data.root_pos_w - env.scene.env_origins
    blue_sorting_bin_pos = blue_sorting_bin.data.root_pos_w - env.scene.env_origins

    # blue exhaust to bin
    blue_exhaust_to_bin_x = torch.abs(blue_exhaust_pipe_pos[:, 0] - blue_sorting_bin_pos[:, 0])
    blue_exhaust_to_bin_y = blue_exhaust_pipe_pos[:, 1] - blue_sorting_bin_pos[:, 1]
    blue_exhaust_to_bin_z = blue_exhaust_pipe_pos[:, 2] - blue_sorting_bin_pos[:, 2]

    done = blue_exhaust_to_bin_x < max_blue_exhaust_to_bin_x
    done = torch.logical_and(done, blue_exhaust_to_bin_y < max_blue_exhaust_to_bin_y)
    done = torch.logical_and(done, blue_exhaust_to_bin_y > min_blue_exhaust_to_bin_y)
    done = torch.logical_and(done, blue_exhaust_to_bin_z < max_blue_exhaust_to_bin_z)

    return done
