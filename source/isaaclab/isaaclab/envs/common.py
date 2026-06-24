# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Dict, Literal, TypeVar  # noqa: UP035

import gymnasium as gym
import torch

from isaaclab.utils import configclass

##
# Configuration.
##


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""
    """场景视角摄像机的配置。"""

    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Initial camera position (in m). Default is (7.5, 7.5, 7.5)."""
    """摄像头的初始位置 (m)。
    默认是 (7.5，7.5，7.5)。
    """

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target position (in m). Default is (0.0, 0.0, 0.0)."""
    """摄像头初始目标位置 (m)。
    默认是 (0.0，0.0，0.0)。
    """

    cam_prim_path: str = "/OmniverseKit_Persp"
    """The camera prim path to record images from. Default is "/OmniverseKit_Persp",
    which is the default camera in the viewport.
    """
    """摄像头prim路径记录图像。
    默认是"/OmniverseKit_Persp"，这是视角中的默认相机。
    """

    resolution: tuple[int, int] = (1280, 720)
    """The resolution (width, height) of the camera specified using :attr:`cam_prim_path`.
    Default is (1280, 720).
    """
    """使用:attr:`cam_prim_path`指定的相机的分辨率 (宽度，高度)。
    默认是 (1280， 720)。
    """

    origin_type: Literal["world", "env", "asset_root", "asset_body"] = "world"
    """The frame in which the camera position (eye) and target (lookat) are defined in. Default is "world".

    Available options are:

    * ``"world"``: The origin of the world.
    * ``"env"``: The origin of the environment defined by :attr:`env_index`.
    * ``"asset_root"``: The center of the asset defined by :attr:`asset_name` in environment :attr:`env_index`.
    * ``"asset_body"``: The center of the body defined by :attr:`body_name` in asset defined by
      :attr:`asset_name` in environment :attr:`env_index`.
    """
    """摄像机位置 (眼睛) 和目标 (视角) 所定义的框架。
    默认是"世界"。

    可供选择的是:

    * ``"world"``世界的起源。
    * ``"env"``:由:attr:`env_index`定义的环境的起源。
    * ``"asset_root"``:在环境 :attr:`env_index` 中，由 :attr:`asset_name` 定义的资产的中心。
    * ``"asset_body"``: 身体的中心由:attr:`body_name`在由:attr:`asset_name`在环境中:attr:`env_index`。
    """

    env_index: int = 0
    """The environment index for frame origin. Default is 0.

    This quantity is only effective if :attr:`origin` is set to "env" or "asset_root".
    """
    """框架来源的环境索引。
    默认是0。

    如果 :attr:`origin` 设置为"env"或"asset_root"，该数量才有效。
    """

    asset_name: str | None = None
    """The asset name in the interactive scene for the frame origin. Default is None.

    This quantity is only effective if :attr:`origin` is set to "asset_root".
    """
    """交互式场景中的资产名称
    默认是None。

    如果设置:attr:`origin`为"asset_root"，这个数量才有效。
    """

    body_name: str | None = None
    """The name of the body in :attr:`asset_name` in the interactive scene for the frame origin. Default is None.

    This quantity is only effective if :attr:`origin` is set to "asset_body".
    """
    """在交互场景中:attr:`asset_name`中的体体名称，用于框架起源。
    默认是None。

    如果设置:attr:`origin`为"asset_body"，这个数量才有效。
    """


##
# Types.
##

SpaceType = TypeVar("SpaceType", gym.spaces.Space, int, set, tuple, list, dict)
"""A sentinel object to indicate a valid space type to specify states, observations and actions."""
"""一个哨兵对象，以指示一个有效的空间类型，以指定状态，观测和动作。"""

VecEnvObs = Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
"""Observation returned by the environment.

The observations are stored in a dictionary. The keys are the group to which the observations belong.
This is useful for various setups such as reinforcement learning with asymmetric actor-critic or
multi-agent learning. For non-learning paradigms, this may include observations for different components
of a system.

Within each group, the observations can be stored either as a dictionary with keys as the names of each
observation term in the group, or a single tensor obtained from concatenating all the observation terms.
For example, for asymmetric actor-critic, the observation for the actor and the critic can be accessed
using the keys ``"policy"`` and ``"critic"`` respectively.

Note:
    By default, most learning frameworks deal with default and privileged observations in different ways.
    This handling must be taken care of by the wrapper around the :class:`ManagerBasedRLEnv` instance.

    For included frameworks (RSL-RL, RL-Games, skrl), the observations must have the key "policy". In case,
    the key "critic" is also present, then the critic observations are taken from the "critic" group.
    Otherwise, they are the same as the "policy" group.

"""
"""环境的观测。

这些观测记录在字典中。
关键是观测所属于的群体。
这对于各种设置有用，例如强化学习与不对称的演员批判性或多代理学习。
对于非学习范式，这可能包括对系统的不同组件的观测。

在每个组内，观测可以作为一个字典存储，其关键是组中每个观测项的名称，或者是从连接所有观测项中获得的单个子。
例如，对于不对称的演员-批评者，可以使用分别的键``"policy"``和``"critic"``访问演员和批评者的观测。

说明：
    默认情况下，大多数学习框架以不同的方式处理默认和特权观测。
    这种处理必须由:class:`ManagerBasedRLEnv`实例周围的包装处理。

    对于包含的框架 (RSL-RL，RL-Games， skrl)，观测必须具有关键的"策略"。
    如果关键的"批评者"也存在，那么批评的观测是从"批评者"群中取出的。
    否则，它们与"策略"群体相同。
"""

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]
"""The environment signals processed at the end of each step.

The tuple contains batched information for each sub-environment. The information is stored in the following order:

1. **Observations**: The observations from the environment.
2. **Rewards**: The rewards from the environment.
3. **Terminated Dones**: Whether the environment reached a terminal state, such as task success or robot falling etc.
4. **Timeout Dones**: Whether the environment reached a timeout state, such as end of max episode length.
5. **Extras**: A dictionary containing additional information from the environment.
"""
"""每一步结束时处理的环境信号。

元组包含每个子环境的批量信息。
信息按下列顺序存储:

1. **观测**:来自环境的观测。
2. **奖励**:环境的奖励。
3. **已完成的Dones**:环境是否达到终端状态，如任务成功或机器人摔倒等。
4. **Timeout Dones**:环境是否达到时间休止状态，例如最长集段时间结束。
5. **额外**:包含环境的额外信息的字典。
"""

AgentID = TypeVar("AgentID")
"""Unique identifier for an agent within a multi-agent environment.

The identifier has to be an immutable object, typically a string (e.g.: ``"agent_0"``).
"""
"""在多代理环境中的代理人的唯一标识符。

标识符必须是一个不可变的对象，通常是一个字符串 (e.g.:``"agent_0"``)。
"""

ObsType = TypeVar("ObsType", torch.Tensor, Dict[str, torch.Tensor])
"""A sentinel object to indicate the data type of the observation.
"""
"""一个哨兵对象，以指示观测的数据类型。
"""

ActionType = TypeVar("ActionType", torch.Tensor, Dict[str, torch.Tensor])
"""A sentinel object to indicate the data type of the action.
"""
"""一个哨兵对象表示动作的数据类型。
"""

StateType = TypeVar("StateType", torch.Tensor, dict)
"""A sentinel object to indicate the data type of the state.
"""
"""监视器对象表示状态的数据类型。
"""

EnvStepReturn = tuple[
    Dict[AgentID, ObsType],
    Dict[AgentID, torch.Tensor],
    Dict[AgentID, torch.Tensor],
    Dict[AgentID, torch.Tensor],
    Dict[AgentID, dict],
]
"""The environment signals processed at the end of each step.

The tuple contains batched information for each sub-environment (keyed by the agent ID).
The information is stored in the following order:

1. **Observations**: The observations from the environment.
2. **Rewards**: The rewards from the environment.
3. **Terminated Dones**: Whether the environment reached a terminal state, such as task success or robot falling etc.
4. **Timeout Dones**: Whether the environment reached a timeout state, such as end of max episode length.
5. **Extras**: A dictionary containing additional information from the environment.
"""
"""每一步结束时处理的环境信号。

元组包含每个子环境的批量信息 (由ID剂关键)。
信息按下列顺序存储:

1. **观测**:来自环境的观测。
2. **奖励**:环境的奖励。
3. **已完成的Dones**:环境是否达到终端状态，如任务成功或机器人摔倒等。
4. **Timeout Dones**:环境是否达到时间休止状态，例如最长集段时间结束。
5. **额外**:包含环境的额外信息的字典。
"""
