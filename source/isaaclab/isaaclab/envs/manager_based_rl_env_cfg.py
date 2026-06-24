# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from .manager_based_env_cfg import ManagerBasedEnvCfg
from .ui import ManagerBasedRLEnvWindow


@configclass
class ManagerBasedRLEnvCfg(ManagerBasedEnvCfg):
    """Configuration for a reinforcement learning environment with the manager-based workflow."""
    """设置强化学习环境，使用基于管理器的工作流。"""

    # ui settings
    ui_window_class_type: type | None = ManagerBasedRLEnvWindow

    # general settings
    is_finite_horizon: bool = False
    """Whether the learning task is treated as a finite or infinite horizon problem for the agent.
    Defaults to False, which means the task is treated as an infinite horizon problem.

    This flag handles the subtleties of finite and infinite horizon tasks:

    * **Finite horizon**: no penalty or bootstrapping value is required by the the agent for
      running out of time. However, the environment still needs to terminate the episode after the
      time limit is reached.
    * **Infinite horizon**: the agent needs to bootstrap the value of the state at the end of the episode.
      This is done by sending a time-limit (or truncated) done signal to the agent, which triggers this
      bootstrapping calculation.

    If True, then the environment is treated as a finite horizon problem and no time-out (or truncated) done signal
    is sent to the agent. If False, then the environment is treated as an infinite horizon problem and a time-out
    (or truncated) done signal is sent to the agent.

    Note:
        The base :class:`ManagerBasedRLEnv` class does not use this flag directly. It is used by the environment
        wrappers to determine what type of done signal to send to the corresponding learning agent.
    """
    """学习任务是否被对待为代理人有限或无限水平问题。
    默认对 False，这意味着任务被视为无限视界问题。

    这一旗处理了有限和无限地平线任务的细节:

    * **终极视野**:经纪人不需要罚款或启动值，因为时间过去了。
    * **无限地平线**:经纪人需要在剧集结束时启动状态值.这通过向经纪人发送截止时间 (或缩短) 的完成信号来实现。

    如果True，则环境被视为有限视野问题，并且没有时间休息 (或缩短) 的信号被发送给代理。
    如果False，则环境被视为无限地平线问题，并发送时间停止 (或缩短) 的信号给代理。

    说明：
        基本:class:`ManagerBasedRLEnv`类不直接使用此标志。
        环境包装器使用它来确定向相应的学习代理发送的完成信号类型。
    """

    episode_length_s: float = MISSING
    """Duration of an episode (in seconds).

    Based on the decimation rate and physics time step, the episode length is calculated as:

    .. code-block:: python

        episode_length_steps = ceil(episode_length_s / (decimation_rate * physics_time_step))

    For example, if the decimation rate is 10, the physics time step is 0.01, and the episode length is 10 seconds,
    then the episode length in steps is 100.
    """
    """一个事件的持续时间 (在秒钟内)。

    根据化速度和物理时间步骤，回合长度计算为:

    .. code-block:: python

        episode_length_steps = ceil(episode_length_s / (decimation_rate * physics_time_step))

    比如，如果十分的速度是10，物理时间步骤是0.01，回合的长度是10秒，然后回合的长度是100步。
    """

    # environment settings
    rewards: object = MISSING
    """Reward settings.

    Please refer to the :class:`isaaclab.managers.RewardManager` class for more details.
    """
    """奖励设置。

    详细请参阅:class:`isaaclab.managers.RewardManager`类。
    """

    terminations: object = MISSING
    """Termination settings.

    Please refer to the :class:`isaaclab.managers.TerminationManager` class for more details.
    """
    """终止设置。

    详细请参阅:class:`isaaclab.managers.TerminationManager`类。
    """

    curriculum: object | None = None
    """Curriculum settings. Defaults to None, in which case no curriculum is applied.

    Please refer to the :class:`isaaclab.managers.CurriculumManager` class for more details.
    """
    """课程设置。
    在 None 中，默认情况下没有应用课程。

    详细请参阅:class:`isaaclab.managers.CurriculumManager`类。
    """

    commands: object | None = None
    """Command settings. Defaults to None, in which case no commands are generated.

    Please refer to the :class:`isaaclab.managers.CommandManager` class for more details.
    """
    """命令设置。
    在 None 中，默认情况下不会生成命令。

    详细请参阅:class:`isaaclab.managers.CommandManager`类。
    """
