# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.devices.openxr import XrCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModelCfg

from .common import AgentID, SpaceType, ViewerCfg
from .ui import BaseEnvWindow


@configclass
class DirectMARLEnvCfg:
    """Configuration for a MARL environment defined with the direct workflow.

    Please refer to the :class:`isaaclab.envs.direct_marl_env.DirectMARLEnv` class for more details.
    """
    """对于直接工作流定义的MARL环境的配置。

    详细请参阅:class:`isaaclab.envs.direct_marl_env.DirectMARLEnv`类。
    """

    # simulation settings
    viewer: ViewerCfg = ViewerCfg()
    """Viewer configuration. Default is ViewerCfg()."""
    """显示器配置
    默认是ViewerCfg()。
    """

    sim: SimulationCfg = SimulationCfg()
    """Physics simulation configuration. Default is SimulationCfg()."""
    """物理仿真配置。
    默认是SimulationCfg()。
    """

    # ui settings
    ui_window_class_type: type | None = BaseEnvWindow
    """The class type of the UI window. Default is None.

    If None, then no UI window is created.

    Note:
        If you want to make your own UI window, you can create a class that inherits from
        from :class:`isaaclab.envs.ui.base_env_window.BaseEnvWindow`. Then, you can set
        this attribute to your class type.
    """
    """在 UI 窗口的类型。
    默认是None。

    如果是None，则不会创建UI窗口。

    说明：
        如果你想创建自己的UI窗口，你可以创建一个继承从
        from :class:`isaaclab.envs.ui.base_env_window.BaseEnvWindow`. Then, you can set
        这种属性是你的类型。
    """

    # general settings
    seed: int | None = None
    """The seed for the random number generator. Defaults to None, in which case the seed is not set.

    Note:
      The seed is set at the beginning of the environment initialization. This ensures that the environment
      creation is deterministic and behaves similarly across different runs.
    """
    """随机数生成器的种子。
    默认为None，在这种情况下，种子没有设置。

    说明：
      种子在环境初始化开始时设置。
      这确保环境的创建是决定性的，并且在不同的行程中表现得类似。
    """

    decimation: int = MISSING
    """Number of control action updates @ sim dt per policy dt.

    For instance, if the simulation dt is 0.01s and the policy dt is 0.1s, then the decimation is 10.
    This means that the control action is updated every 10 simulation steps.
    """
    """控制操作更新次数 @ sim dt 每个策略 dt。

    例如，如果仿真dt是0.01s，策略dt是0.1s，那么数十年是10。
    这意味着每10个仿真步骤都会更新控制操作。
    """

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
    scene: InteractiveSceneCfg = MISSING
    """Scene settings.

    Please refer to the :class:`isaaclab.scene.InteractiveSceneCfg` class for more details.
    """
    """场景设置。

    详细请参阅:class:`isaaclab.scene.InteractiveSceneCfg`类。
    """

    events: object = None
    """Event settings. Defaults to None, in which case no events are applied through the event manager.

    Please refer to the :class:`isaaclab.managers.EventManager` class for more details.
    """
    """事件设置。
    在 None 中，默认情况下，通过事件管理器没有应用任何事件。

    详细请参阅:class:`isaaclab.managers.EventManager`类。
    """

    observation_spaces: dict[AgentID, SpaceType] = MISSING
    """Observation space definition for each agent.

    The space can be defined either using Gymnasium :py:mod:`~gymnasium.spaces` (when a more detailed
    specification of the space is desired) or basic Python data types (for simplicity).

    .. list-table::
        :header-rows: 1

        * - Gymnasium space
          - Python data type
        * - :class:`~gymnasium.spaces.Box`
          - Integer or list of integers (e.g.: ``7``, ``[64, 64, 3]``)
        * - :class:`~gymnasium.spaces.Discrete`
          - Single-element set (e.g.: ``{2}``)
        * - :class:`~gymnasium.spaces.MultiDiscrete`
          - List of single-element sets (e.g.: ``[{2}, {5}]``)
        * - :class:`~gymnasium.spaces.Dict`
          - Dictionary (e.g.: ``{"joints": 7, "rgb": [64, 64, 3], "gripper": {2}}``)
        * - :class:`~gymnasium.spaces.Tuple`
          - Tuple (e.g.: ``(7, [64, 64, 3], {2})``)
    """
    """每个代理的观测空间定义。

    空间可以使用 Gymnasium :py:mod:`~gymnasium.spaces` (如果需要更详细的空间规格) 或基本的Python数据类型来定义。

    ..
    列表表:
        :header-rows: 1

        * - Gym空间
          - Python 数据类型
        * - :class:`~gymnasium.spaces.Box`
          - 整数或整数列表 (e.g.:``7``，``[64， 64， 3]``)
        * - :class:`~gymnasium.spaces.Discrete`
          - 单元组 (e.g.:``{2}``)
        * - :class:`~gymnasium.spaces.MultiDiscrete`
          - 单元组的列表 (e.g.:``[{2}， {5}]``)
        * - :class:`~gymnasium.spaces.Dict`
          - 字典 (e.g.:``{"joints": 7， "rgb": [64， 64， 3]， "gripper": {2}}``)
        * - :class:`~gymnasium.spaces.Tuple`
          - (e.g.:``(7， [64， 64， 3]， {2})``)
    """

    num_observations: dict[AgentID, int] | None = None
    """The dimension of the observation space for each agent.

    .. warning::

        This attribute is deprecated. Use :attr:`~isaaclab.envs.DirectMARLEnvCfg.observation_spaces` instead.
    """
    """每个代理的观测空间的尺寸。

    .. 警告::

        这种属性已被废弃。
        用:attr:`~isaaclab.envs.DirectMARLEnvCfg.observation_spaces`代替。
    """

    state_space: SpaceType = MISSING
    """State space definition.

    The following values are supported:

    * -1: All the observations from the different agents are automatically concatenated.
    * 0: No state-space will be constructed (`state_space` is None).
      This is useful to save computational resources when the algorithm to be trained does not need it.
    * greater than 0: Custom state-space dimension to be provided by the task implementation.

    The space can be defined either using Gymnasium :py:mod:`~gymnasium.spaces` (when a more detailed
    specification of the space is desired) or basic Python data types (for simplicity).

    .. list-table::
        :header-rows: 1

        * - Gymnasium space
          - Python data type
        * - :class:`~gymnasium.spaces.Box`
          - Integer or list of integers (e.g.: ``7``, ``[64, 64, 3]``)
        * - :class:`~gymnasium.spaces.Discrete`
          - Single-element set (e.g.: ``{2}``)
        * - :class:`~gymnasium.spaces.MultiDiscrete`
          - List of single-element sets (e.g.: ``[{2}, {5}]``)
        * - :class:`~gymnasium.spaces.Dict`
          - Dictionary (e.g.: ``{"joints": 7, "rgb": [64, 64, 3], "gripper": {2}}``)
        * - :class:`~gymnasium.spaces.Tuple`
          - Tuple (e.g.: ``(7, [64, 64, 3], {2})``)
    """
    """国家空间定义。

    支持以下值:

    * -1:各个代理人的所有观测都是自动连接的。
    * 0:不会构建任何状态空间 (`state_space`是None).当训练的算法不需要计算资源时，这是有用的。
    * 超过0:任务执行所需的定制状态空间尺寸。

    空间可以使用 Gymnasium :py:mod:`~gymnasium.spaces` (如果需要更详细的空间规格) 或基本的Python数据类型来定义。

    ..
    列表表:
        :header-rows: 1

        * - Gym空间
          - Python 数据类型
        * - :class:`~gymnasium.spaces.Box`
          - 整数或整数列表 (e.g.:``7``，``[64， 64， 3]``)
        * - :class:`~gymnasium.spaces.Discrete`
          - 单元组 (e.g.:``{2}``)
        * - :class:`~gymnasium.spaces.MultiDiscrete`
          - 单元组的列表 (e.g.:``[{2}， {5}]``)
        * - :class:`~gymnasium.spaces.Dict`
          - 字典 (e.g.:``{"joints": 7， "rgb": [64， 64， 3]， "gripper": {2}}``)
        * - :class:`~gymnasium.spaces.Tuple`
          - (e.g.:``(7， [64， 64， 3]， {2})``)
    """

    num_states: int | None = None
    """The dimension of the state space from each environment instance.

    .. warning::

        This attribute is deprecated. Use :attr:`~isaaclab.envs.DirectMARLEnvCfg.state_space` instead.
    """
    """每个环境实例中的状态空间的尺寸。

    .. 警告::

        这种属性已被废弃。
        用:attr:`~isaaclab.envs.DirectMARLEnvCfg.state_space`代替。
    """

    observation_noise_model: dict[AgentID, NoiseModelCfg | None] | None = None
    """The noise model to apply to the computed observations from the environment. Default is None,
    which means no noise is added.

    Please refer to the :class:`isaaclab.utils.noise.NoiseModel` class for more details.
    """
    """对于环境计算的观测而应用的噪音模型。
    默认是None，这意味着没有添加噪音。

    详细请参阅:class:`isaaclab.utils.noise.NoiseModel`类。
    """

    action_spaces: dict[AgentID, SpaceType] = MISSING
    """Action space definition for each agent.

    The space can be defined either using Gymnasium :py:mod:`~gymnasium.spaces` (when a more detailed
    specification of the space is desired) or basic Python data types (for simplicity).

    .. list-table::
        :header-rows: 1

        * - Gymnasium space
          - Python data type
        * - :class:`~gymnasium.spaces.Box`
          - Integer or list of integers (e.g.: ``7``, ``[64, 64, 3]``)
        * - :class:`~gymnasium.spaces.Discrete`
          - Single-element set (e.g.: ``{2}``)
        * - :class:`~gymnasium.spaces.MultiDiscrete`
          - List of single-element sets (e.g.: ``[{2}, {5}]``)
        * - :class:`~gymnasium.spaces.Dict`
          - Dictionary (e.g.: ``{"joints": 7, "rgb": [64, 64, 3], "gripper": {2}}``)
        * - :class:`~gymnasium.spaces.Tuple`
          - Tuple (e.g.: ``(7, [64, 64, 3], {2})``)
    """
    """每个代理的动作空间定义。

    空间可以使用 Gymnasium :py:mod:`~gymnasium.spaces` (如果需要更详细的空间规格) 或基本的Python数据类型来定义。

    ..
    列表表:
        :header-rows: 1

        * - Gym空间
          - Python 数据类型
        * - :class:`~gymnasium.spaces.Box`
          - 整数或整数列表 (e.g.:``7``，``[64， 64， 3]``)
        * - :class:`~gymnasium.spaces.Discrete`
          - 单元组 (e.g.:``{2}``)
        * - :class:`~gymnasium.spaces.MultiDiscrete`
          - 单元组的列表 (e.g.:``[{2}， {5}]``)
        * - :class:`~gymnasium.spaces.Dict`
          - 字典 (e.g.:``{"joints": 7， "rgb": [64， 64， 3]， "gripper": {2}}``)
        * - :class:`~gymnasium.spaces.Tuple`
          - (e.g.:``(7， [64， 64， 3]， {2})``)
    """

    num_actions: dict[AgentID, int] | None = None
    """The dimension of the action space for each agent.

    .. warning::

        This attribute is deprecated. Use :attr:`~isaaclab.envs.DirectMARLEnvCfg.action_spaces` instead.
    """
    """每个代理的动作空间的尺寸。

    .. 警告::

        这种属性已被废弃。
        用:attr:`~isaaclab.envs.DirectMARLEnvCfg.action_spaces`代替。
    """

    action_noise_model: dict[AgentID, NoiseModelCfg | None] | None = None
    """The noise model applied to the actions provided to the environment. Default is None,
    which means no noise is added.

    Please refer to the :class:`isaaclab.utils.noise.NoiseModel` class for more details.
    """
    """对环境提供动作的噪音模型。
    默认是None，这意味着没有添加噪音。

    详细请参阅:class:`isaaclab.utils.noise.NoiseModel`类。
    """

    possible_agents: list[AgentID] = MISSING
    """A list of all possible agents the environment could generate.

    The contents of the list cannot be modified during the entire training process.
    """
    """环境可能产生的所有可能的因素。

    在整个培训过程中，列表的内容不能被修改。
    """

    xr: XrCfg | None = None
    """Configuration for viewing and interacting with the environment through an XR device."""
    """通过XR设备查看和与环境交互的配置。"""

    log_dir: str | None = None
    """Directory for logging experiment artifacts. Defaults to None, in which case no specific log directory is set."""
    """记录实验文物目录。
    默认对None的设置，在这种情况下没有设置特定的日志目录。
    """
