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

from .common import SpaceType, ViewerCfg
from .ui import BaseEnvWindow


@configclass
class DirectRLEnvCfg:
    """Configuration for an RL environment defined with the direct workflow.

    Please refer to the :class:`isaaclab.envs.direct_rl_env.DirectRLEnv` class for more details.
    """
    """对于直接工作流定义的RL环境的配置。

    详细请参阅:class:`isaaclab.envs.direct_rl_env.DirectRLEnv`类。
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

    events: object | None = None
    """Event settings. Defaults to None, in which case no events are applied through the event manager.

    Please refer to the :class:`isaaclab.managers.EventManager` class for more details.
    """
    """事件设置。
    在 None 中，默认情况下，通过事件管理器没有应用任何事件。

    详细请参阅:class:`isaaclab.managers.EventManager`类。
    """

    observation_space: SpaceType = MISSING
    """Observation space definition.

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
    """观测空间的定义。

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

    num_observations: int | None = None
    """The dimension of the observation space from each environment instance.

    .. warning::

        This attribute is deprecated. Use :attr:`~isaaclab.envs.DirectRLEnvCfg.observation_space` instead.
    """
    """从每个环境实例中观测空间的尺寸。

    .. 警告::

        这种属性已被废弃。
        用:attr:`~isaaclab.envs.DirectRLEnvCfg.observation_space`代替。
    """

    state_space: SpaceType | None = None
    """State space definition.

    This is useful for asymmetric actor-critic and defines the observation space for the critic.

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

    这对于不对称的演员-批评者来说是有用的，并为批评者定义了观测空间。

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
    """The dimension of the state-space from each environment instance.

    .. warning::

        This attribute is deprecated. Use :attr:`~isaaclab.envs.DirectRLEnvCfg.state_space` instead.
    """
    """每个环境实例中的状态空间的尺寸。

    .. 警告::

        这种属性已被废弃。
        用:attr:`~isaaclab.envs.DirectRLEnvCfg.state_space`代替。
    """

    observation_noise_model: NoiseModelCfg | None = None
    """The noise model to apply to the computed observations from the environment. Default is None,
    which means no noise is added.

    Please refer to the :class:`isaaclab.utils.noise.NoiseModel` class for more details.
    """
    """对于环境计算的观测而应用的噪音模型。
    默认是None，这意味着没有添加噪音。

    详细请参阅:class:`isaaclab.utils.noise.NoiseModel`类。
    """

    action_space: SpaceType = MISSING
    """Action space definition.

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
    """动作空间的定义。

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

    num_actions: int | None = None
    """The dimension of the action space for each environment.

    .. warning::

        This attribute is deprecated. Use :attr:`~isaaclab.envs.DirectRLEnvCfg.action_space` instead.
    """
    """每个环境的动作空间的尺寸。

    .. 警告::

        这种属性已被废弃。
        用:attr:`~isaaclab.envs.DirectRLEnvCfg.action_space`代替。
    """

    action_noise_model: NoiseModelCfg | None = None
    """The noise model applied to the actions provided to the environment. Default is None,
    which means no noise is added.

    Please refer to the :class:`isaaclab.utils.noise.NoiseModel` class for more details.
    """
    """对环境提供动作的噪音模型。
    默认是None，这意味着没有添加噪音。

    详细请参阅:class:`isaaclab.utils.noise.NoiseModel`类。
    """

    rerender_on_reset: bool = False
    """Whether a render step is performed again after at least one environment has been reset.
    Defaults to False, which means no render step will be performed after reset.

    * When this is False, data collected from sensors after performing reset will be stale and will not reflect the
      latest states in simulation caused by the reset.
    * When this is True, an extra render step will be performed to update the sensor data
      to reflect the latest states from the reset. This comes at a cost of performance as an additional render
      step will be performed after each time an environment is reset.

    .. deprecated:: 2.3.1
        This attribute is deprecated and will be removed in the future. Please use
        :attr:`num_rerenders_on_reset` instead.

        To get the same behaviour as setting this parameter to ``True`` or ``False``, set
        :attr:`num_rerenders_on_reset` to 1 or 0, respectively.
    """
    """在至少设置一个环境后是否再次执行渲染步骤。
    默认对False进行错误，这意味着在重置后不会执行任何 step渲染步骤。

    * 如果是False，重置后从传感器收集的数据将是陈旧的，不会反映重置所导致的仿真中最新状态。
    * 当 True 时，将执行额外的渲染步骤，以更新传感器数据，以反映从重置的最新状态.这 comes at a cost of performance as an additional render
      step will be performed after each time an environment is reset。

    ..
    2.3.1 这个属性已被废除，将来会被删除。
    请使用:attr:`num_rerenders_on_reset`。

        设置这个参数为``True``或``False``设置:attr:`num_rerenders_on_reset`分别为1或0。
    """

    num_rerenders_on_reset: int = 0
    """Number of render steps to perform after reset. Defaults to 0, which means no render step will be performed
    after reset.

    * When this is 0, no render step will be performed after reset. Data collected from sensors after performing
      reset will be stale and will not reflect the latest states in simulation caused by the reset.
    * When this is greater than 0, the specified number of extra render steps will be performed to update the
      sensor data to reflect the latest states from the reset. This comes at a cost of performance as additional
      render steps will be performed after each time an environment is reset.
    """
    """在重置后执行的渲染步骤数。
    默认到0，这意味着在重置后不会执行任何渲染步骤。

    * 在 0 时，重置后不会执行任何渲染步骤.重置后从传感器收集的数据将是陈旧的，不会反映重置所导致的仿真中的最新状态。
    * 当这个数量超过0时，将执行指定的额外渲染步骤，以更新传感器数据，以反映从重置的最新状态.这 comes at a cost of performance as additional render
      steps will be performed after each time an environment is reset。
    """

    wait_for_textures: bool = True
    """True to wait for assets to be loaded completely, False otherwise. Defaults to True."""
    """True等待资产完全充满，False否则。
    默认为 True。
    """

    xr: XrCfg | None = None
    """Configuration for viewing and interacting with the environment through an XR device."""
    """通过XR设备查看和与环境交互的配置。"""

    log_dir: str | None = None
    """Directory for logging experiment artifacts. Defaults to None, in which case no specific log directory is set."""
    """记录实验文物目录。
    默认对None的设置，在这种情况下没有设置特定的日志目录。
    """
