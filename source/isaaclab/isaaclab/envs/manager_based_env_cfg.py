# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""
"""环境的基本配置。

本模块定义了环境的一般配置。
它包括配置环境实例，观众设置和仿真参数的参数。
"""

from dataclasses import MISSING, field

import isaaclab.envs.mdp as mdp
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RecorderManagerBaseCfg as DefaultEmptyRecorderManagerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .common import ViewerCfg
from .ui import BaseEnvWindow


@configclass
class DefaultEventManagerCfg:
    """Configuration of the default event manager.

    This manager is used to reset the scene to a default state. The default state is specified
    by the scene configuration.
    """
    """设置默认事件管理器

    这个管理器用于重置场景到默认状态。
    默认状态由场景配置指定。
    """

    reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class ManagerBasedEnvCfg:
    """Base configuration of the environment."""
    """环境的基本配置。"""

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

    # environment settings
    scene: InteractiveSceneCfg = MISSING
    """Scene settings.

    Please refer to the :class:`isaaclab.scene.InteractiveSceneCfg` class for more details.
    """
    """场景设置。

    详细请参阅:class:`isaaclab.scene.InteractiveSceneCfg`类。
    """

    recorders: object = DefaultEmptyRecorderManagerCfg()
    """Recorder settings. Defaults to recording nothing.

    Please refer to the :class:`isaaclab.managers.RecorderManager` class for more details.
    """
    """记录设置。
    默认的记录没有什么。

    详细请参阅:class:`isaaclab.managers.RecorderManager`类。
    """

    observations: object = MISSING
    """Observation space settings.

    Please refer to the :class:`isaaclab.managers.ObservationManager` class for more details.
    """
    """观测空间设置。

    详细请参阅:class:`isaaclab.managers.ObservationManager`类。
    """

    actions: object = MISSING
    """Action space settings.

    Please refer to the :class:`isaaclab.managers.ActionManager` class for more details.
    """
    """动作空间设置。

    详细请参阅:class:`isaaclab.managers.ActionManager`类。
    """

    events: object = DefaultEventManagerCfg()
    """Event settings. Defaults to the basic configuration that resets the scene to its default state.

    Please refer to the :class:`isaaclab.managers.EventManager` class for more details.
    """
    """事件设置。
    基本配置的默认设置将场景重置到默认状态。

    详细请参阅:class:`isaaclab.managers.EventManager`类。
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
    """Number of render steps to perform after reset. Defaults to 0, which means no render step will be
    performed after reset.

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

    teleop_devices: DevicesCfg = field(default_factory=DevicesCfg)
    """Configuration for teleoperation devices."""
    """远程操作设备的配置。"""

    export_io_descriptors: bool = False
    """Whether to export the IO descriptors for the environment. Defaults to False."""
    """如何出口IO环境描述符。
    默认为 False。
    """

    log_dir: str | None = None
    """Directory for logging experiment artifacts. Defaults to None, in which case no specific log directory is set."""
    """记录实验文物目录。
    默认对None的设置，在这种情况下没有设置特定的日志目录。
    """
