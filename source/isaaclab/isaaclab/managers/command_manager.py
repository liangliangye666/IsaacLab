# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command manager for generating and updating commands."""

from __future__ import annotations
"""命令生成和更新的命令管理器。"""

import inspect
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable

import omni.kit.app

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import CommandTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CommandTerm(ManagerTermBase):
    """The base class for implementing a command term.

    A command term is used to generate commands for goal-conditioned tasks. For example,
    in the case of a goal-conditioned navigation task, the command term can be used to
    generate a target position for the robot to navigate to.

    It implements a resampling mechanism that allows the command to be resampled at a fixed
    frequency. The resampling frequency can be specified in the configuration object.
    Additionally, it is possible to assign a visualization function to the command term
    that can be used to visualize the command in the simulator.
    """
    """执行命令项的基础类。

    一个命令项用于生成目标条件任务的命令。
    例如，在目标条件导航任务的情况下，命令项可以用于生成机器人导航的目标位置。

    它实施了重新样本机制，允许命令在固定频率上重新样本。
    在配置对象中可指定重样频率。
    此外，可以将可视化函数分配给命令项，可用于可视化仿真器中的命令。
    """

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        """启动命令生成器类。

        参数：
            cfg: 命令生成器的配置参数。
            env: 环境对象。
        """
        super().__init__(cfg, env)

        # create buffers to store the command   一个空字典，用于存储可记录的命令指标（如当前命令的距离、角度等），供日志使用。
        # -- metrics that can be used for logging
        self.metrics = dict()
        # -- time left before resampling    命令重采样倒计时器。每个环境有一个独立的倒计时。当倒计时归零时，重新随机生成一个命令。
        self.time_left = torch.zeros(self.num_envs, device=self.device)
        # -- counter for the number of times the command has been resampled within the current episode
        # 记录当前回合内这个环境被重采样了多少次。用于日志统计和课程学习（比如"第 10 次采样后提高难度"）。
        self.command_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        """取消回电话。"""
        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    """
    Properties
    """
    """产品
    """

    @property
    @abstractmethod
    def command(self) -> torch.Tensor:
        """The command tensor. Shape is (num_envs, command_dim)."""
        """命令子。
        形状是 (num_envs，command_dim)。
        """
        raise NotImplementedError

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command generator has a debug visualization implemented."""
        """命令生成器是否实现了调试可视化。"""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    """
    Operations.
    """
    """操作。
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the command data.

        Args:
            debug_vis: Whether to visualize the command data.

        Returns:
            Whether the debug visualization was successfully set. False if the command
            generator does not support debug visualization.
        """
        """设置是否可可视化命令数据。

        参数：
            debug_vis: 是否可视化命令数据。

        返回：
            设置错误可视化是否成功。
            False如果命令生成器不支持调试可视化。
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False
        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True

    '''
    回合结束时清空统计 + 生成新命令
    '''
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command generator and log metrics.

        This function resets the command counter and resamples the command. It should be called
        at the beginning of each episode.

        Args:
            env_ids: The list of environment IDs to reset. Defaults to None.

        Returns:
            A dictionary containing the information to log under the "{name}" key.
        """
        """设置命令生成器和记录数据。

        这个函数重置命令计数器并重新仿真命令。
        应在每个集的开始召开。

        参数：
            env_ids: 环境 IDs的重置列表。
                     默认为 None。

        返回：
            包含在"{name}"键下记录信息的字典。
        """
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)

        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            # reset the metric value
            metric_value[env_ids] = 0.0

        # set the command counter to zero
        self.command_counter[env_ids] = 0
        # resample the command
        self._resample(env_ids)

        return extras
    '''
    reset(env_ids)
    │
    ├── 步骤 1: 输出日志指标
    │     遍历 self.metrics 中所有记录的指标
    │     → 计算均值 → 放入 extras → 归零
    │
    ├── 步骤 2: 重置命令计数器
    │     command_counter[env_ids] = 0
    │
    └── 步骤 3: 重新采样命令
          _resample(env_ids)
          随机生成新的目标（速度、位置等）

    self.metrics 是什么？在 compute() 中动态填充的字典，例如：
        # 一个速度命令 term 的 metrics:
        self.metrics = {
            "vel_command_x":   tensor([1.2, 0.8, 1.5, ...]),   # N 个环境的当前 x 方向命令速度
            "vel_command_y":   tensor([0.3, -0.1, 0.0, ...]),  # y 方向命令速度
            "command_distance": tensor([3.4, 2.1, 5.7, ...]),   # 当前命令的距离
        }

    reset 时：计算这些指标在这些被重置环境上的均值，输出到日志（extras），然后归零，为新回合准备。
        extras["vel_command_x"]   = 1.17   # 例如 (1.2+0.8+1.5)/3
        extras["vel_command_y"]   = 0.07
        extras["command_distance"] = 3.73
    '''


    def compute(self, dt: float):
        """Compute the command.

        Args:
            dt: The time step passed since the last call to compute.
        """
        """计算命令。

        参数：
            dt: 在最后一次电话计算之后，时间已经过去了。
        """
        # update the metrics based on current state
        self._update_metrics()
        # reduce the time left before resampling
        self.time_left -= dt
        # resample the command if necessary
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        '''
        拆解这行张量运算：
            self.time_left <= 0.0
            # → tensor([False, False, False, True])   # 环境 3 到期了

            (self.time_left <= 0.0).nonzero()
            # → tensor([[3]])                          # 非零元素的坐标

            (self.time_left <= 0.0).nonzero().flatten()
            # → tensor([3])                            # 展平为一维
            # resample_env_ids = tensor([3])
        '''

        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
        # update the command
        self._update_command()
    '''
    compute(dt)
    │
    ├── ① _update_metrics()          更新统计指标
    ├── ② time_left -= dt            倒计时递减
    ├── ③ 到期环境重采样              _resample(env_ids)
    └── ④ _update_command()          更新命令值
    '''

    """
    Helper functions.
    """
    """辅助函数。
    """

    def _resample(self, env_ids: Sequence[int]):
        """Resample the command.

        This function resamples the command and time for which the command is applied for the
        specified environment indices.

        Args:
            env_ids: The list of environment IDs to resample.
        """
        """再试命令。

        这项函数为指定的环境索引重新示范了命令和时间。

        参数：
            env_ids: 环境 IDs重新样本的列表。
        """
        if len(env_ids) != 0:
            # resample the time left before resampling
            self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)
            '''
            拆解这行 Python 语法：
                self.time_left[env_ids]	            取出要重采样的环境
                .uniform_(min, max)	                PyTorch 的原地均匀随机采样，在 [min, max) 范围内随机生成值
                *self.cfg.resampling_time_range	    元组解包，把 (5.0, 10.0) 展开为 uniform_(5.0, 10.0)
            末尾的下划线 _ 表示"原地操作"（in-place）——直接修改张量本身，不创建副本。这是 PyTorch 的命名惯例。

            * 元组解包
                resampling_time_range = (5.0, 10.0)
                # 等价写法：
                .uniform_(5.0, 10.0)                    # 手动展开
                .uniform_(*resampling_time_range)        # * 自动解包元组
            '''
            # resample the command
            self._resample_command(env_ids)
            # increment the command counter
            self.command_counter[env_ids] += 1

    """
    Implementation specific functions.
    """
    """具体执行功能。
    """

    @abstractmethod
    def _update_metrics(self):
        """Update the metrics based on the current state."""
        """根据当前状态更新数据。"""
        raise NotImplementedError
    '''
    记录当前命令状态到 self.metrics
    '''

    @abstractmethod
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the specified environments."""
        """对于指定环境来说，重新样本命令。"""
        raise NotImplementedError
    '''
    为指定环境随机生成新命令
    '''

    @abstractmethod
    def _update_command(self):
        """Update the command based on the current state."""
        """根据当前状态更新命令。"""
        raise NotImplementedError
    '''
    每帧更新命令值（如正弦波）
        大部分命令是静态的（重采样后值不变），但有些命令需要随时间变化。
        例如"正弦速度命令"：vx = sin(t)，每帧都在变。
        这个方法允许子类在不需要重采样时也能更新命令值。
    '''

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        """设置调试可视化到可视化对象。

        如果它们不存在，并且输入 ``debug_vis`` 是 True，
        如果可视化对象存在，函数应该将它们的可视性设置在舞台上。
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        This function calls the visualization objects and sets the data to visualize into them.
        """
        """检查错误可视化。

        这个函数将可视化对象调用，并设置数据可视化到它们中。
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

'''
CommandManager 只维护一份数据：
    self._terms = {"base_velocity": ..., "base_pose": ...}  # 只有字典
    # 没有 _term_names！
'''
class CommandManager(ManagerBase):
    """Manager for generating commands.

    The command manager is used to generate commands for an agent to execute. It makes it convenient to switch
    between different command generation strategies within the same environment. For instance, in an environment
    consisting of a quadrupedal robot, the command to it could be a velocity command or position command.
    By keeping the command generation logic separate from the environment, it is easy to switch between different
    command generation strategies.

    The command terms are implemented as classes that inherit from the :class:`CommandTerm` class.
    Each command generator term should also have a corresponding configuration class that inherits from the
    :class:`CommandTermCfg` class.
    """
    """管理器生成命令。

    命令管理器用于生成命令，使代理执行。
    在同一个环境中，它使得更方便地切换不同的命令生成策略。
    例如，在一个由四脚机器人组成的环境中，它的命令可能是速度命令或位置命令。
    通过将命令生成逻辑与环境分开，很容易在不同的命令生成策略之间切换。

    命令项是从:class:`CommandTerm`类继承的类。
    每个命令生成器项还应具有来自:class:`CommandTermCfg`类的相应配置类。
    """

    _env: ManagerBasedRLEnv
    """The environment instance."""
    """环境情况。"""
    '''
    类级别的纯类型注解
        ——它不创建变量、不赋值、不改变运行时行为，只是告诉 IDE "这个类的实例将来会有 _env 属性，类型是 ManagerBasedRLEnv"。
        实际赋值发生在 ManagerBase.__init__ 中的 self._env = env。
    '''

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the command manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, CommandTermCfg]``).
            env: The environment instance.
        """
        """启动命令管理器。

        参数：
            cfg: 配置对象或字典 (``dict[str， CommandTermCfg]``)。
            env: 环境情况。
        """
        # create buffers to parse and store terms
        self._terms: dict[str, CommandTerm] = dict()

        # call the base class constructor (this prepares the terms)
        super().__init__(cfg, env)
        # store the commands
        self._commands = dict() # 存储各 CommandTerm 生成的命令值，供观测函数读取
        if self.cfg:
            self.cfg.debug_vis = False
            for term in self._terms.values():
                self.cfg.debug_vis |= term.cfg.debug_vis

    '''
    命令管理器的信息展示
        <CommandManager> contains 2 active terms.
        +-------------------------------+
        | Active Command Terms          |
        +-------+--------------------+---------------------------+
        | Index | Name               | Type                      |
        +-------+--------------------+---------------------------+
        |   0   | base_velocity      | UniformVelocityCommand    |
        |   1   | base_pose          | UniformPoseCommand        |
        +-------+--------------------+---------------------------+
    '''
    def __str__(self) -> str:
        """Returns: A string representation for the command manager."""
        """Returns: 命令管理器的字符串表示。"""
        msg = f"<CommandManager> contains {len(self._terms.values())} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Command Terms"
        table.field_names = ["Index", "Name", "Type"]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.__class__.__name__])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """
    """属性。
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active command terms."""
        """事件命令项的名称。"""
        return list(self._terms.keys())

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        """命令项是否实现了调试可视化。"""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    """
    Operations.
    """
    """操作。
    """

    '''
    命令数据的 GUI 提取器
    返回示例：
        [
            ("base_velocity", [1.2, 0.3, 0.0]),    # vx, vy, vyaw
            ("base_pose",    [3.5, -1.2, 0.0, 0.0]),  # x, y, z, yaw
        ]
    '''
    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        """返回活跃的项作为可反复的双数序列。

        元组的第一个元素是项的名称，第二个元素是项的原始值。

        参数：
            env_idx: 具体的环境，可以从中提取活跃项。

        返回：
            积极的项。
        """

        terms = []
        idx = 0
        for name, term in self._terms.items():
            terms.append((name, term.command[env_idx].cpu().tolist()))
            idx += term.command.shape[1]
        return terms

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the command data.

        Args:
            debug_vis: Whether to visualize the command data.

        Returns:
            Whether the debug visualization was successfully set. False if the command
            generator does not support debug visualization.
        """
        """设置是否可可视化命令数据。

        参数：
            debug_vis: 是否可视化命令数据。

        返回：
            设置错误可视化是否成功。
            False如果命令生成器不支持调试可视化。
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    '''
    命令管理器的重置汇总
    '''
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Reset the command terms and log their metrics.

        This function resets the command counter and resamples the command for each term. It should be called
        at the beginning of each episode.

        Args:
            env_ids: The list of environment IDs to reset. Defaults to None.

        Returns:
            A dictionary containing the information to log under the "Metrics/{term_name}/{metric_name}" key.
        """
        """重置命令项并记录它们的指标。

        这个函数重置命令计数器，并为每个项重建命令。
        应在每个集的开始召开。

        参数：
            env_ids: 环境 IDs的重置列表。
                     默认为 None。

        返回：
            一个包含"Metrics/{term_name}/{metric_name}"键下记录信息的字典。
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for name, term in self._terms.items():
            # reset the command term
            metrics = term.reset(env_ids=env_ids)
            # compute the mean metric value
            for metric_name, metric_value in metrics.items():
                extras[f"Metrics/{name}/{metric_name}"] = metric_value
                '''
                加前缀汇总
                指标命名的层级结构
                f"Metrics/{name}/{metric_name}" 把每个 term 的指标加上前缀：
                    # term "base_velocity" 的 reset 返回：
                    {"vel_command_x": 1.17, "vel_command_y": 0.07}

                    # CommandManager.reset 汇总后变成：
                    {
                        "Metrics/base_velocity/vel_command_x": 1.17,
                        "Metrics/base_velocity/vel_command_y": 0.07,
                    }
                '''
        # return logged information
        return extras
    '''
    和 CommandTerm.reset() 的关系
        CommandManager.reset(env_ids)          ← Manager 级别（这段代码）
            │
            └── for each term:
                term.reset(env_ids)          ← CommandTerm 级别（之前讲过的）
                    → 输出自己的 metrics
                    → 重置计数器
                    → 重新采样命令
    Manager 只管"汇总"，具体的重置逻辑全部委托给各 term。
    '''

    def compute(self, dt: float):
        """Updates the commands.

        This function calls each command term managed by the class.

        Args:
            dt: The time-step interval of the environment.

        """
        """更新命令。

        这个函数调用每个由类管理的命令项。

        参数：
            dt: 环境的时间间隔。
        """
        # iterate over all the command terms
        for term in self._terms.values():
            # compute term's value
            term.compute(dt)
    '''
    CommandTerm 的工作模式
        训练中:
        ManagerBasedRLEnv.step()
            └── command_manager.compute(dt)
                │
                └── command_term.compute(dt)
                        │
                        ├── time_left -= dt        ← 倒计时递减
                        │
                        ├── if time_left <= 0:
                        │     │
                        │     ├── 随机生成新命令（如速度 vx=1.2, vy=0.3）
                        │     ├── time_left = 随机重采样间隔
                        │     └── command_counter += 1
                        │
                        └── 返回当前命令（观测函数用它来计算奖励）
    '''

    def get_command(self, name: str) -> torch.Tensor:
        """Returns the command for the specified command term.

        Args:
            name: The name of the command term.

        Returns:
            The command tensor of the specified command term.
        """
        """返回指令为指定指令项。

        参数：
            name: 命令项的名称。

        返回：
            指定命令项的命令数。
        """
        return self._terms[name].command

    def get_term(self, name: str) -> CommandTerm:
        """Returns the command term with the specified name.

        Args:
            name: The name of the command term.

        Returns:
            The command term with the specified name.
        """
        """返回指定名称的命令项。

        参数：
            name: 命令项的名称。

        返回：
            指令项与指定名称。
        """
        return self._terms[name]

    """
    Helper functions.
    """
    """辅助函数。
    """

    '''
    配置到 CommandTerm 实例的翻译器
    和 ActionManager._prepare_terms() 一致。
    '''
    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, CommandTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type CommandTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, CommandTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type CommandType.")
            # add class to dict
            self._terms[term_name] = term
