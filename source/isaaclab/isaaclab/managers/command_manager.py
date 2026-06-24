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

        # create buffers to store the command
        # -- metrics that can be used for logging
        self.metrics = dict()
        # -- time left before resampling
        self.time_left = torch.zeros(self.num_envs, device=self.device)
        # -- counter for the number of times the command has been resampled within the current episode
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
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
        # update the command
        self._update_command()

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

    @abstractmethod
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the specified environments."""
        """对于指定环境来说，重新样本命令。"""
        raise NotImplementedError

    @abstractmethod
    def _update_command(self):
        """Update the command based on the current state."""
        """根据当前状态更新命令。"""
        raise NotImplementedError

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
        self._commands = dict()
        if self.cfg:
            self.cfg.debug_vis = False
            for term in self._terms.values():
                self.cfg.debug_vis |= term.cfg.debug_vis

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
        # return logged information
        return extras

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
