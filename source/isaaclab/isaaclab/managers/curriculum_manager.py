# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum manager for updating environment quantities subject to a training curriculum."""

from __future__ import annotations
"""课程管理器更新环境数量"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import CurriculumTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CurriculumManager(ManagerBase):
    """Manager to implement and execute specific curricula.

    The curriculum manager updates various quantities of the environment subject to a training curriculum by
    calling a list of terms. These help stabilize learning by progressively making the learning tasks harder
    as the agent improves.

    The curriculum terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each curriculum term should instantiate the :class:`CurriculumTermCfg` class.
    """
    """管理器实施和执行具体课程。

    课程管理器通过调用项清单来更新一个培训课程所需的各种环境。
    这些帮助稳定学习， 通过逐步使学习任务变得更加困难，

    课程项由包含管理器设置和每个项参数的配置类进行分析。
    每个课程项都应该包含:class:`CurriculumTermCfg`课程。
    """

    _env: ManagerBasedRLEnv
    """The environment instance."""
    """环境情况。"""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, CurriculumTermCfg]``)
            env: An environment object.

        Raises:
            TypeError: If curriculum term is not of type :class:`CurriculumTermCfg`.
            ValueError: If curriculum term configuration does not satisfy its function signature.
        """
        """启动管理器。

        参数：
            cfg: 配置对象或字典 (``dict[str， CurriculumTermCfg]``)
            env: 一个环境对象。

        异常：
            TypeError: 如果课程项不是:class:`CurriculumTermCfg`类型。
            ValueError: 如果课程项配置不符合其功能签名。
        """
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._term_cfgs: list[CurriculumTermCfg] = list()
        self._class_term_cfgs: list[CurriculumTermCfg] = list()

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)

        # prepare logging
        self._curriculum_state = dict()
        for term_name in self._term_names:
            self._curriculum_state[term_name] = None

    def __str__(self) -> str:
        """Returns: A string representation for curriculum manager."""
        """Returns: 课程管理器。"""
        msg = f"<CurriculumManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Curriculum Terms"
        table.field_names = ["Index", "Name"]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for index, name in enumerate(self._term_names):
            table.add_row([index, name])
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
        """Name of active curriculum terms."""
        """事件课程项的名称。"""
        return self._term_names

    """
    Operations.
    """
    """操作。
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Returns the current state of individual curriculum terms.

        Note:
            This function does not use the environment indices :attr:`env_ids`
            and logs the state of all the terms. The argument is only present
            to maintain consistency with other classes.

        Returns:
            Dictionary of curriculum terms and their states.
        """
        """返回个人课程项的现状。

        说明：
            这个函数不使用环境索引:attr:`env_ids`，并记录所有项的状态。
            这种论点只存在于保持与其他类别的一致性。

        返回：
            课程项字典及其状态。
        """
        extras = {}
        for term_name, term_state in self._curriculum_state.items():
            if term_state is not None:
                # deal with dict
                if isinstance(term_state, dict):
                    # each key is a separate state to log
                    for key, value in term_state.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        extras[f"Curriculum/{term_name}/{key}"] = value
                else:
                    # log directly if not a dict
                    if isinstance(term_state, torch.Tensor):
                        term_state = term_state.item()
                    extras[f"Curriculum/{term_name}"] = term_state
        # reset all the curriculum terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self, env_ids: Sequence[int] | None = None):
        """Update the curriculum terms.

        This function calls each curriculum term managed by the class.

        Args:
            env_ids: The list of environment IDs to update.
                If None, all the environments are updated. Defaults to None.
        """
        """更新课程项。

        这种函数将每个课程项称为由课堂管理的。

        参数：
            env_ids: 环境 IDs更新的列表。
                     如果None，所有环境都会更新。
                     默认为 None。
        """
        # resolve environment indices
        if env_ids is None:
            env_ids = slice(None)
        # iterate over all the curriculum terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            state = term_cfg.func(self._env, env_ids, **term_cfg.params)
            self._curriculum_state[name] = state

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

        for term_name, term_state in self._curriculum_state.items():
            if term_state is not None:
                # deal with dict
                data = []

                if isinstance(term_state, dict):
                    # each key is a separate state to log
                    for key, value in term_state.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        terms[term_name].append(value)
                else:
                    # log directly if not a dict
                    if isinstance(term_state, torch.Tensor):
                        term_state = term_state.item()
                    data.append(term_state)
                terms.append((term_name, data))

        return terms

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
            # check if the term is a valid term config
            if not isinstance(term_cfg, CurriculumTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type CurriculumTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
            # add name and config to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            # check if the term is a class
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)
