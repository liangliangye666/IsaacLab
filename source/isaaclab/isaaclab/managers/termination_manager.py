# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination manager for computing done signals for a given world."""
"""终止管理器为计算给定世界的信号。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TerminationManager(ManagerBase):
    """Manager for computing done signals for a given world.

    The termination manager computes the termination signal (also called dones) as a combination
    of termination terms. Each termination term is a function which takes the environment as an
    argument and returns a boolean tensor of shape (num_envs,). The termination manager
    computes the termination signal as the union (logical or) of all the termination terms.

    Following the `Gymnasium API <https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/>`_,
    the termination signal is computed as the logical OR of the following signals:

    * **Time-out**: This signal is set to true if the environment has ended after an externally defined condition
      (that is outside the scope of a MDP). For example, the environment may be terminated if the episode has
      timed out (i.e. reached max episode length).
    * **Terminated**: This signal is set to true if the environment has reached a terminal state defined by the
      environment. This state may correspond to task success, task failure, robot falling, etc.

    These signals can be individually accessed using the :attr:`time_outs` and :attr:`terminated` properties.

    The termination terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each termination term should instantiate the :class:`TerminationTermCfg` class. The term's
    configuration :attr:`TerminationTermCfg.time_out` decides whether the term is a timeout or a termination term.
    """
    """对于一个特定的世界进行了信号的计算管理器。

    终止管理器将终止信号 (也称为 dones) 计算为终止项的组合。
    每个终结项都是一个函数，它将环境作为一个参数，并返回形状的布尔式子 (num_envs，)。
    终止管理器将终止信号计算为所有终止项的联盟 (逻辑或)。

    根据`Gymnasium API
    <https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/>`终止信号计算为OR在以下信号中:

    * **时间停止**:如果环境在外部定义的条件之后结束了，则该信号设置为 true (即 MDP 范围之外).例如，如果事件已结束 (i.e。 达到最高剧集长度)，则环境可能会结束。
    * **完成**:如果环境达到环境定义的终端状态，则该信号设置为 true。 这种状态可能与任务成功，任务失败，机器人摔倒等相符。

    这些信号可以使用:attr:`time_outs`和:attr:`terminated`特性单独访问。

    终止项由包含管理器的设置和每个项的参数的配置类进行分析。
    每个终止项都应标记:class:`TerminationTermCfg`类。
    这一项是
    configuration :attr:`TerminationTermCfg.time_out`决定该项是否是截止时间或终止项。
    """

    _env: ManagerBasedRLEnv
    """The environment instance."""
    """环境情况。"""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initializes the termination manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, TerminationTermCfg]``).
            env: An environment object.
        """
        """启动终止管理器。

        参数：
            cfg: 配置对象或字典 (``dict[str， TerminationTermCfg]``)。
            env: 一个环境对象。
        """
        # create buffers to parse and store terms
        self._term_names: list[str] = list()    # 名字列表
        self._term_cfgs: list[TerminationTermCfg] = list()  # 配置列表
        self._class_term_cfgs: list[TerminationTermCfg] = list()    # 类实现的 term（reset 时额外处理）

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)
        self._term_name_to_term_idx = {name: i for i, name in enumerate(self._term_names)}  # 创建名字→索引快速查找表
        # prepare extra info to store individual termination term information
        self._term_dones = torch.zeros((self.num_envs, len(self._term_names)), device=self.device, dtype=torch.bool)
        # prepare extra info to store last episode done per termination term information
        self._last_episode_dones = torch.zeros_like(self._term_dones)
        # create buffer for managing termination per environment
        self._truncated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._terminated_buf = torch.zeros_like(self._truncated_buf)
        '''
        _term_dones          [N, T]  bool  ← 每 term 是否触发   # 当前帧：每个环境 × 每个 term → 是否触发
        _last_episode_dones  [N, T]  bool  ← 上一步的触发记录   # 上一帧快照：用于在 reset() 时输出"上个回合是因哪个 term 触发的而结束"
        _truncated_buf       [N]     bool  ← 时间截断   ← "超时了，但没失败"
        _terminated_buf      [N]     bool  ← 真实终止   ← "真正失败了"
        '''

    '''
    含超时标志的终止信息面板
    '''
    def __str__(self) -> str:
        """Returns: A string representation for termination manager."""
        """Returns: 终止管理器的字符串表示。"""
        msg = f"<TerminationManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Termination Terms"
        table.field_names = ["Index", "Name", "Time Out"]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name, term_cfg.time_out])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg
    '''
    输出示例
        <TerminationManager> contains 3 active terms.
        +-----------------------------------+
        | Active Termination Terms          |
        +-------+------------------+--------+
        | Index | Name             | Time Out |
        +-------+------------------+--------+
        |   0   | time_out         |   True  |
        |   1   | illegal_contact  |  False  |
        |   2   | bad_orientation  |  False  |
        +-------+------------------+--------+

    Time Out——表示这个终止条件被归类为"时间截断"还是"真实终止"。
        Time Out	归类	            对训练的影响
        True	    _truncated_buf	    只是超时，Bootstrap 继续估计 value
        False	    _terminated_buf	    真正失败，不 Bootstrap
    一眼看出 time_out 是截断型（无害），illegal_contact 和 bad_orientation 是致命型（真的死了）。
    '''

    """
    Properties.
    """
    """属性。
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active termination terms."""
        """事件终止项的名称"""
        return self._term_names

    @property
    def dones(self) -> torch.Tensor:
        """The net termination signal. Shape is (num_envs,)."""
        """网络终止信号。
        形状是 (num_envs，)。
        """
        return self._truncated_buf | self._terminated_buf
    '''
    dones — 总体终止信号
        return self._truncated_buf | self._terminated_buf
        位或运算：只要"截断"或"真实终止"任意一个为 True，dones 就是 True。
        ManagerBasedRLEnv.step() 用它决定是否重置环境：
            # step() 中:
            self.reset_buf = self.termination_manager.compute()
            # 等价于: _truncated_buf | _terminated_buf
    '''

    '''
    TerminationTermCfg.time_out 的配置对应，即如何判断外部条件达到之后是属于 _truncated_buf 还是 _terminated_buf
        # 配置中:
        TerminationsCfg:
            time_out = DoneTermCfg(func=mdp.time_out, time_out=True)          # → _truncated_buf
            illegal_contact = DoneTermCfg(func=mdp.illegal_contact, time_out=False)  # → _terminated_buf
        compute() 中根据每个 term 的 time_out 标志将结果路由到 _truncated_buf 或 _terminated_buf。
    '''

    @property
    def time_outs(self) -> torch.Tensor:
        """The timeout signal (reaching max episode length). Shape is (num_envs,).

        This signal is set to true if the environment has ended after an externally defined condition
        (that is outside the scope of a MDP). For example, the environment may be terminated if the episode has
        timed out (i.e. reached max episode length).
        """
        """截止时间信号 (达到最长的回合长度)。
        形状是 (num_envs，)。

        如果环境在外部定义的条件后结束，则该信号设置为 true (即在外部定义的条件之外).MDP)。
        例如，如果事件已结束时，环境可能会被终止 (i.e.达到最高事件长度)。
        """
        return self._truncated_buf

    @property
    def terminated(self) -> torch.Tensor:
        """The terminated signal (reaching a terminal state). Shape is (num_envs,).

        This signal is set to true if the environment has reached a terminal state defined by the environment.
        This state may correspond to task success, task failure, robot falling, etc.
        """
        """终止信号 (达到终端状态)。
        形状是 (num_envs，)。

        如果环境达到环境定义的终端状态，则该信号将设置为真实。
        这种状态可能与任务成功，任务失败，机器人摔倒等相符。
        """
        return self._terminated_buf

    """
    Operations.
    """
    """操作。
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic counts of individual termination terms.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        """返回个人终止项的次数。

        参数：
            env_ids: 环境 ID。
                     在 None 中，默认情况下考虑所有环境。

        返回：
            单个奖励项的回合总数字典。
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # add to episode dict
        extras = {}
        last_episode_done_stats = self._last_episode_dones.float().mean(dim=0)
        for i, key in enumerate(self._term_names):
            # store information
            extras["Episode_Termination/" + key] = last_episode_done_stats[i].item()
        # reset all the reward terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras
    '''
    _last_episode_dones 是什么？
        回顾 __init__：_last_episode_dones 是 [N, T] 的 bool 张量。在 compute() 中每次更新前，先把当前 _term_dones 备份到这里，然后才计算新的。
        reset 时读的就是被重置环境的上一帧终止状态。
        _last_episode_dones = [
            [ True, False, False],   # 环境 0: time_out=True（超时了）
            [False,  True, False],   # 环境 1: illegal_contact=True（撞了）
            [ True, False, False],   # 环境 2: time_out=True
            ...
        ]
        # 列: [time_out, illegal_contact, bad_orientation]
    .float().mean(dim=0) 做了什么?
        # 原始: Tensor[N, T] bool
        # .float() → Tensor[N, T] float  (True→1.0, False→0.0)
        # .mean(dim=0) → Tensor[T] float  (每列的平均值 = 触发比例)

        last_episode_done_stats = [0.60, 0.35, 0.05]
        #                          ↑     ↑     ↑
        #                       60%    35%   5%
        #                     超时    碰撞   姿态异常
        为什么用比例而非累加值？
            终止是二进制事件，不像奖励是连续值。说"35% 的环境因碰撞终止"比"这个回合积了 1432 次碰撞"更有意义——每个环境只终止一次，重点看分布。
    日志输出
        extras["Episode_Termination/time_out"]        = 0.60
        extras["Episode_Termination/illegal_contact"] = 0.35
        extras["Episode_Termination/bad_orientation"] = 0.05
        和 RewardManager.reset() 的输出并排显示在 TensorBoard 中，一眼看出"当前训练中，60% 的终止是超时，35% 是碰撞"
        ——如果 illegal_contact 比例过高，可能需要调整奖励权重。
    '''

    def compute(self) -> torch.Tensor:
        """Computes the termination signal as union of individual terms.

        This function calls each termination term managed by the class and performs a logical OR operation
        to compute the net termination signal.

        Returns:
            The combined termination signal of shape (num_envs,).
        """
        """计算终止信号作为单个项的结合。

        这个函数将由类管理的每个终止项调用，并执行逻辑OR操作来计算净终止信号。

        返回：
            形状的结合终止信号 (num_envs，)。
        """
        # reset computation
        self._truncated_buf[:] = False
        self._terminated_buf[:] = False
        # iterate over all the termination terms
        for i, term_cfg in enumerate(self._term_cfgs):
            value = term_cfg.func(self._env, **term_cfg.params)
            # store timeout signal separately
            if term_cfg.time_out:
                self._truncated_buf |= value
            else:
                self._terminated_buf |= value
            # add to episode dones
            self._term_dones[:, i] = value
        # update last-episode dones once per compute: for any env where a term fired,
        # reflect exactly which term(s) fired this step and clear others
        rows = self._term_dones.any(dim=1).nonzero(as_tuple=True)[0]
        if rows.numel() > 0:
            self._last_episode_dones[rows] = self._term_dones[rows]
        '''
        _last_episode_dones 的增量更新
            # _term_dones = [
            #     [ True, False, False],   ← env 0: time_out triggered
            #     [False, False, False],   ← env 1: nothing yet
            #     [False,  True, False],   ← env 2: illegal_contact triggered
            # ]

            _term_dones.any(dim=1)  → [ True, False, True ]
            #                          env0  env1   env2

            .nonzero(as_tuple=True)[0]  → tensor([0, 2])
            # 只有 env 0 和 2 发生了终止

            _last_episode_dones[[0, 2]] = _term_dones[[0, 2]]
            # 只更新 env 0 和 2，env 1 保持不变（保留上上次的值）
        为什么只增量更新有终止的环境？
            环境 1 这步没终止，_last_episode_dones[1] 保持上一步的值（None 时为全 False）。
            当 env 1 后面终止时，_last_episode_dones[1] 会被更新。
            这样可以确保 reset() 时每个环境读到的 _last_episode_dones 就是它最后终止那一刻的快照。
        _last_episode_dones 存储的是该环境终止时刻所有 term 的完整 bool 行——既有触发的（True），也有没触发的（False）。
            只有发生终止的环境才会被更新，没终止的环境保留旧值。
            reset() 用这些完整行计算各终止条件的整体触发比例。
        '''
        # return combined termination signal
        return self._truncated_buf | self._terminated_buf

    def get_term(self, name: str) -> torch.Tensor:
        """Returns the termination term value at current step with the specified name.

        Args:
            name: The name of the termination term.

        Returns:
            The corresponding termination term value. Shape is (num_envs,).
        """
        """返回当前步骤的终止期值，使用指定名称。

        参数：
            name: 终止项的名称

        返回：
            相关终止期值
            形状是 (num_envs，)。
        """
        return self._term_dones[:, self._term_name_to_term_idx[name]]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term
        recorded at current step.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        """返回活跃的项作为可反复的双数序列。

        元组的第一个元素是项名称，第二个元素是当前阶段记录的项原始值 (s)。

        参数：
            env_idx: 具体的环境，可以从中提取活跃项。

        返回：
            积极的项。
        """
        terms = []
        for i, key in enumerate(self._term_names):
            terms.append((key, [self._term_dones[env_idx, i].float().cpu().item()]))
        return terms

    """
    Operations - Term settings.
    """
    """运营 - 项设置
    """

    def set_term_cfg(self, term_name: str, cfg: TerminationTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            term_name: The name of the termination term.
            cfg: The configuration for the termination term.

        Raises:
            ValueError: If the term name is not found.
        """
        """设置指定项的配置在管理器中。

        参数：
            term_name: 终止项的名称
            cfg: 终止项的配置

        异常：
            ValueError: 如果没有找到项名称。
        """
        if term_name not in self._term_names:
            raise ValueError(f"Termination term '{term_name}' not found.")
        # set the configuration
        self._term_cfgs[self._term_name_to_term_idx[term_name]] = cfg

    def get_term_cfg(self, term_name: str) -> TerminationTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the termination term.

        Returns:
            The configuration of the termination term.

        Raises:
            ValueError: If the term name is not found.
        """
        """获得指定项的配置。

        参数：
            term_name: 终止项的名称

        返回：
            终止项的配置

        异常：
            ValueError: 如果没有找到项名称。
        """
        if term_name not in self._term_names:
            raise ValueError(f"Termination term '{term_name}' not found.")
        # return the configuration
        return self._term_cfgs[self._term_name_to_term_idx[term_name]]

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
            if not isinstance(term_cfg, TerminationTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type TerminationTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            # check if the term is a class
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)
