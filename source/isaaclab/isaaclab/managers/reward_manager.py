# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward manager for computing reward signals for a given world."""
"""计算一个特定世界的奖励信号的奖励管理器。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RewardManager(ManagerBase):
    """Manager for computing reward signals for a given world.

    The reward manager computes the total reward as a sum of the weighted reward terms. The reward
    terms are parsed from a nested config class containing the reward manger's settings and reward
    terms configuration.

    The reward terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each reward term should instantiate the :class:`RewardTermCfg` class.

    .. note::

        The reward manager multiplies the reward term's ``weight``  with the time-step interval ``dt``
        of the environment. This is done to ensure that the computed reward terms are balanced with
        respect to the chosen time-step interval in the environment.

    """
    """管理一个特定世界的计算奖励信号。

    奖励管理器将总奖励计算为加权奖励项的总和。
    奖励条件由包含奖励的设置和奖励条件配置的嵌套配置类进行分析。

    奖励项由包含管理器的设置和每个项的参数的配置类进行分析。
    每个奖励项都应该标记:class:`RewardTermCfg`类。

    .. 说明::

        奖励管理器将奖励期的``weight``乘以环境的时间步骤间隔``dt``。
        这样才能确保计算的奖励条件与环境中选择的时间步骤间隔保持平衡。
    """

    _env: ManagerBasedRLEnv
    """The environment instance."""
    """环境情况。"""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the reward manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, RewardTermCfg]``).
            env: The environment instance.
        """
        """启动奖励管理器。

        参数：
            cfg: 配置对象或字典 (``dict[str， RewardTermCfg]``)。
            env: 环境情况。
        """
        # create buffers to parse and store terms
        self._term_names: list[str] = list()    # 存放 term 名字
        self._term_cfgs: list[RewardTermCfg] = list()   # 存放所有配置
        self._class_term_cfgs: list[RewardTermCfg] = list() # 存放类实现的 term（reset 时特殊处理）

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)
        # prepare extra info to store individual reward term information
        self._episode_sums = dict() # 每个 term 一个 [N] 张量，记录回合累积奖励
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        '''
        _episode_sums = {
            "alive":       tensor([0.15, 0.12, 0.15, ...]),   # alive 奖励的回合累积
            "terminating": tensor([0.00, 0.00, 0.00, ...]),   # terminating 奖励的回合累积
            "pole_pos":    tensor([-0.05, -0.03, -0.01, ...]), # pole_pos 奖励的回合累积
        }
        '''

        # create buffer for managing reward per environment
        self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)    # [N]   ← 总奖励（所有 term 加权和）
        # reward_buf[i] = alive_weight * alive[i] * dt + terminating_weight * terminating[i] * dt + ...

        # Buffer which stores the current step reward for each term for each environment
        self._step_reward = torch.zeros((self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device)
        # [N, T] ← 每个 term 每步的原始值,N=环境数, T=term 数量
        '''
        用于 get_active_iterable_terms 给 GUI 展示"每项奖励各自是多少"：
            # 第 0 个环境，第 2 个 term (pole_pos) 的当前步奖励:
                _step_reward[0, 2]  →  -0.023
        '''

    '''
    含权重列的奖励信息面板
    '''
    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        """Returns: 一个奖励管理器的字符串表示。"""
        msg = f"<RewardManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Reward Terms"
        table.field_names = ["Index", "Name", "Weight"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Weight"] = "r"
        # add info on each term
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name, term_cfg.weight])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg
    '''
    输出示例
        <RewardManager> contains 3 active terms.
        +----------------------------------+
        | Active Reward Terms              |
        +-------+-------------+-----------+
        | Index | Name        |    Weight |
        +-------+-------------+-----------+
        |   0   | alive       |      1.0  |
        |   1   | terminating |     -2.0  |
        |   2   | pole_pos    |     -1.0  |
        +-------+-------------+-----------+
    '''

    """
    Properties.
    """
    """属性。
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active reward terms."""
        """事件奖励条件的名称。"""
        return self._term_names
    # 返回: ["alive", "terminating", "pole_pos", "cart_vel", "pole_vel"]


    """
    Operations.
    """
    """操作。
    """

    '''
    reset() 输出回合奖励日志：
        遍历 _episode_sums（已包含 weight×dt 的累积值），取被重置环境的均值，再除以 max_episode_length_s 统一为"每秒平均奖励"，存入 Episode_Reward/{term} 前缀的 extras。
        然后归零累积器和重置类 term。
        除以全局最大回合长度的归一化让不同步数的回合在日志中可比。
    '''
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic sum of individual reward terms.

        Args:
            env_ids: The environment ids for which the episodic sum of
                individual reward terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        """返回个别奖励条件的事件总数。

        参数：
            env_ids: 必须返回单个奖励条件的事件总和的环境ID。
                     所有环境 ID的默认。

        返回：
            单个奖励项的回合总数字典。
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for key in self._episode_sums.keys():
            # store information
            # r_1 + r_2 + ... + r_n
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self._env.max_episode_length_s
            # reset episodic sum
            self._episode_sums[key][env_ids] = 0.0
        # reset all the reward terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras
    '''
    为什么要除以 max_episode_length_s？
        假设两个环境都在 Cartpole 任务中：
            环境	结束步数	_episode_sums["alive"] 累积	    直接平均值
            A	    300 步	    300 × 1.0 × 0.02 = 6.0	        6.0
            B	    50 步	    50 × 1.0 × 0.02 = 1.0	        1.0
        环境 A 活了 300 步拿了 6.0，环境 B 活了 50 步拿了 1.0——表面差 6 倍。但如果除以各自步数，两者都会回到约 0.02/步。

        除以 max_episode_length_s（全局常量，如 5 秒）的作用是把所有回合统一到"每秒平均奖励"的尺度，让日志曲线在不同长度的回合之间可比。
            env A: 6.0 / 5.0 = 1.2   ← 每秒 aliving 奖励 = 1.2
            env B: 1.0 / 5.0 = 0.2   ← 每秒 aliving 奖励 = 0.2
        现在你可以看到：A 不仅活得更久（秒均 1.2），B 秒均只有 0.2——说明 B 不仅短命，而且行为质量也差。

        * dt 解决的是控制频率归一化：
            同样跑 1 秒：
            step_dt=0.02 -> 50 步
            step_dt=0.01 -> 100 步
        如果每步都给 1.0，不乘 dt，100 步的环境会拿到 100，50 步的环境只拿到 50。
        所以 [compute() (line 253)] 里乘 dt，让 reward 近似变成时间积分：
            episode_sum = sum(reward_rate * dt) ≈ ∫ reward_rate dt
        但 / max_episode_length_s 解决的是另一个问题：
        把整个 episode return 放到固定最大时长尺度上做日志展示。它不是训练必需的，只影响 [reset() 写入 extras["log"] 的统计值。
        关键区别：
            episode_sum / 当前回合时长
            = 当前存活期间的平均 reward rate

            episode_sum / max_episode_length_s
            = 按最大回合长度归一化后的 episode return
            = 平均 reward rate * 当前回合完成比例
        举例，alive 每秒奖励率为 1.0，最大回合 20 秒：
            A 活了 20 秒: episode_sum = 20
            B 活了  2 秒: episode_sum = 2
        如果除以当前回合总时长：
            A: 20 / 20 = 1.0
            B:  2 /  2 = 1.0
        两个日志一样，看不出 B 很快失败。
        如果除以 max_episode_length_s：
            A: 20 / 20 = 1.0
            B:  2 / 20 = 0.1
        B 的提前终止会体现在日志里。
        所以你的判断可以这样修正：
            乘 dt：有必要，用于让 reward 不依赖环境步频。
            除以 max_episode_length_s：不是训练必要项，是 IsaacLab 选择的一种日志归一化方式。
            如果你想看“当前回合存活期间平均每秒 reward”，那确实应该除以当前回合时长。
            但如果你想让提前失败的 episode 在日志上明显变差，就应该除以最大回合时长。
    '''

    '''
    奖励计算的加权求和引擎
        它是奖励的中央计算器：遍历所有奖励项，调函数 → 加权 → 乘时间步 → 累加 → 更新 3 个缓冲区。
    '''
    def compute(self, dt: float) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        """计算奖励信号作为个体项的权重总数。

        这个函数将由类管理的每个奖励项调用，并添加它们来计算净奖励信号。
        它还更新了各个奖励项相应的集体金额。

        参数：
            dt: 环境的时间间隔。

        返回：
            形状的净奖励信号 (num_envs，)。
        """
        # reset computation
        self._reward_buf[:] = 0.0
        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value

            # Update current reward for this step.
            self._step_reward[:, term_idx] = value / dt

        return self._reward_buf
    '''
    数值对比：
        存储位置	        值	                        含义
        _reward_buf	    func * weight * dt	        加权总奖励（含时间步）,只计算当前回合的
        _episode_sums	累加 func * weight * dt	    回合加权累积
        _step_reward	func * weight	            每步纯值（去掉 dt，给 GUI）
    '''
    '''
    这里的 dt 是环境步时间，不是物理仿真步时间。
        调用点在 [manager_based_rl_env.py ：
            self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        而 step_dt 定义在 [manager_based_env.py (line 303)]：
            return self.cfg.sim.dt * self.cfg.decimation
        所以：
            physics_dt = cfg.sim.dt
            step_dt    = cfg.sim.dt * decimation
        原因是一个环境 step 里面会跑多次 physics step：
            for _ in range(self.cfg.decimation):
                self.sim.step(...)
        但 reward 只在这个环境 step 结束后算一次。因此 reward 用的是这整个环境步跨过的时间 step_dt。

    为什么奖励要乘 dt：
        为了让 reward 对控制频率不敏感。IsaacLab 把很多 reward term 当成“每秒奖励率/惩罚率”来写，然后乘 dt 做时间积分：
            value = reward_term * weight * dt
        例如 alive reward 的原始值恒为 1.0：
            step_dt = 0.02s: 每步奖励 1.0 * 0.02 = 0.02，1 秒 50 步，总计 1.0
            step_dt = 0.01s: 每步奖励 1.0 * 0.01 = 0.01，1 秒 100 步，总计 1.0
        如果不乘 dt，同样 1 秒内，控制频率越高，累计 reward 越大，训练目标就会随仿真/控制频率改变。
    '''

    """
    Operations - Term settings.
    """
    """运营 - 项设置
    """

    '''
    运行时替换奖励项配置
    '''
    def set_term_cfg(self, term_name: str, cfg: RewardTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            term_name: The name of the reward term.
            cfg: The configuration for the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        """设置指定项的配置在管理器中。

        参数：
            term_name: 奖励项的名称。
            cfg: 奖励项的配置。

        异常：
            ValueError: 如果没有找到项名称。
        """
        if term_name not in self._term_names:
            raise ValueError(f"Reward term '{term_name}' not found.")
        # set the configuration
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str) -> RewardTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the reward term.

        Returns:
            The configuration of the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        """获得指定项的配置。

        参数：
            term_name: 奖励项的名称。

        返回：
            奖励项的配置。

        异常：
            ValueError: 如果没有找到项名称。
        """
        if term_name not in self._term_names:
            raise ValueError(f"Reward term '{term_name}' not found.")
        # return the configuration
        return self._term_cfgs[self._term_names.index(term_name)]

    '''
    GUI 面板显示的是 func() * weight——原始奖励函数值乘权重，去掉了时间缩放
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
        for idx, name in enumerate(self._term_names):
            terms.append((name, [self._step_reward[env_idx, idx].cpu().item()]))
        return terms
    '''
    返回示例
        get_active_iterable_terms(env_idx=0)

        [
            ("alive",       [1.0]),
            ("terminating", [0.0]),
            ("pole_pos",    [-0.023]),
            ("cart_vel",    [-0.001]),
        ]
        每个值被包在单元素列表里（[1.0] 而非 1.0），保持和其他 Manager 返回格式一致（Sequence[float]）。
    '''

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
            if not isinstance(term_cfg, RewardTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type RewardTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # check for valid weight type
            if not isinstance(term_cfg.weight, (float, int)):
                raise TypeError(
                    f"Weight for the term '{term_name}' is not of type float or int."
                    f" Received: '{type(term_cfg.weight)}'."
                )
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            # check if the term is a class
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)
