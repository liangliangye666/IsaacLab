# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event manager for orchestrating operations based on different simulation events."""

from __future__ import annotations
"""基于不同的仿真事件的操作调整事件管理器。"""

import inspect
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable

from .manager_base import ManagerBase
from .manager_term_cfg import EventTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# import logger
logger = logging.getLogger(__name__)


class EventManager(ManagerBase):
    """Manager for orchestrating operations based on different simulation events.

    The event manager applies operations to the environment based on different simulation events. For example,
    changing the masses of objects or their friction coefficients during initialization/ reset, or applying random
    pushes to the robot at a fixed interval of steps. The user can specify several modes of events to fine-tune the
    behavior based on when to apply the event.

    The event terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each event term should instantiate the :class:`EventTermCfg` class.

    Event terms can be grouped by their mode. The mode is a user-defined string that specifies when
    the event term should be applied. This provides the user complete control over when event
    terms should be applied.

    For a typical training process, you may want to apply events in the following modes:

    - "prestartup": Event is applied once at the beginning of the training before the simulation starts.
      This is used to randomize USD-level properties of the simulation stage.
    - "startup": Event is applied once at the beginning of the training once simulation is started.
    - "reset": Event is applied at every reset.
    - "interval": Event is applied at pre-specified intervals of time.

    However, you can also define your own modes and use them in the training process as you see fit.
    For this you will need to add the triggering of that mode in the environment implementation as well.

    .. note::

        The triggering of operations corresponding to the mode ``"interval"`` are the only mode that are
        directly handled by the manager itself. The other modes are handled by the environment implementation.

    """
    """基于不同的仿真事件的操作调整管理器。

    事件管理器根据不同的仿真事件对环境进行操作。
    例如，在初始化/重置过程中改变对象的质量或摩擦系数，或在一定步骤间隔下将随机推向机器人。
    用户可以指定几个事件模式，以根据何时应用事件进行细节调整行为。

    事件项由包含管理器的设置和每个项的参数的配置类进行分析。
    每个事件项都应该表示:class:`EventTermCfg`类。

    事件项可以根据其模式进行组合。
    模式是一个用户定义的字符串，该字符串指定该事件项的应用时间。
    这使用户能够完全控制何时应应用事件项。

    对于典型的培训过程，您可能希望在以下模式下应用事件:

    - "prestartup":在仿真开始之前，在训练开始时一次应用事件.USD-仿真阶段的水平特性。
    - "启动":在仿真开始后，训练开始时一次应用事件。
    - "重置":每次重置时都会应用事件。
    - "间隔":事件应在预先指定的时间间隔上进行。

    但是，您也可以定义您自己的模式，并在训练过程中使用它们。
    为此，您还需要在环境实施中添加该模式的触发。

    .. 说明::

        操作的启动与``"interval"``模式相符，是唯一直接由管理器自己处理的模式。
        其他模式由环境实施来处理。
    """

    _env: ManagerBasedEnv
    """The environment instance."""
    """环境情况。"""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the event manager.

        Args:
            cfg: A configuration object or dictionary (``dict[str, EventTermCfg]``).
            env: An environment object.
        """
        """启动事件管理器。

        参数：
            cfg: 一个配置对象或字典 (``dict[str， EventTermCfg]``)。
            env: 一个环境对象。
        """
        # create buffers to parse and store terms
        self._mode_term_names: dict[str, list[str]] = dict()
        self._mode_term_cfgs: dict[str, list[EventTermCfg]] = dict()
        self._mode_class_term_cfgs: dict[str, list[EventTermCfg]] = dict()

        # call the base class (this will parse the terms config)
        super().__init__(cfg, env)
        '''
        它的三个容器不是列表，而是字典的字典。
        外层键是 mode（"reset"、"startup"、"interval"），内层值是列表：
            _mode_term_names = {
                "reset":     ["reset_cart_position", "reset_pole_position"],
                "interval":  ["random_push"],
                "startup":   ["init_material"],
            }

            _mode_term_cfgs = {
                "reset":     [EventTermCfg(func=reset_joints, ...), EventTermCfg(func=reset_joints, ...)],
                "interval":  [EventTermCfg(func=push_robot, ...)],
                "startup":   [EventTermCfg(func=init_material, ...)],
            }
        为什么需要按 mode 分组？
            回顾 EventTermCfg.mode 字段——事件有四类触发时机：
            mode	        触发时机	    谁触发
            "prestartup"	仿真前一次	    ManagerBasedEnv.__init__
            "startup"	    仿真后一次	    ManagerBasedRLEnv.load_managers
            "reset"	        每次环境重置	ManagerBasedRLEnv._reset_idx
            "interval"	    按时间间隔	    ManagerBasedRLEnv.step
            当 event_manager.apply(mode="reset") 被调用时，它只需要遍历 _mode_term_cfgs["reset"] 中的 term，不需要检查每个 term 的 mode 字段。
            按 mode 预分组是空间换时间的优化。
        '''

    '''
    按 mode 分组多表格
    输出示例
        <EventManager> contains 3 active terms.
        +-------------------------------------------+
        | Active Event Terms in Mode: 'reset'       |
        +-------+--------------------------+
        | Index | Name                     |
        +-------+--------------------------+
        |   0   | reset_cart_position      |
        |   1   | reset_pole_position      |
        +-------+--------------------------+
        +-------------------------------------------------+
        | Active Event Terms in Mode: 'startup'           |
        +-------+----------------------------------------+
        | Index | Name                                   |
        +-------+----------------------------------------+
        |   0   | init_material                          |
        +-------+----------------------------------------+
        +------------------------------------------------------+
        | Active Event Terms in Mode: 'interval'               |
        +-------+--------------------+-------------------------+
        | Index | Name               | Interval time range (s) |
        +-------+--------------------+-------------------------+
        |   0   | random_push        | (5.0, 10.0)             |
        +-------+--------------------+-------------------------+
    '''
    def __str__(self) -> str:
        """Returns: A string representation for event manager."""
        """Returns: 为事件管理器提供一个字符串表示。"""
        msg = f"<EventManager> contains {len(self._mode_term_names)} active terms.\n"

        # add info on each mode
        for mode in self._mode_term_names:
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Event Terms in Mode: '{mode}'"
            # add table headers based on mode
            if mode == "interval":
                table.field_names = ["Index", "Name", "Interval time range (s)"]
                table.align["Name"] = "l"
                for index, (name, cfg) in enumerate(zip(self._mode_term_names[mode], self._mode_term_cfgs[mode])):
                    table.add_row([index, name, cfg.interval_range_s])
            else:
                table.field_names = ["Index", "Name"]
                table.align["Name"] = "l"
                for index, name in enumerate(self._mode_term_names[mode]):
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
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active event terms.

        The keys are the modes of event and the values are the names of the event terms.
        """
        """事件项名称。

        关键是事件模式，值是事件项的名称。
        """
        return self._mode_term_names

    @property
    def available_modes(self) -> list[str]:
        """Modes of events."""
        """事件的模式。"""
        return list(self._mode_term_names.keys())

    """
    Operations.
    """
    """操作。
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # call all terms that are classes
        for mode_cfg in self._mode_class_term_cfgs.values():
            for term_cfg in mode_cfg:
                term_cfg.func.reset(env_ids=env_ids)
        '''
        第一部分：类形式 Term 的重置
        遍历所有 mode 中的所有类形式 term，调用它们的 .reset()。
        '''

        # resolve number of environments
        if env_ids is None:
            num_envs = self._env.num_envs
        else:
            num_envs = len(env_ids)
        # if we are doing interval based events then we need to reset the time left
        # when the episode starts. otherwise the counter will start from the last time
        # for that environment
        if "interval" in self._mode_term_cfgs:
            for index, term_cfg in enumerate(self._mode_term_cfgs["interval"]):
                # sample a new interval and set that as time left
                # note: global time events are based on simulation time and not episode time
                #   so we do not reset them
                if not term_cfg.is_global_time:
                    lower, upper = term_cfg.interval_range_s
                    sampled_interval = torch.rand(num_envs, device=self.device) * (upper - lower) + lower
                    self._interval_term_time_left[index][env_ids] = sampled_interval
        '''
        第二部分：interval 倒计时重置
        核心逻辑：只重置 is_global_time=False 的 interval term。
        is_global_time 的真假区别
            is_global_time	    计时基准	    reset 时的行为
            False（默认）	     回合时间	     重新随机倒计时
            True	            仿真绝对时间	不重置，继续用原来的倒计时
        为什么需要区分？
            回合时间事件（如"每个回合的前 3 秒给一个推力"）：重置后需要新计时
            绝对时间事件（如"每 10 秒模拟一次阵风"）：不应该因某个环境重置而打断节奏
            # is_global_time=False: 每个环境刚重置时，等 5~10 秒再推
            # is_global_time=True:  不管哪个环境重置，统一每 10 秒推一次
        '''

        # nothing to log here
        return {}

    '''
    事件触发的核心分发器
        这是 EventManager 中最重要的方法，被四处在 ManagerBasedRLEnv 中调用。它根据 mode 分发到三种完全不同的执行逻辑。

        它在 step() 和 _reset_idx() 中的调用点
            ManagerBasedRLEnv.__init__():
                event_manager.apply(mode="prestartup")                          ← line 162

            ManagerBasedRLEnv.load_managers():
                event_manager.apply(mode="startup")                             ← line 134

            ManagerBasedRLEnv.step():
                event_manager.apply(mode="interval", dt=self.step_dt)           ← line 285

            ManagerBasedRLEnv._reset_idx():
                event_manager.apply(mode="reset", env_ids=..., global_env_step_count=...)  ← line 439
            四次调用，每次传入不同的 mode 和配套参数。
        '''
    def apply(
        self,
        mode: str,
        env_ids: Sequence[int] | None = None,
        dt: float | None = None,
        global_env_step_count: int | None = None,
    ):
        """Calls each event term in the specified mode.

        This function iterates over all the event terms in the specified mode and calls the function
        corresponding to the term. The function is called with the environment instance and the environment
        indices to apply the event to.

        For the "interval" mode, the function is called when the time interval has passed. This requires
        specifying the time step of the environment.

        For the "reset" mode, the function is called when the mode is "reset" and the total number of environment
        steps that have happened since the last trigger of the function is equal to its configured parameter for
        the number of environment steps between resets.

        Args:
            mode: The mode of event.
            env_ids: The indices of the environments to apply the event to.
                Defaults to None, in which case the event is applied to all environments when applicable.
            dt: The time step of the environment. This is only used for the "interval" mode.
                Defaults to None to simplify the call for other modes.
            global_env_step_count: The total number of environment steps that have happened. This is only used
                for the "reset" mode. Defaults to None to simplify the call for other modes.

        Raises:
            ValueError: If the mode is ``"interval"`` and the time step is not provided.
            ValueError: If the mode is ``"interval"`` and the environment indices are provided. This is an undefined
                behavior as the environment indices are computed based on the time left for each environment.
            ValueError: If the mode is ``"reset"`` and the total number of environment steps that have happened
                is not provided.
        """
        """在指定模式中调用每个事件的时间。

        这个函数在指定模式中反复对所有事件项进行反复执行，并调用与该项相应的函数。
        函数与环境实例和环境指标调用事件。

        在"间隔"模式下，函数在时间间隔过去了时被调用。
        这需要指定环境的时间步骤。

        对于"重置"模式，当模式"重置"时调用函数，自函数的最后触发事件以来发生的环境步骤总数等于重置之间的环境步骤数量的配置参数。

        参数：
            mode: 事件的模式。
            env_ids: 适用于事件的环境索引。
                     在 None 中，默认情况下，该事件适用于所有环境。
            dt: 环境的时间步骤。
                这只用于"间隔"模式。
                默认对None来简化调用其他模式。
            global_env_step_count: 发生的环境步骤总数。
                                   这只用
                for the "reset" mode. Defaults to None to simplify the call for other modes.

        异常：
            ValueError: 如果模式是``"interval"``，并且没有提供时间步骤。
            ValueError: 如果模式是``"interval"``，并提供环境指标。
                        这是一种未定义的行为，因为环境索引根据每个环境剩下的时间计算。
            ValueError: 如果模式是``"reset"``，并未提供发生的环境步骤总数。
        """
        ########################## 校验阶段 #############################
        # check if mode is valid    # ① mode 是否注册过？
        if mode not in self._mode_term_names:
            logger.warning(f"Event mode '{mode}' is not defined. Skipping event.")
            return

        # check if mode is interval and dt is not provided  # ② interval 模式必须有 dt
        if mode == "interval" and dt is None:
            raise ValueError(f"Event mode '{mode}' requires the time-step of the environment.")
        if mode == "interval" and env_ids is not None:  # ③ interval 模式不允许外部指定 env_ids # 因为 env_ids 由倒计时自己算出
            raise ValueError(
                f"Event mode '{mode}' does not require environment indices. This is an undefined behavior"
                " as the environment indices are computed based on the time left for each environment."
            )
        # check if mode is reset and env step count is not provided # ④ reset 模式必须有步数计数
        if mode == "reset" and global_env_step_count is None:
            raise ValueError(f"Event mode '{mode}' requires the total number of environment steps to be provided.")

        ########################## 分发阶阶段 #############################
        # iterate over all the event terms
        for index, term_cfg in enumerate(self._mode_term_cfgs[mode]):   # 遍历 _mode_term_cfgs[mode]
            if mode == "interval":
                # extract time left for this term
                time_left = self._interval_term_time_left[index]
                # update the time left for each environment
                time_left -= dt

                # check if the interval has passed and sample a new interval
                # note: we compare with a small value to handle floating point errors
                if term_cfg.is_global_time:
                    if time_left < 1e-6:
                        lower, upper = term_cfg.interval_range_s
                        sampled_interval = torch.rand(1) * (upper - lower) + lower
                        self._interval_term_time_left[index][:] = sampled_interval

                        # call the event term (with None for env_ids)
                        term_cfg.func(self._env, None, **term_cfg.params)
                else:
                    valid_env_ids = (time_left < 1e-6).nonzero().flatten()
                    if len(valid_env_ids) > 0:
                        lower, upper = term_cfg.interval_range_s
                        sampled_time = torch.rand(len(valid_env_ids), device=self.device) * (upper - lower) + lower
                        self._interval_term_time_left[index][valid_env_ids] = sampled_time

                        # call the event term
                        term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
                '''
                is_global_time=False:
                    每个 env reset 后重新计时；
                    该 env 到点就扰动一次；
                    扰动后重新采样下一次；
                    同一 episode 足够长时可多次扰动。

                is_global_time=True:
                    全体 env 共用仿真全局计时；
                    reset 不影响计时；
                    到点时所有 env 一起扰动；
                    某个 env 可能刚 reset 就被全局扰动命中。
                全局 vs 独立：
                                is_global_time=True	        is_global_time=False
                    time_left	标量 Tensor([1])	         向量 Tensor([N])
                    重新采样	 torch.rand(1) 随机一个值	    torch.rand(N) 每个环境独立随机
                    触发范围	 env_ids=None（全部）	        valid_env_ids（只到期的那几个）
                '''
            elif mode == "reset":
                # obtain the minimum step count between resets
                min_step_count = term_cfg.min_step_count_between_reset
                # resolve the environment indices
                if env_ids is None:
                    env_ids = slice(None)

                # We bypass the trigger mechanism if min_step_count is zero, i.e. apply term on every reset call.
                # This should avoid the overhead of checking the trigger condition.
                if min_step_count == 0: # 子情况 A：min_step_count=0（每次重置都触发）
                    self._reset_term_last_triggered_step_id[index][env_ids] = global_env_step_count
                    self._reset_term_last_triggered_once[index][env_ids] = True

                    # call the event term with the environment indices
                    term_cfg.func(self._env, env_ids, **term_cfg.params)
                else:   # 子情况 B：min_step_count>0（有冷却时间）
                    # extract last reset step for this term
                    last_triggered_step = self._reset_term_last_triggered_step_id[index][env_ids]
                    triggered_at_least_once = self._reset_term_last_triggered_once[index][env_ids]
                    # compute the steps since last reset
                    steps_since_triggered = global_env_step_count - last_triggered_step

                    # check if the term can be applied after the minimum step count between triggers has passed
                    valid_trigger = steps_since_triggered >= min_step_count
                    # check if the term has not been triggered yet (in that case, we trigger it at least once)
                    # this is usually only needed at the start of the environment
                    valid_trigger |= (last_triggered_step == 0) & ~triggered_at_least_once

                    # select the valid environment indices based on the trigger
                    if env_ids == slice(None):
                        valid_env_ids = valid_trigger.nonzero().flatten()
                    else:
                        valid_env_ids = env_ids[valid_trigger]

                    # reset the last reset step for each environment to the current env step count
                    if len(valid_env_ids) > 0:
                        self._reset_term_last_triggered_once[index][valid_env_ids] = True
                        self._reset_term_last_triggered_step_id[index][valid_env_ids] = global_env_step_count

                        # call the event term
                        term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
                    '''
                    min_step_count_between_reset 是两次触发之间的最小间隔（节流/限频），不是"重置后延迟一段时间再触发"（延迟）。
                    它防止同一个环境因为频繁摔倒而在短时间内反复触发同一个事件，浪费训练步数。
                    首次触发不受冷却限制，保证训练初期正常运作。
                    '''
            else:   # "startup"、"prestartup" 等模式直接触发，不需要任何条件判断。调用者保证只在合适的时机调用。
                # call the event term
                term_cfg.func(self._env, env_ids, **term_cfg.params)

    """
    Operations - Term settings.
    """
    """运营 - 项设置
    """

    '''
    运行时动态修改事件配置
        EventManager.set_term_cfg 是一个运行时配置修改器，允许你在训练过程中动态替换某个事件 term 的配置。
        它和 ManagerBase.find_terms（manager_base.py:411）配合使用，实现"先搜索后修改"的工作流。
    '''
    def set_term_cfg(self, term_name: str, cfg: EventTermCfg):
        """Sets the configuration of the specified term into the manager.

        The method finds the term by name by searching through all the modes.
        It then updates the configuration of the term with the first matching name.

        Args:
            term_name: The name of the event term.
            cfg: The configuration for the event term.

        Raises:
            ValueError: If the term name is not found.
        """
        """设置指定项的配置在管理器中。

        该方法通过搜索所有模式来找到名字。
        然后它会更新这个项的配置，

        参数：
            term_name: 事件的名称。
            cfg: 事件项的配置。

        异常：
            ValueError: 如果没有找到项名称。
        """
        term_found = False
        for mode, terms in self._mode_term_names.items():
            if term_name in terms:  # 此处是精确全字匹配
                self._mode_term_cfgs[mode][terms.index(term_name)] = cfg
                term_found = True
                break
        '''
        找到第一个匹配就退出。这意味着：如果（意外地）同一个名字出现在两个 mode 下，只会修改第一个匹配到的。不过正常情况下这种情况不会发生。
        '''
        if not term_found:
            raise ValueError(f"Event term '{term_name}' not found.")
        '''
        in 对不同容器的行为
            容器	in 检查什么	                                复杂度
            list	遍历每个元素，做 == 比较	                 O(n)
            dict	检查键是否存在	                            O(1)
            set	    哈希查找	                                O(1)
            str	    子串匹配（"cart" in "reset_cart" → True）	O(n)
            特别注意：in 对字符串是子串匹配，但对列表是精确元素匹配。两者行为不同，这是 Python 初学者容易混淆的地方。
        '''
        '''
        实际使用场景
            # 训练脚本中：训练到 1000 步后，把推力事件的力范围加大
            thrust_terms = event_manager.find_terms("random_push")
            if len(thrust_terms) > 0:
                new_cfg = old_cfg.to_dict()  # 或直接用现有 cfg 对象替换
                event_manager.set_term_cfg("random_push", new_cfg)
        '''

    '''
    运行时读取事件配置
    '''
    def get_term_cfg(self, term_name: str) -> EventTermCfg:
        """Gets the configuration for the specified term.

        The method finds the term by name by searching through all the modes.
        It then returns the configuration of the term with the first matching name.

        Args:
            term_name: The name of the event term.

        Returns:
            The configuration of the event term.

        Raises:
            ValueError: If the term name is not found.
        """
        """获得指定项的配置。

        该方法通过搜索所有模式来找到名字。
        然后返回了这个项的配置，

        参数：
            term_name: 事件的名称。

        返回：
            事件时间的配置。

        异常：
            ValueError: 如果没有找到项名称。
        """
        for mode, terms in self._mode_term_names.items():
            if term_name in terms:
                return self._mode_term_cfgs[mode][terms.index(term_name)]
        raise ValueError(f"Event term '{term_name}' not found.")

    """
    Helper functions.
    """
    """辅助函数。
    """

    def _prepare_terms(self):
        # buffer to store the time left for "interval" mode
        # if interval is global, then it is a single value, otherwise it is per environment
        self._interval_term_time_left: list[torch.Tensor] = list()
        # buffer to store the step count when the term was last triggered for each environment for "reset" mode
        self._reset_term_last_triggered_step_id: list[torch.Tensor] = list()
        self._reset_term_last_triggered_once: list[torch.Tensor] = list()
        '''
        缓冲区	                                存储内容	                            用途
        _interval_term_time_left	            每个 interval term 的倒计时张量	        apply() 中 time_left -= dt
        _reset_term_last_triggered_step_id	    每个 reset term 的上次触发步数	        apply() 中计算冷却是否已过
        _reset_term_last_triggered_once	        每个 reset term 的"是否触发过"标记	    apply() 中首次豁免逻辑
        '''

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
            if not isinstance(term_cfg, EventTermCfg):  # 类型校验（必须是 EventTermCfg）
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type EventTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # min_step_count 只在 reset 模式有效的警告
            if term_cfg.mode != "reset" and term_cfg.min_step_count_between_reset != 0:
                logger.warning(
                    f"Event term '{term_name}' has 'min_step_count_between_reset' set to a non-zero value"
                    " but the mode is not 'reset'. Ignoring the 'min_step_count_between_reset' value."
                )

            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)

            '''
            prestartup 的 scene replication 检查
                背景：
                    prestartup 模式的事件在仿真开始前执行，通常用于 USD 级别的场景随机化（如随机改变地形的 USD 属性）。
                    如果开启了 replicate_physics（PhysX 场景复制优化），多个环境会共享同一个 USD 资产——改了其中一个，所有环境都会受影响。
                这是一个硬性约束，不是警告：如果用 USD 级别的随机化，就不能开启场景复制。
            '''
            # check if mode is pre-startup and scene replication is enabled
            if term_cfg.mode == "prestartup" and self._env.scene.cfg.replicate_physics:
                raise RuntimeError(
                    "Scene replication is enabled, which may affect USD-level randomization."
                    " When assets are replicated, their properties are shared across instances,"
                    " potentially leading to unintended behavior."
                    " For stable USD-level randomization, please disable scene replication"
                    " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
                )

            '''
            prestartup 的特殊预初始化
                这是 EventManager 独有的逻辑。
                回顾 ManagerBase._process_term_cfg_at_play（manager_base.py:748），类实例化通常延迟到仿真 PLAY 事件后才执行。

                但 prestartup 模式的事件必须在仿真播放前执行（因为要改 USD 属性），所以不能等 PLAY 事件。这里提前实例化。

                注意：只对 prestartup 类做预初始化，其他 mode 的类仍走正常的延迟解析流程。
            '''
            # for event terms with mode "prestartup", we assume a callable class term
            # can be initialized before the simulation starts.
            # this is done to ensure that the USD-level randomization is possible before the simulation starts.
            if inspect.isclass(term_cfg.func) and term_cfg.mode == "prestartup":
                logger.info(f"Initializing term '{term_name}' with class '{term_cfg.func.__name__}'.")
                term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)

            '''
            注册 mode 并存入
                自动注册：用户不需要预先声明有哪些 mode。第一个 mode="reset" 的 term 到来时自动创建 _mode_term_names["reset"]，第二个追加进去。
                支持用户自定义 mode（如 mode="my_custom_event"）。
            '''
            # check if mode is a new mode
            if term_cfg.mode not in self._mode_term_names:
                # add new mode
                self._mode_term_names[term_cfg.mode] = list()
                self._mode_term_cfgs[term_cfg.mode] = list()
                self._mode_class_term_cfgs[term_cfg.mode] = list()
            # add term name and parameters
            self._mode_term_names[term_cfg.mode].append(term_name)
            self._mode_term_cfgs[term_cfg.mode].append(term_cfg)

            # check if the term is a class
            if inspect.isclass(term_cfg.func):
                self._mode_class_term_cfgs[term_cfg.mode].append(term_cfg)

            '''
            interval 模式的初始化
                is_global_time=True → 一个标量（所有环境共享），使用绝对时间无视各环境的reset，
                False → 每个环境独立的向量，以各环境的reset为基准。
                这和之前 apply() 中的处理逻辑一致。
            '''
            # resolve the mode of the events
            # -- interval mode
            if term_cfg.mode == "interval":
                if term_cfg.interval_range_s is None:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'interval' but 'interval_range_s' is not specified."
                    )

                # sample the time left for global
                if term_cfg.is_global_time:
                    lower, upper = term_cfg.interval_range_s
                    time_left = torch.rand(1) * (upper - lower) + lower
                    self._interval_term_time_left.append(time_left)
                else:
                    # sample the time left for each environment
                    lower, upper = term_cfg.interval_range_s
                    time_left = torch.rand(self.num_envs, device=self.device) * (upper - lower) + lower
                    self._interval_term_time_left.append(time_left)
                '''
                reset 模式的初始化
                '''
            # -- reset mode
            elif term_cfg.mode == "reset":
                if term_cfg.min_step_count_between_reset < 0:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'reset' but 'min_step_count_between_reset' is"
                        f" negative: {term_cfg.min_step_count_between_reset}. Please provide a non-negative value."
                    )

                # initialize the current step count for each environment to zero    # 上次触发步数 → 初始化为 0（从未触发过）
                step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
                self._reset_term_last_triggered_step_id.append(step_count)
                # initialize the trigger flag for each environment to zero  # "是否触发过"标记 → 初始化为 False
                no_trigger = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                self._reset_term_last_triggered_once.append(no_trigger)
