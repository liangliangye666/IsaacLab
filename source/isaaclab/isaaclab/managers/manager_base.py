# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import inspect
import logging
import weakref
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import omni.timeline

import isaaclab.utils.string as string_utils
from isaaclab.utils import class_to_dict, string_to_callable

from .manager_term_cfg import ManagerTermBaseCfg
from .scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# import logger
logger = logging.getLogger(__name__)


class ManagerTermBase(ABC):
    """Base class for manager terms.

    Manager term implementations can be functions or classes. If the term is a class, it should
    inherit from this base class and implement the required methods.

    Each manager is implemented as a class that inherits from the :class:`ManagerBase` class. Each manager
    class should also have a corresponding configuration class that defines the configuration terms for the
    manager. Each term should the :class:`ManagerTermBaseCfg` class or its subclass.

    Example pseudo-code for creating a manager:

    .. code-block:: python

        from isaaclab.utils import configclass
        from isaaclab.utils.mdp import ManagerBase, ManagerTermBaseCfg


        @configclass
        class MyManagerCfg:
            my_term_1: ManagerTermBaseCfg = ManagerTermBaseCfg(...)
            my_term_2: ManagerTermBaseCfg = ManagerTermBaseCfg(...)
            my_term_3: ManagerTermBaseCfg = ManagerTermBaseCfg(...)


        # define manager instance
        my_manager = ManagerBase(cfg=ManagerCfg(), env=env)

    """
    """管理器项的基类。

    manager term 可以由函数或类实现。若采用类实现，该类应继承本基类并实现所需方法。

    每一种管理器都继承 :class:`ManagerBase`，并具有对应的配置类来声明管理器包含的各个配置项。
    每个配置项都应为 :class:`ManagerTermBaseCfg` 或其子类的实例。

    创建管理器的伪代码如下：

    .. code-block:: python

        from isaaclab.utils import configclass
        from isaaclab.utils.mdp import ManagerBase, ManagerTermBaseCfg


        @configclass
        class MyManagerCfg:
            my_term_1: ManagerTermBaseCfg = ManagerTermBaseCfg(...)
            my_term_2: ManagerTermBaseCfg = ManagerTermBaseCfg(...)
            my_term_3: ManagerTermBaseCfg = ManagerTermBaseCfg(...)


        # define manager instance
        my_manager = ManagerBase(cfg=ManagerCfg(), env=env)
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        """Initialize the manager term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        """初始化管理器项。

        参数：
            cfg: 配置对象。
            env: 环境实例。
        """
        # store the inputs
        self.cfg = cfg
        self._env = env

    """
    Properties.
    """
    """属性。
    """

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        """环境数量"""
        return self._env.num_envs

    @property
    def device(self) -> str:
        """Device on which to perform computations."""
        """用于执行计算的设备。"""
        return self._env.device

    @property
    def __name__(self) -> str:
        """Return the name of the class or subclass."""
        """返回类或子类的名称。"""
        return self.__class__.__name__
    '''
    # 方案 A：直接存字符串
        class 父类:
            __name__ = "父类"

        class 子类(父类):
            __name__ = "子类"           # 每个子类都要手动写一遍

    # 方案 B：返回 self.__class__.__name__（实际采用）
        class 父类:
            @property
            def __name__(self):
                return self.__class__.__name__  # 自动获取！
    方法 B 的好处：子类继承时不需要重定义。无论你有多少层继承，self.__class__ 总是指向真实的子类。
    __name__ 属性 — 多态名字获取器
        @property 装饰的 __name__ 返回 self.__class__.__name__，是一个自动适配多态的名字获取器——子类无需重定义，框架就能拿到真实的类名。
        @property 让它在访问时动态计算，而不是在 __init__ 中固化，保证了子类灵活性最大。
        因为增加了@property 装饰，所以调用时直接使用 term.__name__接口，而无需在后面增加()
    '''

    """
    Operations.
    """
    """操作。
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the manager term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        """重置管理器项。

        参数：
            env_ids: 要重置的环境 ID。默认为 None，表示处理所有环境。
        """
        pass
    '''
    pass 是 Python 的关键字，意思是"什么也不做"。它只有一个功能：占据缩进位置，让空的方法体、空循环、空条件分支语法合法。
        # ❌ 语法错误：方法体不能为空
        def reset(self):
                                # 什么都没有，缩进断了 → IndentationError
        # ✅ 用一个 pass 占住缩进
        def reset(self):
            pass                # 当前不实现具体逻辑，允许子类决定重写
    pass 是 Python 的"合法空语句"——让 reset() 有一个默认的无操作实现，子类可以选择覆盖或保持不动。
    它和 @abstractmethod 不同：不强制子类必须写，只提供一个"默认什么都不做"的钩子。
    '''

    def serialize(self) -> dict:
        """General serialization call. Includes the configuration dict."""
        """执行通用序列化，并包含该项的配置字典。"""
        return {"cfg": class_to_dict(self.cfg)}
    '''
    调用这个方法时，会：
        1. 取 self.cfg（这个 ManagerTerm 实例的配置对象，如 RewardTermCfg）
        2. 通过 class_to_dict() 把它递归转换成纯 Python 字典
        3. 用 {"cfg": ...} 包装后返回
        输入 → 输出示例：
            # 假设 self.cfg 是：
            RewardTermCfg(func=is_alive, weight=1.0, params={})

            # serialize() 返回：
            {
                "cfg": {
                    "func": "is_alive",     # 函数被转成了字符串路径
                    "weight": 1.0,
                    "params": {}
                }
            }
    serialize() 把 ManagerTerm 的配置对象（self.cfg）通过 class_to_dict 转换成纯字典，便于日志记录和 checkpoint 保存。
    返回值用 {"cfg": ...} 包装而非直接返回字典，是预留了未来扩展其他序列化信息的空间。
    '''

    def __call__(self, *args) -> Any:
        """Returns the value of the term required by the manager.

        In case of a class implementation, this function is called by the manager
        to get the value of the term. The arguments passed to this function are
        the ones specified in the term configuration (see :attr:`ManagerTermBaseCfg.params`).

        .. attention::
            To be consistent with memory-less implementation of terms with functions, it is
            recommended to ensure that the returned mutable quantities are cloned before
            returning them. For instance, if the term returns a tensor, it is recommended
            to ensure that the returned tensor is a clone of the original tensor. This prevents
            the manager from storing references to the tensors and altering the original tensors.

        Args:
            *args: Variable length argument list.

        Returns:
            The value of the term.
        """
        """返回管理器所需的该项计算结果。

        对于类形式的实现，管理器会调用此方法获取该项的值。传入参数由项配置中的
        :attr:`ManagerTermBaseCfg.params` 指定。

        .. 注意::
            为了与无状态函数项的内存行为保持一致，建议在返回可变对象前先进行复制。
            例如，若返回 Tensor，应返回原 Tensor 的 clone，避免管理器持有引用并意外修改原数据。

        参数：
            *args: 可变长度参数列表。

        返回：
            该项的计算结果。
        """
        raise NotImplementedError("The method '__call__' should be implemented by the subclass.")
    '''
    __call__ — Python 的"让对象像函数一样被调用"魔法方法
    一、__call__ 是什么？
        __call__ 是 Python 的 5 个双下划线魔法方法之一。当你在一个对象后面加括号 () 时，Python 自动调用它的 __call__ 方法。

            class 加法器:
                def __init__(self, n):
                    self.n = n
                def __call__(self, x):      # ← __call__ 让实例可以像函数一样被调用
                    return self.n + x

            add5 = 加法器(5)
            add5(10)                        # ← 等价于 add5.__call__(10)
            # → 15
    二、为什么 ManagerTermBase 需要 __call__？
        回顾之前学的 ManagerTermBaseCfg（manager_term_cfg.py:35）：
            func: Callable | ManagerTermBase = MISSING
        func 字段可以是两种东西：
            普通函数：def my_reward(env, **params): ... → Manager 直接调用 func(env, **params)
            类实例：class MyTerm(ManagerTermBase): ... → Manager 调用 func.__call__(env, **params)
        无论哪种，使用方式完全一样——func(env, **params)。这就是多态统一接口。
    '''


class ManagerBase(ABC):
    """Base class for all managers."""
    """所有管理器的基类。"""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the manager.

        This function is responsible for parsing the configuration object and creating the terms.

        If the simulation is not playing, the scene entities are not resolved immediately.
        Instead, the resolution is deferred until the simulation starts. This is done to ensure
        that the scene entities are resolved even if the manager is created after the simulation
        has already started.

        Args:
            cfg: The configuration object. If None, the manager is initialized without any terms.
            env: The environment instance.
        """
        """初始化管理器。

        该函数负责解析配置对象并创建各个 manager term。

        如果仿真尚未开始播放，则不会立即解析场景实体，而是将解析过程延迟到仿真开始时。
        如果创建管理器时仿真已经处于播放状态，则会在准备配置项时直接解析场景实体。

        参数：
            cfg: 配置对象。若为 None，则初始化一个不包含任何项的管理器。
            env: 环境实例。
        """
        # store the inputs
        self.cfg = copy.deepcopy(cfg)
        '''
        copy.deepcopy(cfg) 递归克隆整个配置对象树，确保 Manager 内部的 _process_term_cfg_at_play（修改 SceneEntityCfg 的 joint_ids、替换 func 等）只污染副本，不污染原始配置。
        你担心的"cfg 被修改后影响别处"正是它的设计动机——即使不同 Manager 拿的是不同 cfg 属性，同一个 Manager 内部的多个 term 也可能共享子对象（如同一个 SceneEntityCfg），深拷贝保证了互相隔离。
        '''
        self._env = env

        # flag for whether the scene entities have been resolved
        # if sim is playing, we resolve the scene entities directly while preparing the terms
        self._is_scene_entities_resolved = self._env.sim.is_playing()   # 检查仿真状态

        # if the simulation is not playing, we use callbacks to trigger the resolution of the scene
        # entities configuration. this is needed for cases where the manager is created after the
        # simulation, but before the simulation is playing.
        # FIXME: Once Isaac Sim supports storing this information as USD schema, we can remove this
        #   callback and resolve the scene entities directly inside `_prepare_terms`.
        if not self._env.sim.is_playing():  # 注册延迟回调
            # note: Use weakref on all callbacks to ensure that this object can be deleted when its destructor
            # is called
            # The order is set to 20 to allow asset/sensor initialization to complete before the scene entities
            # are resolved. Those have the order 10.
            timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
            self._resolve_terms_handle = timeline_event_stream.create_subscription_to_pop_by_type(
                int(omni.timeline.TimelineEventType.PLAY),
                lambda event, obj=weakref.proxy(self): obj._resolve_terms_callback(event),
                order=20,
            )
        else:
            self._resolve_terms_handle = None
        '''
        路径	条件	                                    处理方式
        路径 A	仿真未播放（大多数情况）	                    注册 PLAY 事件回调，延迟到仿真启动后再解析 SceneEntityCfg
        路径 B	仿真已播放（通过 sim.reset_async() 的手动模式）	 直接在 _prepare_terms 中解析

        代码	                                    含义
        get_timeline_event_stream()	                获取 Isaac Sim 的时间线事件流
        create_subscription_to_pop_by_type(...)	    注册一个"只执行一次"的事件订阅
        TimelineEventType.PLAY	                    订阅的事件类型：仿真开始播放
        order=20	                                执行优先级：排在 10（资产初始化）之后
        weakref.proxy(self)	                        弱引用包装（防止循环引用导致内存泄漏）

        五、通俗类比
            想象你是一家餐厅的经理（ManagerBase），上任时（__init__）：
                复印菜单（deepcopy）：确保自己的那份菜单不被别人涂改
                检查厨房状态（is_playing()）：看看食材供应商到了没
                如果供应商还没到（仿真未播放）：在前台留个便条"供应商一到，立刻通知我"（PLAY 事件回调），然后先去安排其他事
                对照菜单准备工具（_prepare_terms()）：菜单上有"红烧肉"就派一个红烧肉厨师，有"清蒸鱼"就派一个清蒸鱼厨师
            供应商到了（仿真播放），你收到通知，立刻回去把那些需要"新鲜食材"（场景实体引用）的菜品确认一遍。
        一句话总结
            ManagerBase.__init__ 是所有 Manager 的构造骨架——深拷贝配置防篡改，根据仿真状态决定是直接解析还是延迟回调，最后调用子类的 _prepare_terms() 把配置字段翻译成 Term 实例。
            weakref.proxy + order=20 的延迟回调设计巧妙解决了"创建 Manager 时场景实体还不存在"的时序问题，保证了初始化顺序的灵活性。
        '''

        # parse config to create terms information
        if self.cfg:
            self._prepare_terms()

    def __del__(self):
        """Delete the manager."""
        """删除管理器。"""
        if self._resolve_terms_handle:
            self._resolve_terms_handle.unsubscribe()
            self._resolve_terms_handle = None
    '''
    __del__ 是 Python 的析构函数，ManagerBase 用它来取消在 __init__ 中注册的 Isaac Sim PLAY 事件订阅。
    核心原则是"谁订阅谁取消"——避免资源泄漏。因为事件订阅是 Isaac Sim C++ 层的资源，不会随 Python 对象自动释放，所以必须手动清理。
    '''

    """
    Properties.
    """
    """属性。
    """

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        """环境数量"""
        return self._env.num_envs

    @property
    def device(self) -> str:
        """Device on which to perform computations."""
        """用于执行计算的设备。"""
        return self._env.device

    @property
    @abstractmethod
    def active_terms(self) -> list[str] | dict[str, list[str]]:
        """Name of active terms."""
        """当前启用项的名称。"""
        raise NotImplementedError

    """
    Operations.
    """
    """操作。
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Resets the manager and returns logging information for the current time-step.

        Args:
            env_ids: The environment ids for which to log data.
                Defaults None, which logs data for all environments.

        Returns:
            Dictionary containing the logging information.
        """
        """重置管理器，并返回当前时间步的日志信息。

        参数：
            env_ids: 需要重置并记录数据的环境 ID。默认为 None，表示全部环境。

        返回：
            包含记录信息的字典。
        """
        return {}

    def find_terms(self, name_keys: str | Sequence[str]) -> list[str]:
        """Find terms in the manager based on the names.

        This function searches the manager for terms based on the names. The names can be
        specified as regular expressions or a list of regular expressions. The search is
        performed on the active terms in the manager.

        Please check the :meth:`~isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the term names.

        Returns:
            A list of term names that match the input keys.
        """
        """按名称查找管理器中的项。

        ``name_keys`` 可以是一个正则表达式，也可以是正则表达式列表；匹配范围为管理器当前启用的项。
        名称匹配规则请参阅 :meth:`~isaaclab.utils.string_utils.resolve_matching_names`。

        参数：
            name_keys: 用于匹配项名称的正则表达式或正则表达式列表。

        返回：
            与输入表达式匹配的项名称列表。
        """
        # resolve search keys
        if isinstance(self.active_terms, dict):
            list_of_strings = []
            for names in self.active_terms.values():
                list_of_strings.extend(names)
        else:
            list_of_strings = self.active_terms

        # return the matching names
        return string_utils.resolve_matching_names(name_keys, list_of_strings)[1]
    '''
    执行流程
        find_terms(name_keys)
        │
        ├── 步骤 1: 收集所有可搜索的 Term 名字
        │     self.active_terms 是什么？
        │     ├── 如果是 dict (ObservationManager)
        │     │     {"policy": ["joint_pos", "joint_vel"], "critic": ["base_lin_vel"]}
        │     │     → 展平为一维列表: ["joint_pos", "joint_vel", "base_lin_vel"]
        │     └── 如果是 list (RewardManager, TerminationManager 等)
        │           ["alive", "terminating", "pole_pos"]
        │           → 直接用
        │
        └── 步骤 2: 正则匹配
            resolve_matching_names(name_keys, list_of_strings)
            → 返回 (匹配到的索引列表, 匹配到的名字列表)
            → 只取名字列表: [1]
    为什么 active_terms 可能是 dict？
        只有 ObservationManager 的 active_terms 是 dict，因为观测是分组的（policy 组、critic 组）。
        你需要把各组内的 Term 名展平（list_of_strings.extend(names)），才能统一做正则搜索。
            # ObservationManager.active_terms 示例:
            {
                "policy":  ["joint_pos", "joint_vel", "actions"],
                "critic":  ["base_lin_vel", "base_ang_vel"],
            }

            # 展平后:
            ["joint_pos", "joint_vel", "actions", "base_lin_vel", "base_ang_vel"]
    这个功能在什么场景下有用？
        场景一：实现通用的奖励缩放/关闭
            # 训练脚本中：把所有速度相关的奖励权重设为 0
            velocity_terms = reward_manager.find_terms(".*vel.*")
            for name in velocity_terms:
                reward_manager.set_term_cfg(name, weight=0.0)
    一句话总结
        find_terms 允许用正则表达式批量查找 Manager 中的 Term 名字。
        对于 ObservationManager 这种分组结构的（dict），会先把所有分组内的 term 名字展平成列表。
        底层调用 resolve_matching_names 做正则匹配，返回匹配到的名字列表。
        这让训练脚本可以根据模式动态调整奖励权重、观测选择等，而不需要硬编码完整的 Term 名字。
    '''

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Returns:
            The active terms.
        """
        """以可迭代元组序列返回当前启用的项。

        每个元组的第一个元素是项名称，第二个元素是该项的原始值。

        返回：
            当前启用的项。
        """
        raise NotImplementedError
    '''
    为什么是 Sequence[float] 而非 float？
        因为一个 Term 可能返回多维数据。比如 joint_pos 观测返回 7 个关节的角度值：
            # 单维 Term
            ("alive", [1.0])

            # 多维 Term
            ("joint_pos", [0.0, 0.5, -0.3, 0.0, 1.2, 0.0, -0.1])  # 7 个关节
            Sequence[float] 统一用列表表示，无论 1 个值还是多个值。
    '''

    """
    Implementation specific.
    """
    """具体实现。
    """

    @abstractmethod
    def _prepare_terms(self):
        """Prepare terms information from the configuration object."""
        """根据配置对象准备各项信息。"""
        raise NotImplementedError

    """
    Internal callbacks.
    """
    """内部回调。
    """

    def _resolve_terms_callback(self, event):
        """Resolve configurations of terms once the simulation starts.

        Please check the :meth:`_process_term_cfg_at_play` method for more information.
        """
        """在仿真开始播放后解析各项配置。

        更多信息请参阅 :meth:`_process_term_cfg_at_play`。
        """
        # check if scene entities have been resolved
        if self._is_scene_entities_resolved:
            return
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
            # process attributes at runtime
            # these properties are only resolvable once the simulation starts playing
            self._process_term_cfg_at_play(term_name, term_cfg)

        # set the flag
        self._is_scene_entities_resolved = True
    '''
    _resolve_terms_callback 是 Isaac Sim PLAY 事件的回调（在 __init__ 中注册），它的工作是：等仿真启动后（PhysX 已加载所有资产），遍历所有 Term 配置，
    把其中的 SceneEntityCfg 从"只有名字的引用"解析为"有具体 joint_ids / body_ids 的实体"，并把配置中的类（class_type）实例化为对象。_
    is_scene_entities_resolved flag 防止重复解析。
    这是 IsaacLab 中"谁先初始化谁后初始化"时序控制的精妙体现。
    '''

    """
    Internal functions.
    """
    """内部功能。
    """

    '''
    _resolve_common_term_cfg 是一条 6 站流水线：①类型校验→②字符串转函数→③可调用性→④类模式适配→⑤inspect.signature 参数匹配→⑥按需运行时解析。
    每一步都是一道安全闸门，确保每个 TermCfg 在参与实际运算前，其 func 是合法的、参数是对齐的、SceneEntityCfg 在仿真启动后被正确解析。
    这是 IsaacLab "配置驱动"架构的质量保障层。
    '''
    def _resolve_common_term_cfg(self, term_name: str, term_cfg: ManagerTermBaseCfg, min_argc: int = 1):
        """Resolve common attributes of the term configuration.

        Usually, called by the :meth:`_prepare_terms` method to resolve common attributes of the term
        configuration. These include:

        * Resolving the term function and checking if it is callable.
        * Checking if the term function's arguments are matched by the parameters.
        * Resolving special attributes of the term configuration like ``asset_cfg``, ``sensor_cfg``, etc.
        * Initializing the term if it is a class.

        The last two steps are only possible once the simulation starts playing.

        By default, all term functions are expected to have at least one argument, which is the
        environment object. Some other managers may expect functions to take more arguments, for
        instance, the environment indices as the second argument. In such cases, the
        ``min_argc`` argument can be used to specify the minimum number of arguments
        required by the term function to be called correctly by the manager.

        Args:
            term_name: The name of the term.
            term_cfg: The term configuration.
            min_argc: The minimum number of arguments required by the term function to be called correctly
                by the manager.

        Raises:
            TypeError: If the term configuration is not of type :class:`ManagerTermBaseCfg`.
            ValueError: If the scene entity defined in the term configuration does not exist.
            AttributeError: If the term function is not callable.
            ValueError: If the term function's arguments are not matched by the parameters.
        """
        """解析 manager term 配置的通用属性。

        该方法通常由 :meth:`_prepare_terms` 调用，主要完成：

        * 解析项函数，并检查其是否可调用；
        * 校验项函数的参数是否与配置中的 ``params`` 匹配；
        * 解析 ``asset_cfg``、``sensor_cfg`` 等场景实体配置；
        * 如果项由类实现，则实例化该项。

        后两步依赖已经初始化的仿真场景，因此只能在仿真开始播放后完成。

        默认情况下，项函数至少接收环境对象这一个参数。某些管理器还要求额外的固定参数，例如第二个参数
        ``env_ids``。此时可通过 ``min_argc`` 指定管理器调用该函数所需的最少固定参数数量。

        参数：
            term_name: 项名称。
            term_cfg: 项配置。
            min_argc: 管理器正确调用项函数所需的最少固定参数数量。

        异常：
            TypeError: 项配置不是 :class:`ManagerTermBaseCfg` 类型。
            ValueError: 项配置中声明的场景实体不存在。
            AttributeError: 项函数不可调用。
            ValueError: 项函数参数与配置参数不匹配。
        """
        # check if the term is a valid term config    # 第1步: 类型校验        → term_cfg 必须是 ManagerTermBaseCfg 或其子类
        if not isinstance(term_cfg, ManagerTermBaseCfg):
            raise TypeError(
                f"Configuration for the term '{term_name}' is not of type ManagerTermBaseCfg."
                f" Received: '{type(term_cfg)}'."
            )

        # get the corresponding function or functional class
        if isinstance(term_cfg.func, str):
            term_cfg.func = string_to_callable(term_cfg.func)
            '''
            第2步: 字符串→函数      → 把 "module:func" 转成真正的 Python 函数
            如果你的配置中用字符串指定了函数路径，这里会把它转成真正的 Python 函数：
                # 配置中这样写（字符串）：
                func="isaaclab_tasks.xxx.mdp.rewards:is_alive"
                #           ↓ string_to_callable
                # 变成真正的 Python 函数对象
                func=<function is_alive at 0x7f...>
            这让你可以在 YAML 配置文件中用字符串引用函数，而不需要 import Python 模块。
            '''
        # check if function is callable
        if not callable(term_cfg.func):
            raise AttributeError(f"The term '{term_name}' is not callable. Received: {term_cfg.func}")
            '''
            第3步: 可调用性检查     → 是不是真的能 func(env, ...) 调用？
                callable() 是 Python 内置函数，检查一个对象能不能像函数一样被调用。如果传了个数字或字符串，这里直接报错。
            '''

        # check if the term is a class of valid type    # 第4步: 类模式特殊处理   → 如果是类（非实例），检查是否继承 ManagerTermBase
        if inspect.isclass(term_cfg.func):
            if not issubclass(term_cfg.func, ManagerTermBase):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ManagerTermBase."
                    f" Received: '{type(term_cfg.func)}'."
                )
            func_static = term_cfg.func.__call__    # ← 取类的 __call__ 方法做签名检查
            min_argc += 1  # forward by 1 to account for 'self' argument    # ← +1 因为实例方法多一个 self 参数
        else:
            func_static = term_cfg.func
        # check if function is callable
        if not callable(func_static):
            raise AttributeError(f"The term '{term_name}' is not callable. Received: {term_cfg.func}")

        # check statically if the term's arguments are matched by params    第5步: 参数签名匹配     → func 的参数和 params 的关键字是否对得上？
        term_params = list(term_cfg.params.keys())
        args = inspect.signature(func_static).parameters
        args_with_defaults = [arg for arg in args if args[arg].default is not inspect.Parameter.empty]
        args_without_defaults = [arg for arg in args if args[arg].default is inspect.Parameter.empty]
        args = args_without_defaults + args_with_defaults
        # ignore first two arguments for env and env_ids
        # Think: Check for cases when kwargs are set inside the function?
        if len(args) > min_argc:
            if set(args[min_argc:]) != set(term_params + args_with_defaults):
                raise ValueError(
                    f"The term '{term_name}' expects mandatory parameters: {args_without_defaults[min_argc:]}"
                    f" and optional parameters: {args_with_defaults}, but received: {term_params}."
                )

        # process attributes at runtime     第6步: 运行时解析       → 如果仿真已启动，立即解析 SceneEntityCfg
        # these properties are only resolvable once the simulation starts playing
        if self._env.sim.is_playing():
            self._process_term_cfg_at_play(term_name, term_cfg)

    def _process_term_cfg_at_play(self, term_name: str, term_cfg: ManagerTermBaseCfg):
        """Process the term configuration at runtime.

        This function is called when the simulation starts playing. It is used to process the term
        configuration at runtime. This includes:

        * Resolving the scene entity configuration for the term.
        * Initializing the term if it is a class.

        Since the above steps rely on PhysX to parse over the simulation scene, they are deferred
        until the simulation starts playing.

        Args:
            term_name: The name of the term.
            term_cfg: The term configuration.
        """
        """在仿真运行时处理项配置。

        该函数在仿真开始播放时调用，主要完成：

        * 解析该项引用的场景实体配置；
        * 如果项由类实现，则实例化该项。

        这些步骤依赖 PhysX 对仿真场景的解析结果，因此会延迟到仿真开始播放后执行。

        参数：
            term_name: 项名称。
            term_cfg: 项配置。
        """
        for key, value in term_cfg.params.items():  # 阶段 1: 扫描 params 中的 SceneEntityCfg → 解析为具体索引
            if isinstance(value, SceneEntityCfg):
                # load the entity
                try:
                    value.resolve(self._env.scene)  # ← 问 PhysX：这个关节名字对应第几个索引？
                except ValueError as e:
                    raise ValueError(f"Error while parsing '{term_name}:{key}'. {e}")
                # log the entity for checking later
                msg = f"[{term_cfg.__class__.__name__}:{term_name}] Found entity '{value.name}'."
                if value.joint_ids is not None:
                    msg += f"\n\tJoint names: {value.joint_names} [{value.joint_ids}]"
                if value.body_ids is not None:
                    msg += f"\n\tBody names: {value.body_names} [{value.body_ids}]"
                # print the information
                logger.info(msg)
                '''
                这让你在终端启动时看到类似这样的输出，确认关节名解析正确：
                    [EventTermCfg:reset_cart_position] Found entity 'robot'.
                        Joint names: ['slider_to_cart'] [0]
                    [EventTermCfg:reset_pole_position] Found entity 'robot'.
                        Joint names: ['cart_to_pole'] [1]
                '''
            # store the entity
            term_cfg.params[key] = value    # ← 把解析结果写回
        '''
        一个具体例子：
            # 你在配置中写的：
            params = {
                "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])
            }
            # 此时 joint_ids = None，因为 PhysX 还没加载

            # _process_term_cfg_at_play 执行后：
            # value.resolve(scene) 执行：
            #   1. 在 scene 中找到 Articulation 对象 "robot"
            #   2. 在 robot 的关节列表中找到 "slider_to_cart" → 索引 0
            #   3. SceneEntityCfg.joint_ids = [0]
            # 此时 joint_ids = [0]！
        '''

        # initialize the term if it is a class
        if inspect.isclass(term_cfg.func):  # 阶段 2: 如果 func 是类 → 实例化
            logger.info(f"Initializing term '{term_name}' with class '{term_cfg.func.__name__}'.")
            term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)
            '''
            为什么函数不需要实例化？ 因为函数本身就能被调用：func(env, **params)。
            但类不能直接被调用——你需要先 obj = MyClass(...) 然后 obj(env, **params)。这一步做的就是把类变成实例。
            '''
