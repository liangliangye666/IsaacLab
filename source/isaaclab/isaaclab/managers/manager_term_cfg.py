# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations
"""不同管理器使用的配置项定义。"""

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab.utils.noise import NoiseCfg, NoiseModelCfg

from .scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from .action_manager import ActionTerm
    from .command_manager import CommandTerm
    from .manager_base import ManagerTermBase
    from .recorder_manager import RecorderTerm


@configclass
class ManagerTermBaseCfg:
    """Configuration for a manager term."""
    """管理器项的基础配置。"""

    func: Callable | ManagerTermBase = MISSING
    """The function or class to be called for the term.

    The function must take the environment object as the first argument.
    The remaining arguments are specified in the :attr:`params` attribute.

    It also supports `callable classes`_, i.e. classes that implement the :meth:`__call__`
    method. In this case, the class should inherit from the :class:`ManagerTermBase` class
    and implement the required methods.

    .. _`callable classes`: https://docs.python.org/3/reference/datamodel.html#object.__call__
    """
    """该项调用的函数或类。

    函数必须将环境对象作为第一个参数，其余参数通过 :attr:`params` 指定。

    也支持实现 :meth:`__call__` 的 `callable classes`_。此时，该类应继承
    :class:`ManagerTermBase` 并实现所需方法。

    .. _`callable classes`: https://docs.python.org/3/reference/datamodel.html#object.__call__
    """
    '''
    类型	            含义	                            例子
    Callable	        一个普通 Python 函数	            def my_reward(env): ...
    ManagerTermBase	    一个实现了 __call__ 方法的类实例	  class MyTerm(ManagerTermBase): def __call__(self, env): ...
    什么是 Callable？ 
        在 Python 中，函数、lambda、实现了 __call__ 的类的实例都是"可调用对象"（Callable）。
        Callable 就是"任何可以像函数一样被调用的东西"的统称。
    通俗理解：
        这个字段可以填一个函数，也可以填一个"能当函数用的类对象"。
        两者最终使用方式一样（都是 func(env, **params)），只是来源不同。
    '''

    params: dict[str, Any | SceneEntityCfg] = dict()
    """The parameters to be passed to the function as keyword arguments. Defaults to an empty dict.

    .. note::
        If the value is a :class:`SceneEntityCfg` object, the manager will query the scene entity
        from the :class:`InteractiveScene` and process the entity's joints and bodies as specified
        in the :class:`SceneEntityCfg` object.
    """
    """以关键字参数形式传递给函数的参数。默认为空字典。

    .. 说明::
        若参数值是 :class:`SceneEntityCfg`，管理器会从 :class:`InteractiveScene` 中查询对应场景实体，
        并按 :class:`SceneEntityCfg` 的定义解析其关节和刚体。
    """
    '''
    部分	                        含义
    dict	                        字典类型
    [str, ...]	                    键（key）是字符串
    [..., Any | SceneEntityCfg]	    值（value）可以是任意类型，或者是一个 SceneEntityCfg 对象
    = dict()	                    默认值是一个空字典
        Any 是 Python 类型系统中的"万能类型"——表示"什么都行"，不做类型检查。
        这里用 Any | SceneEntityCfg 表示：大部分参数值是普通数值（被归为 Any），但如果是 SceneEntityCfg 类型，框架会特殊处理——自动查询场景中的对应实体。
    dict() 等价于 {}，都是创建一个空字典。
    '''


##
# Recorder manager.
##


@configclass
class RecorderTermCfg:
    """Configuration for an recorder term."""
    """记录器项的配置。"""

    class_type: type[RecorderTerm] = MISSING
    """The associated recorder term class.

    The class should inherit from :class:`isaaclab.managers.recorder_manager.RecorderTerm`.
    """
    """关联的记录器项类。

    该类应继承 :class:`isaaclab.managers.recorder_manager.RecorderTerm`。
    """
    '''
    type[X] 是 Python 的类型注解语法，读作"X 这个类本身"，而不是"X 的实例"。
        写法	        含义	                举例
        x: Dog	        x 是 Dog 的一个实例	    x = Dog(name="旺财")
        x: type[Dog]	x 是 Dog 这个类本身	    x = Dog（没有括号！）

    对比一下：
        配置类	                核心字段	     存的是什么	    框架怎么用
        ManagerTermBaseCfg	    func	        一个函数	func(env, **params) — 直接调用
        RecorderTermCfg	        class_type	    一个类	    class_type(cfg, env) — 先实例化，再调用
        ActionTermCfg	        class_type	    一个类	    class_type(cfg, env) — 先实例化，再调用
    class_type 的值必须是 RecorderTerm 的子类（或其本身）

    为什么 RecorderTermCfg 和 ActionTermCfg 不用 func + params 模式？
        因为 Recorder 和 Action 的逻辑更复杂，不是一个简单函数能搞定的——它们需要维护内部状态（如 IK 求解器的历史数据、录制缓冲等），所以用类而非函数。
        class_type 就是告诉框架"用哪个类来创建处理对象"。
    '''


##
# Action manager.
##


@configclass
class ActionTermCfg:
    """Configuration for an action term."""
    """动作项的配置。"""

    class_type: type[ActionTerm] = MISSING
    """The associated action term class.

    The class should inherit from :class:`isaaclab.managers.action_manager.ActionTerm`.
    """
    """关联的动作项类。

    该类应继承 :class:`isaaclab.managers.action_manager.ActionTerm`。
    """

    asset_name: str = MISSING
    """The name of the scene entity.

    This is the name defined in the scene configuration file. See the :class:`InteractiveSceneCfg`
    class for more details.
    """
    """场景实体名称。

    该名称在场景配置中定义，详见 :class:`InteractiveSceneCfg`。
    """

    '''
    父类字段	     含义	            示例
    class_type	    指向实现类	        joint_actions.JointPositionAction       谁继承它就指向谁
    asset_name	    场景中机器人名称	"robot"
    '''

    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""
    """是否显示调试可视化信息。默认为 False。"""

    clip: dict[str, tuple] | None = None
    """Clip range for the action (dict of regex expressions). Defaults to None."""
    """动作的裁剪范围，以正则表达式到范围的字典表示。默认为 None。"""
    '''
    dict[str, tuple] 表示："键是字符串，值是元组"的字典。
    关键：
        tuple 后面没写具体类型，表示"任何元组都行"——不限制元组的长度和元素类型。
        如果写成 dict[str, tuple[float, float]] 的话，那就限制了必须是两个浮点数的元组。
    '''


##
# Command manager.
##


@configclass
class CommandTermCfg:
    """Configuration for a command generator term."""
    """命令生成器项的配置。"""

    class_type: type[CommandTerm] = MISSING
    """The associated command term class to use.

    The class should inherit from :class:`isaaclab.managers.command_manager.CommandTerm`.
    """
    """关联的命令项类。

    该类应继承 :class:`isaaclab.managers.command_manager.CommandTerm`。
    """

    resampling_time_range: tuple[float, float] = MISSING
    """Time before commands are changed [s]."""
    """命令重新采样前的时间范围，单位为秒。"""
    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""
    """是否显示调试可视化信息。默认为 False。"""


##
# Curriculum manager.
##


@configclass
class CurriculumTermCfg(ManagerTermBaseCfg):
    """Configuration for a curriculum term."""
    """课程项（curriculum term）的配置。"""

    func: Callable[..., float | dict[str, float] | None] = MISSING
    """The name of the function to be called.

    This function should take the environment object, environment indices
    and any other parameters as input and return the curriculum state for
    logging purposes. If the function returns None, the curriculum state
    is not logged.
    """
    """要调用的函数。

    该函数接收环境对象、环境索引及其他参数，并返回用于日志记录的课程状态。
    若返回 None，则不记录课程状态。
    """
    '''
    规则：Callable 里必须有一个"返回类型"，且它总是在最后
        Python 规定 Callable 的写法只有两种：
        # 形式一：不关心输入参数
        Callable[..., 返回值类型]
                │     │
                └─── 固定写法，三个点，表示"任何参数都行"
                    └── 最后一个位置 = 返回类型（永远在这里）

        # 形式二：指定输入参数类型
        Callable[[参数1类型, 参数2类型], 返回值类型]
                └───────┬──────────┘     │
                    输入参数（内层方括号）   └── 最后一个逗号后的类型 = 返回类型
        "最后一个"就是返回值——这是 Callable 泛型的语法规定，不是推导出来的。
        总结：
            Callable[[A, B], C] 中，嵌套在里层方括号里的是输入，最后一个逗号后面的是返回值。
            Callable[..., C] 是说"输入无所谓，返回是 C"。
            这是 Python typing 模块的硬规定，就像 dict[str, int] 的键在前、值在后一样，位置决定含义。

    Callable[  ...,  float | dict[str, float] | None  ]
    │       │                   │
    │       │                   └── 返回类型（三种可能）
    │       └── 参数：... 表示"什么参数都行，数量不限"
    └── 这是一个可调用对象（函数）

    '''


##
# Observation manager.
##


@configclass
class ObservationTermCfg(ManagerTermBaseCfg):
    """Configuration for an observation term."""
    """观测项的配置。"""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the observation signal as torch float tensors of
    shape (num_envs, obs_term_dim).
    """
    """要调用的函数。

    该函数接收环境对象及其他参数，并返回形状为 ``(num_envs, obs_term_dim)`` 的
    torch 浮点观测张量。
    """

    modifiers: list[ModifierCfg] | None = None
    """The list of data modifiers to apply to the observation in order. Defaults to None,
    in which case no modifications will be applied.

    Modifiers are applied in the order they are specified in the list. They can be stateless
    or stateful, and can be used to apply transformations to the observation data. For example,
    a modifier can be used to normalize the observation data or to apply a rolling average.

    For more information on modifiers, see the :class:`~isaaclab.utils.modifiers.ModifierCfg` class.
    """
    """按顺序应用于观测的数据修改器列表。默认为 None，表示不应用修改器。

    修改器可以是无状态或有状态，用于对观测数据执行变换，例如归一化或滑动平均。
    更多信息请参阅 :class:`~isaaclab.utils.modifiers.ModifierCfg`。
    """
    '''
    modifiers: list[ModifierCfg] | None = None
    │         │         │      │       │
    │         │         │      │       └── 默认值：None（不应用修改器）
    │         │         │      └── 也可以是 None
    │         │         └── ModifierCfg 对象
    │         └── list 里装的是 ...
    └── 字段名
    '''

    noise: NoiseCfg | NoiseModelCfg | None = None
    """The noise to add to the observation. Defaults to None, in which case no noise is added."""
    """添加到观测中的噪声配置。默认为 None，表示不添加噪声。"""

    clip: tuple[float, float] | None = None
    """The clipping range for the observation after adding noise. Defaults to None,
    in which case no clipping is applied."""
    """添加噪声后对观测执行裁剪的范围。默认为 None，表示不裁剪。"""

    scale: tuple[float, ...] | float | None = None
    """The scale to apply to the observation after clipping. Defaults to None,
    in which case no scaling is applied (same as setting scale to :obj:`1`).

    We leverage PyTorch broadcasting to scale the observation tensor with the provided value. If a tuple is provided,
    please make sure the length of the tuple matches the dimensions of the tensor outputted from the term.
    """
    """裁剪后应用于观测的缩放系数。

    默认为 None，等价于缩放系数为 :obj:`1`。缩放使用 PyTorch broadcasting；
    若提供元组，其长度必须与该观测项输出张量的对应维度匹配。
    """
    '''
    核心语法：tuple[float, ...] — 可变长度元组
        tuple[float, ...]
        │     │      │
        │     │      └── ... 表示"里面有多少个 float 都行"
        │     └── 元素类型是 float
        └── 元组类型
    在 tuple[X, ...] 中的含义：和 Callable[..., X] 中的 ... 完全不同！这里是表示"元组长度不固定，里面全是 float"。

    值	                    含义	                        举例
    tuple[float, ...]	    每个观测维度用不同的缩放系数	    (1.0, 2.0, 0.5) → 维度0×1.0, 维度1×2.0, 维度2×0.5
    float	                所有维度用同一个缩放系数	        2.0 → 所有维度都 ×2.0
    None	                不缩放（等价于 ×1.0）	            默认值
    '''

    history_length: int = 0
    """Number of past observations to store in the observation buffers. Defaults to 0, meaning no history.

    Observation history initializes to empty, but is filled with the first append after reset or initialization.
    Subsequent history only adds a single entry to the history buffer. If flatten_history_dim is set to True,
    the source data of shape (N, H, D, ...) where N is the batch dimension and H is the history length will
    be reshaped to a 2-D tensor of shape (N, H*D*...). Otherwise, the data will be returned as is.
    """
    """观测缓冲区中保存的历史观测数量。默认为 0，表示不保存历史。

    历史缓冲区初始为空，并在 reset 或初始化后的第一次 append 时填充。
    后续每次仅追加一个条目。若 ``flatten_history_dim`` 为 True，形状为 ``(N, H, D, ...)``
    的数据会重塑为 ``(N, H*D*...)``；否则保持原形状。
    """

    flatten_history_dim: bool = True
    """Whether or not the observation manager should flatten history-based observation terms to a 2-D (N, D) tensor.
    Defaults to True."""
    """是否将基于历史的观测项展平为二维 ``(N, D)`` 张量。默认为 True。"""


@configclass
class ObservationGroupCfg:
    """Configuration for an observation group."""
    """观测组的配置。"""

    concatenate_terms: bool = True
    """Whether to concatenate the observation terms in the group. Defaults to True.

    If true, the observation terms in the group are concatenated along the dimension specified through
    :attr:`concatenate_dim`. Otherwise, they are kept separate and returned as a dictionary.

    If the observation group contains terms of different dimensions, it must be set to False.
    """
    """是否拼接该组中的观测项。默认为 True。

    若为 True，各观测项沿 :attr:`concatenate_dim` 指定的维度拼接；否则保持分离并以字典返回。
    如果组内观测项的维数不同，必须将该值设为 False。
    """

    concatenate_dim: int = -1
    """Dimension along to concatenate the different observation terms. Defaults to -1, which
    means the last dimension of the observation terms.

    If :attr:`concatenate_terms` is True, this parameter specifies the dimension along which the observation
    terms are concatenated. The indicated dimension depends on the shape of the observations. For instance,
    for a 2-D RGB image of shape (H, W, C), the dimension 0 means concatenating along the height, 1 along the
    width, and 2 along the channels. The offset due to the batched environment is handled automatically.
    """
    """拼接不同观测项时使用的维度。默认为 -1，即观测项的最后一个维度。

    当 :attr:`concatenate_terms` 为 True 时，该参数指定拼接维度。例如，对于形状为 ``(H, W, C)``
    的二维 RGB 图像，0、1、2 分别表示沿高度、宽度和通道维度拼接。批量环境引入的维度偏移会自动处理。
    """

    enable_corruption: bool = False
    """Whether to enable corruption for the observation group. Defaults to False.

    If true, the observation terms in the group are corrupted by adding noise (if specified).
    Otherwise, no corruption is applied.
    """
    """是否为该观测组启用扰动。默认为 False。

    若为 True，则按各项配置向观测添加噪声；否则不施加扰动。
    """

    history_length: int | None = None
    """Number of past observation to store in the observation buffers for all observation terms in group.

    This parameter will override :attr:`ObservationTermCfg.history_length` if set. Defaults to None.
    If None, each terms history will be controlled on a per term basis. See :class:`ObservationTermCfg`
    for details on :attr:`ObservationTermCfg.history_length` implementation.
    """
    """应用于组内所有观测项的历史长度。

    设置后会覆盖 :attr:`ObservationTermCfg.history_length`。默认为 None，表示各观测项分别控制自己的
    历史长度。具体行为请参阅 :class:`ObservationTermCfg`。
    """

    flatten_history_dim: bool = True
    """Flag to flatten history-based observation terms to a 2-D (num_env, D) tensor for all observation terms in group.
    Defaults to True.

    This parameter will override all :attr:`ObservationTermCfg.flatten_history_dim` in the group if
    ObservationGroupCfg.history_length is set.
    """
    """是否将组内基于历史的观测项展平为二维 ``(num_env, D)`` 张量。默认为 True。

    当设置 ``ObservationGroupCfg.history_length`` 时，该参数会覆盖组内所有
    :attr:`ObservationTermCfg.flatten_history_dim` 设置。
    """


##
# Event manager
##


@configclass
class EventTermCfg(ManagerTermBaseCfg):
    """Configuration for a event term."""
    """事件项的配置。"""

    func: Callable[..., None] = MISSING
    """The name of the function to be called.

    This function should take the environment object, environment indices
    and any other parameters as input.
    """
    """要调用的函数。

    该函数接收环境对象、环境索引及其他参数。
    """

    mode: str = MISSING
    """The mode in which the event term is applied.

    Note:
        The mode name ``"interval"`` is a special mode that is handled by the
        manager Hence, its name is reserved and cannot be used for other modes.
    """
    """事件项的执行模式。

    说明：
        ``"interval"`` 是由事件管理器专门处理的保留模式名称，不能用于其他模式。
    """

    interval_range_s: tuple[float, float] | None = None
    """The range of time in seconds at which the term is applied. Defaults to None.

    Based on this, the interval is sampled uniformly between the specified
    range for each environment instance. The term is applied on the environment
    instances where the current time hits the interval time.

    Note:
        This is only used if the mode is ``"interval"``.
    """
    """执行该项的时间间隔采样范围，单位为秒。默认为 None。

    每个环境实例会在给定范围内均匀采样自己的间隔时间，并在当前时间达到该间隔时执行该项。

    说明：
        只有在 ``"interval"`` 模式下才使用。
    """

    is_global_time: bool = False
    """Whether randomization should be tracked on a per-environment basis. Defaults to False.

    If True, the same interval time is used for all the environment instances.
    If False, the interval time is sampled independently for each environment instance
    and the term is applied when the current time hits the interval time for that instance.

    Note:
        This is only used if the mode is ``"interval"``.
    """
    """是否对所有环境使用全局统一的间隔时间。默认为 False。

    如果 True，则对所有环境实例使用相同的间隔时间。
    如果 False，则为每个环境实例独立采样间隔，并在该实例达到对应间隔时执行该项。

    说明：
        只有在 ``"interval"`` 模式下才使用。
    """

    min_step_count_between_reset: int = 0
    """The number of environment steps after which the term is applied since its last application. Defaults to 0.

    When the mode is "reset", the term is only applied if the number of environment steps since
    its last application exceeds this quantity. This helps to avoid calling the term too often,
    thereby improving performance.

    If the value is zero, the term is applied on every call to the manager with the mode "reset".

    Note:
        This is only used if the mode is ``"reset"``.
    """
    """两次 reset 模式执行之间要求的最小环境步数。默认为 0。

    当模式为 ``"reset"`` 时，只有自上次执行以来的环境步数达到该值才会再次执行此项，
    从而避免过于频繁的调用。若为 0，则每次以 ``"reset"`` 模式调用事件管理器时都执行该项。

    说明：
        只有在 ``"reset"`` 模式下才使用。
    """


##
# Reward manager.
##


@configclass
class RewardTermCfg(ManagerTermBaseCfg):
    """Configuration for a reward term."""
    """奖励项的配置。"""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the reward signals as torch float tensors of
    shape (num_envs,).
    """
    """要调用的函数。

    该函数接收环境对象及其他参数，并返回形状为 ``(num_envs,)`` 的 torch 浮点奖励张量。
    """

    weight: float = MISSING
    """The weight of the reward term.

    This is multiplied with the reward term's value to compute the final
    reward.

    Note:
        If the weight is zero, the reward term is ignored.
    """
    """奖励项权重。

    最终奖励通过该权重乘以奖励项的原始值计算。

    说明：
        权重为 0 时忽略该奖励项。
    """


##
# Termination manager.
##


@configclass
class TerminationTermCfg(ManagerTermBaseCfg):
    """Configuration for a termination term."""
    """终止项的配置。"""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the termination signals as torch boolean tensors of
    shape (num_envs,).
    """
    """要调用的函数。

    该函数接收环境对象及其他参数，并返回形状为 ``(num_envs,)`` 的 torch 布尔终止张量。
    """

    time_out: bool = False
    """Whether the termination term contributes towards episodic timeouts. Defaults to False.

    Note:
        These usually correspond to tasks that have a fixed time limit.
    """
    """该终止项是否表示回合超时。默认为 False。

    说明：
        此类终止项通常对应具有固定时间上限的任务。
    """
