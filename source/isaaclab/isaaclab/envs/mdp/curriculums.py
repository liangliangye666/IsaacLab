# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""
"""可用于为学习环境创建课程的共同功能。

函数可以传递到:class:`isaaclab.managers.CurriculumTermCfg`对象，使函数引入的课程能够实现。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

'''
curriculums.py 定义了三种课程（Curriculum）策略：
    ManagerTermBase
        ├── modify_reward_weight       ← 【本类】动态调整奖励项权重
        ├── modify_env_param           ← 动态修改环境的任意属性（如摩擦力）
        └── modify_term_cfg            ← 继承 modify_env_param，修改管理器项的配置
**课程（Curriculum）**在强化学习中是什么？
    训练初期任务简单（如低速行走），训练后期任务变难（如高速奔跑）。
    课程学习通过逐渐改变环境参数或奖励权重来实现这种"由易到难"的过渡。
'''


class modify_reward_weight(ManagerTermBase):
    """Curriculum that modifies the reward weight based on a step-wise schedule."""
    """课程，根据步骤的时间表修改奖励权重。"""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # obtain term configuration
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)
        '''
        根据配置中指定的 term_name，从 RewardManager 中找到对应的奖励项配置对象，拿到它的引用。
        之后 __call__ 方法就可以在训练过程中动态修改这个奖励项的权重（weight）。
        '''

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        weight: float,
        num_steps: int,
    ) -> float:
        # update term settings
        if env.common_step_counter > num_steps:
            self._term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)

        return self._term_cfg.weight
    '''
    __init__ 和 __call__ 的分工：
        __init__（初始化时执行一次）:
            从配置中读 term_name = "track_lin_vel_xy_exp"
            找到对应的 RewardTermCfg 对象
            存为 self._term_cfg

            → 这是一个"查找并缓存引用"的操作

        __call__（每步调用）:
            检查训练步数是否超过阈值 num_steps
            如果超过 → 修改 self._term_cfg.weight = weight（新权重）
            把修改后的配置写回 RewardManager
            返回当前权重

            → 这是一个"定时修改"的操作
    例子：
        # 配置文件中:
            curriculum_terms = {
                "boost_speed": CurriculumTermCfg(
                    func=mdp.modify_reward_weight,
                    params={
                        "term_name": "track_lin_vel_xy_exp",  # ← __init__ 用这个找奖励项
                        "weight": 2.0,                         # ← __call__ 用这个设新权重
                        "num_steps": 50000,                    # ← __call__ 用这个判断时机
                    },
                ),
            }
    '''

'''
是三种课程中最通用的一种——它不关心你要改什么，只提供一个"根据路径读-调用修改函数-写回"的通用包装器。
'''
class modify_env_param(ManagerTermBase):
    """Curriculum term for modifying an environment parameter at runtime.

    This term helps modify an environment parameter (or attribute) at runtime.
    This parameter can be any attribute of the environment, such as the physics material properties,
    observation ranges, or any other configurable parameter that can be accessed via a dotted path.

    The term uses the ``address`` parameter to specify the target attribute as a dotted path string.
    For instance, "event_manager.cfg.object_physics_material.func.material_buckets" would
    refer to the attribute ``material_buckets`` in the event manager's event term "object_physics_material",
    which is a tensor of sampled physics material properties.

    The term uses the ``modify_fn`` parameter to specify the function that modifies the value of the target attribute.
    The function should have the signature:

    .. code-block:: python

        def modify_fn(env, env_ids, old_value, **modify_params) -> new_value | modify_env_param.NO_CHANGE:
            # modify the value based on the old value and the modify parameters
            new_value = old_value + modify_params["value"]
            return new_value

    where ``env`` is the learning environment, ``env_ids`` are the sub-environment indices,
    ``old_value`` is the current value of the target attribute, and ``modify_params``
    are additional parameters that can be passed to the function. The function should return
    the new value to be set for the target attribute, or the special token ``modify_env_param.NO_CHANGE``
    to indicate that the value should not be changed.

    At the first call to the term after initialization, it compiles getter and setter functions
    for the target attribute specified by the ``address`` parameter. The getter retrieves the
    current value, and the setter writes a new value back to the attribute.

    This term processes getter/setter accessors for a target attribute in an(specified by
    as an "address" in the term configuration :attr:`cfg.params["address"]`) the first time it is called,
    then on each invocation reads the current value, applies a user-provided :attr:`modify_fn`,
    and writes back the result. Since :obj:`None` in this case can sometime be desirable value
    to write, we use token, :attr:`NO_CHANGE`, as non-modification signal to this class, see usage below.

    Usage:
        .. code-block:: python

            def resample_bucket_range(
                env, env_id, data, static_friction_range, dynamic_friction_range, restitution_range, num_steps
            ):
                if env.common_step_counter > num_steps:
                    range_list = [static_friction_range, dynamic_friction_range, restitution_range]
                    ranges = torch.tensor(range_list, device="cpu")
                    new_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(data), 3), device="cpu")
                    return new_buckets

                # if the step counter is not reached, return NO_CHANGE to indicate no modification.
                # we do this instead of returning None, since None is a valid value to set.
                # additionally, returning the input data would not change the value but still lead
                # to the setter being called, which may add overhead.
                return mdp.modify_env_param.NO_CHANGE


            object_physics_material_curriculum = CurrTerm(
                func=mdp.modify_env_param,
                params={
                    "address": "event_manager.cfg.object_physics_material.func.material_buckets",
                    "modify_fn": resample_bucket_range,
                    "modify_params": {
                        "static_friction_range": [0.5, 1.0],
                        "dynamic_friction_range": [0.3, 1.0],
                        "restitution_range": [0.0, 0.5],
                        "num_step": 120000,
                    },
                },
            )
    """
    """在运行时间修改环境参数的课程项。

    这个项有助于在运行时修改环境参数 (或属性)。
    这个参数可以是环境的任何属性，例如物理材料属性，观测范围或可以通过点路径访问的任何其他可配置参数。

    该项使用``address``参数来指定目标属性为点路线字符串。
    例如"，event_manager.cfg.object_physics_material.func.material_buckets"在事件管理器的事件项"object_physics_mater
    ial"中的``material_buckets``属性，这是样本物理材料属性的数。

    该项使用``modify_fn``参数来指定修改目标属性的值函数。
    函数应具有以下签名:

    .. code-block:: python

        def modify_fn(env, env_ids, old_value, **modify_params) -> new_value | modify_env_param.NO_CHANGE:
            # modify the value based on the old value and the modify parameters
            new_value = old_value + modify_params["value"]
            return new_value

    在 ``env`` 是学习环境， ``env_ids`` 是子环境索引， ``old_value`` 是目标属性的当前值， ``modify_params`` 是可传递到函数的额外参数。
    函数应返回对目标属性设置的新值，或表示值不应改变的特殊代币``modify_env_param.NO_CHANGE``。

    在初始化后，它编译了getter和setter函数
    for the target attribute specified by the ``address`` parameter. The getter retrieves the
    设置器将新值返回属性。

    这一项处理一个目标属性的getter/setter accessors 在一个 ((指定为一个"地址"在项配置:attr:`cfg.params["address"]`)
    上第一次调用它，然后在每个调用时读取当前值，应用用户提供的:attr:`modify_fn`，并写回结果。
    由于:obj:`None`在这种情况下有时可以是值值值，我们使用代币，:attr:`NO_CHANGE`，作为这个类的非修改信号。

    Usage:
        .. code-block:: python

            def resample_bucket_range(
                env, env_id, data, static_friction_range, dynamic_friction_range, restitution_range, num_steps
            ):
                if env.common_step_counter > num_steps:
                    range_list = [static_friction_range, dynamic_friction_range, restitution_range]
                    ranges = torch.tensor(range_list, device="cpu")
                    new_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(data), 3), device="cpu")
                    return new_buckets

                # if the step counter is not reached, return NO_CHANGE to indicate no modification.
                # we do this instead of returning None, since None is a valid value to set.
                # additionally, returning the input data would not change the value but still lead
                # to the setter being called, which may add overhead.
                return mdp.modify_env_param.NO_CHANGE


            object_physics_material_curriculum = CurrTerm(
                func=mdp.modify_env_param,
                params={
                    "address": "event_manager.cfg.object_physics_material.func.material_buckets",
                    "modify_fn": resample_bucket_range,
                    "modify_params": {
                        "static_friction_range": [0.5, 1.0],
                        "dynamic_friction_range": [0.3, 1.0],
                        "restitution_range": [0.0, 0.5],
                        "num_step": 120000,
                    },
                },
            )
    """

    NO_CHANGE: ClassVar = object()
    """Special token to indicate no change in the value to be set.

    This token is used to signal that the `modify_fn` did not produce a new value. It can
    be returned by the `modify_fn` to indicate that the current value should remain unchanged.
    """
    """标志着未需设置的值的变化。

    这种代币用于表示`modify_fn`没有产生新的值。
    `modify_fn`可以返回它，表示当前值不变。
    """
    '''
    哨兵值
        object() 创建一个裸 Python 对象实例——没有任何属性、没有任何方法、唯一的特点是"存在且唯一"。
        这个对象被用作哨兵（Sentinel）——一个特殊的标记，区别于所有正常的数据值。

        为什么选择 object() 而不是 None？
            来看使用场景：
                def modify_fn(env, env_ids, data, **params):
                    if step_counter > num_steps:
                        return new_value      # 正常修改 → 返回新值
                    return modify_env_param.NO_CHANGE  # 不修改 → 返回哨兵
            如果哨兵用 None：
                return None   # ← 歧义！是"不修改"还是"把值设为 None"？
            有些环境属性的合法值就是 None（比如关闭某个功能），这时无法区分"故意设为 None"和"不修改"。
            object() 创建的是一个独一无二的新对象——世界上只有这一个实例，return modify_env_param.NO_CHANGE 和 return None 在 __call__ 的 is 判断中可以明确区分：
                    if new_val is not self.NO_CHANGE:
                        self._set_fn(new_val)
            is 是身份比较（同一块内存），NO_CHANGE 作为 object() 的唯一实例，不存在任何值会和它"is 相同"。

    ClassVar 类型注解：
        表示这是类级别的变量，所有实例共享同一个值。它本质是文档标注——不影响运行时行为，但 IDE 和类型检查器（如 mypy）会据此判断访问方式。

    通俗类比：
        NO_CHANGE 就像快递员按门铃时你挂出的"请勿打扰"牌子——它不是快递，而是一个信号，告诉快递员"今天别送，原路返回"。
        如果用 None 做这个信号，那"空包裹"和"请勿打扰"就无法区分了。
    '''

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # resolve term configuration
        if "address" not in cfg.params: # cfg.params 字典中必须包含 "address" 键
            raise ValueError("The 'address' parameter must be specified in the curriculum term configuration.")

        # store current address
        self._address: str = cfg.params["address"]
        '''
        address 是一个点号分隔的路径字符串，例如：
            "event_manager.cfg.object_physics_material.func.material_buckets" — 指向事件管理器中的物理材质配置
            "scene.terrain.cfg.mesh_type" — 指向场景地形类型
        这个路径描述了"要修改的环境属性在哪里"。
        '''
        # store accessor functions
        self._get_fn: callable = None
        self._set_fn: callable = None
        '''
        惰性函数槽
            初始化为 None——不立即编译，等到第一次 __call__ 时才通过 _process_accessors 动态生成 getter/setter 函数。这样做的好处是：
                初始化时的环境对象可能还未完全就绪（某些属性还不存在）
                如果课程项从未被触发，就永远不需要编译（节省开销）
            这是延迟绑定——把耗时的路径解析推迟到必要时刻。
        '''

    def __del__(self):
        """Destructor to clean up the compiled functions."""
        """为了清理编译的功能。"""
        # clear the getter and setter functions
        self._get_fn = None
        self._set_fn = None
        self._container = None
        self._last_path = None

    """
    Operations.
    """
    """操作。
    """

    '''
    每步被 CurriculumManager 调用一次，实现一个通用的 读-改-写 循环：从环境配置中读取某个属性的当前值 → 调用用户提供的修改函数 → 把结果写回（如果用户不想修改则跳过）。
    这是 Isaac Lab 中最通用的课程学习机制——可以修改环境的任意属性而不需要为每种属性写专门的课程类。
    '''
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        address: str,
        modify_fn: callable,
        modify_params: dict | None = None,
    ):
        # fetch the getter and setter functions if not already compiled
        if not self._get_fn:
            self._get_fn, self._set_fn = self._process_accessors(self._env, self._address)
        '''
        第一步：惰性编译 getter/setter
            第一次调用时（_get_fn 还是 None），调用 _process_accessors 解析 self._address
            （如 "event_manager.cfg.object_physics_material.func.material_buckets"），生成一对函数：
                _get_fn() — 读取当前值（如返回当前物理材质张量）
                _set_fn(new_val) — 写入新值
            之后每次调用直接复用已编译的函数，不再做路径解析。这是编译一次，调用千次的优化策略。
            _process_accessors 内部做什么？
                把点号路径逐段解析，最终定位到目标属性所在的容器，然后生成针对该容器的 getter/setter 闭包。
        '''

        # resolve none type
        modify_params = {} if modify_params is None else modify_params
        '''
        第二步：空值安全
            如果 modify_params 是 None（没有额外参数），替换为空字典 {}。
            这样后续的 **modify_params 解包时不会报错——**{} 就是零个关键字参数。
            一行守卫避免了 modify_fn(**None) 导致的 TypeError。
        '''

        # get the current value of the target attribute
        data = self._get_fn()
        # modify the value using the provided function
        new_val = modify_fn(self._env, env_ids, data, **modify_params)
        # set the modified value back to the target attribute
        # note: if the modify_fn return NO_CHANGE signal, we do not invoke self.set_fn
        if new_val is not self.NO_CHANGE:
            self._set_fn(new_val)
        '''
        第三步：读-改-写
            ① 读: data = _get_fn()
                ↓   当前属性值
            ② 改: new_val = modify_fn(env, env_ids, data, **params)
                ↓   用户函数决定新值或 NO_CHANGE
            ③ 写: if new_val is not NO_CHANGE:
                    _set_fn(new_val)       ← 写回属性
                else:
                    什么都不做               ← 保持原值

        '''
        '''
        例子：
        典型使用场景
            # 配置文件中定义一个课程项:
            physics_curriculum = CurriculumTermCfg(
                func=mdp.modify_env_param,
                params={
                    "address": "event_manager.cfg.object_physics_material.func.material_buckets",
                    "modify_fn": harden_friction_over_time,
                    "modify_params": {"num_steps": 120000},
                },
            )

            # 用户定义的修改函数:
            def harden_friction_over_time(env, env_ids, data, num_steps):
                if env.common_step_counter > num_steps:
                    # 120000 步后，把摩擦力提高到 [0.5, 1.0]
                    ranges = torch.tensor([[0.5, 1.0], [0.5, 1.0], [0.0, 0.2]], device="cpu")
                    return math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(data), 3), device="cpu")
                return modify_env_param.NO_CHANGE  # 没到时间，不改
        训练效果：
            前 120000 步机器人在地面摩擦力较低的环境中学习行走（容易滑，但不容易摔倒）。
            120000 步后摩擦力提高，机器人必须适应更难的控制环境——这就是"由易到难"的课程学习。
        '''

    """
    Helper functions.
    """
    """辅助函数。
    """

    '''
    把一条点号路径字符串编译成一对 getter/setter 闭包。
        这是 modify_env_param 的核心引擎——通过字符串路径动态访问任意嵌套深度的环境属性，而不需要在代码中硬编码访问路径
        通俗类比： _process_accessors 就像一个"地址翻译器"。
            你告诉它一个人类可读的地址 "北京市.海淀区.中关村[3].101室"，它解析这个地址、走到对应的大楼、找到房间，然后给你一把万能钥匙
            ——get_value() 是"看一眼房间里的东西"，set_value(val) 是"把东西换成新的"。
            之后你再也不需要用地址字符串了，直接用钥匙就行。
    '''
    def _process_accessors(self, root: ManagerBasedRLEnv, path: str) -> tuple[callable, callable]:
        """Process and return the (getter, setter) functions for a dotted attribute path.

        This function resolves a dotted path string to an attribute in the given root object.
        The dotted path can include nested attributes, dictionary keys, and sequence indexing.

        For instance, the path "foo.bar[2].baz" would resolve to `root.foo.bar[2].baz`. This
        allows accessing attributes in a nested structure, such as a dictionary or a list.

        Args:
            root: The main object from which to resolve the attribute.
            path: Dotted path string to the attribute variable. For e.g., "foo.bar[2].baz".

        Returns:
            A tuple of two functions (getter, setter), where:
            the getter retrieves the current value of the attribute, and
            the setter writes a new value back to the attribute.
        """
        """处理和返回点点属性路径的 (getter， setter) 函数。

        这个函数解决了给定的根对象中的属性。
        点的路径可能包括嵌套的属性，字典键和序列索引。

        例如，路径"foo.bar[2].baz"将解决为`root.foo.bar[2].baz`。
        这允许访问嵌入式结构中的属性，例如字典或列表。

        参数：
            root: 解决属性的主要对象。
            path: 给属性变量带点路径。
                  对于e.g."，foo.bar[2].baz"。

        返回：
            两个函数 (getter， setter) 的元组，其中:getter 检索属性的当前值，而 setter 返回属性的新值。
        """
        # Turn "a.b[2].c" into ["a", ("b", 2), "c"] and store in parts
        path_parts: list[str | tuple[str, int]] = []
        for part in path.split("."):
            m = re.compile(r"^(\w+)\[(\d+)\]$").match(part)
            if m:
                path_parts.append((m.group(1), int(m.group(2))))
            else:
                path_parts.append(part)

        # Traverse the parts to find the container
        container = root
        for container_path in path_parts[:-1]:
            if isinstance(container_path, tuple):
                # we are accessing a list element
                name, idx = container_path
                # find underlying attribute
                if isinstance(container_path, dict):
                    seq = container[name]  # type: ignore[assignment]
                else:
                    seq = getattr(container, name)
                # save the container for the next iteration
                container = seq[idx]
            else:
                # we are accessing a dictionary key or an attribute
                if isinstance(container, dict):
                    container = container[container_path]
                else:
                    container = getattr(container, container_path)

        # save the container and the last part of the path
        self._container = container
        self._last_path = path_parts[-1]  # for "a.b[2].c", this is "c", while for "a.b[2]" it is 2

        # build the getter and setter
        if isinstance(self._container, tuple):
            get_value = lambda: self._container[self._last_path]  # noqa: E731

            def set_value(val):
                tuple_list = list(self._container)
                tuple_list[self._last_path] = val
                self._container = tuple(tuple_list)

        elif isinstance(self._container, (list, dict)):
            get_value = lambda: self._container[self._last_path]  # noqa: E731

            def set_value(val):
                self._container[self._last_path] = val

        elif isinstance(self._container, object):
            get_value = lambda: getattr(self._container, self._last_path)  # noqa: E731
            set_value = lambda val: setattr(self._container, self._last_path, val)  # noqa: E731
        else:
            raise TypeError(
                f"Unable to build accessors for address '{path}'. Unknown type found for access variable:"
                f" '{type(self._container)}'. Expected a list, dict, or object with attributes."
            )

        return get_value, set_value


class modify_term_cfg(modify_env_param):
    """Curriculum for modifying a manager term configuration at runtime.

    This class inherits from :class:`modify_env_param` and is specifically designed to modify
    the configuration of a manager term in the environment. It mainly adds the convenience of
    using a simplified address style that uses "s." as a prefix to refer to the manager's configuration.

    For instance, instead of writing "event_manager.cfg.object_physics_material.func.material_buckets",
    you can write "events.object_physics_material.func.material_buckets" to refer to the same term configuration.
    The same applies to other managers, such as "observations", "commands", "rewards", and "terminations".

    Internally, it replaces the first occurrence of "s." in the address with "_manager.cfg.",
    thus transforming the simplified address into a full manager path.

    Usage:
        .. code-block:: python

            def override_value(env, env_ids, data, value, num_steps):
                if env.common_step_counter > num_steps:
                    return value
                return mdp.modify_term_cfg.NO_CHANGE


            command_object_pose_xrange_adr = CurrTerm(
                func=mdp.modify_term_cfg,
                params={
                    "address": "commands.object_pose.ranges.pos_x",  # note: `_manager.cfg` is omitted
                    "modify_fn": override_value,
                    "modify_params": {"value": (-0.75, -0.25), "num_steps": 12000},
                },
            )
    """
    """在运行时修改管理器项配置的课程。

    这个类继承了:class:`modify_env_param`，并且专门旨在在环境中修改管理器项的配置。
    它主要增加了使用简单的地址风格的便利性，使用"s"作为指向管理器的配置的前置。

    例如，在写"event_manager.cfg.object_physics_material.func.material_buckets"的位置，你可以写"events.object_physics
    _material.func.material_buckets"来指同一个项配置。
    同样适用于其他管理器，例如"观测"，"命令"，"奖励"和"终结"。

    内部，它将"s"在地址中的第一个出现取代为"_manager.cfg"。

    Usage:
        .. code-block:: python

            def override_value(env, env_ids, data, value, num_steps):
                if env.common_step_counter > num_steps:
                    return value
                return mdp.modify_term_cfg.NO_CHANGE


            command_object_pose_xrange_adr = CurrTerm(
                func=mdp.modify_term_cfg,
                params={
                    "address": "commands.object_pose.ranges.pos_x",  # note: `_manager.cfg` is omitted
                    "modify_fn": override_value,
                    "modify_params": {"value": (-0.75, -0.25), "num_steps": 12000},
                },
            )
    """

    '''
    它继承了 modify_env_param 的全部能力，只做了一件事：把用户写的简化路径自动展开成完整路径。
    '''
    def __init__(self, cfg, env):
        # initialize the parent
        super().__init__(cfg, env)
        # overwrite the simplified address with the full manager path
        self._address = self._address.replace("s.", "_manager.cfg.", 1)
        '''
        replace("s.", "_manager.cfg.", 1) 的含义：
            参数	    值	                含义
            第 1 参数	"s."	            要查找的子串
            第 2 参数	"_manager.cfg."	    替换为的子串
            第 3 参数	1	                只替换第一次出现（防止误伤路径后面的 s.）
        为什么是 "_manager.cfg."？
            Isaac Lab 的命名规则是：命令管理器叫 command_manager，它内部的配置叫 command_manager.cfg。"s." 中的 s 是 *_manager 中 s 的占位符。

        实际转换示例：
            简写:  "commands.object_pose.ranges.pos_x"
                    │      └─────────────┬─────────────┘
                    │              完整路径的"尾巴"
                    │
                    │  "commands".replace("s.", "_manager.cfg.", 1)
                    ▼
            完整:  "command_manager.cfg.object_pose.ranges.pos_x"
                    │              │  └─────────────┬─────────────┘
            command_manager    .cfg          完整路径不变

            简写:  "events.object_physics_material.func.material_buckets"
            完整:  "event_manager.cfg.object_physics_material.func.material_buckets"

            简写:  "rewards.track_lin_vel_xy_exp.weight"
            完整:  "reward_manager.cfg.track_lin_vel_xy_exp.weight"
        为什么需要这个简写？
            没有简写时，修改事件管理器中某个项配置的地址需要写成：
                address: "event_manager.cfg.object_physics_material.func.material_buckets"
            有了简写只需要：
                address: "events.object_physics_material.func.material_buckets"
            省了 10 个字符（vent_mana...→s），但更重要的是减少了心智负担——用户不需要记住"事件管理器在代码里叫 event_manager"，只需要知道它的简写是 "events"。
            命名规则统一：命令 → "commands"，观测 → "observations"，奖励 → "rewards"，终止 → "terminations"，事件 → "events"。
        '''
