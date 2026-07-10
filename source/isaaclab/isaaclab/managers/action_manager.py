# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action manager for processing actions sent to the environment."""

from __future__ import annotations
"""处理向环境发送的动作管理器。"""

import inspect
import re
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from prettytable import PrettyTable

import omni.kit.app

from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ActionTermCfg

if TYPE_CHECKING:
    from isaaclab.assets import AssetBase
    from isaaclab.envs import ManagerBasedEnv

'''
ActionTerm 继承自 ManagerTermBase（action_manager.py:33），
是所有动作类型（JointEffortAction、JointPositionAction、DifferentialInverseKinematicsAction 等）的共同父类。
一、它在继承链中的位置
    ManagerTermBase                             ← 存 cfg + env，解析通用属性
        │
        └── ActionTerm(ManagerTermBase)          ← 加入动作特有属性
                │
                ├── JointEffortAction           ← 力矩控制
                ├── JointPositionAction          ← 位置控制
                ├── DifferentialInverseKinematicsAction  ← IK 控制
                ├── BinaryJointPositionAction    ← 夹爪开合
                └── ... 等
'''
class ActionTerm(ManagerTermBase):
    """Base class for action terms.

    The action term is responsible for processing the raw actions sent to the environment
    and applying them to the asset managed by the term. The action term is comprised of two
    operations:

    * Processing of actions: This operation is performed once per **environment step** and
      is responsible for pre-processing the raw actions sent to the environment.
    * Applying actions: This operation is performed once per **simulation step** and is
      responsible for applying the processed actions to the asset managed by the term.
    """
    """基本类别的动作条件。

    动作项负责处理向环境发送的原始动作，并将其应用到该项管理的资产上。
    动作项由两个
    operations:

    * 处理动作:该操作每次**环境步骤**进行一次，负责向环境发送的原始动作的预处理。
    * 执行动作:此操作每**仿真步骤**进行一次，负责将处理的动作应用于该期内管理的资产。
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        """开始动作项。

        参数：
            cfg: 配置对象。
            env: 环境情况。
        """
        # call the base class constructor
        super().__init__(cfg, env)
        # parse config to obtain asset to which the term is applied
        self._asset: AssetBase = self._env.scene[self.cfg.asset_name]
        '''
        这是最关键的一行。ActionTermCfg.asset_name 是 ActionTermCfg 的特有字段（manager_term_cfg.py:152），值是你在配置中写的资产名（如 "robot"）。
        这里用这个名字从场景中取出实际的资产对象。
            # 你的配置:
            JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)

            # __init__ 执行后:
            self._asset = <Articulation 对象，指向 Cartpole 机器人>
        之后所有对机器人的操作（读关节位置、写力矩目标）都通过 self._asset 进行。
        '''

        '''
        IO 描述符用于导出动作的输入/输出规格（维度、范围等），供外部工具（如部署、模型转换）使用。
        _export_IO_descriptor = True 表示当前 ActionTerm 在请求导出 IO 时会自动填充描述符。
        '''
        self._IO_descriptor = GenericActionIODescriptor()
        self._export_IO_descriptor = True

        '''
        调试可视化在 Isaac Sim 的视口中绘制箭头、轨迹等，帮助理解动作如何影响机器人。
        '''
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
    Properties.
    """
    """属性。
    """

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action term."""
        """动作项的维度。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        """输入/原始动作向该项发送。"""
        raise NotImplementedError
    '''
    返回策略网络输出的原始动作值（还没经过缩放、偏移等处理）。
    '''

    @property
    @abstractmethod
    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the term after applying any processing."""
        """经过任何处理后按项计算的动作。"""
        raise NotImplementedError
    '''
    返回经过缩放/偏移/IK 转换后的动作，即真正要写入机器人硬件的值。
    '''

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the action term has a debug visualization implemented."""
        """操作项是否实现了调试可视化。"""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor for the action term."""
        """动作项的IO描述符。"""
        self._IO_descriptor.name = re.sub(r"([a-z])([A-Z])", r"\1_\2", self.__class__.__name__).lower()
        self._IO_descriptor.full_path = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self._IO_descriptor.description = " ".join(self.__class__.__doc__.split())
        self._IO_descriptor.export = self.export_IO_descriptor
        return self._IO_descriptor

    @property
    def export_IO_descriptor(self) -> bool:
        """Whether to export the IO descriptor for the action term."""
        """是否出口IO描述符用于动作项。"""
        return self._export_IO_descriptor

    """
    Operations.
    """
    """操作。
    """

    '''
    调试可视化的开关与资源管理
    这是 ActionTerm 中控制视口内调试可视化的方法。
    当你在 Isaac Sim 的 GUI 中看到机器人关节上画着箭头、轨迹等辅助线时，就是这个方法在背后工作。
    '''
    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the action term data.
        Args:
            debug_vis: Whether to visualize the action term data.
        Returns:
            Whether the debug visualization was successfully set. False if the action term does
            not support debug visualization.
        """
        """设定是否可可视化动作项数据。
        参数：
            debug_vis: 是否可视化动作项数据。
        返回：
            设置错误可视化是否成功。
            False如果操作项不支持调试可视化。
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

    @abstractmethod
    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The actions to process.
        """
        """处理向环境发送的动作。

        说明：
            该函数由管理器每次调用一次。

        参数：
            actions: 处理的动作。
        """
        raise NotImplementedError

    @abstractmethod
    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """
        """投资指数

        说明：
            管理器在每一步仿真时都会调用。
        """
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


'''
里面的一个 _terms 应该就代表了一个 ActionTerm

ActionManager 维护了两份数据：
    self._term_names = ["arm_action", "gripper_action"]    # 独立列表
    self._terms = {"arm_action": ..., "gripper_action": ...}  # 字典
'''
class ActionManager(ManagerBase):
    """Manager for processing and applying actions for a given world.

    The action manager handles the interpretation and application of user-defined
    actions on a given world. It is comprised of different action terms that decide
    the dimension of the expected actions.

    The action manager performs operations at two stages:

    * processing of actions: It splits the input actions to each term and performs any
      pre-processing needed. This should be called once at every environment step.
    * apply actions: This operation typically sets the processed actions into the assets in the
      scene (such as robots). It should be called before every simulation step.
    """
    """管理一个特定世界的处理和应用动作。

    动作管理器处理用户定义的操作在给定的世界上的解释和应用。
    它由不同的动作项组成，决定预期动作的规模。

    动作管理器在两个阶段执行操作:

    * 处理操作:它将输入操作分为每个项，并执行任何必要的预处理。
    * 运行操作:这种操作通常将处理的操作设置在场景的资产中 (如机器人)。
    """

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the action manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ActionTermCfg]``).
            env: The environment instance.

        Raises:
            ValueError: If the configuration is None.
        """
        """启动动作管理器。

        参数：
            cfg: 配置对象或字典 (``dict[str， ActionTermCfg]``)。
            env: 环境情况。

        异常：
            ValueError: 如果配置是None。
        """
        # check if config is None
        if cfg is None:
            raise ValueError("Action manager configuration is None. Please provide a valid configuration.")

        # call the base class constructor (this prepares the terms)
        super().__init__(cfg, env)
        # create buffers to store actions
        self._action = torch.zeros((self.num_envs, self.total_action_dim), device=self.device)  # 当前帧的策略输出动作
        self._prev_action = torch.zeros_like(self._action)  # 上一帧的动作（可供观测使用）

        # check if any term has debug visualization implemented
        self.cfg.debug_vis = False
        for term in self._terms.values():
            self.cfg.debug_vis |= term.cfg.debug_vis

    '''
    动作管理器的信息展示面板
    这是 ActionManager.__str__ 方法。当你执行 print(action_manager) 时，它生成一个格式化的 ASCII 表格，展示当前有哪些动作项。
    一、输出示例
        <ActionManager> contains 2 active terms.
        +---------------------------------------+
        | Active Action Terms (shape: 8)        |
        +-------+--------------------+-----------+
        | Index | Name               | Dimension |
        +-------+--------------------+-----------+
        |   0   | arm_action         |     7     |
        |   1   | gripper_action     |     1     |
        +-------+--------------------+-----------+
        一目了然：2 个动作项，总维度 8（7+1），arm 控制末端位姿（7 维），gripper 控制夹爪开合（1 维）。
    '''
    def __str__(self) -> str:
        """Returns: A string representation for action manager."""
        """Returns: 动作管理器的字符串表示。"""
        msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = f"Active Action Terms (shape: {self.total_action_dim})"
        table.field_names = ["Index", "Name", "Dimension"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Dimension"] = "r"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.action_dim])
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
    def total_action_dim(self) -> int:
        """Total dimension of actions."""
        """动作的总尺寸。"""
        return sum(self.action_term_dim)

    @property
    def active_terms(self) -> list[str]:
        """Name of active action terms."""
        """事件项名称。"""
        return self._term_names

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        """每个动作项的形状。"""
        return [term.action_dim for term in self._terms.values()]

    @property
    def action(self) -> torch.Tensor:
        """The actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        """动作向环境发送。
        形状是 (num_envs，total_action_dim)。
        """
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        """The previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        """之前的动作向环境发送。
        形状是 (num_envs，total_action_dim)。
        """
        return self._prev_action

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        """命令项是否实现了调试可视化。"""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    '''
    动作 IO 规格导出器
    这个方法收集所有 ActionTerm 的 IO 描述符，并重新格式化为标准结构，用于模型导出和部署——让外部工具知道策略网络的输入输出规格。
    最终输出是可直接序列化为 YAML/JSON 的纯字典列表，供模型部署和外部工具使用。
    '''
    @property
    def get_IO_descriptors(self) -> list[dict[str, Any]]:
        """Get the IO descriptors for the action manager.

        Returns:
            A dictionary with keys as the term names and values as the IO descriptors.
        """
        """给动作管理器提供IO描述符。

        返回：
            一个字典，用键作为项名称和值作为IO描述符。
        """

        data = []

        for term_name, term in self._terms.items():
            try:
                data.append(term.IO_descriptor.__dict__.copy())
            except Exception as e:
                print(f"Error getting IO descriptor for term '{term_name}': {e}")

        formatted_data = []
        for item in data:
            name = item.pop("name")
            formatted_item = {"name": name, "extras": item.pop("extras")}
            print(item["export"])
            if not item.pop("export"):
                continue
            for k, v in item.items():
                # Check if v is a tuple and convert to list
                if isinstance(v, tuple):
                    v = list(v)
                if k in ["description", "units"]:
                    formatted_item["extras"][k] = v
                else:
                    formatted_item[k] = v
            formatted_data.append(formatted_item)

        return formatted_data

    """
    Operations.
    """
    """操作。
    """

    '''
    拆分拼接的动作向量
    这是 ActionManager 对 ManagerBase 抽象方法的实现（manager_base.py:504）。
    它从拼接的大动作向量中按各 Term 的维度切分，返回每个 Term 的名字和对应值，供 GUI 面板实时显示。
        返回示例
            get_active_iterable_terms(env_idx=0)
            # 返回:
            [
                ("arm_action", [0.3, -0.1, 0.0, 0.5, -0.2, 0.1, 0.4]),
                ("gripper_action", [1.0]),
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
            term_actions = self._action[env_idx, idx : idx + term.action_dim].cpu()
            terms.append((name, term_actions.tolist()))
            idx += term.action_dim
        return terms

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the action data.
        Args:
            debug_vis: Whether to visualize the action data.
        Returns:
            Whether the debug visualization was successfully set. False if the action
            does not support debug visualization.
        """
        """设定是否可可视化动作数据。
        参数：
            debug_vis: 是否可视化动作数据。
        返回：
            设置错误可视化是否成功。
            False如果该操作不支持调试可视化。
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    '''
    环境重置时的动作历史清零
    这是 ActionManager.reset()，在 ManagerBasedRLEnv._reset_idx() 中被调用（manager_based_rl_env.py:450）。
    当一个环境被重置（机器人摔倒/超时）时，这个环境对应的动作缓冲区必须清零。
    '''
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the action history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """
        """恢复动作历史。

        参数：
            env_ids: 环境 ID。
                     在 None 中，默认情况下考虑所有环境。

        返回：
            一个空白的字典。
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the action history
        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0
        # reset all action terms
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    '''
    动作接收与分发中心
        这是 ActionManager 最核心的方法之一，被 ManagerBasedRLEnv.step() 在每步环境步中调用一次。
        它负责接收策略网络输出的完整动作向量，保存历史，并按各 ActionTerm 的维度切片分发。

    它在 step() 中的执行时机
        ManagerBasedRLEnv.step(action)
            │
            ├── action_manager.process_action(action)    ← 每环境步调用一次（这里！）
            │
            ├── for _ in range(decimation):             ← 高频物理循环
            │       ├── action_manager.apply_action()    ← 每物理步调用一次
            │       ├── scene.write_data_to_sim()
            │       └── sim.step()
            │
            ├── termination_manager.compute()
            ├── reward_manager.compute()
            └── observation_manager.compute()
    process_action 每环境步调用一次（做预处理），apply_action 每物理步调用一次（写硬件）。这就是 ActionTerm 文档中说的"两阶段"机制。
    '''
    def process_action(self, action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        """处理向环境发送的动作。

        说明：
            这个函数应每次调用一次。

        参数：
            action: 处理的动作。
        """
        # check if action dimension is valid
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def apply_action(self) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        """适用于环境/仿真。

        说明：
            在每个仿真步骤中都应调用。
        """
        for term in self._terms.values():
            term.apply_actions()

    def get_term(self, name: str) -> ActionTerm:
        """Returns the action term with the specified name.

        Args:
            name: The name of the action term.

        Returns:
            The action term with the specified name.
        """
        """返回使用指定名称的操作项。

        参数：
            name: 动作项的名称。

        返回：
            用指定名称的动作项。
        """
        return self._terms[name]

    def serialize(self) -> dict:
        """Serialize the action manager configuration.

        Returns:
            A dictionary of serialized action term configurations.
        """
        """连载动作管理器配置。

        返回：
            一个系列动作项配置字典。
        """
        return {term_name: term.serialize() for term_name, term in self._terms.items()}

    """
    Helper functions.
    """
    """辅助函数。
    """

    '''
    把配置字段变成活的 ActionTerm
        这是 ActionManager 对 ManagerBase 抽象方法的实现（manager_base.py:370）。
        它的职责是遍历 ActionsCfg 的每个字段，用 class_type 创建对应的 ActionTerm 实例。
    '''
    def _prepare_terms(self):
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._terms: dict[str, ActionTerm] = dict()
        '''
        self._term_names 和 self._terms 是实例变量——每个 ActionManager 对象拥有独立的一份，只在对象存活期间存在。
        self. 前缀是区分实例变量和局部变量的唯一标志。
        '''

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # parse action terms from the config
        for term_name, term_cfg in cfg_items:
            # check if term config is None
            if term_cfg is None:
                continue
            # check valid type
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, ActionTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ActionType.")
            # add term name and parameters
            self._term_names.append(term_name)
            self._terms[term_name] = term
            '''
            _prepare_terms()
            │
            ├── 初始化两个容器
            │     self._term_names = []           ActionTerm 名字列表
            │     self._terms = {}                {名字 → ActionTerm 实例}
            │
            ├── 获取配置项
            │     if dict → cfg.items()
            │     if @configclass → cfg.__dict__.items()
            │
            └── 对每个字段:
                ├── 跳过 None
                ├── 校验类型 (必须是 ActionTermCfg)
                ├── 用 class_type 创建实例
                ├── 校验实例类型 (必须是 ActionTerm)
                └── 存入 self._terms

            一句话总结
                _prepare_terms 是 ActionManager 的配置解析器——遍历 ActionsCfg 的每个字段，用 class_type 将 ActionTermCfg 变成活的 ActionTerm 实例（如 JointEffortAction），存入 self._terms 字典。
                和其他 Manager（如 RewardManager）不同，ActionManager 的 term 是类实例而非纯函数，因为动作需要维护内部状态（buffer、asset 引用）。
            '''
