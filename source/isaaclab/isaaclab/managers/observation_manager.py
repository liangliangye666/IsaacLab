# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation manager for computing observation signals for a given world."""

from __future__ import annotations
"""计算一个特定世界的观测信号的观测管理器。"""

import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from prettytable import PrettyTable

from isaaclab.utils import class_to_dict, modifiers, noise
from isaaclab.utils.buffers import CircularBuffer

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ObservationGroupCfg, ObservationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ObservationManager(ManagerBase):
    """Manager for computing observation signals for a given world.

    Observations are organized into groups based on their intended usage. This allows having different observation
    groups for different types of learning such as asymmetric actor-critic and student-teacher training. Each
    group contains observation terms which contain information about the observation function to call, the noise
    corruption model to use, and the sensor to retrieve data from.

    Each observation group should inherit from the :class:`ObservationGroupCfg` class. Within each group, each
    observation term should instantiate the :class:`ObservationTermCfg` class. Based on the configuration, the
    observations in a group can be concatenated into a single tensor or returned as a dictionary with keys
    corresponding to the term's name.

    If the observations in a group are concatenated, the shape of the concatenated tensor is computed based on the
    shapes of the individual observation terms. This information is stored in the :attr:`group_obs_dim` dictionary
    with keys as the group names and values as the shape of the observation tensor. When the terms in a group are not
    concatenated, the attribute stores a list of shapes for each term in the group.

    .. note::
        When the observation terms in a group do not have the same shape, the observation terms cannot be
        concatenated. In this case, please set the :attr:`ObservationGroupCfg.concatenate_terms` attribute in the
        group configuration to False.

    Observations can also have history. This means a running history is updated per sim step. History can be controlled
    per :class:`ObservationTermCfg` (See the :attr:`ObservationTermCfg.history_length` and
    :attr:`ObservationTermCfg.flatten_history_dim`). History can also be controlled via :class:`ObservationGroupCfg`
    where group configuration overwrites per term configuration if set. History follows an oldest to newest ordering.

    The observation manager can be used to compute observations for all the groups or for a specific group. The
    observations are computed by calling the registered functions for each term in the group. The functions are
    called in the order of the terms in the group. The functions are expected to return a tensor with shape
    (num_envs, ...).

    If a noise model or custom modifier is registered for a term, the function is called to corrupt
    the observation. The corruption function is expected to return a tensor with the same shape as the observation.
    The observations are clipped and scaled as per the configuration settings.
    """
    """对于一个特定的世界来说，计算观测信号的管理器。

    观测根据其预期使用进行组合。
    这使得不同类型的学习，如不对称的演员-批评者和学生-教师培训，具有不同的观测群体。
    每个组包含关于调用的观测函数，使用的噪声损坏模型和检测器的数据信息的观测项。

    每个观测组应继承:class:`ObservationGroupCfg`类。
    在每个组内，每个观测项都应该表示:class:`ObservationTermCfg`类。
    根据配置，一个组中的观测可以连接到单个子中，或者作为一个字典返回，按项名称的键。

    如果组中的观测是连锁的，则根据单个观测项的形状计算连锁子的形状。
    这些信息存储在:attr:`group_obs_dim`字典中
    with keys as the group names and values as the shape of the observation tensor. When the terms in a group are not
    连接，属性存储了组中每个项的形状列表。

    .. 说明::
        当一个组中的观测项没有相同的形状时，观测项不能连接。
        在此情况下，请设置组配置中的:attr:`ObservationGroupCfg.concatenate_terms`属性为False。

    观测也可以有历史。
    这意味着运行历史每sim步骤更新。
    历史可以控制
    per :类:`ObservationTermCfg` (见:attr:`ObservationTermCfg.history_length`和
    :attr:`ObservationTermCfg.flatten_history_dim`)。
    历史也可以通过:class:`ObservationGroupCfg`来控制，如果设置，则组配置按项配置覆盖。
    历史从最古老到最新的秩序。

    观测管理器可用于计算对所有组或特定组的观测。
    通过调用该组中每个项的注册函数来计算观测。
    函数以组中的项顺序调用。
    函数预计将返回一个形状的子 (num_envs， ...)。

    如果一个噪音模型或定制修改器被注册为一段时间，该函数被要求破坏观测。
    预计腐败函数将返回与观测相同的形状的子。
    根据配置设置，观测被裁剪和扩展。
    """
    '''
    剥离 batch
    就是丢掉第一个维度（batch 维度）。
    一、为什么批量维度是重复信息？
        观测函数总是返回形状 [num_envs, ...] 的张量：
            def joint_pos(env):
                return env.scene["robot"].data.joint_pos   # → [4096, 7]
            #                                                   ↑      ↑
            #                                               num_envs  每个环境的维度
        num_envs 永远是已知的（self._env.num_envs），存它没意义。而且拼接观测组时，需要的是"每个环境内部有多少维"，不是"总共有多少个环境"。所以只保留 batch 之后的部分。

    二、[1:] 切片的意思
            obs_dims = (4096, 7)
            obs_dims[1:]   →   (7,)
            #             ↑
            #       从索引 1 开始取到末尾，丢掉索引 0
        obs_dims	        含义	                    obs_dims[1:]	存下来的
        (4096, 7)	        4096 环境，各 7 个关节位置	    (7,)	        每个关节维度
        (4096, 3, 64, 64)	4096 环境，各 RGB 图像	        (3, 64, 64)	    图像通道和尺寸
        (4096, 21)	        4096 环境，各 21 维展平历史	    (21,)	        特征维度
    三、存在哪里、用在哪里
            # 存储（_prepare_terms 末尾）:
            self._group_obs_term_dim[group_name].append(obs_dims[1:])
            # {"policy": [(7,), (7,), (1,)]}

            # 使用（__init__ 中计算拼接总维度）:
            self._group_obs_dim[group_name] = (15,)   # 7+7+1
        如果不剥离 batch，就会变成 (4096, 7) + (4096, 7) + (4096, 1) → 拼接逻辑需要额外处理第一个维度，徒增复杂度。
    '''

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize observation manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ObservationGroupCfg]``).
            env: The environment instance.

        Raises:
            ValueError: If the configuration is None.
            RuntimeError: If the shapes of the observation terms in a group are not compatible for concatenation
                and the :attr:`~ObservationGroupCfg.concatenate_terms` attribute is set to True.
        """
        """启动观测管理器。

        参数：
            cfg: 配置对象或字典 (``dict[str， ObservationGroupCfg]``)。
            env: 环境情况。

        异常：
            ValueError: 如果配置是None。
            RuntimeError: 如果一个组中的观测项的形状不适合连锁，并且:attr:`~ObservationGroupCfg.concatenate_terms`属性设置为True。
        """
        # check that cfg is not None
        if cfg is None:
            raise ValueError("Observation manager configuration is None. Please provide a valid configuration.")

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)

        # compute combined vector for obs group
        self._group_obs_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = dict()
        for group_name, group_term_dims in self._group_obs_term_dim.items():
            # if terms are concatenated, compute the combined shape into a single tuple
            # otherwise, keep the list of shapes as is
            if self._group_obs_concatenate[group_name]: # 分支 A: 拼接模式 → 算出一个总维度,[N, 总维度] 一个大张量
                try:
                    term_dims = torch.stack([torch.tensor(dims, device="cpu") for dims in group_term_dims], dim=0)
                    if len(term_dims.shape) > 1:    # 子情况 A1：所有 term 输出一维特征（最常见）——关节位置 7 维、关节速度 7 维、动作 1 维，全部拼接成 15 维向量。
                        if self._group_obs_concatenate_dim[group_name] >= 0:
                            dim = self._group_obs_concatenate_dim[group_name] - 1  # account for the batch offset
                        else:
                            dim = self._group_obs_concatenate_dim[group_name]
                        dim_sum = torch.sum(term_dims[:, dim], dim=0)
                        term_dims[0, dim] = dim_sum
                        term_dims = term_dims[0]
                    else:
                        term_dims = torch.sum(term_dims, dim=0)
                    self._group_obs_dim[group_name] = tuple(term_dims.tolist())
                except RuntimeError:
                    raise RuntimeError(
                        f"Unable to concatenate observation terms in group '{group_name}'."
                        f" The shapes of the terms are: {group_term_dims}."
                        " Please ensure that the shapes are compatible for concatenation."
                        " Otherwise, set 'concatenate_terms' to False in the group configuration."
                    )
            else:   # 分支 B: 字典模式 → 保留各 term 独立维度,{"joint_pos": [N,7], "joint_vel": [N,7]}
                self._group_obs_dim[group_name] = group_term_dims

        # Stores the latest observations.
        self._obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None
    '''
    观测组（Group）是什么？
        和其他 Manager 不同，ObservationManager 不是把所有 term 拍平成一个列表，而是按用途分组：
        ObservationsCfg:
            policy: ObservationGroupCfg(...)     # ← 给策略网络看的
                ├── joint_pos     → [N, 7]
                ├── joint_vel     → [N, 7]
                └── actions       → [N, 1]
            critic: ObservationGroupCfg(...)     # ← 给 Critic 网络看的（可能含特权信息）
                ├── joint_pos     → [N, 7]
                ├── joint_vel     → [N, 7]
                └── base_height   → [N, 1]      # ← critic 独有，policy 看不到
        这样不对称 Actor-Critic 训练中，Actor 和 Critic 可以看不同的信息。

    __init__ 做了一件核心事：计算 _group_obs_dim
        输入（_prepare_terms 中已填充的）:
            _group_obs_term_dim = {
                "policy": [(7,), (7,), (1,)],    # joint_pos(7) + joint_vel(7) + actions(1)
                "critic": [(7,), (7,), (1,)],    # 同上
            }
        输出:
            _group_obs_dim = {
                "policy":  (15,),                # 7+7+1 = 15 维
                "critic":  (15,),
            }
    '''

    '''
    按组展示的观测信息面板
    '''
    def __str__(self) -> str:
        """Returns: A string representation for the observation manager."""
        """Returns: 对观测管理器的字符串表示。"""
        msg = f"<ObservationManager> contains {len(self._group_obs_term_names)} groups.\n"

        # add info for each group
        for group_name, group_dim in self._group_obs_dim.items():
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Observation Terms in Group: '{group_name}'"
            if self._group_obs_concatenate[group_name]:
                table.title += f" (shape: {group_dim})"
            table.field_names = ["Index", "Name", "Shape"]
            # set alignment of table columns
            table.align["Name"] = "l"
            # add info for each term
            obs_terms = zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
            )
            for index, (name, dims) in enumerate(obs_terms):
                # resolve inputs to simplify prints
                tab_dims = tuple(dims)
                # add row
                table.add_row([index, name, tab_dims])
            # convert table to string
            msg += table.get_string()
            msg += "\n"

        return msg
    '''
    输出示例
        <ObservationManager> contains 2 groups.

        +--------------------------------------------------+
        | Active Observation Terms in Group: 'policy' (shape: (15,)) |
        +-------+------------+-------+
        | Index | Name       | Shape |
        +-------+------------+-------+
        |   0   | joint_pos  | (7,)  |
        |   1   | joint_vel  | (7,)  |
        |   2   | actions    | (1,)  |
        +-------+------------+-------+

        +--------------------------------------------------+
        | Active Observation Terms in Group: 'critic' (shape: (16,)) |
        +-------+-------------+-------+
        | Index | Name        | Shape |
        +-------+-------------+-------+
        |   0   | joint_pos   | (7,)  |
        |   1   | joint_vel   | (7,)  |
        |   2   | actions     | (1,)  |
        |   3   | base_height | (1,)  |
        +-------+-------------+-------+
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

        if self._obs_buffer is None:    # 懒加载: 如果 _obs_buffer 为空 → 调 compute() 计算一次
            self.compute()
        obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = self._obs_buffer

        for group_name, _ in self._group_obs_dim.items():
            if not self.group_obs_concatenate[group_name]:  # 路径 A: concatenate_terms=False (字典模式),obs_buffer[group] 是 dict → 直接取 term[env_idx]
                for name, term in obs_buffer[group_name].items():
                    terms.append((group_name + "-" + name, term[env_idx].cpu().tolist()))
                continue
            '''
            数据结构：obs_buffer["policy"] 是一个 dict：
                {
                    "joint_pos": Tensor[N, 7],
                    "joint_vel": Tensor[N, 7],
                }
            直接取 dict["joint_pos"][env_idx]，简单直接。命名加上了 group 前缀（"policy-joint_pos"）以区分同名 term 在不同 group 中的值。
            '''

            # 路径 B: concatenate_terms=True (拼接模式),obs_buffer[group] 是大张量 → narrow 切片提取各 term
            idx = 0
            concat_dim = self._group_obs_concatenate_dim[group_name]
            # handle cases where concat dim is positive, account for the batch dimension
            if concat_dim > 0:
                concat_dim -= 1
            # add info for each term
            data = obs_buffer[group_name]
            for name, shape in zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
            ):
                # extract the term from the buffer based on the shape
                term = data[env_idx].narrow(dim=concat_dim, start=idx, length=shape[concat_dim])
                terms.append((group_name + "-" + name, term.cpu().tolist()))
                idx += shape[concat_dim]

        return terms
    '''
    返回示例
        get_active_iterable_terms(env_idx=0)
        # 返回:
        [
            ("policy-joint_pos", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("policy-joint_vel", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("policy-actions",   [0.0]),
            ("critic-joint_pos", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("critic-joint_vel", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("critic-actions",   [0.0]),
            ("critic-base_height", [0.5]),
        ]
    '''

    """
    Properties.
    """
    """属性。
    """

    '''
    各组的 term 名字
        {"policy": ["joint_pos", "joint_vel", "actions"], "critic": [...]}
    '''
    @property
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active observation terms in each group.

        The keys are the group names and the values are the list of observation term names in the group.
        """
        """每组的活跃观测项名称。

        关键是组名，值是组中的观测项名单。
        """
        return self._group_obs_term_names

    '''
    各组的总维度
        返回值有两种形态，取决于 concatenate_terms：
        # 拼接模式: {"policy": (15,), "critic": (16,)},单个元组,拼接后总形状
        # 字典模式: {"policy": [(7,), (7,), (1,)], ...},元组列表,各 term 独立形状
    '''
    @property
    def group_obs_dim(self) -> dict[str, tuple[int, ...] | list[tuple[int, ...]]]:
        """Shape of computed observations in each group.

        The key is the group name and the value is the shape of the observation tensor.
        If the terms in the group are concatenated, the value is a single tuple representing the
        shape of the concatenated observation tensor. Otherwise, the value is a list of tuples,
        where each tuple represents the shape of the observation tensor for a term in the group.
        """
        """在每个组中计算观测的形状。

        关键是组名，值是观测张量的形状。
        如果组中的项是连环的，则值是单一的元组，代表了连环的观测张量的形状。
        否则，值是 tuples 的列表，其中每个 tuple 代表了对组中的一个项的观测张量的形状。
        """
        return self._group_obs_dim

    '''
    各 term 的独立维度
        # {"policy": [(7,), (7,), (1,)]}
    '''
    @property
    def group_obs_term_dim(self) -> dict[str, list[tuple[int, ...]]]:
        """Shape of individual observation terms in each group.

        The key is the group name and the value is a list of tuples representing the shape of the observation terms
        in the group. The order of the tuples corresponds to the order of the terms in the group.
        This matches the order of the terms in the :attr:`active_terms`.
        """
        """每组的个别观测项的形状。

        关键是组名，值是表达组中的观测项的形状的体列表。
        双数的顺序与组中的项的顺序相符。
        这符合:attr:`active_terms`中的项顺序。
        """
        return self._group_obs_term_dim
    '''
    和 group_obs_dim 的区别：
        属性	                存什么	                        何时用
        group_obs_term_dim	    每个 term 的形状（不拼接）	    get_active_iterable_terms 切片时
        group_obs_dim	        整组的形状（拼接后或列表）	    构建 observation_space 时
        即使 concatenate_terms=True，group_obs_term_dim 仍然保持各 term 的独立形状——因为后续 get_active_iterable_terms 需要用这些形状从拼接张量中切片。
    '''

    '''
    各组是否拼接
        # {"policy": True, "critic": True}
        一个简单的布尔标志字典，来自 ObservationGroupCfg.concatenate_terms 配置。
        使用方根据这个标志决定数据是直接取（字典模式）还是切片取（拼接模式）。
    '''
    @property
    def group_obs_concatenate(self) -> dict[str, bool]:
        """Whether the observation terms are concatenated in each group or not.

        The key is the group name and the value is a boolean specifying whether the observation terms in the group
        are concatenated into a single tensor. If True, the observations are concatenated along the last dimension.

        The values are set based on the :attr:`~ObservationGroupCfg.concatenate_terms` attribute in the group
        configuration.
        """
        """观测项是否在每个组中连锁。

        关键是组名，值是布尔式，指定该组中的观测项是否被连接到单个子中。
        如果是True，则观测在最后一个维度上连接。

        根据组配置中的:attr:`~ObservationGroupCfg.concatenate_terms`属性设置值。
        """
        return self._group_obs_concatenate

    '''
    观测 IO 规格导出器
    '''
    @property
    def get_IO_descriptors(self, group_names_to_export: list[str] = ["policy"]):
        """Get the IO descriptors for the observation manager.

        Returns:
            A dictionary with keys as the group names and values as the IO descriptors.
        """
        """给观测管理器提供IO描述器。

        返回：
            一个字典，键为组名、值为 IO 描述符。
        """
        '''
        get_IO_descriptors(group_names_to_export=["policy"])
            │
            ├── 阶段一: 收集原始描述符（按 group → term 两层遍历）
            │     ├── 调 func(env, **params, inspect=True) 触发描述符生成
            │     ├── 拷贝 _descriptor 并注入 term 的 overloads
            │     └── 按 group 分组存储
            │
            ├── 阶段二: 格式化重构
            │     ├── name → 顶层
            │     ├── extras → 顶层
            │     ├── scale/clip/history → overloads
            │     ├── modifiers/description/units → extras
            │     ├── tuple/list/tensor → 纯 Python 类型
            │     └── 其余字段 → 顶层
            │
            └── 过滤: 只保留 group_names_to_export 中的组（默认只有 "policy"）
        '''

        group_data = {}

        for group_name in self._group_obs_term_names:
            group_data[group_name] = []
            # check if group name is valid
            if group_name not in self._group_obs_term_names:
                raise ValueError(
                    f"Unable to find the group '{group_name}' in the observation manager."
                    f" Available groups are: {list(self._group_obs_term_names.keys())}"
                )
            # iterate over all the terms in each group
            group_term_names = self._group_obs_term_names[group_name]
            # read attributes for each term
            obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])

            for term_name, term_cfg in obs_terms:
                # Call to the observation function to get the IO descriptor with the inspect flag set to True
                try:
                    term_cfg.func(self._env, **term_cfg.params, inspect=True)
                    # Copy the descriptor and update with the term's own extra parameters
                    desc = term_cfg.func._descriptor.__dict__.copy()
                    # Create a dictionary to store the overloads
                    overloads = {}
                    # Iterate over the term's own parameters and add them to the overloads dictionary
                    for k, v in term_cfg.__dict__.items():
                        # For now we do not add the noise modifier
                        if k in ["modifiers", "clip", "scale", "history_length", "flatten_history_dim"]:
                            overloads[k] = v
                    desc.update(overloads)
                    group_data[group_name].append(desc)
                except Exception as e:
                    print(f"Error getting IO descriptor for term '{term_name}' in group '{group_name}': {e}")
        # Format the data for YAML export
        formatted_data = {}
        for group_name, data in group_data.items():
            formatted_data[group_name] = []
            for item in data:
                name = item.pop("name")
                formatted_item = {"name": name, "overloads": {}, "extras": item.pop("extras")}
                for k, v in item.items():
                    # Check if v is a tuple and convert to list
                    if isinstance(v, tuple):
                        v = list(v)
                    # Check if v is a tensor and convert to list
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().numpy().tolist()
                    if k in ["scale", "clip", "history_length", "flatten_history_dim"]:
                        formatted_item["overloads"][k] = v
                    elif k in ["modifiers", "description", "units"]:
                        formatted_item["extras"][k] = v
                    else:
                        formatted_item[k] = v
                formatted_data[group_name].append(formatted_item)
        formatted_data = {k: v for k, v in formatted_data.items() if k in group_names_to_export}
        return formatted_data

    """
    Operations.
    """
    """操作。
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # call all terms that are classes
        for group_name, group_cfg in self._group_obs_class_term_cfgs.items():
            '''
            第 1 层：类形式 term 重置
                遍历 _group_obs_class_term_cfgs——仅在 _prepare_terms 中当 isinstance(term_cfg.func, ManagerTermBase) 时才加入的列表。
                只有类实现的观测 term（非纯函数）才有内部状态需要重置。
            '''
            for term_cfg in group_cfg:
                term_cfg.func.reset(env_ids=env_ids)
            '''
            第 2 层：历史缓冲区清理（Observation 独有）
            背景：观测可以配置 history_length > 0，此时会用一个 CircularBuffer 保存过去 N 步的观测值：
                # 配置: ObsTermCfg(func=joint_pos, history_length=3)
                # 效果: 观测不是 [N, 7]，而是 [N, 3, 7]  ← 3 步历史
                #                          └─ 当前步
                #                          └─ 上一步
                #                          └─ 上一步的上一步
            环境重置后，旧的 3 步历史都是"上一个回合"的数据，必须清空。CircularBuffer.reset(batch_ids=env_ids) 把指定环境的历史槽位全部置零。
            '''
            # reset terms with history
            for term_name in self._group_obs_term_names[group_name]:
                if term_name in self._group_obs_term_history_buffer[group_name]:
                    self._group_obs_term_history_buffer[group_name][term_name].reset(batch_ids=env_ids)
        '''
        第 3 层：噪声/修改器重置
            如果噪声模型或观测修改器（modifiers）是用类实现的（有内部状态，如自适应噪声），也需要重置。
        '''
        # call all modifiers that are classes
        for mod in self._group_obs_class_instances:
            mod.reset(env_ids=env_ids)

        # nothing to log here
        return {}

    '''
    所有组观测的统一计算入口
        ObservationManager.compute() 只有一个职责：遍历所有观测组，逐个调 compute_group()，缓存结果，返回。
    '''
    def compute(self, update_history: bool = False) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute the observations per group for all groups.

        The method computes the observations for all the groups handled by the observation manager.
        Please check the :meth:`compute_group` on the processing of observations per group.

        Args:
            update_history: The boolean indicator without return obs should be appended to observation history.
                Default to False, in which case calling compute_group does not modify history. This input is no-ops
                if the group's history_length == 0.

        Returns:
            A dictionary with keys as the group names and values as the computed observations.
            The observations are either concatenated into a single tensor or returned as a dictionary
            with keys corresponding to the term's name.
        """
        """计算每个组的观测结果。

        该方法计算了观测管理器处理的所有组的观测。
        请检查对每个组的观测处理的:meth:`compute_group`。

        参数：
            update_history: 没有返回obs的布尔指标应添加到观测历史。
                            默认为False，在这种情况下调用compute_group不会改变历史记录。
                            这个输入是无运行
                if the group's history_length == 0.

        返回：
            一个字典，按组名字和计算观测值的关键。
            这些观测要么被连接到单个子中，要么作为字典返回。
            with keys corresponding to the term's name.
        """
        # create a buffer for storing obs from all the groups
        obs_buffer = dict()
        # iterate over all the terms in each group
        for group_name in self._group_obs_term_names:
            obs_buffer[group_name] = self.compute_group(group_name, update_history=update_history)
        # otherwise return a dict with observations of all groups
        '''
            ├── for group_name in ["policy", "critic"]:
            │     obs_buffer["policy"]  = self.compute_group("policy")
            │     obs_buffer["critic"]  = self.compute_group("critic")
        '''

        # Cache the observations.
        self._obs_buffer = obs_buffer
        return obs_buffer

    def compute_group(self, group_name: str, update_history: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        """Computes the observations for a given group.

        The observations for a given group are computed by calling the registered functions for each
        term in the group. The functions are called in the order of the terms in the group. The functions
        are expected to return a tensor with shape (num_envs, ...).

        The following steps are performed for each observation term:

        1. Compute observation term by calling the function
        2. Apply custom modifiers in the order specified in :attr:`ObservationTermCfg.modifiers`
        3. Apply corruption/noise model based on :attr:`ObservationTermCfg.noise`
        4. Apply clipping based on :attr:`ObservationTermCfg.clip`
        5. Apply scaling based on :attr:`ObservationTermCfg.scale`

        We apply noise to the computed term first to maintain the integrity of how noise affects the data
        as it truly exists in the real world. If the noise is applied after clipping or scaling, the noise
        could be artificially constrained or amplified, which might misrepresent how noise naturally occurs
        in the data.

        Args:
            group_name: The name of the group for which to compute the observations. Defaults to None,
                in which case observations for all the groups are computed and returned.
            update_history: The boolean indicator without return obs should be appended to observation group's history.
                Default to False, in which case calling compute_group does not modify history. This input is no-ops
                if the group's history_length == 0.

        Returns:
            Depending on the group's configuration, the tensors for individual observation terms are
            concatenated along the last dimension into a single tensor. Otherwise, they are returned as
            a dictionary with keys corresponding to the term's name.

        Raises:
            ValueError: If input ``group_name`` is not a valid group handled by the manager.
        """
        """计算给定组的观测。

        对给定的组的观测是通过调用该组中每个项的注册函数来计算的。
        函数以组中的项顺序调用。
        函数预计将返回一个形状的子 (num_envs， ...)。

        每个观测项都执行以下步骤:

        1. 通过调用函数来计算观测项
        2. 在:attr:`ObservationTermCfg.modifiers`中指定的顺序中应用定制修改器
        3. 基于:attr:`ObservationTermCfg.noise`的腐败/噪声模型
        4. 基于:attr:`ObservationTermCfg.clip`的裁剪
        5. 基于:attr:`ObservationTermCfg.scale`的扩展应用

        我们首先将噪音应用于计算项，以保持噪音如何影响数据的完整性，
        如果在裁剪或扩展后应用噪音，噪音可能会被人工限制或放大，这可能会错误地描述数据中自然发生的噪音。

        参数：
            group_name: 对观测计算的组名称。
                        在 None 中，默认情况下计算并返回所有组的观测。
            update_history: 没有返回obs的布鲁尔指标应添加到观测组的历史。
                            默认为False，在这种情况下调用compute_group不会改变历史记录。
                            这个输入是无运行
                if the group's history_length == 0.

        返回：
            根据组的配置，单个观测项的子沿着最后一个维度连接成一个 single子。
            否则，它们将作为一个字典返回，

        异常：
            ValueError: 如果输入``group_name``不是管理器处理的有效组。
        """
        # check ig group name is valid
        if group_name not in self._group_obs_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the observation manager."
                f" Available groups are: {list(self._group_obs_term_names.keys())}"
            )
        # iterate over all the terms in each group
        group_term_names = self._group_obs_term_names[group_name]
        # buffer to store obs per group
        group_obs = dict.fromkeys(group_term_names, None)
        # read attributes for each term
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])

        '''
        单个 ObsTerm 的处理顺序：
            func(env, **params)        ← 第 1 步: 计算原始值,原始观测 (raw data),例如: joint_pos = [N, 7]
                ↓
            modifiers (自定义修改器)    ← 第 2 步: 用户自定义变换,例如: 归一化到 [0, 1]
                ↓
            noise (噪声模型)           ← 第 3 步: 注入噪声,例如: + N(0, 0.01)
                ↓
            clip (裁剪)                ← 第 4 步: 钳制范围,例如: clamp 到 [-5, 5]
                ↓
            scale (缩放)               ← 第 5 步: 乘以系数,例如: ×2.0
                ↓
            history buffer (历史)      ← 第 6 步: 追加到历史队列
                ↓
            存储到 group_obs[term_name] ← 存入组字典
                ↓
            (全部 term 处理完后) concatenate → 拼接或字典返回
        '''
        # evaluate terms: compute, add noise, clip, scale, custom modifiers
        for term_name, term_cfg in obs_terms:
            '''
                调用你写的观测函数（如 mdp.joint_pos），传入环境对象和参数。
                .clone() 至关重要——因为 func 可能返回 env.scene 内部张量的视图，直接修改会破坏物理状态。
                .clone() 创建独立副本，后面放心改。
            '''
            # compute term's value
            obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()

            '''
                用户自定义的变换序列，按列表顺序依次应用。典型用途：归一化（减均值除方差）、指数平滑等。
                这一步在噪声之前，因为原始数据应该先被正规化再考虑加噪的尺度。
            '''
            # apply post-processing
            if term_cfg.modifiers is not None:
                for modifier in term_cfg.modifiers:
                    obs = modifier.func(obs, **modifier.params)

            '''
                两种噪声模型：
                    类型	        适用场景
                    NoiseCfg	    简单高斯噪声（固定参数）
                    NoiseModelCfg	有状态噪声模型（如自适应噪声，需要训练）
                噪声在 clip/scale 之前的原因（docstring 原文）：现实中噪声是在传感器层面存在的，裁剪和缩放是后处理。
                如果先裁剪再加噪，噪声会被人工约束；先加噪再裁剪更贴近真实传感器行为。
            '''
            if isinstance(term_cfg.noise, noise.NoiseCfg):
                obs = term_cfg.noise.func(obs, term_cfg.noise)
            elif isinstance(term_cfg.noise, noise.NoiseModelCfg) and term_cfg.noise.func is not None:
                obs = term_cfg.noise.func(obs)

            '''
                clip_() 是 PyTorch 的原地操作（末尾 _ 表示 in-place），直接修改张量不创建副本——4096 个并行环境，每帧省一次内存分配。
                term_cfg.clip 是 (min, max) 元组（见 manager_term_cfg.py:167）
            '''
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])

            '''
                mul_() 也是原地操作。term_cfg.scale 是 float 或 tuple[float, ...]（manager_term_cfg.py:314）：
                    scale=2.0 → 所有维度 ×2
                    scale=(1.0, 0.5) → 维度0×1.0, 维度1×0.5
            '''
            if term_cfg.scale is not None:
                obs = obs.mul_(term_cfg.scale)
            # Update the history buffer if observation term has history enabled
            if term_cfg.history_length > 0:
                circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                if update_history:
                    circular_buffer.append(obs)
                elif circular_buffer._buffer is None:   # ← 特殊情况
                    # because circular buffer only exits after the simulation steps,
                    # this guards history buffer from corruption by external calls before simulation start
                    circular_buffer = CircularBuffer(    #    重建缓冲区
                        max_len=circular_buffer.max_length,
                        batch_size=circular_buffer.batch_size,
                        device=circular_buffer.device,
                    )
                    circular_buffer.append(obs)

                if term_cfg.flatten_history_dim:
                    group_obs[term_name] = circular_buffer.buffer.reshape(self._env.num_envs, -1)   # [N, 3, 7] → [N, 21]
                else:
                    group_obs[term_name] = circular_buffer.buffer   # [N, 3, 7]
                '''
                flatten_history_dim：
                    # 不展平: [N, 3, 7] → 策略网络看到 3×7 的 2D 矩阵
                    # 展平:   [N, 21]  → 策略网络看到 21 维向量（更简单但丢失时间结构）
                '''
            else:
                group_obs[term_name] = obs  # 无历史，直接存

        '''
            模式	返回值	                            示例
            拼接	Tensor[N, 15]	                    `[joint_pos(7)
            字典	{"joint_pos": Tensor[N,7], ...}	    各 term 独立
        '''
        # concatenate all observations in the group together
        if self._group_obs_concatenate[group_name]:
            # set the concatenate dimension, account for the batch dimension if positive dimension is given
            return torch.cat(list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name])
        else:
            return group_obs

    '''
    嵌套字典推导式的观测配置导出
        ObservationManager.serialize() 把当前所有活跃的观测 term 配置序列化为纯字典，用于日志保存和 checkpoint 恢复。
        和其他 Manager 的 serialize 最大的不同是嵌套了两层：group 层和 term 层。
    '''
    def serialize(self) -> dict:
        """Serialize the observation term configurations for all active groups.

        Returns:
            A dictionary where each group name maps to its serialized observation term configurations.
        """
        """为所有活跃组进行观测项配置的序列化。

        返回：
            一个字典，其中每个组名单将其序列化观测项配置映射。
        """
        output = {
            group_name: {
                term_name: (
                    term_cfg.func.serialize()
                    if isinstance(term_cfg.func, ManagerTermBase)
                    else {"cfg": class_to_dict(term_cfg)}
                )
                for term_name, term_cfg in zip(
                    self._group_obs_term_names[group_name],
                    self._group_obs_term_cfgs[group_name],
                )
            }
            for group_name in self.active_terms.keys()
        }

        return output
    '''
    输出结构
        {
            "policy": {
                "joint_pos": {"cfg": {"func": "isaaclab.xxx:joint_pos", "params": {...}}},
                "joint_vel": {"cfg": {"func": "isaaclab.xxx:joint_vel", "params": {...}}},
                "actions":   {"cfg": {"func": "isaaclab.xxx:actions",   "params": {...}}},
            },
            "critic": {
                ...
            },
        }
    '''

    """
    Helper functions.
    """
    """辅助函数。
    """

    def _prepare_terms(self):
        """Prepares a list of observation terms functions."""
        """编制观测项功能列表。"""

        '''
        整体架构
            _prepare_terms()
                │
                ├── 阶段 0: 初始化全局容器
                │     9 个 group 级字典 + 1 个 class 实例列表
                │
                ├── 前置条件: 仿真必须已播放
                │     (需要调用 func 来获取观测维度)
                │
                └── 双层遍历:
                    ├── 外层: 遍历 group_cfg_items
                    │     ├── 校验: ObservationGroupCfg
                    │     ├── 初始化 group 的子列表
                    │     ├── 读取 concatenate 设置
                    │     │
                    │     └── 内层: 遍历 term_cfg_items (各 term)
                    │           ├── 过滤非 term 字段
                    │           ├── 校验: ObservationTermCfg
                    │           ├── _resolve_common_term_cfg (min_argc=1)
                    │           ├── group 级覆盖 (noise/history)
                    │           ├── 调用 func 获取 obs_dims
                    │           ├── 验证 scale 维度
                    │           ├── 初始化 modifiers
                    │           ├── 初始化 noise model
                    │           ├── 创建 history buffer
                    │           └── 存储 term dims / class term cfgs
                    │
                    └── 存储 group 的 history buffers
        '''

        '''
        容器	                        结构	                                        用途
        _group_obs_term_names	        {"policy": ["joint_pos", ...]}	            每组有哪些 term
        _group_obs_term_dim	            {"policy": [(7,), (7,), ...]}	            每个 term 的形状（剥离 batch）
        _group_obs_term_cfgs	        {"policy": [ObsTermCfg, ...]}	            每个 term 的配置对象
        _group_obs_class_term_cfgs	    同上	                                    类实现的 term（reset() 时额外处理）
        _group_obs_concatenate	        {"policy": True}	                        是否拼接
        _group_obs_concatenate_dim	    {"policy": 1}	                            沿哪个维度拼接
        _group_obs_term_history_buffer	{"policy": {"joint_pos": CircularBuffer}}	历史观测缓冲区
        _group_obs_class_instances	    [...]	                                    类形式的 modifier/noise model 实例
        '''
        # create buffers to store information for each observation group
        # TODO: Make this more convenient by using data structures.
        self._group_obs_term_names: dict[str, list[str]] = dict()
        self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = dict()
        self._group_obs_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_concatenate: dict[str, bool] = dict()
        self._group_obs_concatenate_dim: dict[str, int] = dict()

        self._group_obs_term_history_buffer: dict[str, dict] = dict()
        # create a list to store classes instances, e.g., for modifiers and noise models
        # we store it as a separate list to only call reset on them and prevent unnecessary calls
        self._group_obs_class_instances: list[modifiers.ModifierBase | noise.NoiseModel] = list()

        # make sure the simulation is playing since we compute obs dims which needs asset quantities
        if not self._env.sim.is_playing():
            raise RuntimeError(
                "Simulation is not playing. Observation manager requires the simulation to be playing"
                " to compute observation dimensions. Please start the simulation before using the"
                " observation manager."
            )

        # check if config is dict already
        if isinstance(self.cfg, dict):
            group_cfg_items = self.cfg.items()
        else:
            group_cfg_items = self.cfg.__dict__.items()
        # iterate over all the groups
        for group_name, group_cfg in group_cfg_items:
            # check for non config
            if group_cfg is None:
                continue
            # check if the term is a curriculum term
            if not isinstance(group_cfg, ObservationGroupCfg):
                raise TypeError(
                    f"Observation group '{group_name}' is not of type 'ObservationGroupCfg'."
                    f" Received: '{type(group_cfg)}'."
                )
            # initialize list for the group settings
            self._group_obs_term_names[group_name] = list()
            self._group_obs_term_dim[group_name] = list()
            self._group_obs_term_cfgs[group_name] = list()
            self._group_obs_class_term_cfgs[group_name] = list()

            # history buffers
            group_entry_history_buffer: dict[str, CircularBuffer] = dict()

            # read common config for the group
            self._group_obs_concatenate[group_name] = group_cfg.concatenate_terms
            self._group_obs_concatenate_dim[group_name] = (
                group_cfg.concatenate_dim + 1 if group_cfg.concatenate_dim >= 0 else group_cfg.concatenate_dim
            )

            # check if config is dict already
            if isinstance(group_cfg, dict):
                term_cfg_items = group_cfg.items()
            else:
                term_cfg_items = group_cfg.__dict__.items()
            # iterate over all the terms in each group
            for term_name, term_cfg in term_cfg_items:
                # skip non-obs settings
                if term_name in [
                    "enable_corruption",
                    "concatenate_terms",
                    "history_length",
                    "flatten_history_dim",
                    "concatenate_dim",
                ]:
                    continue
                # check for non config
                if term_cfg is None:
                    continue
                if not isinstance(term_cfg, ObservationTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type ObservationTermCfg."
                        f" Received: '{type(term_cfg)}'."
                    )
                # resolve common terms in the config
                self._resolve_common_term_cfg(f"{group_name}/{term_name}", term_cfg, min_argc=1)

                # check noise settings
                if not group_cfg.enable_corruption:
                    term_cfg.noise = None
                # check group history params and override terms
                if group_cfg.history_length is not None:
                    term_cfg.history_length = group_cfg.history_length
                    term_cfg.flatten_history_dim = group_cfg.flatten_history_dim
                # add term config to list to list
                self._group_obs_term_names[group_name].append(term_name)
                self._group_obs_term_cfgs[group_name].append(term_cfg)

                # call function the first time to fill up dimensions
                obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)

                # if scale is set, check if single float or tuple
                if term_cfg.scale is not None:
                    if not isinstance(term_cfg.scale, (float, int, tuple)):
                        raise TypeError(
                            f"Scale for observation term '{term_name}' in group '{group_name}'"
                            f" is not of type float, int or tuple. Received: '{type(term_cfg.scale)}'."
                        )
                    if isinstance(term_cfg.scale, tuple) and len(term_cfg.scale) != obs_dims[1]:
                        raise ValueError(
                            f"Scale for observation term '{term_name}' in group '{group_name}'"
                            f" does not match the dimensions of the observation. Expected: {obs_dims[1]}"
                            f" but received: {len(term_cfg.scale)}."
                        )

                    # cast the scale into torch tensor
                    term_cfg.scale = torch.tensor(term_cfg.scale, dtype=torch.float, device=self._env.device)

                # prepare modifiers for each observation
                if term_cfg.modifiers is not None:
                    # initialize list of modifiers for term
                    for mod_cfg in term_cfg.modifiers:
                        # check if class modifier and initialize with observation size when adding
                        if isinstance(mod_cfg, modifiers.ModifierCfg):
                            # to list of modifiers
                            if inspect.isclass(mod_cfg.func):
                                if not issubclass(mod_cfg.func, modifiers.ModifierBase):
                                    raise TypeError(
                                        f"Modifier function '{mod_cfg.func}' for observation term '{term_name}'"
                                        f" is not a subclass of 'ModifierBase'. Received: '{type(mod_cfg.func)}'."
                                    )
                                mod_cfg.func = mod_cfg.func(cfg=mod_cfg, data_dim=obs_dims, device=self._env.device)

                                # add to list of class modifiers
                                self._group_obs_class_instances.append(mod_cfg.func)
                        else:
                            raise TypeError(
                                f"Modifier configuration '{mod_cfg}' of observation term '{term_name}' is not of"
                                f" required type ModifierCfg, Received: '{type(mod_cfg)}'"
                            )

                        # check if function is callable
                        if not callable(mod_cfg.func):
                            raise AttributeError(
                                f"Modifier '{mod_cfg}' of observation term '{term_name}' is not callable."
                                f" Received: {mod_cfg.func}"
                            )

                        # check if term's arguments are matched by params
                        term_params = list(mod_cfg.params.keys())
                        args = inspect.signature(mod_cfg.func).parameters
                        args_with_defaults = [arg for arg in args if args[arg].default is not inspect.Parameter.empty]
                        args_without_defaults = [arg for arg in args if args[arg].default is inspect.Parameter.empty]
                        args = args_without_defaults + args_with_defaults
                        # ignore first two arguments for env and env_ids
                        # Think: Check for cases when kwargs are set inside the function?
                        if len(args) > 1:
                            if set(args[1:]) != set(term_params + args_with_defaults):
                                raise ValueError(
                                    f"Modifier '{mod_cfg}' of observation term '{term_name}' expects"
                                    f" mandatory parameters: {args_without_defaults[1:]}"
                                    f" and optional parameters: {args_with_defaults}, but received: {term_params}."
                                )

                # prepare noise model classes
                if term_cfg.noise is not None and isinstance(term_cfg.noise, noise.NoiseModelCfg):
                    noise_model_cls = term_cfg.noise.class_type
                    if not issubclass(noise_model_cls, noise.NoiseModel):
                        raise TypeError(
                            f"Class type for observation term '{term_name}' NoiseModelCfg"
                            f" is not a subclass of 'NoiseModel'. Received: '{type(noise_model_cls)}'."
                        )
                    # initialize func to be the noise model class instance
                    term_cfg.noise.func = noise_model_cls(
                        term_cfg.noise, num_envs=self._env.num_envs, device=self._env.device
                    )
                    self._group_obs_class_instances.append(term_cfg.noise.func)

                # create history buffers and calculate history term dimensions
                if term_cfg.history_length > 0:
                    group_entry_history_buffer[term_name] = CircularBuffer(
                        max_len=term_cfg.history_length, batch_size=self._env.num_envs, device=self._env.device
                    )
                    old_dims = list(obs_dims)
                    old_dims.insert(1, term_cfg.history_length)
                    obs_dims = tuple(old_dims)
                    if term_cfg.flatten_history_dim:
                        obs_dims = (obs_dims[0], np.prod(obs_dims[1:]))

                self._group_obs_term_dim[group_name].append(obs_dims[1:])

                # add term in a separate list if term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_obs_class_term_cfgs[group_name].append(term_cfg)
                    # call reset (in-case above call to get obs dims changed the state)
                    term_cfg.func.reset()
            # add history buffers for each group
            self._group_obs_term_history_buffer[group_name] = group_entry_history_buffer
