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
            if self._group_obs_concatenate[group_name]:
                try:
                    term_dims = torch.stack([torch.tensor(dims, device="cpu") for dims in group_term_dims], dim=0)
                    if len(term_dims.shape) > 1:
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
            else:
                self._group_obs_dim[group_name] = group_term_dims

        # Stores the latest observations.
        self._obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None

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

        if self._obs_buffer is None:
            self.compute()
        obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = self._obs_buffer

        for group_name, _ in self._group_obs_dim.items():
            if not self.group_obs_concatenate[group_name]:
                for name, term in obs_buffer[group_name].items():
                    terms.append((group_name + "-" + name, term[env_idx].cpu().tolist()))
                continue

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

    """
    Properties.
    """
    """属性。
    """

    @property
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active observation terms in each group.

        The keys are the group names and the values are the list of observation term names in the group.
        """
        """每组的活跃观测项名称。

        关键是组名，值是组中的观测项名单。
        """
        return self._group_obs_term_names

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
            for term_cfg in group_cfg:
                term_cfg.func.reset(env_ids=env_ids)
            # reset terms with history
            for term_name in self._group_obs_term_names[group_name]:
                if term_name in self._group_obs_term_history_buffer[group_name]:
                    self._group_obs_term_history_buffer[group_name][term_name].reset(batch_ids=env_ids)
        # call all modifiers that are classes
        for mod in self._group_obs_class_instances:
            mod.reset(env_ids=env_ids)

        # nothing to log here
        return {}

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

        # evaluate terms: compute, add noise, clip, scale, custom modifiers
        for term_name, term_cfg in obs_terms:
            # compute term's value
            obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
            # apply post-processing
            if term_cfg.modifiers is not None:
                for modifier in term_cfg.modifiers:
                    obs = modifier.func(obs, **modifier.params)
            if isinstance(term_cfg.noise, noise.NoiseCfg):
                obs = term_cfg.noise.func(obs, term_cfg.noise)
            elif isinstance(term_cfg.noise, noise.NoiseModelCfg) and term_cfg.noise.func is not None:
                obs = term_cfg.noise.func(obs)
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale is not None:
                obs = obs.mul_(term_cfg.scale)
            # Update the history buffer if observation term has history enabled
            if term_cfg.history_length > 0:
                circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                if update_history:
                    circular_buffer.append(obs)
                elif circular_buffer._buffer is None:
                    # because circular buffer only exits after the simulation steps,
                    # this guards history buffer from corruption by external calls before simulation start
                    circular_buffer = CircularBuffer(
                        max_len=circular_buffer.max_length,
                        batch_size=circular_buffer.batch_size,
                        device=circular_buffer.device,
                    )
                    circular_buffer.append(obs)

                if term_cfg.flatten_history_dim:
                    group_obs[term_name] = circular_buffer.buffer.reshape(self._env.num_envs, -1)
                else:
                    group_obs[term_name] = circular_buffer.buffer
            else:
                group_obs[term_name] = obs

        # concatenate all observations in the group together
        if self._group_obs_concatenate[group_name]:
            # set the concatenate dimension, account for the batch dimension if positive dimension is given
            return torch.cat(list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name])
        else:
            return group_obs

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

    """
    Helper functions.
    """
    """辅助函数。
    """

    def _prepare_terms(self):
        """Prepares a list of observation terms functions."""
        """编制观测项功能列表。"""
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
