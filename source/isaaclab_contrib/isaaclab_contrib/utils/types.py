# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for multirotor-specific data types.

This module defines data container classes used for passing multirotor-specific
information between components (e.g., between action terms and actuator models).
"""

from __future__ import annotations
"""对多轮机特定数据类型的子模块。

该模块定义了用于组件之间传输多轮机特定信息的数据容器类 (e.g.，操作条件和执行器模型之间)。
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class MultiRotorActions:
    """Data container to store multirotor thruster actions.

    This dataclass is used to pass thrust commands and thruster indices between
    components in the multirotor control pipeline. It is primarily used internally
    by the :class:`~isaaclab_contrib.assets.Multirotor` class to communicate with
    :class:`~isaaclab_contrib.actuators.Thruster` actuator models.

    The container supports partial actions by allowing specification of which
    thrusters the actions apply to through the :attr:`thruster_indices` field.

    Attributes:
        thrusts: Thrust values for the specified thrusters. Shape is typically
            ``(num_envs, num_selected_thrusters)``.
        thruster_indices: Indices of thrusters that the thrust values apply to.
            Can be a tensor of indices, a sequence, a slice, or None for all thrusters.

    Example:
        .. code-block:: python

            # Create actions for all thrusters
            actions = MultiRotorActions(
                thrusts=torch.ones(num_envs, 4) * 5.0,
                thruster_indices=slice(None),  # All thrusters
            )

            # Create actions for specific thrusters
            actions = MultiRotorActions(
                thrusts=torch.tensor([[6.0, 7.0]]),
                thruster_indices=[0, 2],  # Only thrusters 0 and 2
            )

    Note:
        If both fields are ``None``, no action is taken. This is useful for
        conditional action application.

    .. seealso::
        - :class:`~isaaclab.utils.types.ArticulationActions`: Similar container for joint actions
        - :class:`~isaaclab_contrib.actuators.Thruster`: Thruster actuator that consumes these actions
    """
    """数据容器用于存储多轮驱动器操作。

    该数据类用于在多轮机控制管道中的组件之间传递推力命令和推力索引。
    它主要由:class:`~isaaclab_contrib.assets.Multirotor`类内部用于与:class:`~isaaclab_contrib.actuators.Thruster`执
    行器模型通信。

    容器支持部分操作，通过 :attr:`thruster_indices` 字段允许通过哪些推进器进行操作的规范。

    属性：
        thrusts: 指定驱动器的推力值。
                 形状通常是``(num_envs， num_selected_thrusters)``。
        thruster_indices: 适用于推力值的推进器指标。
                          对于所有推进器来说，它可以是索引数，序列，片段或None。

    示例：
        .. code-block:: python

            # Create actions for all thrusters
            actions = MultiRotorActions(
                thrusts=torch.ones(num_envs, 4) * 5.0,
                thruster_indices=slice(None),  # All thrusters
            )

            # Create actions for specific thrusters
            actions = MultiRotorActions(
                thrusts=torch.tensor([[6.0, 7.0]]),
                thruster_indices=[0, 2],  # Only thrusters 0 and 2
            )

    说明：
        如果两个字段都是``None``，则不会采取任何动作。
        这对于有条件动作的应用是有用的。

    ..
    查看:
        - :class:`~isaaclab.utils.types.ArticulationActions`:联合动作的类似容器
        - :class:`~isaaclab_contrib.actuators.Thruster`:使用这些动作的推进动机
    """

    thrusts: torch.Tensor | None = None
    """Thrust values for the multirotor thrusters.

    Shape: ``(num_envs, num_thrusters)`` or ``(num_envs, num_selected_thrusters)``

    The units depend on the actuator model configuration:
        - For force-based control: Newtons (N)
        - For RPS-based control: Revolutions per second (1/s)

    If ``None``, no thrust commands are specified.
    """
    """多轮驱动器的推力值。

    Shape: ``(num_envs， num_thrusters)``或``(num_envs， num_selected_thrusters)``

    单元取决于动机模型配置:
        - 用于基于力控制:纽顿 (N)
        - 对于基于RPS的控制:每秒旋转 (1/s)

    如果 ``None``，没有指定推力命令。
    """

    thruster_indices: torch.Tensor | Sequence[int] | slice | None = None
    """Indices of thrusters that the thrust values apply to.

    This field specifies which thrusters the :attr:`thrusts` values correspond to.
    It can be:
        - A torch.Tensor of integer indices: ``torch.tensor([0, 2, 3])``
        - A sequence of integers: ``[0, 2, 3]``
        - A slice: ``slice(None)`` for all thrusters, ``slice(0, 2)`` for first two
        - ``None``: Defaults to all thrusters

    Using a slice is more efficient for contiguous thruster ranges as it avoids
    creating intermediate index tensors.

    Example:
        .. code-block:: python

            # All thrusters (most efficient)
            thruster_indices = slice(None)

            # First two thrusters
            thruster_indices = slice(0, 2)

            # Specific thrusters
            thruster_indices = [0, 2, 3]
    """
    """适用于推力值的推进器指标。

    这段指明:attr:`thrusts`值相应的推进器。
    它可能是:
        - 整数索引的torch.Tensor:``torch.tensor([0， 2， 3])``
        - 整数序列:``[0， 2， 3]``
        - 一片:所有推进器的``slice(None)``，前两个的``slice(0， 2)``
        - ``None``:所有推进器的默认故障

    使用切片对连接推进器范围更有效，因为它避免创建中间索引位器。

    示例：
        .. code-block:: python

            # All thrusters (most efficient)
            thruster_indices = slice(None)

            # First two thrusters
            thruster_indices = slice(0, 2)

            # Specific thrusters
            thruster_indices = [0, 2, 3]
    """
