# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .modifier_cfg import ModifierCfg


class ModifierBase(ABC):
    """Base class for modifiers implemented as classes.

    Modifiers implementations can be functions or classes. If a modifier is a class, it should
    inherit from this class and implement the required methods.

    A class implementation of a modifier can be used to store state information between calls.
    This is useful for modifiers that require stateful operations, such as rolling averages
    or delays or decaying filters.

    Example pseudo-code to create and use the class:

    .. code-block:: python

        from isaaclab.utils import modifiers

        # define custom keyword arguments to pass to ModifierCfg
        kwarg_dict = {"arg_1": VAL_1, "arg_2": VAL_2}

        # create modifier configuration object
        # func is the class name of the modifier and params is the dictionary of arguments
        modifier_config = modifiers.ModifierCfg(func=modifiers.ModifierBase, params=kwarg_dict)

        # define modifier instance
        my_modifier = modifiers.ModifierBase(cfg=modifier_config)

    """
    """作为类实现的修改器的基类。

    修改器实现可以是函数或类。
    如果修改器是类，则应继承该类并实施所需的方法。

    修改器的类实现可用于调用间存储状态信息。
    这对于需要状态操作的修改器有用，例如滚动平均值或延迟或腐蚀过器。

    创建和使用类型的假代码示例:

    .. code-block:: python

        from isaaclab.utils import modifiers

        # define custom keyword arguments to pass to ModifierCfg
        kwarg_dict = {"arg_1": VAL_1, "arg_2": VAL_2}

        # create modifier configuration object
        # func is the class name of the modifier and params is the dictionary of arguments
        modifier_config = modifiers.ModifierCfg(func=modifiers.ModifierBase, params=kwarg_dict)

        # define modifier instance
        my_modifier = modifiers.ModifierBase(cfg=modifier_config)
    """

    def __init__(self, cfg: ModifierCfg, data_dim: tuple[int, ...], device: str) -> None:
        """Initializes the modifier class.

        Args:
            cfg: Configuration parameters.
            data_dim: The dimensions of the data to be modified. First element is the batch size
                which usually corresponds to number of environments in the simulation.
            device: The device to run the modifier on.
        """
        """启动修改器类。

        参数：
            cfg: 配置参数
            data_dim: 要修改的数据的尺寸。
                      第一个元素是批量大小，通常与仿真环境数量相符。
            device: 调节器的设备。
        """
        self._cfg = cfg
        self._data_dim = data_dim
        self._device = device

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the Modifier.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        """修改器重置。

        参数：
            env_ids: 环境 ID。
                     在 None 中，默认情况下考虑所有环境。
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Abstract method for defining the modification function.

        Args:
            data: The data to be modified. Shape should match the data_dim passed during initialization.

        Returns:
            Modified data. Shape is the same as the input data.
        """
        """定义修改函数的抽象方法。

        参数：
            data: 要修改的数据。
                  形状应与启动过程中传递的data_dim相匹配。

        返回：
            修改数据。
            它的形状与输入数据相同。
        """
        raise NotImplementedError
