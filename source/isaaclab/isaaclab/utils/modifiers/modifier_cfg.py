# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

import torch

from isaaclab.utils import configclass

from . import modifier


@configclass
class ModifierCfg:
    """Configuration parameters modifiers"""
    """配置参数修改器"""

    func: Callable[..., torch.Tensor] = MISSING
    """Function or callable class used by modifier.

    The function must take a torch tensor as the first argument. The remaining arguments are specified
    in the :attr:`params` attribute.

    It also supports `callable classes <https://docs.python.org/3/reference/datamodel.html#object.__call__>`_,
    i.e. classes that implement the ``__call__()`` method. In this case, the class should inherit from the
    :class:`ModifierBase` class and implement the required methods.
    """
    """修改器使用的函数或可调用类。

    函数必须作为第一个参数使用火。
    其他参数在:attr:`params`属性中指定。

    它还支持`callable classes <https://docs.python.org/3/reference/datamodel.html#object.__call__>`_，
    i.e.实施的类``__call__()``方法。
    在这种情况下，该类应继承:class:`ModifierBase`类，并实施所需的方法。
    """

    params: dict[str, Any] = dict()
    """The parameters to be passed to the function or callable class as keyword arguments. Defaults to
    an empty dictionary."""
    """作为关键字参数将将参数传递到函数或可调用类。
    默认的空白字典。
    """


@configclass
class DigitalFilterCfg(ModifierCfg):
    """Configuration parameters for a digital filter modifier.

    For more information, please check the :class:`DigitalFilter` class.
    """
    """数字过器修改器的配置参数

    更多信息请查看:class:`DigitalFilter`类。
    """

    func: type[modifier.DigitalFilter] = modifier.DigitalFilter
    """The digital filter function to be called for applying the filter."""
    """应调用的数字过器功能。"""

    A: list[float] = MISSING
    """The coefficients corresponding the the filter's response to past outputs.

    These correspond to the weights of the past outputs of the filter. The first element is the coefficient
    for the output at the previous time step, the second element is the coefficient for the output at two
    time steps ago, and so on.

    It is the denominator coefficients of the transfer function of the filter.
    """
    """符合过器对过去输出反应的系数。

    这些相应于过器过去输出的重量。
    第一个元素是系数
    for the output at the previous time step, the second element is the coefficient for the output at two
    在时间的步骤之前，等等。

    它是过器传输函数的命名系数。
    """

    B: list[float] = MISSING
    """The coefficients corresponding the the filter's response to current and past inputs.

    These correspond to the weights of the current and past inputs of the filter. The first element is the
    coefficient for the current input, the second element is the coefficient for the input at the previous
    time step, and so on.

    It is the numerator coefficients of the transfer function of the filter.
    """
    """符合过器对当前和过去输入的反应的系数。

    这些与过器的当前和过去输入的权重相符。
    第一个元素是电流输入的系数，第二个元素是前一个时间步骤的输入系数等。

    它是过器传输函数的数值系数。
    """


@configclass
class IntegratorCfg(ModifierCfg):
    """Configuration parameters for an integrator modifier.

    For more information, please check the :class:`Integrator` class.
    """
    """集成器修改器的配置参数

    更多信息请查看:class:`Integrator`类。
    """

    func: type[modifier.Integrator] = modifier.Integrator
    """The integrator function to be called for applying the integrator."""
    """应用集成器的集成器函数。"""

    dt: float = MISSING
    """The time step of the integrator."""
    """集成器的时间步骤。"""
