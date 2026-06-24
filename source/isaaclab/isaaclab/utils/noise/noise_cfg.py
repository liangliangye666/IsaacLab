# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

import torch

from isaaclab.utils import configclass

from . import noise_model


@configclass
class NoiseCfg:
    """Base configuration for a noise term."""
    """对于噪音项的基础配置。"""

    func: Callable[[torch.Tensor, NoiseCfg], torch.Tensor] = MISSING
    """The function to be called for applying the noise.

    Note:
        The shape of the input and output tensors must be the same.
    """
    """应调用噪音的功能。

    说明：
        输入和输出 tensor 的形状必须相同。
    """
    operation: Literal["add", "scale", "abs"] = "add"
    """The operation to apply the noise on the data. Defaults to "add"."""
    """对数据的噪音应用操作。
    默认的"添加"。
    """


@configclass
class ConstantNoiseCfg(NoiseCfg):
    """Configuration for an additive constant noise term."""
    """添加常量噪音项的配置。"""

    func = noise_model.constant_noise

    bias: torch.Tensor | float = 0.0
    """The bias to add. Defaults to 0.0."""
    """增加的偏见。
    默认为0.0。
    """


@configclass
class UniformNoiseCfg(NoiseCfg):
    """Configuration for a additive uniform noise term."""
    """对添加剂均噪音项的配置。"""

    func = noise_model.uniform_noise

    n_min: torch.Tensor | float = -1.0
    """The minimum value of the noise. Defaults to -1.0."""
    """噪音的最小值。
    设置为 -1.0。
    """
    n_max: torch.Tensor | float = 1.0
    """The maximum value of the noise. Defaults to 1.0."""
    """噪音的最大值。
    默认到1.0。
    """


@configclass
class GaussianNoiseCfg(NoiseCfg):
    """Configuration for an additive gaussian noise term."""
    """对添加加加斯语噪声的配置。"""

    func = noise_model.gaussian_noise

    mean: torch.Tensor | float = 0.0
    """The mean of the noise. Defaults to 0.0."""
    """这种噪音的平均值。
    默认为0.0。
    """
    std: torch.Tensor | float = 1.0
    """The standard deviation of the noise. Defaults to 1.0."""
    """噪音的标准偏差。
    默认到1.0。
    """


##
# Noise models
##


@configclass
class NoiseModelCfg:
    """Configuration for a noise model."""
    """对于噪音模型的配置。"""

    class_type: type = noise_model.NoiseModel
    """The class type of the noise model."""
    """噪音模型的类型。"""

    noise_cfg: NoiseCfg = MISSING
    """The noise configuration to use."""
    """使用的噪音配置。"""

    func: Callable[[torch.Tensor], torch.Tensor] | None = None
    """Function or callable class used by this noise model.

    The function must take a single `torch.Tensor` (the batch of observations) as input
    and return a `torch.Tensor` of the same shape with noise applied.

    It also supports `callable classes <https://docs.python.org/3/reference/datamodel.html#object.__call__>`_,
    i.e. classes that implement the ``__call__()`` method. In this case, the class should inherit from the
    :class:`NoiseModel` class and implement the required methods.

    This field is used internally by :class:ObservationManager and is not meant to be set directly.
    """
    """在此噪音模型中使用的函数或可调用类。

    函数必须以单个`torch.Tensor` (观测批量) 为输入，并以噪音的形式返回相同形状的`torch.Tensor`。

    它还支持`callable classes <https://docs.python.org/3/reference/datamodel.html#object.__call__>`_，
    i.e.实施的类``__call__()``方法。
    在这种情况下，该类应继承:class:`NoiseModel`类，并实施所需的方法。

    这个字段由:class:ObservationManager内部使用，并不是直接设置的。
    """


@configclass
class NoiseModelWithAdditiveBiasCfg(NoiseModelCfg):
    """Configuration for an additive gaussian noise with bias model."""
    """配置为具有偏差模型的加固高斯噪音。"""

    class_type: type = noise_model.NoiseModelWithAdditiveBias

    bias_noise_cfg: NoiseCfg = MISSING
    """The noise configuration for the bias.

    Based on this configuration, the bias is sampled at every reset of the noise model.
    """
    """对于偏差的噪音配置。

    根据这种配置，在噪音模型的每次重置时，
    """

    sample_bias_per_component: bool = True
    """Whether to sample a separate bias for each data component.

    Defaults to True.
    """
    """对于每个数据组件是否要采样单独的偏差。

    默认为 True。
    """
