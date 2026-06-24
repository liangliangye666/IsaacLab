# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .modifier_base import ModifierBase

if TYPE_CHECKING:
    from . import modifier_cfg

##
# Modifiers as functions
##


def scale(data: torch.Tensor, multiplier: float) -> torch.Tensor:
    """Scales input data by a multiplier.

    Args:
        data: The data to apply the scale to.
        multiplier: Value to scale input by.

    Returns:
        Scaled data. Shape is the same as data.
    """
    """通过乘法测量输入数据。

    参数：
        data: 适用于规模的数据。
        multiplier: 输入量值

    返回：
        规模数据。
        形状和数据是一样的。
    """
    return data * multiplier


def clip(data: torch.Tensor, bounds: tuple[float | None, float | None]) -> torch.Tensor:
    """Clips the data to a minimum and maximum value.

    Args:
        data: The data to apply the clip to.
        bounds: A tuple containing the minimum and maximum values to clip data to.
            If the value is None, that bound is not applied.

    Returns:
        Clipped data. Shape is the same as data.
    """
    """将数据裁剪到最低和最高值。

    参数：
        data: 应用该裁剪的数据。
        bounds: 一个包含裁剪数据的最小和最大值的图布。
                如果值为None，则不应使用该绑定。

    返回：
        裁剪数据。
        形状和数据是一样的。
    """
    return data.clip(min=bounds[0], max=bounds[1])


def bias(data: torch.Tensor, value: float) -> torch.Tensor:
    """Adds a uniform bias to the data.

    Args:
        data: The data to add bias to.
        value: Value of bias to add to data.

    Returns:
        Biased data. Shape is the same as data.
    """
    """增加一个统一的偏见数据。

    参数：
        data: 增加偏见的数据。
        value: 增加数据的偏见值。

    返回：
        有偏见的数据。
        形状和数据是一样的。
    """
    return data + value


##
# Sample of class based modifiers
##


class DigitalFilter(ModifierBase):
    r"""Modifier used to apply digital filtering to the input data.

    `Digital filters <https://en.wikipedia.org/wiki/Digital_filter>`_ are used to process discrete-time
    signals to extract useful parts of the signal, such as smoothing, noise reduction, or frequency separation.

    The filter can be implemented as a linear difference equation in the time domain. This equation
    can be used to calculate the output at each time-step based on the current and previous inputs and outputs.

    .. math::
         y_{i} = X B - Y A = \sum_{j=0}^{N} b_j x_{i-j} - \sum_{j=1}^{M} a_j y_{i-j}

    where :math:`y_{i}` is the current output of the filter. The array :math:`Y` contains previous
    outputs from the filter :math:`\{y_{i-j}\}_{j=1}^M` for :math:`M` previous time-steps. The array
    :math:`X` contains current :math:`x_{i}` and previous inputs to the filter
    :math:`\{x_{i-j}\}_{j=1}^N` for :math:`N` previous time-steps respectively.
    The filter coefficients :math:`A` and :math:`B` are used to design the filter. They are column vectors of
    length :math:`M` and :math:`N + 1` respectively.

    Different types of filters can be implemented by choosing different values for :math:`A` and :math:`B`.
    We provide some examples below.

    Examples
    ^^^^^^^^

    **Unit Delay Filter**

    A filter that delays the input signal by a single time-step simply outputs the previous input value.

    .. math:: y_{i} = x_{i-1}

    This can be implemented as a digital filter with the coefficients :math:`A = [0.0]` and :math:`B = [0.0, 1.0]`.

    **Moving Average Filter**

    A moving average filter is used to smooth out noise in a signal. It is similar to a low-pass filter
    but has a finite impulse response (FIR) and is non-recursive.

    The filter calculates the average of the input signal over a window of time-steps. The linear difference
    equation for a moving average filter is:

    .. math:: y_{i} = \frac{1}{N} \sum_{j=0}^{N} x_{i-j}

    This can be implemented as a digital filter with the coefficients :math:`A = [0.0]` and
    :math:`B = [1/N, 1/N, \cdots, 1/N]`.

    **First-order recursive low-pass filter**

    A recursive low-pass filter is used to smooth out high-frequency noise in a signal. It is a first-order
    infinite impulse response (IIR) filter which means it has a recursive component (previous output) in the
    linear difference equation.

    A first-order low-pass IIR filter has the difference equation:

    .. math:: y_{i} = \alpha y_{i-1} + (1-\alpha)x_{i}

    where :math:`\alpha` is a smoothing parameter between 0 and 1. Typically, the value of :math:`\alpha` is
    chosen based on the desired cut-off frequency of the filter.

    This filter can be implemented as a digital filter with the coefficients :math:`A = [-\alpha]` and
    :math:`B = [1 - \alpha]`.
    """
    """用于对输入数据进行数字过的修改器。

    `Digital filters
    <https://en.wikipedia.org/wiki/Digital_filter>`它们用于处理离散时间信号，以提取信号的有用部分，如平滑，噪音降低或频率分离。

    过器可以作为时间域的线性差异方程实现。
    这一方程可根据当前和以前的输入和输出计算每一步输出。

    .. math::
         y_{i} = X B - Y A = \sum_{j=0}^{N} b_j x_{i-j} - \sum_{j=1}^{M} a_j y_{i-j}

    where :数学:`y_{i}`是过器的电流输出。
           阵列:math:`Y`包含之前的
    过器的输出:`\{y_{i-j}\}_{j=1}^M`在数学上:`M`之前的时间步骤。
    阵列
    :math:`X`包含当前的:math:`x_{i}`和过器的前进输入
    :math:`\{x_{i-j}\}_{j=1}^N`为:数学:`N` 之前的时间步骤。
    为设计过器使用过器系数:math:`A`和:math:`B`。
    它们是
    length :分别是数学:`M`和数学:`N + 1`。

    通过选择:math:`A`和:math:`B`的不同值来实现不同的类型的过器。
    下面我们提供了一些例子。

    举个例子^^^^^^

    **单位延迟过器**

    一个延迟输入信号的过器仅仅输出了之前的输入值。

    .. math:: y_{i} = x_{i-1}

    这可以作为一个数字过器实现:math:`A = [0.0]`和:math:`B = [0.0， 1.0]`的系数。

    **移动平均过器**

    通过移动平均过器来缓解信号中的噪音。
    它类似于低通行过器，但具有有限的冲动反应 (FIR)，并且是非递归的。

    过器在时间步骤窗口内计算输入信号的平均值。
    移动平均过器的线性差异方程是:

    .. math:: y_{i} = \frac{1}{N} \sum_{j=0}^{N} x_{i-j}

    这可以作为一个数字过器实现:数学:`A = [0.0]`和
    :math:`B = [1/N， 1/N， \cdots， 1/N]`。

    **第一级递归低通道过器**

    通过低通道过器来平滑信号中的高频噪音。
    它是一个第一级无限冲动响应 (IIR) 过器，这意味着它在线性差异方程中具有递归组件 (前输出)。

    一级低通过IIR过器具有差异方程:

    .. math:: y_{i} = \alpha y_{i-1} + (1-\alpha)x_{i}

    where :数学:`\alpha`是0到1之间的平滑参数。
           通常，数值:`\alpha`是
    根据过器所需的切断频率选择。

    这种过器可以作为一个数字过器实现，具有系数:math:`A = [-\alpha]`和
    :math:`B = [1 - \alpha]`。
    """

    def __init__(self, cfg: modifier_cfg.DigitalFilterCfg, data_dim: tuple[int, ...], device: str):
        """Initializes digital filter.

        Args:
            cfg: Configuration parameters.
            data_dim: The dimensions of the data to be modified. First element is the batch size
                which usually corresponds to number of environments in the simulation.
            device: The device to run the modifier on.

        Raises:
            ValueError: If filter coefficients are None.
        """
        """启动数字过器。

        参数：
            cfg: 配置参数
            data_dim: 要修改的数据的尺寸。
                      第一个元素是批量大小，通常与仿真环境数量相符。
            device: 调节器的设备。

        异常：
            ValueError: 如果过系数是None。
        """
        # check that filter coefficients are not None
        if cfg.A is None or cfg.B is None:
            raise ValueError("Digital filter coefficients A and B must not be None. Please provide valid coefficients.")

        # initialize parent class
        super().__init__(cfg, data_dim, device)

        # assign filter coefficients and make sure they are column vectors
        self.A = torch.tensor(self._cfg.A, device=self._device).unsqueeze(1)
        self.B = torch.tensor(self._cfg.B, device=self._device).unsqueeze(1)

        # create buffer for input and output history
        self.x_n = torch.zeros(self._data_dim + (self.B.shape[0],), device=self._device)
        self.y_n = torch.zeros(self._data_dim + (self.A.shape[0],), device=self._device)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets digital filter history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        """重置数字过器历史。

        参数：
            env_ids: 环境 ID。
                     在 None 中，默认情况下考虑所有环境。
        """
        if env_ids is None:
            env_ids = slice(None)
        # reset history buffers
        self.x_n[env_ids] = 0.0
        self.y_n[env_ids] = 0.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Applies digital filter modification with a rolling history window inputs and outputs.

        Args:
            data: The data to apply filter to.

        Returns:
            Filtered data. Shape is the same as data.
        """
        """使用滚动历史窗口输入和输出的数字过器修改。

        参数：
            data: 应用过器的数据。

        返回：
            过数据。
            形状和数据是一样的。
        """
        # move history window for input
        self.x_n = torch.roll(self.x_n, shifts=1, dims=-1)
        self.x_n[..., 0] = data

        # calculate current filter value: y[i] = Y*A - X*B
        y_i = torch.matmul(self.x_n, self.B) - torch.matmul(self.y_n, self.A)
        y_i.squeeze_(-1)

        # move history window for output and add current filter value to history
        self.y_n = torch.roll(self.y_n, shifts=1, dims=-1)
        self.y_n[..., 0] = y_i

        return y_i


class Integrator(ModifierBase):
    r"""Modifier that applies a numerical forward integration based on a middle Reimann sum.

    An integrator is used to calculate the integral of a signal over time. The integral of a signal
    is the area under the curve of the signal. The integral can be approximated using numerical methods
    such as the `Riemann sum <https://en.wikipedia.org/wiki/Riemann_sum>`_.

    The middle Riemann sum is a method to approximate the integral of a function by dividing the area
    under the curve into rectangles. The height of each rectangle is the value of the function at the
    midpoint of the interval. The area of each rectangle is the width of the interval multiplied by the
    height of the rectangle.

    This integral method is useful for signals that are sampled at regular intervals. The integral
    can be written as:

    .. math::
        \int_{t_0}^{t_n} f(t) dt & \approx \int_{t_0}^{t_{n-1}} f(t) dt + \frac{f(t_{n-1}) + f(t_n)}{2} \Delta t

    where :math:`f(t)` is the signal to integrate, :math:`t_i` is the time at the i-th sample, and
    :math:`\Delta t` is the time step between samples.
    """
    """基于中部雷曼总数的数字前进整合的修改器。

    一个集成器用于时间计算信号的整体。
    信号的整体是信号曲线下的区域。
    总数可以通过数值方法进行近似计算，如`Riemann sum <https://en.wikipedia.org/wiki/Riemann_sum>`_。

    中部里曼总数是通过将曲线下面面积分为矩形来接近函数的整体的方法。
    每个矩形的高度是间隔中点函数的值。
    每个矩形的面积是间隔的宽度乘以矩形的高度。

    这种整体方法对于定期采样信号来说是有用的。
    整体可以写成:

    .. math::
        \int_{t_0}^{t_n} f(t) dt & \approx \int_{t_0}^{t_{n-1}} f(t) dt + \frac{f(t_{n-1}) + f(t_n)}{2} \Delta t

    where :数学:`f(t)`是集成的信号，`t_i`是第1样本的时间，
    :math:样本之间的时间步骤是`\Delta t`。
    """

    def __init__(self, cfg: modifier_cfg.IntegratorCfg, data_dim: tuple[int, ...], device: str):
        """Initializes the integrator configuration and state.

        Args:
            cfg: Integral parameters.
            data_dim: The dimensions of the data to be modified. First element is the batch size
                which usually corresponds to number of environments in the simulation.
            device: The device to run the modifier on.
        """
        """启动集成器配置和状态。

        参数：
            cfg: 整体参数。
            data_dim: 要修改的数据的尺寸。
                      第一个元素是批量大小，通常与仿真环境数量相符。
            device: 调节器的设备。
        """
        # initialize parent class
        super().__init__(cfg, data_dim, device)

        # assign buffer for integral and previous value
        self.integral = torch.zeros(self._data_dim, device=self._device)
        self.y_prev = torch.zeros(self._data_dim, device=self._device)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets integrator state to zero.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        """将集成器状态重置为零。

        参数：
            env_ids: 环境 ID。
                     在 None 中，默认情况下考虑所有环境。
        """
        if env_ids is None:
            env_ids = slice(None)
        # reset history buffers
        self.integral[env_ids] = 0.0
        self.y_prev[env_ids] = 0.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Applies integral modification to input data.

        Args:
            data: The data to integrate.

        Returns:
            Integral of input signal. Shape is the same as data.
        """
        """对输入数据进行整体修改。

        参数：
            data: 集成的数据。

        返回：
            输入信号的整体。
            形状和数据是一样的。
        """
        # integrate using middle Riemann sum
        self.integral += (data + self.y_prev) / 2 * self._cfg.dt
        # update previous value
        self.y_prev[:] = data

        return self.integral
