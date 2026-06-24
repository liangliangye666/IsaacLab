# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed because we concatenate int and torch.Tensor in the type hints
from __future__ import annotations

from collections.abc import Sequence

import torch

from .circular_buffer import CircularBuffer


class DelayBuffer:
    """Delay buffer that allows retrieving stored data with delays.

    This class uses a batched circular buffer to store input data. Different to a standard circular buffer,
    which uses the LIFO (last-in-first-out) principle to retrieve the data, the delay buffer class allows
    retrieving data based on the lag set by the user. For instance, if the delay set inside the buffer
    is 1, then the second last entry from the stream is retrieved. If it is 2, then the third last entry
    and so on.

    The class supports storing a batched tensor data. This means that the shape of the appended data
    is expected to be (batch_size, ...), where the first dimension is the batch dimension. Correspondingly,
    the delay can be set separately for each batch index. If the requested delay is larger than the current
    length of the underlying buffer, the most recent entry is returned.

    .. note::
        By default, the delay buffer has no delay, meaning that the data is returned as is.
    """
    """延迟缓冲器，允许随着延迟获取存储的数据。

    该类使用批量循环缓冲来存储输入数据。
    与标准循环缓冲器不同，使用LIFO根据用户设置的延迟，延误缓冲类可以根据用户设置的延迟检索数据。
    例如，如果缓冲器内设置的延迟为1，则从流中获取第二次最后的输入。
    如果是2，那么第三个是最后一个输入，等等等。

    该类支持存储批量数数据。
    这意味着附加数据的形状预计是 (batch_size， ...)，其中第一个维度是批量维度。
    根据此，每批量索引的延迟可以单独设置。
    如果要求的延迟超过底层缓冲的当前长度，则返回最新的输入。

    .. 说明::
        默认情况下，延迟缓冲器没有延迟，这意味着数据会像现在一样返回。
    """

    def __init__(self, history_length: int, batch_size: int, device: str):
        """Initialize the delay buffer.

        Args:
            history_length: The history of the buffer, i.e., the number of time steps in the past that the data
                will be buffered. It is recommended to set this value equal to the maximum time-step lag that
                is expected. The minimum acceptable value is zero, which means only the latest data is stored.
            batch_size: The batch dimension of the data.
            device: The device used for processing.
        """
        """启动延迟缓冲器。

        参数：
            history_length: 缓冲器的历史，i.e.，数据将被缓冲的过去时间步骤数。
                            建议设置这个值等于预期的最大时间步骤延迟。
                            最低可接受值为零，这意味着只存储最新数据。
            batch_size: 数据的批量尺寸。
            device: 用于加工的装置。
        """
        # set the parameters
        self._history_length = max(0, history_length)

        # the buffer size: current data plus the history length
        self._circular_buffer = CircularBuffer(self._history_length + 1, batch_size, device)

        # the minimum and maximum lags across all batch indices.
        self._min_time_lag = 0
        self._max_time_lag = 0
        # the lags for each batch index.
        self._time_lags = torch.zeros(batch_size, dtype=torch.int, device=device)

    """
    Properties.
    """
    """属性。
    """

    @property
    def batch_size(self) -> int:
        """The batch size of the ring buffer."""
        """环境器的批量。"""
        return self._circular_buffer.batch_size

    @property
    def device(self) -> str:
        """The device used for processing."""
        """用于加工的装置。"""
        return self._circular_buffer.device

    @property
    def history_length(self) -> int:
        """The history length of the delay buffer.

        If zero, only the latest data is stored. If one, the latest and the previous data are stored, and so on.
        """
        """延迟缓冲器的历史长度。

        如果是零，只有最新的数据才能存储。
        如果一个，最新的和以前的数据存储，
        """
        return self._history_length

    @property
    def min_time_lag(self) -> int:
        """Minimum amount of time steps that can be delayed.

        This value cannot be negative or larger than :attr:`max_time_lag`.
        """
        """最少可延迟的时间步骤。

        这一值不能为负或超过:attr:`max_time_lag`。
        """
        return self._min_time_lag

    @property
    def max_time_lag(self) -> int:
        """Maximum amount of time steps that can be delayed.

        This value cannot be greater than :attr:`history_length`.
        """
        """最多可延迟的时间步骤。

        这一值不能超过:attr:`history_length`。
        """
        return self._max_time_lag

    @property
    def time_lags(self) -> torch.Tensor:
        """The time lag across each batch index.

        The shape of the tensor is (batch_size, ). The value at each index represents the delay for that index.
        This value is used to retrieve the data from the buffer.
        """
        """每个批量索引的时间延误。

        子的形状是 (batch_size， )。
        每个索引的值代表该索引的延迟。
        这一值用于从缓冲器中获取数据。
        """
        return self._time_lags

    """
    Operations.
    """
    """操作。
    """

    def set_time_lag(self, time_lag: int | torch.Tensor, batch_ids: Sequence[int] | None = None):
        """Sets the time lag for the delay buffer across the provided batch indices.

        Args:
            time_lag: The desired delay for the buffer.

              * If an integer is provided, the same delay is set for the provided batch indices.
              * If a tensor is provided, the delay is set for each batch index separately. The shape of the tensor
                should be (len(batch_ids),).

            batch_ids: The batch indices for which the time lag is set. Default is None, which sets the time lag
                for all batch indices.

        Raises:
            TypeError: If the type of the :attr:`time_lag` is not int or integer tensor.
            ValueError: If the minimum time lag is negative or the maximum time lag is larger than the history length.
        """
        """设置在提供的批量索引中延迟缓冲的时间延迟。

        参数：
            time_lag: 缓冲器所需的延迟。

              * 如果提供整数，则为提供批量索引设定相同的延迟。
              * 如果提供一个子，则每个批量索引的延迟是单独设置的。 tens子的形状应该是 (len(batch_ids)，)。

            batch_ids: 时间延误设置的批量索引。
                       默认是None，
                for all batch indices.

        异常：
            TypeError: 如果:attr:`time_lag`的类型不是int或整数子。
            ValueError: 如果最小时间延迟是负值的，或者最大时间延迟超过历史长度。
        """
        # resolve batch indices
        if batch_ids is None:
            batch_ids = slice(None)

        # parse requested time_lag
        if isinstance(time_lag, int):
            # set the time lags across provided batch indices
            self._time_lags[batch_ids] = time_lag
        elif isinstance(time_lag, torch.Tensor):
            # check valid dtype for time_lag: must be int or long
            if time_lag.dtype not in [torch.int, torch.long]:
                raise TypeError(f"Invalid dtype for time_lag: {time_lag.dtype}. Expected torch.int or torch.long.")
            # set the time lags
            self._time_lags[batch_ids] = time_lag.to(device=self.device)
        else:
            raise TypeError(f"Invalid type for time_lag: {type(time_lag)}. Expected int or integer tensor.")

        # compute the min and max time lag
        self._min_time_lag = int(torch.min(self._time_lags).item())
        self._max_time_lag = int(torch.max(self._time_lags).item())
        # check that time_lag is feasible
        if self._min_time_lag < 0:
            raise ValueError(f"The minimum time lag cannot be negative. Received: {self._min_time_lag}")
        if self._max_time_lag > self._history_length:
            raise ValueError(
                f"The maximum time lag cannot be larger than the history length. Received: {self._max_time_lag}"
            )

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the data in the delay buffer at the specified batch indices.

        Args:
            batch_ids: Elements to reset in the batch dimension. Default is None, which resets all the batch indices.
        """
        """在延迟缓冲器中的数据重置到指定批量索引。

        参数：
            batch_ids: 在批量维度中重置元素。
                       默认是None，它重置了所有批量索引。
        """
        self._circular_buffer.reset(batch_ids)

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        """Append the input data to the buffer and returns a stale version of the data based on time lag delay.

        If the requested delay is larger than the number of buffered data points since the last reset,
        the function returns the latest data. For instance, if the delay is set to 2 and only one data point
        is stored in the buffer, the function will return the latest data. If the delay is set to 2 and three
        data points are stored, the function will return the first data point.

        Args:
           data: The input data. Shape is (batch_size, ...).

        Returns:
            The delayed version of the data from the stored buffer. Shape is (batch_size, ...).
        """
        """将输入数据添加到缓冲器中，并返回基于时间延迟的数据版本。

        如果要求的延迟超过自上一次重置以来缓冲数据点数量，函数将返回最新数据。
        例如，如果延迟设置为2并且只有一个数据点存储在缓冲器中，函数将返回最新数据。
        如果延误设置为2个数据点，并且存储了3个数据点，函数将返回第一个数据点。

        参数：
           data: 输入数据。
                 形状是 (batch_size， ...)。

        返回：
            存储缓冲器中的数据的延迟版本。
            形状是 (batch_size， ...)。
        """
        # add the new data to the last layer
        self._circular_buffer.append(data)
        # return output
        delayed_data = self._circular_buffer[self._time_lags]
        return delayed_data.clone()
