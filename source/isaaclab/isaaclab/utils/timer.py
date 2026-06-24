# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for a timer class that can be used for performance measurements."""

from __future__ import annotations
"""计时器类的子模块，可用于性能测量。"""

import time
from contextlib import ContextDecorator
from typing import Any, ClassVar


class TimerError(Exception):
    """A custom exception used to report errors in use of :class:`Timer` class."""
    """用于报告:class:`Timer`类的使用错误的定制例外。"""

    pass


class Timer(ContextDecorator):
    """A timer for performance measurements.

    A class to keep track of time for performance measurement.
    It allows timing via context managers and decorators as well.

    It uses the `time.perf_counter` function to measure time. This function
    returns the number of seconds since the epoch as a float. It has the
    highest resolution available on the system.

    As a regular object:

    .. code-block:: python

        import time

        from isaaclab.utils.timer import Timer

        timer = Timer()
        timer.start()
        time.sleep(1)
        print(1 <= timer.time_elapsed <= 2)  # Output: True

        time.sleep(1)
        timer.stop()
        print(2 <= stopwatch.total_run_time)  # Output: True

    As a context manager:

    .. code-block:: python

        import time

        from isaaclab.utils.timer import Timer

        with Timer() as timer:
            time.sleep(1)
            print(1 <= timer.time_elapsed <= 2)  # Output: True

    Reference: https://gist.github.com/sumeet/1123871
    """
    """计时器用于性能测量。

    一个用于追踪性能测量时间的课程。
    它还允许通过语境管理器和装饰师进行定时。

    它使用`time.perf_counter`函数来测量时间。
    这个函数将从时代以漂浮为数的秒数返回。
    它在系统中可用的最高分辨率。

    作为一个普通的对象:

    .. code-block:: python

        import time

        from isaaclab.utils.timer import Timer

        timer = Timer()
        timer.start()
        time.sleep(1)
        print(1 <= timer.time_elapsed <= 2)  # Output: True

        time.sleep(1)
        timer.stop()
        print(2 <= stopwatch.total_run_time)  # Output: True

    作为环境管理器:

    .. code-block:: python

        import time

        from isaaclab.utils.timer import Timer

        with Timer() as timer:
            time.sleep(1)
            print(1 <= timer.time_elapsed <= 2)  # Output: True

    Reference: https://gist.github.com/sumeet/1123871
    """

    timing_info: ClassVar[dict[str, float]] = dict()
    """Dictionary for storing the elapsed time per timer instances globally.

    This dictionary logs the timer information. The keys are the names given to the timer class
    at its initialization. If no :attr:`name` is passed to the constructor, no time
    is recorded in the dictionary.
    """
    """全球范围内存储每次计时器实例的时间。

    这个字典记录了计时器信息。
    按时器类在启动时所给出的名称。
    如果没有传递:attr:`name`给构造器，字典中没有记录时间。
    """

    def __init__(self, msg: str | None = None, name: str | None = None):
        """Initializes the timer.

        Args:
            msg: The message to display when using the timer
                class in a context manager. Defaults to None.
            name: The name to use for logging times in a global
                dictionary. Defaults to None.
        """
        """启动计时器。

        参数：
            msg: 使用计时器显示的消息
                class in a context manager. Defaults to None.
            name: 在全球字典中用于记录时间。
                  默认为 None。
        """
        self._msg = msg
        self._name = name
        self._start_time = None
        self._stop_time = None
        self._elapsed_time = None

    def __str__(self) -> str:
        """A string representation of the class object.

        Returns:
            A string containing the elapsed time.
        """
        """类对象的字符串表示。

        返回：
            一个包含过去的时间的字符串。
        """
        return f"{self.time_elapsed:0.6f} seconds"

    """
    Properties
    """
    """产品
    """

    @property
    def time_elapsed(self) -> float:
        """The number of seconds that have elapsed since this timer started timing.

        Note:
            This is used for checking how much time has elapsed while the timer is still running.
        """
        """自此时刻开始时刻以来的数秒。

        说明：
            这用于检查计时器仍在运行期间的时间。
        """
        return time.perf_counter() - self._start_time

    @property
    def total_run_time(self) -> float:
        """The number of seconds that elapsed from when the timer started to when it ended."""
        """计时器开始到结束时的数秒。"""
        return self._elapsed_time

    """
    Operations
    """
    """运营
    """

    def start(self):
        """Start timing."""
        """开始时间。"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop timing."""
        """停止时间。"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        self._stop_time = time.perf_counter()
        self._elapsed_time = self._stop_time - self._start_time
        self._start_time = None

        if self._name:
            Timer.timing_info[self._name] = self._elapsed_time

    """
    Context managers
    """
    """语境管理器
    """

    def __enter__(self) -> Timer:
        """Start timing and return this `Timer` instance."""
        """开始定时，然后返回这个`Timer`实例。"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any):
        """Stop timing."""
        """停止时间。"""
        self.stop()
        # print message
        if self._msg is not None:
            print(self._msg, f": {self._elapsed_time:0.6f} seconds")

    """
    Static Methods
    """
    """静态方法
    """

    @staticmethod
    def get_timer_info(name: str) -> float:
        """Retrieves the time logged in the global dictionary
            based on name.

        Args:
            name: Name of the the entry to be retrieved.

        Raises:
            TimerError: If name doesn't exist in the log.

        Returns:
            A float containing the time logged if the name exists.
        """
        """根据名字恢复了全球字典中登录的时间。

        参数：
            name: 获取的输入名称。

        异常：
            TimerError: 如果名字不在日志中。

        返回：
            如果名称存在，则包含记录的时间。
        """
        if name not in Timer.timing_info:
            raise TimerError(f"Timer {name} does not exist")
        return Timer.timing_info.get(name)
