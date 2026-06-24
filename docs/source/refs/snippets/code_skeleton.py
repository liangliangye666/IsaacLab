# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import ClassVar

DEFAULT_TIMEOUT: int = 30
"""Default timeout for the task."""
"""默认的任务截止时间。"""

_MAX_RETRIES: int = 3  # private constant (note the underscore)
"""Maximum number of retries for the task."""
"""任务重复尝试的最大数量。"""


def run_task(task_name: str):
    """Run a task by name.

    Args:
        task_name: The name of the task to run.
    """
    """按名字执行任务。

    参数：
        task_name: 执行任务的名称。
    """
    print(f"Running task: {task_name}")


class TaskRunner:
    """Runs and manages tasks."""
    """运行和管理任务。"""

    DEFAULT_NAME: ClassVar[str] = "runner"
    """Default name for the runner."""
    """跑者的默认名称。"""

    _registry: ClassVar[dict] = {}
    """Registry of runners."""
    """跑者名单。"""

    def __init__(self, name: str):
        """Initialize the runner.

        Args:
            name: The name of the runner.
        """
        """启动运行器。

        参数：
            name: 跑者的名字。
        """
        self.name = name
        self._tasks = []  # private instance variable

    def __del__(self):
        """Clean up the runner."""
        """清理跑步机。"""
        print(f"Cleaning up {self.name}")

    def __repr__(self) -> str:
        return f"TaskRunner(name={self.name!r})"

    def __str__(self) -> str:
        return f"TaskRunner: {self.name}"

    """
    Properties.
    """
    """属性。
    """

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    """
    Operations.
    """
    """操作。
    """

    def initialize(self):
        """Initialize the runner."""
        """启动运行器。"""
        print("Initializing runner...")

    def update(self, task: str):
        """Update the runner with a new task.

        Args:
            task: The task to add.
        """
        """更新运行员一个新的任务。

        参数：
            task: 增加任务。
        """
        self._tasks.append(task)
        print(f"Added task: {task}")

    def close(self):
        """Close the runner."""
        """关闭跑步机。"""
        print("Closing runner...")

    """
    Operations: Registration.
    """
    """Operations: 报名。
    """

    @classmethod
    def register(cls, name: str, runner: "TaskRunner"):
        """Register a runner.

        Args:
            name: The name of the runner.
            runner: The runner to register.
        """
        """报名一个跑步者。

        参数：
            name: 跑者的名字。
            runner: 跑步者注册。
        """
        if name in cls._registry:
            _log_error(f"Runner {name} already registered. Skipping registration.")
            return
        cls._registry[name] = runner

    @staticmethod
    def validate_task(task: str) -> bool:
        """Validate a task.

        Args:
            task: The task to validate.

        Returns:
            True if the task is valid, False otherwise.
        """
        """验证任务。

        参数：
            task: 验证的任务。

        返回：
            如果任务是有效的 True，否则 False。
        """
        return bool(task and task.strip())

    """
    Internal operations.
    """
    """内部运营。
    """

    def _reset(self):
        """Reset the runner."""
        """重置运行器。"""
        self._tasks.clear()

    @classmethod
    def _get_registry(cls) -> dict:
        """Get the registry."""
        """拿起注册表。"""
        return cls._registry

    @staticmethod
    def _internal_helper():
        """Internal helper."""
        """内部助理。"""
        print("Internal helper called.")


"""
Helper operations.
"""
"""助手动作。
"""


def _log_error(message: str):
    """Internal helper to log errors.

    Args:
        message: The message to log.
    """
    """内部帮助记录错误。

    参数：
        message: 记录信息。
    """
    print(f"[ERROR] {message}")


class _TaskHelper:
    """Private utility class for internal task logic."""
    """内部任务逻辑的私用类。"""

    def compute(self) -> int:
        """Compute the result.

        Returns:
            The result of the computation.
        """
        """计算结果。

        返回：
            计算的结果。
        """
        return 42
