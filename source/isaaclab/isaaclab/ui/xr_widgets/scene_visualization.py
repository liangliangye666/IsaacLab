# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import inspect
import logging
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, Union

import numpy as np
import torch

from pxr import Gf

from isaaclab.sim import SimulationContext
from isaaclab.ui.xr_widgets import show_instruction

# import logger
logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Enumeration of trigger types for visualization callbacks.

    Defines when callbacks should be executed:
    - TRIGGER_ON_EVENT: Execute when a specific event occurs
    - TRIGGER_ON_PERIOD: Execute at regular time intervals
    - TRIGGER_ON_CHANGE: Execute when a specific data variable changes
    - TRIGGER_ON_UPDATE: Execute every frame
    """
    """视觉回调的触发器类型列表。

    定义应执行回调时间:
    - TRIGGER_ON_EVENT:执行当特定事件发生时
    - 执行时间间隔
    - TRIGGER_ON_CHANGE:当特定数据变量发生变化时执行
    - 执行每个框架
    """

    TRIGGER_ON_EVENT = 0
    TRIGGER_ON_PERIOD = 1
    TRIGGER_ON_CHANGE = 2
    TRIGGER_ON_UPDATE = 3


class DataCollector:
    """Collects and manages data for visualization purposes.

    This class provides a centralized data store for visualization data,
    with change detection and callback mechanisms for real-time updates.
    """
    """收集和管理数据以实现可视化目的。

    这类为可视化数据提供一个集中数据存储，
    with change detection and callback mechanisms for real-time updates.
    """

    def __init__(self):
        """Initialize the data collector with empty data store and callback system."""
        """启动数据收集器，使用空格数据存储和回调系统。"""
        self._data: dict[str, Any] = {}
        self._visualization_callback: Callable | None = None
        self._changed_flags: set[str] = set()

    def _values_equal(self, existing_value: Any, new_value: Any) -> bool:
        """Compare two values using appropriate method based on their types.

        Handles different data types including None, NumPy arrays, PyTorch tensors,
        and standard Python types for accurate change detection.

        Args:
            existing_value: The current value stored in the data collector
            new_value: The new value to compare against

        Returns:
            bool: True if values are equal, False otherwise
        """
        """根据其类型，使用适当的方法比较两个值。

        处理包括None，NumPy阵列，PyTorch子和标准Python类型的不同数据类型，以准确检测变化。

        参数：
            existing_value: 在数据收集器中存储的当前值
            new_value: 新的比较值

        返回：
            bool: 如果值等于True，否则False
        """
        # If both are None or one is None
        if existing_value is None or new_value is None:
            return existing_value is new_value

        # If types are different, they're not equal
        if type(existing_value) is not type(new_value):
            return False

        # Handle NumPy arrays
        if isinstance(existing_value, np.ndarray):
            return np.array_equal(existing_value, new_value)

        # Handle torch tensors (if they exist)
        if hasattr(existing_value, "equal"):
            with contextlib.suppress(Exception):
                return torch.equal(existing_value, new_value)

        # For all other types (int, float, string, bool, list, dict, set), use regular equality
        with contextlib.suppress(Exception):
            return existing_value == new_value
        # If comparison fails for any reason, assume they're different
        return False

    def update_data(self, name: str, value: Any) -> None:
        """Update a data field and trigger change detection.

        This method handles data updates with intelligent change detection.
        It also performs pre-processing and post-processing based on the field name.

        Args:
            name: The name/key of the data field to update
            value: The new value to store (None to remove the field)
        """
        """更新数据字段并触发变化检测。

        这种方法通过智能变化检测来处理数据更新。
        它还根据领域名称进行预处理和后处理。

        参数：
            name: 更新的数据字段名称/关键
            value: 存储的新值 (None删除该字段)
        """
        existing_value = self.get_data(name)

        if value is None:
            self._data.pop(name)
            if existing_value is not None:
                self._changed_flags.add(name)
            return

        # Todo: for list or array, the change won't be detected
        # Check if the value has changed using appropriate comparison method
        if self._values_equal(existing_value, value):
            return

        # Save it
        self._data[name] = value
        self._changed_flags.add(name)

    def update_loop(self) -> None:
        """Process pending changes and trigger visualization callbacks.

        This method should be called regularly to ensure visualization updates
        are processed in a timely manner.
        """
        """处理正在进行的变化并触发可视化回调。

        这种方法应定期调用，以确保可视化更新及时处理。
        """
        if len(self._changed_flags) > 0:
            if self._visualization_callback:
                self._visualization_callback(self._changed_flags)
            self._changed_flags.clear()

    def get_data(self, name: str) -> Any:
        """Retrieve data by name.

        Args:
            name: The name/key of the data field to retrieve

        Returns:
            The stored value, or None if the field doesn't exist
        """
        """取名数据。

        参数：
            name: 获取数据字段名称/关键

        返回：
            存储值，或者None如果该领域不存在
        """
        return self._data.get(name)

    def set_visualization_callback(self, callback: Callable) -> None:
        """Set the VisualizationManager callback function to be called when data changes.

        Args:
            callback: Function to call when data changes, receives set of changed field names
        """
        """设置数据变化时调用VisualizationManager回调函数。

        参数：
            callback: 当数据变化时调用函数，接收了已更改的字段名称集
        """
        self._visualization_callback = callback


class VisualizationManager:
    """Base class for managing visualization rules and callbacks.

    Provides a framework for registering and executing callbacks based on
    different trigger conditions (events, time periods, data changes).
    """
    """管理可视化规则和回调的基础类。

    根据不同触发条件 (事件，时间段，数据变化) 进行回调记录和执行的框架。
    """

    # Type aliases for different callback signatures
    StandardCallback = Callable[["VisualizationManager", "DataCollector"], None]
    EventCallback = Callable[["VisualizationManager", "DataCollector", Any], None]
    CallbackType = Union[StandardCallback, EventCallback]  # noqa: UP007

    class TimeCountdown:
        """Internal class for managing periodic timer-based callbacks."""
        """内部类型用于管理周期性定制器回调。"""

        period: float
        countdown: float
        last_time: float

        def __init__(self, period: float, initial_countdown: float = 0.0):
            """Initialize a countdown timer.

            Args:
                period: Time interval in seconds between callback executions
            """
            """启动倒计时器。

            参数：
                period: 召回执行之间的数秒时间间隔
            """
            self.period = period
            self.countdown = initial_countdown
            self.last_time = time.time()

        def update(self, current_time: float) -> bool:
            """Update the countdown timer and check if callback should be triggered.

            Args:
                current_time: Current time in seconds

            Returns:
                bool: True if callback should be triggered, False otherwise
            """
            """更新倒计时器，检查是否应该启动回调。

            参数：
                current_time: 目前时间

            返回：
                bool: 如果应启动回调，True，否则False
            """
            self.countdown -= current_time - self.last_time
            self.last_time = current_time
            if self.countdown <= 0.0:
                self.countdown = self.period
                return True
            return False

    # Widget presets for common visualization configurations
    @classmethod
    def message_widget_preset(cls) -> dict[str, Any]:
        """Get the message widget preset configuration.

        Returns:
            dict: Configuration dictionary for message widgets
        """
        """获取消息预设配置。

        返回：
            dict: 信息 widget 的配置字典
        """
        return {
            "prim_path_source": "/_xr/stage/xrCamera",
            "translation": Gf.Vec3f(0, 0, -2),
            "display_duration": 3.0,
            "max_width": 2.5,
            "min_width": 1.0,
            "font_size": 0.1,
            "text_color": 0xFF00FFFF,
        }

    @classmethod
    def panel_widget_preset(cls) -> dict[str, Any]:
        """Get the panel widget preset configuration.

        Returns:
            dict: Configuration dictionary for panel widgets
        """
        """获取面板插件预设配置。

        返回：
            dict: 面板 widget 的配置字典
        """
        return {
            "prim_path_source": "/XRAnchor",
            "translation": Gf.Vec3f(0, 2, 2),  # hard-coded temporarily
            "display_duration": 0.0,
            "font_size": 0.13,
            "max_width": 2,
            "min_width": 2,
        }

    def display_widget(self, text: str, name: str, args: dict[str, Any]) -> None:
        """Display a widget with the given text and configuration.

        Args:
            text: Text content to display in the widget
            name: Unique identifier for the widget. If duplicated, the old one will be removed from scene.
            args: Configuration dictionary for widget appearance and behavior
        """
        """显示给定的文本和配置。

        参数：
            text: 在 widget 中显示文本内容
            name: 唯一的标识符。
                  如果复制，旧的将从场景删除。
            args: 配置字典对小工具的外观和行为
        """
        widget_config = args | {"text": text, "target_prim_path": name}
        show_instruction(**widget_config)

    def __init__(self, data_collector: DataCollector):
        """Initialize the visualization manager.

        Args:
            data_collector: DataCollector instance to access the data for visualization use.
        """
        """启动可视化管理器。

        参数：
            data_collector: 访问数据可用于可视化使用的DataCollector实例。
        """
        self.data_collector: DataCollector = data_collector
        data_collector.set_visualization_callback(self.on_change)

        self._rules_on_period: dict[VisualizationManager.TimeCountdown, VisualizationManager.StandardCallback] = {}
        self._rules_on_event: dict[str, list[VisualizationManager.EventCallback]] = {}
        self._rules_on_change: dict[str, list[VisualizationManager.StandardCallback]] = {}
        self._rules_on_update: list[VisualizationManager.StandardCallback] = []

    # Todo: add support to registering same callbacks for different names
    def on_change(self, names: set[str]) -> None:
        """Handle data changes by executing registered callbacks.

        Args:
            names: Set of data field names that have changed
        """
        """通过执行注册回调来处理数据变化。

        参数：
            names: 已改变的数据字段名称集合
        """
        for name in names:
            callbacks = self._rules_on_change.get(name)
            if callbacks:
                # Create a copy of the list to avoid modification during iteration
                for callback in list(callbacks):
                    callback(self, self.data_collector)
        if len(names) > 0:
            self.on_event("default_event_has_change")

    def update_loop(self) -> None:
        """Update periodic timers and execute callbacks as needed.

        This method should be called regularly to ensure periodic callbacks
        are executed at the correct intervals.
        """
        """根据需要更新定期计时时间并执行回调。

        这种方法应定期调用，以确保定期回调在正确的间隔中执行。
        """

        # Create a copy of the list to avoid modification during iteration
        for callback in list(self._rules_on_update):
            callback(self, self.data_collector)

        current_time = time.time()
        # Create a copy of the items to avoid modification during iteration
        for timer, callback in list(self._rules_on_period.items()):
            triggered = timer.update(current_time)
            if triggered:
                callback(self, self.data_collector)

    def on_event(self, event: str, params: Any = None) -> None:
        """Handle events by executing registered callbacks.

        Args:
            event: Name of the event that occurred
        """
        """通过执行注册回调来处理事件。

        参数：
            event: 发生的事件名称
        """
        callbacks = self._rules_on_event.get(event)
        if callbacks is None:
            return
        # Create a copy of the list to avoid modification during iteration
        for callback in list(callbacks):
            callback(self, self.data_collector, params)

    # Todo: better organization of callbacks
    def register_callback(self, trigger: TriggerType, arg: dict, callback: CallbackType) -> Any:
        """Register a callback function to be executed based on trigger conditions.

        Args:
            trigger: Type of trigger that should execute the callback
            arg: Dictionary containing trigger-specific parameters:
                - For TRIGGER_ON_PERIOD: {"period": float}
                - For TRIGGER_ON_EVENT: {"event_name": str}
                - For TRIGGER_ON_CHANGE: {"variable_name": str}
                - For TRIGGER_ON_UPDATE: {}
            callback: Function to execute when trigger condition is met. The callback should have
                the following signatures according to the trigger type:
                - For TRIGGER_ON_EVENT:
                    callback(
                        manager: VisualizationManager,
                        data_collector: DataCollector,
                        event_params: Any,
                    )
                - For others:
                    callback(
                        manager: VisualizationManager,
                        data_collector: DataCollector,
                    )

        Raises:
            TypeError: If callback signature doesn't match the expected signature for the trigger type
        """
        """根据触发条件执行回调函数。

        参数：
            trigger: 应执行回调的触发器类型
            arg: 含有触发器特定参数的字典:
                - 对于TRIGGER_ON_PERIOD: {"period": float}
                - 对于 TRIGGER_ON_EVENT: {"event_name": str}
                - 对于 TRIGGER_ON_CHANGE: {"variable_name": str}
                - 为了TIGGER_ON_UPDATE: {}
            callback: 在触发条件达到时执行函数。
                      根据触发器类型，回调应应具有以下签名:
                - 对于TIGGER_ON_EVENT:回调
                        manager: VisualizationManager，
                        data_collector: DataCollector，
                        event_params: 任何一个，
                    )
                - 其他:回调
                        manager: VisualizationManager，
                        data_collector: DataCollector，
                    )

        异常：
            TypeError: 如果回调签名不符合预期的签名，
        """
        # Validate callback signature based on trigger type
        self._validate_callback_signature(trigger, callback)

        match trigger:
            case TriggerType.TRIGGER_ON_PERIOD:
                period = arg.get("period")
                initial_countdown = arg.get("initial_countdown", 0.0)
                if isinstance(period, float) and isinstance(initial_countdown, float):
                    timer = VisualizationManager.TimeCountdown(period=period, initial_countdown=initial_countdown)
                    # Type cast since we've validated the signature
                    self._rules_on_period[timer] = callback  # type: ignore
                    return timer
            case TriggerType.TRIGGER_ON_EVENT:
                event = arg.get("event_name")
                if isinstance(event, str):
                    callbacks = self._rules_on_event.get(event)
                    if callbacks is None:
                        # Type cast since we've validated the signature
                        self._rules_on_event[event] = [callback]  # type: ignore
                    else:
                        # Type cast since we've validated the signature
                        self._rules_on_event[event].append(callback)  # type: ignore
                    return event
            case TriggerType.TRIGGER_ON_CHANGE:
                variable_name = arg.get("variable_name")
                if isinstance(variable_name, str):
                    callbacks = self._rules_on_change.get(variable_name)
                    if callbacks is None:
                        # Type cast since we've validated the signature
                        self._rules_on_change[variable_name] = [callback]  # type: ignore
                    else:
                        # Type cast since we've validated the signature
                        self._rules_on_change[variable_name].append(callback)  # type: ignore
                    return variable_name
            case TriggerType.TRIGGER_ON_UPDATE:
                # Type cast since we've validated the signature
                self._rules_on_update.append(callback)  # type: ignore
        return None

    # Todo: better callback-cancel method
    def cancel_rule(self, trigger: TriggerType, arg: str | TimeCountdown, callback: Callable | None = None) -> None:
        """Remove a previously registered callback.

        Periodic callbacks are not supported to be cancelled for now.

        Args:
            trigger: Type of trigger for the callback to remove
            arg: Trigger-specific identifier (event name or variable name)
            callback: The callback function to remove
        """
        """删除已注册的回调。

        暂时不支持取消定期回调。

        参数：
            trigger: 取消回调的触发器类型
            arg: 触发器特定标识符 (事件名称或变量名称)
            callback: 删除回调函数
        """
        callbacks = None
        match trigger:
            case TriggerType.TRIGGER_ON_CHANGE:
                callbacks = self._rules_on_change.get(arg)
            case TriggerType.TRIGGER_ON_EVENT:
                callbacks = self._rules_on_event.get(arg)
            case TriggerType.TRIGGER_ON_PERIOD:
                self._rules_on_period.pop(arg)
            case TriggerType.TRIGGER_ON_UPDATE:
                callbacks = self._rules_on_update
        if callbacks is not None:
            if callback is not None:
                callbacks.remove(callback)
            else:
                callbacks.clear()

    def set_attr(self, name: str, value: Any) -> None:
        """Set an attribute of the visualization manager.

        Args:
            name: Name of the attribute to set
            value: Value to set the attribute to
        """
        """设置可视化管理器的属性。

        参数：
            name: 集合属性的名称
            value: 将属性设置为
        """
        setattr(self, name, value)

    def _validate_callback_signature(self, trigger: TriggerType, callback: Callable) -> None:
        """Validate that the callback has the correct signature for the trigger type.

        Args:
            trigger: Type of trigger for the callback
            callback: The callback function to validate

        Raises:
            TypeError: If callback signature doesn't match expected signature
        """
        """验证回调具有触发器类型的正确签名。

        参数：
            trigger: 回调的触发器类型
            callback: 验证的回调函数

        异常：
            TypeError: 如果回调签名不符合预期签名
        """
        try:
            sig = inspect.signature(callback)
            params = list(sig.parameters.values())

            # Remove 'self' parameter if it's a bound method
            if params and params[0].name == "self":
                params = params[1:]

            param_count = len(params)

            if trigger == TriggerType.TRIGGER_ON_EVENT:
                # Event callbacks should have 3 parameters: (manager, data_collector, event_params)
                expected_count = 3
                expected_sig = (
                    "callback(manager: VisualizationManager, data_collector: DataCollector, event_params: Any)"
                )
            else:
                # Other callbacks should have 2 parameters: (manager, data_collector)
                expected_count = 2
                expected_sig = "callback(manager: VisualizationManager, data_collector: DataCollector)"

            if param_count != expected_count:
                raise TypeError(
                    f"Callback for {trigger.name} must have {expected_count} parameters, "
                    f"but got {param_count}. Expected signature: {expected_sig}. "
                    f"Actual signature: {sig}"
                )

        except Exception as e:
            if isinstance(e, TypeError):
                raise
            # If we can't inspect the signature (e.g., built-in functions),
            # just log a warning and proceed
            logger.warning(f"Could not validate callback signature for {trigger.name}: {e}")


class XRVisualization:
    """Singleton class providing XR visualization functionality.

    This class implements the singleton pattern to ensure only one instance
    of the visualization system exists across the application. It provides
    a centralized API for managing XR visualization features.

    When manage a new event ordata field, please add a comment to the following list.

    Event names:
        "ik_solver_failed"

    Data fields:
        "manipulability_ellipsoid" : list[float]
        "device_raw_data" : dict
        "joints_distance_percentage_to_limit" : list[float]
        "joints_torque" : list[float]
        "joints_torque_limit" : list[float]
        "joints_name" : list[str]
        "wrist_pose" : list[float]
        "approximated_working_space" : list[float]
        "hand_torque_mapping" : list[str]
    """
    """提供XR可视化功能的Singleton类。

    这个类实现单元模式，以确保整个应用程序中只存在一个可视化系统的实例。
    它提供了一个集中 API 管理XR可视化功能。

    当管理一个新的事件数据字段时，请在下列列表中添加评论。

    事件名称: "ik_solver_failed"

    数据场:"manipulability_ellipsoid" : list[float] "device_raw_data" : dict
    "joints_distance_percentage_to_limit" : list[float] "joints_torque" : list[float]
    "joints_torque_limit" : list[float] "joints_name" : list[str] "wrist_pose" : list[float]
    "approximated_working_space" : list[float] "hand_torque_mapping" : list[str]
    """

    _lock = threading.Lock()
    _instance: XRVisualization | None = None
    _registered = False

    def __init__(self):
        """Prevent direct instantiation."""
        """防止直接实例化。"""
        raise RuntimeError("Use VisualizationInterface classmethods instead of direct instantiation")

    @classmethod
    def __create_instance(cls, manager: type[VisualizationManager] = VisualizationManager) -> XRVisualization:
        """Get the visualization manager instance.

        Returns:
            VisualizationManager: The visualization manager instance
        """
        """查看视觉管理器实例。

        返回：
            VisualizationManager: 视觉管理器实例
        """
        with cls._lock:
            if cls._instance is None:
                # Bypass __init__ by calling __new__ directly
                cls._instance = super().__new__(cls)
                cls._instance._initialize(manager)
        return cls._instance

    @classmethod
    def __get_instance(cls) -> XRVisualization:
        """Thread-safe singleton access.

        Returns:
            XRVisualization: The singleton instance of the visualization system
        """
        """无线单机接入。

        返回：
            XRVisualization: 视觉化系统的单个实例
        """
        if cls._instance is None:
            return cls.__create_instance()
        elif not cls._instance._registered:
            cls._instance._register()
        return cls._instance

    def _register(self) -> bool:
        """Register the visualization system.

        Returns:
            bool: True if the visualization system is registered, False otherwise
        """
        """记录视觉系统。

        返回：
            bool: 如果可视化系统已注册，则True，否则False
        """
        if self._registered:
            return True

        sim = SimulationContext.instance()
        if sim is not None:
            sim.add_render_callback("visualization_render_callback", self.update_loop)
            self._registered = True
        return self._registered

    def _initialize(self, manager: type[VisualizationManager]) -> None:
        """Initialize the singleton instance with data collector and visualization manager."""
        """使用数据收集器和可视化管理器启动单元实例。"""

        self._data_collector = DataCollector()
        self._visualization_manager = manager(self._data_collector)

        self._register()

        self._initialized = True

    # APIs

    def update_loop(self, event) -> None:
        """Update the visualization system.

        This method should be called regularly (e.g., every frame) to ensure
        visualization updates are processed and periodic callbacks are executed.
        """
        """更新可视化系统。

        这种方法应该定期调用 (e.g.，每一个框架)，以确保可视化更新进行处理，并执行定期回调。
        """
        self._visualization_manager.update_loop()
        self._data_collector.update_loop()

    @classmethod
    def push_event(cls, name: str, args: Any = None) -> None:
        """Push an event to trigger registered callbacks.

        Args:
            name: Name of the event to trigger
            args: Optional arguments for the event (currently unused)
        """
        """推出一个事件以触发注册回调。

        参数：
            name: 引发事件名称
            args: 对事件的可选参数 (目前未使用)
        """
        instance = cls.__get_instance()
        instance._visualization_manager.on_event(name, args)

    @classmethod
    def push_data(cls, item: dict[str, Any]) -> None:
        """Push data to the visualization system.

        Updates multiple data fields at once. Each key-value pair in the
        dictionary will be processed by the data collector.

        Args:
            item: Dictionary containing data field names and their values
        """
        """将数据输入到可视化系统中。

        一次更新多个数据字段。
        字典中的每个关键值对将由数据收集器处理。

        参数：
            item: 包含数据字段名称及其值的字典
        """
        instance = cls.__get_instance()
        for name, value in item.items():
            instance._data_collector.update_data(name, value)

    @classmethod
    def set_attrs(cls, attributes: dict[str, Any]) -> None:
        """Set configuration data for the visualization system. Not currently used.

        Args:
            attributes: Dictionary containing configuration keys and values
        """
        """设置可视化系统的配置数据。
        目前未使用。

        参数：
            attributes: 包含配置键和值的字典
        """

        instance = cls.__get_instance()
        for name, data in attributes.items():
            instance._visualization_manager.set_attr(name, data)

    @classmethod
    def get_attr(cls, name: str) -> Any:
        """Get configuration data for the visualization system. Not currently used.

        Args:
            name: Configuration key
        """
        """获取可视化系统的配置数据。
        目前未使用。

        参数：
            name: 配置键
        """
        instance = cls.__get_instance()
        return getattr(instance._visualization_manager, name)

    @classmethod
    def register_callback(cls, trigger: TriggerType, arg: dict, callback: VisualizationManager.CallbackType) -> None:
        """Register a callback function for visualization events.

        Args:
            trigger: Type of trigger that should execute the callback
            arg: Dictionary containing trigger-specific parameters:
                - For TRIGGER_ON_PERIOD: {"period": float}
                - For TRIGGER_ON_EVENT: {"event_name": str}
                - For TRIGGER_ON_CHANGE: {"variable_name": str}
            callback: Function to execute when trigger condition is met
        """
        """记录视觉事件的回调函数。

        参数：
            trigger: 应执行回调的触发器类型
            arg: 含有触发器特定参数的字典:
                - 对于TRIGGER_ON_PERIOD: {"period": float}
                - 对于 TRIGGER_ON_EVENT: {"event_name": str}
                - 对于 TRIGGER_ON_CHANGE: {"variable_name": str}
            callback: 当触发条件达到时执行的函数
        """
        instance = cls.__get_instance()
        instance._visualization_manager.register_callback(trigger, arg, callback)

    @classmethod
    def assign_manager(cls, manager: type[VisualizationManager]) -> None:
        """Assign a visualization manager type to the visualization system.

        Args:
            manager: Type of the visualization manager to assign
        """
        """将可视化管理器类型分配到可视化系统中。

        参数：
            manager: 指定可视化管理器类型
        """
        if cls._instance is not None:
            logger.error(
                f"Visualization system already initialized to {type(cls._instance._visualization_manager).__name__},"
                f" cannot assign manager {manager.__name__}"
            )
            return

        cls.__create_instance(manager)
