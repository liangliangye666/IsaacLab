# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import omni.ui


class UiVisualizerBase:
    """Base Class for components that support debug visualizations that requires access to some UI elements.

    This class provides a set of functions that can be used to assign ui interfaces.

    The following functions are provided:

    * :func:`set_debug_vis`: Assigns a debug visualization interface. This function is called by the main UI
        when the checkbox for debug visualization is toggled.
    * :func:`set_vis_frame`: Assigns a small frame within the isaac lab tab that can be used to visualize debug
        information. Such as e.g. plots or images. It is called by the main UI on startup to create the frame.
    * :func:`set_window`: Assigngs the main window that is used by the main UI. This allows the user
        to have full controller over all UI elements. But be warned, with great power comes great responsibility.
    """
    """支持需要访问某些UI元素的调试可视化组件的基类。

    这类提供了一组可用于分配UI接口的函数。

    提供以下功能:

    * :func:`set_debug_vis`:分配一个调试可视化界面.当调试可视化的选号框被切换时，该函数由主UI调用。
    * :func:`set_vis_frame`:在 isaac 实验室 tabb 中分配一个小框架，可用于可视化调试信息.例如 e.g。 图片或图像.在启动时，主要UI 调用它来创建框架。
    * :func:`set_window`:分配了主要UI所使用的主窗口。 这允许用户对所有UI元素进行全面控制。
    """

    """
    Exposed Properties
    """
    """暴露的属性
    """

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the component has a debug visualization implemented."""
        """如果组件实现了调试可视化。"""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    @property
    def has_vis_frame_implementation(self) -> bool:
        """Whether the component has a debug visualization implemented."""
        """如果组件实现了调试可视化。"""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_vis_frame_impl)
        return "NotImplementedError" not in source_code

    @property
    def has_window_implementation(self) -> bool:
        """Whether the component has a debug visualization implemented."""
        """如果组件实现了调试可视化。"""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_window_impl)
        return "NotImplementedError" not in source_code

    @property
    def has_env_selection_implementation(self) -> bool:
        """Whether the component has a debug visualization implemented."""
        """如果组件实现了调试可视化。"""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_env_selection_impl)
        return "NotImplementedError" not in source_code

    """
    Exposed Setters
    """
    """暴露的设置器
    """

    def set_env_selection(self, env_selection: int) -> bool:
        """Sets the selected environment id.

        This function is called by the main UI when the user selects a different environment.

        Args:
            env_selection: The currently selected environment id.

        Returns:
            Whether the environment selection was successfully set. False if the component
            does not support environment selection.
        """
        """设置选择的环境ID。

        当用户选择不同的环境时，这个函数由主UI调用。

        参数：
            env_selection: 目前选择的环境ID。

        返回：
            环境选择是否成功设置。
            如果组件不支持环境选择，False
        """
        # check if environment selection is supported
        if not self.has_env_selection_implementation:
            return False
        # set environment selection
        self._set_env_selection_impl(env_selection)
        return True

    def set_window(self, window: omni.ui.Window) -> bool:
        """Sets the current main ui window.

        This function is called by the main UI when the window is created. It allows the component
        to add custom UI elements to the window or to control the window and its elements.

        Args:
            window: The ui window.

        Returns:
            Whether the window was successfully set. False if the component
            does not support this functionality.
        """
        """设置当前的主要窗口。

        当创建窗口时，这个函数被主UI调用。
        它允许组件将自定义UI元素添加到窗口中或控制窗口及其元素。

        参数：
            window: 窗口。

        返回：
            窗户是否成功设置。
            False如果该组件不支持此功能。
        """
        # check if window is supported
        if not self.has_window_implementation:
            return False
        # set window
        self._set_window_impl(window)
        return True

    def set_vis_frame(self, vis_frame: omni.ui.Frame) -> bool:
        """Sets the debug visualization frame.

        This function is called by the main UI when the window is created. It allows the component
        to modify a small frame within the orbit tab that can be used to visualize debug information.

        Args:
            vis_frame: The debug visualization frame.

        Returns:
            Whether the debug visualization frame was successfully set. False if the component
            does not support debug visualization.
        """
        """设置调试可视化框架。

        当创建窗口时，这个函数被主UI调用。
        它允许组件修改轨道图表内的小框架，可用于可视化调试信息。

        参数：
            vis_frame: 错误可视化框架。

        返回：
            否成功设置调试可视化框架
            False如果该组件不支持调试可视化。
        """
        # check if debug visualization is supported
        if not self.has_vis_frame_implementation:
            return False
        # set debug visualization frame
        self._set_vis_frame_impl(vis_frame)
        return True

    """
    Internal Implementation
    """
    """内部实施
    """

    def _set_env_selection_impl(self, env_idx: int):
        """Set the environment selection."""
        """设置环境选择。"""
        raise NotImplementedError(f"Environment selection is not implemented for {self.__class__.__name__}.")

    def _set_window_impl(self, window: omni.ui.Window):
        """Set the window."""
        """设置窗户。"""
        raise NotImplementedError(f"Window is not implemented for {self.__class__.__name__}.")

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization state."""
        """设置调试可视化状态。"""
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _set_vis_frame_impl(self, vis_frame: omni.ui.Frame):
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        """设置调试可视化到可视化对象。

        如果它们不存在，并且输入 ``debug_vis`` 是 True，
        如果可视化对象存在，函数应该将它们的可视性设置在舞台上。
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")
