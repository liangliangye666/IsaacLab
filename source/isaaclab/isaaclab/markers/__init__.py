# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for marker utilities to simplify creation of UI elements in the GUI.

Currently, the sub-package provides the following classes:

* :class:`VisualizationMarkers` for creating a group of markers using `UsdGeom.PointInstancer
  <https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html>`_.


.. note::

    For some simple use-cases, it may be sufficient to use the debug drawing utilities from Isaac Sim.
    The debug drawing API is available in the `isaacsim.util.debug_drawing`_ module. It allows drawing of
    points and splines efficiently on the UI.

    .. _isaacsim.util.debug_drawing: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_debug_drawing.html

"""
"""在GUI中简化创建UI元素的标记工具子包。

目前，该子包包含以下类别:

* :class:`VisualizationMarkers`用于创建一个标记组`UsdGeom.PointInstancer
  <https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html>`_。


.. 说明::

    对于一些简单的使用案例，使用Isaac Sim的调试绘图工具可能足够。
    在`isaacsim.util.debug_drawing`_模块中可使用API调试图。
    它允许在UI上有效地绘制点和斜率。

    .. _isaacsim.util.debug_drawing: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_debug_drawing.html
"""

from .config import *  # noqa: F401, F403
from .visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
