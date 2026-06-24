# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing simulation-specific functionalities.

These include:

* Ability to spawn different objects and materials into Omniverse
* Define and modify various schemas on USD prims
* Converters to obtain USD file from other file formats (such as URDF, OBJ, STL, FBX)
* Utility class to control the simulator

.. note::
    Currently, only a subset of all possible schemas and prims in Omniverse are supported.
    We are expanding the these set of functions on a need basis. In case, there are
    specific prims or schemas that you would like to include, please open an issue on GitHub
    as a feature request elaborating on the required application.

To make it convenient to use the module, we recommend importing the module as follows:

.. code-block:: python

    import isaaclab.sim as sim_utils

"""
"""包含仿真特定功能的子包。

这些包括:

* 能够将不同的物体和材料产生到全宇宙中
* 定义和修改USD prims的各种方案
* 转换器从其他文件格式获取USD文件 (如URDF，OBJ，STL，FBX)
* 控制仿真器的实用类

.. 说明::
    目前，只支持所有可能的方案和全宇宙中的prims的子集。
    我们正在根据需要扩大这些功能。
    如果您希望包含特定的prims或方案，请在GitHub上打开一个问题，作为详细的功能请求。

为了方便使用模块，我们建议导入模块如下:

.. code-block:: python

    import isaaclab.sim as sim_utils
"""

from .converters import *  # noqa: F401, F403
from .schemas import *  # noqa: F401, F403
from .simulation_cfg import PhysxCfg, RenderCfg, SimulationCfg  # noqa: F401, F403
from .simulation_context import SimulationContext, build_simulation_context  # noqa: F401, F403
from .spawners import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from .views import *  # noqa: F401, F403
