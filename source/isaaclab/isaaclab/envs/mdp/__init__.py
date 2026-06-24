# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with implementation of manager terms.

The functions can be provided to different managers that are responsible for the
different aspects of the MDP. These include the observation, reward, termination,
actions, events and curriculum managers.

The terms are defined under the ``envs`` module because they are used to define
the environment. However, they are not part of the environment directly, but
are used to define the environment through their managers.

"""
"""执行管理项的子模块

这些职能可以交给负责MDP的不同方面的不同管理器。
这些包括观测，奖励，终止，动作，事件和课程管理器。

这些项在``envs``模块中定义，因为它们用于定义环境。
然而，它们不是直接的环境的一部分，而是通过管理器来定义环境。
"""

from .actions import *  # noqa: F401, F403
from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .recorders import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
