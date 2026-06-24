# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package for environment wrappers to different learning frameworks.

Wrappers allow you to modify the behavior of an environment without modifying the environment itself.
This is useful for modifying the observation space, action space, or reward function. Additionally,
they can be used to cast a given environment into the respective environment class definition used by
different learning frameworks. This operation may include handling of asymmetric actor-critic observations,
casting the data between different backends such `numpy` and `pytorch`, or organizing the returned data
into the expected data structure by the learning framework.

All wrappers work similar to the :class:`gymnasium.Wrapper` class. Using a wrapper is as simple as passing
the initialized environment instance to the wrapper constructor. However, since learning frameworks
expect different input and output data structures, their wrapper classes are not compatible with each other.
Thus, they should always be used in conjunction with the respective learning framework.
"""
"""包装环境包装不同学习框架。

包装器允许你改变环境的行为，而不改变环境本身。
这对于修改观测空间，动作空间或奖励功能来说是有用的。
此外，它们可以用于将特定环境纳入不同的学习框架所使用的各自环境类定义中。
这项操作可能包括处理不对称的演员-关键观测，将数据投放在`numpy`和`pytorch`等不同后台之间，或通过学习框架将返回的数据组织到预期的数据结构中。

所有包装都类似于:class:`gymnasium.Wrapper`类。
使用包装器就像将初始化环境实例传递到包装架构器一样简单。
然而，由于学习框架期望不同的输入和输出数据结构，因此它们的包装类别不兼容。
因此，它们应始终与各自的学习框架相结合使用。
"""
