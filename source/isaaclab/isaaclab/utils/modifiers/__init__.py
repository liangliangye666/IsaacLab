# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing different modifiers implementations.

Modifiers are used to apply stateful or stateless modifications to tensor data. They take
in a tensor and a configuration and return a tensor with the modification applied. This way users
can define custom operations to apply to a tensor. For instance, a modifier can be used to normalize
the input data or to apply a rolling average.

They are primarily used to apply custom operations in the :class:`~isaaclab.managers.ObservationManager`
as an alternative to the built-in noise, clip and scale post-processing operations. For more details, see
the :class:`~isaaclab.managers.ObservationTermCfg` class.

Usage with a function modifier:

.. code-block:: python

    import torch
    from isaaclab.utils import modifiers

    # create a random tensor
    my_tensor = torch.rand(256, 128, device="cuda")

    # create a modifier configuration
    cfg = modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (0.0, torch.inf)})

    # apply the modifier
    my_modified_tensor = cfg.func(my_tensor, cfg)


Usage with a class modifier:

.. code-block:: python

    import torch
    from isaaclab.utils import modifiers

    # create a random tensor
    my_tensor = torch.rand(256, 128, device="cuda")

    # create a modifier configuration
    # a digital filter with a simple delay of 1 timestep
    cfg = modifiers.DigitalFilterCfg(A=[0.0], B=[0.0, 1.0])

    # create the modifier instance
    my_modifier = modifiers.DigitalFilter(cfg, my_tensor.shape, "cuda")

    # apply the modifier as a callable object
    my_modified_tensor = my_modifier(my_tensor)

"""
"""包含不同修改器实现的子模块。

修改器用于对 data子数据进行状态或无状态修改。
他们采用一个子和一个配置，然后将一个子返回了修改。
这样，用户可以定义定制操作，
例如，可以使用修改器来正常化输入数据或应用滚动平均值。

它们主要用于在:class:`~isaaclab.managers.ObservationManager`中应用定制操作，代替内置的噪音，裁剪和尺度后处理操作。
更多详情请见
the :类:`~isaaclab.managers.ObservationTermCfg`类。

使用函数修改器:

.. code-block:: python

    import torch
    from isaaclab.utils import modifiers

    # create a random tensor
    my_tensor = torch.rand(256, 128, device="cuda")

    # create a modifier configuration
    cfg = modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (0.0, torch.inf)})

    # apply the modifier
    my_modified_tensor = cfg.func(my_tensor, cfg)


用于类修改器:

.. code-block:: python

    import torch
    from isaaclab.utils import modifiers

    # create a random tensor
    my_tensor = torch.rand(256, 128, device="cuda")

    # create a modifier configuration
    # a digital filter with a simple delay of 1 timestep
    cfg = modifiers.DigitalFilterCfg(A=[0.0], B=[0.0, 1.0])

    # create the modifier instance
    my_modifier = modifiers.DigitalFilter(cfg, my_tensor.shape, "cuda")

    # apply the modifier as a callable object
    my_modified_tensor = my_modifier(my_tensor)
"""

# isort: off
from .modifier_cfg import ModifierCfg
from .modifier_base import ModifierBase
from .modifier import DigitalFilter
from .modifier_cfg import DigitalFilterCfg
from .modifier import Integrator
from .modifier_cfg import IntegratorCfg

# isort: on
from .modifier import bias, clip, scale
