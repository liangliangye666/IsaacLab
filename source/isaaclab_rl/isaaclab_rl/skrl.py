# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an environment instance to skrl environment.

The following example shows how to wrap an environment for skrl:

.. code-block:: python

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # or ml_framework="jax"

Or, equivalently, by directly calling the skrl library API as follows:

.. code-block:: python

    from skrl.envs.torch.wrappers import wrap_env  # for PyTorch, or...
    from skrl.envs.jax.wrappers import wrap_env  # for JAX

    env = wrap_env(env, wrapper="isaaclab")

"""

# needed to import for type hinting: Agent | list[Agent]
from __future__ import annotations
"""包装器来配置一个环境实例到 skrl环境。

下面的例子显示了如何包装skrl的环境:

.. code-block:: python

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # or ml_framework="jax"

或同样，直接调用 skrl 库API，如下:

.. code-block:: python

    from skrl.envs.torch.wrappers import wrap_env  # for PyTorch, or...
    from skrl.envs.jax.wrappers import wrap_env  # for JAX

    env = wrap_env(env, wrapper="isaaclab")
"""

from typing import Literal

from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedRLEnv

"""
Vectorized environment wrapper.
"""
"""面向环境包装。
"""


def SkrlVecEnvWrapper(
    env: ManagerBasedRLEnv | DirectRLEnv | DirectMARLEnv,
    ml_framework: Literal["torch", "jax", "jax-numpy"] = "torch",
    wrapper: Literal["auto", "isaaclab", "isaaclab-single-agent", "isaaclab-multi-agent"] = "isaaclab",
):
    """Wraps around Isaac Lab environment for skrl.

    This function wraps around the Isaac Lab environment. Since the wrapping
    functionality is defined within the skrl library itself, this implementation
    is maintained for compatibility with the structure of the extension that contains it.
    Internally it calls the :func:`wrap_env` from the skrl library API.

    Args:
        env: The environment to wrap around.
        ml_framework: The ML framework to use for the wrapper. Defaults to "torch".
        wrapper: The wrapper to use. Defaults to "isaaclab": leave it to skrl to determine if the environment
            will be wrapped as single-agent or multi-agent.

    Raises:
        ValueError: When the environment is not an instance of any Isaac Lab environment interface.
        ValueError: If the specified ML framework is not valid.

    Reference:
        https://skrl.readthedocs.io/en/latest/api/envs/wrapping.html
    """
    """围绕艾萨克实验室环境进行 skr缩。

    这种功能围绕着艾萨克实验室的环境。
    由于包装功能是定义在 Skrl 库本身，因此该实现保持与包含它的扩展结构的兼容性。
    在内部，它从 Skrl 库中调用 :func:`wrap_env` API。

    参数：
        env: 周围的环境。
        ml_framework: 包装使用的ML框架。
                      默认的"火"。
        wrapper: 包装要使用。
                 "isaaclab"的默认设置:将其留给 skrl 确定环境是否将被包装为单机或多机。

    异常：
        ValueError: 当环境不是任何Isaac Lab环境界面的实例时。
        ValueError: 如果指定的ML框架不有效。

    Reference:
        https://skrl.readthedocs.io其他国家envs/wrapping.html
    """
    # check that input is valid
    if (
        not isinstance(env.unwrapped, ManagerBasedRLEnv)
        and not isinstance(env.unwrapped, DirectRLEnv)
        and not isinstance(env.unwrapped, DirectMARLEnv)
    ):
        raise ValueError(
            "The environment must be inherited from ManagerBasedRLEnv, DirectRLEnv or DirectMARLEnv. Environment type:"
            f" {type(env)}"
        )

    # import statements according to the ML framework
    if ml_framework.startswith("torch"):
        from skrl.envs.wrappers.torch import wrap_env
    elif ml_framework.startswith("jax"):
        from skrl.envs.wrappers.jax import wrap_env
    else:
        ValueError(
            f"Invalid ML framework for skrl: {ml_framework}. Available options are: 'torch', 'jax' or 'jax-numpy'"
        )

    # wrap and return the environment
    return wrap_env(env, wrapper)
