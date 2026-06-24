# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to specify the symmetry in the observation and action space for cartpole."""

from __future__ import annotations
"""函数，用于指定车杆的观测和动作空间中的对称性。"""

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    two symmetrical transformations: original, left-right. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """
    """通过应用对称变化来增加给定的观测和动作。

    这种函数通过应用两个对称变化来创建所提供的观测和动作的增强版本:原始，左-右。
    对称性转型为强化学习任务有利，因为它们提供了额外的多元化数据，而不需要额外的数据收集。

    参数：
        env: 环境情况。
        obs: 观测数字典的原始版本。
             默认为 None。
        actions: 原始的动作张量。
                 默认为 None。

    返回：
        增强的观测和动作张量，或None，如果各自的输入是None。
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = -obs["policy"]
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = -actions
    else:
        actions_aug = None

    return obs_aug, actions_aug
