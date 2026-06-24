# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for ANYmal."""

from __future__ import annotations
"""为了指定ANYmal的观测和动作空间的对称性。"""

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
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
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

    这种函数通过应用四个对称变化来创建所提供的观测和动作的增强版本:原始，左向右，前向后和横向。
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
        # since we have 4 different symmetries, we need to augment the batch size by 4
        obs_aug = obs.repeat(4)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
        # -- front-back
        obs_aug["policy"][2 * batch_size : 3 * batch_size] = _transform_policy_obs_front_back(
            env.unwrapped, obs["policy"]
        )
        # -- diagonal
        obs_aug["policy"][3 * batch_size :] = _transform_policy_obs_front_back(
            env.unwrapped, obs_aug["policy"][batch_size : 2 * batch_size]
        )
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        actions_aug = torch.zeros(batch_size * 4, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
        # -- front-back
        actions_aug[2 * batch_size : 3 * batch_size] = _transform_actions_front_back(actions)
        # -- diagonal
        actions_aug[3 * batch_size :] = _transform_actions_front_back(actions_aug[batch_size : 2 * batch_size])
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""
"""对称函数用于观测。
"""


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    negating certain components of the linear and angular velocities, projected gravity,
    velocity commands, and flipping the joint positions, joint velocities, and last actions
    for the ANYmal robot. Additionally, if height-scan data is present, it is flipped
    along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    """应对观测张量进行左向右对称转换。

    这个函数通过应用对左-右轴的对称性表示转换来修改给定的观测张量。
    这包括否定线性和角的速度，预测重力，速度命令，并翻转关节位置，关节速度和最后的动作的某些组件
    for the ANYmal robot. Additionally, if height-scan data is present, it is flipped
    在相关的维度上。

    参数：
        env: 观测得到的环境实例。
        obs: 需要转换的观测张量。

    返回：
        转换的观测张量与左-右对称应用。
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # lin vel
    obs[:, :3] = obs[:, :3] * torch.tensor([1, -1, 1], device=device)
    # ang vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)
    # velocity command
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    obs[:, 12:24] = _switch_anymal_joints_left_right(obs[:, 12:24])
    # joint vel
    obs[:, 24:36] = _switch_anymal_joints_left_right(obs[:, 24:36])
    # last actions
    obs[:, 36:48] = _switch_anymal_joints_left_right(obs[:, 36:48])

    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    if "height_scan" in env.observation_manager.active_terms["policy"]:
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[1]).view(-1, 11 * 17)

    return obs


def _transform_policy_obs_front_back(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the front-back axis. This includes negating
    certain components of the linear and angular velocities, projected gravity, velocity commands,
    and flipping the joint positions, joint velocities, and last actions for the ANYmal robot.
    Additionally, if height-scan data is present, it is flipped along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with front-back symmetry applied.
    """
    """应用前后对称性转换到观测张量。

    这项函数通过应用对前后轴的对称性表示转换来修改给定的观测度。
    这包括否定线性和角的速度，预测重力，速度指令的某些组件，并翻转关节位置，关节速度和ANYmal机器人的最后动作。
    此外，如果有高度扫描数据，则将其翻向相关维度。

    参数：
        env: 观测得到的环境实例。
        obs: 需要转换的观测张量。

    返回：
        转换的观测张量与前后对称应用。
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # lin vel
    obs[:, :3] = obs[:, :3] * torch.tensor([-1, 1, 1], device=device)
    # ang vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, -1], device=device)
    # projected gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([-1, 1, 1], device=device)
    # velocity command
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([-1, 1, -1], device=device)
    # joint pos
    obs[:, 12:24] = _switch_anymal_joints_front_back(obs[:, 12:24])
    # joint vel
    obs[:, 24:36] = _switch_anymal_joints_front_back(obs[:, 24:36])
    # last actions
    obs[:, 36:48] = _switch_anymal_joints_front_back(obs[:, 36:48])

    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    if "height_scan" in env.observation_manager.active_terms["policy"]:
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[2]).view(-1, 11 * 17)

    return obs


"""
Symmetry functions for actions.
"""
"""对象函数为动作。
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    """应用左向右对称转换到动作张量。

    这个函数通过应用对左-右轴的对称性表示转换来修改给定的动作张量。
    这包括转换关节位置，关节速度和ANYmal机器人最后的操作。

    参数：
        actions: 动作张量要转换。

    返回：
        转换的动作张量与左-右对称应用。
    """
    actions = actions.clone()
    actions[:] = _switch_anymal_joints_left_right(actions[:])
    return actions


def _transform_actions_front_back(actions: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the front-back axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with front-back symmetry applied.
    """
    """应用前后对称性转换到动作张量。

    这种函数通过应用对前后轴的对称性表示转换来修改给定的动作张量。
    这包括转换关节位置，关节速度和ANYmal机器人最后的操作。

    参数：
        actions: 动作张量要转换。

    返回：
        转换的动作张量与前后对称应用。
    """
    actions = actions.clone()
    actions[:] = _switch_anymal_joints_front_back(actions[:])
    return actions


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[
    'LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA',
    'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE',
    'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE'
]

Correspondingly, the joint ordering for the ANYmal robot is:

* LF = left front --> [0, 4, 8]
* LH = left hind --> [1, 5, 9]
* RF = right front --> [2, 6, 10]
* RH = right hind --> [3, 7, 11]
"""
"""辅助函数为对称。

在Isaac Sim中，联合序列是如下: ["LF_HAA"，"LH_HAA"，"RF_HAA"，"RH_HAA"，"LF_HFE"，"LH_HFE"，"RF_HFE"，"RH_HFE"，"LF_KF
E"，"LH_KFE"，"RF_KFE"，"RH_KFE" ]

相应，ANYmal机器人的联合订单是:

* LF现在，我们要做什么?
* LH后面的左侧
* RF在前面的右侧 --> [2， 6， 10]
* RH现在，我们要做什么?
"""


def _switch_anymal_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    """应用左向右对称转换到联合数据子。"""
    joint_data_switched = torch.zeros_like(joint_data)
    # left <-- right
    joint_data_switched[..., [0, 4, 8, 1, 5, 9]] = joint_data[..., [2, 6, 10, 3, 7, 11]]
    # right <-- left
    joint_data_switched[..., [2, 6, 10, 3, 7, 11]] = joint_data[..., [0, 4, 8, 1, 5, 9]]

    # Flip the sign of the HAA joints
    joint_data_switched[..., [0, 1, 2, 3]] *= -1.0

    return joint_data_switched


def _switch_anymal_joints_front_back(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the joint data tensor."""
    """应用前后对称性转换到联合数据子。"""
    joint_data_switched = torch.zeros_like(joint_data)
    # front <-- hind
    joint_data_switched[..., [0, 4, 8, 2, 6, 10]] = joint_data[..., [1, 5, 9, 3, 7, 11]]
    # hind <-- front
    joint_data_switched[..., [1, 5, 9, 3, 7, 11]] = joint_data[..., [0, 4, 8, 2, 6, 10]]

    # Flip the sign of the HFE and KFE joints
    joint_data_switched[..., 4:] *= -1

    return joint_data_switched
