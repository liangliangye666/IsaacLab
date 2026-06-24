# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

"""
Drone control rewards.
"""
"""无人机控制奖励。
"""


def distance_to_goal_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.0,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Reward the distance to a goal position using an exponential kernel.

    This reward computes an exponential falloff of the squared Euclidean distance
    between the commanded target position and the asset (robot) root position.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel; larger values
            produce a gentler falloff.
        command_name: Name of the command to read the target pose from the
            environment's command manager. The function expects the command
            tensor to contain positions in its first three columns.

    Returns:
        A 1-D tensor of shape (num_envs,) containing the per-environment reward
        values in [0, 1], with 1.0 when the position error is zero.
    """
    """使用指数核来奖励距离到目标位置。

    这种奖励计算了指数式下降，命令的目标位置和资产 (机器人) 根位置之间的方方曲方曲方程。

    参数：
        env: 基于管理器的RL环境实例。
        asset_cfg: SceneEntityCfg识别资产 (默认为"机器人")。
        std: 在指数核中使用的标准偏差；较大的值产生更轻微的下降。
        command_name: 环境命令管理器的命令名。
                      该函数预计命令子将在其前三列中包含位置。

    返回：
        包含[0， 1]中每个环境奖励值的1D形状数 (num_envs，) 且在位置错误为零时为1.0。
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    # compute the error
    position_error_square = torch.sum(torch.square(target_position_w - current_position), dim=1)
    return torch.exp(-position_error_square / std**2)


def ang_vel_xyz_exp(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 1.0
) -> torch.Tensor:
    """Penalize angular velocity magnitude with an exponential kernel.

    This reward computes exp(-||omega||^2 / std^2) where omega is the body-frame
    angular velocity of the asset. It is useful for encouraging low rotational
    rates.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel; controls
            sensitivity to angular velocity magnitude.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates
        zero angular velocity.
    """
    """用指数级核来惩罚角速度大小。

    奖励计算出 exp(- 这2 时时时时时时时^/std^2) 在何处 omega 是资产的体格框架角速度。
    它有助于鼓励低转速率。

    参数：
        env: 基于管理器的RL环境实例。
        asset_cfg: SceneEntityCfg识别资产 (默认为"机器人")。
        std: 在指数核中使用的标准偏差；对角速度大小的敏感性控制。

    返回：
        一个1D形状张量 (num_envs，) 值为 (0， 1)，其中1表示零角速度。
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # compute squared magnitude of angular velocity (all axes)
    ang_vel_squared = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)

    return torch.exp(-ang_vel_squared / std**2)


def lin_vel_xyz_exp(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 1.0
) -> torch.Tensor:
    """Penalize linear velocity magnitude with an exponential kernel.

    Computes exp(-||v||^2 / std^2) where v is the asset's linear velocity in
    world frame. Useful for encouraging the agent to reduce translational speed.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates
        zero linear velocity.
    """
    """通过指数核来惩罚线性速度大小。

    计算出 exp(- 时代时代时代^2 / std^2) ，其中v是资产在世界框架中的线性速度。
    有助于鼓励代理减少转换速度。

    参数：
        env: 基于管理器的RL环境实例。
        asset_cfg: SceneEntityCfg识别资产 (默认为"机器人")。
        std: 在指数核中使用的标准偏差。

    返回：
        一个形状 (num_envs，) 的1-D子，值为 (0， 1)，其中1表示零线性速度。
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # compute squared magnitude of linear velocity (all axes)
    lin_vel_squared = torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)

    return torch.exp(-lin_vel_squared / std**2)


def yaw_aligned(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.5
) -> torch.Tensor:
    """Reward alignment of the vehicle's yaw to zero using an exponential kernel.

    The function extracts the yaw (rotation about Z) from the world-frame root
    quaternion and computes exp(-yaw^2 / std^2). This encourages heading
    alignment to a zero-yaw reference.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel; smaller values
            make the reward more sensitive to yaw deviations.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates
        perfect yaw alignment (yaw == 0).
    """
    """使用指数核来将车辆的向零的奖励对齐。

    这个函数从世界框架根四方体中提取 (旋转约为Z) 并计算exp ((-yaw^2 / std^2)。
    这鼓励标题与零yaw引用保持一致。

    参数：
        env: 基于管理器的RL环境实例。
        asset_cfg: SceneEntityCfg识别资产 (默认为"机器人")。
        std: 在指数核中使用的标准偏差；较小的值使奖励对偏差更加敏感。

    返回：
        一个形状 (num_envs，) 的1-D子，值为 (0， 1)，其中1表示完美的线 (yaw == 0)。
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # extract yaw from current orientation
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)

    # normalize yaw to [-pi, pi] (target is 0)
    yaw = math_utils.wrap_to_pi(yaw)

    # return exponential reward (1 when yaw=0, approaching 0 when rotated)
    return torch.exp(-(yaw**2) / std**2)
