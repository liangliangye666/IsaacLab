# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create drone observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations
"""可用于创建无人机观测项的共同函数。

函数可以传递到:class:`isaaclab.managers.ObservationTermCfg`对象，以实现函数引入的观测。
"""

from typing import TYPE_CHECKING

import torch
import torch.jit

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab_contrib.assets import Multirotor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_shape

"""
State.
"""
"""州。
"""


def base_roll_pitch(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the base roll and pitch in the simulation world frame.

    Parameters:
        env: Manager-based environment providing the scene and tensors.
        asset_cfg: Scene entity config pointing to the target robot (default: "robot").

    Returns:
        torch.Tensor: Shape (num_envs, 2). Column 0 is roll, column 1 is pitch.
        Values are radians normalized to [-pi, pi], expressed in the world frame.

    Notes:
        - Euler angles are computed from asset.data.root_quat_w using XYZ convention.
        - Only roll and pitch are returned; yaw is omitted.
    """
    """在仿真世界框架中返回基滚动。

    Parameters:
        env: 管理器提供场景和子。
        asset_cfg: 对目标机器人的场景实体配置 (默认:"机器人")。

    返回：
        torch.Tensor: 形状 (num_envs， 2)。
                      列0是滚动，列1是发射。
        值是以 [-pi，pi]为正常的半径，

    说明：
        - 勒角是从asset.data.root_quat_w计算的，使用XYZ公约。
        - 只返回滚动和；是省略的。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to [-pi, pi]
    roll = math_utils.wrap_to_pi(roll)
    pitch = math_utils.wrap_to_pi(pitch)

    return torch.cat((roll.unsqueeze(-1), pitch.unsqueeze(-1)), dim=-1)


"""
Commands.
"""
"""命令。
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def generated_drone_commands(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Generate a body-frame direction and distance to the commanded position.

    This observation reads a command from env.command_manager identified by command_name,
    interprets its first three components as a target position in the world frame, and
    returns:
        [dir_x, dir_y, dir_z, distance]
    where dir_* is the unit vector from the current body origin to the target, expressed
    in the multirotor body (root link) frame, and distance is the Euclidean separation.

    Parameters:
        env: Manager-based RL environment providing scene and command manager.
        command_name: Name of the command term to query from the command manager.
        asset_cfg: Scene entity config for the multirotor asset (default: "robot").

    Returns:
        torch.Tensor: Shape (num_envs, 4) with body-frame unit direction (3) and distance (1).

    Frame conventions:
        - Current position is asset.data.root_pos_w relative to env.scene.env_origins (world frame).
        - Body orientation uses asset.data.root_link_quat_w to rotate world vectors into the body frame.

    Assumptions:
        - env.command_manager.get_command(command_name) returns at least three values
          representing a world-frame target position per environment.
        - A small epsilon (1e-8) is used to guard against zero-length direction vectors.
    """
    """产生一个身体框架方向和距离到命令位置。

    这项观测是从env.command_manager由command_name，将其前三个组成部分解释为世界框架中的目标位置，
    returns: [dir_x，dir_y，dir_z，距离]
    在此，dir_*是从当前体源到目标的单元向量，表达在多轮体 (根链) 框架中，距离是尤克利德分离。

    Parameters:
        env: 基于管理器的RL环境提供场景和命令管理器。
        command_name: 命令管理器的查询命令项名称。
        asset_cfg: 场景实体对多机器资产的配置 (默认:"机器人")。

    返回：
        torch.Tensor: 形状 (num_envs， 4) 与车身框架单元方向 (3) 和距离 (1)。

    框架会议:
        - 目前位置是asset.data.root_pos_w与env.scene.env_origins相对 (世界框架)。
        - 身体导向使用asset.data.root_link_quat_w将世界向量转换到身体框架中。

    Assumptions:
        - env.command_manager.get_command(command_name) 返回每环境至少3个代表世界框架目标位置的值。
        - 为了保护对零长度方向向量，使用一个小的子 (1e-8)。
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    current_position_w = asset.data.root_pos_w - env.scene.env_origins
    command = env.command_manager.get_command(command_name)
    current_position_b = math_utils.quat_apply_inverse(asset.data.root_link_quat_w, command[:, :3] - current_position_w)
    current_position_b_dir = current_position_b / (torch.norm(current_position_b, dim=-1, keepdim=True) + 1e-8)
    current_position_b_mag = torch.norm(current_position_b, dim=-1, keepdim=True)
    return torch.cat((current_position_b_dir, current_position_b_mag), dim=-1)
