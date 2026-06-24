# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This sub-module contains the functions that can be used to enable Spot randomizations.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the randomization introduced by the function.
"""

from __future__ import annotations
"""该子模块包含可用于实现 Spot 随机定位的函数。

函数可以传递到:class:`isaaclab.managers.EventTermCfg`对象，以实现函数引入的随机化。
"""

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_joints_around_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints in the interval around the default position and velocity by the given ranges.

    This function samples random values from the given ranges around the default joint positions and velocities.
    The ranges are clipped to fit inside the soft joint limits. The sampled values are then set into the physics
    simulation.
    """
    """按给定的范围重置机器人关节，

    这种函数在默认的关节位置和速度周围的给定的范围中抽取随机值。
    它们的范围被裁剪，以便适合柔软关节的边界。
    然后将样本值设置在物理仿真中。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_min_pos = asset.data.default_joint_pos[env_ids] + position_range[0]
    joint_max_pos = asset.data.default_joint_pos[env_ids] + position_range[1]
    joint_min_vel = asset.data.default_joint_vel[env_ids] + velocity_range[0]
    joint_max_vel = asset.data.default_joint_vel[env_ids] + velocity_range[1]
    # clip pos to range
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, ...]
    joint_min_pos = torch.clamp(joint_min_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    joint_max_pos = torch.clamp(joint_max_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    # clip vel to range
    joint_vel_abs_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_min_vel = torch.clamp(joint_min_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    joint_max_vel = torch.clamp(joint_max_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    # sample these values randomly
    joint_pos = sample_uniform(joint_min_pos, joint_max_pos, joint_min_pos.shape, joint_min_pos.device)
    joint_vel = sample_uniform(joint_min_vel, joint_max_vel, joint_min_vel.shape, joint_min_vel.device)
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
