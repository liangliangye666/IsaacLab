# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils import configclass

from .drone_pose_command import DroneUniformPoseCommand


@configclass
class DroneUniformPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for uniform drone pose command generator."""
    """机器人统一姿势命令生成器的配置。"""

    class_type: type = DroneUniformPoseCommand
