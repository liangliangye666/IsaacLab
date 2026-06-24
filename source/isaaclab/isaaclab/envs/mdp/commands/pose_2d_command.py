# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations
"""含有2D位置机动任务命令生成器的子模块。"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import TerrainBasedPose2dCommandCfg, UniformPose2dCommandCfg


class UniformPose2dCommand(CommandTerm):
    """Command generator that generates pose commands containing a 3-D position and heading.

    The command generator samples uniform 2D positions around the environment origin. It sets
    the height of the position command to the default root height of the robot. The heading
    command is either set to point towards the target or is sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.
    """
    """命令生成器，生成包含3D位置和方向的姿势命令。

    命令生成器在环境来源周围采样一致的2D位置。
    它设置位置命令的高度为机器人的默认根高度。
    方向指令要么设置向目标指向，要么采样均。
    这可以通过配置中的:attr:`Pose2dCommandCfg.simple_heading`参数进行配置。
    """

    cfg: UniformPose2dCommandCfg
    """Configuration for the command generator."""
    """命令生成器的配置。"""

    def __init__(self, cfg: UniformPose2dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        """启动命令生成器类。

        参数：
            cfg: 命令生成器的配置参数。
            env: 环境对象。
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """
    """产品
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        """在基础框架中所需的2D姿势。
        形状是 (num_envs， 4)。
        """
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

    """
    Implementation specific functions.
    """
    """具体执行功能。
    """

    def _update_metrics(self):
        # logs data
        self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))

    def _resample_command(self, env_ids: Sequence[int]):
        # obtain env origins for the environments
        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        # offset the position command by the current root position
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root state."""
        """将位置命令重定向到当前的根状态。"""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )


class TerrainBasedPose2dCommand(UniformPose2dCommand):
    """Command generator that generates pose commands based on the terrain.

    This command generator samples the position commands from the valid patches of the terrain.
    The heading commands are either set to point towards the target or are sampled uniformly.

    It expects the terrain to have a valid flat patches under the key 'target'.
    """
    """根据地形生成姿势命令的命令生成器。

    这种命令生成器从地形的有效地块中采样位置命令。
    方向指令要么设置向目标指向，要么采样均。

    它预计地形将在"目标"键下有一个有效的平坦地板。
    """

    cfg: TerrainBasedPose2dCommandCfg
    """Configuration for the command generator."""
    """命令生成器的配置。"""

    def __init__(self, cfg: TerrainBasedPose2dCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the terrain asset
        self.terrain: TerrainImporter = env.scene["terrain"]

        # obtain the valid targets from the terrain
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "The terrain-based command generator requires a valid flat patch under 'target' in the terrain."
                f" Found: {list(self.terrain.flat_patches.keys())}"
            )
        # valid targets: (terrain_level, terrain_type, num_patches, 3)
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets from the terrain
        ids = torch.randint(0, self.valid_targets.shape[2], size=(len(env_ids),), device=self.device)
        self.pos_command_w[env_ids] = self.valid_targets[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids], ids
        ]
        # offset the position command by the current root height
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)
