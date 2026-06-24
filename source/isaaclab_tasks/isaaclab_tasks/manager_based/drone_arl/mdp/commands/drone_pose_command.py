# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations
"""含有可追踪姿势的命令生成器的子模块。"""

import torch

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error


class DroneUniformPoseCommand(UniformPoseCommand):
    """Drone-specific UniformPoseCommand extensions.

    This class customizes the generic :class:`UniformPoseCommand` for drone (multirotor)
    use-cases. Main differences and additions:

    - Transforms pose commands from the drone's base frame to the world frame before use.
    - Accounts for per-environment origin offsets (``scene.env_origins``) when computing
        position errors so tasks running on shifted/sub-terrain environments compute
        meaningful errors.
    - Computes and exposes simple metrics used by higher-level code: ``position_error``
        and ``orientation_error`` (stored in ``self.metrics``).
    - Provides a debug visualization callback that renders the goal pose (with
        sub-terrain shift) and current body pose using the existing visualizers.

    The implementation overrides :meth:`_update_metrics` and :meth:`_debug_vis_callback`
    from the base class to implement these drone-specific behaviors.
    """
    """专用无人机的UniformPoseCommand扩展。

    该类为无人机 (多动机) 使用案例定制了通用:class:`UniformPoseCommand`。
    主要的区别和补充:

    - 在使用前将无人机的基框架转换为世界框架。
    - 计算位置错误时，每环境的原始抵消 (``scene.env_origins``) 计算，因此在移动/地下环境中运行的任务计算了有意义的错误。
    - 计算和揭露高层代码使用的简单指标:``position_error``和``orientation_error`` (存储在``self.metrics``中)。
    - 提供一个调试可视化回调，通过现有可视化器将目标姿势 (地下转移) 和当前身体姿势转换。

    执行将取决于:meth:`_update_metrics`和:meth:`_debug_vis_callback`
    from the base class to implement these drone-specific behaviors.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            # Sub-terrain shift for correct position error calculation @grzemal
            self.pose_command_b[:, :3] + self._env.scene.env_origins,
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        # Sub-terrain shift for visualization purposes @grzemal
        self.goal_pose_visualizer.visualize(
            self.pose_command_b[:, :3] + self._env.scene.env_origins, self.pose_command_b[:, 3:]
        )
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])
