# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class LocomanipulationG1MimicEnv(ManagerBasedRLMimicEnv):
    """G1 Locomanipulation Mimic environment."""
    """模仿环境。"""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        """现在就把机器人最终效果姿势。
        机器人终端效应控制器使用的框架应该相同。

        参数：
            eef_name: 终端有效者的名称。
            env_ids: 环境索引，让你做好姿势。
                     如果是None，则考虑所有envs。

        返回：
            一个torch.Tensoreef姿势矩阵。
            形状是 (len(env_ids)， 4， 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        eef_pos_name = f"{eef_name}_eef_pos"
        eef_quat_name = f"{eef_name}_eef_quat"

        target_wrist_position = self.obs_buf["policy"][eef_pos_name][env_ids]
        target_rot_mat = PoseUtils.matrix_from_quat(self.obs_buf["policy"][eef_quat_name][env_ids])

        return PoseUtils.make_pose(target_wrist_position, target_rot_mat)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            action_noise_dict: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        """执行目标姿势和对最终效果控制器的抓住作用，并返回一个操作 (通常是正常化的三角形姿势) 试图实现目标姿势。
        如果指定，将噪音添加到目标姿势操作中。

        参数：
            target_eef_pose_dict: 每个末端执行器的4×4目标效应。
            gripper_action_dict: 每个末端执行器的抓住器操作字典。
            action_noise_dict: 噪音增加了动作。
                               如果None，则不会增加噪音。
            env_id: 环境索引，以获得动作。

        返回：
            一个与env.step兼容的torch.Tensor动作。
        """

        # target position and rotation
        target_left_eef_pos, left_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["left"])
        target_right_eef_pos, right_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["right"])

        target_left_eef_rot_quat = PoseUtils.quat_from_matrix(left_target_rot)
        target_right_eef_rot_quat = PoseUtils.quat_from_matrix(right_target_rot)

        # gripper actions
        left_gripper_action = gripper_action_dict["left"]
        right_gripper_action = gripper_action_dict["right"]

        if action_noise_dict is not None:
            pos_noise_left = action_noise_dict["left"] * torch.randn_like(target_left_eef_pos)
            pos_noise_right = action_noise_dict["right"] * torch.randn_like(target_right_eef_pos)
            quat_noise_left = action_noise_dict["left"] * torch.randn_like(target_left_eef_rot_quat)
            quat_noise_right = action_noise_dict["right"] * torch.randn_like(target_right_eef_rot_quat)

            target_left_eef_pos += pos_noise_left
            target_right_eef_pos += pos_noise_right
            target_left_eef_rot_quat += quat_noise_left
            target_right_eef_rot_quat += quat_noise_right

        return torch.cat(
            (
                target_left_eef_pos,
                target_left_eef_rot_quat,
                target_right_eef_pos,
                target_right_eef_rot_quat,
                left_gripper_action,
                right_gripper_action,
            ),
            dim=0,
        )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action: Environment action. Shape is (num_envs, action_dim).

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to.
        """
        """将动作 (与env.step兼容) 转换为最终效应控制器的目标姿势。
        转换为 @target_eef_pose_to_action。
        通常用于推断目标控制器姿势的序列
        from a demonstration trajectory using the recorded actions.

        参数：
            action: 环境动作
                    形状是 (num_envs，action_dim)。

        返回：
            一个eef字典 torch.Tensor的字典， @action与。
        """
        target_poses = {}

        target_left_wrist_position = action[:, 0:3]
        target_left_rot_mat = PoseUtils.matrix_from_quat(action[:, 3:7])
        target_pose_left = PoseUtils.make_pose(target_left_wrist_position, target_left_rot_mat)
        target_poses["left"] = target_pose_left

        target_right_wrist_position = action[:, 7:10]
        target_right_rot_mat = PoseUtils.matrix_from_quat(action[:, 10:14])
        target_pose_right = PoseUtils.make_pose(target_right_wrist_position, target_right_rot_mat)
        target_poses["right"] = target_pose_right

        return target_poses

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        """从一系列env动作中提取抓住器动力部分 (与env.step兼容)。

        参数：
            actions: 环境动作。
                     形状是 (num_envs，演示中的步骤数，action_dim)。

        返回：
            一个torch.Tensor抓住器动作字典。
            每个句子的关键是eef_name。
        """
        return {"left": actions[:, 14:21], "right": actions[:, 21:]}
