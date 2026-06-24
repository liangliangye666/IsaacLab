# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class FrankaCubeStackIKRelMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for Franka Cube Stack IK Rel env.
    """
    """艾萨克实验室仿真环境包装类IK铁路env。
    """

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

        # Retrieve end effector pose from the observation buffer
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

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
            noise: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        """执行目标姿势和对最终效果控制器的抓住作用，并返回一个操作 (通常是正常化的三角形姿势) 试图实现目标姿势。
        如果指定，将噪音添加到目标姿势操作中。

        参数：
            target_eef_pose_dict: 每个末端执行器的4×4目标效应。
            gripper_action_dict: 每个末端执行器的抓住器操作字典。
            noise: 噪音增加了动作。
                   如果None，则不会增加噪音。
            env_id: 环境索引，以获得动作。

        返回：
            一个与env.step兼容的torch.Tensor动作。
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # normalized delta position action
        delta_position = target_pos - curr_pos

        # normalized delta rotation action
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        # get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        # add noise to action
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action: Environment action. Shape is (num_envs, action_dim)

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to
        """
        """将动作 (与env.step兼容) 转换为最终效应控制器的目标姿势。
        转换为 @target_eef_pose_to_action。
        通常用于推断目标控制器姿势的序列
        from a demonstration trajectory using the recorded actions.

        参数：
            action: 环境动作
                    形状是 (num_envs，action_dim)

        返回：
            一个字典的eef形象torch.Tensor，
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # get pose target
        target_pos = curr_pos + delta_position

        # Convert delta_rotation to axis angle form
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        # Handle invalid division for the case when delta_rotation_angle is close to zero
        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

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
        # last dimension is gripper action
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. The implementation of this method is
        required if intending to enable automatic subtask term signal annotation when running the
        dataset annotation tool. This method can be kept unimplemented if intending to use manual
        subtask term signal annotation.

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        """在任务中的每个子任务中，得到终止信号标志的字典。
        在完成子任务时，标志是1和否则是0。
        如果打算在运行数据集注释工具时启用自动子任务项信号注释，则需要实施这种方法。
        如果要使用手动的子任务项信号注释，则可以保持这种方法未实施。

        参数：
            env_ids: 环境索引，以获得终止信号。
                     如果是None，则考虑所有envs。

        返回：
            字典终止信号标志 (False或True) 对于每个子任务。
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["grasp_1"] = subtask_terms["grasp_1"][env_ids]
        signals["grasp_2"] = subtask_terms["grasp_2"][env_ids]
        signals["stack_1"] = subtask_terms["stack_1"][env_ids]
        # final subtask is placing cubeC on cubeA (motion relative to cubeA) - but final subtask signal is not needed
        return signals

    def get_expected_attached_object(self, eef_name: str, subtask_index: int, env_cfg) -> str | None:
        """
        (SkillGen) Return the expected attached object for the given EEF/subtask.

        Assumes 'stack' subtasks place the object grasped in the preceding 'grasp' subtask.
        Returns None for 'grasp' (or others) at subtask start.
        """
        """(SkillGen) 返回给定的EEF/子任务的预期附件对象。

        假设"堆"子任务将被抓到的对象放置在之前的"抓"子任务中。
        在子任务开始时返回None为"grasp" (或其他)
        """
        if eef_name not in env_cfg.subtask_configs:
            return None

        subtask_configs = env_cfg.subtask_configs[eef_name]
        if not (0 <= subtask_index < len(subtask_configs)):
            return None

        current_cfg = subtask_configs[subtask_index]
        # If stacking, expect we are holding the object grasped in the prior subtask
        if "stack" in str(current_cfg.subtask_term_signal).lower():
            if subtask_index > 0:
                prev_cfg = subtask_configs[subtask_index - 1]
                if "grasp" in str(prev_cfg.subtask_term_signal).lower():
                    return prev_cfg.object_ref
        return None
