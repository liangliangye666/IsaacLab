# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils

from .franka_stack_ik_abs_mimic_env import FrankaCubeStackIKAbsMimicEnv
from .franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv


class PickPlaceRelMimicEnv(FrankaCubeStackIKRelMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for DiffIK / RmpFlow Relative Pose Control env.

    This MimicEnv is used when all observations are in the robot base frame.
    """
    """艾萨克实验室仿真环境包装类 DiffIK / RmpFlow 相对姿势控制 env。

    在所有观测都在机器人基础框架中使用 MimicEnv。
    """

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Gets the pose of each object (including rigid objects and articulated objects) in the robot base frame.

        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A dictionary that maps object names to object pose matrix in robot base frame (4x4 torch.Tensor)
        """
        """在机器人基架中得到每个对象的姿势 (包括刚性对象和关节对象)。

        参数：
            env_ids: 环境索引，让你做好姿势。
                     如果是None，则考虑所有envs。

        返回：
            在机器人基础框架中映射对象名称到对象姿势矩阵的字典 (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        # Get scene state
        scene_state = self.scene.get_state(is_relative=True)
        rigid_object_states = scene_state["rigid_object"]
        articulation_states = scene_state["articulation"]

        # Get robot root pose
        robot_root_pose = articulation_states["robot"]["root_pose"]
        root_pos = robot_root_pose[env_ids, :3]
        root_quat = robot_root_pose[env_ids, 3:7]

        object_pose_matrix = dict()

        # Process rigid objects
        for obj_name, obj_state in rigid_object_states.items():
            pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                root_pos, root_quat, obj_state["root_pose"][env_ids, :3], obj_state["root_pose"][env_ids, 3:7]
            )
            rot_obj_base = PoseUtils.matrix_from_quat(quat_obj_base)
            object_pose_matrix[obj_name] = PoseUtils.make_pose(pos_obj_base, rot_obj_base)

        # Process articulated objects (except robot)
        for art_name, art_state in articulation_states.items():
            if art_name != "robot":  # Skip robot
                pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                    root_pos, root_quat, art_state["root_pose"][env_ids, :3], art_state["root_pose"][env_ids, 3:7]
                )
                rot_obj_base = PoseUtils.matrix_from_quat(quat_obj_base)
                object_pose_matrix[art_name] = PoseUtils.make_pose(pos_obj_base, rot_obj_base)

        return object_pose_matrix

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
        if "grasp" in subtask_terms:
            signals["grasp"] = subtask_terms["grasp"][env_ids]

        # Handle multiple grasp signals
        for i in range(0, len(self.cfg.subtask_configs)):
            grasp_key = f"grasp_{i + 1}"
            if grasp_key in subtask_terms:
                signals[grasp_key] = subtask_terms[grasp_key][env_ids]
        # final subtask signal is not needed
        return signals


class PickPlaceAbsMimicEnv(FrankaCubeStackIKAbsMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for DiffIK / RmpFlow Absolute Pose Control env.

    This MimicEnv is used when all observations are in the robot base frame.
    """
    """艾萨克实验室仿真环境包装类 DiffIK / RmpFlow 绝对姿势控制 env。

    在所有观测都在机器人基础框架中使用 MimicEnv。
    """

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Gets the pose of each object (including rigid objects and articulated objects) in the robot base frame.

        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A dictionary that maps object names to object pose matrix in robot base frame (4x4 torch.Tensor)
        """
        """在机器人基架中得到每个对象的姿势 (包括刚性对象和关节对象)。

        参数：
            env_ids: 环境索引，让你做好姿势。
                     如果是None，则考虑所有envs。

        返回：
            在机器人基础框架中映射对象名称到对象姿势矩阵的字典 (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        # Get scene state
        scene_state = self.scene.get_state(is_relative=True)
        rigid_object_states = scene_state["rigid_object"]
        articulation_states = scene_state["articulation"]

        # Get robot root pose
        robot_root_pose = articulation_states["robot"]["root_pose"]
        root_pos = robot_root_pose[env_ids, :3]
        root_quat = robot_root_pose[env_ids, 3:7]

        object_pose_matrix = dict()

        # Process rigid objects
        for obj_name, obj_state in rigid_object_states.items():
            pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                root_pos, root_quat, obj_state["root_pose"][env_ids, :3], obj_state["root_pose"][env_ids, 3:7]
            )
            rot_obj_base = PoseUtils.matrix_from_quat(quat_obj_base)
            object_pose_matrix[obj_name] = PoseUtils.make_pose(pos_obj_base, rot_obj_base)

        # Process articulated objects (except robot)
        for art_name, art_state in articulation_states.items():
            if art_name != "robot":  # Skip robot
                pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                    root_pos, root_quat, art_state["root_pose"][env_ids, :3], art_state["root_pose"][env_ids, 3:7]
                )
                rot_obj_base = PoseUtils.matrix_from_quat(quat_obj_base)
                object_pose_matrix[art_name] = PoseUtils.make_pose(pos_obj_base, rot_obj_base)

        return object_pose_matrix

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
        if "grasp" in subtask_terms:
            signals["grasp"] = subtask_terms["grasp"][env_ids]

        # Handle multiple grasp signals
        for i in range(0, len(self.cfg.subtask_configs)):
            grasp_key = f"grasp_{i + 1}"
            if grasp_key in subtask_terms:
                signals[grasp_key] = subtask_terms[grasp_key][env_ids]
        # final subtask signal is not needed
        return signals
