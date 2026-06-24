# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLEnv


def optional_method(func):
    """Decorator to mark a method as optional."""
    """装饰仪标记方法为可选。"""
    func.__is_optional__ = True
    return func


class ManagerBasedRLMimicEnv(ManagerBasedRLEnv):
    """The superclass for the Isaac Lab Mimic environments.

    This class inherits from :class:`ManagerBasedRLEnv` and provides a template for the functions that
    need to be defined to run the Isaac Lab Mimic data generation workflow. The Isaac Lab data generation
    pipeline, inspired by the MimicGen system, enables the generation of new datasets based on a few human
    collected demonstrations. MimicGen is a novel approach designed to automatically synthesize large-scale,
    rich datasets from a sparse set of human demonstrations by adapting them to new contexts. It manages to
    replicate the benefits of large datasets while reducing the immense time and effort usually required to
    gather extensive human demonstrations.

    The MimicGen system works by parsing demonstrations into object-centric segments. It then adapts
    these segments to new scenes by transforming each segment according to the new scene’s context, stitching
    them into a coherent trajectory for a robotic end-effector to execute. This approach allows learners to train
    proficient agents through imitation learning on diverse configurations of scenes, object instances, etc.

    Key Features:
        - Efficient Dataset Generation: Utilizes a small set of human demos to produce large scale demonstrations.
        - Broad Applicability: Capable of supporting tasks that require a range of manipulation skills, such as
          pick-and-place and interacting with articulated objects.
        - Dataset Versatility: The synthetic data retains a quality that compares favorably with additional human demos.
    """
    """对于艾萨克实验室模仿环境的超类型。

    这个类继承了:class:`ManagerBasedRLEnv`，并为运行Isaac Lab Mimic数据生成工作流需要定义的函数提供了一个模板。
    以MimicGen系统为灵感，基于少数收集的人类示范的数据集，
    MimicGen是一种新型方法，旨在自动合成大规模，丰富的数据集，
    它能够复制大型数据集的好处，同时减少通常收集广泛的人类示范所需的巨大时间和精力。

    在MimicGen系统通过将示范分为对象中心的细分进行分析。
    然后通过根据新场景的背景改变每个段落，将这些段落调整到新的场景中，将它们编织成一个连贯的轨迹，用于机器人最终效应器执行。
    这种方法允许学习者通过模仿学习培训熟练的代理人，

    主要特征:
        - 有效的数据集生成:利用一个小组的人类演示进行大规模演示。
        - 广泛适用性:能够支持需要多种操纵技能的任务，例如选择和放置和与关键物体交互。
        - 数据集多功能性:合成数据保持了与额外的人类演示相比较优质的质量。
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
        raise NotImplementedError

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
            env_id: Environment index to compute the action for.

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
            env_id: 环境索引来计算动作。

        返回：
            一个与env.step兼容的torch.Tensor动作。
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Gets the pose of each object relevant to Isaac Lab Mimic data generation in the current scene.

        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A dictionary that maps object names to object pose matrix (4x4 torch.Tensor)
        """
        """在当前场景中，每个对象的姿势与艾萨克实验室仿真数据生成相关。

        参数：
            env_ids: 环境索引，让你做好姿势。
                     如果是None，则考虑所有envs。

        返回：
            一个对象名称映射到对象姿势矩阵的字典 (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        rigid_object_states = self.scene.get_state(is_relative=True)["rigid_object"]
        object_pose_matrix = dict()
        for obj_name, obj_state in rigid_object_states.items():
            object_pose_matrix[obj_name] = PoseUtils.make_pose(
                obj_state["root_pose"][env_ids, :3], PoseUtils.matrix_from_quat(obj_state["root_pose"][env_ids, 3:7])
            )
        return object_pose_matrix

    def get_subtask_start_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of start signal flags for each subtask in a task. The flag is 1
        when the subtask has started and 0 otherwise. The implementation of this method is
        required if intending to enable automatic subtask start signal annotation when running the
        dataset annotation tool. This method can be kept unimplemented if intending to use manual
        subtask start signal annotation.

        Args:
            env_ids: Environment indices to get the start signals for. If None, all envs are considered.

        Returns:
            A dictionary start signal flags (False or True) for each subtask.
        """
        """在任务中的每一个子任务中得到了启动信号标志的字典。
        如果该子任务开始，则标志为1，否则为0。
        如果打算在运行数据集注释工具时启用自动子任务启动信号注释，则需要实现此方法。
        如果打算使用手动的子任务启动信号注释，则可保持这种方法未实施。

        参数：
            env_ids: 环境索引，以获得启动信号。
                     如果是None，则考虑所有envs。

        返回：
            每个子任务的字典启动信号标志 (False或True)。
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        """在字典中保存所有必要的信息，
        这与在hdf5数据集中存储的 @env_meta -环境元数据相同，并用于utils/env_utils.py。
        """
        return dict(env_name=self.spec.id, type=2, env_kwargs=dict())

    @optional_method
    def get_navigation_state(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Optional method. Only required when using navigation controller locomanipulation data generation.

        Gets the navigation state of the robot. Required when use of the navigation controller is
        enabled. The navigation state includes a boolean flag "is_navigating" to indicate when the
        robot is under control by the navigation controller, and a boolean flag "navigation_goal_reached"
        to indicate when the navigation goal has been reached.

        Args:
            env_ids: The environment index to get the navigation state for. If None, all envs are considered.

        Returns:
            A dictionary that of navigation state flags (False or True).
        """
        """选择性方法。
        只有在使用导航控制器位置操作数据生成时才需要。

        得到机器人的导航状态。
        在启用导航控制器时要求。
        导航状态包括"is_navigating"的布鲁尔旗，用于表示机器人在导航控制器控制下时，以及"navigation_goal_reached"的布鲁尔旗，用于表示导航目标达到什么时候。

        参数：
            env_ids: 导航状态的环境索引。
                     如果是None，则考虑所有envs。

        返回：
            导航国旗的字典 (False或True)。
        """
        raise NotImplementedError
