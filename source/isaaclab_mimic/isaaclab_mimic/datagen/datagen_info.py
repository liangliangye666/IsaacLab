# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Defines the structure of information required from an environment for data generation processes."""
"""定义数据生成过程中环境所需的信息结构。"""

from copy import deepcopy


class DatagenInfo:
    """Defines the structure of information required from an environment for data generation processes.

    The :class:`DatagenInfo` class centralizes all essential data elements needed for data generation in one place,
    reducing the overhead and complexity of repeatedly querying the environment whenever this information
    is needed.

    To allow for flexibility,not all information must be present.

    Core Elements:

    - **eef_pose**: Captures the current 6 dimensional poses of the robot's end-effector.
    - **object_poses**: Captures the 6 dimensional poses of relevant objects in the scene.
    - **subtask_start_signals**: Captures subtask start signals. Used by skillgen to identify
      the precise start of a subtask from a demonstration.
    - **subtask_term_signals**: Captures subtask completions signals.
    - **target_eef_pose**: Captures the target 6 dimensional poses for robot's end effector at each time step.
    - **gripper_action**:  Captures the gripper's state.
    """
    """定义数据生成过程中环境所需的信息结构。

    The :类:`DatagenInfo`类将所有数据生成所需的基本数据元素集中在一个地方，
    减少每当需要这些信息时，不断查询环境的总费用和复杂性。

    为了实现灵活性，并非所有信息都必须存在。

    核心元素:

    - **eef_pose**:捕捉机器人的末端执行器的当前六维姿势。
    - **object_poses**:捕捉场景中相关物体的六维姿势。
    - **subtask_start_signals**:捕获子任务启动信号。 用于识别从演示中确切地启动子任务。
    - **subtask_term_signals**:捕获子任务完成信号。
    - **target_eef_pose**:每个时间步骤都捕捉到机器人的最终效应器的目标六维姿势。
    - **gripper_action**: 捕获抓住器的状态。
    """

    def __init__(
        self,
        eef_pose=None,
        object_poses=None,
        subtask_term_signals=None,
        subtask_start_signals=None,
        target_eef_pose=None,
        gripper_action=None,
    ):
        """Initialize the DatagenInfo object.

        Args:
            eef_pose (torch.Tensor or None): robot end effector poses of shape [..., 4, 4]
            object_poses (dict or None): dictionary mapping object name to object poses
                of shape [..., 4, 4]
            subtask_start_signals (dict or None): dictionary mapping subtask name to a binary
                indicator (0 or 1) on whether subtask has started. This is required when using skillgen.
                Each value in the dictionary could be an int, float, or torch.Tensor of shape [..., 1].
            subtask_term_signals (dict or None): dictionary mapping subtask name to a binary
                indicator (0 or 1) on whether subtask has been completed. Each value in the
                dictionary could be an int, float, or torch.Tensor of shape [..., 1].
            target_eef_pose (torch.Tensor or None): target end effector poses of shape [..., 4, 4]
            gripper_action (torch.Tensor or None): gripper actions of shape [..., D] where D
                is the dimension of the gripper actuation action for the robot arm
        """
        """启动DatagenInfo对象。

        参数：
            eef_pose (torch.Tensor or None): 机器人末端执行器姿势的形状 [...， 4， 4]
            object_poses (dict or None): 字典映射对象名称对象姿态的形状 [...， 4， 4]
            subtask_start_signals (dict or None): 字典映射子任务名称向二进制指标 (0或 1) 确定子任务是否启动。
                                                  在使用skillgen时需要这样做。
                                                  字典中的每个值可以是 int， float，或 torch.Tensor的形状 [...， 1]。
            subtask_term_signals (dict or None): 字典映射子任务名称向二进制指标 (0或 1) 确定子任务是否完成。
                                                 字典中的每个值可以是 int， float，或 torch.Tensor的形状 [...， 1]。
            target_eef_pose (torch.Tensor or None): 形状的目标末端执行器姿势 [...， 4， 4]
            gripper_action (torch.Tensor or None): [...， D]形状的抓紧机动，其中D是机器人臂的抓紧机动动的尺寸
        """
        self.eef_pose = None
        if eef_pose is not None:
            self.eef_pose = eef_pose

        self.object_poses = None
        if object_poses is not None:
            self.object_poses = {k: object_poses[k] for k in object_poses}

        # When using skillgen, demonstrations must be annotated with subtask start signals.
        self.subtask_start_signals = None
        if subtask_start_signals is not None:
            self.subtask_start_signals = dict()
            for k in subtask_start_signals:
                if isinstance(subtask_start_signals[k], (float, int)):
                    self.subtask_start_signals[k] = subtask_start_signals[k]
                else:
                    # Only create torch tensor if value is not a single value
                    self.subtask_start_signals[k] = subtask_start_signals[k]

        self.subtask_term_signals = None
        if subtask_term_signals is not None:
            self.subtask_term_signals = dict()
            for k in subtask_term_signals:
                if isinstance(subtask_term_signals[k], (float, int)):
                    self.subtask_term_signals[k] = subtask_term_signals[k]
                else:
                    # only create torch tensor if value is not a single value
                    self.subtask_term_signals[k] = subtask_term_signals[k]

        self.target_eef_pose = None
        if target_eef_pose is not None:
            self.target_eef_pose = target_eef_pose

        self.gripper_action = None
        if gripper_action is not None:
            self.gripper_action = gripper_action

    def to_dict(self) -> dict:
        """Convert this instance to a dictionary containing the same information.

        Returns:
            A dictionary containing the same information as this instance.
        """
        """将此例转换为包含相同信息的字典。

        返回：
            一个包含与本案相同的信息的字典。
        """
        ret = dict()
        if self.eef_pose is not None:
            ret["eef_pose"] = self.eef_pose
        if self.object_poses is not None:
            ret["object_poses"] = deepcopy(self.object_poses)
        if self.subtask_start_signals is not None:
            ret["subtask_start_signals"] = deepcopy(self.subtask_start_signals)
        if self.subtask_term_signals is not None:
            ret["subtask_term_signals"] = deepcopy(self.subtask_term_signals)
        if self.target_eef_pose is not None:
            ret["target_eef_pose"] = self.target_eef_pose
        if self.gripper_action is not None:
            ret["gripper_action"] = self.gripper_action
        return ret
