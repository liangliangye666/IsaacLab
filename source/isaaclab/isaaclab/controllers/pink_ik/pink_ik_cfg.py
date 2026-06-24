# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Pink IK controller."""
"""IK控制器的配置。"""

from dataclasses import MISSING

from pink.tasks import FrameTask

from isaaclab.utils import configclass


@configclass
class PinkIKControllerCfg:
    """Configuration settings for the Pink IK Controller.

    The Pink IK controller can be found at: https://github.com/stephane-caron/pink
    """
    """粉红色IK控制器的配置设置。

    粉红色IK控制器可在:https://github.com/stephane-caron/pink
    """

    urdf_path: str | None = None
    """Path to the robot's URDF file. This file is used by Pinocchio's ``robot_wrapper.BuildFromURDF``
    to load the robot model.
    """
    """路径到机器人的URDF文件。
    这份文件被皮诺基奥的``robot_wrapper.BuildFromURDF``用来加载机器人模型。
    """

    mesh_path: str | None = None
    """Path to the mesh files associated with the robot. These files are also loaded by Pinocchio's
    ``robot_wrapper.BuildFromURDF``.
    """
    """路径到与机器人相关的网格文件。
    这些文件也被皮诺基奥的``robot_wrapper.BuildFromURDF``加载。
    """

    num_hand_joints: int = 0
    """The number of hand joints in the robot.

    The action space for the controller contains the ``pose_dim(7) * num_controlled_frames + num_hand_joints``.
    The last ``num_hand_joints`` values of the action are the hand joint angles.
    """
    """机器人中的手关节数量。

    控制器的操作空间包含``pose_dim(7) * num_controlled_frames + num_hand_joints``。
    最后的``num_hand_joints``值是手关角。
    """

    variable_input_tasks: list[FrameTask] = MISSING
    """A list of tasks for the Pink IK controller.

    These tasks are controllable by the environment action.

    These tasks can be used to control the pose of a frame or the angles of joints.
    For more details, visit: https://github.com/stephane-caron/pink
    """
    """色IK控制器的任务列表。

    这些任务可以通过环境动作控制。

    这些任务可以用来控制一个框架的姿势或关节的角度。
    更多信息请访问:https://github.com/stephane-caron/pink
    """

    fixed_input_tasks: list[FrameTask] = MISSING
    """
    A list of tasks for the Pink IK controller. These tasks are fixed and not controllable by the env action.

    These tasks can be used to fix the pose of a frame or the angles of joints to a desired configuration.
    For more details, visit: https://github.com/stephane-caron/pink
    """
    """色IK控制器的任务列表。
    这些任务是固定的，不能通过env动作控制。

    这些任务可以用来将框架的姿势或关节的角度固定到所需的配置。
    更多信息请访问:https://github.com/stephane-caron/pink
    """

    joint_names: list[str] | None = None
    """A list of joint names in the USD asset controlled by the Pink IK controller.

    This is required because the joint naming conventions differ between USD and URDF files. This value is
    currently designed to be automatically populated by the action term in a manager based environment.
    """
    """在USD资产中，由粉红色IK控制器控制的联合名称列表。

    由于USD和URDF文件之间的联合命名约定不同，因此需要这样做。
    目前，该值被设计为在基于管理器的环境中自动填充的动作项。
    """

    all_joint_names: list[str] | None = None
    """A list of joint names in the USD asset.

    This is required because the joint naming conventions differ between USD and URDF files. This value is
    currently designed to be automatically populated by the action term in a manager based environment.
    """
    """在USD资产中的共同名称列表。

    由于USD和URDF文件之间的联合命名约定不同，因此需要这样做。
    目前，该值被设计为在基于管理器的环境中自动填充的动作项。
    """

    articulation_name: str = "robot"
    """The name of the articulation USD asset in the scene."""
    """在场景的USD元件的名字。"""

    base_link_name: str = "base_link"
    """The name of the base link in the USD asset."""
    """在USD资产中的基链的名称。"""

    show_ik_warnings: bool = True
    """Show warning if IK solver fails to find a solution."""
    """如果IK溶剂找不到解决方案，则显示警告。"""

    fail_on_joint_limit_violation: bool = True
    """Whether to fail on joint limit violation.

    If True, the Pink IK solver will fail and raise an error if any joint limit is violated during optimization.
    The PinkIKController will handle the error by setting the last joint positions.

    If False, the solver will ignore joint limit violations and return the closest solution found.
    """
    """无论是否会在违反联合限制方面失败。

    如果True，粉红色IK解决器会失败，如果在优化过程中违反任何关节限制，则会产生错误。
    在PinkIKController通过设置最后的关节位置来处理错误。

    如果False，解决器将忽略关节限制违规，并返回找到的最接近解决方案。
    """

    xr_enabled: bool = False
    """If True, the Pink IK controller will send information to the XRVisualization."""
    """如果True，粉红色IK控制器将向XRVisualization发送信息。"""
