# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.controllers.pink_ik import PinkIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import pink_task_space_actions


@configclass
class PinkInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for Pink inverse kinematics action term.

    This configuration is used to define settings for the Pink inverse kinematics action term,
    which is a inverse kinematics framework.
    """
    """Pink色反动动力学动作项的配置。

    这种配置用于定义粉红色逆动力学动作项的设置，这是逆动力学框架。
    """

    class_type: type[ActionTerm] = pink_task_space_actions.PinkInverseKinematicsAction
    """Specifies the action term class type for Pink inverse kinematics action."""
    """指定为粉红色反动动力学动作的动作项类型。"""

    pink_controlled_joint_names: list[str] = MISSING
    """List of joint names or regular expression patterns that specify the joints controlled by pink IK."""
    """列出由粉红色IK控制的关节的联合名称或定期表达模式。"""

    hand_joint_names: list[str] = MISSING
    """List of joint names or regular expression patterns that specify the joints controlled by hand retargeting."""
    """列出指标或常规表达模式的联合名称，指标指标的关节由手动重定向控制。"""

    controller: PinkIKControllerCfg = MISSING
    """Configuration for the Pink IK controller that will be used to solve the inverse kinematics."""
    """Pink色IK控制器的配置，用于解决逆动力学。"""

    enable_gravity_compensation: bool = True
    """Whether to compensate for gravity in the Pink IK controller."""
    """在粉红色IK控制器中是否补偿重力。"""

    target_eef_link_names: dict[str, str] = MISSING
    """Dictionary mapping task names to controlled link names for the Pink IK controller.

    This dictionary should map the task names (e.g., 'left_wrist', 'right_wrist') to the
    corresponding link names in the URDF that will be controlled by the IK solver.
    """
    """字典映射任务名称到 Pink IK控制器的控制链接名称。

    该字典应将任务名称 (e.g.， "left_wrist"， "right_wrist") 映射到URDF中的相应链接名称，这些链接将由IK解决器控制。
    """
