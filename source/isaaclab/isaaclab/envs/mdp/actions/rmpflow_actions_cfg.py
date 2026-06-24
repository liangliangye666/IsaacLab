# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

from isaaclab.controllers.rmp_flow import RmpFlowControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import rmpflow_task_space_actions


@configclass
class RMPFlowActionCfg(ActionTermCfg):
    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """
        """从父母的框架到孩子的框架。

        在许多机器人上，最终效应器框架是虚构的框架，
        在这种情况下，更容易定义这个变化w.r.t。
        它们的父母的身体是固定的。
        例如，对于Franka Emika臂，末端执行器的定义是对"panda_hand"框架的偏移。
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        """翻译w.r.t
        它们的母体。
        在 (0.0，0.0，0.0) 之前的默认设置。
        """
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    class_type: type[ActionTerm] = rmpflow_task_space_actions.RMPFlowAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    """列出将该动作映射到的联合名称或regex表达式。"""
    body_name: str = MISSING
    """Name of the body or frame for which IK is performed."""
    """执行IK的机体或框架名称。"""
    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""
    """目标框架 w.r.t的抵消。
    在身体框架。
    默认对None的缺陷，在这种情况下，不使用任何抵消。
    """
    scale: float | tuple[float, ...] = 1.0

    controller: RmpFlowControllerCfg = MISSING

    articulation_prim_expr: str = MISSING  # The expression to find the articulation prim paths.
    """The configuration for the RMPFlow controller."""
    """RMPFlow控制器的配置。"""

    use_relative_mode: bool = False
    """
    Defaults to False.
    If True, then the controller treats the input command as a delta change in the position/pose.
    Otherwise, the controller treats the input command as the absolute position/pose.
    """
    """默认为 False。
    如果 True，则控制器将输入命令视为位置/位置的多角变化。
    否则，控制器将输入命令视为绝对位置/位置。
    """
