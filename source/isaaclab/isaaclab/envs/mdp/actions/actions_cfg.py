# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import (
    binary_joint_actions,
    joint_actions,
    joint_actions_to_limits,
    non_holonomic_actions,
    surface_gripper_actions,
    task_space_actions,
)

##
# Joint actions.
##


@configclass
class JointActionCfg(ActionTermCfg):
    """Configuration for the base joint action term.

    See :class:`JointAction` for more details.
    """
    """基本联合动作项的配置

    See :分类:`JointAction` 详细信息。
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    """列出将该动作映射到的联合名称或regex表达式。"""
    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    """动作的尺度因子 (regex表达式的浮动或定位)。
    默认到1.0。
    """
    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
    """动作的抵消因素 (regex表达式的浮动或定值)。
    默认为0.0。
    """
    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""
    """在动作输出中是否保留联合名称的顺序。
    默认为 False。
    """


@configclass
class JointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """
    """联合立场动作项的配置

    See :分类:`JointPositionAction` 详细信息。
    """

    class_type: type[ActionTerm] = joint_actions.JointPositionAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """
    """在关节资产中配置的默认关联位置是否应作为抵消。
    默认为 True。

    如果 True，该标志将:attr:`offset`的值重写到默认的关节位置
    from the articulation asset.
    """


@configclass
class RelativeJointPositionActionCfg(JointActionCfg):
    """Configuration for the relative joint position action term.

    See :class:`RelativeJointPositionAction` for more details.
    """
    """对相对关节位置动作项的配置。

    See :分类:`RelativeJointPositionAction` 详细信息。
    """

    class_type: type[ActionTerm] = joint_actions.RelativeJointPositionAction

    use_zero_offset: bool = True
    """Whether to ignore the offset defined in articulation asset. Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to zero.
    """
    """在关节资产中定义的抵消是否被忽视。
    默认为 True。

    如果是True，这个标志将:attr:`offset`的值重写为零。
    """


@configclass
class JointVelocityActionCfg(JointActionCfg):
    """Configuration for the joint velocity action term.

    See :class:`JointVelocityAction` for more details.
    """
    """对于关节速度动作项的配置。

    See :分类:`JointVelocityAction` 详细信息。
    """

    class_type: type[ActionTerm] = joint_actions.JointVelocityAction

    use_default_offset: bool = True
    """Whether to use default joint velocities configured in the articulation asset as offset.
    Defaults to True.

    This overrides the settings from :attr:`offset` if set to True.
    """
    """在关节资产中配置的默认关节速度是否应作为抵消。
    默认为 True。

    如果设置为True，则将:attr:`offset`的设置取消。
    """


@configclass
class JointEffortActionCfg(JointActionCfg):
    """Configuration for the joint effort action term.

    See :class:`JointEffortAction` for more details.
    """
    """共同努力动作项的配置。

    See :分类:`JointEffortAction` 详细信息。
    """

    class_type: type[ActionTerm] = joint_actions.JointEffortAction


##
# Joint actions rescaled to limits.
##


@configclass
class JointPositionToLimitsActionCfg(ActionTermCfg):
    """Configuration for the bounded joint position action term.

    See :class:`JointPositionToLimitsAction` for more details.
    """
    """限制关节位置动作项的配置

    See :分类:`JointPositionToLimitsAction` 详细信息。
    """

    class_type: type[ActionTerm] = joint_actions_to_limits.JointPositionToLimitsAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    """列出将该动作映射到的联合名称或regex表达式。"""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    """动作的尺度因子 (regex表达式的浮动或定位)。
    默认到1.0。
    """

    rescale_to_limits: bool = True
    """Whether to rescale the action to the joint limits. Defaults to True.

    If True, the input actions are rescaled to the joint limits, i.e., the action value in
    the range [-1, 1] corresponds to the joint lower and upper limits respectively.

    Note:
        This operation is performed after applying the scale factor.
    """
    """否将动作重新扩展到联合限制。
    默认为 True。

    如果True，输入操作将重新扩展到联合限度，i.e.，范围内的动作值[-1， 1]相应于分别的联合下限和上限。

    说明：
        这种操作是在使用尺度因子后进行的。
    """

    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""
    """在动作输出中是否保留联合名称的顺序。
    默认为 False。
    """


@configclass
class EMAJointPositionToLimitsActionCfg(JointPositionToLimitsActionCfg):
    """Configuration for the exponential moving average (EMA) joint position action term.

    See :class:`EMAJointPositionToLimitsAction` for more details.
    """
    """对指数移动平均值 (EMA) 关节位置动作项的配置。

    See :分类:`EMAJointPositionToLimitsAction` 详细信息。
    """

    class_type: type[ActionTerm] = joint_actions_to_limits.EMAJointPositionToLimitsAction

    alpha: float | dict[str, float] = 1.0
    """The weight for the moving average (float or dict of regex expressions). Defaults to 1.0.

    If set to 1.0, the processed action is applied directly without any moving average window.
    """
    """移动平均值的重量 (regex表达式的浮动或定值)。
    默认到1.0。

    如果设置为1.0，处理的操作将直接进行，没有任何移动平均窗口。
    """


##
# Gripper actions.
##


@configclass
class BinaryJointActionCfg(ActionTermCfg):
    """Configuration for the base binary joint action term.

    See :class:`BinaryJointAction` for more details.
    """
    """基本二元联合动作项的配置

    See :分类:`BinaryJointAction` 详细信息。
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    """列出将该动作映射到的联合名称或regex表达式。"""
    open_command_expr: dict[str, float] = MISSING
    """The joint command to move to *open* configuration."""
    """移动到#开#配置的联合命令。"""
    close_command_expr: dict[str, float] = MISSING
    """The joint command to move to *close* configuration."""
    """移动到"接近"配置。"""


@configclass
class BinaryJointPositionActionCfg(BinaryJointActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """
    """对二进制关节位置动作项的配置

    See :分类:`BinaryJointPositionAction` 详细信息。
    """

    class_type: type[ActionTerm] = binary_joint_actions.BinaryJointPositionAction


@configclass
class BinaryJointVelocityActionCfg(BinaryJointActionCfg):
    """Configuration for the binary joint velocity action term.

    See :class:`BinaryJointVelocityAction` for more details.
    """
    """对二进制关节速度操作项的配置。

    See :分类:`BinaryJointVelocityAction` 详细信息。
    """

    class_type: type[ActionTerm] = binary_joint_actions.BinaryJointVelocityAction


@configclass
class AbsBinaryJointPositionActionCfg(ActionTermCfg):
    """Configuration for the absolute binary joint position action term.

    This action term is used for robust grasping by converting continuous gripper joint position actions
    into binary open/close commands. Unlike directly applying continuous gripper joint position actions, this class
    applies a threshold-based decision mechanism to determine whether to
    open or close the gripper.

    The action works by:
    1. Taking a continuous input action value
    2. Comparing it against a configurable threshold
    3. Mapping the result to either open or close commands based on the threshold comparison
    4. Applying the corresponding gripper open/close commands

    This approach provides more predictable and stable grasping behavior compared to directly applying
    continuous gripper joint position actions.

    See :class:`AbsBinaryJointPositionAction` for more details.
    """
    """对于绝对二元关节位置动作项的配置。

    这种动作项用于通过将连续抓住器关节位置动作转换为双式开/关闭命令来强大的抓住。
    与直接应用连续抓紧机关位置操作不同，该类应用基于门的决策机制来确定是否打开或关闭抓紧机。

    动作由:
    1. 采用连续输入动作值
    2. 与可配置的门进行比较
    3. 根据门比较，将结果映射成开放或关闭命令
    4. 使用相应的抓住器打开/关闭命令

    这种方法比直接应用连续抓住器关节位置操作更具可预测性和稳定的抓住行为。

    See :分类:`AbsBinaryJointPositionAction` 详细信息。
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    """列出将该动作映射到的联合名称或regex表达式。"""
    open_command_expr: dict[str, float] = MISSING
    """The joint command to move to *open* configuration."""
    """移动到#开#配置的联合命令。"""
    close_command_expr: dict[str, float] = MISSING
    """The joint command to move to *close* configuration."""
    """移动到"接近"配置。"""
    threshold: float = 0.5
    """The threshold for the binary action. Defaults to 0.5."""
    """双向动作的门。
    默认为0.5。
    """
    positive_threshold: bool = True
    """Whether to use positive (Open actions > Close actions) threshold. Defaults to True."""
    """是否使用正值 (开放动作>关闭动作) 门。
    默认为 True。
    """

    class_type: type[ActionTerm] = binary_joint_actions.AbsBinaryJointPositionAction


##
# Non-holonomic actions.
##


@configclass
class NonHolonomicActionCfg(ActionTermCfg):
    """Configuration for the non-holonomic action term with dummy joints at the base.

    See :class:`NonHolonomicAction` for more details.
    """
    """设置非全态动作项，底部设置模具关节。

    See :分类:`NonHolonomicAction` 详细信息。
    """

    class_type: type[ActionTerm] = non_holonomic_actions.NonHolonomicAction

    body_name: str = MISSING
    """Name of the body which has the dummy mechanism connected to."""
    """机器的名称。"""
    x_joint_name: str = MISSING
    """The dummy joint name in the x direction."""
    """在 x 方向的仿真联合名称。"""
    y_joint_name: str = MISSING
    """The dummy joint name in the y direction."""
    """在"y"方向的模糊名称。"""
    yaw_joint_name: str = MISSING
    """The dummy joint name in the yaw direction."""
    """在的方向上，的联合名字。"""
    scale: tuple[float, float] = (1.0, 1.0)
    """Scale factor for the action. Defaults to (1.0, 1.0)."""
    """动作的规模因素。
    默认的 (1.0， 1.0)。
    """
    offset: tuple[float, float] = (0.0, 0.0)
    """Offset factor for the action. Defaults to (0.0, 0.0)."""
    """这种动作的抵消因素。
    默认值为 (0.0，0.0)。
    """


##
# Task-space Actions.
##


@configclass
class DifferentialInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for inverse differential kinematics action term.

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """
    """对逆差差动力学操作项的配置。

    See :分类:`DifferentialInverseKinematicsAction` 详细信息。
    """

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
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""
        """四元数旋转 ``(w， x， y， z)`` w.r.t。
        它们的母体。
        在 (1.0，0.0，0.0，0.0，0.0的默认情况下。
        """

    class_type: type[ActionTerm] = task_space_actions.DifferentialInverseKinematicsAction

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
    """Scale factor for the action. Defaults to 1.0."""
    """动作的规模因素。
    默认到1.0。
    """
    controller: DifferentialIKControllerCfg = MISSING
    """The configuration for the differential IK controller."""
    """对差异性IK控制器的配置。"""


@configclass
class OperationalSpaceControllerActionCfg(ActionTermCfg):
    """Configuration for operational space controller action term.

    See :class:`OperationalSpaceControllerAction` for more details.
    """
    """操作空间控制器操作项的配置

    See :分类:`OperationalSpaceControllerAction` 详细信息。
    """

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
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""
        """四元数旋转 ``(w， x， y， z)`` w.r.t。
        它们的母体。
        在 (1.0，0.0，0.0，0.0，0.0的默认情况下。
        """

    class_type: type[ActionTerm] = task_space_actions.OperationalSpaceControllerAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    """列出将该动作映射到的联合名称或regex表达式。"""

    body_name: str = MISSING
    """Name of the body or frame for which motion/force control is performed."""
    """运动/力控制所执行的车身或框架名称。"""

    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""
    """目标框架 w.r.t的抵消。
    在身体框架。
    默认对None的缺陷，在这种情况下，不使用任何抵消。
    """

    task_frame_rel_path: str = None
    """The path of a ``RigidObject``, relative to the sub-environment, representing task frame. Defaults to None."""
    """对子环境的 ``RigidObject`` 路径，代表任务框架。
    默认为 None。
    """

    controller_cfg: OperationalSpaceControllerCfg = MISSING
    """The configuration for the operational space controller."""
    """操作空间控制器的配置。"""

    position_scale: float = 1.0
    """Scale factor for the position targets. Defaults to 1.0."""
    """位置目标的规模因素。
    默认到1.0。
    """

    orientation_scale: float = 1.0
    """Scale factor for the orientation (quad for ``pose_abs`` or axis-angle for ``pose_rel``). Defaults to 1.0."""
    """方向的尺度因子 (``pose_abs``的四角或``pose_rel``的轴角)。
    默认到1.0。
    """

    wrench_scale: float = 1.0
    """Scale factor for the wrench targets. Defaults to 1.0."""
    """关目标的规模因素。
    默认到1.0。
    """

    stiffness_scale: float = 1.0
    """Scale factor for the stiffness commands. Defaults to 1.0."""
    """硬度指令的尺度因素。
    默认到1.0。
    """

    damping_ratio_scale: float = 1.0
    """Scale factor for the damping ratio commands. Defaults to 1.0."""
    """放缓率指令的尺度因子。
    默认到1.0。
    """

    nullspace_joint_pos_target: str = "none"
    """The joint targets for the null-space control: ``"none"``, ``"zero"``, ``"default"``, ``"center"``.

    Note: Functional only when ``nullspace_control`` is set to ``"position"`` within the
        ``OperationalSpaceControllerCfg``.
    """
    """零空间控制的共同目标:``"none"``，``"zero"``，``"default"``，``"center"``。

    Note: 只有当``nullspace_control``在``OperationalSpaceControllerCfg``内设置为``"position"``时才会运行。
    """


##
# Surface Gripper actions.
##


@configclass
class SurfaceGripperBinaryActionCfg(ActionTermCfg):
    """Configuration for the binary surface gripper action term.

    See :class:`SurfaceGripperBinaryAction` for more details.
    """
    """对二进制表面抓住器操作项的配置。

    See :分类:`SurfaceGripperBinaryAction` 详细信息。
    """

    asset_name: str = MISSING
    """Name of the surface gripper asset in the scene."""
    """在场景的表面抓住器的名称。"""
    open_command: float = -1.0
    """The command value to open the gripper. Defaults to -1.0."""
    """打开抓住器的命令值。
    设置为 -1.0。
    """
    close_command: float = 1.0
    """The command value to close the gripper. Defaults to 1.0."""
    """关闭抓住器的命令值。
    默认到1.0。
    """

    class_type: type[ActionTerm] = surface_gripper_actions.SurfaceGripperBinaryAction
