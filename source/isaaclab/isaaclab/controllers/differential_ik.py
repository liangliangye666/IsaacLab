# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import apply_delta_pose, compute_pose_error

if TYPE_CHECKING:
    from .differential_ik_cfg import DifferentialIKControllerCfg


class DifferentialIKController:
    r"""Differential inverse kinematics (IK) controller.

    This controller is based on the concept of differential inverse kinematics [1, 2] which is a method for computing
    the change in joint positions that yields the desired change in pose.

    .. math::

        \Delta \mathbf{q} &= \mathbf{J}^{\dagger} \Delta \mathbf{x} \\
        \mathbf{q}_{\text{desired}} &= \mathbf{q}_{\text{current}} + \Delta \mathbf{q}

    where :math:`\mathbf{J}^{\dagger}` is the pseudo-inverse of the Jacobian matrix :math:`\mathbf{J}`,
    :math:`\Delta \mathbf{x}` is the desired change in pose, and :math:`\mathbf{q}_{\text{current}}`
    is the current joint positions.

    To deal with singularity in Jacobian, the following methods are supported for computing inverse of the Jacobian:

    - "pinv": Moore-Penrose pseudo-inverse
    - "svd": Adaptive singular-value decomposition (SVD)
    - "trans": Transpose of matrix
    - "dls": Damped version of Moore-Penrose pseudo-inverse (also called Levenberg-Marquardt)


    .. caution::
        The controller does not assume anything about the frames of the current and desired end-effector pose,
        or the joint-space velocities. It is up to the user to ensure that these quantities are given
        in the correct format.

    Reference:

    1. `Robot Dynamics Lecture Notes <https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf>`_
       by Marco Hutter (ETH Zurich)
    2. `Introduction to Inverse Kinematics <https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf>`_
       by Samuel R. Buss (University of California, San Diego)

    """
    """变化逆动力 (IK) 控制器。

    这种控制器基于差异逆动力学[1， 2]的概念，这是计算关联位置的变化方法，从而产生所需的姿势变化。

    .. math::

        \Delta \mathbf{q} &= \mathbf{J}^{\dagger} \Delta \mathbf{x} \\
        \mathbf{q}_{\text{desired}} &= \mathbf{q}_{\text{current}} + \Delta \mathbf{q}

    where :数学:`\mathbf{J}^{\dagger}`是雅哥比亚矩阵的伪逆:`\mathbf{J}`，
    :math:`\Delta \mathbf{x}`是所需的姿势变化，`\mathbf{q}_{\text{current}}`
    是目前的联合立场。

    为了处理Jacobian中的单一性，用于Jacobian的计算逆，支持以下方法:

    - "pinv":摩尔-罗斯伪逆
    - "svd":适应单值分解 (SVD)
    - "trans":矩阵的转移
    - "dls":摩尔-宾罗斯伪逆 (也称为莱文伯格-马卡尔特) 的化版本


    .. 谨慎::
        控制器不假设任何关于电流和所需的末端执行器姿势的框架，或联合空间速度的东西。
        用户必须确保这些量以正确的格式提供。

    Reference:

    1. `Robot Dynamics Lecture Notes <https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-
       intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf>`马可·哈特 (ETH苏黎世
    2. `Introduction to Inverse Kinematics
       <https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf>`通过Samuel R。 Buss
       (加利福尼亚大学，圣地亚哥)
    """

    def __init__(self, cfg: DifferentialIKControllerCfg, num_envs: int, device: str):
        """Initialize the controller.

        Args:
            cfg: The configuration for the controller.
            num_envs: The number of environments.
            device: The device to use for computations.
        """
        """启动控制器。

        参数：
            cfg: 控制器的配置。
            num_envs: 环境的数量。
            device: 用于计算的设备。
        """
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device
        # create buffers
        self.ee_pos_des = torch.zeros(self.num_envs, 3, device=self._device)
        self.ee_quat_des = torch.zeros(self.num_envs, 4, device=self._device)
        # -- input command
        self._command = torch.zeros(self.num_envs, self.action_dim, device=self._device)

    """
    Properties.
    """
    """属性。
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
        """控制器输入命令的尺寸。"""
        if self.cfg.command_type == "position":
            return 3  # (x, y, z)
        elif self.cfg.command_type == "pose" and self.cfg.use_relative_mode:
            return 6  # (dx, dy, dz, droll, dpitch, dyaw)
        else:
            return 7  # (x, y, z, qw, qx, qy, qz)

    """
    Operations.
    """
    """操作。
    """

    def reset(self, env_ids: torch.Tensor = None):
        """Reset the internals.

        Args:
            env_ids: The environment indices to reset. If None, then all environments are reset.
        """
        """重置内部。

        参数：
            env_ids: 环境索引要重置。
                     如果是None，则所有环境都会重置。
        """
        pass

    def set_command(
        self, command: torch.Tensor, ee_pos: torch.Tensor | None = None, ee_quat: torch.Tensor | None = None
    ):
        """Set target end-effector pose command.

        Based on the configured command type and relative mode, the method computes the desired end-effector pose.
        It is up to the user to ensure that the command is given in the correct frame. The method only
        applies the relative mode if the command type is ``position_rel`` or ``pose_rel``.

        Args:
            command: The input command in shape (N, 3) or (N, 6) or (N, 7).
            ee_pos: The current end-effector position in shape (N, 3).
                This is only needed if the command type is ``position_rel`` or ``pose_rel``.
            ee_quat: The current end-effector orientation (w, x, y, z) in shape (N, 4).
                This is only needed if the command type is ``position_*`` or ``pose_rel``.

        Raises:
            ValueError: If the command type is ``position_*`` and :attr:`ee_quat` is None.
            ValueError: If the command type is ``position_rel`` and :attr:`ee_pos` is None.
            ValueError: If the command type is ``pose_rel`` and either :attr:`ee_pos` or :attr:`ee_quat` is None.
        """
        """设置目标末端执行器姿势命令。

        基于配置命令类型和相对模式，该方法计算了所需的末端执行器姿势。
        用户必须确保命令是在正确的框架中进行。
        如果命令类型是``position_rel``或``pose_rel``，则该方法只适用于相对模式。

        参数：
            command: 输入命令的形状 (N， 3) 或 (N， 6) 或 (N， 7)。
            ee_pos: 目前末端执行器的形状位置 (N， 3)。
                    如果命令类型是``position_rel``或``pose_rel``，只需要这样做。
            ee_quat: 目前的末端执行器方向 (w， x， y， z) 形状 (N， 4)。
                     如果命令类型是``position_*``或``pose_rel``，只需要这样做。

        异常：
            ValueError: 如果命令类型是``position_*``，:attr:`ee_quat`是None。
            ValueError: 如果命令类型是``position_rel``，:attr:`ee_pos`是None。
            ValueError: 如果命令类型是``pose_rel``，则:attr:`ee_pos`或:attr:`ee_quat`是None。
        """
        # store command
        self._command[:] = command
        # compute the desired end-effector pose
        if self.cfg.command_type == "position":
            # we need end-effector orientation even though we are in position mode
            # this is only needed for display purposes
            if ee_quat is None:
                raise ValueError("End-effector orientation can not be None for `position_*` command type!")
            # compute targets
            if self.cfg.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("End-effector position can not be None for `position_rel` command type!")
                self.ee_pos_des[:] = ee_pos + self._command
                self.ee_quat_des[:] = ee_quat
            else:
                self.ee_pos_des[:] = self._command
                self.ee_quat_des[:] = ee_quat
        else:
            # compute targets
            if self.cfg.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    raise ValueError(
                        "Neither end-effector position nor orientation can be None for `pose_rel` command type!"
                    )
                self.ee_pos_des, self.ee_quat_des = apply_delta_pose(ee_pos, ee_quat, self._command)
            else:
                self.ee_pos_des = self._command[:, 0:3]
                self.ee_quat_des = self._command[:, 3:7]

    def compute(
        self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Computes the target joint positions that will yield the desired end effector pose.

        Args:
            ee_pos: The current end-effector position in shape (N, 3).
            ee_quat: The current end-effector orientation in shape (N, 4).
            jacobian: The geometric jacobian matrix in shape (N, 6, num_joints).
            joint_pos: The current joint positions in shape (N, num_joints).

        Returns:
            The target joint positions commands in shape (N, num_joints).
        """
        """计算目标关节位置，将产生所需的最终效果剂姿势。

        参数：
            ee_pos: 目前末端执行器的形状位置 (N， 3)。
            ee_quat: 目前的终端效应的形状方向 (N， 4)。
            jacobian: 形状的几何雅可比矩阵 (N，6，num_joints)。
            joint_pos: 现在的形状结合位置 (N，num_joints)。

        返回：
            目标关键位置以形状 (N，num_joints) 命令。
        """
        # compute the delta in joint-space
        if "position" in self.cfg.command_type:
            position_error = self.ee_pos_des - ee_pos
            jacobian_pos = jacobian[:, 0:3]
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=position_error, jacobian=jacobian_pos)
        else:
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
            )
            pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian)
        # return the desired joint positions
        return joint_pos + delta_joint_pos

    """
    Helper functions.
    """
    """辅助函数。
    """

    def _compute_delta_joint_pos(self, delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Computes the change in joint position that yields the desired change in pose.

        The method uses the Jacobian mapping from joint-space velocities to end-effector velocities
        to compute the delta-change in the joint-space that moves the robot closer to a desired
        end-effector position.

        Args:
            delta_pose: The desired delta pose in shape (N, 3) or (N, 6).
            jacobian: The geometric jacobian matrix in shape (N, 3, num_joints) or (N, 6, num_joints).

        Returns:
            The desired delta in joint space. Shape is (N, num-jointsß).
        """
        """计算关节位置的变化，从而产生所需的姿势变化。

        这种方法使用Jacobian地图绘制，从关联空间速度到末端执行器速度，以计算在关联空间中的多角变化，使机器人接近所需的末端执行器位置。

        参数：
            delta_pose: 想要的三角形姿势 (N， 3) 或 (N， 6)
            jacobian: 形状的几何雅可比矩阵 (N， 3， num_joints) 或 (N， 6， num_joints)。

        返回：
            在关节空间中。
            形状是 (N， num-jointsß)。
        """
        if self.cfg.ik_params is None:
            raise RuntimeError(f"Inverse-kinematics parameters for method '{self.cfg.ik_method}' is not defined!")
        # compute the delta in joint-space
        if self.cfg.ik_method == "pinv":  # Jacobian pseudo-inverse
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            # computation
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "svd":  # adaptive SVD
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            min_singular_value = self.cfg.ik_params["min_singular_value"]
            # computation
            # U: 6xd, S: dxd, V: d x num-joint
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1.0 / S
            S_inv = torch.where(min_singular_value < S, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = (
                torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6]
                @ torch.diag_embed(S_inv)
                @ torch.transpose(U, dim0=1, dim1=2)
            )
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "trans":  # Jacobian transpose
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_joint_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "dls":  # damped least squares
            # parameters
            lambda_val = self.cfg.ik_params["lambda_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
            delta_joint_pos = (
                jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
            )
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        else:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.cfg.ik_method}")

        return delta_joint_pos
