# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

# import logger
logger = logging.getLogger(__name__)


class BinaryJointAction(ActionTerm):
    """Base class for binary joint actions.

    This action term maps a binary action to the *open* or *close* joint configurations. These configurations are
    specified through the :class:`BinaryJointActionCfg` object. If the input action is a float vector, the action
    is considered binary based on the sign of the action values.

    Based on above, we follow the following convention for the binary action:

    1. Open action: 1 (bool) or positive values (float).
    2. Close action: 0 (bool) or negative values (float).

    The action term can mostly be used for gripper actions, where the gripper is either open or closed. This
    helps in devising a mimicking mechanism for the gripper, since in simulation it is often not possible to
    add such constraints to the gripper.
    """
    """双元联合动作的基类。

    这个动作项将二进制动作映射到 *开放*或 *关闭*联合配置。
    这些配置通过:class:`BinaryJointActionCfg`对象进行指定。
    如果输入动作是浮动向量，则根据动作值的标志，该动作被认为是二进制的。

    基于上述，我们遵循以下二进制作用的惯例:

    1. 开放动作:1 (bool) 或正值 (float)。
    2. 接近作用:0 (bool) 或负值 (float)。

    动作项主要可用于抓住器操作，在抓住器是开放或关闭的。
    这有助于设计仿真机器的机制，因为在仿真中，往往无法将这种限制添加到抓住机上。
    """

    cfg: actions_cfg.BinaryJointActionCfg
    """The configuration of the action term."""
    """动作项的配置。"""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    """动作项适用于的关节资产。"""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    """在输入操作中应用的裁剪。"""

    def __init__(self, cfg: actions_cfg.BinaryJointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        logger.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # parse open command
        self._open_command = torch.zeros(self._num_joints, device=self.device)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for the action term. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._open_command[index_list] = torch.tensor(value_list, device=self.device)

        # parse close command
        self._close_command = torch.zeros_like(self._open_command)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.close_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"Could not resolve all joints for the action term. Missing: {set(self._joint_names) - set(name_list)}"
            )
        self._close_command[index_list] = torch.tensor(value_list, device=self.device)

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """
    """属性。
    """

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "JointAction"
        self._IO_descriptor.joint_names = self._joint_names
        return self._IO_descriptor

    """
    Operations.
    """
    """操作。
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions < 0
        # compute the command
        self._processed_actions = torch.where(binary_mask, self._close_command, self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class BinaryJointPositionAction(BinaryJointAction):
    """Binary joint action that sets the binary action into joint position targets."""
    """双元联合动作，将双元动作设置为共同位置目标。"""

    cfg: actions_cfg.BinaryJointPositionActionCfg
    """The configuration of the action term."""
    """动作项的配置。"""

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)


class BinaryJointVelocityAction(BinaryJointAction):
    """Binary joint action that sets the binary action into joint velocity targets."""
    """双向联合动作，将双向动作设置为关节速度目标。"""

    cfg: actions_cfg.BinaryJointVelocityActionCfg
    """The configuration of the action term."""
    """动作项的配置。"""

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self._processed_actions, joint_ids=self._joint_ids)


class AbsBinaryJointPositionAction(BinaryJointAction):
    """Absolute Binary joint action that sets the binary action into joint position targets.

    This class extends BinaryJointAction to accept absolute position control
    for gripper joints. It converts continuous input actions into binary open/close commands
    using a configurable threshold mechanism.

    The key difference from the base BinaryJointAction is that this class:
    - Receives absolute joint position actions for gripper control
    - Implements a threshold-based decision system to determine open/close state

    The action processing works by:
    1. Taking a continuous input action value
    2. Comparing it against the configured threshold value
    3. Based on the threshold comparison and positive_threshold flag, determining
       whether to open or close the gripper
    4. Setting the target joint positions to either the open or close configuration

    """
    """绝对二元联合动作，将二元动作设置为关节位置目标。

    这类扩展到BinaryJointAction以接受绝对位置控制
    for gripper joints. It converts continuous input actions into binary open/close commands
    使用可配置的门机制。

    根据BinaryJointAction的关键区别是，
    - 接收绝对关节位置操作，用于控制抓住器
    - 实施基于门的决策系统，以确定开放/关闭状态

    动作处理由:
    1. 采用连续输入动作值
    2. 与配置的门值进行比较
    3. 基于门比较和positive_threshold标志，确定是否打开或关闭
    4. 设置目标关键位置为开放或关闭配置
    """

    cfg: actions_cfg.AbsBinaryJointPositionActionCfg
    """The configuration of the action term."""
    """动作项的配置。"""

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if self.cfg.positive_threshold:
            # true: open 0.785, false: close 0.0
            binary_mask = actions > self.cfg.threshold
        else:
            # true: close 0.0, false: open 0.785
            binary_mask = actions < self.cfg.threshold
        # compute the command
        self._processed_actions = torch.where(binary_mask, self._open_command, self._close_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
