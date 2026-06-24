# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

# import logger
logger = logging.getLogger(__name__)


class JointPositionToLimitsAction(ActionTerm):
    """Joint position action term that scales the input actions to the joint limits and applies them to the
    articulation's joints.

    This class is similar to the :class:`JointPositionAction` class. However, it performs additional
    re-scaling of input actions to the actuator joint position limits.

    While processing the actions, it performs the following operations:

    1. Apply scaling to the raw actions based on :attr:`actions_cfg.JointPositionToLimitsActionCfg.scale`.
    2. Clip the scaled actions to the range [-1, 1] and re-scale them to the joint limits if
       :attr:`actions_cfg.JointPositionToLimitsActionCfg.rescale_to_limits` is set to True.

    The processed actions are then sent as position commands to the articulation's joints.
    """
    """关节位置动作项，将输入动作扩展到关节极限，并应用于关节。

    这类类似于:class:`JointPositionAction`类。
    然而，它还将输入操作重新扩展到执行器关节位置限制。

    在处理动作时，它执行以下操作:

    1. 基于:attr:`actions_cfg.JointPositionToLimitsActionCfg.scale`的原始动作应进行扩展。
    2. 如果设置:attr:`actions_cfg.JointPositionToLimitsActionCfg.rescale_to_limits`为True，将扩展的操作裁剪到范围 [-1，
       1]并重新扩展到联合限度。

    然后将处理的操作作为位置命令发送到关节。
    """

    cfg: actions_cfg.JointPositionToLimitsActionCfg
    """The configuration of the action term."""
    """动作项的配置。"""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    """动作项适用于的关节资产。"""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    """对输入操作所应用的扩展因素。"""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    """在输入操作中应用的裁剪。"""

    def __init__(self, cfg: actions_cfg.JointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        logger.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.scale, self._joint_names, preserve_order=cfg.preserve_order
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names, preserve_order=cfg.preserve_order
                )
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
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the joint position to limits action.
        It adds the following information to the base descriptor:
        - joint_names: The names of the joints.
        - scale: The scale of the action term.
        - offset: The offset of the action term.
        - clip: The clip of the action term.

        Returns:
            The IO descriptor of the action term.
        """
        """动作项的IO描述符。

        该描述符用于描述联合限制作用位置的作用项。
        它将以下信息添加到基础描述符中:
        - joint_names关节的名称。
        - 规模:动作项的规模。
        - 抵消:动作项的抵消。
        - 动作项的裁剪。

        返回：
            动作项的IO描述符。
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "JointAction"
        self._IO_descriptor.joint_names = self._joint_names
        self._IO_descriptor.scale = self._scale
        # This seems to be always [4xNum_joints] IDK why. Need to check.
        if isinstance(self._offset, torch.Tensor):
            self._IO_descriptor.offset = self._offset[0].detach().cpu().numpy().tolist()
        else:
            self._IO_descriptor.offset = self._offset
        if self.cfg.clip is not None:
            self._IO_descriptor.clip = self._clip
        else:
            self._IO_descriptor.clip = None
        return self._IO_descriptor

    """
    Operations.
    """
    """操作。
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply affine transformations
        self._processed_actions = self._raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # rescale the position targets if configured
        # this is useful when the input actions are in the range [-1, 1]
        if self.cfg.rescale_to_limits:
            # clip to [-1, 1]
            actions = self._processed_actions.clamp(-1.0, 1.0)
            # rescale within the joint limits
            actions = math_utils.unscale_transform(
                actions,
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
            )
            self._processed_actions[:] = actions[:]

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class EMAJointPositionToLimitsAction(JointPositionToLimitsAction):
    r"""Joint action term that applies exponential moving average (EMA) over the processed actions as the
    articulation's joints position commands.

    Exponential moving average (EMA) is a type of moving average that gives more weight to the most recent data points.
    This action term applies the processed actions as moving average position action commands.
    The moving average is computed as:

    .. math::

        \text{applied action} =
            \alpha \times \text{processed actions} +
            (1 - \alpha) \times \text{previous applied action}

    where :math:`\alpha` is the weight for the moving average, :math:`\text{processed actions}` are the
    processed actions, and :math:`\text{previous action}` is the previous action that was applied to the articulation's
    joints.

    In the trivial case where the weight is 1.0, the action term behaves exactly like
    the :class:`JointPositionToLimitsAction` class.

    On reset, the previous action is initialized to the current joint positions of the articulation's joints.
    """
    """联合动作项，适用于指数移动平均值 (EMA) 作为关节位置指令处理的动作。

    指数移动平均值 (EMA) 是一种移动平均值，给最近数据点的重量增加。
    这种操作项将处理的操作作为移动平均位置操作命令。
    移动平均值为:

    .. math::

        \text{applied action} =
            \alpha \times \text{processed actions} +
            (1 - \alpha) \times \text{previous applied action}

    where :数学:`\alpha`是移动平均值的重量:`\text{processed actions}`是
    通过处理的操作，和:math:`\text{previous action}`是之前的操作，

    在小的情况下，在重量为1.0，操作项表现得完全像
    the :类:`JointPositionToLimitsAction`类。

    在重置时，先前的动作开始到关节关节的当前关节位置。
    """

    cfg: actions_cfg.EMAJointPositionToLimitsActionCfg
    """The configuration of the action term."""
    """动作项的配置。"""

    def __init__(self, cfg: actions_cfg.EMAJointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # parse and save the moving average weight
        if isinstance(cfg.alpha, float):
            # check that the weight is in the valid range
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(f"Moving average weight must be in the range [0, 1]. Got {cfg.alpha}.")
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((env.num_envs, self.action_dim), device=self.device)
            # resolve the dictionary config
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(
                cfg.alpha, self._joint_names
            )
            # check that the weights are in the valid range
            for name, value in zip(names_list, value_list):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Moving average weight must be in the range [0, 1]. Got {value} for joint {name}."
                    )
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported moving average weight type: {type(cfg.alpha)}. Supported types are float and dict."
            )

        # initialize the previous targets
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the EMA joint position to limits action.
        It adds the following information to the base descriptor:
        - joint_names: The names of the joints.
        - scale: The scale of the action term.
        - offset: The offset of the action term.
        - clip: The clip of the action term.
        - alpha: The moving average weight.

        Returns:
            The IO descriptor of the action term.
        """
        """动作项的IO描述符。

        这种描述符用于描述EMA关节位置限制作用的作用项。
        它将以下信息添加到基础描述符中:
        - joint_names关节的名称。
        - 规模:动作项的规模。
        - 抵消:动作项的抵消。
        - 动作项的裁剪。
        - 移动平均重量。

        返回：
            动作项的IO描述符。
        """
        super().IO_descriptor
        if isinstance(self._alpha, float):
            self._IO_descriptor.alpha = self._alpha
        elif isinstance(self._alpha, torch.Tensor):
            self._IO_descriptor.alpha = self._alpha[0].detach().cpu().numpy().tolist()
        else:
            raise ValueError(
                f"Unsupported moving average weight type: {type(self._alpha)}. Supported types are float and"
                " torch.Tensor."
            )
        return self._IO_descriptor

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # check if specific environment ids are provided
        if env_ids is None:
            super().reset(slice(None))
            self._prev_applied_actions[:] = self._asset.data.joint_pos[:, self._joint_ids]
        else:
            super().reset(env_ids)
            curr_applied_actions = self._asset.data.joint_pos[env_ids[:, None], self._joint_ids].view(len(env_ids), -1)
            self._prev_applied_actions[env_ids, :] = curr_applied_actions

    def process_actions(self, actions: torch.Tensor):
        # apply affine transformations
        super().process_actions(actions)
        # set position targets as moving average
        ema_actions = self._alpha * self._processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        # clamp the targets
        self._processed_actions[:] = torch.clamp(
            ema_actions,
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
        )
        # update previous targets
        self._prev_applied_actions[:] = self._processed_actions[:]
