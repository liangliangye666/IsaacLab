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
from isaaclab.managers.action_manager import ActionTerm

from isaaclab_contrib.assets import Multirotor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import thrust_actions_cfg

# import logger
logger = logging.getLogger(__name__)


class ThrustAction(ActionTerm):
    """Thrust action term that applies the processed actions as thrust commands.

    This action term is designed specifically for controlling multirotor vehicles by mapping
    action inputs to thruster commands. It provides flexible preprocessing of actions through:

    - **Scaling**: Multiply actions by a scale factor to adjust command magnitudes
    - **Offset**: Add an offset to center actions around a baseline (e.g., hover thrust)
    - **Clipping**: Constrain actions to valid ranges to prevent unsafe commands

    The action term integrates with Isaac Lab's :class:`~isaaclab.managers.ActionManager`
    framework and is specifically designed to work with :class:`~isaaclab_contrib.assets.Multirotor`
    assets.

    Key Features:
        - Supports per-thruster or uniform scaling and offsets
        - Optional automatic offset computation based on hover thrust
        - Action clipping for safety and constraint enforcement
        - Regex-based thruster selection for flexible control schemes

    Example:
        .. code-block:: python

            from isaaclab.envs import ManagerBasedRLEnvCfg
            from isaaclab_contrib.mdp.actions import ThrustActionCfg


            @configclass
            class MyEnvCfg(ManagerBasedRLEnvCfg):
                # ... other configuration ...

                @configclass
                class ActionsCfg:
                    # Direct thrust control (normalized actions)
                    thrust = ThrustActionCfg(
                        asset_name="robot",
                        scale=5.0,  # Convert [-1, 1] to [-5, 5] N
                        use_default_offset=True,  # Add hover thrust as offset
                        clip={".*": (-2.0, 8.0)},  # Clip to safe thrust range
                    )

    """
    """推动动作项，将处理的动作作为推动命令。

    这种动作项专门用于通过将动作输入映射到推进器命令来控制多轮动力车辆。
    它通过:

    - **扩展**:乘以规模因子来调整命令大小
    - **偏移**:在基线周围的中心动作中添加偏移 (e.g.，悬浮推力)
    - **裁剪**:限制操作到有效范围以防止不安全的命令

    动作项与艾萨克实验室的:class:`~isaaclab.managers.ActionManager`框架相结合，并专门为:class:`~isaaclab_contrib.assets.Mult
    irotor`资产设计。

    主要特征:
        - 支持每驱动器或均的扩展和抵消
        - 基于浮动推力的可选自动抵消计算
        - 减少安全和强制执行的动作
        - 灵活控制方案的基于Regex的推进器选择

    示例：
        .. code-block:: python

            from isaaclab.envs import ManagerBasedRLEnvCfg
            from isaaclab_contrib.mdp.actions import ThrustActionCfg


            @configclass
            class MyEnvCfg(ManagerBasedRLEnvCfg):
                # ... other configuration ...

                @configclass
                class ActionsCfg:
                    # Direct thrust control (normalized actions)
                    thrust = ThrustActionCfg(
                        asset_name="robot",
                        scale=5.0,  # Convert [-1, 1] to [-5, 5] N
                        use_default_offset=True,  # Add hover thrust as offset
                        clip={".*": (-2.0, 8.0)},  # Clip to safe thrust range
                    )
    """

    cfg: thrust_actions_cfg.ThrustActionCfg
    """The configuration of the action term."""
    """动作项的配置。"""
    _asset: Multirotor
    """The articulation asset on which the action term is applied."""
    """动作项适用于的关节资产。"""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    """对输入操作所应用的扩展因素。"""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    """对输入操作所应用的抵消。"""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    """在输入操作中应用的裁剪。"""

    def __init__(self, cfg: thrust_actions_cfg.ThrustActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        thruster_names_expr = self._asset.actuators["thrusters"].cfg.thruster_names_expr

        # resolve the thrusters over which the action term is applied
        self._thruster_ids, self._thruster_names = self._asset.find_bodies(
            thruster_names_expr, preserve_order=self.cfg.preserve_order
        )
        self._num_thrusters = len(self._thruster_ids)
        # log the resolved thruster names for debugging
        logger.info(
            f"Resolved thruster names for the action term {self.__class__.__name__}:"
            f" {self._thruster_names} [{self._thruster_ids}]"
        )

        # Avoid indexing across all thrusters for efficiency
        if self._num_thrusters == self._asset.num_thrusters and not self.cfg.preserve_order:
            self._thruster_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._thruster_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")

        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.offset, self._thruster_names
            )
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

        # parse clip
        if cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._thruster_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

        # Handle use_default_offset
        if cfg.use_default_offset:
            # Use default thruster RPS as offset
            self._offset = self._asset.data.default_thruster_rps[:, self._thruster_ids].clone()

    """
    Properties
    """
    """产品
    """

    @property
    def action_dim(self) -> int:
        return self._num_thrusters

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term."""
        """动作项的IO描述符。"""
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "ThrustAction"
        self._IO_descriptor.thruster_names = self._thruster_names
        self._IO_descriptor.scale = self._scale
        if isinstance(self._offset, torch.Tensor):
            self._IO_descriptor.offset = self._offset[0].detach().cpu().numpy().tolist()
        else:
            self._IO_descriptor.offset = self._offset
        if self.cfg.clip is not None:
            if isinstance(self._clip, torch.Tensor):
                self._IO_descriptor.clip = self._clip[0].detach().cpu().numpy().tolist()
            else:
                self._IO_descriptor.clip = self._clip
        else:
            self._IO_descriptor.clip = None
        return self._IO_descriptor

    """
    Methods
    """
    """方法
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term.

        This method resets the raw actions to zero for the specified environments.
        The processed actions will be recomputed during the next :meth:`process_actions` call.

        Args:
            env_ids: Environment indices to reset. Defaults to None (all environments).
        """
        """重置动作项。

        这种方法将原始动作重置为指定环境的零。
        在下一次:meth:`process_actions`电话中将重新计算处理的操作。

        参数：
            env_ids: 环境索引要重置。
                     在 None (所有环境) 中默认设置。
        """
        self._raw_actions[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        r"""Process actions by applying scaling, offset, and clipping.

        This method transforms raw policy actions into thrust commands through
        an affine transformation followed by optional clipping. The transformation is:

        .. math::
            \text{processed} = \text{raw} \times \text{scale} + \text{offset}

        If clipping is configured, the processed actions are then clamped:

        .. math::
            \text{processed} = \text{clamp}(\text{processed}, \text{min}, \text{max})

        Args:
            actions: Raw action tensor from the policy. Shape is ``(num_envs, action_dim)``.
                Typically in the range [-1, 1] for normalized policies.

        Note:
            The processed actions are stored internally and applied during the next
            :meth:`apply_actions` call.
        """
        """通过应用扩展，偏移和裁剪来处理操作。

        这种方法将原始的策略动作转化为推力命令，
        转变是:

        .. math::
            \text{processed} = \text{raw} \times \text{scale} + \text{offset}

        如果设置裁剪，则将处理的操作紧缩:

        .. math::
            \text{processed} = \text{clamp}(\text{processed}, \text{min}, \text{max})

        参数：
            actions: 策略的原始动作张量。
                     形状是``(num_envs， action_dim)``。
                     通常在正常化策略的范围 [-1， 1]。

        说明：
            处理的操作将内部存储并在下一次:meth:`apply_actions`调用中应用。
        """
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        """Apply the processed actions as thrust commands.

        This method sets the processed actions as thrust targets on the multirotor
        asset. The thrust targets are then used by the thruster actuator models
        to compute actual thrust forces during the simulation step.

        The method calls :meth:`~isaaclab_contrib.assets.Multirotor.set_thrust_target`
        on the multirotor asset with the appropriate thruster IDs.
        """
        """执行处理的操作作为推命令。

        这种方法将处理的动作设定为对多机动资产的推进目标。
        然后推进动力模型使用推进目标来计算在仿真阶段的实际推进力。

        该方法在适当的推进器 IDs 上调用 multirotor 资产 :meth:`~isaaclab_contrib.assets.Multirotor.set_thrust_target`。
        """
        # Set thrust targets using thruster IDs
        self._asset.set_thrust_target(self.processed_actions, thruster_ids=self._thruster_ids)
