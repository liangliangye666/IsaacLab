# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.surface_gripper import SurfaceGripper
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg

# import logger
logger = logging.getLogger(__name__)


class SurfaceGripperBinaryAction(ActionTerm):
    """Surface gripper binary action.

    This action term maps a binary action to the *open* or *close* surface gripper configurations.
    The surface gripper behavior is as follows:
    - [-1, -0.3] --> Gripper is Opening
    - [-0.3, 0.3] --> Gripper is Idle (do nothing)
    - [0.3, 1] --> Gripper is Closing

    Based on above, we follow the following convention for the binary action:

    1. Open action: 1 (bool) or positive values (float).
    2. Close action: 0 (bool) or negative values (float).

    The action term is specifically designed for surface grippers, which use a different
    interface than joint-based grippers.
    """
    """表面抓住器的二进制作用。

    这种动作项将二进制动作映射到 * 开放 * 或 * 关闭 * 表面抓住器配置。
    表面的行为如下:
    - [-1， -0.3] --> 抓住器正在开放
    - [-0.3，0.3] --> 抓住者是无力 (什么都不要做)
    - [0.3， 1] --> 抓住器正在关闭

    基于上述，我们遵循以下二进制作用的惯例:

    1. 开放动作:1 (bool) 或正值 (float)。
    2. 接近作用:0 (bool) 或负值 (float)。

    操作项专门用于使用不同于基于关节的接口的表面接口。
    """

    cfg: actions_cfg.SurfaceGripperBinaryActionCfg
    """The configuration of the action term."""
    """动作项的配置。"""
    _asset: SurfaceGripper
    """The surface gripper asset on which the action term is applied."""
    """操作项适用于的表面抓住器资产。"""

    def __init__(self, cfg: actions_cfg.SurfaceGripperBinaryActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # log the resolved asset name for debugging
        logger.info(
            f"Resolved surface gripper asset for the action term {self.__class__.__name__}: {self.cfg.asset_name}"
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 1, device=self.device)

        # parse open command
        self._open_command = torch.tensor(self.cfg.open_command, device=self.device)
        # parse close command
        self._close_command = torch.tensor(self.cfg.close_command, device=self.device)

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

    def apply_actions(self):
        """Apply the processed actions to the surface gripper."""
        """应将加工的操作应用于表面抓住器。"""
        self._asset.set_grippers_command(self._processed_actions.view(-1))
        self._asset.write_data_to_sim()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
