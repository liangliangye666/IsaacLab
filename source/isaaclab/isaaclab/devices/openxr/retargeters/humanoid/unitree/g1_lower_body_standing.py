# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class G1LowerBodyStandingRetargeter(RetargeterBase):
    """Provides lower body standing commands for the G1 robot."""
    """为G1机器人提供底部站立指令。"""

    def __init__(self, cfg: G1LowerBodyStandingRetargeterCfg):
        """Initialize the retargeter."""
        """启动重定位器。"""
        super().__init__(cfg)
        self.cfg = cfg

    def retarget(self, data: dict) -> torch.Tensor:
        return torch.tensor([0.0, 0.0, 0.0, self.cfg.hip_height], device=self.cfg.sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        # This retargeter does not consume any device data
        return []


@dataclass
class G1LowerBodyStandingRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 lower body standing retargeter."""
    """设置G1底部站立回器。"""

    hip_height: float = 0.72
    """Height of the G1 robot hip in meters. The value is a fixed height suitable for G1 to do tabletop manipulation."""
    """机器人G1的部的高度在米。
    值为G1进行桌面操作的固定高度。
    """
    retargeter_type: type[RetargeterBase] = G1LowerBodyStandingRetargeter
