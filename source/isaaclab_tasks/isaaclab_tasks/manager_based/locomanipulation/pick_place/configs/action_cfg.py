# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..mdp.actions import AgileBasedLowerBodyAction


@configclass
class AgileBasedLowerBodyActionCfg(ActionTermCfg):
    """Configuration for the lower body action term that is based on Agile lower body RL policy."""
    """基于敏捷下部 RL策略的下部动作项配置。"""

    class_type: type[ActionTerm] = AgileBasedLowerBodyAction
    """The class type for the lower body action term."""
    """身体下部作用项的类型。"""

    joint_names: list[str] = MISSING
    """The names of the joints to control."""
    """控制关节的名称。"""

    obs_group_name: str = MISSING
    """The name of the observation group to use."""
    """使用的观测组名称。"""

    policy_path: str = MISSING
    """The path to the policy model."""
    """策略模式的道路。"""

    policy_output_offset: float = 0.0
    """Offsets the output of the policy."""
    """抵消策略的产出。"""

    policy_output_scale: float = 1.0
    """Scales the output of the policy."""
    """衡量策略的产出。"""
