# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch


@dataclass
class LocomanipulationSDGInputData:
    """Data container for in-place manipulation recording state.  Used during locomanipulation replay."""
    """场景操纵记录状态的数据容器。
    在位置操作重播中使用。
    """

    left_hand_pose_target: torch.Tensor
    """The pose of the left hand in world coordinates."""
    """在世界坐标中左手的姿势。"""

    right_hand_pose_target: torch.Tensor
    """The pose of the right hand in world coordinates."""
    """在世界坐标中右手的姿势。"""

    left_hand_joint_positions_target: torch.Tensor
    """The left hand joint positions."""
    """左手关节位置。"""

    right_hand_joint_positions_target: torch.Tensor
    """The right hand joint positions."""
    """右手关节位置。"""

    base_pose: torch.Tensor
    """The robot base pose in world coordinates."""
    """机器人基地在世界坐标中姿势。"""

    object_pose: torch.Tensor
    """The target object pose in world coordinates."""
    """目标对象在世界坐标中姿势。"""

    fixture_pose: torch.Tensor
    """The fixture (ie: table) pose in world coordinates."""
    """固定式 (即表) 呈现于世界坐标。"""


@dataclass
class LocomanipulationSDGOutputData:
    """A container for data that is recorded during locomanipulation replay.
    This is the final output of the pipeline.
    """
    """在位置操作重播过程中记录的数据容器。
    这是管道的最终输出。
    """

    left_hand_pose_target: torch.Tensor | None = None
    """The left hand's target pose."""
    """左手的目标姿势。"""

    right_hand_pose_target: torch.Tensor | None = None
    """The right hand's target pose."""
    """右手的目标姿势。"""

    left_hand_joint_positions_target: torch.Tensor | None = None
    """The left hand's target joint positions"""
    """左手的目标关节位置"""

    right_hand_joint_positions_target: torch.Tensor | None = None
    """The right hand's target joint positions"""
    """右手的目标关节位置"""

    base_velocity_target: torch.Tensor | None = None
    """The target velocity of the robot base.  This value is provided to the underlying base controller or policy."""
    """机器人基地的目标速度。
    该值应向底层基准控制器或策略提供。
    """

    start_fixture_pose: torch.Tensor | None = None
    """The pose of the start fixture (ie: pick-up table)."""
    """起始装置的姿势 (即接机)。"""

    end_fixture_pose: torch.Tensor | None = None
    """The pose of the end / destination fixture (ie: drop-off table)"""
    """终端/目的地装置的姿势 (即:降落表)"""

    object_pose: torch.Tensor | None = None
    """The pose of the target object."""
    """目标对象的姿势。"""

    base_pose: torch.Tensor | None = None
    """The pose of the robot base."""
    """机器人基地的姿势。"""

    data_generation_state: int | None = None
    """The state of the the locomanipulation SDG replay script's state machine."""
    """位置操作SDG重播脚本的状态机。"""

    base_goal_pose: torch.Tensor | None = None
    """The goal pose of the robot base (ie: the final destination before dropping off the object)"""
    """机器人基地的目标姿势 (即:在放下物体之前的最终目的地)"""

    base_goal_approach_pose: torch.Tensor | None = None
    """The goal pose provided to the path planner (this may be offset from the final destination to enable approach.)"""
    """为路径规划者提供的目标姿势 (这可以与最终目的地相对应，以便实现接近)。"""

    base_path: torch.Tensor | None = None
    """The robot base path as determined by the path planner."""
    """根据路径规划器的决定，机器人基路。"""

    recording_step: int | None = None
    """The current recording step used for upper body replay."""
    """现在的记录步骤用于身体上部重播。"""

    obstacle_fixture_poses: torch.Tensor | None = None
    """The pose of all obstacle fixtures in the scene."""
    """在场景中所有障碍装置的姿势。"""
