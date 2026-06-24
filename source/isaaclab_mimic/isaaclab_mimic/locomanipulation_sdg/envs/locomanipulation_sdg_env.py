# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils.datasets import EpisodeData

from isaaclab_mimic.locomanipulation_sdg.data_classes import LocomanipulationSDGInputData, LocomanipulationSDGOutputData
from isaaclab_mimic.locomanipulation_sdg.scene_utils import HasPose, SceneFixture


class LocomanipulationSDGOutputDataRecorder(RecorderTerm):
    """Recorder for Locomanipulation SDG output data."""
    """记录器用于Locomanipulation SDG输出数据。"""

    def record_pre_step(self):
        output_data: LocomanipulationSDGOutputData = self._env._locomanipulation_sdg_output_data

        output_data_dict = {
            "left_hand_pose_target": output_data.left_hand_pose_target[None, :],
            "right_hand_pose_target": output_data.right_hand_pose_target[None, :],
            "left_hand_joint_positions_target": output_data.left_hand_joint_positions_target[None, :],
            "right_hand_joint_positions_target": output_data.right_hand_joint_positions_target[None, :],
            "base_velocity_target": output_data.base_velocity_target[None, :],
            "start_fixture_pose": output_data.start_fixture_pose,
            "end_fixture_pose": output_data.end_fixture_pose,
            "object_pose": output_data.object_pose,
            "base_pose": output_data.base_pose,
            "task": torch.tensor([[output_data.data_generation_state]]),
            "base_goal_pose": output_data.base_goal_pose,
            "base_goal_approach_pose": output_data.base_goal_approach_pose,
            "base_path": output_data.base_path[None, :],
            "recording_step": torch.tensor([[output_data.recording_step]]),
            "obstacle_fixture_poses": output_data.obstacle_fixture_poses,
        }

        return "locomanipulation_sdg_output_data", output_data_dict


class LocomanipulationSDGEnv(ManagerBasedRLEnv):
    """An abstract base class that wraps the underlying environment, exposing methods needed for integration with
    locomanipulation replay.

    This class defines the core methods needed to integrate an environment with the locomanipulation SDG pipeline for
    locomanipulation replay.  By implementing these methods for a new environment, the environment can be used with
    the locomanipulation SDG replay function.
    """
    """一个抽象的基础类，包裹了底层环境，揭示了集成与位置操纵重播所需的方法。

    该类定义了将环境与位置操作SDG管道集成用于位置操作重播所需的核心方法。
    通过在新环境中实现这些方法，环境可以使用SDG重播功能。
    """

    def load_input_data(self, episode_data: EpisodeData, step: int) -> LocomanipulationSDGInputData:
        raise NotImplementedError

    def build_action_vector(
        self,
        left_hand_pose_target: torch.Tensor,
        right_hand_pose_target: torch.Tensor,
        left_hand_joint_positions_target: torch.Tensor,
        right_hand_joint_positions_target: torch.Tensor,
        base_velocity_target: torch.Tensor,
    ):
        raise NotImplementedError

    def get_base(self) -> HasPose:
        """Get the robot base body."""
        """拿起机器人基体。"""
        raise NotImplementedError

    def get_left_hand(self) -> HasPose:
        """Get the robot left hand body."""
        """拿着机器人左手身体。"""
        raise NotImplementedError

    def get_right_hand(self) -> HasPose:
        """Get the robot right hand body."""
        """拿着机器人右手身体。"""
        raise NotImplementedError

    def get_object(self) -> HasPose:
        """Get the target object body."""
        """拿到目标物体。"""
        raise NotImplementedError

    def get_start_fixture(self) -> SceneFixture:
        """Get the start fixture body."""
        """拿起启动装置的机体。"""
        raise NotImplementedError

    def get_end_fixture(self) -> SceneFixture:
        """Get the end fixture body."""
        """拿到终端装置的身体。"""
        raise NotImplementedError

    def get_obstacle_fixtures(self) -> list[SceneFixture]:
        """Get the set of obstacle fixtures."""
        """拿出障碍灯。"""
        raise NotImplementedError
