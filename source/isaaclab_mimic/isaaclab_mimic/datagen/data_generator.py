# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Base class for data generator."""
"""数据生成器的基础类。"""

import asyncio
import copy
import logging
from typing import Any

import numpy as np
import torch

import isaaclab.utils.math as PoseUtils

logger = logging.getLogger(__name__)

from isaaclab.envs import (
    ManagerBasedRLMimicEnv,
    MimicEnvCfg,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from isaaclab.managers import TerminationTermCfg

from isaaclab_mimic.datagen.datagen_info import DatagenInfo
from isaaclab_mimic.datagen.selection_strategy import make_selection_strategy
from isaaclab_mimic.datagen.waypoint import MultiWaypoint, Waypoint, WaypointSequence, WaypointTrajectory

from .datagen_info_pool import DataGenInfoPool


def transform_source_data_segment_using_delta_object_pose(
    src_eef_poses: torch.Tensor,
    delta_obj_pose: torch.Tensor,
) -> torch.Tensor:
    """
    Transform a source data segment (object-centric subtask segment from source demonstration) using
    a delta object pose.

    Args:
        src_eef_poses: pose sequence (shape [T, 4, 4]) for the sequence of end effector control poses
            from the source demonstration
        delta_obj_pose: 4x4 delta object pose

    Returns:
        transformed_eef_poses: transformed pose sequence (shape [T, 4, 4])
    """
    """转换一个源数据段 (从源示范中进行对象中心子任务段) 使用多角对象姿势。

    参数：
        src_eef_poses: 对最终效应控制姿势的序列 (形状 [T， 4， 4])
            from the source demonstration
        delta_obj_pose: 4x4 delta对象姿势

    返回：
        transformed_eef_poses: 转型姿势序列 (形状 [T， 4， 4])
    """
    return PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses,
        pose_A_in_B=delta_obj_pose[None],
    )


def transform_source_data_segment_using_object_pose(
    obj_pose: torch.Tensor,
    src_eef_poses: torch.Tensor,
    src_obj_pose: torch.Tensor,
) -> torch.Tensor:
    """
    Transform a source data segment (object-centric subtask segment from source demonstration) such that
    the relative poses between the target eef pose frame and the object frame are preserved. Recall that
    each object-centric subtask segment corresponds to one object, and consists of a sequence of
    target eef poses.

    Args:
        obj_pose: 4x4 object pose in current scene
        src_eef_poses: pose sequence (shape [T, 4, 4]) for the sequence of end effector control poses
            from the source demonstration
        src_obj_pose: 4x4 object pose from the source demonstration

    Returns:
        transformed_eef_poses: transformed pose sequence (shape [T, 4, 4])
    """
    """转换一个源数据段 (从源示范中以对象为中心的子任务段)，使目标eef姿势框架和对象框架之间的相对姿势保持。
    请记住，每个对象中心的子任务段都对应一个对象，并且由目标eef姿势的序列组成。

    参数：
        obj_pose: 在当前场景中4×4物体姿势
        src_eef_poses: 对最终效应控制姿势的序列 (形状 [T， 4， 4])
            from the source demonstration
        src_obj_pose: 来源示范中的4×4物体姿势

    返回：
        transformed_eef_poses: 转型姿势序列 (形状 [T， 4， 4])
    """

    # Transform source end effector poses to be relative to source object frame
    src_eef_poses_rel_obj = PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses,
        pose_A_in_B=PoseUtils.pose_inv(src_obj_pose[None]),
    )

    # Apply relative poses to current object frame to obtain new target eef poses
    transformed_eef_poses = PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses_rel_obj,
        pose_A_in_B=obj_pose[None],
    )
    return transformed_eef_poses


def get_delta_pose_with_scheme(
    src_obj_pose: torch.Tensor,
    cur_obj_pose: torch.Tensor,
    task_constraint: dict,
) -> torch.Tensor:
    """
    Get the delta pose with the given coordination scheme.

    Args:
        src_obj_pose: 4x4 object pose in source scene
        cur_obj_pose: 4x4 object pose in current scene
        task_constraint: task constraint dictionary

    Returns:
        delta_pose: 4x4 delta pose
    """
    """通过给定的协调方案来获得三角形姿势。

    参数：
        src_obj_pose: 4x4物体在源场景中姿势
        cur_obj_pose: 在当前场景中4×4物体姿势
        task_constraint: 任务限制字典

    返回：
        delta_pose: 4x4 达尔塔姿势
    """
    coord_transform_scheme = task_constraint["coordination_scheme"]
    device = src_obj_pose.device
    if coord_transform_scheme == SubTaskConstraintCoordinationScheme.TRANSFORM:
        delta_pose = PoseUtils.get_delta_object_pose(cur_obj_pose, src_obj_pose)
        # add noise to delta pose position
    elif coord_transform_scheme == SubTaskConstraintCoordinationScheme.TRANSLATE:
        delta_pose = torch.eye(4, device=device)
        delta_pose[:3, 3] = cur_obj_pose[:3, 3] - src_obj_pose[:3, 3]
    elif coord_transform_scheme == SubTaskConstraintCoordinationScheme.REPLAY:
        delta_pose = torch.eye(4, device=device)
    else:
        raise ValueError(
            f"coordination coord_transform_scheme {coord_transform_scheme} not supported, only"
            f" {[e.value for e in SubTaskConstraintCoordinationScheme]} are supported"
        )

    pos_noise_scale = task_constraint["coordination_scheme_pos_noise_scale"]
    rot_noise_scale = task_constraint["coordination_scheme_rot_noise_scale"]
    if pos_noise_scale != 0.0 or rot_noise_scale != 0.0:
        pos = delta_pose[:3, 3]
        rot = delta_pose[:3, :3]
        pos_new, rot_new = PoseUtils.add_uniform_noise_to_pose(pos, rot, pos_noise_scale, rot_noise_scale)
        delta_pose = torch.eye(4, device=device)
        delta_pose[:3, 3] = pos_new
        delta_pose[:3, :3] = rot_new
    return delta_pose


class DataGenerator:
    """The main data generator class that generates new trajectories from source datasets.

    The data generator, inspired by the MimicGen, enables the generation of new datasets based on a
    few human collected source demonstrations.

    The data generator works by parsing demonstrations into object-centric subtask segments, stored in
    :class:`DataGenInfoPool`. It then adapts these subtask segments to new scenes by transforming each
    segment according to the new scene's context, stitching them into a coherent trajectory for a robotic
    end-effector to execute.
    """
    """从源数据集中生成新的轨迹的主要数据生成器类。

    数据生成器，MimicGen根据人类收集的几个源头示范，可以生成新的数据集。

    数据生成器通过分析示范，将其分为对象中心的子任务段，存储在:class:`DataGenInfoPool`中。
    然后通过根据新场景的背景改变每个段落来将这些子任务段调整到新的场景中，将它们编织成一个连贯的轨迹，
    """

    def __init__(
        self,
        env: ManagerBasedRLMimicEnv,
        src_demo_datagen_info_pool: DataGenInfoPool | None = None,
        dataset_path: str | None = None,
        demo_keys: list[str] | None = None,
    ):
        """
        Args:
            env: environment to use for data generation
            src_demo_datagen_info_pool: source demo datagen info pool
            dataset_path: path to hdf5 dataset to use for generation
            demo_keys: list of demonstration keys to use in file. If not provided,
                all demonstration keys will be used.
        """
        """参数：
            env: 用于数据生成的环境
            src_demo_datagen_info_pool: 来源示范数据集
            dataset_path: 用于生成的hdf5数据集的路径
            demo_keys: 在文件中使用的示范键列表。
                       如果没有提供，将使用所有示范键。
        """
        self.env = env
        self.env_cfg = env.cfg
        assert isinstance(self.env_cfg, MimicEnvCfg)
        self.dataset_path = dataset_path

        # Sanity check on task spec offset ranges - final subtask should not have any offset randomization
        for subtask_configs in self.env_cfg.subtask_configs.values():
            assert subtask_configs[-1].subtask_term_offset_range[0] == 0
            assert subtask_configs[-1].subtask_term_offset_range[1] == 0

        self.demo_keys = demo_keys

        if src_demo_datagen_info_pool is not None:
            self.src_demo_datagen_info_pool = src_demo_datagen_info_pool
        elif dataset_path is not None:
            self.src_demo_datagen_info_pool = DataGenInfoPool(
                env=self.env, env_cfg=self.env_cfg, device=self.env.device
            )
            self.src_demo_datagen_info_pool.load_from_dataset_file(dataset_path, select_demo_keys=self.demo_keys)
        else:
            raise ValueError("Either src_demo_datagen_info_pool or dataset_path must be provided")

    def __repr__(self):
        """Pretty print this object."""
        """这个物体很漂亮。"""
        msg = str(self.__class__.__name__)
        msg += f" (\n\tdataset_path={self.dataset_path}\n\tdemo_keys={self.demo_keys}\n)"
        return msg

    def randomize_subtask_boundaries(self) -> dict[str, np.ndarray]:
        """Apply random offsets to sample subtask boundaries according to the task spec.

        Recall that each demonstration is segmented into a set of subtask segments, and the
        end index (and start index when skillgen is enabled) of each subtask can have a random offset.
        """
        """根据任务规范，将随机抵消应用到样本子子任务边界。

        请记住，每个示范都被划分为一组子任务段，每个子任务的终端索引 (以及启动时的启动索引) 可以有随机的抵消。
        """

        randomized_subtask_boundaries = {}

        for eef_name, subtask_boundaries in self.src_demo_datagen_info_pool.subtask_boundaries.items():
            # Initial subtask start and end indices - shape (N, S, 2)
            subtask_boundaries = np.array(subtask_boundaries)

            # Randomize the start of the first subtask
            first_subtask_start_offsets = np.random.randint(
                low=self.env_cfg.subtask_configs[eef_name][0].first_subtask_start_offset_range[0],
                high=self.env_cfg.subtask_configs[eef_name][0].first_subtask_start_offset_range[0] + 1,
                size=subtask_boundaries.shape[0],
            )
            subtask_boundaries[:, 0, 0] += first_subtask_start_offsets

            # For each subtask, sample all end offsets at once for each demonstration
            # Add them to subtask end indices, and then set them as the start indices of next subtask too
            for i in range(subtask_boundaries.shape[1]):
                # If skillgen is enabled, sample a random start offset to increase demonstration variety.
                if self.env_cfg.datagen_config.use_skillgen:
                    start_offset = np.random.randint(
                        low=self.env_cfg.subtask_configs[eef_name][i].subtask_start_offset_range[0],
                        high=self.env_cfg.subtask_configs[eef_name][i].subtask_start_offset_range[1] + 1,
                        size=subtask_boundaries.shape[0],
                    )
                    subtask_boundaries[:, i, 0] += start_offset
                elif i > 0:
                    # Without skillgen, the start of a subtask is the end of the previous one.
                    subtask_boundaries[:, i, 0] = subtask_boundaries[:, i - 1, 1]

                # Sample end offset for each demonstration
                end_offsets = np.random.randint(
                    low=self.env_cfg.subtask_configs[eef_name][i].subtask_term_offset_range[0],
                    high=self.env_cfg.subtask_configs[eef_name][i].subtask_term_offset_range[1] + 1,
                    size=subtask_boundaries.shape[0],
                )
                subtask_boundaries[:, i, 1] = subtask_boundaries[:, i, 1] + end_offsets

            # Ensure non-empty subtasks
            assert np.all((subtask_boundaries[:, :, 1] - subtask_boundaries[:, :, 0]) > 0), "got empty subtasks!"

            # Ensure subtask indices increase (both starts and ends)
            assert np.all((subtask_boundaries[:, 1:, :] - subtask_boundaries[:, :-1, :]) > 0), (
                "subtask indices do not strictly increase"
            )

            # Ensure subtasks are in order
            subtask_inds_flat = subtask_boundaries.reshape(subtask_boundaries.shape[0], -1)
            assert np.all((subtask_inds_flat[:, 1:] - subtask_inds_flat[:, :-1]) >= 0), "subtask indices not in order"

            randomized_subtask_boundaries[eef_name] = subtask_boundaries

        return randomized_subtask_boundaries

    def select_source_demo(
        self,
        eef_name: str,
        eef_pose: np.ndarray,
        object_pose: np.ndarray,
        src_demo_current_subtask_boundaries: np.ndarray,
        subtask_object_name: str,
        selection_strategy_name: str,
        selection_strategy_kwargs: dict | None = None,
    ) -> int:
        """Helper method to run source subtask segment selection.

        Args:
            eef_name: name of end effector
            eef_pose: current end effector pose
            object_pose: current object pose for this subtask
            src_demo_current_subtask_boundaries: start and end indices for subtask segment
                in source demonstrations of shape (N, 2)
            subtask_object_name: name of reference object for this subtask
            selection_strategy_name: name of selection strategy
            selection_strategy_kwargs: extra kwargs for running selection strategy

        Returns:
            The selected source demo index
        """
        """运行源子任务段选择的辅助方法。

        参数：
            eef_name: 终端有效者的名称
            eef_pose: 终端效应的当前姿势
            object_pose: 为此子任务的当前对象姿势
            src_demo_current_subtask_boundaries: 在原始形状示范中的子任务段的起始和终索引 (N，2)
            subtask_object_name: 本子任务的参考对象名称
            selection_strategy_name: 选择战略名称
            selection_strategy_kwargs: 执行选择策略的额外kwargs

        返回：
            选择的源示范索引
        """
        if subtask_object_name is None:
            # no reference object - only random selection is supported
            assert selection_strategy_name == "random", selection_strategy_name

        # We need to collect the datagen info objects over the timesteps for the subtask segment in each source
        # demo, so that it can be used by the selection strategy.
        src_subtask_datagen_infos = []
        for i in range(len(self.src_demo_datagen_info_pool.datagen_infos)):
            # Datagen info over all timesteps of the src trajectory
            src_ep_datagen_info = self.src_demo_datagen_info_pool.datagen_infos[i]

            # Time indices for subtask
            subtask_start_ind = src_demo_current_subtask_boundaries[i][0]
            subtask_end_ind = src_demo_current_subtask_boundaries[i][1]

            # Get subtask segment using indices
            src_subtask_datagen_infos.append(
                DatagenInfo(
                    eef_pose=src_ep_datagen_info.eef_pose[eef_name][subtask_start_ind:subtask_end_ind],
                    # Only include object pose for relevant object in subtask
                    object_poses=(
                        {
                            subtask_object_name: src_ep_datagen_info.object_poses[subtask_object_name][
                                subtask_start_ind:subtask_end_ind
                            ]
                        }
                        if (subtask_object_name is not None)
                        else None
                    ),
                    # Subtask termination signal is unused
                    subtask_term_signals=None,
                    target_eef_pose=src_ep_datagen_info.target_eef_pose[eef_name][subtask_start_ind:subtask_end_ind],
                    gripper_action=src_ep_datagen_info.gripper_action[eef_name][subtask_start_ind:subtask_end_ind],
                )
            )

        # Make selection strategy object
        selection_strategy_obj = make_selection_strategy(selection_strategy_name)

        # Run selection
        if selection_strategy_kwargs is None:
            selection_strategy_kwargs = dict()
        selected_src_demo_ind = selection_strategy_obj.select_source_demo(
            eef_pose=eef_pose,
            object_pose=object_pose,
            src_subtask_datagen_infos=src_subtask_datagen_infos,
            **selection_strategy_kwargs,
        )

        return selected_src_demo_ind

    def generate_eef_subtask_trajectory(
        self,
        env_id: int,
        eef_name: str,
        subtask_ind: int,
        all_randomized_subtask_boundaries: dict,
        runtime_subtask_constraints_dict: dict,
        selected_src_demo_inds: dict,
    ) -> WaypointTrajectory:
        """Build a transformed waypoint trajectory for a single subtask of an end-effector.

        This method selects a source demonstration segment for the specified subtask,
        slices the corresponding EEF poses/targets/gripper actions using the randomized
        subtask boundaries, optionally prepends the first robot EEF pose (to interpolate
        from the robot pose instead of the first target), applies an object/coordination
        based transform to the pose sequence, and returns the result as a `WaypointTrajectory`.

        Selection and transforms:

        - Source demo selection is controlled by `SubTaskConfig.selection_strategy` (and kwargs) and by
          `datagen_config.generation_select_src_per_subtask` / `generation_select_src_per_arm`.
        - For coordination constraints, the method reuses/sets the selected source demo ID across
          concurrent subtasks, computes `synchronous_steps`, and stores the pose `transform` used
          to ensure consistent relative motion between tasks.
        - Pose transforms are computed either from object poses (`object_ref`) or via a delta pose
          provided by a concurrent task/coordination scheme.


        Args:
            env_id: Environment index used to query current robot/object poses.
            eef_name: End-effector key whose subtask trajectory is being generated.
            subtask_ind: Index of the subtask within `subtask_configs[eef_name]`.
            all_randomized_subtask_boundaries: For each EEF, an array of per-demo
                randomized (start, end) indices for every subtask.
            runtime_subtask_constraints_dict: In/out dictionary carrying runtime fields
                for constraints (e.g., selected source ID, delta transform, synchronous steps).
            selected_src_demo_inds: Per-EEF mapping for the currently selected source demo index
                (may be reused across arms if configured).

        Returns:
            WaypointTrajectory: The transformed trajectory for the selected subtask segment.
        """
        """构建一个转换的路线点轨迹，用于单个子任务的最终效应。

        这种方法选择了指定的子任务的源示范段，使用随机的子任务边界切割相应的EEF姿势/目标/抓住动作，可选地预定第一个机器人EEF姿势 (进行回合)
        from the robot pose instead of the first target), applies an object/coordination
        基于转换到姿势序列，并返回结果为`WaypointTrajectory`。

        选择和转换:

        - 来源演示选择由`SubTaskConfig.selection_strategy` (和kwargs)
          和`datagen_config.generation_select_src_per_subtask` / `generation_select_src_per_arm`控制。
        - 对于协调约束，该方法在同时的子任务中重复使用/设置选定的源演示 ID，计算 `synchronous_steps`，并存储用于确保任务之间的相对运动一致的姿势 `transform`。
        - 姿势转换是从对象姿势 (`object_ref`) 或通过由同时任务/协调方案提供的三角姿势计算的。


        参数：
            env_id: 环境索引用于查询当前机器人/物体姿势。
            eef_name: 终端效应键，其子任务轨迹正在生成。
            subtask_ind: 在 `subtask_configs[eef_name]` 中的子任务索引。
            all_randomized_subtask_boundaries: 对于每一个EEF，每一个子任务的每一个示范随机索引 (开始，结束)。
            runtime_subtask_constraints_dict: 运行时间字段的进/出字典
                for constraints (e.g., selected source ID, delta transform, synchronous steps).
            selected_src_demo_inds: 目前选择的源演示索引的每EEF映射 (如果配置，可在各臂上重复使用)。

        返回：
            WaypointTrajectory: 选择的子任务段的转换轨迹。
        """
        subtask_configs = self.env_cfg.subtask_configs[eef_name]
        # name of object for this subtask
        subtask_object_name = self.env_cfg.subtask_configs[eef_name][subtask_ind].object_ref
        subtask_object_pose = (
            self.env.get_object_poses(env_ids=[env_id])[subtask_object_name][0]
            if (subtask_object_name is not None)
            else None
        )

        is_first_subtask = subtask_ind == 0

        need_source_demo_selection = is_first_subtask or self.env_cfg.datagen_config.generation_select_src_per_subtask

        if not self.env_cfg.datagen_config.generation_select_src_per_arm:
            need_source_demo_selection = need_source_demo_selection and selected_src_demo_inds[eef_name] is None

        use_delta_transform = None
        coord_transform_scheme = None
        if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
            if runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["type"] == SubTaskConstraintType.COORDINATION:
                # Avoid selecting source demo if it has already been selected by the concurrent task
                concurrent_task_spec_key = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_task_spec_key"
                ]
                concurrent_subtask_ind = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_subtask_ind"
                ]
                concurrent_selected_src_ind = runtime_subtask_constraints_dict[
                    (concurrent_task_spec_key, concurrent_subtask_ind)
                ]["selected_src_demo_ind"]
                if concurrent_selected_src_ind is not None:
                    # The concurrent task has started, so we should use the same source demo
                    selected_src_demo_inds[eef_name] = concurrent_selected_src_ind
                    need_source_demo_selection = False
                    # This transform is set at after the first data generation iteration/first
                    # run of the main while loop
                    use_delta_transform = runtime_subtask_constraints_dict[
                        (concurrent_task_spec_key, concurrent_subtask_ind)
                    ]["transform"]
                else:
                    assert "transform" not in runtime_subtask_constraints_dict[(eef_name, subtask_ind)], (
                        "transform should not be set for concurrent task"
                    )
                    # Need to transform demo according to scheme
                    coord_transform_scheme = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                        "coordination_scheme"
                    ]
                    if coord_transform_scheme != SubTaskConstraintCoordinationScheme.REPLAY:
                        assert subtask_object_name is not None, (
                            f"object reference should not be None for {coord_transform_scheme} coordination scheme"
                        )

        if need_source_demo_selection:
            selected_src_demo_inds[eef_name] = self.select_source_demo(
                eef_name=eef_name,
                eef_pose=self.env.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0],
                object_pose=subtask_object_pose,
                src_demo_current_subtask_boundaries=all_randomized_subtask_boundaries[eef_name][:, subtask_ind],
                subtask_object_name=subtask_object_name,
                selection_strategy_name=self.env_cfg.subtask_configs[eef_name][subtask_ind].selection_strategy,
                selection_strategy_kwargs=self.env_cfg.subtask_configs[eef_name][subtask_ind].selection_strategy_kwargs,
            )

        assert selected_src_demo_inds[eef_name] is not None
        selected_src_demo_ind = selected_src_demo_inds[eef_name]

        if not self.env_cfg.datagen_config.generation_select_src_per_arm and need_source_demo_selection:
            for itrated_eef_name in self.env_cfg.subtask_configs.keys():
                selected_src_demo_inds[itrated_eef_name] = selected_src_demo_ind

        # Selected subtask segment time indices
        selected_src_subtask_boundary = all_randomized_subtask_boundaries[eef_name][selected_src_demo_ind, subtask_ind]

        if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
            if runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["type"] == SubTaskConstraintType.COORDINATION:
                # Store selected source demo ind for concurrent task
                runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["selected_src_demo_ind"] = (
                    selected_src_demo_ind
                )
                concurrent_task_spec_key = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_task_spec_key"
                ]
                concurrent_subtask_ind = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_subtask_ind"
                ]
                concurrent_src_subtask_inds = all_randomized_subtask_boundaries[concurrent_task_spec_key][
                    selected_src_demo_ind, concurrent_subtask_ind
                ]
                subtask_len = selected_src_subtask_boundary[1] - selected_src_subtask_boundary[0]
                concurrent_subtask_len = concurrent_src_subtask_inds[1] - concurrent_src_subtask_inds[0]
                runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["synchronous_steps"] = min(
                    subtask_len, concurrent_subtask_len
                )

        # Get subtask segment, consisting of the sequence of robot eef poses, target poses, gripper actions
        src_ep_datagen_info = self.src_demo_datagen_info_pool.datagen_infos[selected_src_demo_ind]
        src_subtask_eef_poses = src_ep_datagen_info.eef_pose[eef_name][
            selected_src_subtask_boundary[0] : selected_src_subtask_boundary[1]
        ]
        src_subtask_target_poses = src_ep_datagen_info.target_eef_pose[eef_name][
            selected_src_subtask_boundary[0] : selected_src_subtask_boundary[1]
        ]
        src_subtask_gripper_actions = src_ep_datagen_info.gripper_action[eef_name][
            selected_src_subtask_boundary[0] : selected_src_subtask_boundary[1]
        ]

        # Get reference object pose from source demo
        src_subtask_object_pose = (
            src_ep_datagen_info.object_poses[subtask_object_name][selected_src_subtask_boundary[0]]
            if (subtask_object_name is not None)
            else None
        )

        if is_first_subtask or self.env_cfg.datagen_config.generation_transform_first_robot_pose:
            # Source segment consists of first robot eef pose and the target poses. This ensures that
            # We will interpolate to the first robot eef pose in this source segment, instead of the
            # first robot target pose.
            src_eef_poses = torch.cat([src_subtask_eef_poses[0:1], src_subtask_target_poses], dim=0)
            # Account for extra timestep added to @src_eef_poses
            src_subtask_gripper_actions = torch.cat(
                [src_subtask_gripper_actions[0:1], src_subtask_gripper_actions], dim=0
            )
        else:
            # Source segment consists of just the target poses.
            src_eef_poses = src_subtask_target_poses.clone()
            src_subtask_gripper_actions = src_subtask_gripper_actions.clone()

        # Transform source demonstration segment using relevant object pose.
        if use_delta_transform is not None:
            # Use delta transform from concurrent task
            transformed_eef_poses = transform_source_data_segment_using_delta_object_pose(
                src_eef_poses, use_delta_transform
            )

        else:
            if coord_transform_scheme is not None:
                delta_obj_pose = get_delta_pose_with_scheme(
                    src_subtask_object_pose,
                    subtask_object_pose,
                    runtime_subtask_constraints_dict[(eef_name, subtask_ind)],
                )
                transformed_eef_poses = transform_source_data_segment_using_delta_object_pose(
                    src_eef_poses, delta_obj_pose
                )
                runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["transform"] = delta_obj_pose
            else:
                if subtask_object_name is not None:
                    transformed_eef_poses = transform_source_data_segment_using_object_pose(
                        subtask_object_pose,
                        src_eef_poses,
                        src_subtask_object_pose,
                    )
                else:
                    print(f"skipping transformation for {subtask_object_name}")

                    # Skip transformation if no reference object is provided
                    transformed_eef_poses = src_eef_poses

        # Construct trajectory for the transformed segment.
        transformed_seq = WaypointSequence.from_poses(
            poses=transformed_eef_poses,
            gripper_actions=src_subtask_gripper_actions,
            action_noise=subtask_configs[subtask_ind].action_noise,
        )
        transformed_traj = WaypointTrajectory()
        transformed_traj.add_waypoint_sequence(transformed_seq)

        return transformed_traj

    def merge_eef_subtask_trajectory(
        self,
        env_id: int,
        eef_name: str,
        subtask_index: int,
        prev_executed_traj: list[Waypoint] | None,
        subtask_trajectory: WaypointTrajectory,
    ) -> list[Waypoint]:
        """Merge a subtask trajectory into an executable trajectory for the robot end-effector.

        This constructs a new `WaypointTrajectory` by first creating an initial
        interpolation segment, then merging the provided `subtask_trajectory` onto it.
        The initial segment begins either from the last executed target waypoint of the
        previous subtask (if configured) or from the robot's current end-effector pose.

        Behavior:

        - If `datagen_config.generation_interpolate_from_last_target_pose` is True and
          this is not the first subtask, interpolation starts from the last waypoint of
          `prev_executed_traj`.
        - Otherwise, interpolation starts from the current robot EEF pose (queried from the env)
          and uses the first waypoint's gripper action and the subtask's action noise.
        - The merge uses `num_interpolation_steps`, `num_fixed_steps`, and optionally
          `apply_noise_during_interpolation` from the corresponding `SubTaskConfig`.
        - The temporary initial waypoint used to enable interpolation is removed before returning.

        Args:
            env_id: Environment index to query the current robot EEF pose when needed.
            eef_name: Name/key of the end-effector whose trajectory is being merged.
            subtask_index: Index of the subtask within `subtask_configs[eef_name]` driving interpolation parameters.
            prev_executed_traj: The previously executed trajectory used to
                seed interpolation from its last target waypoint. Required when interpolation-from-last-target
                is enabled and this is not the first subtask.
            subtask_trajectory:
                Trajectory segment for the current subtask that will be merged after the initial interpolation segment.

        Returns:
            The full sequence of waypoints to execute (initial interpolation segment followed by the subtask segment),
            with the temporary initial waypoint removed.
        """
        """融合一个子任务轨迹到一个可执行的轨迹，用于机器人最终效应器。

        这是在创建一个新的`WaypointTrajectory`，首先创建一个初始的插孔段，然后将提供的`subtask_trajectory`合并到它上。
        最初的段落从前次任务的最后执行目标路线点 (如果配置) 或从机器人的当前终端效果器姿势开始。

        Behavior:

        - 如果`datagen_config.generation_interpolate_from_last_target_pose`是True，而不是第一个子任务，则插射从`prev_executed_
          traj`的最后路线点开始。
        - 否则，插射从当前的机器人EEF姿势开始 (从env要求) 并使用第一步点的抓住器操作和子任务的操作噪音。
        - 合并使用`num_interpolation_steps`，`num_fixed_steps`，并从相应的`SubTaskConfig`中选择
          `apply_noise_during_interpolation`。
        - 在返回之前，将用于实现插射的临时初始路线点移除。

        参数：
            env_id: 在需要时，查询当前机器人EEF姿势。
            eef_name: 结合轨迹的末端执行器名称/关键。
            subtask_index: 在 `subtask_configs[eef_name]` 驱动插孔参数内的子任务索引。
            prev_executed_traj: 之前执行的轨迹用于从最后一个目标路线点开始插射。
                                需要在启用最后目标中断时，而这不是第一个子任务。
            subtask_trajectory: 目前的子任务轨迹段，将在最初的插射段后合并。

        返回：
            执行的通路点的完整序列 (初始插图段，随后是子任务段)，
            with the temporary initial waypoint removed.
        """
        is_first_subtask = subtask_index == 0
        # We will construct a WaypointTrajectory instance to keep track of robot control targets
        # and then execute it once we have the trajectory.
        traj_to_execute = WaypointTrajectory()

        if self.env_cfg.datagen_config.generation_interpolate_from_last_target_pose and (not is_first_subtask):
            # Interpolation segment will start from last target pose (which may not have been achieved).
            assert prev_executed_traj is not None
            last_waypoint = prev_executed_traj[-1]
            init_sequence = WaypointSequence(sequence=[last_waypoint])
        else:
            # Interpolation segment will start from current robot eef pose.
            init_sequence = WaypointSequence.from_poses(
                poses=self.env.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0].unsqueeze(0),
                gripper_actions=subtask_trajectory[0].gripper_action.unsqueeze(0),
                action_noise=self.env_cfg.subtask_configs[eef_name][subtask_index].action_noise,
            )
        traj_to_execute.add_waypoint_sequence(init_sequence)

        # Merge this trajectory into our trajectory using linear interpolation.
        # Interpolation will happen from the initial pose (@init_sequence) to the first element of @transformed_seq.
        traj_to_execute.merge(
            subtask_trajectory,
            num_steps_interp=self.env_cfg.subtask_configs[eef_name][subtask_index].num_interpolation_steps,
            num_steps_fixed=self.env_cfg.subtask_configs[eef_name][subtask_index].num_fixed_steps,
            action_noise=(
                float(self.env_cfg.subtask_configs[eef_name][subtask_index].apply_noise_during_interpolation)
                * self.env_cfg.subtask_configs[eef_name][subtask_index].action_noise
            ),
        )

        # We initialized @traj_to_execute with a pose to allow @merge to handle linear interpolation
        # for us. However, we can safely discard that first waypoint now, and just start by executing
        # the rest of the trajectory (interpolation segment and transformed subtask segment).
        traj_to_execute.pop_first()

        # Return the generated trajectory
        return traj_to_execute.get_full_sequence().sequence

    async def generate(  # noqa: C901
        self,
        env_id: int,
        success_term: TerminationTermCfg,
        env_reset_queue: asyncio.Queue | None = None,
        env_action_queue: asyncio.Queue | None = None,
        pause_subtask: bool = False,
        export_demo: bool = True,
        motion_planner: Any | None = None,
    ) -> dict:
        """Attempt to generate a new demonstration.

        Args:
            env_id: environment ID
            success_term: success function to check if the task is successful
            env_reset_queue: queue to store environment IDs for reset
            env_action_queue: queue to store actions for each environment
            pause_subtask: whether to pause the subtask generation
            export_demo: whether to export the demo
            motion_planner: motion planner to use for motion planning

        Returns:
            A dictionary containing the following items:
                - initial_state (dict): initial simulator state for the executed trajectory
                - states (list): simulator state at each timestep
                - observations (list): observation dictionary at each timestep
                - datagen_infos (list): datagen_info at each timestep
                - actions (np.array): action executed at each timestep
                - success (bool): whether the trajectory successfully solved the task or not
                - src_demo_inds (list): list of selected source demonstration indices for each subtask
                - src_demo_labels (np.array): same as @src_demo_inds, but repeated to have a label for
                  each timestep of the trajectory.
        """
        """试图产生新的示范。

        参数：
            env_id: 环境 ID
            success_term: 成功函数检查任务是否成功
            env_reset_queue: 排列存储环境 IDs重置
            env_action_queue: 排队存储每个环境的操作
            pause_subtask: 是否暂停子任务生成
            export_demo: 是否出口示范
            motion_planner: 运动规划器用于运动规划

        返回：
            包含以下内容的字典:
                - initial_state (dict):执行轨迹的初始仿真器状态
                - 状态 (列表):每个时间步骤的仿真器状态
                - 观测 (列表):每一步的观测字典
                - datagen_infos (列表):每个时间步骤 datagen_info
                - 动作 (np.array):每一步执行的动作
                - 成功 (bool):轨迹是否成功解决任务
                - src_demo_inds (列表):每个子任务所选择的源示范索引列表
                - src_demo_labels (np.array):与 @src_demo_inds相同，但重复为轨道的每一步都有标签。
        """
        # With skillgen, a motion planner is required to generate collision-free transitions between subtasks.
        if self.env_cfg.datagen_config.use_skillgen and motion_planner is None:
            raise ValueError("motion_planner must be provided if use_skillgen is True")

        # reset the env to create a new task demo instance
        env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=self.env.device)
        self.env.recorder_manager.reset(env_ids=env_id_tensor)
        await env_reset_queue.put(env_id)
        await env_reset_queue.join()
        new_initial_state = self.env.scene.get_state(is_relative=True)

        # create runtime subtask constraint rules from subtask constraint configs
        runtime_subtask_constraints_dict = {}
        for subtask_constraint in self.env_cfg.task_constraint_configs:
            runtime_subtask_constraints_dict.update(subtask_constraint.generate_runtime_subtask_constraints())

        # save generated data in these variables
        generated_states = []
        generated_obs = []
        generated_actions = []
        generated_success = False

        # some eef-specific state variables used during generation
        current_eef_selected_src_demo_indices = {}
        current_eef_subtask_trajectories: dict[str, list[Waypoint]] = {}
        current_eef_subtask_indices = {}
        next_eef_subtask_indices_after_motion = {}
        next_eef_subtask_trajectories_after_motion = {}
        current_eef_subtask_step_indices = {}
        eef_subtasks_done = {}
        for eef_name in self.env_cfg.subtask_configs.keys():
            current_eef_selected_src_demo_indices[eef_name] = None
            current_eef_subtask_trajectories[eef_name] = []  # type of list of Waypoint
            current_eef_subtask_indices[eef_name] = 0
            next_eef_subtask_indices_after_motion[eef_name] = None
            next_eef_subtask_trajectories_after_motion[eef_name] = None
            current_eef_subtask_step_indices[eef_name] = None
            eef_subtasks_done[eef_name] = False

        prev_src_demo_datagen_info_pool_size = 0

        if self.env_cfg.datagen_config.use_navigation_controller:
            was_navigating = False

        # While loop that runs per time step
        while True:
            async with self.src_demo_datagen_info_pool.asyncio_lock:
                if len(self.src_demo_datagen_info_pool.datagen_infos) > prev_src_demo_datagen_info_pool_size:
                    # src_demo_datagen_info_pool at this point may be updated with new demos,
                    # So we need to update subtask boundaries again
                    randomized_subtask_boundaries = (
                        self.randomize_subtask_boundaries()
                    )  # shape [N, S, 2], last dim is start and end action lengths
                    prev_src_demo_datagen_info_pool_size = len(self.src_demo_datagen_info_pool.datagen_infos)

                # Generate trajectory for a subtask for the eef that is currently at the beginning of a subtask
                for eef_name, eef_subtask_step_index in current_eef_subtask_step_indices.items():
                    if eef_subtask_step_index is None:
                        # Trajectory stored in current_eef_subtask_trajectories[eef_name] has been executed,
                        # So we need to determine the next trajectory
                        # Note: This condition is the "resume-after-motion-plan" gate for skillgen. When
                        # use_skillgen=False (vanilla Mimic), next_eef_subtask_indices_after_motion[eef_name]
                        # remains None, so this condition is always True and the else-branch below is never taken.
                        # The else-branch is only used right after executing a motion-planned transition (skillgen)
                        # to resume the actual subtask trajectory.
                        if next_eef_subtask_indices_after_motion[eef_name] is None:
                            # This is the beginning of a new subtask, so generate a new trajectory accordingly
                            eef_subtask_trajectory = self.generate_eef_subtask_trajectory(
                                env_id,
                                eef_name,
                                current_eef_subtask_indices[eef_name],
                                randomized_subtask_boundaries,
                                runtime_subtask_constraints_dict,
                                current_eef_selected_src_demo_indices,  # updated in the method
                            )
                            # With skillgen, use a motion planner to transition between subtasks.
                            if self.env_cfg.datagen_config.use_skillgen:
                                # Define the goal for the motion planner: the start of the next subtask.
                                target_eef_pose = eef_subtask_trajectory[0].pose
                                target_gripper_action = eef_subtask_trajectory[0].gripper_action

                                # Determine expected object attachment using environment-specific logic (optional)
                                expected_attached_object = None
                                if hasattr(self.env, "get_expected_attached_object"):
                                    expected_attached_object = self.env.get_expected_attached_object(
                                        eef_name, current_eef_subtask_indices[eef_name], self.env.cfg
                                    )

                                # Plan motion using motion planner with comprehensive world update
                                # and attachment handling
                                if motion_planner:
                                    print(f"\n--- Environment {env_id}: Planning motion to target pose ---")
                                    print(f"Target pose: {target_eef_pose}")
                                    print(f"Expected attached object: {expected_attached_object}")

                                    # This call updates the planner's world model and computes the trajectory.
                                    planning_success = motion_planner.update_world_and_plan_motion(
                                        target_pose=target_eef_pose,
                                        expected_attached_object=expected_attached_object,
                                        env_id=env_id,
                                        step_size=getattr(motion_planner, "step_size", None),
                                        enable_retiming=hasattr(motion_planner, "step_size")
                                        and motion_planner.step_size is not None,
                                    )

                                    # If planning succeeds, execute the planner's trajectory first.
                                    if planning_success:
                                        print(f"Env {env_id}: Motion planning succeeded")
                                        # The original subtask trajectory is stored to be executed after the transition.
                                        next_eef_subtask_trajectories_after_motion[eef_name] = eef_subtask_trajectory
                                        next_eef_subtask_indices_after_motion[eef_name] = current_eef_subtask_indices[
                                            eef_name
                                        ]
                                        # Mark the current subtask as invalid (-1) until the transition is done.
                                        current_eef_subtask_indices[eef_name] = -1

                                        # Convert the planner's output into a sequence of waypoints to be executed.
                                        current_eef_subtask_trajectories[eef_name] = (
                                            self._convert_planned_trajectory_to_waypoints(
                                                motion_planner, target_gripper_action
                                            )
                                        )
                                        current_eef_subtask_step_indices[eef_name] = 0
                                        print(
                                            f"Generated {len(current_eef_subtask_trajectories[eef_name])} waypoints"
                                            " from motion plan"
                                        )

                                    else:
                                        # If planning fails, abort the data generation trial.
                                        print(f"Env {env_id}: Motion planning failed for {eef_name}")
                                        return {"success": False}
                            else:
                                # Without skillgen, transition using simple interpolation.
                                current_eef_subtask_trajectories[eef_name] = self.merge_eef_subtask_trajectory(
                                    env_id,
                                    eef_name,
                                    current_eef_subtask_indices[eef_name],
                                    current_eef_subtask_trajectories[eef_name],
                                    eef_subtask_trajectory,
                                )
                                current_eef_subtask_step_indices[eef_name] = 0
                        else:
                            # Motion-planned trajectory has been executed, so we are ready to move to
                            # execute the next subtask
                            print("Finished executing motion-planned trajectory")
                            # It is important to pass the prev_executed_traj to merge_eef_subtask_trajectory
                            # so that it can correctly interpolate from the last pose of the motion-planned trajectory
                            prev_executed_traj = current_eef_subtask_trajectories[eef_name]
                            current_eef_subtask_indices[eef_name] = next_eef_subtask_indices_after_motion[eef_name]
                            current_eef_subtask_trajectories[eef_name] = self.merge_eef_subtask_trajectory(
                                env_id,
                                eef_name,
                                current_eef_subtask_indices[eef_name],
                                prev_executed_traj,
                                next_eef_subtask_trajectories_after_motion[eef_name],
                            )
                            current_eef_subtask_step_indices[eef_name] = 0
                            next_eef_subtask_trajectories_after_motion[eef_name] = None
                            next_eef_subtask_indices_after_motion[eef_name] = None

            # Determine the next waypoint for each eef based on the current subtask constraints
            eef_waypoint_dict = {}
            for eef_name in sorted(self.env_cfg.subtask_configs.keys()):
                # Handle constraints
                step_ind = current_eef_subtask_step_indices[eef_name]
                subtask_ind = current_eef_subtask_indices[eef_name]
                if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
                    task_constraint = runtime_subtask_constraints_dict[(eef_name, subtask_ind)]
                    if task_constraint["type"] == SubTaskConstraintType._SEQUENTIAL_LATTER:
                        min_time_diff = task_constraint["min_time_diff"]
                        if not task_constraint["fulfilled"]:
                            if (
                                min_time_diff == -1
                                or step_ind >= len(current_eef_subtask_trajectories[eef_name]) - min_time_diff
                            ):
                                if step_ind > 0:
                                    # Wait at the same step
                                    step_ind -= 1
                                    current_eef_subtask_step_indices[eef_name] = step_ind

                    elif task_constraint["type"] == SubTaskConstraintType.COORDINATION:
                        synchronous_steps = task_constraint["synchronous_steps"]
                        concurrent_task_spec_key = task_constraint["concurrent_task_spec_key"]
                        concurrent_subtask_ind = task_constraint["concurrent_subtask_ind"]
                        concurrent_task_fulfilled = runtime_subtask_constraints_dict[
                            (concurrent_task_spec_key, concurrent_subtask_ind)
                        ]["fulfilled"]

                        if (
                            task_constraint["coordination_synchronize_start"]
                            and current_eef_subtask_indices[concurrent_task_spec_key] < concurrent_subtask_ind
                        ):
                            # The concurrent eef is not yet at the concurrent subtask, so wait at the first action
                            # This also makes sure that the concurrent task starts at the same time as this task
                            step_ind = 0
                            current_eef_subtask_step_indices[eef_name] = 0
                        else:
                            if (
                                not concurrent_task_fulfilled
                                and step_ind >= len(current_eef_subtask_trajectories[eef_name]) - synchronous_steps
                            ):
                                # Trigger concurrent task
                                runtime_subtask_constraints_dict[(concurrent_task_spec_key, concurrent_subtask_ind)][
                                    "fulfilled"
                                ] = True

                            if not task_constraint["fulfilled"]:
                                if step_ind >= len(current_eef_subtask_trajectories[eef_name]) - synchronous_steps:
                                    if step_ind > 0:
                                        step_ind -= 1
                                        current_eef_subtask_step_indices[eef_name] = step_ind  # wait here

                waypoint = current_eef_subtask_trajectories[eef_name][step_ind]

                # Update visualization if motion planner is available
                if motion_planner and motion_planner.visualize_spheres:
                    current_joints = self.env.scene["robot"].data.joint_pos[env_id]
                    motion_planner._update_visualization_at_joint_positions(current_joints)

                eef_waypoint_dict[eef_name] = waypoint
            multi_waypoint = MultiWaypoint(eef_waypoint_dict)

            # Execute the next waypoints for all eefs
            exec_results = await multi_waypoint.execute(
                env=self.env,
                success_term=success_term,
                env_id=env_id,
                env_action_queue=env_action_queue,
            )

            # Update execution state buffers
            if len(exec_results["states"]) > 0:
                generated_states.extend(exec_results["states"])
                generated_obs.extend(exec_results["observations"])
                generated_actions.extend(exec_results["actions"])
                generated_success = generated_success or exec_results["success"]

            # Get the navigation state
            if self.env_cfg.datagen_config.use_navigation_controller:
                processed_nav_subtask = False
                navigation_state = self.env.get_navigation_state(env_id)
                assert navigation_state is not None, "Navigation state cannot be None when using navigation controller"
                is_navigating = navigation_state["is_navigating"]
                navigation_goal_reached = navigation_state["navigation_goal_reached"]

            for eef_name in self.env_cfg.subtask_configs.keys():
                current_eef_subtask_step_indices[eef_name] += 1

                # Execute locomanip navigation controller if it is enabled via the use_navigation_controller flag
                if self.env_cfg.datagen_config.use_navigation_controller:
                    if "body" not in self.env_cfg.subtask_configs.keys():
                        error_msg = (
                            'End effector with name "body" not found in subtask configs. "body" must be a valid end'
                            " effector to use the navigation controller.\n"
                        )
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)

                    # Repeat the last nav subtask action if the robot is navigating and hasn't reached the waypoint goal
                    if (
                        current_eef_subtask_step_indices["body"] == len(current_eef_subtask_trajectories["body"]) - 1
                        and not processed_nav_subtask
                    ):
                        if is_navigating and not navigation_goal_reached:
                            for name in self.env_cfg.subtask_configs.keys():
                                current_eef_subtask_step_indices[name] -= 1
                            processed_nav_subtask = True

                    # Else skip to the end of the nav subtask if the robot has reached the waypoint goal before the end
                    # of the human recorded trajectory
                    elif was_navigating and not is_navigating and not processed_nav_subtask:
                        number_of_steps_to_skip = len(current_eef_subtask_trajectories["body"]) - (
                            current_eef_subtask_step_indices["body"] + 1
                        )
                        for name in self.env_cfg.subtask_configs.keys():
                            if current_eef_subtask_step_indices[name] + number_of_steps_to_skip < len(
                                current_eef_subtask_trajectories[name]
                            ):
                                current_eef_subtask_step_indices[name] = (
                                    current_eef_subtask_step_indices[name] + number_of_steps_to_skip
                                )
                            else:
                                current_eef_subtask_step_indices[name] = len(current_eef_subtask_trajectories[name]) - 1
                        processed_nav_subtask = True

                subtask_ind = current_eef_subtask_indices[eef_name]
                if current_eef_subtask_step_indices[eef_name] == len(
                    current_eef_subtask_trajectories[eef_name]
                ):  # Subtask done
                    if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
                        task_constraint = runtime_subtask_constraints_dict[(eef_name, subtask_ind)]
                        if task_constraint["type"] == SubTaskConstraintType._SEQUENTIAL_FORMER:
                            constrained_task_spec_key = task_constraint["constrained_task_spec_key"]
                            constrained_subtask_ind = task_constraint["constrained_subtask_ind"]
                            runtime_subtask_constraints_dict[(constrained_task_spec_key, constrained_subtask_ind)][
                                "fulfilled"
                            ] = True
                        elif task_constraint["type"] == SubTaskConstraintType.COORDINATION:
                            concurrent_task_spec_key = task_constraint["concurrent_task_spec_key"]
                            concurrent_subtask_ind = task_constraint["concurrent_subtask_ind"]
                            # Concurrent_task_spec_idx = task_spec_keys.index(concurrent_task_spec_key)
                            task_constraint["finished"] = True
                            # Check if concurrent task has been finished
                            assert (
                                runtime_subtask_constraints_dict[(concurrent_task_spec_key, concurrent_subtask_ind)][
                                    "finished"
                                ]
                                or current_eef_subtask_step_indices[concurrent_task_spec_key]
                                >= len(current_eef_subtask_trajectories[concurrent_task_spec_key]) - 1
                            )

                    if pause_subtask:
                        input(
                            f"Pausing after subtask {current_eef_subtask_indices[eef_name]} of {eef_name} execution."
                            " Press any key to continue..."
                        )
                    # This is a check to see if this arm has completed all the subtasks
                    if current_eef_subtask_indices[eef_name] == len(self.env_cfg.subtask_configs[eef_name]) - 1:
                        eef_subtasks_done[eef_name] = True
                        # If all subtasks done for this arm, repeat last waypoint to make sure this arm does not move
                        current_eef_subtask_trajectories[eef_name].append(
                            current_eef_subtask_trajectories[eef_name][-1]
                        )
                    else:
                        current_eef_subtask_step_indices[eef_name] = None
                        current_eef_subtask_indices[eef_name] += 1

            if self.env_cfg.datagen_config.use_navigation_controller:
                was_navigating = copy.deepcopy(is_navigating)

            # Check if all eef_subtasks_done values are True
            if all(eef_subtasks_done.values()):
                break

        # Merge numpy arrays
        if len(generated_actions) > 0:
            generated_actions = torch.cat(generated_actions, dim=0)

        # Set success to the recorded episode data and export to file
        self.env.recorder_manager.set_success_to_episodes(
            env_id_tensor, torch.tensor([[generated_success]], dtype=torch.bool, device=self.env.device)
        )
        if export_demo:
            self.env.recorder_manager.export_episodes(env_id_tensor)

        results = dict(
            initial_state=new_initial_state,
            states=generated_states,
            observations=generated_obs,
            actions=generated_actions,
            success=generated_success,
        )
        return results

    def _convert_planned_trajectory_to_waypoints(
        self, motion_planner: Any, gripper_action: torch.Tensor
    ) -> list[Waypoint]:
        """
        (skillgen) Convert a motion planner's output trajectory into a list of Waypoint objects.

        The motion planner provides a sequence of planned 4x4 poses. This method wraps each
        pose into a `Waypoint`, pairing it with the provided `gripper_action` and an optional
        per-timestep noise value sourced from the planner config (`motion_noise_scale`).

        Args:
            motion_planner: Planner instance exposing `get_planned_poses()` and an optional
                `config.motion_noise_scale` float.
            gripper_action: Gripper actuation to associate with each planned pose.

        Returns:
            list[Waypoint]: Sequence of waypoints corresponding to the planned trajectory.
        """
        """(技能) 将运动规划器的输出轨迹转化为Waypoint对象列表。

        运动规划器提供了计划的4×4姿势。
        这种方法将每个姿势包装成`Waypoint`，将其与提供的`gripper_action`和从规划器配置 (`motion_noise_scale`) 来源的可选每步噪音值结合。

        参数：
            motion_planner: 规划器实例暴露`get_planned_poses()`和可选`config.motion_noise_scale`浮动。
            gripper_action: 按动作，与每个计划的姿势相结合。

        返回：
            列表[路线点]:符合计划轨迹的路线点序列。
        """
        # Get motion noise scale from the planner's configuration
        motion_noise_scale = getattr(motion_planner.config, "motion_noise_scale", 0.0)

        waypoints = []
        planned_poses = motion_planner.get_planned_poses()

        for planned_pose in planned_poses:
            waypoint = Waypoint(pose=planned_pose, gripper_action=gripper_action, noise=motion_noise_scale)
            waypoints.append(waypoint)

        return waypoints
