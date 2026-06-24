# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from pink.tasks import FrameTask

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.pink_ik import PinkIKController
from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import pink_actions_cfg


class PinkInverseKinematicsAction(ActionTerm):
    r"""Pink Inverse Kinematics action term.

    This action term processes the action tensor and sets these setpoints in the pink IK framework.
    The action tensor is ordered in the order of the tasks defined in PinkIKControllerCfg.
    """
    """粉红色反向动力学动作项。

    这种动作项处理动作张量，并在粉红色IK框架中设置这些设置点。
    动作张量按PinkIKControllerCfg中定义的任务顺序排列。
    """

    cfg: pink_actions_cfg.PinkInverseKinematicsActionCfg
    """Configuration for the Pink Inverse Kinematics action term."""
    """粉红色逆动力学动作项的配置。"""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""
    """动作项适用于的关节资产。"""

    def __init__(self, cfg: pink_actions_cfg.PinkInverseKinematicsActionCfg, env: ManagerBasedEnv):
        """Initialize the Pink Inverse Kinematics action term.

        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        """开始使用"粉红色反动动力学"动作项。

        参数：
            cfg: 这个操作项的配置。
            env: 动作项将应用于的环境。
        """
        super().__init__(cfg, env)

        self._env = env
        self._sim_dt = env.sim.get_physics_dt()

        # Initialize joint information
        self._initialize_joint_info()

        # Initialize IK controllers
        self._initialize_ik_controllers()

        # Initialize action tensors
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # PhysX Articulation Floating joint indices offset from IsaacLab Articulation joint indices
        self._physx_floating_joint_indices_offset = 6

        # Pre-allocate tensors for runtime use
        self._initialize_helper_tensors()

    def _initialize_joint_info(self) -> None:
        """Initialize joint IDs and names based on configuration."""
        """根据配置初始化联合IDs和名称。"""
        # Resolve pink controlled joints
        self._isaaclab_controlled_joint_ids, self._isaaclab_controlled_joint_names = self._asset.find_joints(
            self.cfg.pink_controlled_joint_names
        )
        self.cfg.controller.joint_names = self._isaaclab_controlled_joint_names
        self._isaaclab_all_joint_ids = list(range(len(self._asset.data.joint_names)))
        self.cfg.controller.all_joint_names = self._asset.data.joint_names

        # Resolve hand joints
        self._hand_joint_ids, self._hand_joint_names = self._asset.find_joints(self.cfg.hand_joint_names)

        # Combine all joint information
        self._controlled_joint_ids = self._isaaclab_controlled_joint_ids + self._hand_joint_ids
        self._controlled_joint_names = self._isaaclab_controlled_joint_names + self._hand_joint_names

    def _initialize_ik_controllers(self) -> None:
        """Initialize Pink IK controllers for all environments."""
        """启动所有环境的粉色IK控制器。"""
        assert self._env.num_envs > 0, "Number of environments specified are less than 1."

        self._ik_controllers = []
        for _ in range(self._env.num_envs):
            self._ik_controllers.append(
                PinkIKController(
                    cfg=self.cfg.controller.copy(),
                    robot_cfg=self._env.scene.cfg.robot,
                    device=self.device,
                    controlled_joint_indices=self._isaaclab_controlled_joint_ids,
                )
            )

    def _initialize_helper_tensors(self) -> None:
        """Pre-allocate tensors and cache values for performance optimization."""
        """为优化性能预先分配子和缓存值。"""
        # Cache frequently used tensor versions of joint IDs to avoid repeated creation
        self._controlled_joint_ids_tensor = torch.tensor(self._controlled_joint_ids, device=self.device)

        # Cache base link index to avoid string lookup every time
        articulation_data = self._env.scene[self.cfg.controller.articulation_name].data
        self._base_link_idx = articulation_data.body_names.index(self.cfg.controller.base_link_name)

        # Pre-allocate working tensors
        # Count only FrameTask instances in variable_input_tasks (not all tasks)
        num_frame_tasks = sum(
            1 for task in self._ik_controllers[0].cfg.variable_input_tasks if isinstance(task, FrameTask)
        )
        self._num_frame_tasks = num_frame_tasks
        self._controlled_frame_poses = torch.zeros(num_frame_tasks, self.num_envs, 4, 4, device=self.device)

        # Pre-allocate tensor for base frame computations
        self._base_link_frame_buffer = torch.zeros(self.num_envs, 4, 4, device=self.device)

    # ==================== Properties ====================

    @property
    def hand_joint_dim(self) -> int:
        """Dimension for hand joint positions."""
        """适用于手关键位置的尺寸。"""
        return self.cfg.controller.num_hand_joints

    @property
    def position_dim(self) -> int:
        """Dimension for position (x, y, z)."""
        """位置的尺寸 (x，y，z)。"""
        return 3

    @property
    def orientation_dim(self) -> int:
        """Dimension for orientation (w, x, y, z)."""
        """导向尺寸 (w，x，y，z)。"""
        return 4

    @property
    def pose_dim(self) -> int:
        """Total pose dimension (position + orientation)."""
        """总姿势尺寸 (位置+方向)。"""
        return self.position_dim + self.orientation_dim

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (based on number of tasks and pose dimension)."""
        """动作空间的尺寸 (基于任务数量和姿势尺寸)。"""
        # Count only FrameTask instances in variable_input_tasks
        frame_tasks_count = sum(
            1 for task in self._ik_controllers[0].cfg.variable_input_tasks if isinstance(task, FrameTask)
        )
        return frame_tasks_count * self.pose_dim + self.hand_joint_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions tensor."""
        """得到原始的动作张量。"""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions tensor."""
        """得到处理的动作张量。"""
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the pink inverse kinematics action.
        It adds the following information to the base descriptor:
        - scale: The scale of the action term.
        - offset: The offset of the action term.
        - clip: The clip of the action term.
        - pink_controller_joint_names: The names of the pink controller joints.
        - hand_joint_names: The names of the hand joints.
        - controller_cfg: The configuration of the pink controller.

        Returns:
            The IO descriptor of the action term.
        """
        """动作项的IO描述符。

        这种描述符用于描述粉红色逆动力动力动力动力的作用项。
        它将以下信息添加到基础描述符中:
        - 规模:动作项的规模。
        - 抵消:动作项的抵消。
        - 动作项的裁剪。
        - pink_controller_joint_names:粉红色控制器关节的名称。
        - hand_joint_names:手关节的名称。
        - controller_cfg:粉红色控制器的配置。

        返回：
            动作项的IO描述符。
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "PinkInverseKinematicsAction"
        self._IO_descriptor.pink_controller_joint_names = self._isaaclab_controlled_joint_names
        self._IO_descriptor.hand_joint_names = self._hand_joint_names
        self._IO_descriptor.extras["controller_cfg"] = self.cfg.controller.__dict__
        return self._IO_descriptor

    # """
    # Operations.
    # """

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process the input actions and set targets for each task.

        Args:
            actions: The input actions tensor.
        """
        """处理输入动作，并为每个任务设定目标。

        参数：
            actions: 输入动作张量。
        """
        # Store raw actions
        self._raw_actions[:] = actions

        # Extract hand joint positions directly (no cloning needed)
        self._target_hand_joint_positions = actions[:, -self.hand_joint_dim :]

        # Get base link frame transformation
        self.base_link_frame_in_world_rf = self._get_base_link_frame_transform()

        # Process controlled frame poses (pass original actions, no clone needed)
        controlled_frame_poses = self._extract_controlled_frame_poses(actions)
        transformed_poses = self._transform_poses_to_base_link_frame(controlled_frame_poses)

        # Set targets for all tasks
        self._set_task_targets(transformed_poses)

    def _get_base_link_frame_transform(self) -> torch.Tensor:
        """Get the base link frame transformation matrix.

        Returns:
            Base link frame transformation matrix.
        """
        """获取基链框架转换矩阵。

        返回：
            基链框架转换矩阵
        """
        # Get base link frame pose in world origin using cached index
        articulation_data = self._env.scene[self.cfg.controller.articulation_name].data
        base_link_frame_in_world_origin = articulation_data.body_link_state_w[:, self._base_link_idx, :7]

        # Transform to environment origin frame (reuse buffer to avoid allocation)
        torch.sub(
            base_link_frame_in_world_origin[:, :3],
            self._env.scene.env_origins,
            out=self._base_link_frame_buffer[:, :3, 3],
        )

        # Copy orientation (avoid clone)
        base_link_frame_quat = base_link_frame_in_world_origin[:, 3:7]

        # Create transformation matrix
        return math_utils.make_pose(
            self._base_link_frame_buffer[:, :3, 3], math_utils.matrix_from_quat(base_link_frame_quat)
        )

    def _extract_controlled_frame_poses(self, actions: torch.Tensor) -> torch.Tensor:
        """Extract controlled frame poses from action tensor.

        Args:
            actions: The action tensor.

        Returns:
            Stacked controlled frame poses tensor.
        """
        """从动作张量中提取控制框架姿势。

        参数：
            actions: 动作张量。

        返回：
            堆叠的控制框架呈现度。
        """
        # Use pre-allocated tensor instead of list operations
        for task_index in range(self._num_frame_tasks):
            # Extract position and orientation for this task
            pos_start = task_index * self.pose_dim
            pos_end = pos_start + self.position_dim
            quat_start = pos_end
            quat_end = (task_index + 1) * self.pose_dim

            position = actions[:, pos_start:pos_end]
            quaternion = actions[:, quat_start:quat_end]

            # Create pose matrix directly into pre-allocated tensor
            self._controlled_frame_poses[task_index] = math_utils.make_pose(
                position, math_utils.matrix_from_quat(quaternion)
            )

        return self._controlled_frame_poses

    def _transform_poses_to_base_link_frame(self, poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform poses from world frame to base link frame.

        Args:
            poses: Poses in world frame.

        Returns:
            Tuple of (positions, rotation_matrices) in base link frame.
        """
        """从世界框架转换为基链框架。

        参数：
            poses: 在世界框架中姿势。

        返回：
            在基链框中 (位置，rotation_matrices) 的双倍。
        """
        # Transform poses to base link frame
        base_link_inv = math_utils.pose_inv(self.base_link_frame_in_world_rf)
        transformed_poses = math_utils.pose_in_A_to_pose_in_B(poses, base_link_inv)

        # Extract position and rotation
        positions, rotation_matrices = math_utils.unmake_pose(transformed_poses)

        return positions, rotation_matrices

    def _set_task_targets(self, transformed_poses: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Set targets for all tasks across all environments.

        Args:
            transformed_poses: Tuple of (positions, rotation_matrices) in base link frame.
        """
        """在所有环境中设定所有任务的目标。

        参数：
            transformed_poses: 在基链框中 (位置，rotation_matrices) 的双倍。
        """
        positions, rotation_matrices = transformed_poses

        for env_index, ik_controller in enumerate(self._ik_controllers):
            for frame_task_index, task in enumerate(ik_controller.cfg.variable_input_tasks):
                if isinstance(task, LocalFrameTask):
                    target = task.transform_target_to_base
                elif isinstance(task, FrameTask):
                    target = task.transform_target_to_world
                else:
                    continue

                # Set position and rotation targets using frame_task_index
                target.translation = positions[frame_task_index, env_index, :].cpu().numpy()
                target.rotation = rotation_matrices[frame_task_index, env_index, :].cpu().numpy()

                task.set_target(target)

    # ==================== Action Application ====================

    def apply_actions(self) -> None:
        """Apply the computed joint positions based on the inverse kinematics solution."""
        """根据反动动力学解决方案应用计算的关节位置。"""
        # Compute IK solutions for all environments
        ik_joint_positions = self._compute_ik_solutions()

        # Combine IK and hand joint positions
        all_joint_positions = torch.cat((ik_joint_positions, self._target_hand_joint_positions), dim=1)
        self._processed_actions = all_joint_positions

        # Apply gravity compensation to arm joints
        if self.cfg.enable_gravity_compensation:
            self._apply_gravity_compensation()

        # Apply joint position targets
        self._asset.set_joint_position_target(self._processed_actions, self._controlled_joint_ids)

    def _apply_gravity_compensation(self) -> None:
        """Apply gravity compensation to arm joints if not disabled in props."""
        """应对手臂关节的重力补偿，如果没有在道具中残疾。"""
        if not self._asset.cfg.spawn.rigid_props.disable_gravity:
            # Get gravity compensation forces using cached tensor
            if self._asset.is_fixed_base:
                gravity = torch.zeros_like(
                    self._asset.root_physx_view.get_gravity_compensation_forces()[:, self._controlled_joint_ids_tensor]
                )
            else:
                # If floating base, then need to skip the first 6 joints (base)
                gravity = self._asset.root_physx_view.get_gravity_compensation_forces()[
                    :, self._controlled_joint_ids_tensor + self._physx_floating_joint_indices_offset
                ]

            # Apply gravity compensation to arm joints
            self._asset.set_joint_effort_target(gravity, self._controlled_joint_ids)

    def _compute_ik_solutions(self) -> torch.Tensor:
        """Compute IK solutions for all environments.

        Returns:
            IK joint positions tensor for all environments.
        """
        """为所有环境计算IK解决方案。

        返回：
            对于所有环境来说，IK联合定位光器。
        """
        ik_solutions = []

        for env_index, ik_controller in enumerate(self._ik_controllers):
            # Get current joint positions for this environment
            current_joint_pos = self._asset.data.joint_pos.cpu().numpy()[env_index]

            # Compute IK solution
            joint_pos_des = ik_controller.compute(current_joint_pos, self._sim_dt)
            ik_solutions.append(joint_pos_des)

        return torch.stack(ik_solutions)

    # ==================== Reset ====================

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.

        Args:
            env_ids: A list of environment IDs to reset. If None, all environments are reset.
        """
        """为指定环境重置操作时间。

        参数：
            env_ids: 设置环境 IDs的列表。
                     如果None，所有环境都会重置。
        """
        self._raw_actions[env_ids] = torch.zeros(self.action_dim, device=self.device)
