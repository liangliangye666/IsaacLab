# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


class Se3RelRetargeter(RetargeterBase):
    """Retargets OpenXR hand tracking data to end-effector commands using relative positioning.

    This retargeter calculates delta poses between consecutive hand joint poses to generate incremental robot movements.
    It can either:
    - Use the wrist position and orientation
    - Use the midpoint between thumb and index finger (pinch position)

    Features:
    - Optional constraint to zero out X/Y rotations (keeping only Z-axis rotation)
    - Motion smoothing with adjustable parameters
    - Optional visualization of the target end-effector pose
    """
    """将OpenXR手动跟踪数据返回使用相对定位的末端执行器命令。

    这种重定位器计算连续的手关键姿势之间的三角形姿势，以产生增量机器人运动。
    它可以:
    - 使用手腕的位置和方向
    - 使用指公和指公之间的中点 (点位置)

    Features:
    - 选择性限制为零掉X/Y旋转 (仅保持Z轴旋转)
    - 具有可调节参数的运动平滑
    - 目标末端执行器姿势的可选可视化
    """

    def __init__(
        self,
        cfg: Se3RelRetargeterCfg,
    ):
        """Initialize the relative motion retargeter.

        Args:
            bound_hand: The hand to track (DeviceBase.TrackingTarget.HAND_LEFT or DeviceBase.TrackingTarget.HAND_RIGHT)
            zero_out_xy_rotation: If True, ignore rotations around x and y axes, allowing only z-axis rotation
            use_wrist_rotation: If True, use wrist rotation for control instead of averaging finger orientations
            use_wrist_position: If True, use wrist position instead of pinch position (midpoint between fingers)
            delta_pos_scale_factor: Amplification factor for position changes (higher = larger robot movements)
            delta_rot_scale_factor: Amplification factor for rotation changes (higher = larger robot rotations)
            alpha_pos: Position smoothing parameter (0-1); higher values track more closely to input,
                lower values smooth more
            alpha_rot: Rotation smoothing parameter (0-1); higher values track more closely to input,
                lower values smooth more
            enable_visualization: If True, show a visual marker representing the target end-effector pose
            device: The device to place the returned tensor on ('cpu' or 'cuda')
        """
        """启动相对运动重定向器。

        参数：
            bound_hand: 追踪手 (DeviceBase.TrackingTarget.HAND_LEFT或DeviceBase.TrackingTarget.HAND_RIGHT)
            zero_out_xy_rotation: 如果True，忽略在x和y轴周围的旋转，只允许z轴旋转
            use_wrist_rotation: 如果True，使用手腕旋转来控制，而不是平均指指向
            use_wrist_position: 如果True，使用手腕位置而不是点位置 (指之间的中点)
            delta_pos_scale_factor: 对位置变化的放大因素 (更高 =更大的机器人运动)
            delta_rot_scale_factor: 转动变化放大因子 (更高 =更大的机器人转动)
            alpha_pos: 位置平滑参数 (0-1)；较高的值更接近输入，较低的值更平滑
            alpha_rot: 旋转平滑参数 (0-1)；较高的值更接近输入，较低的值更平滑
            enable_visualization: 如果True，显示目标末端执行器姿势的视觉标记
            device: 返回门器的装置 ("cpu"或"cuda")
        """
        # Store the hand to track
        if cfg.bound_hand not in [DeviceBase.TrackingTarget.HAND_LEFT, DeviceBase.TrackingTarget.HAND_RIGHT]:
            raise ValueError(
                "bound_hand must be either DeviceBase.TrackingTarget.HAND_LEFT or DeviceBase.TrackingTarget.HAND_RIGHT"
            )
        super().__init__(cfg)
        self.bound_hand = cfg.bound_hand

        self._zero_out_xy_rotation = cfg.zero_out_xy_rotation
        self._use_wrist_rotation = cfg.use_wrist_rotation
        self._use_wrist_position = cfg.use_wrist_position
        self._delta_pos_scale_factor = cfg.delta_pos_scale_factor
        self._delta_rot_scale_factor = cfg.delta_rot_scale_factor
        self._alpha_pos = cfg.alpha_pos
        self._alpha_rot = cfg.alpha_rot

        # Initialize smoothing state
        self._smoothed_delta_pos = np.zeros(3)
        self._smoothed_delta_rot = np.zeros(3)

        # Define thresholds for small movements
        self._position_threshold = 0.001
        self._rotation_threshold = 0.01

        # Initialize visualization if enabled
        self._enable_visualization = cfg.enable_visualization
        if cfg.enable_visualization:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self._goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
            self._goal_marker.set_visibility(True)
            self._visualization_pos = np.zeros(3)
            self._visualization_rot = np.array([1.0, 0.0, 0.0, 0.0])

        self._previous_thumb_tip = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._previous_index_tip = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._previous_wrist = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def retarget(self, data: dict) -> torch.Tensor:
        """Convert hand joint poses to robot end-effector command.

        Args:
            data: Dictionary mapping tracking targets to joint data dictionaries.
                The joint names are defined in isaaclab.devices.openxr.common.HAND_JOINT_NAMES

        Returns:
            torch.Tensor: 6D tensor containing position (xyz) and rotation vector (rx,ry,rz)
                for the robot end-effector
        """
        """转换手关姿势为机器人终端执行器命令。

        参数：
            data: 根据"数据字典"的定义，
                  联合名称在isaaclab.devices.openxr.common.HAND_JOINT_NAMES中定义

        返回：
            torch.Tensor: 包含位置 (xyz) 和旋转向量 (rx，ry，rz) 的6D子
                for the robot end-effector
        """
        # Extract key joint poses from the bound hand
        hand_data = data[self.bound_hand]
        thumb_tip = hand_data.get("thumb_tip")
        index_tip = hand_data.get("index_tip")
        wrist = hand_data.get("wrist")

        delta_thumb_tip = self._calculate_delta_pose(thumb_tip, self._previous_thumb_tip)
        delta_index_tip = self._calculate_delta_pose(index_tip, self._previous_index_tip)
        delta_wrist = self._calculate_delta_pose(wrist, self._previous_wrist)
        ee_command_np = self._retarget_rel(delta_thumb_tip, delta_index_tip, delta_wrist)

        self._previous_thumb_tip = thumb_tip.copy()
        self._previous_index_tip = index_tip.copy()
        self._previous_wrist = wrist.copy()

        # Convert to torch tensor
        ee_command = torch.tensor(ee_command_np, dtype=torch.float32, device=self._sim_device)

        return ee_command

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]

    def _calculate_delta_pose(self, joint_pose: np.ndarray, previous_joint_pose: np.ndarray) -> np.ndarray:
        """Calculate delta pose from previous joint pose.

        Args:
            joint_pose: Current joint pose (position and orientation)
            previous_joint_pose: Previous joint pose for the same joint

        Returns:
            np.ndarray: 6D array with position delta (xyz) and rotation delta as axis-angle (rx,ry,rz)
        """
        """从前的联合姿势计算多达姿势。

        参数：
            joint_pose: 目前的联合姿势 (位置和方向)
            previous_joint_pose: 在同一关节上，以前的关节姿势

        返回：
            np.ndarray: 6D阵列，位置 delta (xyz) 和旋转 delta作为轴角 (rx，ry，rz)
        """
        delta_pos = joint_pose[:3] - previous_joint_pose[:3]
        abs_rotation = Rotation.from_quat([*joint_pose[4:7], joint_pose[3]])
        previous_rot = Rotation.from_quat([*previous_joint_pose[4:7], previous_joint_pose[3]])
        relative_rotation = abs_rotation * previous_rot.inv()
        return np.concatenate([delta_pos, relative_rotation.as_rotvec()])

    def _retarget_rel(self, thumb_tip: np.ndarray, index_tip: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        """Handle relative (delta) pose retargeting.

        Args:
            thumb_tip: Delta pose of thumb tip
            index_tip: Delta pose of index tip
            wrist: Delta pose of wrist

        Returns:
            np.ndarray: 6D array with position delta (xyz) and rotation delta (rx,ry,rz)
        """
        """处理相对 (delta) 姿势重定向。

        参数：
            thumb_tip: 指尖的直角姿势
            index_tip: 索引尖端的三角姿势
            wrist: 手腕的多角姿势

        返回：
            np.ndarray: 6D阵列与位置 delta (xyz) 和旋转 delta (rx，ry，rz)
        """
        # Get position
        if self._use_wrist_position:
            position = wrist[:3]
        else:
            position = (thumb_tip[:3] + index_tip[:3]) / 2

        # Get rotation
        if self._use_wrist_rotation:
            rotation = wrist[3:6]  # rx, ry, rz
        else:
            rotation = (thumb_tip[3:6] + index_tip[3:6]) / 2

        # Apply zero_out_xy_rotation regardless of rotation source
        if self._zero_out_xy_rotation:
            rotation[0] = 0  # x-axis
            rotation[1] = 0  # y-axis

        # Smooth and scale position
        self._smoothed_delta_pos = self._alpha_pos * position + (1 - self._alpha_pos) * self._smoothed_delta_pos
        if np.linalg.norm(self._smoothed_delta_pos) < self._position_threshold:
            self._smoothed_delta_pos = np.zeros(3)
        position = self._smoothed_delta_pos * self._delta_pos_scale_factor

        # Smooth and scale rotation
        self._smoothed_delta_rot = self._alpha_rot * rotation + (1 - self._alpha_rot) * self._smoothed_delta_rot
        if np.linalg.norm(self._smoothed_delta_rot) < self._rotation_threshold:
            self._smoothed_delta_rot = np.zeros(3)
        rotation = self._smoothed_delta_rot * self._delta_rot_scale_factor

        # Update visualization if enabled
        if self._enable_visualization:
            # Convert rotation vector to quaternion and combine with current rotation
            delta_quat = Rotation.from_rotvec(rotation).as_quat()  # x, y, z, w format
            current_rot = Rotation.from_quat([self._visualization_rot[1:], self._visualization_rot[0]])
            new_rot = Rotation.from_quat(delta_quat) * current_rot
            self._visualization_pos = self._visualization_pos + position
            # Convert back to w, x, y, z format
            self._visualization_rot = np.array([new_rot.as_quat()[3], *new_rot.as_quat()[:3]])
            self._update_visualization()

        return np.concatenate([position, rotation])

    def _update_visualization(self):
        """Update visualization markers with current pose."""
        """更新可视化标记与当前姿势。"""
        if self._enable_visualization:
            trans = np.array([self._visualization_pos])
            quat = Rotation.from_matrix(self._visualization_rot).as_quat()
            rot = np.array([np.array([quat[3], quat[0], quat[1], quat[2]])])
            self._goal_marker.visualize(translations=trans, orientations=rot)


@dataclass
class Se3RelRetargeterCfg(RetargeterCfg):
    """Configuration for relative position retargeter."""
    """对相对位置重定位器的配置。"""

    zero_out_xy_rotation: bool = True
    use_wrist_rotation: bool = False
    use_wrist_position: bool = True
    delta_pos_scale_factor: float = 10.0
    delta_rot_scale_factor: float = 10.0
    alpha_pos: float = 0.5
    alpha_rot: float = 0.5
    enable_visualization: bool = False
    bound_hand: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.HAND_RIGHT
    retargeter_type: type[RetargeterBase] = Se3RelRetargeter
