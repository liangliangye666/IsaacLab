# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class G1TriHandUpperBodyMotionControllerGripperRetargeter(RetargeterBase):
    """Retargeter for G1 gripper that outputs a boolean state based on controller trigger input,
    concatenated with the retargeted wrist pose.

    Gripper:
    - Uses hysteresis to prevent flickering when the trigger is near the threshold.
    - Output is 0.0 for open, 1.0 for close.

    Wrist:
    - Retargets absolute pose from controller to robot frame.
    - Applies a fixed offset rotation for comfort/alignment.
    """
    """基于控制器触发输入的布尔状态的G1抓住器重定向，与重定向手腕姿势相连。

    Gripper:
    - 在触发器接近门时使用歇斯底里来防止闪。
    - 开放时是0.0，关闭时是1.0。

    Wrist:
    - 从控制器到机器人框架。
    - 适用于舒适性/调整性固定偏移旋转。
    """

    def __init__(self, cfg: G1TriHandUpperBodyMotionControllerGripperRetargeterCfg):
        """Initialize the retargeter.

        Args:
            cfg: Configuration for the retargeter.
        """
        """启动重定位器。

        参数：
            cfg: 预定重定目标。
        """
        super().__init__(cfg)
        self._cfg = cfg
        # Track previous state for hysteresis (left, right)
        self._prev_left_state: float = 0.0
        self._prev_right_state: float = 0.0

    def retarget(self, data: dict) -> torch.Tensor:
        """Retarget controller inputs to gripper boolean state and wrist pose.

        Args:
            data: Dictionary with MotionControllerTrackingTarget.LEFT/RIGHT keys
                 Each value is a 2D array: [pose(7), inputs(7)]

        Returns:
            Tensor: [left_gripper_state(1), right_gripper_state(1), left_wrist(7), right_wrist(7)]
            Wrist format: [x, y, z, qw, qx, qy, qz]
        """
        """转向控制器输入到抓住器的布尔状态和手腕姿势。

        参数：
            data: 字典MotionControllerTrackingTarget.LEFT/RIGHT每个值都是2D数组: [pose(7)，输入(7)]

        返回：
            Tensor: [left_gripper_state(1)，right_gripper_state(1)，left_wrist(7)，right_wrist(7)]
            手腕格式: [x， y， z， qw， qx， qy， qz]
        """
        # Get controller data
        left_controller_data = data.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT, np.array([]))
        right_controller_data = data.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT, np.array([]))

        # --- Gripper Logic ---
        # Extract hand state from controller data with hysteresis
        left_hand_state: float = self._extract_hand_state(left_controller_data, self._prev_left_state)
        right_hand_state: float = self._extract_hand_state(right_controller_data, self._prev_right_state)

        # Update previous states
        self._prev_left_state = left_hand_state
        self._prev_right_state = right_hand_state

        gripper_tensor = torch.tensor([left_hand_state, right_hand_state], dtype=torch.float32, device=self._sim_device)

        # --- Wrist Logic ---
        # Default wrist poses (position + quaternion [w, x, y, z] as per default_wrist init)
        # Note: default_wrist is [x, y, z, w, x, y, z] in reference, but seemingly used as [x,y,z, w,x,y,z]
        default_wrist = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # Extract poses from controller data
        left_wrist = self._extract_wrist_pose(left_controller_data, default_wrist)
        right_wrist = self._extract_wrist_pose(right_controller_data, default_wrist)

        # Convert to tensors
        left_wrist_tensor = torch.tensor(self._retarget_abs(left_wrist), dtype=torch.float32, device=self._sim_device)
        right_wrist_tensor = torch.tensor(self._retarget_abs(right_wrist), dtype=torch.float32, device=self._sim_device)

        # Concatenate: [gripper(2), left_wrist(7), right_wrist(7)]
        return torch.cat([gripper_tensor, left_wrist_tensor, right_wrist_tensor])

    def _extract_hand_state(self, controller_data: np.ndarray, prev_state: float) -> float:
        """Extract hand state from controller data with hysteresis.

        Args:
            controller_data: 2D array [pose(7), inputs(7)]
            prev_state: Previous hand state (0.0 or 1.0)

        Returns:
            Hand state as float (0.0 for open, 1.0 for close)
        """
        """通过歇斯底里取出控制器数据的手动状态。

        参数：
            controller_data: 2D阵列 [pose(7)，输入(7)]
            prev_state: 前手状态 (0.0或1.0)

        返回：
            作为浮动的手状态 (0.0为开放，1.0为关闭)
        """
        if len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            return 0.0

        # Extract inputs from second row
        inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
        if len(inputs) < len(DeviceBase.MotionControllerInputIndex):
            return 0.0

        # Extract specific inputs using enum
        trigger = inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]  # 0.0 to 1.0 (analog)

        # Apply hysteresis
        if prev_state < 0.5:  # Currently open
            return 1.0 if trigger > self._cfg.threshold_high else 0.0
        else:  # Currently closed
            return 0.0 if trigger < self._cfg.threshold_low else 1.0

    def _extract_wrist_pose(self, controller_data: np.ndarray, default_pose: np.ndarray) -> np.ndarray:
        """Extract wrist pose from controller data.

        Args:
            controller_data: 2D array [pose(7), inputs(7)]
            default_pose: Default pose to use if no data

        Returns:
            Wrist pose array [x, y, z, w, x, y, z]
        """
        """从控制器数据中提取手腕姿势。

        参数：
            controller_data: 2D阵列 [pose(7)，输入(7)]
            default_pose: 如果没有数据，则使用默认状态

        返回：
            手腕姿势阵列 [x， y， z， w， x， y， z]
        """
        if len(controller_data) > DeviceBase.MotionControllerDataRowIndex.POSE.value:
            return controller_data[DeviceBase.MotionControllerDataRowIndex.POSE.value]
        return default_pose

    def _retarget_abs(self, wrist: np.ndarray) -> np.ndarray:
        """Handle absolute pose retargeting for controller wrists."""
        """控制器手腕的绝对姿势重定向。"""
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)

        # Combined -75° (rather than -90° for wrist comfort) Y rotation + 90° Z rotation
        # This is equivalent to (0, -75, 90) in euler angles
        combined_quat = torch.tensor([0.5358, -0.4619, 0.5358, 0.4619], dtype=torch.float32)

        openxr_pose = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))
        transform_pose = PoseUtils.make_pose(torch.zeros(3), PoseUtils.matrix_from_quat(combined_quat))

        result_pose = PoseUtils.pose_in_A_to_pose_in_B(transform_pose, openxr_pose)
        pos, rot_mat = PoseUtils.unmake_pose(result_pose)
        quat = PoseUtils.quat_from_matrix(rot_mat)

        return np.concatenate([pos.numpy(), quat.numpy()])

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]


@dataclass
class G1TriHandUpperBodyMotionControllerGripperRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 boolean gripper and wrist retargeter."""
    """为G1布尔式抓住器和手腕回器的配置。"""

    threshold_high: float = 0.6  # Threshold to close hand
    threshold_low: float = 0.4  # Threshold to open hand
    retargeter_type: type[RetargeterBase] = G1TriHandUpperBodyMotionControllerGripperRetargeter
