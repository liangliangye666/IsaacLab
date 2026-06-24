# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import numpy as np
import torch


class MotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """
    """从NumPy文件格式上载和样本运动数据的辅助类。
    """

    def __init__(self, motion_file: str, device: torch.device) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        """运载一个动作文件，并初始化内部变量。

        参数：
            motion_file: 移动文件路径进行加载。
            device: 输入数据的设备。

        异常：
            AssertionError: 如果指定的运动文件不存在。
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)

        self.device = device
        self._dof_names = data["dof_names"].tolist()
        self._body_names = data["body_names"].tolist()

        self.dof_positions = torch.tensor(data["dof_positions"], dtype=torch.float32, device=self.device)
        self.dof_velocities = torch.tensor(data["dof_velocities"], dtype=torch.float32, device=self.device)
        self.body_positions = torch.tensor(data["body_positions"], dtype=torch.float32, device=self.device)
        self.body_rotations = torch.tensor(data["body_rotations"], dtype=torch.float32, device=self.device)
        self.body_linear_velocities = torch.tensor(
            data["body_linear_velocities"], dtype=torch.float32, device=self.device
        )
        self.body_angular_velocities = torch.tensor(
            data["body_angular_velocities"], dtype=torch.float32, device=self.device
        )

        self.dt = 1.0 / data["fps"]
        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.dt * (self.num_frames - 1)
        print(f"Motion loaded ({motion_file}): duration: {self.duration} sec, frames: {self.num_frames}")

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        """骨架DOF名字。"""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names."""
        """骨硬体名字。"""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        """骨的数量是DOFs。"""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        """骨的硬体数量。"""
        return len(self._body_names)

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        """连续值之间的线性插角。

        参数：
            a: 第一个值。
               形状是 (N，X) 或 (N，M，X)。
            b: 第二个值。
               形状是 (N，X) 或 (N，M，X)。
            blend: 在0 (a) 到1 (b) 间的插射系数。
            start: 索引以获得第一个值。
                   如果指定``start``和 ``end`，则第一值和第二值将是从参数``a`` (维度0) 中得到的。
            end: 索引以获取第二个值。
                 如果指定``start``和 ``end`，则第一值和第二值将是从参数``a`` (维度0) 中得到的。

        返回：
            间接值。
            形状是 (N，X) 或 (N，M，X)。
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        """连续旋转之间的回合 (球状线性回合)。

        参数：
            q0: 第一个四 (wxyz)。
                形状是 (N， 4) 或 (N， M， 4)。
            q1: 第二个四nion (wxyz)。
                形状是 (N， 4) 或 (N， M， 4)。
            blend: 在0 (q0) 和1 (q1) 间的插射系数。
            start: 索引找出第一个四元数。
                   如果指定了``start``和 ``end`，则第一个和第二个四分母将是从参数``q0`` (维度0) 中取出的。
            end: 索引取第二个四元数。
                 如果指定了``start``和 ``end`，则第一个和第二个四分母将是从参数``q0`` (维度0) 中取出的。

        返回：
            交叉的四元数。
            形状是 (N， 4) 或 (N， M， 4)。
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and motion duration, to sample motion values.
                Specified times will be clipped to fall within the range of the motion duration.

        Returns:
            First value indexes, Second value indexes, and blending time between 0 (first value) and 1 (second value).
        """
        """计算第一个和第二个值的索引，以及它们和给定的时间之间的混合时间。

        参数：
            times: 时间在0到运动时间之间，以样本运动值。
                   指定时间将被裁剪，以使其进入运动时间范围。

        返回：
            第一个值索引，第二值索引和0 (第一个值) 和1 (第二值) 之间的混合时间。
        """
        phase = np.clip(times / self.duration, 0.0, 1.0)
        index_0 = (phase * (self.num_frames - 1)).round(decimals=0).astype(int)
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = ((times - index_0 * self.dt) / self.dt).round(decimals=5)
        return index_0, index_1, blend

    def sample_times(self, num_samples: int, duration: float | None = None) -> np.ndarray:
        """Sample random motion times uniformly.

        Args:
            num_samples: Number of time samples to generate.
            duration: Maximum motion duration to sample.
                If not defined samples will be within the range of the motion duration.

        Raises:
            AssertionError: If the specified duration is longer than the motion duration.

        Returns:
            Time samples, between 0 and the specified/motion duration.
        """
        """随机运动时间均。

        参数：
            num_samples: 需要生成的时间样本数。
            duration: 检测时间:
                      如果没有定义，样本将在运动时间范围内。

        异常：
            AssertionError: 如果指定时间比运动时间长。

        返回：
            时间样本，在0到指定/运动时间之间。
        """
        duration = self.duration if duration is None else duration
        assert duration <= self.duration, (
            f"The specified duration ({duration}) is longer than the motion duration ({self.duration})"
        )
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self, num_samples: int, times: np.ndarray | None = None, duration: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            A tuple containing sampled motion data:
                - DOF positions (with shape (N, num_dofs))
                - DOF velocities (with shape (N, num_dofs))
                - Body positions (with shape (N, num_bodies, 3))
                - Body rotations (with shape (N, num_bodies, 4), as wxyz quaternion)
                - Body linear velocities (with shape (N, num_bodies, 3))
                - Body angular velocities (with shape (N, num_bodies, 3))
        """
        """运动数据样本。

        参数：
            num_samples: 需要生成的时间样本数。
                         如果定义``times``，则忽略这个参数。
            times: 用于采样的运动时间。
                   如果没有定义，运动数据将随机抽取时间均。
            duration: 检测时间:
                      如果没有定义，样本将在运动时间范围内。
                      如果定义``times``，则忽略这个参数。

        返回：
            含有采样运动数据的图普:
                - DOF位置 (形状 (N， num_dofs))
                - DOF速度 (形状 (N， num_dofs))
                - 身体位置 (形状 (N，num_bodies，3))
                - 身体旋转 (形状 (N，num_bodies，4) 作为wxyz四元数)
                - 身体线性速度 (形状 (N，num_bodies，3))
                - 体角速度 (形状 (N，num_bodies，3))
        """
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(self.dof_positions, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_positions, blend=blend, start=index_0, end=index_1),
            self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_linear_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_angular_velocities, blend=blend, start=index_0, end=index_1),
        )

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """
        """按DOFs名字进行 DOFs索引。

        参数：
            dof_names: 列出DOFs的名字。

        异常：
            AssertionError: 如果指定的DOFs名称不存在。

        返回：
            列出DOFs索引。
        """
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        """按身体名字查看骨架身体索引。

        参数：
            dof_names: 尸体名单。

        异常：
            AssertionError: 如果没有指定的尸体名称。

        返回：
            列表的索引。
        """
        indexes = []
        for name in body_names:
            assert name in self._body_names, f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = MotionLoader(args.file, "cpu")

    print("- number of frames:", motion.num_frames)
    print("- number of DOFs:", motion.num_dofs)
    print("- number of bodies:", motion.num_bodies)
