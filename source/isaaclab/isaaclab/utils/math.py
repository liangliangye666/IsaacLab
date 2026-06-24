# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for various math operations."""

# needed to import for allowing type-hinting: torch.Tensor | np.ndarray
from __future__ import annotations
"""包含各种数学操作的工具的子模块。"""

import logging
import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional

# import logger
logger = logging.getLogger(__name__)

"""
General
"""
"""一般
"""


@torch.jit.script
def scale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Normalizes a given input tensor to a range of [-1, 1].

    .. note::
        It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape is (N, dims) or (dims,).
        upper: The maximum value of the tensor. Shape is (N, dims) or (dims,).

    Returns:
        Normalized transform of the tensor. Shape is (N, dims).
    """
    """将给定的输入数正常化为 [-1， 1] 范围。

    .. 说明::
        它使用pytorch播放功能来处理批量输入。

    参数：
        x: 输入形状张量 (N，色)。
        lower: 子的最小值。
               形状是 (N，暗色) 或 (暗色)。
        upper: 子的最大值。
               形状是 (N，暗色) 或 (暗色)。

    返回：
        子的正常化转变。
        形状是 (N，色)。
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return 2 * (x - offset) / (upper - lower)


@torch.jit.script
def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """De-normalizes a given input tensor from range of [-1, 1] to (lower, upper).

    .. note::
        It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape is (N, dims) or (dims,).
        upper: The maximum value of the tensor. Shape is (N, dims) or (dims,).

    Returns:
        De-normalized transform of the tensor. Shape is (N, dims).
    """
    """从 [-1， 1] 范围到 (下，上) 范围的给定的输入子。

    .. 说明::
        它使用pytorch播放功能来处理批量输入。

    参数：
        x: 输入形状张量 (N，色)。
        lower: 子的最小值。
               形状是 (N，暗色) 或 (暗色)。
        upper: 子的最大值。
               形状是 (N，暗色) 或 (暗色)。

    返回：
        变态变化。
        形状是 (N，色)。
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return x * (upper - lower) * 0.5 + offset


@torch.jit.script
def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Clamps a given input tensor to (lower, upper).

    It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape is (N, dims) or (dims,).
        upper: The maximum value of the tensor. Shape is (N, dims) or (dims,).

    Returns:
        Clamped transform of the tensor. Shape is (N, dims).
    """
    """将给定的输入子紧缩到 (下，上)。

    它使用pytorch播放功能来处理批量输入。

    参数：
        x: 输入形状张量 (N，色)。
        lower: 子的最小值。
               形状是 (N，暗色) 或 (暗色)。
        upper: 子的最大值。
               形状是 (N，暗色) 或 (暗色)。

    返回：
        紧器的转换。
        形状是 (N，色)。
    """
    return torch.max(torch.min(x, upper), lower)


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    """将给定的输入子正常化为单位长度。

    参数：
        x: 输入形状张量 (N，色)。
        eps: 一个小值以避免零分。
             默认调整1e-9。

    返回：
        规范化形状度 (N，暗)。
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    r"""Wraps input angles (in radians) to the range :math:`[-\pi, \pi]`.

    This function wraps angles in radians to the range :math:`[-\pi, \pi]`, such that
    :math:`\pi` maps to :math:`\pi`, and :math:`-\pi` maps to :math:`-\pi`. In general,
    odd positive multiples of :math:`\pi` are mapped to :math:`\pi`, and odd negative
    multiples of :math:`\pi` are mapped to :math:`-\pi`.

    The function behaves similar to MATLAB's `wrapToPi <https://www.mathworks.com/help/map/ref/wraptopi.html>`_
    function.

    Args:
        angles: Input angles of any shape.

    Returns:
        Angles in the range :math:`[-\pi, \pi]`.
    """
    """将输入角度 (在半径中) 卷入为:数学:`[-\pi， \pi]`范围。

    这种函数将角在半径中包裹到范围:数学:`[-\pi， \pi]`，这样
    :math:运算的 `\pi` 地图:`\pi`，和运算的 `-\pi` 地图:`-\pi`。
    :math:`\pi`的奇数正倍数被映射到:math:`\pi`，和:math:`\pi`的奇数负倍数被映射到:math:`-\pi`。

    函数的行为类似于MATLAB现在`wrapToPi <https://www.mathworks.com/help/map/ref/wraptopi.html>`函数

    参数：
        angles: 任何形状的输入角。

    返回：
        在范围中的角度:数学:`[-\pi， \pi]`。
    """
    # wrap to [0, 2*pi)
    wrapped_angle = (angles + torch.pi) % (2 * torch.pi)
    # map to [-pi, pi]
    # we check for zero in wrapped angle to make it go to pi when input angle is odd multiple of pi
    return torch.where((wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi)


@torch.jit.script
def copysign(mag: float, other: torch.Tensor) -> torch.Tensor:
    """Create a new floating-point tensor with the magnitude of input and the sign of other, element-wise.

    Note:
        The implementation follows from `torch.copysign`. The function allows a scalar magnitude.

    Args:
        mag: The magnitude scalar.
        other: The tensor containing values whose signbits are applied to magnitude.

    Returns:
        The output tensor.
    """
    """创建一个新的浮点子， 输入大小和其他元素的标志。

    说明：
        实施是从`torch.copysign`中得到的。
        函数允许一个 skalar 大小。

    参数：
        mag: 这就是大小尺度。
        other: 含有值的子，其信号位应用于大小。

    返回：
        输出子。
    """
    mag_torch = abs(mag) * torch.ones_like(other)
    return torch.copysign(mag_torch, other)


"""
Rotation
"""
"""旋转
"""


@torch.jit.script
def quat_unique(q: torch.Tensor) -> torch.Tensor:
    """Convert a unit quaternion to a standard form where the real part is non-negative.

    Quaternion representations have a singularity since ``q`` and ``-q`` represent the same
    rotation. This function ensures the real part of the quaternion is non-negative.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Standardized quaternions. Shape is (..., 4).
    """
    """转换一个单元四角形到一个标准形式，其中真实部分是非负的。

    四元数表示具有单一性，因为``q``和``-q``代表相同的旋转。
    这种函数确保四ater的真实部分是非负的。

    参数：
        q: 在 (w，x，y，z) 中的四元数方向。
           形状是 (...， 4)。

    返回：
        标准化四季节。
        形状是 (...， 4)。
    """
    return torch.where(q[..., 0:1] < 0, -q, q)


@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    """转换作为四元数的旋转为旋转矩阵。

    参数：
        quaternions: 在 (w，x，y，z) 中的四元数方向。
                     形状是 (...， 4)。

    返回：
        旋转矩阵。
        形状是 (...， 3， 3)。

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversion
              s.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def convert_quat(quat: torch.Tensor | np.ndarray, to: Literal["xyzw", "wxyz"] = "xyzw") -> torch.Tensor | np.ndarray:
    """Converts quaternion from one convention to another.

    The convention to convert TO is specified as an optional argument. If to == 'xyzw',
    then the input is in 'wxyz' format, and vice-versa.

    Args:
        quat: The quaternion of shape (..., 4).
        to: Convention to convert quaternion to.. Defaults to "xyzw".

    Returns:
        The converted quaternion in specified convention.

    Raises:
        ValueError: Invalid input argument `to`, i.e. not "xyzw" or "wxyz".
        ValueError: Invalid shape of input `quat`, i.e. not (..., 4,).
    """
    """转换四元数从一个会议到另一个。

    转换TO的约定是选项参数。
    如果 == 'xyzw'，则输入是'wxyz'格式，反之亦然。

    参数：
        quat: 形状的四角形 (...， 4)。
        to: 为了将四元数转换为..。
            在"xyzw"上默认设置。

    返回：
        在规定的公约中转换的四元数。

    异常：
        ValueError: 不有效的输入参数 `to`， i.e。 不是"xyzw"或"wxyz"。
        ValueError: 输入 `quat`， i.e。 不有效的形状 (...， 4，)。
    """
    # check input is correct
    if quat.shape[-1] != 4:
        msg = f"Expected input quaternion shape mismatch: {quat.shape} != (..., 4)."
        raise ValueError(msg)
    if to not in ["xyzw", "wxyz"]:
        msg = f"Expected input argument `to` to be 'xyzw' or 'wxyz'. Received: {to}."
        raise ValueError(msg)
    # check if input is numpy array (we support this backend since some classes use numpy)
    if isinstance(quat, np.ndarray):
        # use numpy functions
        if to == "xyzw":
            # wxyz -> xyzw
            return np.roll(quat, -1, axis=-1)
        else:
            # xyzw -> wxyz
            return np.roll(quat, 1, axis=-1)
    else:
        # convert to torch (sanity check)
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=float)
        # convert to specified quaternion type
        if to == "xyzw":
            # wxyz -> xyzw
            return quat.roll(-1, dims=-1)
        else:
            # xyzw -> wxyz
            return quat.roll(1, dims=-1)


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    """计算一个四元数的结合。

    参数：
        q: 在 (w，x，y，z) 中的四元数方向。
           形状是 (...， 4)。

    返回：
        在 (w，x，y，z) 中的合并四方体。
        形状是 (...， 4)。
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Computes the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        The inverse quaternion in (w, x, y, z). Shape is (N, 4).
    """
    """计算一个四元数的逆向。

    参数：
        q: 在 (w，x，y，z) 中的四元数方向。
           形状是 (N， 4)。
        eps: 一个小值以避免零分。
             默认调整1e-9。

    返回：
        在 (w，x，y，z) 中的逆四元数。
        形状是 (N， 4)。
    """
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to Quaternions.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians). Shape is (N,).
        pitch: Rotation around y-axis (in radians). Shape is (N,).
        yaw: Rotation around z-axis (in radians). Shape is (N,).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    """转换为维勒角的旋转为四元数。

    说明：
        在XYZ公约中假设了尤勒角。

    参数：
        roll: 在 x 轴周围的旋转 (在半径中)。
              形状是 (N，)。
        pitch: 绕在 y 轴的旋转 (在半径中)。
               形状是 (N，)。
        yaw: 在z轴周围的旋转 (在半径中)。
             形状是 (N，)。

    返回：
        在 (w，x，y，z) 中的四元数。
        形状是 (N， 4)。
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qw, qx, qy, qz], dim=-1)


@torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) but with a zero sub-gradient where x is 0.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L91-L99
    """
    """返回torch.sqrt(torch.max(0，x)) 但以零次分数，其中x是0。

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversion
              s.py#L91-L99
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch.jit.script
def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: The rotation matrices. Shape is (..., 3, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (..., 4).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L102-L161
    """
    """转换作为转动矩阵的旋转为四元数。

    参数：
        matrix: 转动矩阵。
                形状是 (...， 3， 3)。

    返回：
        在 (w，x，y，z) 中的四元数。
        形状是 (...， 4)。

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversion
              s.py#L102-L161
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    return quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )


def _axis_angle_rotation(axis: Literal["X", "Y", "Z"], angle: torch.Tensor) -> torch.Tensor:
    """Return the rotation matrices for one of the rotations about an axis of which Euler angles describe,
    for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: Euler angles in radians of any shape.

    Returns:
        Rotation matrices. Shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L164-L191
    """
    """返回旋转矩阵，为一个关于尤勒角描述的轴的旋转，
    for each value of the angle given.

    参数：
        axis: 轴标签"X"或"Y"或"Z"。
        angle: 在任何形状的半径中。

    返回：
        旋转矩阵。
        形状是 (...， 3， 3)。

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversion
              s.py#L164-L191
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def matrix_from_euler(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles (intrinsic) in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians. Shape is (..., 3).
        convention: Convention string of three uppercase letters from {"X", "Y", and "Z"}.
            For example, "XYZ" means that the rotations should be applied first about x,
            then y, then z.

    Returns:
        Rotation matrices. Shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L194-L220
    """
    """转换在半径中作为尤勒角 (内在) 的旋转为旋转矩阵。

    参数：
        euler_angles: 在半径中，
                      形状是 (...， 3)。
        convention: 从{"X"， "Y"， and "Z"}起3个大字母的会议字符串。
                    例如"，XYZ"意味着轮换应该首先应用到x，然后y，然后z。

    返回：
        旋转矩阵。
        形状是 (...， 3， 3)。

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversion
              s.py#L194-L220
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [_axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler_angles, -1))]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


@torch.jit.script
def euler_xyz_from_quat(
    quat: torch.Tensor, wrap_to_2pi: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert rotations given as quaternions to Euler angles in radians.

    Note:
        The euler angles are assumed in XYZ extrinsic convention.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
        wrap_to_2pi (bool): Whether to wrap output Euler angles into [0, 2π). If
            False, angles are returned in the default range (−π, π]. Defaults to
            False.

    Returns:
        A tuple containing roll-pitch-yaw. Each element is a tensor of shape (N,).

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    """转换作为四元数的旋转为雷达人的尤勒角。

    说明：
        在XYZ外观约定中，

    参数：
        quat: 在 (w，x，y，z) 中的四元数方向。
              形状是 (N， 4)。
        wrap_to_2pi (bool): 要否将输出尤勒角卷入 [0， 2π)。
                            如果 False，则返回默认范围的角度 (−π， π)。
                            默认为 False。

    返回：
        一个含有滚球yaw的。
        每个元素都是形状张量 (N，)。

    Reference:
        https://en.wikipedia.org/wiki/转换_之间的_四角_和_艾勒_角
    """
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(torch.abs(sin_pitch) >= 1, copysign(torch.pi / 2.0, sin_pitch), torch.asin(sin_pitch))

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    if wrap_to_2pi:
        return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)
    return roll, pitch, yaw


@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    """转换作为四元数的旋转为轴/角。

    参数：
        quat: 在 (w，x，y，z) 中的四元数方向。
              形状是 (...， 4)。
        eps: 对于泰勒近似的宽容。
             在1.0e-6上默认。

    返回：
        在轴角形式中作为向量的旋转。
        形状是 (...， 3)。
        矢量的大小是绕向向的方向转向反时针的角。

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversion
              s.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


@torch.jit.script
def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as angle-axis to quaternions.

    Args:
        angle: The angle turned anti-clockwise in radians around the vector's direction. Shape is (N,).
        axis: The axis of rotation. Shape is (N, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    """转换作为角轴的旋转为四元数。

    参数：
        angle: 角度在向量方向周围的半径中反时针转向。
               形状是 (N，)。
        axis: 旋转的轴。
              形状是 (N， 3)。

    返回：
        在 (w，x，y，z) 中的四元数。
        形状是 (N， 4)。
    """
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    """乘以两个四元数。

    参数：
        q1: 在 (w，x，y，z) 中的第一个四角形。
            形状是 (...， 4)。
        q2: 在 (w，x，y，z) 中的第二个四元数。
            形状是 (...， 4)。

    返回：
        在 (w，x，y，z) 中的两个四元数的产量。
        形状是 (...， 4)。

    异常：
        ValueError: ``q1``和``q2``的输入形状不匹配。
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)


@torch.jit.script
def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    """提取一个四元数的部件。

    参数：
        quat: 在 (w，x，y，z) 中的方向。
              形状是 (...， 4)

    返回：
        一个只有的四元数。
    """
    shape = quat.shape
    quat_yaw = quat.view(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = torch.zeros_like(quat_yaw)
    quat_yaw[:, 3] = torch.sin(yaw / 2)
    quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.view(shape)


@torch.jit.script
def quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """The box-minus operator (quaternion difference) between two quaternions.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (N, 4).
        q2: The second quaternion in (w, x, y, z). Shape is (N, 4).

    Returns:
        The difference between the two quaternions. Shape is (N, 3).

    Reference:
        https://github.com/ANYbotics/kindr/blob/master/doc/cheatsheet/cheatsheet_latest.pdf
    """
    """两个四元数之间的盒子减值运算器 (四元数区别)。

    参数：
        q1: 在 (w，x，y，z) 中的第一个四角形。
            形状是 (N， 4)。
        q2: 在 (w，x，y，z) 中的第二个四元数。
            形状是 (N， 4)。

    返回：
        这两个四元数的区别。
        形状是 (N， 3)。

    Reference:
        https://github.com/ANYbotics/kindr/blob/master/doc/cheatsheet/cheatsheet_latest.pdf
    """
    quat_diff = quat_mul(q1, quat_conjugate(q2))  # q1 * q2^-1
    return axis_angle_from_quat(quat_diff)  # log(qd)


@torch.jit.script
def quat_box_plus(q: torch.Tensor, delta: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """The box-plus operator (quaternion update) to apply an increment to a quaternion.

    Args:
        q: The initial quaternion in (w, x, y, z). Shape is (N, 4).
        delta: The axis-angle perturbation. Shape is (N, 3).
            eps: A small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        The updated quaternion after applying the perturbation. Shape is (N, 4).

    Reference:
        https://github.com/ANYbotics/kindr/blob/master/doc/cheatsheet/cheatsheet_latest.pdf
    """
    """方块加算器 (四元数更新) 将增值用于四元数。

    参数：
        q: 在 (w，x，y，z) 中的初始四分之一。
           形状是 (N， 4)。
        delta: 轴角的扰乱。
               形状是 (N， 3)。
            eps: 一个小值以避免零分。
                 默认的1e-6。

    返回：
        在使用乱后更新的四元数。
        形状是 (N， 4)。

    Reference:
        https://github.com/ANYbotics/kindr/blob/master/doc/cheatsheet/cheatsheet_latest.pdf
    """
    delta_norm = torch.clamp_min(torch.linalg.norm(delta, dim=-1, keepdim=True), min=eps)
    delta_quat = quat_from_angle_axis(delta_norm.squeeze(-1), delta / delta_norm)  # exp(dq)
    new_quat = quat_mul(delta_quat, q)  # Apply perturbation
    return quat_unique(new_quat)


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    """运用四元数转向向量。

    参数：
        quat: 在 (w，x，y，z) 中的四元数。
              形状是 (...， 4)。
        vec: 在 (x，y，z) 中的向量。
             形状是 (...， 3)。

    返回：
        在 (x，y，z) 中旋转的向量。
        形状是 (...， 3)。
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    """运用反向四元数转向向量。

    参数：
        quat: 在 (w，x，y，z) 中的四元数。
              形状是 (...， 4)。
        vec: 在 (x，y，z) 中的向量。
             形状是 (...， 3)。

    返回：
        在 (x，y，z) 中旋转的向量。
        形状是 (...， 3)。
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector only around the yaw-direction.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (N, 4).
        vec: The vector in (x, y, z). Shape is (N, 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (N, 3).
    """
    """只有围绕方向旋转向量。

    参数：
        quat: 在 (w，x，y，z) 中的方向。
              形状是 (N， 4)。
        vec: 在 (x，y，z) 中的向量。
             形状是 (N， 3)。

    返回：
        在 (x，y，z) 中旋转的向量。
        形状是 (N， 3)。
    """
    quat_yaw = yaw_quat(quat)
    return quat_apply(quat_yaw, vec)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion along the last dimension of q and v.

    .. deprecated v2.1.0:
         This function will be removed in a future release in favor of the faster implementation :meth:`quat_apply`.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    """在q和v的最后一个维度上旋转一个向量，

    ..
    已过时 v2.1.0:该函数将在未来的版本中被删除以支持更快的实现:meth:`quat_apply`。

    参数：
        q: 在 (w，x，y，z) 中的四元数。
           形状是 (...， 4)。
        v: 在 (x，y，z) 中的向量。
           形状是 (...， 3)。

    返回：
        在 (x，y，z) 中旋转的向量。
        形状是 (...， 3)。
    """
    # deprecation
    logger.warning(
        "The function 'quat_rotate' will be deprecated in favor of the faster method 'quat_apply'."
        " Please use 'quat_apply' instead...."
    )
    return quat_apply(q, v)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v.

    .. deprecated v2.1.0:
         This function will be removed in a future release in favor of the faster implementation
         :meth:`quat_apply_inverse`.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    """在q和v的最后一个维度上旋转向量，

    ..
    已过时 v2.1.0:该函数将在未来的版本中被删除以支持更快的实现:meth:`quat_apply_inverse`。

    参数：
        q: 在 (w，x，y，z) 中的四元数。
           形状是 (...， 4)。
        v: 在 (x，y，z) 中的向量。
           形状是 (...， 3)。

    返回：
        在 (x，y，z) 中旋转的向量。
        形状是 (...， 3)。
    """
    logger.warning(
        "The function 'quat_rotate_inverse' will be deprecated in favor of the faster method 'quat_apply_inverse'."
        " Please use 'quat_apply_inverse' instead...."
    )
    return quat_apply_inverse(q, v)


@torch.jit.script
def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Computes the rotation difference between two quaternions.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        Angular error between input quaternions in radians.
    """
    """计算两个四元数之间的旋转差异。

    参数：
        q1: 在 (w，x，y，z) 中的第一个四角形。
            形状是 (...， 4)。
        q2: 在 (w，x，y，z) 中的第二个四元数。
            形状是 (...， 4)。

    返回：
        在射线中输入四元数之间的角错误。
    """
    axis_angle_error = quat_box_minus(q1, q2)
    return torch.norm(axis_angle_error, dim=-1)


@torch.jit.script
def skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:
    """Computes the skew-symmetric matrix of a vector.

    Args:
        vec: The input vector. Shape is (3,) or (N, 3).

    Returns:
        The skew-symmetric matrix. Shape is (1, 3, 3) or (N, 3, 3).

    Raises:
        ValueError: If input tensor is not of shape (..., 3).
    """
    """计算一个向量的偏差对称矩阵。

    参数：
        vec: 输入向量。
             形状为 (3，) 或 (N， 3)。

    返回：
        偏差对称矩阵。
        形状是 (1， 3， 3) 或 (N， 3， 3)。

    异常：
        ValueError: 如果输入子没有形状 (...， 3)。
    """
    # check input is correct
    if vec.shape[-1] != 3:
        raise ValueError(f"Expected input vector shape mismatch: {vec.shape} != (..., 3).")
    # unsqueeze the last dimension
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    # create a skew-symmetric matrix
    skew_sym_mat = torch.zeros(vec.shape[0], 3, 3, device=vec.device, dtype=vec.dtype)
    skew_sym_mat[:, 0, 1] = -vec[:, 2]
    skew_sym_mat[:, 0, 2] = vec[:, 1]
    skew_sym_mat[:, 1, 2] = -vec[:, 0]
    skew_sym_mat[:, 1, 0] = vec[:, 2]
    skew_sym_mat[:, 2, 0] = -vec[:, 1]
    skew_sym_mat[:, 2, 1] = vec[:, 0]

    return skew_sym_mat


"""
Transformations
"""
"""变化
"""


def is_identity_pose(pos: torch.tensor, rot: torch.tensor) -> bool:
    """Checks if input poses are identity transforms.

    The function checks if the input position and orientation are close to zero and
    identity respectively using L2-norm. It does NOT check the error in the orientation.

    Args:
        pos: The cartesian position. Shape is (N, 3).
        rot: The quaternion in (w, x, y, z). Shape is (N, 4).

    Returns:
        True if all the input poses result in identity transform. Otherwise, False.
    """
    """检查是否输入姿势是身份转变。

    该函数通过L2标准检查输入位置和方向是否接近零和身份。
    它会NOT检查方向的错误。

    参数：
        pos: 卡特斯人的位置。
             形状是 (N， 3)。
        rot: 在 (w，x，y，z) 中的四元数。
             形状是 (N， 4)。

    返回：
        True如果所有输入构成了身份变化。
        否则，False。
    """
    # create identity transformations
    pos_identity = torch.zeros_like(pos)
    rot_identity = torch.zeros_like(rot)
    rot_identity[..., 0] = 1
    # compare input to identity
    return torch.allclose(pos, pos_identity) and torch.allclose(rot, rot_identity)


@torch.jit.script
def combine_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t12: torch.Tensor | None = None, q12: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Combine transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{02} = T_{01} \times T_{12}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
        t12: Position of frame 2 w.r.t. frame 1. Shape is (N, 3).
            Defaults to None, in which case the position is assumed to be zero.
        q12: Quaternion orientation of frame 2 w.r.t. frame 1 in (w, x, y, z). Shape is (N, 4).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 0.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    """将两个参考框架之间的转换组合成静止框架。

    它执行以下转换操作:`T_{02} = T_{01} \times T_{12}`，
    where :数学:`T_{AB}`是从框架A到B的均质转换矩阵。

    参数：
        t01: 框架1 w.r.t的位置。
             框架 0
             形状是 (N， 3)。
        q01: 框架1 w.r.t的四元数方向。
             在 (w，x，y，z) 中的框架0
             形状是 (N， 4)。
        t12: 框架2 w.r.t的位置
             框架1
             形状是 (N， 3)。
             在 None 时的默认值，在这种情况下，假设位置为零。
        q12: 框架2 w.r.t的四元数方向。
             在 (w，x，y，z) 中的框架1
             形状是 (N， 4)。
             在 None 时的默认状态下，在这种情况下，取向被认为是身份。

    返回：
        包含2 w.r.t框架的位置和方向的图布。
        框架 0
        紧器的形状分别为 (N， 3) 和 (N， 4)。
    """
    # compute orientation
    if q12 is not None:
        q02 = quat_mul(q01, q12)
    else:
        q02 = q01
    # compute translation
    if t12 is not None:
        t02 = t01 + quat_apply(q01, t12)
    else:
        t02 = t01

    return t02, q02


def rigid_body_twist_transform(
    v0: torch.Tensor, w0: torch.Tensor, t01: torch.Tensor, q01: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Transform the linear and angular velocity of a rigid body between reference frames.

    Given the twist of 0 relative to frame 0, this function computes the twist of 1 relative to frame 1
    from the position and orientation of frame 1 relative to frame 0. The transformation follows the
    equations:

    .. math::

        w_11 = R_{10} w_00 = R_{01}^{-1} w_00
        v_11 = R_{10} v_00 + R_{10} (w_00 \times t_01) = R_{01}^{-1} (v_00 + (w_00 \times t_01))

    where

        - :math:`R_{01}` is the rotation matrix from frame 0 to frame 1 derived from quaternion :math:`q_{01}`.
        - :math:`t_{01}` is the position of frame 1 relative to frame 0 expressed in frame 0
        - :math:`w_0` is the angular velocity of 0 in frame 0
        - :math:`v_0` is the linear velocity of 0 in frame 0

    Args:
        v0: Linear velocity of 0 in frame 0. Shape is (N, 3).
        w0: Angular velocity of 0 in frame 0. Shape is (N, 3).
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).

    Returns:
        A tuple containing:
        - The transformed linear velocity in frame 1. Shape is (N, 3).
        - The transformed angular velocity in frame 1. Shape is (N, 3).
    """
    """在参考框架之间转换硬体的线性和角性速度。

    鉴于0的扭曲相对于 0 ，这个函数计算了1的扭曲相对于 1
    from the position and orientation of frame 1 relative to frame 0. The transformation follows the
    equations:

    .. math::

        w_11 = R_{10} w_00 = R_{01}^{-1} w_00
        v_11 = R_{10} v_00 + R_{10} (w_00 \times t_01) = R_{01}^{-1} (v_00 + (w_00 \times t_01))

    在哪里

        - 数学:`R_{01}`是从四元数中取出的从0到1的轮回矩阵:`q_{01}`。
        - :数学:`t_{01}`是框架1对框架0的位置，表达为框架0
        - 数学:`w_0`是 0 的角度速度在 0 框架中
        - 数学:`v_0`是 0 的线性速度

    参数：
        v0: 在 0 架子中的线性速度
            形状是 (N， 3)。
        w0: 在 0 中 0 的角速度
            形状是 (N， 3)。
        t01: 框架1 w.r.t的位置。
             框架 0
             形状是 (N， 3)。
        q01: 框架1 w.r.t的四元数方向。
             在 (w，x，y，z) 中的框架0
             形状是 (N， 4)。

    返回：
        含有:
        - 形状是 (N，3)
        - 形状是 (N， 3)。
    """
    w1 = quat_rotate_inverse(q01, w0)
    v1 = quat_rotate_inverse(q01, v0 + torch.cross(w0, t01, dim=-1))
    return v1, w1


# @torch.jit.script
def subtract_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor | None = None, q02: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Subtract transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{12} = T_{01}^{-1} \times T_{02}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
        t02: Position of frame 2 w.r.t. frame 0. Shape is (N, 3).
            Defaults to None, in which case the position is assumed to be zero.
        q02: Quaternion orientation of frame 2 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 1.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    """减去两个参考框架之间的转换成静止框架。

    它执行以下转换操作:`T_{12} = T_{01}^{-1} \times T_{02}`，
    where :数学:`T_{AB}`是从框架A到B的均质转换矩阵。

    参数：
        t01: 框架1 w.r.t的位置。
             框架 0
             形状是 (N， 3)。
        q01: 框架1 w.r.t的四元数方向。
             在 (w，x，y，z) 中的框架0
             形状是 (N， 4)。
        t02: 框架2 w.r.t的位置
             框架 0
             形状是 (N， 3)。
             在 None 时的默认值，在这种情况下，假设位置为零。
        q02: 框架2 w.r.t的四元数方向。
             在 (w，x，y，z) 中的框架0
             形状是 (N， 4)。
             在 None 时的默认状态下，在这种情况下，取向被认为是身份。

    返回：
        包含2 w.r.t框架的位置和方向的图布。
        框架1
        紧器的形状分别为 (N， 3) 和 (N， 4)。
    """
    # compute orientation
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    # compute translation
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12


# @torch.jit.script
def compute_pose_error(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor,
    q02: torch.Tensor,
    rot_error_type: Literal["quat", "axis_angle"] = "axis_angle",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the position and orientation error between source and target frames.

    Args:
        t01: Position of source frame. Shape is (N, 3).
        q01: Quaternion orientation of source frame in (w, x, y, z). Shape is (N, 4).
        t02: Position of target frame. Shape is (N, 3).
        q02: Quaternion orientation of target frame in (w, x, y, z). Shape is (N, 4).
        rot_error_type: The rotation error type to return: "quat", "axis_angle".
            Defaults to "axis_angle".

    Returns:
        A tuple containing position and orientation error. Shape of position error is (N, 3).
        Shape of orientation error depends on the value of :attr:`rot_error_type`:

        - If :attr:`rot_error_type` is "quat", the orientation error is returned
          as a quaternion. Shape is (N, 4).
        - If :attr:`rot_error_type` is "axis_angle", the orientation error is
          returned as an axis-angle vector. Shape is (N, 3).

    Raises:
        ValueError: Invalid rotation error type.
    """
    """计算源和目标框架之间的位置和方向错误。

    参数：
        t01: 源框架的位置。
             形状是 (N， 3)。
        q01: 在 (w，x，y，z) 中源框架的四元数定向。
             形状是 (N， 4)。
        t02: 目标框架的位置。
             形状是 (N， 3)。
        q02: 在 (w，x，y，z) 中目标框架的四元数定向。
             形状是 (N， 4)。
        rot_error_type: 转换错误类型: "quat"， "axis_angle"。
                        在"axis_angle"上默认。

    返回：
        包含位置和方向错误的元组。
        位置错误的形状是 (N， 3)。
        导向错误的形状取决于:attr:`rot_error_type`的值:

        - 如果:attr:`rot_error_type`是"quat"，则导向错误被返回为quaternion。
        - 如果:attr:`rot_error_type`是"axis_angle"，导向错误将作为轴角向量返回。

    异常：
        ValueError: 无效的旋转错误类型
    """
    # Compute quaternion error (i.e., difference quaternion)
    # Reference: https://personal.utdallas.edu/~sxb027100/dock/quaternion.html
    # q_current_norm = q_current * q_current_conj
    source_quat_norm = quat_mul(q01, quat_conjugate(q01))[:, 0]
    # q_current_inv = q_current_conj / q_current_norm
    source_quat_inv = quat_conjugate(q01) / source_quat_norm.unsqueeze(-1)
    # q_error = q_target * q_current_inv
    quat_error = quat_mul(q02, source_quat_inv)

    # Compute position error
    pos_error = t02 - t01

    # return error based on specified type
    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)
        return pos_error, axis_angle_error
    else:
        raise ValueError(f"Unsupported orientation error type: {rot_error_type}. Valid: 'quat', 'axis_angle'.")


@torch.jit.script
def apply_delta_pose(
    source_pos: torch.Tensor, source_rot: torch.Tensor, delta_pose: torch.Tensor, eps: float = 1.0e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies delta pose transformation on source pose.

    The first three elements of `delta_pose` are interpreted as cartesian position displacement.
    The remaining three elements of `delta_pose` are interpreted as orientation displacement
    in the angle-axis format.

    Args:
        source_pos: Position of source frame. Shape is (N, 3).
        source_rot: Quaternion orientation of source frame in (w, x, y, z). Shape is (N, 4)..
        delta_pose: Position and orientation displacements. Shape is (N, 6).
        eps: The tolerance to consider orientation displacement as zero. Defaults to 1.0e-6.

    Returns:
        A tuple containing the displaced position and orientation frames.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    """在源姿势上应用 delta姿势转换。

    `delta_pose`的前三个元素被解释为卡特式位置移动。
    在角度轴格式中，`delta_pose`的剩余三个元素被解释为方向移动。

    参数：
        source_pos: 源框架的位置。
                    形状是 (N， 3)。
        source_rot: 在 (w，x，y，z) 中源框架的四元数定向。
                    形状是 (N，4).。
        delta_pose: 位置和方向转移。
                    形状为 (N， 6)。
        eps: 视导向移动为零的宽容。
             在1.0e-6上默认。

    返回：
        包含移动位置和方向框架的图普。
        紧器的形状分别为 (N， 3) 和 (N， 4)。
    """
    # number of poses given
    num_poses = source_pos.shape[0]
    device = source_pos.device

    # interpret delta_pose[:, 0:3] as target position displacements
    target_pos = source_pos + delta_pose[:, 0:3]
    # interpret delta_pose[:, 3:6] as target rotation displacements
    rot_actions = delta_pose[:, 3:6]
    angle = torch.linalg.vector_norm(rot_actions, dim=1)
    axis = rot_actions / angle.unsqueeze(-1)
    # change from axis-angle to quat convention
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_poses, 1)
    rot_delta_quat = torch.where(
        angle.unsqueeze(-1).repeat(1, 4) > eps, quat_from_angle_axis(angle, axis), identity_quat
    )
    # TODO: Check if this is the correct order for this multiplication.
    target_rot = quat_mul(rot_delta_quat, source_rot)

    return target_pos, target_rot


# @torch.jit.script
def transform_points(
    points: torch.Tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Transform input points in a given frame to a target frame.

    This function transform points from a source frame to a target frame. The transformation is defined by the
    position :math:`t` and orientation :math:`R` of the target frame in the source frame.

    .. math::
        p_{target} = R_{target} \times p_{source} + t_{target}

    If the input `points` is a batch of points, the inputs `pos` and `quat` must be either a batch of
    positions and quaternions or a single position and quaternion. If the inputs `pos` and `quat` are
    a single position and quaternion, the same transformation is applied to all points in the batch.

    If either the inputs :attr:`pos` and :attr:`quat` are None, the corresponding transformation is not applied.

    Args:
        points: Points to transform. Shape is (N, P, 3) or (P, 3).
        pos: Position of the target frame. Shape is (N, 3) or (3,).
            Defaults to None, in which case the position is assumed to be zero.
        quat: Quaternion orientation of the target frame in (w, x, y, z). Shape is (N, 4) or (4,).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        Transformed points in the target frame. Shape is (N, P, 3) or (P, 3).

    Raises:
        ValueError: If the inputs `points` is not of shape (N, P, 3) or (P, 3).
        ValueError: If the inputs `pos` is not of shape (N, 3) or (3,).
        ValueError: If the inputs `quat` is not of shape (N, 4) or (4,).
    """
    """将给定的框架中的输入点转换为目标框架。

    这个函数将点从源框转换为目标框。
    转型由
    position :数学:`t`和方向:数学:`R`的目标框架在源框架。

    .. math::
        p_{target} = R_{target} \times p_{source} + t_{target}

    如果输入 `points` 是一个点分组，则输入 `pos` 和 `quat` 必须是位置和四元数的分组或单个位置和四元数。
    如果输入`pos`和`quat`是单个位置和四方位，则应对批次的所有点进行相同的转换。

    如果输入 :attr:`pos` 和 :attr:`quat` 是 None，则不应运行相应的转换。

    参数：
        points: 转换的点。
                形状是 (N，P，3) 或 (P，3)。
        pos: 目标框架的位置。
             形状是 (N， 3) 或 (3，)。
             在 None 时的默认值，在这种情况下，假设位置为零。
        quat: 在 (w，x，y，z) 中目标框架的四元数定向。
              形状是 (N， 4) 或 (4，)。
              在 None 时的默认状态下，在这种情况下，取向被认为是身份。

    返回：
        目标框架中的转换点。
        形状是 (N，P，3) 或 (P，3)。

    异常：
        ValueError: 如果输入 `points` 没有形状 (N，P，3) 或 (P，3)。
        ValueError: 如果输入 `pos` 没有形状 (N， 3) 或 (3，)。
        ValueError: 如果输入 `quat` 没有形状 (N， 4) 或 (4，)。
    """
    points_batch = points.clone()
    # check if inputs are batched
    is_batched = points_batch.dim() == 3
    # -- check inputs
    if points_batch.dim() == 2:
        points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
    if points_batch.dim() != 3:
        raise ValueError(f"Expected points to have dim = 2 or dim = 3: got shape {points.shape}")
    if not (pos is None or pos.dim() == 1 or pos.dim() == 2):
        raise ValueError(f"Expected pos to have dim = 1 or dim = 2: got shape {pos.shape}")
    if not (quat is None or quat.dim() == 1 or quat.dim() == 2):
        raise ValueError(f"Expected quat to have dim = 1 or dim = 2: got shape {quat.shape}")
    # -- rotation
    if quat is not None:
        # convert to batched rotation matrix
        rot_mat = matrix_from_quat(quat)
        if rot_mat.dim() == 2:
            rot_mat = rot_mat[None]  # (3, 3) -> (1, 3, 3)
        # convert points to matching batch size (N, P, 3) -> (N, 3, P)
        # and apply rotation
        points_batch = torch.matmul(rot_mat, points_batch.transpose_(1, 2))
        # (N, 3, P) -> (N, P, 3)
        points_batch = points_batch.transpose_(1, 2)
    # -- translation
    if pos is not None:
        # convert to batched translation vector
        if pos.dim() == 1:
            pos = pos[None, None, :]  # (3,) -> (1, 1, 3)
        else:
            pos = pos[:, None, :]  # (N, 3) -> (N, 1, 3)
        # apply translation
        points_batch += pos
    # -- return points in same shape as input
    if not is_batched:
        points_batch = points_batch.squeeze(0)  # (1, P, 3) -> (P, 3)

    return points_batch


"""
Projection operations.
"""
"""投影动作。
"""


@torch.jit.script
def orthogonalize_perspective_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Converts perspective depth image to orthogonal depth image.

    Perspective depth images contain distances measured from the camera's optical center.
    Meanwhile, orthogonal depth images provide the distance from the camera's image plane.
    This method uses the camera geometry to convert perspective depth to orthogonal depth image.

    The function assumes that the width and height are both greater than 1.

    Args:
        depth: The perspective depth images. Shape is (H, W) or or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        intrinsics: The camera's calibration matrix. If a single matrix is provided, the same
            calibration matrix is used across all the depth images in the batch.
            Shape is (3, 3) or (N, 3, 3).

    Returns:
        The orthogonal depth images. Shape matches the input shape of depth images.

    Raises:
        ValueError: When depth is not of shape (H, W) or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        ValueError: When intrinsics is not of shape (3, 3) or (N, 3, 3).
    """
    """将视角深度图像转换为直角深度图像。

    视角深度图像包含从相机的光学中心测量的距离。
    与此同时，直角深度图像提供了相机的图像平面距离。
    这种方法使用摄像头几何来将视角深度转换为直角深度图像。

    函数假设宽度和高度都超过1。

    参数：
        depth: 视角深度图像。
               形状是 (H， W) 或 (H， W， 1) 或 (N， H， W) 或 (N， H， W， 1)。
        intrinsics: 摄像机的校准矩阵。
                    如果提供单个矩阵，则在批量中的所有深度图像中使用相同的校准矩阵。
                    形状是 (3， 3) 或 (N， 3， 3)。

    返回：
        ort形深度图像。
        形状与深度图像的输入形状一致。

    异常：
        ValueError: 如果深度没有形状 (H， W) 或 (H， W， 1) 或 (N， H， W) 或 (N， H， W， 1)。
        ValueError: 当本质不具有形状 (3， 3) 或 (N， 3， 3)
    """
    # Clone inputs to avoid in-place modifications
    perspective_depth_batch = depth.clone()
    intrinsics_batch = intrinsics.clone()

    # Check if inputs are batched
    is_batched = perspective_depth_batch.dim() == 4 or (
        perspective_depth_batch.dim() == 3 and perspective_depth_batch.shape[-1] != 1
    )

    # Track whether the last dimension was singleton
    add_last_dim = False
    if perspective_depth_batch.dim() == 4 and perspective_depth_batch.shape[-1] == 1:
        add_last_dim = True
        perspective_depth_batch = perspective_depth_batch.squeeze(dim=3)  # (N, H, W, 1) -> (N, H, W)
    if perspective_depth_batch.dim() == 3 and perspective_depth_batch.shape[-1] == 1:
        add_last_dim = True
        perspective_depth_batch = perspective_depth_batch.squeeze(dim=2)  # (H, W, 1) -> (H, W)

    if perspective_depth_batch.dim() == 2:
        perspective_depth_batch = perspective_depth_batch[None]  # (H, W) -> (1, H, W)

    if intrinsics_batch.dim() == 2:
        intrinsics_batch = intrinsics_batch[None]  # (3, 3) -> (1, 3, 3)

    if is_batched and intrinsics_batch.shape[0] == 1:
        intrinsics_batch = intrinsics_batch.expand(perspective_depth_batch.shape[0], -1, -1)  # (1, 3, 3) -> (N, 3, 3)

    # Validate input shapes
    if perspective_depth_batch.dim() != 3:
        raise ValueError(f"Expected depth images to have 2, 3, or 4 dimensions; got {depth.shape}.")
    if intrinsics_batch.dim() != 3:
        raise ValueError(f"Expected intrinsics to have shape (3, 3) or (N, 3, 3); got {intrinsics.shape}.")

    # Image dimensions
    im_height, im_width = perspective_depth_batch.shape[1:]

    # Get the intrinsics parameters
    fx = intrinsics_batch[:, 0, 0].view(-1, 1, 1)
    fy = intrinsics_batch[:, 1, 1].view(-1, 1, 1)
    cx = intrinsics_batch[:, 0, 2].view(-1, 1, 1)
    cy = intrinsics_batch[:, 1, 2].view(-1, 1, 1)

    # Create meshgrid of pixel coordinates
    u_grid = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
    v_grid = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
    u_grid, v_grid = torch.meshgrid(u_grid, v_grid, indexing="xy")

    # Expand the grids for batch processing
    u_grid = u_grid.unsqueeze(0).expand(perspective_depth_batch.shape[0], -1, -1)
    v_grid = v_grid.unsqueeze(0).expand(perspective_depth_batch.shape[0], -1, -1)

    # Compute the squared terms for efficiency
    x_term = ((u_grid - cx) / fx) ** 2
    y_term = ((v_grid - cy) / fy) ** 2

    # Calculate the orthogonal (normal) depth
    orthogonal_depth = perspective_depth_batch / torch.sqrt(1 + x_term + y_term)

    # Restore the last dimension if it was present in the input
    if add_last_dim:
        orthogonal_depth = orthogonal_depth.unsqueeze(-1)

    # Return to original shape if input was not batched
    if not is_batched:
        orthogonal_depth = orthogonal_depth.squeeze(0)

    return orthogonal_depth


@torch.jit.script
def unproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor, is_ortho: bool = True) -> torch.Tensor:
    r"""Un-project depth image into a pointcloud.

    This function converts orthogonal or perspective depth images into points given the calibration matrix
    of the camera. It uses the following transformation based on camera geometry:

    .. math::
        p_{3D} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`p_{3D}` is the 3D point, :math:`d` is the depth value (measured from the image plane),
    :math:`u` and :math:`v` are the pixel coordinates and :math:`K` is the intrinsic matrix.

    The function assumes that the width and height are both greater than 1. This makes the function
    deal with many possible shapes of depth images and intrinsics matrices.

    .. note::
        If :attr:`is_ortho` is False, the input depth images are transformed to orthogonal depth images
        by using the :meth:`orthogonalize_perspective_depth` method.

    Args:
        depth: The depth measurement. Shape is (H, W) or or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        intrinsics: The camera's calibration matrix. If a single matrix is provided, the same
            calibration matrix is used across all the depth images in the batch.
            Shape is (3, 3) or (N, 3, 3).
        is_ortho: Whether the input depth image is orthogonal or perspective depth image. If True, the input
            depth image is considered as the *orthogonal* type, where the measurements are from the camera's
            image plane. If False, the depth image is considered as the *perspective* type, where the
            measurements are from the camera's optical center. Defaults to True.

    Returns:
        The 3D coordinates of points. Shape is (P, 3) or (N, P, 3).

    Raises:
        ValueError: When depth is not of shape (H, W) or (H, W, 1) or (N, H, W) or (N, H, W, 1).
        ValueError: When intrinsics is not of shape (3, 3) or (N, 3, 3).
    """
    """没有投影的深度图像进入点云。

    这种函数将直角或视角深度图像转换为相机校准矩阵的点。
    它使用基于相机几何学的以下变化:

    .. math::
        p_{3D} = K^{-1} \times [u, v, 1]^T \times d

    where :数学:`p_{3D}`是3D点， 数学:`d`是深度值 (从图像平面测量)，
    :math:`u`和数学:`v`是像素坐标和数学:`K`是内在的矩阵。

    函数假设宽度和高度都超过1。
    这使得函数处理许多可能的深度图像和内在矩阵。

    .. 说明::
        If :attr:`is_ortho`是False，输入深度图像转换为直角深度图像
        使用:meth:`orthogonalize_perspective_depth`方法。

    参数：
        depth: 测量深度。
               形状是 (H， W) 或 (H， W， 1) 或 (N， H， W) 或 (N， H， W， 1)。
        intrinsics: 摄像机的校准矩阵。
                    如果提供单个矩阵，则在批量中的所有深度图像中使用相同的校准矩阵。
                    形状是 (3， 3) 或 (N， 3， 3)。
        is_ortho: 如果输入深度图像是直角或视角深度图像。
                  如果True，输入深度图像将被视为*直角*类型，测量来自相机的图像平面。
                  如果False，深度图像将被视为 *视角*类型，测量来自相机的光学中心。
                  默认为 True。

    返回：
        它们的3D坐标。
        形状是 (P， 3) 或 (N， P， 3)。

    异常：
        ValueError: 如果深度没有形状 (H， W) 或 (H， W， 1) 或 (N， H， W) 或 (N， H， W， 1)。
        ValueError: 当本质不具有形状 (3， 3) 或 (N， 3， 3)
    """
    # clone inputs to avoid in-place modifications
    intrinsics_batch = intrinsics.clone()
    # convert depth image to orthogonal if needed
    if not is_ortho:
        depth_batch = orthogonalize_perspective_depth(depth, intrinsics)
    else:
        depth_batch = depth.clone()

    # check if inputs are batched
    is_batched = depth_batch.dim() == 4 or (depth_batch.dim() == 3 and depth_batch.shape[-1] != 1)
    # make sure inputs are batched
    if depth_batch.dim() == 3 and depth_batch.shape[-1] == 1:
        depth_batch = depth_batch.squeeze(dim=2)  # (H, W, 1) -> (H, W)
    if depth_batch.dim() == 2:
        depth_batch = depth_batch[None]  # (H, W) -> (1, H, W)
    if depth_batch.dim() == 4 and depth_batch.shape[-1] == 1:
        depth_batch = depth_batch.squeeze(dim=3)  # (N, H, W, 1) -> (N, H, W)
    if intrinsics_batch.dim() == 2:
        intrinsics_batch = intrinsics_batch[None]  # (3, 3) -> (1, 3, 3)
    # check shape of inputs
    if depth_batch.dim() != 3:
        raise ValueError(f"Expected depth images to have dim = 2 or 3 or 4: got shape {depth.shape}")
    if intrinsics_batch.dim() != 3:
        raise ValueError(f"Expected intrinsics to have shape (3, 3) or (N, 3, 3): got shape {intrinsics.shape}")

    # get image height and width
    im_height, im_width = depth_batch.shape[1:]
    # create image points in homogeneous coordinates (3, H x W)
    indices_u = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
    indices_v = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
    img_indices = torch.stack(torch.meshgrid([indices_u, indices_v], indexing="ij"), dim=0).reshape(2, -1)
    pixels = torch.nn.functional.pad(img_indices, (0, 0, 0, 1), mode="constant", value=1.0)
    pixels = pixels.unsqueeze(0)  # (3, H x W) -> (1, 3, H x W)

    # unproject points into 3D space
    points = torch.matmul(torch.inverse(intrinsics_batch), pixels)  # (N, 3, H x W)
    points = points / points[:, -1, :].unsqueeze(1)  # normalize by last coordinate
    # flatten depth image (N, H, W) -> (N, H x W)
    depth_batch = depth_batch.transpose_(1, 2).reshape(depth_batch.shape[0], -1).unsqueeze(2)
    depth_batch = depth_batch.expand(-1, -1, 3)
    # scale points by depth
    points_xyz = points.transpose_(1, 2) * depth_batch  # (N, H x W, 3)

    # return points in same shape as input
    if not is_batched:
        points_xyz = points_xyz.squeeze(0)

    return points_xyz


@torch.jit.script
def project_points(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    r"""Projects 3D points into 2D image plane.

    This project 3D points into a 2D image plane. The transformation is defined by the intrinsic
    matrix of the camera.

    .. math::

        \begin{align}
            p &= K \times p_{3D}  = \\
            p_{2D} &= \begin{pmatrix} u \\ v \\  d \end{pmatrix}
                    = \begin{pmatrix} p[0] / p[2] \\  p[1] / p[2] \\ Z \end{pmatrix}
        \end{align}

    where :math:`p_{2D} = (u, v, d)` is the projected 3D point, :math:`p_{3D} = (X, Y, Z)` is the
    3D point and :math:`K \in \mathbb{R}^{3 \times 3}` is the intrinsic matrix.

    If `points` is a batch of 3D points and `intrinsics` is a single intrinsic matrix, the same
    calibration matrix is applied to all points in the batch.

    Args:
        points: The 3D coordinates of points. Shape is (P, 3) or (N, P, 3).
        intrinsics: Camera's calibration matrix. Shape is (3, 3) or (N, 3, 3).

    Returns:
        Projected 3D coordinates of points. Shape is (P, 3) or (N, P, 3).
    """
    """项目3D指向2D图像平面。

    这个项目3D指向2D图像平面。
    变化是由摄像机内在矩阵定义的。

    .. math::

        \begin{align}
            p &= K \times p_{3D}  = \\
            p_{2D} &= \begin{pmatrix} u \\ v \\  d \end{pmatrix}
                    = \begin{pmatrix} p[0] / p[2] \\  p[1] / p[2] \\ Z \end{pmatrix}
        \end{align}

    where :数学:`p_{2D} = (u， v， d)`是预测的3D点:`p_{3D} = (X， Y， Z)`是
    3D点和数学:`K \in \mathbb{R}^{3 \times 3}`是内在矩阵。

    如果`points`是一个3D点的批量，而`intrinsics`是一个单一的内在矩阵，则应对该批量的所有点应用相同的校准矩阵。

    参数：
        points: 它们的3D坐标。
                形状是 (P， 3) 或 (N， P， 3)。
        intrinsics: 摄像头的校准矩阵。
                    形状是 (3， 3) 或 (N， 3， 3)。

    返回：
        预测点的3D坐标。
        形状是 (P， 3) 或 (N， P， 3)。
    """
    # clone the inputs to avoid in-place operations modifying the original data
    points_batch = points.clone()
    intrinsics_batch = intrinsics.clone()

    # check if inputs are batched
    is_batched = points_batch.dim() == 2
    # make sure inputs are batched
    if points_batch.dim() == 2:
        points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
    if intrinsics_batch.dim() == 2:
        intrinsics_batch = intrinsics_batch[None]  # (3, 3) -> (1, 3, 3)
    # check shape of inputs
    if points_batch.dim() != 3:
        raise ValueError(f"Expected points to have dim = 3: got shape {points.shape}.")
    if intrinsics_batch.dim() != 3:
        raise ValueError(f"Expected intrinsics to have shape (3, 3) or (N, 3, 3): got shape {intrinsics.shape}.")

    # project points into 2D image plane
    points_2d = torch.matmul(intrinsics_batch, points_batch.transpose(1, 2))
    points_2d = points_2d / points_2d[:, -1, :].unsqueeze(1)  # normalize by last coordinate
    points_2d = points_2d.transpose_(1, 2)  # (N, 3, P) -> (N, P, 3)
    # replace last coordinate with depth
    points_2d[:, :, -1] = points_batch[:, :, -1]

    # return points in same shape as input
    if not is_batched:
        points_2d = points_2d.squeeze(0)  # (1, 3, P) -> (3, P)

    return points_2d


"""
Sampling
"""
"""采样
"""


@torch.jit.script
def default_orientation(num: int, device: str) -> torch.Tensor:
    """Returns identity rotation transform.

    Args:
        num: The number of rotations to sample.
        device: Device to create tensor on.

    Returns:
        Identity quaternion in (w, x, y, z). Shape is (num, 4).
    """
    """返回身份转换。

    参数：
        num: 取样的旋转数量
        device: 电脑的电压。

    返回：
        在 (w，x，y，z) 中的身份四方体。
        形状是 (第4号)。
    """
    quat = torch.zeros((num, 4), dtype=torch.float, device=device)
    quat[..., 0] = 1.0

    return quat


@torch.jit.script
def random_orientation(num: int, device: str) -> torch.Tensor:
    """Returns sampled rotation in 3D as quaternion.

    Args:
        num: The number of rotations to sample.
        device: Device to create tensor on.

    Returns:
        Sampled quaternion in (w, x, y, z). Shape is (num, 4).

    Reference:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html
    """
    """返回样本中的3D旋转作为四元数。

    参数：
        num: 取样的旋转数量
        device: 电脑的电压。

    返回：
        在 (w， x， y， z) 中采样四角形。
        形状是 (第4号)。

    Reference:
        https://docs.scipy.org/doc/scipy/引用/生成/scipy.spatial.transform.Rotation.random.html
    """
    # sample random orientation from normal distribution
    quat = torch.randn((num, 4), dtype=torch.float, device=device)
    # normalize the quaternion
    return torch.nn.functional.normalize(quat, p=2.0, dim=-1, eps=1e-12)


@torch.jit.script
def random_yaw_orientation(num: int, device: str) -> torch.Tensor:
    """Returns sampled rotation around z-axis.

    Args:
        num: The number of rotations to sample.
        device: Device to create tensor on.

    Returns:
        Sampled quaternion in (w, x, y, z). Shape is (num, 4).
    """
    """返回在z轴周围的样本旋转。

    参数：
        num: 取样的旋转数量
        device: 电脑的电压。

    返回：
        在 (w， x， y， z) 中采样四角形。
        形状是 (第4号)。
    """
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = 2 * torch.pi * torch.rand(num, dtype=torch.float, device=device)

    return quat_from_euler_xyz(roll, pitch, yaw)


def sample_triangle(lower: float, upper: float, size: int | tuple[int, ...], device: str) -> torch.Tensor:
    """Randomly samples tensor from a triangular distribution.

    Args:
        lower: The lower range of the sampled tensor.
        upper: The upper range of the sampled tensor.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor. Shape is based on :attr:`size`.
    """
    """随机抽取来自三角分布的子。

    参数：
        lower: 采样子的较低范围。
        upper: 采样子的上层范围。
        size: 子的形状。
        device: 电脑的电压。

    返回：
        抽取了子。
        形状是基于:attr:`size`的。
    """
    # convert to tuple
    if isinstance(size, int):
        size = (size,)
    # create random tensor in the range [-1, 1]
    r = 2 * torch.rand(*size, device=device) - 1
    # convert to triangular distribution
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    # rescale back to [0, 1]
    r = (r + 1.0) / 2.0
    # rescale to range [lower, upper]
    return (upper - lower) * r + lower


def sample_uniform(
    lower: torch.Tensor | float, upper: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    """Sample uniformly within a range.

    Args:
        lower: Lower bound of uniform range.
        upper: Upper bound of uniform range.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor. Shape is based on :attr:`size`.
    """
    """在范围内均的样本。

    参数：
        lower: 统一范围的下边界。
        upper: 统一范围的上限。
        size: 子的形状。
        device: 电脑的电压。

    返回：
        抽取了子。
        形状是基于:attr:`size`的。
    """
    # convert to tuple
    if isinstance(size, int):
        size = (size,)
    # return tensor
    return torch.rand(*size, device=device) * (upper - lower) + lower


def sample_log_uniform(
    lower: torch.Tensor | float, upper: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    r"""Sample using log-uniform distribution within a range.

    The log-uniform distribution is defined as a uniform distribution in the log-space. It
    is useful for sampling values that span several orders of magnitude. The sampled values
    are uniformly distributed in the log-space and then exponentiated to get the final values.

    .. math::

        x = \exp(\text{uniform}(\log(\text{lower}), \log(\text{upper})))

    Args:
        lower: Lower bound of uniform range.
        upper: Upper bound of uniform range.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor. Shape is based on :attr:`size`.
    """
    """采样使用在范围内的日志均分布。

    记载均分布定义为记载空间中的均分布。
    它对于跨越数个大小顺序的样本取值是有用的。
    在日记空间中，采样值均分布，然后指数化以获得最终值。

    .. math::

        x = \exp(\text{uniform}(\log(\text{lower}), \log(\text{upper})))

    参数：
        lower: 统一范围的下边界。
        upper: 统一范围的上限。
        size: 子的形状。
        device: 电脑的电压。

    返回：
        抽取了子。
        形状是基于:attr:`size`的。
    """
    # cast to tensor if not already
    if not isinstance(lower, torch.Tensor):
        lower = torch.tensor(lower, dtype=torch.float, device=device)
    if not isinstance(upper, torch.Tensor):
        upper = torch.tensor(upper, dtype=torch.float, device=device)
    # sample in log-space and exponentiate
    return torch.exp(sample_uniform(torch.log(lower), torch.log(upper), size, device))


def sample_gaussian(
    mean: torch.Tensor | float, std: torch.Tensor | float, size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    """Sample using gaussian distribution.

    Args:
        mean: Mean of the gaussian.
        std: Std of the gaussian.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor.
    """
    """使用高斯分布的样本。

    参数：
        mean: 这就是高斯人。
        std: 斯人。
        size: 子的形状。
        device: 电脑的电压。

    返回：
        抽取了子。
    """
    if isinstance(mean, float):
        if isinstance(size, int):
            size = (size,)
        return torch.normal(mean=mean, std=std, size=size).to(device=device)
    else:
        return torch.normal(mean=mean, std=std).to(device=device)


def sample_cylinder(
    radius: float, h_range: tuple[float, float], size: int | tuple[int, ...], device: str
) -> torch.Tensor:
    """Sample 3D points uniformly on a cylinder's surface.

    The cylinder is centered at the origin and aligned with the z-axis. The height of the cylinder is
    sampled uniformly from the range :obj:`h_range`, while the radius is fixed to :obj:`radius`.

    The sampled points are returned as a tensor of shape :obj:`(*size, 3)`, i.e. the last dimension
    contains the x, y, and z coordinates of the sampled points.

    Args:
        radius: The radius of the cylinder.
        h_range: The minimum and maximum height of the cylinder.
        size: The shape of the tensor.
        device: Device to create tensor on.

    Returns:
        Sampled tensor. Shape is :obj:`(*size, 3)`.
    """
    """在面上均的3D点样本。

    圆柱以原点为中心，并与z轴一致。
    cyl的高度从:obj:`h_range`范围均抽样，而半径固定在:obj:`radius`。

    返回样本点为 :obj:`(*size， 3)`， i.e.的形状，最后一个维度包含样本点的x，y和z坐标。

    参数：
        radius: 圆柱的半径。
        h_range: minimum筒的最小和最大高度。
        size: 子的形状。
        device: 电脑的电压。

    返回：
        抽取了子。
        形状是:obj:`(*size， 3)`。
    """
    # sample angles
    angles = (torch.rand(size, device=device) * 2 - 1) * torch.pi
    h_min, h_max = h_range
    # add shape
    if isinstance(size, int):
        size = (size, 3)
    else:
        size += (3,)
    # allocate a tensor
    xyz = torch.zeros(size, device=device)
    xyz[..., 0] = radius * torch.cos(angles)
    xyz[..., 1] = radius * torch.sin(angles)
    xyz[..., 2].uniform_(h_min, h_max)
    # return positions
    return xyz


"""
Orientation Conversions
"""
"""方向转变
"""


def convert_camera_frame_orientation_convention(
    orientation: torch.Tensor,
    origin: Literal["opengl", "ros", "world"] = "opengl",
    target: Literal["opengl", "ros", "world"] = "ros",
) -> torch.Tensor:
    r"""Converts a quaternion representing a rotation from one convention to another.

    In USD, the camera follows the ``"opengl"`` convention. Thus, it is always in **Y up** convention.
    This means that the camera is looking down the -Z axis with the +Y axis pointing up , and +X axis pointing right.
    However, in ROS, the camera is looking down the +Z axis with the +Y axis pointing down, and +X axis pointing right.
    Thus, the camera needs to be rotated by :math:`180^{\circ}` around the X axis to follow the ROS convention.

    .. math::

        T_{ROS} =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1
            \end{bmatrix} T_{USD}

    On the other hand, the typical world coordinate system is with +X pointing forward, +Y pointing left,
    and +Z pointing up. The camera can also be set in this convention by rotating the camera by :math:`90^{\circ}`
    around the X axis and :math:`-90^{\circ}` around the Y axis.

    .. math::

        T_{WORLD} =
            \begin{bmatrix}
                0 & 0 & -1 & 0 \\ -1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1
            \end{bmatrix} T_{USD}

    Thus, based on their application, cameras follow different conventions for their orientation. This function
    converts a quaternion from one convention to another.

    Possible conventions are:

    - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
    - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
    - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

    Args:
        orientation: Quaternion of form `(w, x, y, z)` with shape (..., 4) in source convention.
        origin: Convention to convert from. Defaults to "opengl".
        target: Convention to convert to. Defaults to "ros".

    Returns:
        Quaternion of form `(w, x, y, z)` with shape (..., 4) in target convention
    """
    """转换一个四元数，代表从一个会议转换到另一个会议。

    在USD中，相机遵循``"opengl"``规则。
    因此，它总是在**Y up**会议。
    这意味着摄像头正在向下看 -Z轴， +Y轴向上， +X轴向右。
    然而，在ROS中，摄像头看着+Z轴， +Y轴向下， +X轴向右。
    因此，摄像头需要通过:math:`180^{\circ}`绕X轴旋转以遵循ROS规则。

    .. math::

        T_{ROS} =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \ 0 & -1 & 0 & 0 \ 0 & 0 & -1 & 0 \ 0 & 0 & 0 & 1
            \end{bmatrix} T_{USD}

    另一方面，典型的世界坐标系统是+X向前，+Y向左，+Z向上。
    通过通过:`90^{\circ}`在 X 轴周围:`-90^{\circ}`在 Y 轴周围。

    .. math::

        T_{WORLD} =
            \begin{bmatrix}
                0 & 0 & -1 & 0 \ -1 & 0 & 0 & 0 \ 0 & 1 & 0 & 0 \ 0 & 0 & 0 & 1
            \end{bmatrix} T_{USD}

    因此，根据它们的应用，摄像头遵循不同的定向规范。
    这种函数将一个四元数从一个会议转换到另一个。

    可能的会议是:

    - 在OpenGL (Usd.Camera) 公约中，应用:obj:`"opengl"` - 前轴: -Z - 上轴 +Y - 抵消
    - :obj:`"ros"`- 前向轴: +Z - 上向轴 -YROS公约
    - 在"世界框架"公约中，应用:obj:`"world"` - 前轴:+X - 上轴 +Z - 偏移

    参数：
        orientation: 形状`(w， x， y， z)`的四角形，形状 (...， 4) 在源合约中。
        origin: 让我们转换。
                默认的"开放"
        target: 会议要转换。
                默认的"ros"。

    返回：
        形状`(w， x， y， z)`的四角形，形状 (...， 4) 在目标合约中
    """
    if target == origin:
        return orientation.clone()

    # -- unify input type
    if origin == "ros":
        # convert from ros to opengl convention
        rotm = matrix_from_quat(orientation)
        rotm[:, :, 2] = -rotm[:, :, 2]
        rotm[:, :, 1] = -rotm[:, :, 1]
        # convert to opengl convention
        quat_gl = quat_from_matrix(rotm)
    elif origin == "world":
        # convert from world (x forward and z up) to opengl convention
        rotm = matrix_from_quat(orientation)
        rotm = torch.matmul(
            rotm,
            matrix_from_euler(torch.tensor([math.pi / 2, -math.pi / 2, 0], device=orientation.device), "XYZ"),
        )
        # convert to isaac-sim convention
        quat_gl = quat_from_matrix(rotm)
    else:
        quat_gl = orientation

    # -- convert to target convention
    if target == "ros":
        # convert from opengl to ros convention
        rotm = matrix_from_quat(quat_gl)
        rotm[:, :, 2] = -rotm[:, :, 2]
        rotm[:, :, 1] = -rotm[:, :, 1]
        return quat_from_matrix(rotm)
    elif target == "world":
        # convert from opengl to world (x forward and z up) convention
        rotm = matrix_from_quat(quat_gl)
        rotm = torch.matmul(
            rotm,
            matrix_from_euler(torch.tensor([math.pi / 2, -math.pi / 2, 0], device=orientation.device), "XYZ").T,
        )
        return quat_from_matrix(rotm)
    else:
        return quat_gl.clone()


def create_rotation_matrix_from_view(
    eyes: torch.Tensor,
    targets: torch.Tensor,
    up_axis: Literal["Y", "Z"] = "Z",
    device: str = "cpu",
) -> torch.Tensor:
    """Compute the rotation matrix from world to view coordinates.

    This function takes a vector ''eyes'' which specifies the location
    of the camera in world coordinates and the vector ''targets'' which
    indicate the position of the object.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

        The inputs eyes and targets can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)

    Args:
        eyes: Position of the camera in world coordinates.
        targets: Position of the object in world coordinates.
        up_axis: The up axis of the camera. Defaults to "Z".
        device: The device to create torch tensors on. Defaults to "cpu".

    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (N, 3, 3) batched rotation matrices

    Reference:
    Based on PyTorch3D (https://github.com/facebookresearch/pytorch3d/blob/eaf0709d6af0025fe94d1ee7cec454bc3054826a/pytorch3d/renderer/cameras.py#L1635-L1685)
    """
    """计算来自世界的旋转矩阵，以查看坐标。

    这种函数采用"眼睛"向量，指明相机在世界坐标中的位置，以及"目标"向量，指明对象的位置。
    输出是一个转换矩阵
    from world coordinates -> view coordinates.

        输入眼睛和目标都可以
        - 3个元素tuple/列表
        - 形状的火 (1， 3)
        - 形状的火 (N，3)

    参数：
        eyes: 摄像机在世界坐标中的位置。
        targets: 在世界坐标中对象的位置。
        up_axis: 摄像头的上轴。
                 默认的"Z"。
        device: 发射火的装置。
                默认的"CPU"。

    这些向量相互传播，所以它们都有形状 (N， 3)。

    返回：
        R: (N， 3， 3) 批量旋转矩阵

    Reference:
    基于PyTorch3D (https://github.com/facebookresearch/pytorch3d/blob/eaf0709d6af0025fe94d1ee7cec454bc3054
    826a/pytorch3d/renderer/cameras.py#L1635-L1685)
    """
    if up_axis == "Y":
        up_axis_vec = torch.tensor((0, 1, 0), device=device, dtype=torch.float32).repeat(eyes.shape[0], 1)
    elif up_axis == "Z":
        up_axis_vec = torch.tensor((0, 0, 1), device=device, dtype=torch.float32).repeat(eyes.shape[0], 1)
    else:
        raise ValueError(f"Invalid up axis: {up_axis}. Valid options are 'Y' and 'Z'.")

    # get rotation matrix in opengl format (-Z forward, +Y up)
    z_axis = -torch.nn.functional.normalize(targets - eyes, eps=1e-5)
    x_axis = torch.nn.functional.normalize(torch.cross(up_axis_vec, z_axis, dim=1), eps=1e-5)
    y_axis = torch.nn.functional.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(dim=1, keepdim=True)
    if is_close.any():
        replacement = torch.nn.functional.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)


def make_pose(pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Creates transformation matrices from positions and rotation matrices.

    Args:
        pos: Batch of position vectors with last dimension of 3.
        rot: Batch of rotation matrices with last 2 dimensions of (3, 3).

    Returns:
        Batch of pose matrices with last 2 dimensions of (4, 4).
    """
    """从位置和旋转矩阵创建转换矩阵。

    参数：
        pos: 位置向量组，最后的维度为3。
        rot: 轮换矩阵的批量，最后的2个尺寸为 (3， 3)。

    返回：
        有 (4，4) 的最后2个维度的姿势矩阵组。
    """
    assert isinstance(pos, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(rot, torch.Tensor), "Input must be a torch tensor"
    assert pos.shape[:-1] == rot.shape[:-2]
    assert pos.shape[-1] == rot.shape[-2] == rot.shape[-1] == 3
    pose = torch.zeros(pos.shape[:-1] + (4, 4), dtype=pos.dtype, device=pos.device)
    pose[..., :3, :3] = rot
    pose[..., :3, 3] = pos
    pose[..., 3, 3] = 1.0
    return pose


def unmake_pose(pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Splits transformation matrices into positions and rotation matrices.

    Args:
        pose: Batch of pose matrices with last 2 dimensions of (4, 4).

    Returns:
        Tuple containing:
            - Batch of position vectors with last dimension of 3.
            - Batch of rotation matrices with last 2 dimensions of (3, 3).
    """
    """分开转换矩阵成位置和旋转矩阵。

    参数：
        pose: 有 (4，4) 的最后2个维度的姿势矩阵组。

    返回：
        含有:
            - 位置向量组，最后的维度为3。
            - 轮换矩阵的批量，最后的2个尺寸为 (3， 3)。
    """
    assert isinstance(pose, torch.Tensor), "Input must be a torch tensor"
    return pose[..., :3, 3], pose[..., :3, :3]


def pose_inv(pose: torch.Tensor) -> torch.Tensor:
    """Computes the inverse of transformation matrices.

    The inverse of a pose matrix [R t; 0 1] is [R.T -R.T*t; 0 1].

    Args:
        pose: Batch of pose matrices with last 2 dimensions of (4, 4).

    Returns:
        Batch of inverse pose matrices with last 2 dimensions of (4, 4).
    """
    """计算转换矩阵的逆向。

    一个姿势矩阵的逆向 [R t； 0 1] 是 [R.T -R.T*t； 0 1]。

    参数：
        pose: 有 (4，4) 的最后2个维度的姿势矩阵组。

    返回：
        逆姿矩阵的批量，最后的2个维度为 (4， 4)。
    """
    assert isinstance(pose, torch.Tensor), "Input must be a torch tensor"
    num_axes = len(pose.shape)
    assert num_axes >= 2

    inv_pose = torch.zeros_like(pose)

    # Take transpose of last 2 dimensions
    inv_pose[..., :3, :3] = pose[..., :3, :3].transpose(-1, -2)

    # note: PyTorch matmul wants shapes [..., 3, 3] x [..., 3, 1] -> [..., 3, 1]
    # so we add a dimension and take it away after
    inv_pose[..., :3, 3] = torch.matmul(-inv_pose[..., :3, :3], pose[..., :3, 3:4])[..., 0]
    inv_pose[..., 3, 3] = 1.0
    return inv_pose


def pose_in_A_to_pose_in_B(pose_in_A: torch.Tensor, pose_A_in_B: torch.Tensor) -> torch.Tensor:
    """Converts poses from one coordinate frame to another.

    Transforms matrices representing point C in frame A
    to matrices representing the same point C in frame B.

    Example usage:

    frame_C_in_B = pose_in_A_to_pose_in_B(frame_C_in_A, frame_A_in_B)

    Args:
        pose_in_A: Batch of transformation matrices of point C in frame A.
        pose_A_in_B: Batch of transformation matrices of frame A in frame B.

    Returns:
        Batch of transformation matrices of point C in frame B.
    """
    """转换从一个坐标框架到另一个。

    转换图形 A 中点 C 的矩阵为图形 B 中点 C 的矩阵。

    例如使用:

    frame_C_in_B = pose_in_A_to_pose_in_B(frame_C_in_A, frame_A_in_B)

    参数：
        pose_in_A: 在格A中的C点转换矩阵的批量
        pose_A_in_B: 框架 A 在框架 B 中的转换矩阵组

    返回：
        在图B中的点C的转换矩阵批量
    """
    assert isinstance(pose_in_A, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(pose_A_in_B, torch.Tensor), "Input must be a torch tensor"
    return torch.matmul(pose_A_in_B, pose_in_A)


def quat_slerp(q1: torch.Tensor, q2: torch.Tensor, tau: float) -> torch.Tensor:
    """Performs spherical linear interpolation (SLERP) between two quaternions.

    This function does not support batch processing.

    Args:
        q1: First quaternion in (w, x, y, z) format.
        q2: Second quaternion in (w, x, y, z) format.
        tau: Interpolation coefficient between 0 (q1) and 1 (q2).

    Returns:
        Interpolated quaternion in (w, x, y, z) format.
    """
    """在两个四元数之间执行圆形线性插射 (SLERP)。

    该函数不支持批量加工。

    参数：
        q1: 在 (w，x，y，z) 格式中，
        q2: 在 (w，x，y，z) 格式中进行第二个四元数。
        tau: 在0 (q1) 和1 (q2) 间的插射系数。

    返回：
        在 (w， x， y， z) 格式中回合的四元数。
    """
    assert isinstance(q1, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(q2, torch.Tensor), "Input must be a torch tensor"
    if tau == 0.0:
        return q1
    elif tau == 1.0:
        return q2
    d = torch.dot(q1, q2)
    if abs(abs(d) - 1.0) < torch.finfo(q1.dtype).eps * 4.0:
        return q1
    if d < 0.0:
        # Invert rotation
        d = -d
        q2 *= -1.0
    angle = torch.acos(torch.clamp(d, -1, 1))
    if abs(angle) < torch.finfo(q1.dtype).eps * 4.0:
        return q1
    isin = 1.0 / torch.sin(angle)
    q1 = q1 * torch.sin((1.0 - tau) * angle) * isin
    q2 = q2 * torch.sin(tau * angle) * isin
    q1 = q1 + q2
    return q1


def interpolate_rotations(R1: torch.Tensor, R2: torch.Tensor, num_steps: int, axis_angle: bool = True) -> torch.Tensor:
    """Interpolates between two rotation matrices.

    Args:
        R1: First rotation matrix. (4x4).
        R2: Second rotation matrix. (4x4).
        num_steps: Number of desired interpolated rotations (excluding start and end).
        axis_angle: If True, interpolate in axis-angle representation;
                   otherwise use slerp. Defaults to True.

    Returns:
        Stack of interpolated rotation matrices of shape (num_steps + 1, 4, 4),
        including the start and end rotations.
    """
    """在两个旋转矩阵之间进行回合。

    参数：
        R1: 第一个旋转矩阵。
            (4x4)
        R2: 第二旋转矩阵。
            (4x4)
        num_steps: 预期的间断旋转数量 (不包括开始和结束)。
        axis_angle: 如果 True，则在轴角表示中插入；否则使用slerp。
                    默认为 True。

    返回：
        交叉旋转矩阵的形状 (num_steps + 1， 4， 4)，包括开始和结束旋转。
    """
    assert isinstance(R1, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(R2, torch.Tensor), "Input must be a torch tensor"
    if axis_angle:
        # Delta rotation expressed as axis-angle
        delta_rot_mat = torch.matmul(R2, R1.transpose(-1, -2))
        delta_quat = quat_from_matrix(delta_rot_mat)
        delta_axis_angle = axis_angle_from_quat(delta_quat)

        # Grab angle
        delta_angle = torch.linalg.norm(delta_axis_angle)

        # Fix the axis, and chunk the angle up into steps
        rot_step_size = delta_angle / num_steps

        # Convert into delta rotation matrices, and then convert to absolute rotations
        if delta_angle < 0.05:
            # Small angle - don't bother with interpolation
            rot_steps = torch.stack([R2 for _ in range(num_steps)])
        else:
            # Make sure that axis is a unit vector
            delta_axis = delta_axis_angle / delta_angle
            delta_rot_steps = [
                matrix_from_quat(quat_from_angle_axis(i * rot_step_size, delta_axis)) for i in range(num_steps)
            ]
            rot_steps = torch.stack([torch.matmul(delta_rot_steps[i], R1) for i in range(num_steps)])
    else:
        q1 = quat_from_matrix(R1)
        q2 = quat_from_matrix(R2)
        rot_steps = torch.stack(
            [matrix_from_quat(quat_slerp(q1, q2, tau=float(i) / num_steps)) for i in range(num_steps)]
        )

    # Add in endpoint
    rot_steps = torch.cat([rot_steps, R2[None]], dim=0)

    return rot_steps


def interpolate_poses(
    pose_1: torch.Tensor,
    pose_2: torch.Tensor,
    num_steps: int = None,
    step_size: float = None,
    perturb: bool = False,
) -> tuple[torch.Tensor, int]:
    """Performs linear interpolation between two poses.

    Args:
        pose_1: 4x4 start pose.
        pose_2: 4x4 end pose.
        num_steps: If provided, specifies the number of desired interpolated points.
                  Passing 0 corresponds to no interpolation. If None, step_size must be provided.
        step_size: If provided, determines number of steps based on distance between poses.
        perturb: If True, randomly perturbs interpolated position points.

    Returns:
        Tuple containing:
            - Array of shape (N + 2, 4, 4) corresponding to the interpolated pose path.
            - Number of interpolated points (N) in the path.
    """
    """执行两种姿势之间的线性回合。

    参数：
        pose_1: 4x4开始姿势。
        pose_2: 4x4终端姿势。
        num_steps: 如果提供，指定所需的插点数量。
                   通过0对应无回合。
                   如果None，必须提供step_size。
        step_size: 如果提供，根据姿势之间的距离确定步骤数。
        perturb: 如果True，随机扰乱插入位置点。

    返回：
        含有:
            - 形状阵列 (N + 2， 4， 4) 与回合的姿势路径相应。
            - 路径中插入点数 (N)。
    """
    assert isinstance(pose_1, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(pose_2, torch.Tensor), "Input must be a torch tensor"
    assert step_size is None or num_steps is None

    pos1, rot1 = unmake_pose(pose_1)
    pos2, rot2 = unmake_pose(pose_2)

    if num_steps == 0:
        # Skip interpolation
        return (
            torch.cat([pos1[None], pos2[None]], dim=0),
            torch.cat([rot1[None], rot2[None]], dim=0),
            num_steps,
        )

    delta_pos = pos2 - pos1
    if num_steps is None:
        assert torch.norm(delta_pos) > 0
        num_steps = math.ceil(torch.norm(delta_pos) / step_size)

    num_steps += 1  # Include starting pose
    assert num_steps >= 2

    # Linear interpolation of positions
    pos_step_size = delta_pos / num_steps
    grid = torch.arange(num_steps, dtype=torch.float32)
    if perturb:
        # Move interpolation grid points by up to half-size forward or backward
        perturbations = torch.rand(num_steps - 2) - 0.5
        grid[1:-1] += perturbations
    pos_steps = torch.stack([pos1 + grid[i] * pos_step_size for i in range(num_steps)])

    # Add in endpoint
    pos_steps = torch.cat([pos_steps, pos2[None]], dim=0)

    # Interpolate rotations
    rot_steps = interpolate_rotations(R1=rot1, R2=rot2, num_steps=num_steps, axis_angle=True)

    pose_steps = make_pose(pos_steps, rot_steps)
    return pose_steps, num_steps - 1


def transform_poses_from_frame_A_to_frame_B(
    src_poses: torch.Tensor, frame_A: torch.Tensor, frame_B: torch.Tensor
) -> torch.Tensor:
    """Transforms poses from one coordinate frame to another preserving relative poses.

    Args:
        src_poses: Input pose sequence (shape [T, 4, 4]) from source demonstration.
        frame_A: 4x4 frame A pose.
        frame_B: 4x4 frame B pose.

    Returns:
        Transformed pose sequence of shape [T, 4, 4].
    """
    """转换从一个坐标框架到另一个保存相对姿势。

    参数：
        src_poses: 输入姿势序列 (形状 [T， 4， 4]) 来自源示范。
        frame_A: 4x4框架A姿势。
        frame_B: 4x4框架B姿势。

    返回：
        转型姿势序列的形状 [T， 4， 4]。
    """
    # Transform source end effector poses to be relative to source object frame
    src_poses_rel_frame_B = pose_in_A_to_pose_in_B(
        pose_in_A=src_poses,
        pose_A_in_B=pose_inv(frame_B[None]),
    )

    # Apply relative poses to current object frame to obtain new target eef poses
    transformed_poses = pose_in_A_to_pose_in_B(
        pose_in_A=src_poses_rel_frame_B,
        pose_A_in_B=frame_A[None],
    )
    return transformed_poses


def generate_random_rotation(rot_boundary: float = (2 * math.pi)) -> torch.Tensor:
    """Generates a random rotation matrix using Euler angles.

    Args:
        rot_boundary: Range for random rotation angles around each axis (x, y, z).

    Returns:
        3x3 rotation matrix.
    """
    """使用尤勒角度生成一个随机旋转矩阵。

    参数：
        rot_boundary: 每个轴 (x，y，z) 周围随机旋转角的范围。

    返回：
        3x3旋转矩阵。
    """
    angles = torch.rand(3) * rot_boundary
    Rx = torch.tensor(
        [[1, 0, 0], [0, torch.cos(angles[0]), -torch.sin(angles[0])], [0, torch.sin(angles[0]), torch.cos(angles[0])]]
    )

    Ry = torch.tensor(
        [[torch.cos(angles[1]), 0, torch.sin(angles[1])], [0, 1, 0], [-torch.sin(angles[1]), 0, torch.cos(angles[1])]]
    )

    Rz = torch.tensor(
        [[torch.cos(angles[2]), -torch.sin(angles[2]), 0], [torch.sin(angles[2]), torch.cos(angles[2]), 0], [0, 0, 1]]
    )

    # Combined rotation matrix
    R = torch.matmul(torch.matmul(Rz, Ry), Rx)
    return R


def generate_random_translation(pos_boundary: float = 1) -> torch.Tensor:
    """Generates a random translation vector.

    Args:
        pos_boundary: Range for random translation values in 3D space.

    Returns:
        3-element translation vector.
    """
    """产生一个随机翻译向量。

    参数：
        pos_boundary: 在3D空间中随机翻译值的范围。

    返回：
        三个元素的转换向量。
    """
    return torch.rand(3) * 2 * pos_boundary - pos_boundary  # Random translation in 3D space


def generate_random_transformation_matrix(pos_boundary: float = 1, rot_boundary: float = (2 * math.pi)) -> torch.Tensor:
    """Generates a random transformation matrix combining rotation and translation.

    Args:
        pos_boundary: Range for random translation values.
        rot_boundary: Range for random rotation angles.

    Returns:
        4x4 transformation matrix.
    """
    """产生一个结合转换和转换的随机转换矩阵。

    参数：
        pos_boundary: 随机翻译值的范围。
        rot_boundary: 随机旋转角的范围。

    返回：
        4x4转换矩阵。
    """
    R = generate_random_rotation(rot_boundary)
    translation = generate_random_translation(pos_boundary)

    # Create the transformation matrix
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    return T
