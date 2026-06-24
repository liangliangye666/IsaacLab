# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions to project between pointcloud and depth images."""

# needed to import for allowing type-hinting: torch.device | str | None
from __future__ import annotations
"""帮助函数在点云和深度图像之间投影。"""

from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.utils.array import TensorData, convert_to_torch

"""
Depth <-> Pointcloud conversions.
"""
"""Dep度 <->点云转换。
"""


def transform_points(
    points: TensorData,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray | torch.Tensor:
    r"""Transform input points in a given frame to a target frame.

    This function transform points from a source frame to a target frame. The transformation is defined by the
    position ``t`` and orientation ``R`` of the target frame in the source frame.

    .. math::
        p_{target} = R_{target} \times p_{source} + t_{target}

    If either the inputs `position` and `orientation` are None, the corresponding transformation is not applied.

    Args:
        points: a tensor of shape (p, 3) or (n, p, 3) comprising of 3d points in source frame.
        position: The position of source frame in target frame. Defaults to None.
        orientation: The orientation (w, x, y, z) of source frame in target frame.
            Defaults to None.
        device: The device for torch where the computation
            should be executed. Defaults to None, i.e. takes the device that matches the depth image.

    Returns:
        A tensor of shape (N, 3) comprising of 3D points in target frame.
        If the input is a numpy array, the output is a numpy array. Otherwise, it is a torch tensor.
    """
    """将给定的框架中的输入点转换为目标框架。

    这个函数将点从源框转换为目标框。
    转换由目标框架的位置``t``和方向``R``在源框架中定义。

    .. math::
        p_{target} = R_{target} \times p_{source} + t_{target}

    如果输入 `position` 和 `orientation` 是 None，则不应运行相应的转换。

    参数：
        points: 一个形状 (p，3) 或 (n，p，3) 的子，由源框中的3d点组成。
        position: 目标框架中的源框架位置。
                  默认为 None。
        orientation: 目标框架中源框架的方向 (w，x，y，z)。
                     默认为 None。
        device: 计算应执行的火装置。
                默认为 None，i.e.将与深度图像相匹配的设备。

    返回：
        一个形状张量 (N，3) 包含目标框架中的3D点。
        如果输入是 numpy array，输出是 numpy array。
        否则，它是火。
    """
    # check if numpy
    is_numpy = isinstance(points, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert to torch
    points = convert_to_torch(points, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = points.device
    # apply rotation
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)
    # apply translation
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    # apply transformation
    points = math_utils.transform_points(points, position, orientation)

    # return everything according to input type
    if is_numpy:
        return points.detach().cpu().numpy()
    else:
        return points


def create_pointcloud_from_depth(
    intrinsic_matrix: np.ndarray | torch.Tensor | wp.array,
    depth: np.ndarray | torch.Tensor | wp.array,
    keep_invalid: bool = False,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray | torch.Tensor:
    r"""Creates pointcloud from input depth image and camera intrinsic matrix.

    This function creates a pointcloud from a depth image and camera intrinsic matrix. The pointcloud is
    computed using the following equation:

    .. math::
        p_{camera} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`K` is the camera intrinsic matrix, :math:`u` and :math:`v` are the pixel coordinates and
    :math:`d` is the depth value at the pixel.

    Additionally, the pointcloud can be transformed from the camera frame to a target frame by providing
    the position ``t`` and orientation ``R`` of the camera in the target frame:

    .. math::
        p_{target} = R_{target} \times p_{camera} + t_{target}

    Args:
        intrinsic_matrix: A (3, 3) array providing camera's calibration matrix.
        depth: An array of shape (H, W) with values encoding the depth measurement.
        keep_invalid: Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation (w, x, y, z) of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed.
            Defaults to None, i.e. takes the device that matches the depth image.

    Returns:
        An array/tensor of shape (N, 3) comprising of 3D coordinates of points.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.
    """
    """通过输入深度图像和摄像头内在矩阵创建点云。

    这种函数从深度图像和摄像头内在矩阵中创建一个点云。
    计算点云使用以下方程:

    .. math::
        p_{camera} = K^{-1} \times [u, v, 1]^T \times d

    where :数学:`K`是摄像头内在矩阵， 数学:`u`和 数学:`v`是像素坐标和
    :math:`d`是像素的深度值。

    此外，点云可以通过在目标框架中提供相机的位置``t``和方向``R``来从相机框架转换为目标框架:

    .. math::
        p_{target} = R_{target} \times p_{camera} + t_{target}

    参数：
        intrinsic_matrix: 提供相机校准矩阵的 (3，3) 阵列。
        depth: 一个形状阵列 (H，W) 含有编码深度测量值。
        keep_invalid: 在云中保留不行点还是不保留的。
                      无效点与深度值为0.0或NaN的像素相符。
                      默认为 False。
        position: 摄像机在目标框架中的位置。
                  默认为 None。
        orientation: 目标框架中的相机的方向 (w， x， y， z)。
                     默认为 None。
        device: 计算应执行的火装置。
                默认为 None，i.e.将与深度图像相匹配的设备。

    返回：
        一个由3D点坐标组成的形状阵列/ensor (N，3)。
        如果输入深度是torch.tensor或wp.array类型，返回的数据类型是火。
        否则，将返回np.ndarray。
    """
    # We use PyTorch here for matrix multiplication since it is compiled with Intel MKL while numpy
    # by default uses OpenBLAS. With PyTorch (CPU), we could process a depth image of size (480, 640)
    # in 0.0051 secs, while with numpy it took 0.0292 secs.

    # convert to numpy matrix
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    depth = convert_to_torch(depth, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = depth.device
    # convert inputs to torch tensors
    intrinsic_matrix = convert_to_torch(intrinsic_matrix, dtype=torch.float32, device=device)
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)
    # compute pointcloud
    depth_cloud = math_utils.unproject_depth(depth, intrinsic_matrix)
    # convert 3D points to world frame
    depth_cloud = math_utils.transform_points(depth_cloud, position, orientation)

    # keep only valid entries if flag is set
    if not keep_invalid:
        pts_idx_to_keep = torch.all(torch.logical_and(~torch.isnan(depth_cloud), ~torch.isinf(depth_cloud)), dim=1)
        depth_cloud = depth_cloud[pts_idx_to_keep, ...]

    # return everything according to input type
    if is_numpy:
        return depth_cloud.detach().cpu().numpy()
    else:
        return depth_cloud


def create_pointcloud_from_rgbd(
    intrinsic_matrix: torch.Tensor | np.ndarray | wp.array,
    depth: torch.Tensor | np.ndarray | wp.array,
    rgb: torch.Tensor | wp.array | np.ndarray | tuple[float, float, float] = None,
    normalize_rgb: bool = False,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
    num_channels: int = 3,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray]:
    """Creates pointcloud from input depth image and camera transformation matrix.

    This function provides the same functionality as :meth:`create_pointcloud_from_depth` but also allows
    to provide the RGB values for each point.

    The ``rgb`` attribute is used to resolve the corresponding point's color:

    - If a ``np.array``/``wp.array``/``torch.tensor`` of shape (H, W, 3), then the corresponding channels
      encode the RGB values.
    - If a tuple, then the point cloud has a single color specified by the values (r, g, b).
    - If None, then default color is white, i.e. (0, 0, 0).

    If the input ``normalize_rgb`` is set to :obj:`True`, then the RGB values are normalized to be in the range [0, 1].

    Args:
        intrinsic_matrix: A (3, 3) array/tensor providing camera's calibration matrix.
        depth: An array/tensor of shape (H, W) with values encoding the depth measurement.
        rgb: Color for generated point cloud. Defaults to None.
        normalize_rgb: Whether to normalize input rgb. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation `(w, x, y, z)` of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed. Defaults to None, in which case
            it takes the device that matches the depth image.
        num_channels: Number of channels in RGB pointcloud. Defaults to 3.

    Returns:
        A tuple of (N, 3) arrays or tensors containing the 3D coordinates of points and their RGB color respectively.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.

    Raises:
        ValueError:  When rgb image is a numpy array but not of shape (H, W, 3) or (H, W, 4).
    """
    """从输入深度图像和摄像头转换矩阵创建点云。

    这个函数提供了与:meth:`create_pointcloud_from_depth`相同的功能，但也允许为每个点提供RGB值。

    用``rgb``属性来解决相应点的颜色:

    - 如果 ``np.array``/``wp.array``/``torch.tensor``的形状 (H，W，3)，则相应的道将RGB值编码。
    - 如果是tuple，那么点云的颜色是指值 (r，g，b) 所指定的。
    - 如果None，则默认颜色是白色，i.e。 (0， 0， 0)。

    如果输入 ``normalize_rgb`` 设置为 :obj:`True`，则RGB 值将正常化为 [0， 1] 范围。

    参数：
        intrinsic_matrix: 一个 (3， 3) 阵列/ensor提供摄像头的校准矩阵。
        depth: 形状 (H，W) 的阵列/度器，含有编码深度测量值。
        rgb: 产生点云的颜色。
             默认为 None。
        normalize_rgb: 是否正常化输入rgb。
                       默认为 False。
        position: 摄像机在目标框架中的位置。
                  默认为 None。
        orientation: 在目标框架中的相机的`(w， x， y， z)`导向。
                     默认为 None。
        device: 计算应执行的火装置。
                默认为 None，在这种情况下，它采用与深度图像相匹配的设备。
        num_channels: 在RGB点云中的频道数量。
                      默认的3。

    返回：
        包含分别点的3D坐标和它们的RGB颜色的 (N，3) 阵列或子的图普。
        如果输入深度是torch.tensor或wp.array类型，返回的数据类型是火。
        否则，将返回np.ndarray。

    异常：
        ValueError:  当 rgb 图像是个 numpy 阵列，但不是形状 (H， W， 3) 或 (H， W， 4)。
    """
    # check valid inputs
    if rgb is not None and not isinstance(rgb, tuple):
        if len(rgb.shape) == 3:
            if rgb.shape[2] not in [3, 4]:
                raise ValueError(f"Input rgb image of invalid shape: {rgb.shape} != (H, W, 3) or (H, W, 4).")
        else:
            raise ValueError(f"Input rgb image not three-dimensional. Received shape: {rgb.shape}.")
    if num_channels not in [3, 4]:
        raise ValueError(f"Invalid number of channels: {num_channels} != 3 or 4.")

    # check if input depth is numpy array
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    if is_numpy:
        depth = torch.from_numpy(depth).to(device=device)
    # retrieve XYZ pointcloud
    points_xyz = create_pointcloud_from_depth(intrinsic_matrix, depth, True, position, orientation, device=device)

    # get image height and width
    im_height, im_width = depth.shape[:2]
    # total number of points
    num_points = im_height * im_width
    # extract color value
    if rgb is not None:
        if isinstance(rgb, (np.ndarray, torch.Tensor, wp.array)):
            # copy numpy array to preserve
            rgb = convert_to_torch(rgb, device=device, dtype=torch.float32)
            rgb = rgb[:, :, :3]
            # convert the matrix to (W, H, 3) from (H, W, 3) since depth processing
            # is done in the order (u, v) where u: (0, W-1) and v: (0 - H-1)
            points_rgb = rgb.permute(1, 0, 2).reshape(-1, 3)
        elif isinstance(rgb, (tuple, list)):
            # same color for all points
            points_rgb = torch.Tensor((rgb,) * num_points, device=device, dtype=torch.uint8)
        else:
            # default color is white
            points_rgb = torch.Tensor(((0, 0, 0),) * num_points, device=device, dtype=torch.uint8)
    else:
        points_rgb = torch.Tensor(((0, 0, 0),) * num_points, device=device, dtype=torch.uint8)
    # normalize color values
    if normalize_rgb:
        points_rgb = points_rgb.float() / 255

    # remove invalid points
    pts_idx_to_keep = torch.all(torch.logical_and(~torch.isnan(points_xyz), ~torch.isinf(points_xyz)), dim=1)
    points_rgb = points_rgb[pts_idx_to_keep, ...]
    points_xyz = points_xyz[pts_idx_to_keep, ...]

    # add additional channels if required
    if num_channels == 4:
        points_rgb = torch.nn.functional.pad(points_rgb, (0, 1), mode="constant", value=1.0)

    # return everything according to input type
    if is_numpy:
        return points_xyz.cpu().numpy(), points_rgb.cpu().numpy()
    else:
        return points_xyz, points_rgb


def save_images_to_file(images: torch.Tensor, file_path: str):
    """Save images to file.

    Args:
        images: A tensor of shape (N, H, W, C) containing the images.
        file_path: The path to save the images to.
    """
    """保存图像以文件。

    参数：
        images: 包含图像的形状张量 (N，H，W，C)。
        file_path: 保存图像的路径。
    """
    from torchvision.utils import make_grid, save_image

    save_image(
        make_grid(torch.swapaxes(images.unsqueeze(1), 1, -1).squeeze(-1), nrow=round(images.shape[0] ** 0.5)), file_path
    )
