# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for working with different array backends."""

# needed to import for allowing type-hinting: torch.device | str | None
from __future__ import annotations
"""包含与不同阵列后端工作的工具的子模块。"""

from typing import Union

import numpy as np
import torch
import warp as wp

TensorData = Union[np.ndarray, torch.Tensor, wp.array]  # noqa: UP007
"""Type definition for a tensor data.

Union of numpy, torch, and warp arrays.
"""
"""为 data光数据的类型定义。

Union，火和曲阵列的联盟。
"""

TENSOR_TYPES = {
    "numpy": np.ndarray,
    "torch": torch.Tensor,
    "warp": wp.array,
}
"""A dictionary containing the types for each backend.

The keys are the name of the backend ("numpy", "torch", "warp") and the values are the corresponding type
(``np.ndarray``, ``torch.Tensor``, ``wp.array``).
"""
"""一个包含每个后端类型的字典。

键是后端的名称 ("numpy"，"torch"，"warp") 和值是相应的类型 (``np.ndarray``，``torch.Tensor``，``wp.array``)。
"""

TENSOR_TYPE_CONVERSIONS = {
    "numpy": {wp.array: lambda x: x.numpy(), torch.Tensor: lambda x: x.detach().cpu().numpy()},
    "torch": {wp.array: lambda x: wp.torch.to_torch(x), np.ndarray: lambda x: torch.from_numpy(x)},
    "warp": {np.array: lambda x: wp.array(x), torch.Tensor: lambda x: wp.torch.from_torch(x)},
}
"""A nested dictionary containing the conversion functions for each backend.

The keys of the outer dictionary are the name of target backend ("numpy", "torch", "warp"). The keys of the
inner dictionary are the source backend (``np.ndarray``, ``torch.Tensor``, ``wp.array``).
"""
"""包含每个后端的转换函数的嵌入式字典。

外部字典的键是目标后端的名称 ("numpy"，"torch"，"warp")。
内部字典的键是源后端 (``np.ndarray``，``torch.Tensor``，``wp.array``)。
"""


def convert_to_torch(
    array: TensorData,
    dtype: torch.dtype = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Converts a given array into a torch tensor.

    The function tries to convert the array to a torch tensor. If the array is a numpy/warp arrays, or python
    list/tuples, it is converted to a torch tensor. If the array is already a torch tensor, it is returned
    directly.

    If ``device`` is None, then the function deduces the current device of the data. For numpy arrays,
    this defaults to "cpu", for torch tensors it is "cpu" or "cuda", and for warp arrays it is "cuda".

    Note:
        Since PyTorch does not support unsigned integer types, unsigned integer arrays are converted to
        signed integer arrays. This is done by casting the array to the corresponding signed integer type.

    Args:
        array: The input array. It can be a numpy array, warp array, python list/tuple, or torch tensor.
        dtype: Target data-type for the tensor.
        device: The target device for the tensor. Defaults to None.

    Returns:
        The converted array as torch tensor.
    """
    """转换给定的数组为火。

    函数试图将数组转换为火。
    如果数组是一个 numpy/warp数组，或者python列表/tuples，则将其转换为火。
    如果数组已经是火，则直接返回。

    如果``device``是None，那么函数将数据的当前设备推断。
    对于 numpy 阵列，这个默认是"cpu"，对于火器，它是"cpu"或"cuda"，而对于变形阵列，它是"cuda"。

    说明：
        由于PyTorch不支持未签名整数类型，未签名整数阵列转换为签名整数阵列。
        通过将数组投放到相应的签名整数类型上，这样做。

    参数：
        array: 输入阵列。
               它可以是 numpy array， warp array， python list/tuple，或火。
        dtype: 为光器的目标数据类型。
        device: 子的目标装置。
                默认为 None。

    返回：
        转换为火。
    """
    # Convert array to tensor
    # if the datatype is not currently supported by torch we need to improvise
    # supported types are: https://pytorch.org/docs/stable/tensors.html
    if isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint32:
            array = array.astype(np.int32)
        # need to deal with object arrays (np.void) separately
        tensor = torch.from_numpy(array)
    elif isinstance(array, wp.array):
        if array.dtype == wp.uint32:
            array = array.view(wp.int32)
        tensor = wp.to_torch(array)
    else:
        tensor = torch.Tensor(array)
    # Convert tensor to the right device
    if device is not None and str(tensor.device) != str(device):
        tensor = tensor.to(device)
    # Convert dtype of tensor if requested
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.type(dtype)

    return tensor
