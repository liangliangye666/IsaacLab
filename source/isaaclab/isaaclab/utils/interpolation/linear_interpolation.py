# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


class LinearInterpolation:
    """Linearly interpolates a sampled scalar function for arbitrary query points.

    This class implements a linear interpolation for a scalar function. The function maps from real values, x, to
    real values, y. It expects a set of samples from the function's domain, x, and the corresponding values, y.
    The class allows querying the function's values at any arbitrary point.

    The interpolation is done by finding the two closest points in x to the query point and then linearly
    interpolating between the corresponding y values. For the query points that are outside the input points,
    the class does a zero-order-hold extrapolation based on the boundary values. This means that the class
    returns the value of the closest point in x.
    """
    """在任意查询点中线性插入样本的尺度函数。

    这个类实现了对 skalar 函数的线性回合。
    函数从实际值，x，到实际值，y。
    它预计从函数域，x，和相应的值，y，
    该类允许在任意点查询函数的值。

    通过找到 x 中最接近查询点的两个点，然后在相应的 y 值之间线性地进行回合。
    在输入点之外的查询点上，该类根据边界值进行了零顺序持久外分。
    这意味着该类返回x中最接近点的值。
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, device: str):
        """Initializes the linear interpolation.

        The scalar function maps from real values, x, to real values, y. The input to the class is a set of samples
        from the function's domain, x, and the corresponding values, y.

        Note:
            The input tensor x should be sorted in ascending order.

        Args:
            x: An vector of samples from the function's domain. The values should be sorted in ascending order.
                Shape is (num_samples,)
            y: The function's values associated to the input x. Shape is (num_samples,)
            device: The device used for processing.

        Raises:
            ValueError: If the input tensors are empty or have different sizes.
            ValueError: If the input tensor x is not sorted in ascending order.
        """
        """开始线性插射。

        尺度函数从实值，x，到实值，y。
        类的输入是样本集
        from the function's domain, x, and the corresponding values, y.

        说明：
            输入子x应按上升顺序进行排序。

        参数：
            x: 函数域中的样本向量。
               值应按上升顺序进行排序。
               形状是 (num_samples，)
            y: 与输入 x 相关的函数值。
               形状是 (num_samples，)
            device: 用于加工的装置。

        异常：
            ValueError: 如果输入子是空的或有不同的尺寸。
            ValueError: 如果输入子 x 不按上升顺序排序。
        """
        # make sure that input tensors are 1D of size (num_samples,)
        self._x = x.view(-1).clone().to(device=device)
        self._y = y.view(-1).clone().to(device=device)

        # make sure sizes are correct
        if self._x.numel() == 0:
            raise ValueError("Input tensor x is empty!")
        if self._x.numel() != self._y.numel():
            raise ValueError(f"Input tensors x and y have different sizes: {self._x.numel()} != {self._y.numel()}")
        # make sure that x is sorted
        if torch.any(self._x[1:] < self._x[:-1]):
            raise ValueError("Input tensor x is not sorted in ascending order!")

    def compute(self, q: torch.Tensor) -> torch.Tensor:
        """Calculates a linearly interpolated values for the query points.

        Args:
           q: The query points. It can have any arbitrary shape.

        Returns:
            The interpolated values at query points. It has the same shape as the input tensor.
        """
        """计算对查询点的线性插入值。

        参数：
           q: 查询点。
              它可以有任意的形状。

        返回：
            在查询点中插入的值。
            它的形状与输入子相同。
        """
        # serialized q
        q_1d = q.view(-1)
        # Number of elements in the x that are strictly smaller than query points (use int32 instead of int64)
        num_smaller_elements = torch.sum(self._x.unsqueeze(1) < q_1d.unsqueeze(0), dim=0, dtype=torch.int)

        # The index pointing to the first element in x such that x[lower_bound_i] < q_i
        # If a point is smaller that all x elements, it will assign 0
        lower_bound = torch.clamp(num_smaller_elements - 1, min=0)
        # The index pointing to the first element in x such that x[upper_bound_i] >= q_i
        # If a point is greater than all x elements, it will assign the last elements' index
        upper_bound = torch.clamp(num_smaller_elements, max=self._x.numel() - 1)

        # compute the weight as: (q_i - x_lb) / (x_ub - x_lb)
        weight = (q_1d - self._x[lower_bound]) / (self._x[upper_bound] - self._x[lower_bound])
        # If a point is out of bounds assign weight 0.0
        weight[upper_bound == lower_bound] = 0.0

        # Perform linear interpolation
        fq = self._y[lower_bound] + weight * (self._y[upper_bound] - self._y[lower_bound])

        # deserialized fq
        fq = fq.view(q.shape)
        return fq
