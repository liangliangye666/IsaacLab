# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for different data types."""

from __future__ import annotations
"""对不同数据类型的子模块。"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class ArticulationActions:
    """Data container to store articulation's joints actions.

    This class is used to store the actions of the joints of an articulation.
    It is used to store the joint positions, velocities, efforts, and indices.

    If the actions are not provided, the values are set to None.
    """
    """数据容器存储关节的动作。

    这类用于存储关节的动作。
    它用于存储关节位置，速度，努力和索引。

    如果没有提供操作，则设置为None。
    """

    joint_positions: torch.Tensor | None = None
    """The joint positions of the articulation. Defaults to None."""
    """关节的关节位置。
    默认为 None。
    """

    joint_velocities: torch.Tensor | None = None
    """The joint velocities of the articulation. Defaults to None."""
    """关节的关节速度。
    默认为 None。
    """

    joint_efforts: torch.Tensor | None = None
    """The joint efforts of the articulation. Defaults to None."""
    """共同努力。
    默认为 None。
    """

    joint_indices: torch.Tensor | Sequence[int] | slice | None = None
    """The joint indices of the articulation. Defaults to None.

    If the joint indices are a slice, this indicates that the indices are continuous and correspond
    to all the joints of the articulation. We use a slice to make the indexing more efficient.
    """
    """关节的联合索引。
    默认为 None。

    如果关节指标是片段，这表明指标是连续的，并与关节的所有关节相符。
    我们使用一块，使索引更有效。
    """
