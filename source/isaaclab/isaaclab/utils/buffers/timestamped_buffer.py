# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

import torch


@dataclass
class TimestampedBuffer:
    """A buffer class containing data and its timestamp.

    This class is a simple data container that stores a tensor and its timestamp. The timestamp is used to
    track the last update of the buffer. The timestamp is set to -1.0 by default, indicating that the buffer
    has not been updated yet. The timestamp should be updated whenever the data in the buffer is updated. This
    way the buffer can be used to check whether the data is outdated and needs to be refreshed.

    The buffer is useful for creating lazy buffers that only update the data when it is outdated. This can be
    useful when the data is expensive to compute or retrieve. For example usage, refer to the data classes in
    the :mod:`isaaclab.assets` module.
    """
    """包含数据及其时间标签的缓冲类。

    这类是一个简单的数据容器，存储一个子及其时间标签。
    时间用于追踪缓冲器的最后更新。
    按默认设置时间为 -1.0，表明缓冲器尚未更新。
    每当缓冲器中的数据更新时，应更新时间。
    这样，缓冲器可以用来检查数据是否过时，需要更新。

    缓冲器对于创建 update惰的缓冲器是有用的，只有当数据过时时才会更新数据。
    如果数据计算或检索昂贵，
    例如使用，请参见数据类
    the :模块:`isaaclab.assets`
    """

    data: torch.Tensor = None  # type: ignore
    """The data stored in the buffer. Default is None, indicating that the buffer is empty."""
    """在缓冲器中存储的数据。
    默认是None，表示缓冲器空。
    """

    timestamp: float = -1.0
    """Timestamp at the last update of the buffer. Default is -1.0, indicating that the buffer has not been updated."""
    """在缓冲器的最后更新时刻。
    默认为 -1.0，表明缓冲器尚未更新。
    """
