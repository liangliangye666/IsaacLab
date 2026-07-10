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

    '''
    数据管道的惰性求值体系
        所谓懒加载，就是：
            初始化时不立刻读取数据；
            等你真正访问某个属性时，才去 PhysX 读取；
            读完后缓存起来；
            同一帧内再次访问就不重复读取。
        所以这些 buffer 可以理解为：
            机器人状态数据的缓存槽

    二、结构拆解
        TimestampedBuffer
        │
        ├── data: torch.Tensor = None
        │        实际存储的张量数据，如关节位置 [4096, 12]
        │        默认 None 表示"还没算过"
        │
        └── timestamp: float = -1.0
                数据上次更新的"时间戳"
                默认 -1.0 表示"从未更新"
                每次数据刷新时被设为当前的 _sim_timestamp
    三、核心设计模式：惰性求值
            TimestampedBuffer 本身不做任何计算。
            它的威力在于配合时间戳检查实现惰性求值。来看它在 ArticulationData 中的典型使用（articulation_data.py:778-786）：
                if self._root_link_pose_w.timestamp < self._sim_timestamp:
                    # read data from simulation
                    pose = self._root_physx_view.get_root_transforms().clone()
                    pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
                    # set the buffer data and timestamp
                    self._root_link_pose_w.data = pose
                    self._root_link_pose_w.timestamp = self._sim_timestamp

                return self._root_link_pose_w.data
        工作流程：
            用户访问 data.root_link_pose_w
                    │
                    ▼
                buffer.timestamp < _sim_timestamp ?
                    │
                    ├── 是（数据过期 / 从未计算）
                    │       ├── 从 PhysX 读取数据（昂贵！）
                    │       ├── buffer.data = 新数据
                    │       ├── buffer.timestamp = _sim_timestamp
                    │       └── return buffer.data
                    │
                    └── 否（数据是最新的）
                            └── return buffer.data  （直接返回缓存）
            同一帧内多次访问同一个属性，PhysX 只被查询一次。这就是惰性求值 + 缓存的完整实现。

    四、为什么 timestamp 默认是 -1.0？
        这是一个精妙的哨兵值设计：
                _sim_timestamp 从 0.0 开始，每步 + dt（通常是 0.005）
                -1.0 < 0.0，所以第一次访问一定会触发计算
                不需要额外的 is_initialized 标志——-1.0 本身就是"未初始化"的语义
            如果用 0.0 作为默认值，第一次访问时 0.0 < 0.0 为 False，就不会计算——访问者拿到一个 None 而不是真实数据，会直接崩溃。
        -1.0 只保证第一次访问一定会触发计算。计算完 buffer.timestamp 就被立刻设为 _sim_timestamp，所以"一直更新直到相等"的情况不会发生——因为一次更新就让它们相等了。
    _sim_timestamp 是物理步
        def update(self, dt: float):
            # update the simulation timestamp
            self._sim_timestamp += dt
        而 update(dt) 在每个物理子步循环中都被调用一次：
    '''
