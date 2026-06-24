# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

import torch


@dataclass
class FrameTransformerData:
    """Data container for the frame transformer sensor."""
    """框架变压器传感器的数据容器。"""

    target_frame_names: list[str] = None
    """Target frame names (this denotes the order in which that frame data is ordered).

    The frame names are resolved from the :attr:`FrameTransformerCfg.FrameCfg.name` field.
    This does not necessarily follow the order in which the frames are defined in the config due to
    the regex matching.
    """
    """目标框架名称 (这表示该框架数据的顺序)。

    图像名称是从:attr:`FrameTransformerCfg.FrameCfg.name`字段中得到的。
    这不一定是由于regex匹配而在配置中定义的框架的顺序。
    """

    target_pos_source: torch.Tensor = None
    """Position of the target frame(s) relative to source frame.

    Shape is (N, M, 3), where N is the number of environments, and M is the number of target frames.
    """
    """目标框架的位置与源框架相比。

    形状是 (N，M，3)，其中N是环境数量，M是目标框架数量。
    """

    target_quat_source: torch.Tensor = None
    """Orientation of the target frame(s) relative to source frame quaternion (w, x, y, z).

    Shape is (N, M, 4), where N is the number of environments, and M is the number of target frames.
    """
    """目标框架的导向 (x，y，z) 与源框架四元数相对。

    形状是 (N，M，4) ，其中N是环境的数量，M是目标框架的数量。
    """

    target_pos_w: torch.Tensor = None
    """Position of the target frame(s) after offset (in world frame).

    Shape is (N, M, 3), where N is the number of environments, and M is the number of target frames.
    """
    """目标框架的位置 (在世界框架中) 偏移后。

    形状是 (N，M，3)，其中N是环境数量，M是目标框架数量。
    """

    target_quat_w: torch.Tensor = None
    """Orientation of the target frame(s) after offset (in world frame) quaternion (w, x, y, z).

    Shape is (N, M, 4), where N is the number of environments, and M is the number of target frames.
    """
    """目标框架的定向 (在世界框架中) 偏移后 (w， x， y， z) 四角形。

    形状是 (N，M，4) ，其中N是环境的数量，M是目标框架的数量。
    """

    source_pos_w: torch.Tensor = None
    """Position of the source frame after offset (in world frame).

    Shape is (N, 3), where N is the number of environments.
    """
    """偏移后源框架的位置 (世界框架)。

    形状是 (N， 3)，其中N是环境的数量。
    """

    source_quat_w: torch.Tensor = None
    """Orientation of the source frame after offset (in world frame) quaternion (w, x, y, z).

    Shape is (N, 4), where N is the number of environments.
    """
    """源框架的偏移 (世界框架) 后的导向 (w， x， y， z)。

    形状是 (N， 4)，其中N是环境的数量。
    """
