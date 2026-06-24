# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from . import actuator_net
from .actuator_pd_cfg import DCMotorCfg


@configclass
class ActuatorNetLSTMCfg(DCMotorCfg):
    """Configuration for LSTM-based actuator model."""
    """基于LSTM的执行器模型的配置。"""

    class_type: type = actuator_net.ActuatorNetLSTM
    # we don't use stiffness and damping for actuator net
    stiffness = None
    damping = None

    network_file: str = MISSING
    """Path to the file containing network weights."""
    """网络权重的文件的路径。"""


@configclass
class ActuatorNetMLPCfg(DCMotorCfg):
    """Configuration for MLP-based actuator model."""
    """基于MLP的执行器模型的配置。"""

    class_type: type = actuator_net.ActuatorNetMLP
    # we don't use stiffness and damping for actuator net

    stiffness = None
    damping = None

    network_file: str = MISSING
    """Path to the file containing network weights."""
    """网络权重的文件的路径。"""

    pos_scale: float = MISSING
    """Scaling of the joint position errors input to the network."""
    """网络输入的关节位置错误扩展。"""
    vel_scale: float = MISSING
    """Scaling of the joint velocities input to the network."""
    """扩展对网络输入的关节速度。"""
    torque_scale: float = MISSING
    """Scaling of the joint efforts output from the network."""
    """从网络中产出的联合努力规模化。"""

    input_order: Literal["pos_vel", "vel_pos"] = MISSING
    """Order of the inputs to the network.

    The order can be one of the following:

    * ``"pos_vel"``: joint position errors followed by joint velocities
    * ``"vel_pos"``: joint velocities followed by joint position errors
    """
    """网络输入的顺序

    订单可能是以下内容之一:

    * ``"pos_vel"``:关节位置错误，随后是关节速度
    * ``"vel_pos"``:关节速度，随后是关节位置错误
    """

    input_idx: Iterable[int] = MISSING
    """
    Indices of the actuator history buffer passed as inputs to the network.

    The index *0* corresponds to current time-step, while *n* corresponds to n-th
    time-step in the past. The allocated history length is `max(input_idx) + 1`.
    """
    """作为输入到网络的执行器历史缓冲的索引。

    索引*0*与当前的时间步骤相符，而*n*与过去的n-th时间步骤相符。
    分配的历史长度是`max(input_idx) + 1`。
    """
