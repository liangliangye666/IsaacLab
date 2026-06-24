# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rl_cfg import RslRlBaseRunnerCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlDistillationStudentTeacherCfg:
    """Configuration for the distillation student-teacher networks."""
    """蒸学生与教师网络的配置。"""

    class_name: str = "StudentTeacher"
    """The policy class name. Default is StudentTeacher."""
    """策略类名字。
    默认是StudentTeacher。
    """

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the student policy."""
    """学生策略的初始噪音标准偏差。"""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""
    """策略的噪音标准偏差类型。
    默认是 skalar。
    """

    student_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the student network."""
    """是否将观测正常化为学生网络。"""

    teacher_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the teacher network."""
    """对于教师网络来说，"""

    student_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the student network."""
    """学生网络的隐藏维度。"""

    teacher_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the teacher network."""
    """教授网络的隐藏维度。"""

    activation: str = MISSING
    """The activation function for the student and teacher networks."""
    """学生和教师网络的激活功能。"""


@configclass
class RslRlDistillationStudentTeacherRecurrentCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for the distillation student-teacher recurrent networks."""
    """蒸学生和教师复发网络的配置。"""

    class_name: str = "StudentTeacherRecurrent"
    """The policy class name. Default is StudentTeacherRecurrent."""
    """策略类名字。
    默认是StudentTeacherRecurrent。
    """

    rnn_type: str = MISSING
    """The type of the RNN network. Either "lstm" or "gru"."""
    """RNN网络的类型。
    无论是"Istm"还是"Gru"。
    """

    rnn_hidden_dim: int = MISSING
    """The hidden dimension of the RNN network."""
    """隐藏的维度RNN网络。"""

    rnn_num_layers: int = MISSING
    """The number of layers of the RNN network."""
    """RNN网络的层次数。"""

    teacher_recurrent: bool = MISSING
    """Whether the teacher network is recurrent too."""
    """如果教师网络也是重复性的。"""


############################
# Algorithm configurations #
############################


@configclass
class RslRlDistillationAlgorithmCfg:
    """Configuration for the distillation algorithm."""
    """蒸算法的配置"""

    class_name: str = "Distillation"
    """The algorithm class name. Default is Distillation."""
    """算法类名称。
    默认是蒸。
    """

    num_learning_epochs: int = MISSING
    """The number of updates performed with each sample."""
    """每个样本进行的更新数量。"""

    learning_rate: float = MISSING
    """The learning rate for the student policy."""
    """学生策略的学习率。"""

    gradient_length: int = MISSING
    """The number of environment steps the gradient flows back."""
    """随着环境的增加，梯度向后流动。"""

    max_grad_norm: None | float = None
    """The maximum norm the gradient is clipped to."""
    """梯度的最高标准。"""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """The optimizer to use for the student policy."""
    """学生策略的优化器。"""

    loss_type: Literal["mse", "huber"] = "mse"
    """The loss type to use for the student policy."""
    """学生保险的损失类型。"""


#########################
# Runner configurations #
#########################


@configclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for distillation algorithms."""
    """蒸算法运行器的配置"""

    class_name: str = "DistillationRunner"
    """The runner class name. Default is DistillationRunner."""
    """跑者类名字。
    默认是DistillationRunner。
    """

    policy: RslRlDistillationStudentTeacherCfg = MISSING
    """The policy configuration."""
    """策略配置。"""

    algorithm: RslRlDistillationAlgorithmCfg = MISSING
    """The algorithm configuration."""
    """算法配置。"""
