# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class RslRlRndCfg:
    """Configuration for the Random Network Distillation (RND) module.

    For more information, please check the work from :cite:`schwarke2023curiosity`.
    """
    """随机网络蒸 (RND) 模块的配置。

    更多信息请查看下文:`schwarke2023curiosity`。
    """

    @configclass
    class WeightScheduleCfg:
        """Configuration for the weight schedule."""
        """体重安排的配置。"""

        mode: str = "constant"
        """The type of weight schedule. Default is "constant"."""
        """体重时间表。
        默认是"常态"。
        """

    @configclass
    class LinearWeightScheduleCfg(WeightScheduleCfg):
        """Configuration for the linear weight schedule.

        This schedule decays the weight linearly from the initial value to the final value
        between :attr:`initial_step` and before :attr:`final_step`.
        """
        """为线性体重表的配置。

        这一时间表将重量从初始值降到最终值。
        between :吸引:`initial_step`在此之前:attr:`final_step`。
        """

        mode: str = "linear"

        final_value: float = MISSING
        """The final value of the weight parameter."""
        """重量参数的最终值。"""

        initial_step: int = MISSING
        """The initial step of the weight schedule.

        For steps before this step, the weight is the initial value specified in :attr:`RslRlRndCfg.weight`.
        """
        """体重计划的第一步。

        在此步骤前的步骤中，重量是:attr:`RslRlRndCfg.weight`中指定的初始值。
        """

        final_step: int = MISSING
        """The final step of the weight schedule.

        For steps after this step, the weight is the final value specified in :attr:`final_value`.
        """
        """体重计划的最后一步。

        在此步骤之后的步骤中，重量是:attr:`final_value`中指定的最终值。
        """

    @configclass
    class StepWeightScheduleCfg(WeightScheduleCfg):
        """Configuration for the step weight schedule.

        This schedule sets the weight to the value specified in :attr:`final_value` at step :attr:`final_step`.
        """
        """按步骤体重时间表的配置。

        该表设定重量为:attr:`final_value`中的:attr:`final_step`步骤中的值。
        """

        mode: str = "step"

        final_step: int = MISSING
        """The final step of the weight schedule.

        For steps after this step, the weight is the value specified in :attr:`final_value`.
        """
        """体重计划的最后一步。

        在此步骤之后的步骤中，重量是:attr:`final_value`中指定的值。
        """

        final_value: float = MISSING
        """The final value of the weight parameter."""
        """重量参数的最终值。"""

    weight: float = 0.0
    """The weight for the RND reward (also known as intrinsic reward). Default is 0.0.

    Similar to other reward terms, the RND reward is scaled by this weight.
    """
    """对于RND奖励的重量 (也称为内在奖励)。
    默认是0.0。

    类似于其他奖励条件，RND奖励是按这个权重进行的。
    """

    weight_schedule: WeightScheduleCfg | None = None
    """The weight schedule for the RND reward. Default is None, which means the weight is constant."""
    """对于RND奖励的体重表。
    默认是None，这意味着重量是恒定的。
    """

    reward_normalization: bool = False
    """Whether to normalize the RND reward. Default is False."""
    """是否正常化RND奖励。
    默认是False。
    """

    state_normalization: bool = False
    """Whether to normalize the RND state. Default is False."""
    """是否正常化RND状态。
    默认是False。
    """

    learning_rate: float = 1e-3
    """The learning rate for the RND module. Default is 1e-3."""
    """RND模块的学习速度。
    默认是1e-3。
    """

    num_outputs: int = 1
    """The number of outputs for the RND module. Default is 1."""
    """RND模块输出数量
    默认是1。
    """

    predictor_hidden_dims: list[int] = [-1]
    """The hidden dimensions for the RND predictor network. Default is [-1].

    If the list contains -1, then the hidden dimensions are the same as the input dimensions.
    """
    """对于RND预测器网络的隐藏维度。
    默认是 [-1]。

    如果列表包含 -1，那么隐藏的尺寸与输入尺寸相同。
    """

    target_hidden_dims: list[int] = [-1]
    """The hidden dimensions for the RND target network. Default is [-1].

    If the list contains -1, then the hidden dimensions are the same as the input dimensions.
    """
    """隐藏的尺寸为RND目标网络。
    默认是 [-1]。

    如果列表包含 -1，那么隐藏的尺寸与输入尺寸相同。
    """
