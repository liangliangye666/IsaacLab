# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class RslRlSymmetryCfg:
    """Configuration for the symmetry-augmentation in the training.

    When :meth:`use_data_augmentation` is True, the :meth:`data_augmentation_func` is used to generate
    augmented observations and actions. These are then used to train the model.

    When :meth:`use_mirror_loss` is True, the :meth:`mirror_loss_coeff` is used to weight the
    symmetry-mirror loss. This loss is directly added to the agent's loss function.

    If both :meth:`use_data_augmentation` and :meth:`use_mirror_loss` are False, then no symmetry-based
    training is enabled. However, the :meth:`data_augmentation_func` is called to compute and log
    symmetry metrics. This is useful for performing ablations.

    For more information, please check the work from :cite:`mittal2024symmetry`.
    """
    """在训练中增加对称的配置。

    When ::`use_data_augmentation`是True，:meth:`data_augmentation_func`用于生成
    增加观测和动作。
    然后用这些来训练模型。

    When ::`use_mirror_loss`是True，:meth:`mirror_loss_coeff`用于权重
    它们的相对度和镜子损失。
    这种损失直接增加到代理的损失函数。

    如果:meth:`use_data_augmentation`和:meth:`use_mirror_loss`都是False，则不启用基于对称的训练。
    然而，:meth:`data_augmentation_func`被调用为计算和记录对称度的指标。
    这对于执行除是有用的。

    更多信息请查看下文:`mittal2024symmetry`。
    """

    use_data_augmentation: bool = False
    """Whether to use symmetry-based data augmentation. Default is False."""
    """是否使用基于对称的数据增强。
    默认是False。
    """

    use_mirror_loss: bool = False
    """Whether to use the symmetry-augmentation loss. Default is False."""
    """如果使用对称增长损失。
    默认是False。
    """

    data_augmentation_func: callable = MISSING
    """The symmetry data augmentation function.

    The function signature should be as follows:

    Args:

        env (VecEnv): The environment object. This is used to access the environment's properties.
        obs (tensordict.TensorDict | None): The observation tensor dictionary. If None, the observation is not used.
        action (torch.Tensor | None): The action tensor. If None, the action is not used.

    Returns:
        A tuple containing the augmented observation dictionary and action tensors. The tensors can be None,
        if their respective inputs are None.
    """
    """交对称数据增强函数。

    函数签名应如下:

    参数：

        env (VecEnv): 环境对象。
                      它们用于访问环境的特性。
        obs (tensordict.TensorDict | None): 观测张量字典。
                                            如果 None，则不使用观测。
        action (torch.Tensor | None): 动作张量。
                                      如果 None，该动作不会使用。

    返回：
        包含增强的观测字典和动作张量的元组。
        子可以是None，
        if their respective inputs are None.
    """

    mirror_loss_coeff: float = 0.0
    """The weight for the symmetry-mirror loss. Default is 0.0."""
    """对于对称镜子损失的重量。
    默认是0.0。
    """
