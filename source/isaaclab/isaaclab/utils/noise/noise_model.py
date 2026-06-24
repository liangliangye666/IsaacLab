# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from . import noise_cfg

##
# Noise as functions.
##


def constant_noise(data: torch.Tensor, cfg: noise_cfg.ConstantNoiseCfg) -> torch.Tensor:
    """Applies a constant noise bias to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for constant noise.

    Returns:
        The data modified by the noise parameters provided.
    """
    """对给定的数据集应用一个恒定的噪音偏差。

    参数：
        data: 适用于噪音的未修改数据集。
        cfg: 常规噪音的配置参数。

    返回：
        通过提供噪音参数修改的数据。
    """

    # fix tensor device for bias on first call and update config parameters
    if isinstance(cfg.bias, torch.Tensor):
        cfg.bias = cfg.bias.to(device=data.device)

    if cfg.operation == "add":
        return data + cfg.bias
    elif cfg.operation == "scale":
        return data * cfg.bias
    elif cfg.operation == "abs":
        return torch.zeros_like(data) + cfg.bias
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def uniform_noise(data: torch.Tensor, cfg: noise_cfg.UniformNoiseCfg) -> torch.Tensor:
    """Applies a uniform noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for uniform noise.

    Returns:
        The data modified by the noise parameters provided.
    """
    """适用于给定的数据集。

    参数：
        data: 适用于噪音的未修改数据集。
        cfg: 均噪音的配置参数。

    返回：
        通过提供噪音参数修改的数据。
    """

    # fix tensor device for n_max on first call and update config parameters
    if isinstance(cfg.n_max, torch.Tensor):
        cfg.n_max = cfg.n_max.to(data.device)
    # fix tensor device for n_min on first call and update config parameters
    if isinstance(cfg.n_min, torch.Tensor):
        cfg.n_min = cfg.n_min.to(data.device)

    if cfg.operation == "add":
        return data + torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    elif cfg.operation == "scale":
        return data * (torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min)
    elif cfg.operation == "abs":
        return torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def gaussian_noise(data: torch.Tensor, cfg: noise_cfg.GaussianNoiseCfg) -> torch.Tensor:
    """Applies a gaussian noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for gaussian noise.

    Returns:
        The data modified by the noise parameters provided.
    """
    """对给定的数据集应用高斯噪音。

    参数：
        data: 适用于噪音的未修改数据集。
        cfg: 对于高斯噪音的配置参数。

    返回：
        通过提供噪音参数修改的数据。
    """

    # fix tensor device for mean on first call and update config parameters
    if isinstance(cfg.mean, torch.Tensor):
        cfg.mean = cfg.mean.to(data.device)
    # fix tensor device for std on first call and update config parameters
    if isinstance(cfg.std, torch.Tensor):
        cfg.std = cfg.std.to(data.device)

    if cfg.operation == "add":
        return data + cfg.mean + cfg.std * torch.randn_like(data)
    elif cfg.operation == "scale":
        return data * (cfg.mean + cfg.std * torch.randn_like(data))
    elif cfg.operation == "abs":
        return cfg.mean + cfg.std * torch.randn_like(data)
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


##
# Noise models as classes
##


class NoiseModel:
    """Base class for noise models."""
    """噪音模型的基础类。"""

    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelCfg, num_envs: int, device: str):
        """Initialize the noise model.

        Args:
            noise_model_cfg: The noise configuration to use.
            num_envs: The number of environments.
            device: The device to use for the noise model.
        """
        """启动噪音模型。

        参数：
            noise_model_cfg: 使用的噪音配置。
            num_envs: 环境的数量。
            device: 用于噪音模型的设备。
        """
        self._noise_model_cfg = noise_model_cfg
        self._num_envs = num_envs
        self._device = device

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method can be implemented by derived classes to reset the noise model.
        This is useful when implementing temporal noise models such as random walk.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        """调整噪音模型。

        这种方法可以通过衍生类实现，以重置噪音模型。
        这在实施时代噪音模型时是有用的，例如随机步行。

        参数：
            env_ids: 环境识别器将噪声模型重置为。
                     在 None 中，默认情况下考虑所有环境。
        """
        pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the noise to the data.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        """将噪音应用于数据。

        参数：
            data: 应对噪音的数据。
                  形状是 (num_envs， ...)。

        返回：
            采用噪音的数据。
            它的形状与输入数据相同。
        """
        return self._noise_model_cfg.noise_cfg.func(data, self._noise_model_cfg.noise_cfg)


class NoiseModelWithAdditiveBias(NoiseModel):
    """Noise model with an additive bias.

    The bias term is sampled from a the specified distribution on reset.
    """
    """噪音模型具有添加偏见。

    偏差项在重置时从 a 指定分布中取样。
    """

    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg, num_envs: int, device: str):
        # initialize parent class
        super().__init__(noise_model_cfg, num_envs, device)
        # store the bias noise configuration
        self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
        self._bias = torch.zeros((num_envs, 1), device=self._device)
        self._num_components: int | None = None
        self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method resets the bias term for the specified environments.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        """调整噪音模型。

        这种方法为指定环境重置偏差项。

        参数：
            env_ids: 环境识别器将噪声模型重置为。
                     在 None 中，默认情况下考虑所有环境。
        """
        # resolve the environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the bias term
        self._bias[env_ids] = self._bias_noise_cfg.func(self._bias[env_ids], self._bias_noise_cfg)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply bias noise to the data.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        """将偏差噪音应用于数据。

        参数：
            data: 应对噪音的数据。
                  形状是 (num_envs， ...)。

        返回：
            采用噪音的数据。
            它的形状与输入数据相同。
        """
        # if sample_bias_per_component, on first apply, expand bias to match last dim of data
        if self._sample_bias_per_component and self._num_components is None:
            *_, self._num_components = data.shape
            # expand bias from (num_envs,1) to (num_envs, num_components)
            self._bias = self._bias.repeat(1, self._num_components)
            # now re-sample that expanded bias in-place
            self.reset()
        return super().__call__(data) + self._bias
