# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Noise models specific to deployment tasks."""

from __future__ import annotations
"""特定部署任务的噪音模型。"""

__all__ = ["ResetSampledConstantNoiseModel", "ResetSampledConstantNoiseModelCfg"]

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModel, NoiseModelCfg

if TYPE_CHECKING:
    from isaaclab.utils.noise import NoiseCfg


class ResetSampledConstantNoiseModel(NoiseModel):
    """Noise model that samples noise ONLY during reset and applies it consistently.

    The noise is sampled from the configured distribution ONLY during reset and applied consistently
    until the next reset. Unlike regular noise that generates new random values every step,
    this model maintains the same noise values throughout an episode.

    Note:
        This noise model was used since the noise randimization should only be done at reset time.
        Other noise models(Eg: GaussianNoise) were not used since this randomizes the noise at every time-step.
    """
    """噪音模型在重置时采样噪音ONLY，并一致应用。

    在重置期间，从配置分布 ONLY中采集噪音，并持续应用到下一次重置。
    与每一步都产生新的随机值的常规噪音不同，这个模型在一集中保持相同的噪音值。

    说明：
        这种噪音模型是使用的，因为噪音随机化只应在重置时进行。
        其他噪音模型 ((例如:GaussianNoise) 没有被使用，因为这随机对噪音进行了每一步的随机排序。
    """

    def __init__(self, noise_model_cfg: NoiseModelCfg, num_envs: int, device: str):
        # initialize parent class
        super().__init__(noise_model_cfg, num_envs, device)
        # store the noise configuration
        self._noise_cfg = noise_model_cfg.noise_cfg
        self._sampled_noise = torch.zeros((num_envs, 1), device=self._device)
        self._num_components: int | None = None

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model by sampling NEW noise values.

        This method samples new noise for the specified environments using the configured noise function.
        The sampled noise will remain constant until the next reset.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        """通过采样NEW噪音值重置噪音模型。

        该方法采用配置噪音函数对指定环境进行新的噪音样本。
        在下一次重置之前，采样噪音将保持一致。

        参数：
            env_ids: 环境识别器将噪声模型重置为。
                     在 None 中，默认情况下考虑所有环境。
        """
        # resolve the environment ids
        if env_ids is None:
            env_ids = slice(None)

        # Use the existing noise function to sample new noise
        # Create dummy data to sample from the noise function
        dummy_data = torch.zeros(
            (env_ids.stop - env_ids.start if isinstance(env_ids, slice) else len(env_ids), 1), device=self._device
        )

        # Sample noise using the configured noise function
        sampled_noise = self._noise_model_cfg.noise_cfg.func(dummy_data, self._noise_model_cfg.noise_cfg)

        self._sampled_noise[env_ids] = sampled_noise

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the pre-sampled noise to the data.

        This method applies the noise that was sampled during the last reset.
        No new noise is generated - the same values are used consistently.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        """应用预测采样噪音到数据中。

        该方法适用于最后一次重置时采样的噪音。
        没有新的噪音产生 - 始终使用相同的值。

        参数：
            data: 应对噪音的数据。
                  形状是 (num_envs， ...)。

        返回：
            采用噪音的数据。
            它的形状与输入数据相同。
        """
        # on first apply, expand noise to match last dim of data
        if self._num_components is None:
            *_, self._num_components = data.shape
            # expand noise from (num_envs,1) to (num_envs, num_components)
            self._sampled_noise = self._sampled_noise.repeat(1, self._num_components)

        # apply the noise based on operation
        if self._noise_cfg.operation == "add":
            return data + self._sampled_noise
        elif self._noise_cfg.operation == "scale":
            return data * self._sampled_noise
        elif self._noise_cfg.operation == "abs":
            return self._sampled_noise
        else:
            raise ValueError(f"Unknown operation in noise: {self._noise_cfg.operation}")


@configclass
class ResetSampledConstantNoiseModelCfg(NoiseModelCfg):
    """Configuration for a noise model that samples noise ONLY during reset."""
    """对于在重置过程中采样噪音ONLY的噪音模型的配置。"""

    class_type: type = ResetSampledConstantNoiseModel

    noise_cfg: NoiseCfg = MISSING
    """The noise configuration for the noise.

    Based on this configuration, the noise is sampled at every reset of the noise model.
    """
    """噪音配置。

    根据这种配置，噪音模型的每次重置都会采样噪音。
    """
