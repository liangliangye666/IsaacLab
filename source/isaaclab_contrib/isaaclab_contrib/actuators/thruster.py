# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

from isaaclab_contrib.utils.types import MultiRotorActions

if TYPE_CHECKING:
    from .thruster_cfg import ThrusterCfg


class Thruster:
    """Low-level motor/thruster dynamics with separate rise/fall time constants.

    Integration scheme is Euler or RK4. All internal buffers are shaped (num_envs, num_motors).
    Units: thrust [N], rates [N/s], time [s].
    """
    """低级发动机/推进器动力，有单独的上/下时间常量。

    集成方案是尤勒或RK4。
    所有内部缓冲器均有形状 (num_envs，num_motors)。
    Units: 推力 [N]，速率 [N/s]，时间 [s]。
    """

    computed_thrust: torch.Tensor
    """The computed thrust for the actuator group. Shape is (num_envs, num_thrusters)."""
    """执行器组的计算推力。
    形状是 (num_envs，num_thrusters)。
    """

    applied_thrust: torch.Tensor
    """The applied thrust for the actuator group. Shape is (num_envs, num_thrusters).

    This is the thrust obtained after clipping the :attr:`computed_thrust` based on the
    actuator characteristics.
    """
    """执行器组的应用于推力。
    形状是 (num_envs，num_thrusters)。

    根据动机特性，在裁剪:attr:`computed_thrust`后获得的推力。
    """

    cfg: ThrusterCfg

    def __init__(
        self,
        cfg: ThrusterCfg,
        thruster_names: list[str],
        thruster_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        init_thruster_rps: torch.Tensor,
    ):
        """Construct buffers and sample per-motor parameters.

        Args:
            cfg: Thruster configuration.
            thruster_names: List of thruster names belonging to this group.
            thruster_ids: Slice or tensor of indices into the articulation thruster array.
            num_envs: Number of parallel/vectorized environments.
            device: PyTorch device string or device identifier.
            init_thruster_rps: Initial per-thruster rotations-per-second tensor used when
                the configuration uses RPM-based thrust modelling.
        """
        """构建缓冲器和每发动机参数的样本。

        参数：
            cfg: 推进器配置。
            thruster_names: 属于该组的推进器名称列表。
            thruster_ids: 在关节推进器阵列中切片或索引子。
            num_envs: 平行/向量化环境数量
            device: PyTorch设备字符串或设备标识符。
            init_thruster_rps: 当配置采用RPM基于的推力建模时，使用的每推力初始旋转-每秒紧张器。
        """
        self.cfg = cfg
        self._num_envs = num_envs
        self._device = device
        self._thruster_names = thruster_names
        self._thruster_indices = thruster_ids
        self._init_thruster_rps = init_thruster_rps

        # Range tensors, shaped (num_envs, 2, num_motors); [:,0,:]=min, [:,1,:]=max
        self.num_motors = len(thruster_names)
        self.thrust_r = torch.tensor(cfg.thrust_range).to(self._device)
        self.tau_inc_r = torch.tensor(cfg.tau_inc_range).to(self._device)
        self.tau_dec_r = torch.tensor(cfg.tau_dec_range).to(self._device)

        self.max_rate = torch.tensor(cfg.max_thrust_rate).expand(self._num_envs, self.num_motors).to(self._device)

        self.max_thrust = self.cfg.thrust_range[1]
        self.min_thrust = self.cfg.thrust_range[0]

        # State & randomized per-motor parameters
        self.tau_inc_s = math_utils.sample_uniform(*self.tau_inc_r, (self._num_envs, self.num_motors), self._device)
        self.tau_dec_s = math_utils.sample_uniform(*self.tau_dec_r, (self._num_envs, self.num_motors), self._device)
        self.thrust_const_r = torch.tensor(cfg.thrust_const_range, device=self._device, dtype=torch.float32)
        self.thrust_const = math_utils.sample_uniform(
            *self.thrust_const_r, (self._num_envs, self.num_motors), self._device
        ).clamp(min=1e-6)

        self.curr_thrust = self.thrust_const * (self._init_thruster_rps.to(self._device).float() ** 2)

        # Mixing factor (discrete vs continuous form)
        if self.cfg.use_discrete_approximation:
            self.mixing_factor_function = self.discrete_mixing_factor
        else:
            self.mixing_factor_function = self.continuous_mixing_factor

        # Choose stepping kernel once (avoids per-step branching)
        if self.cfg.integration_scheme == "euler":
            self._step_thrust = self.compute_thrust_with_rpm_time_constant
        elif self.cfg.integration_scheme == "rk4":
            self._step_thrust = self.compute_thrust_with_rpm_time_constant_rk4
        else:
            raise ValueError("integration scheme unknown")

    @property
    def num_thrusters(self) -> int:
        """Number of actuators in the group."""
        """集团中的执行器数量"""
        return len(self._thruster_names)

    @property
    def thruster_names(self) -> list[str]:
        """Articulation's thruster names that are part of the group."""
        """关节的推进器名称是集团的一部分。"""
        return self._thruster_names

    @property
    def thruster_indices(self) -> slice | torch.Tensor:
        """Articulation's thruster indices that are part of the group.

        Note:
            If :obj:`slice(None)` is returned, then the group contains all the thrusters in the articulation.
            We do this to avoid unnecessary indexing of the thrusters for performance reasons.
        """
        """关节的推进索引是集团的一部分。

        说明：
            If :转换到obj:`slice(None)`，然后集团包含了关节中的所有推进器。
            我们这样做是为了避免由于性能原因，
        """
        return self._thruster_indices

    def compute(self, control_action: MultiRotorActions) -> MultiRotorActions:
        """Advance the thruster state one step.

        Applies saturation, chooses rise/fall tau per motor, computes mixing factor,
        and integrates with the selected kernel.

        Args:
            control_action: (num_envs, num_thrusters) commanded per-thruster thrust [N].

        Returns:
            (num_envs, num_thrusters) updated thrust state [N].

        """
        """一步推进推进器状态。

        应用和 tau，选择每个发动机的 tau/tau，计算混合因素，并与所选的内核集成。

        参数：
            control_action: (num_envs，num_thrusters) 命令每驱动器推力 [N]。

        返回：
            (num_envs，num_thrusters) 更新的推力状态 [N]。
        """
        des_thrust = control_action.thrusts
        des_thrust = torch.clamp(des_thrust, *self.thrust_r)

        thrust_decrease_mask = torch.sign(self.curr_thrust) * torch.sign(des_thrust - self.curr_thrust)
        motor_tau = torch.where(thrust_decrease_mask < 0, self.tau_dec_s, self.tau_inc_s)
        mixing = self.mixing_factor_function(motor_tau)

        self.curr_thrust[:] = self._step_thrust(des_thrust, self.curr_thrust, mixing)

        self.computed_thrust = self.curr_thrust
        self.applied_thrust = torch.clamp(self.computed_thrust, self.min_thrust, self.max_thrust)

        control_action.thrusts = self.applied_thrust

        return control_action

    def reset_idx(self, env_ids=None) -> None:
        """Re-sample parameters and reinitialize state.

        Args:
            env_ids: Env indices to reset. If ``None``, resets all envs.
        """
        """再样本参数并重新启动状态。

        参数：
            env_ids: 设置重置的Env索引。
                     如果 ``None``，将所有 envs重置。
        """
        if env_ids is None:
            env_ids = slice(None)

        if isinstance(env_ids, slice):
            num_resets = self._num_envs
        else:
            num_resets = len(env_ids)

        self.tau_inc_s[env_ids] = math_utils.sample_uniform(
            *self.tau_inc_r,
            (num_resets, self.num_motors),
            self._device,
        )
        self.tau_dec_s[env_ids] = math_utils.sample_uniform(
            *self.tau_dec_r,
            (num_resets, self.num_motors),
            self._device,
        )
        self.thrust_const[env_ids] = math_utils.sample_uniform(
            *self.thrust_const_r,
            (num_resets, self.num_motors),
            self._device,
        )
        self.curr_thrust[env_ids] = self.thrust_const[env_ids] * self._init_thruster_rps[env_ids] ** 2

    def reset(self, env_ids: Sequence[int]) -> None:
        """Reset all envs."""
        """设置所有envs。"""
        self.reset_idx(env_ids)

    def motor_model_rate(self, error: torch.Tensor, mixing_factor: torch.Tensor):
        return torch.clamp(mixing_factor * (error), -self.max_rate, self.max_rate)

    def rk4_integration(self, error: torch.Tensor, mixing_factor: torch.Tensor):
        k1 = self.motor_model_rate(error, mixing_factor)
        k2 = self.motor_model_rate(error + 0.5 * self.cfg.dt * k1, mixing_factor)
        k3 = self.motor_model_rate(error + 0.5 * self.cfg.dt * k2, mixing_factor)
        k4 = self.motor_model_rate(error + self.cfg.dt * k3, mixing_factor)
        return (self.cfg.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def discrete_mixing_factor(self, time_constant: torch.Tensor):
        return 1.0 / (self.cfg.dt + time_constant)

    def continuous_mixing_factor(self, time_constant: torch.Tensor):
        return 1.0 / time_constant

    def compute_thrust_with_rpm_time_constant(
        self,
        des_thrust: torch.Tensor,
        curr_thrust: torch.Tensor,
        mixing_factor: torch.Tensor,
    ):
        # Avoid negative or NaN values inside sqrt by clamping the ratio to >= 0.
        current_ratio = torch.clamp(curr_thrust / self.thrust_const, min=0.0)
        desired_ratio = torch.clamp(des_thrust / self.thrust_const, min=0.0)
        current_rpm = torch.sqrt(current_ratio)
        desired_rpm = torch.sqrt(desired_ratio)
        rpm_error = desired_rpm - current_rpm
        current_rpm += self.motor_model_rate(rpm_error, mixing_factor) * self.cfg.dt
        return self.thrust_const * current_rpm**2

    def compute_thrust_with_rpm_time_constant_rk4(
        self,
        des_thrust: torch.Tensor,
        curr_thrust: torch.Tensor,
        mixing_factor: torch.Tensor,
    ) -> torch.Tensor:
        current_ratio = torch.clamp(curr_thrust / self.thrust_const, min=0.0)
        desired_ratio = torch.clamp(des_thrust / self.thrust_const, min=0.0)
        current_rpm = torch.sqrt(current_ratio)
        desired_rpm = torch.sqrt(desired_ratio)
        rpm_error = desired_rpm - current_rpm
        current_rpm += self.rk4_integration(rpm_error, mixing_factor)
        return self.thrust_const * current_rpm**2
