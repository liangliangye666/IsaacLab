# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations
"""可用于启用奖励函数的共同函数。

函数可以传递到:class:`isaaclab.managers.RewardTermCfg`对象，包括函数引入的奖励。
"""

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
General.
"""
"""总理。
"""


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    """为了活着而得到奖励。"""
    return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    """惩罚那些不符合回合时间的终止回合。"""
    return env.termination_manager.terminated.float()


class is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """
    """针对不符合回合时间的特定项的终止。

    参数如下:

    * attr:`term_keys`: 终止项。 这可能是字符串，字符串列表或正则表达式.默认是 ".*" 处罚所有终止。

    奖励计算为终止条件的总和，而不是事件时间。
    这意味着，如果剧情因剧情时间间断而结束，奖励为0。
    否则，
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        return (reset_buf * (~env.termination_manager.time_outs)).float()


"""
Root penalties.
"""
"""基本罚款。
"""


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    """使用L2平方内核来惩罚z轴基线性速度。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    """使用L2平方内核来惩罚xy轴基角速度。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    """使用L2平方内核进行非平的基准定向。

    这通过对预测重力向量的xy组件进行惩罚来计算。
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    """通过L2平方内核来惩罚其目标的资产高度。

    说明：
        对于平面地形，目标高度在世界框架内。
        对于粗的地形，传感器读数可以调整目标高度，以考虑到地形。
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    """通过L2核来惩罚身体的线性加速。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)


"""
Joint penalties.
"""
"""共同处罚。
"""


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques
        contribute to the term.
    """
    """通过使用L2平方核在关节上施加的关节扭矩。

    .. 说明::
        只有在:attr:`asset_cfg.joint_ids`中配置的关节将有它们的关节扭矩为这个项做出贡献。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    """通过L1核使用关节速度进行惩罚。"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities
        contribute to the term.
    """
    """使用L2平方核来将关节速度处罚。

    .. 说明::
        只有在:attr:`asset_cfg.joint_ids`中配置的关节，它们的关节速度才能为这个项做出贡献。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations
        contribute to the term.
    """
    """使用L2平方核来将关节加速处罚。

    .. 说明::
        只有在:attr:`asset_cfg.joint_ids`中配置的关节，它们的关节加速将有助于这个项。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    """处罚违反默认的关节位置。"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    """如果它们超越柔软的边界，

    这将被计算为关节位置和软极之间的差异的绝对值的总和。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    """如果它们超越柔软的极限，

    这被计算为关节速度和软极之间的差异的绝对值的总和。

    参数：
        soft_ratio: 使用的软极限的比例。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


"""
Action penalties.
"""
"""动作处罚。
"""


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    """如果扭矩超越了限量，应处罚。

    这将被计算为应用扭矩和限制之间的差异的绝对值的总和。

    .. 谨慎::
        目前，这仅适用于明确的动力器，因为我们手动计算应用的扭矩。
        对于隐含动机，我们目前无法从物理引擎中检索应用的扭矩。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    """使用L2平方内核来惩罚操作的变化速度。"""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    """使用L2平方内核进行惩罚。"""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


"""
Contact sensor.
"""
"""接触传感器。
"""


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    """处罚不必要的联系人，因为违规行为超过门。"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def desired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize if none of the desired contacts are present."""
    """如果没有所需的接触者，请惩罚。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    zero_contact = (~contacts).all(dim=1)
    return 1.0 * zero_contact


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    """根据网络接触力违规的数量，惩罚接触力。"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1)


"""
Velocity-tracking rewards.
"""
"""追踪速度的奖励。
"""


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    """使用指数内核的线性速度命令 (xy轴) 的奖励跟踪。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    """使用指数内核的角速度命令 (yaw) 的奖励跟踪。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)
