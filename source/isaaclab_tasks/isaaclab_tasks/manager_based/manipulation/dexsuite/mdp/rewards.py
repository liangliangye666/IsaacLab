# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    """使用L2平方内核来惩罚操作的变化速度。"""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-1000, 1000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    """使用L2平方内核进行惩罚。"""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance.

    The reward is close to 1 when the maximum distance between the object and any end-effector body is small.
    """
    """奖励使用端效应器距离的 tan核达到对象。

    如果对象和任何最终效应体之间的最大距离很小，则奖励接近1。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    object_pos = object.data.root_pos_w
    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    return 1 - torch.tanh(object_ee_distance / std)


def contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    """处罚不必要的联系人，因为违规行为超过门。"""

    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_link_3_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["index_link_3_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["middle_link_3_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["ring_link_3_object_s"]
    # check if contact force is above threshold
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    good_contact_cond1 = (thumb_contact_mag > threshold) & (
        (index_contact_mag > threshold) | (middle_contact_mag > threshold) | (ring_contact_mag > threshold)
    )

    return good_contact_cond1


def success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_std: float,
    rot_std: float | None = None,
) -> torch.Tensor:
    """Reward success by comparing commanded pose to the object pose using tanh kernels on error."""
    """通过使用tanh内核对错误进行命令姿势与对象姿势进行比较来奖励成功。"""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3], command[:, 3:7]
    )
    pos_err, rot_err = compute_pose_error(des_pos_w, des_quat_w, object.data.root_pos_w, object.data.root_quat_w)
    pos_dist = torch.norm(pos_err, dim=1)
    if not rot_std:
        # square is not necessary but this help to keep the final value between having rot_std or not roughly the same
        return (1 - torch.tanh(pos_dist / pos_std)) ** 2
    rot_dist = torch.norm(rot_err, dim=1)
    return (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by contact presence."""
    """奖励跟踪命令位置使用tanh核，通过接触存在。"""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * contacts(env, 1.0).float()


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded orientation using tanh kernel, gated by contact presence."""
    """通过TANH内核进行命令导向的奖励跟踪，通过接触存在被关闭。"""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_quat_w)

    return (1 - torch.tanh(quat_distance / std)) * contacts(env, 1.0).float()
