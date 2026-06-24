# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class AgileTeacherPolicyObservationsCfg(ObsGroup):
    """Observation specifications for the Agile lower body policy.

    Note: This configuration defines only part of the observation input to the Agile lower body policy.
    The lower body command portion is appended to the observation tensor in the action term, as that
    is where the environment has access to those commands.
    """
    """对敏捷下体策略的观测规范。

    Note: 这种配置只定义了对敏捷下体策略的观测输入的一部分。
    在动作项中，下部部部的命令部分附加到观测张量中，因为这是环境可以访问这些命令的地方。
    """

    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        scale=1.0,
    )

    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                    ".*_hip_.*_joint",
                    ".*_knee_joint",
                    ".*_ankle_.*_joint",
                    "waist_.*_joint",
                ],
            ),
        },
    )

    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        scale=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                    ".*_hip_.*_joint",
                    ".*_knee_joint",
                    ".*_ankle_.*_joint",
                    "waist_.*_joint",
                ],
            ),
        },
    )

    actions = ObsTerm(
        func=mdp.last_action,
        scale=1.0,
        params={
            "action_name": "lower_body_joint_pos",
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
