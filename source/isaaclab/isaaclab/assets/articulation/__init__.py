# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for rigid articulated assets."""
"""固体关节资产子模块"""

from .articulation import Articulation
from .articulation_cfg import ArticulationCfg
from .articulation_data import ArticulationData

'''
articulation_data.py 属性全景图         需重点关注的函数
总览：五层结构
    ┌─────────────────────────────────────────────────────────────┐
    │                    ArticulationData                          │
    │                                                             │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ 第0层：生命周期方法 (2个)                            │  │
    │  │   __init__(), update(dt)                             │  │
    │  └──────────────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ 第1层：根状态 (root_*) —— ⭐⭐⭐ 最重要              │  │
    │  │   root_link_pose_w, root_link_vel_w,                  │  │
    │  │   root_com_pose_w,  root_com_vel_w,                   │  │
    │  │   root_state_w, root_link_state_w, root_com_state_w   │  │
    │  └──────────────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ 第2层：全身连杆状态 (body_*) —— ⭐⭐ 重要            │  │
    │  │   body_link_pose_w, body_link_vel_w,                  │  │
    │  │   body_com_pose_w,  body_com_vel_w,                   │  │
    │  │   body_state_w, body_link_state_w, body_com_state_w   │  │
    │  │   body_com_acc_w                                      │  │
    │  └──────────────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ 第3层：关节状态 (joint_*) —— ⭐⭐ 重要               │  │
    │  │   joint_pos, joint_vel, joint_acc                     │  │
    │  └──────────────────────────────────────────────────────┘  │
    │    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ 第4层：便捷派生量 —— ⭐ 按需关注                     │  │
    │  │   projected_gravity_b, heading_w,                     │  │
    │  │   root_link_lin_vel_b, root_com_ang_vel_w, ...        │  │
    │  └──────────────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ 第5层：向后兼容别名 —— ❌ 可跳过                     │  │
    │  │   root_pose_w → root_link_pose_w                       │  │
    │  │   root_vel_w  → root_com_vel_w                        │  │
    │  │   ... (约15个别名)                                     │  │
    │  └──────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
⭐⭐⭐ 第1层：根状态（必须掌握，7个属性）
    这是 RL 训练中最常用的部分。之前已经详细讲过其中三个：
        属性	            形状	    含义	                                        用途频率
        root_link_pose_w	(N, 7)	    根连杆位姿 [pos, quat]	                        极高
        root_link_vel_w	    (N, 6)	    根连杆速度 [lin, ang]（需 v_COM→v_link 变换）	    中
        root_com_pose_w	    (N, 7)	    根质心位姿（= link_pose + COM 偏移）	            中
        root_com_vel_w	    (N, 6)	    根质心速度（PhysX 原生输出，最快）	                高
        root_state_w	    (N, 13)	    混合组合：link_pose + com_vel	                极高
        root_link_state_w	(N, 13)	    纯 link 组合：link_pose + link_vel	            中
        root_com_state_w	(N, 13)	    纯 COM 组合：com_pose + com_vel	                低
    记忆口诀：
        想要最快的速度 → root_com_vel_w
        想要位置+速度一起 → root_state_w（最常用）
        想要严格物理一致性 → root_link_state_w
⭐⭐ 第2层：全身连杆状态（需要了解，10个属性）
    当你需要知道机器人脚在哪、膝盖在哪时用这些。比根状态多一个 num_bodies 维度。
        属性	                形状	    含义
        body_link_pose_w	(N, B, 7)	    所有连杆的位姿
        body_link_vel_w	    (N, B, 6)	    所有连杆的速度（link frame）
        body_com_pose_w	    (N, B, 7)	    所有连杆的 COM 位姿
        body_com_vel_w	    (N, B, 6)	    所有连杆的 COM 速度（PhysX 原生）
        body_state_w	    (N, B, 13)	    混合组合：link_pose + com_vel
        body_link_state_w	(N, B, 13)	    纯 link 组合
        body_com_state_w	(N, B, 13)	    纯 COM 组合
        body_com_acc_w	    (N, B, 6)	    所有连杆的 COM 加速度
        body_com_pose_b	    (N, B, 7)	    COM 在各自连杆本体系下的位姿（常量）
        body_incoming_joint_wrench_b	(N, B, 6)	关节传递到连杆的力/力矩
⭐⭐ 第3层：关节状态（需要了解，3个属性）
        属性	    形状	    含义
        joint_pos	(N, J)	关节角度（rad）
        joint_vel	(N, J)	关节角速度（rad/s）
        joint_acc	(N, J)	关节角加速度（差分计算，rad/s²）
    这三个在观测空间和奖励计算中频繁出现，但通常通过 manager_based_rl_env 的观测管理器自动处理，你不需要手动访问。

⭐ 第4层：便捷派生量（按需关注，约20个属性）
    这些是上面核心属性的"切片"或"坐标系转换"，比如：
        主干属性                     切片属性（只取部分数据）
        ──────────────────────────────────────────────────
        root_link_pose_w  ──→  root_link_pos_w       [N,3]  只取位置
                        ──→  root_link_quat_w      [N,4]  只取姿态

        root_link_vel_w   ──→  root_link_lin_vel_w   [N,3]  只取线速度
                        ──→  root_link_ang_vel_w   [N,3]  只取角速度

        坐标系转换（世界系 → 机器人本体系）:
        root_link_vel_w   ──→  root_link_lin_vel_b   [N,3]  本体系线速度
                        ──→  root_link_ang_vel_b   [N,3]  本体系角速度
        root_com_vel_w    ──→  root_com_lin_vel_b    [N,3]
                        ──→  root_com_ang_vel_b    [N,3]
    需要重点关注的2个派生量：
        属性	                形状	    含义
        projected_gravity_b	    (N, 3)	重力在机器人本体系的方向（判断倾斜程度）
        heading_w	            (N,)	机器人在世界系中的朝向角（弧度）
❌ 第5层：向后兼容别名（直接跳过，约15个属性）
    这些是旧 API 名的别名，指向新名字。比如 root_pose_w → root_link_pose_w，root_vel_w → root_com_vel_w。
    永远不要用这些别名，它们只是为了兼容老代码保留的。
'''