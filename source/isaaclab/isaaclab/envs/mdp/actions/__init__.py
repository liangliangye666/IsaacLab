# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Various action terms that can be used in the environment."""
"""在环境中可以使用的各种动作项。"""

from .actions_cfg import *
from .binary_joint_actions import *
from .joint_actions import *
from .joint_actions_to_limits import *
from .non_holonomic_actions import *
from .surface_gripper_actions import *

'''
动作系统的配置与实现分离
    也就是说，整个动作系统由配置cfg和实现组成，每个实现对应了一个配置cfg,其中 actions_cfg.py 就包含了多个实现所对应的配置cfg
        actions_cfg.py:    "契约" — 类名、参数、默认值、class_type 指向
        joint_actions.py:  "执行" — __init__, process_actions, apply_actions

    actions/
        │
        ├── actions_cfg.py              ← [核心配置中心] 所有核心动作的配置类
        │
        ├── joint_actions.py            ← 关节空间动作的实现
        ├── joint_actions_to_limits.py  ← 关节限位映射动作的实现
        ├── binary_joint_actions.py     ← 二值(开/关)关节动作的实现
        ├── non_holonomic_actions.py    ← 移动底盘动作的实现
        ├── task_space_actions.py       ← 任务空间(IK/OSC)动作的实现
        ├── surface_gripper_actions.py  ← 表面夹具动作的实现
        │
        ├── rmpflow_actions_cfg.py      ← RMPFlow 配置(独立扩展)
        ├── rmpflow_task_space_actions.py ← RMPFlow 实现
        ├── pink_actions_cfg.py         ← Pink IK 配置(独立扩展)
        ├── pink_task_space_actions.py  ← Pink IK 实现
        │
        └── __init__.py                 ← 导出核心模块(不含 RMPFlow/Pink)

    actions_cfg.py — 核心配置中心（11 个 Config 类）
        ActionTermCfg                          ← 来自 action_manager.py 的基类
        │
        ├── JointActionCfg                    ← 关节动作基类 (scale + offset + clip)
        │   ├── JointPositionActionCfg        → joint_actions.JointPositionAction
        │   ├── RelativeJointPositionActionCfg → joint_actions.RelativeJointPositionAction
        │   ├── JointVelocityActionCfg        → joint_actions.JointVelocityAction
        │   └── JointEffortActionCfg          → joint_actions.JointEffortAction
        │
        ├── JointPositionToLimitsActionCfg    → joint_actions_to_limits.JointPositionToLimitsAction
        │   └── EMAJointPositionToLimitsActionCfg → ...EMAJointPositionToLimitsAction
        │
        ├── BinaryJointActionCfg              ← 夹具基类 (open/close_command_expr)
        │   ├── BinaryJointPositionActionCfg  → binary_joint_actions.BinaryJointPositionAction
        │   └── BinaryJointVelocityActionCfg  → binary_joint_actions.BinaryJointVelocityAction
        │
        ├── AbsBinaryJointPositionActionCfg   → binary_joint_actions.AbsBinaryJointPositionAction
        │
        ├── NonHolonomicActionCfg             → non_holonomic_actions.NonHolonomicAction
        │
        ├── DifferentialInverseKinematicsActionCfg → task_space_actions.DifferentialInverseKinematicsAction
        ├── OperationalSpaceControllerActionCfg    → task_space_actions.OperationalSpaceControllerAction
        │
        └── SurfaceGripperBinaryActionCfg     → surface_gripper_actions.SurfaceGripperBinaryAction
    五大动作类别的功能概览（实现类）
        ① 关节空间动作（joint_actions.py）— 最基础的底层动作
                Config	                        策略输出含义	        物理层效果
                JointPositionAction	            关节目标角度	        set_joint_position_target(), PD 控制器驱动
                RelativeJointPositionAction	    相对当前角度的增量	    set_joint_position_target(joint_pos + delta)
                JointVelocityAction	            关节目标角速度	        set_joint_velocity_target()
                JointEffortAction	            关节力矩	            set_joint_effort_target()
            JointAction 基类的核心公式：
                action = offset + scale × input_action
            offset 和 scale 可以按 joint 名称的正则表达式分别设置——如 scale={".*shoulder.*": 0.5, ".*elbow.*": 1.0}，让策略网络对不同关节有不同的动作灵敏度。
        
        ② 关节限位映射（joint_actions_to_limits.py）— 归一化动作
                # 策略输出 [-1, 1] → 映射到 [joint_lower_limit, joint_upper_limit]
                # 如关节限位 [-2.0, 2.0]:
                #   策略输出 0.5 → 实际目标 = 1.0
                #   策略输出 -1.0 → 实际目标 = -2.0
            EMAJointPositionToLimitsAction 是增强版——在限位映射的基础上加指数移动平均平滑，减少抖动：
                q_target_t = alpha × q_raw + (1 - alpha) × q_target_{t-1}
                alpha=1.0 意味着不做平滑，直接应用。

        ③ 二值关节动作（binary_joint_actions.py）— 夹具专用
            策略输出是二值信号（开/关），而非连续值：
                # BinaryJointPositionAction: 策略输出 > 0 → 张开配置, < 0 → 闭合配置
                # AbsBinaryJointPositionAction: 策略输出 > threshold(0.5) → 张开, < threshold → 闭合
            两种的区别：
                BinaryJointPositionAction 用策略值的正负号判断（更简单），AbsBinaryJointPositionAction 用绝对值和阈值比较（更鲁棒，适合连续策略网络输出）。
        
        ④ 任务空间动作（task_space_actions.py）— IK 驱动的末端位姿控制
            策略输出是末端执行器的目标位姿（6 或 7 维），内部通过逆运动学（IK）转为关节目标：
                策略输出: [dx, dy, dz, dqw, dqx, dqy, dqz]  ← 末端位姿增量
                    process_actions():
                        IK 求解: Δq = J^T (J J^T + λ²I)^(-1) Δx
                        → set_joint_position_target(q_current + Δq)
                Config	                                控制器	                    适用场景
                DifferentialInverseKinematicsAction	    差分 IK（数值雅可比）	    标准机械臂末端控制
                OperationalSpaceControllerAction	    操作空间控制器	            力/阻抗控制场景

        ⑤ 非完整约束动作（non_holonomic_actions.py）— 移动底盘专用
            策略输出是 2 维向量 [v_x, ω_z]（前进速度 + 转弯速率），转为底盘的虚拟关节速度：
                策略输出: [v_x, ω_z]          ← 2 维

                物理映射:
                    q̇_0 = v_x * cos(θ)      ← x 方向虚拟棱柱关节
                    q̇_1 = v_x * sin(θ)      ← y 方向虚拟棱柱关节
                    q̇_2 = ω_z               ← z 方向虚拟旋转关节
            移动底盘被建模为三个虚拟关节（x 平移、y 平移、z 旋转），而非真实轮子——这样避免复杂的轮地摩擦仿真。

        ⑥ 表面夹具动作（surface_gripper_actions.py）
            和关节夹具不同，表面夹具用不同的 PhysX 接口（SurfaceGripper 而非 Articulation）。三段式逻辑：
                策略输出:
                    [-1.0, -0.3] → 张开
                    [-0.3,  0.3] → 无动作（空闲）
                    [ 0.3,  1.0] → 闭合

    RMPFlow 和 Pink IK — 独立扩展
        两者都在主 __init__.py 之外，因为它们作为可选扩展，不在核心导出列表中。
                        RMPFlow	                                                                        Pink IK
            控制器来源	 NVIDIA RMPFlow	                                                                Stéphane Caron 的 Pink 库
            特点	    黎曼运动策略，无碰撞路径	                                                        基于任务优先级的二次规划 IK
            配置类	    RMPFlowActionCfg	                                                            PinkInverseKinematicsActionCfg
            使用方式	from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg	    同理
        __init__.py 选择性导出
            RMPFlow 和 Pink IK 不进 __init__.py——用户需要时才显式 import，避免强迫安装可选依赖。这是 Python 包管理中"可选依赖"的最佳实践。
    
    【建议】确认后续是否会真实采用？？？
        轮式双足的常见用法：底盘用 NonHolonomicAction，策略学的是"往前走多快、转多大弯"，不直接控制轮子转速。
        对于轮式双足的腿关节：优先考虑 JointPositionToLimitsAction——策略只需输出 [-1, 1]，框架自动映射到关节真实限位。
                            这比直接输出角度值更鲁棒（策略不会输出超限值）。
'''