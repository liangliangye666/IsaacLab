# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Various command terms that can be used in the environment."""
"""在环境中可以使用的各种命令项。"""

from .commands_cfg import (
    NormalVelocityCommandCfg,
    NullCommandCfg,
    TerrainBasedPose2dCommandCfg,
    UniformPose2dCommandCfg,
    UniformPoseCommandCfg,
    UniformVelocityCommandCfg,
)
from .null_command import NullCommand
from .pose_2d_command import TerrainBasedPose2dCommand, UniformPose2dCommand
from .pose_command import UniformPoseCommand
from .velocity_command import NormalVelocityCommand, UniformVelocityCommand

'''
基类 → 配置 → 实现"三层模式。
    文件关系一览
        commands/
        ├── __init__.py              ← 对外导出接口
        ├── commands_cfg.py           ← 所有 Command 的配置类（Config）
        ├── null_command.py           ← NullCommand（空命令，占位用）
        ├── velocity_command.py       ← 速度命令（行走/奔跑任务）
        │     ├── UniformVelocityCommand
        │     └── NormalVelocityCommand (继承 UniformVelocityCommand)
        ├── pose_command.py           ← 3D 位姿命令（机械臂操作任务）
        │     └── UniformPoseCommand
        └── pose_2d_command.py        ← 2D 位置+朝向命令（地面导航任务）
            ├── UniformPose2dCommand
            └── TerrainBasedPose2dCommand (继承 UniformPose2dCommand)

    继承关系
        命令类（实现）
                            CommandTerm (isaaclab.managers)
                            │   提供: _resample_command, _update_command, _update_metrics
                            │         _set_debug_vis_impl, _debug_vis_callback
                            │
                    ┌─────────┼────────────┬──────────────┐
                    │         │            │              │
            NullCommand  UniformPoseCommand  UniformPose2dCommand  UniformVelocityCommand
                            (3D位姿)           (2D地面目标)         (速度指令)
                                                │                      │
                                                │                      │
                                        TerrainBasedPose2dCommand   NormalVelocityCommand
                                        (地形感知的2D目标)          (正态分布速度)
        配置类（Config）
                            CommandTermCfg
                                    │
                    ┌───────────────┼───────────────┬───────────────┐
                    │               │               │               │
            NullCommandCfg  UniformPoseCommandCfg UniformPose2dCommandCfg  UniformVelocityCommandCfg
                                                                                │
                                                                                │
                                                                        NormalVelocityCommandCfg
                                                                        TerrainBasedPose2dCommandCfg
        每个命令类型的作用与使用场景
            命令类型	                策略要做什么	            命令内容	                            典型任务
            NullCommand	                不需要命令	                空（访问会抛 RuntimeError）	            倒立摆平衡、纯稳定任务
            UniformVelocityCommand	    以指定速度移动	            [vx, vy, ωz] 机体坐标系	                四足/人形机器人行走
            NormalVelocityCommand	    以正态分布采样的速度移动	  [vx, vy, ωz] 正态分布	                更自然的行走速度分布
            UniformPoseCommand	        末端执行器到达某个位姿	      [x, y, z, qw, qx, qy, qz] 基座系	    机械臂抓取、灵巧手操作
            UniformPose2dCommand	    走到地面上某个点	        [x, y, z, heading] 世界系 → 基座系	    盲犬导盲、仓库机器人导航
            TerrainBasedPose2dCommand	走到地形上的有效位置	    [x, y, z, heading] 来自地形信息	        越野导航、非平坦地形
        命令生成的生命周期（共同流程）
            每种命令都遵循 CommandTerm 基类定义的同一套生命周期。
            这是 Isaac Lab 中 模板方法模式 的又一次应用——父类 CommandTerm 定义流程框架，子类覆写具体步骤：
                时间推进 dt
                    │
                    ▼
                CommandManager 调度:
                    │
                    ├── compute(dt)                     ← 父类统一入口
                    │     │
                    │     ├── 检查是否需要重采样?      
                    │     │   是 → _resample_command(env_ids)  ← 子类覆写（核心）
                    │     │         │
                    │     │         └── 重新随机生成命令值（速度/位置/朝向）
                    │     │
                    │     └── _update_command()         ← 子类覆写（后处理）
                    │           │
                    │           └── 后处理（如: 站立环境速度清零、heading→角速度转换）
                    │
                    └── _update_metrics()              ← 子类覆写（追踪误差）
                        │
                        └── 计算跟踪误差，存入 self.metrics 字典
        各命令的独特机制
            ① UniformVelocityCommand — 速度命令（用途最广）
                观测空间: [vx_des, vy_des, ωz_des]  在机器人机体坐标系

                特殊机制:
                    heading_command=True:
                        角速度不由随机采样，而是根据 heading 误差计算:
                        ωz = stiffness × wrap_to_pi(heading_target - heading_current)
                        类似于一个 P 控制器，让机器人自动转向目标方向

                rel_standing_envs:
                    有一定比例的环境速度命令 = [0, 0, 0]（训练机器人学会静止站立）
            ② NormalVelocityCommand — 正态分布速度
                继承 UniformVelocityCommand，差异在 _resample_command:

                Uniform:     vx ~ U(min, max)                ← 均匀分布
                Normal:      vx ~ N(mean, std) × (±1 随机)   ← 正态分布，随机翻转符号

                多了 zero_prob 参数:
                    每个速度分量单独有概率被设为 0
                    例如: zero_prob=[0.3, 0.3, 0.5]
                        → 30% 的环境 vx=0, 30% 的环境 vy=0, 50% 的环境 ωz=0
            ③ UniformPoseCommand — 3D 位姿命令（机械臂）
                观测空间: [x, y, z, qw, qx, qy, qz]  在机器人基座坐标系

                命令在基座系生成（不是世界系）——这意味着:
                    基座转动时，目标位姿跟着转
                    策略需要学习的是"相对于基座"的运动

                _update_metrics 中将命令从基座系转换到世界系，再计算误差
            ④ UniformPose2dCommand — 2D 地面导航
                观测空间: [Δx, Δy, Δz, Δheading]  在机器人基座坐标系

                _update_command 中将世界系目标转为基座系相对量:
                    pos_command_b = quat_apply_inverse(yaw_quat(root_quat_w), target_vec)
                    → 策略看到的是"目标在我前方 3 米，偏左 1 米"

                simple_heading: 朝向是自动"看向目标"还是随机采样
                    自动: heading = atan2(target_y - robot_y, target_x - robot_x)
                    随机: heading ~ U(min, max)
            ⑤ TerrainBasedPose2dCommand — 地形感知导航
                继承 UniformPose2dCommand，但位置不从均匀分布采样，
                而是从地面上预先计算好的"平坦区域"中随机选取:
                    self.terrain.flat_patches["target"]  ← 地形预计算的可行走区域坐标
                    形状: [terrain_level, terrain_type, num_patches, 3]

                采样:
                    ids = randint(0, num_patches)
                    target = flat_patches[terrain_level, terrain_type, ids]
                    target.z += default_root_height  ← 加上机器人站立高度
'''