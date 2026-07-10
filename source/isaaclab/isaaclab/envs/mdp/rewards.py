# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""
"""可用于启用奖励函数的共同函数。

函数可以传递到:class:`isaaclab.managers.RewardTermCfg`对象，包括函数引入的奖励。
"""

from __future__ import annotations

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
'''
表达式	                            含义
termination_manager.terminated	    [N] bool，True = 该环境已终止（摔倒了/出界了）
~terminated	                        取反，False→True（还活着），True→False（已死）
.float()	                        bool→float，1.0 = 活着，0.0 = 死了
'''


'''
惩罚非超时导致的终止（摔倒、出界、自碰撞等）。通常配合负权重使用：摔倒扣分。
'''
def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    """惩罚那些不符合回合时间的终止回合。"""
    return env.termination_manager.terminated.float()   # → _terminated_buf（不含 time_out）
'''
关键概念：terminated ≠ time_out
    TerminationManager 区分两种终止：
        信号	                含义	                示例
        truncated (time_out)	episode 到最大长度	    20 秒到了，正常结束
        terminated	            真正失败了	            摔倒、出界、自碰撞
'''
'''
示例
    @configclass
    class RewardsCfg:
        alive = RewardTermCfg(
            func=rewards.is_alive,
            weight=1.0,           # 活着：每步 +1
        )
        terminated = RewardTermCfg(
            func=rewards.is_terminated,
            weight=-100.0,        # 摔倒：一次扣 100！
        )
    # 总回报 = 活着步数 × 1 - 摔倒 × 100
'''


'''
is_terminated 的增强版——可以选择性惩罚特定的终止条件，而不是一刀切惩罚所有终止。
'''
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

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv): # 预处理匹配的终止项
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)    # find_terms 用正则匹配 TerminationManager 中注册的终止项名称，返回匹配到的名称列表。
        '''
        .get() — 字典的安全取值
            cfg.params 是一个 Python 字典，get() 是字典的方法：
                d.get(key, default)
                    情况	    行为
                    key 存在	返回对应的值
                    key 不存在	返回 default（不报错）
            对比 [] 直接取值
                # 方式 1: 直接取值 — key 不存在时崩溃
                cfg.params["term_keys"]          # KeyError 如果没配这个参数

                # 方式 2: get() — key 不存在时返回默认值
                cfg.params.get("term_keys", ".*")  # 没配 → 返回 ".*"，不崩溃
        '''

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:  # 累计特定终止项的值
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)
            # 如果同时触发两个终止 → 值为 2 → 惩罚加倍

        return (reset_buf * (~env.termination_manager.time_outs)).float()
        #             ↑ 累加触发次数    ↑ 排除超时终止
        # ~time_outs 的过滤： 如果 episode 是因为超时结束的，time_outs=True，取反后乘 0 → 不惩罚。
'''
使用示例
    @configclass
    class RewardsCfg:
        """只惩罚摔倒，不管超时和出界"""

        fell_down = RewardTermCfg(
            func=rewards.is_terminated_term,
            weight=-50.0,
            params={"term_keys": "bad_orientation"},  # 只惩罚这一种终止
        )
'''


"""
Root penalties.
"""
"""基本罚款。
"""


'''
惩罚机器人基座的垂直方向速度（Z 轴）。
    机器人往前走的垂直速度应该趋近于 0——有垂直速度意味着在上下弹跳、消耗多余能量。
数学公式
    reward = -(vz)²     # 配合负权重使用
'''
def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    """使用L2平方内核来惩罚z轴基线性速度。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])
'''
为什么用平方（L2）？
    大的速度偏差惩罚更重。小幅弹跳（0.1 m/s → 惩罚 0.01）几乎不计较，大幅弹跳（0.5 m/s → 惩罚 0.25）会严重扣分。
'''


'''
惩罚机器人上下摆动和左右摇摆（roll + pitch 旋转），只允许绕 Z 轴原地转圈（yaw）。
数学公式
    # asset.data.root_ang_vel_b = [ωx, ωy, ωz]  (roll, pitch, yaw)
    return ωx² + ωy²    # 只惩罚 roll 和 pitch，不管 yaw
'''
def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    """使用L2平方内核来惩罚xy轴基角速度。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


'''
惩罚机器人身体倾斜——机器人应该保持直立。利用之前学过的 projected_gravity_b 来偷懒地衡量倾斜程度。
(gx²+gy²)
'''
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
'''
数学原理
    机器人直立:    projected_gravity = [0, 0, -1]
    → XY 分量 = [0, 0]  →  penalty = 0² + 0² = 0 ✅

    机器人前倾30°: projected_gravity = [0.5, 0, -0.87]
    → XY 分量 = [0.5, 0]  →  penalty = 0.5² + 0² = 0.25 ⚠️

    机器人侧躺:    projected_gravity = [0, 1, 0]
    → XY 分量 = [0, 1]  →  penalty = 0² + 1² = 1.0 ❌
优雅之处：
    不用算角度（欧拉角），直接用 projected_gravity_b[:, :2] 的 L2 范数来衡量倾斜——两行代码实现了一个精确的"直立度"奖励。
'''


'''
惩罚机器人偏离目标高度。
    在平地上，目标高度是固定值；在崎岖地形上，通过RayCaster 传感器动态调整——机器人踩在高地上和踩在洼地里应有不同的"正常高度"。
    (h - h_target)²
'''
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
'''
    # 模式 A: 平坦地形 — 固定目标高度
    adjusted_target = target_height         # 如 0.5m

    # 模式 B: 崎岖地形 — 传感器修正
    adjusted_target = target_height + mean(ray_hits_z)
    #                  ↑ 预设高度        ↑ 地形起伏修正
为什么崎岖地形需要传感器修正？
    平地:  目标高度 = 0.5m
        地面高度   = 0.0m
        robot 实际高度 = 0.48m → 误差 0.02m ✅ 正常

    崎岖: 目标高度 = 0.5m
        地面高度   = 0.2m（站在小坡上）
        robot 实际高度 = 0.72m → 误差 0.22m ❌ 但机器人站得很稳！

        修正后: 目标高度 = 0.5 + 0.2 = 0.7m
        robot 实际高度 = 0.72m → 误差 0.02m ✅ 正确！
使用示例
    @configclass
    class RewardsCfg:
        # 平坦地形:
        flat_height = RewardTermCfg(
            func=rewards.base_height_l2,
            weight=-50.0,
            params={"target_height": 0.5},  # sensor_cfg 不填 → None → 固定模式
        )

        # 崎岖地形:
        rough_height = RewardTermCfg(
            func=rewards.base_height_l2,
            weight=-50.0,
            params={
                "target_height": 0.5,
                "sensor_cfg": SceneEntityCfg("height_scanner"),  # 用传感器修正
            },
        )
'''


'''
惩罚所有身体的加速度——鼓励平滑的全身运动。
    和之前只惩罚根状态不同，这里对所有连杆都做惩罚。
公式
    # body_lin_acc_w: [N, B, 3]  — 每个身体的 XYZ 线加速度
    for each body:
        ‖a_body‖ = √(ax² + ay² + az²)     # 该身体的加速度大小

    penalty = Σ ‖a_body‖                  # 所有身体的加速度之和
加速度大意味着"动作突然、生硬"——突然踢腿、猛然摆臂。惩罚加速度鼓励柔和平滑的运动。
'''
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


'''
惩罚关节力矩的平方和——鼓励节能、省力。机器人不该用"蛮力"完成任务。
公式
    penalty = Σ τ²     # 所有关节的力矩平方之和
'''
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
'''
力矩越大扣分越多——鼓励策略找到"最省力"的方式完成任务：
    正常行走: τ₁=5, τ₂=8  →  25+64=89   小幅扣分
    暴力行走: τ₁=20, τ₂=30 → 400+900=1300 大幅扣分
使用示例
    @configclass
    class RewardsCfg:
        energy_efficient = RewardTermCfg(
            func=rewards.joint_torques_l2,
            weight=-0.0001,     # 极轻微惩罚，顺势而为即可
        )
'''


'''
和 joint_torques_l2 对称的"省力"奖励——惩罚关节速度，鼓励关节不要转太快。
'''
def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    """通过L1核使用关节速度进行惩罚。"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

'''
L1 vs L2 差异
    # L1 (绝对值): 线性惩罚
    penalty = Σ |ω|           # 关节 1 rad/s → 罚 1，3 rad/s → 罚 3

    # L2 (平方): 二次惩罚
    penalty = Σ ω²            # 关节 1 rad/s → 罚 1，3 rad/s → 罚 9

                L2（平方）	        L1（绝对值）
        小速度	几乎不惩罚	            线性惩罚
        大速度	大幅惩罚	            成比例惩罚
        效果	抑制极端值	            均匀推动归零
        例子	joint_torques_l2	joint_vel_l1
    L1 鼓励关节尽量静止——任何非零速度都线性扣分，推动策略在不必要时不转动关节。
    L2 主要抑制疯狂动作。
使用示例
    @configclass
    class RewardsCfg:
        joint_speed = RewardTermCfg(
            func=rewards.joint_vel_l1,
            weight=-0.001,
            params={"asset_cfg": SceneEntityCfg("robot")},   # ← 必须填
        )
'''

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


'''
惩罚关节加速度——鼓励关节匀速运动，拒绝突然加速或急停。
    和 body_lin_acc_l2（身体级加速度）形成互补：身体级管"整体动作是否柔"，关节级管"每个关节的转动是否平滑"。
'''
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
'''
使用示例
    @configclass
    class RewardsCfg:
        smooth_joints = RewardTermCfg(
            func=rewards.joint_acc_l2,
            weight=-2.5e-7,    # 极微小惩罚，只要不突然抖就行
        )
'''


'''
惩罚关节偏离默认站姿——鼓励机器人在不需要动作时回到默认姿态。
    penalty = Σ |q_current - q_default|
'''
def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    """处罚违反默认的关节位置。"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)
'''
为什么需要？
    没有这个奖励，策略可能学会"完成任务但姿态怪异"——比如用半蹲马步走完全程。这个奖励约束"完成任务的同时要保持正常的站姿"。
使用示例
    @configclass
    class RewardsCfg:
        natural_pose = RewardTermCfg(
            func=rewards.joint_deviation_l1,
            weight=-1.0,    # 偏离默认就扣分
        )
'''


'''
惩罚关节超出软限制（soft limits）——防止关节过度弯曲、过度伸直。
    和 joint_deviation_l1（惩罚偏离默认站姿）互补：deviation 鼓励"靠近默认"，limits 惩罚"越界"。
'''
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
'''
分段惩罚的技巧
    只用 clip 实现了"范围内不罚、越界才罚"的分段函数：
        # 低于下限: -(q - lower).clip(max=0.0)
        q - lower = -2.5 - (-2.0) = -0.5
        .clip(max=0.0) → -0.5  (不截断，因为是负数)
        -( -0.5 ) = 0.5  ← 惩罚值 ✅

        # 在范围内: -(q - lower).clip(max=0.0)
        q - lower = 0.0 - (-2.0) = 2.0
        .clip(max=0.0) → 0  (截断为正数)
        -(0) = 0  ← 不惩罚 ✅

        # 高于上限: (q - upper).clip(min=0.0)
        q - upper = 2.5 - 2.0 = 0.5
        .clip(min=0.0) → 0.5  (不截断，因为是正数)
        = 0.5  ← 惩罚值 ✅
使用示例
    @configclass
    class RewardsCfg:
        joint_limits = RewardTermCfg(
            func=rewards.joint_pos_limits,
            weight=-10.0,    # 越界狠狠扣分——相当于"墙"
        )
'''


'''
joint_pos_limits 的速度版本——惩罚关节转速超过软限制。
    penalty = clip(|ω| - soft_limit, 0, 1)
    # soft_limit = soft_joint_vel_limits × soft_ratio
'''
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
'''
为什么 clip 到 1.0？
        out_of_limits.clip_(min=0.0, max=1.0)
    防止极端速度时惩罚爆炸——关节速度可能瞬间很大（碰撞、掉落），无限惩罚会让训练不稳定。上限 1.0 rad/s per joint 提供了一个"惩罚天花板"：
        正常行走: ω=2 rad/s, limit=3, ratio=0.8 → soft=2.4 → |2|-2.4= -0.4 → clip→0 不罚 ✅
        快速转动: ω=5 rad/s, limit=3, ratio=0.8 → soft=2.4 → |5|-2.4= 2.6 → clip→1.0 惩罚 ✅
        碰撞瞬时: ω=50 rad/s, ... → 惩罚=1.0 (不是 47.6!) → 训练稳定 ✅
使用示例
    @configclass
    class RewardsCfg:
        joint_speed_limit = RewardTermCfg(
            func=rewards.joint_vel_limits,
            weight=-1.0,
            params={"soft_ratio": 0.5},   # 用物理极限的 50% 作软限制
        )
'''


"""
Action penalties.
"""
"""动作处罚。
"""


'''
检测并惩罚执行器饱和——当电机想输出更大力矩但被限制住时，策略应该感受到惩罚。这是 Isaac Lab 中最独特的奖励函数之一。
核心公式
    out_of_limits = |applied_torque - computed_torque|
'''
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
'''
    力矩	            含义
    computed_torque	    PD 控制器想输出的力矩（τ = Kp×Δq + Kd×Δq̇）
    applied_torque	    实际作用在关节上的力矩（被限制后裁剪的值）
    两者差值	            电机"有力使不出"的程度（饱和量）
        正常情况: computed=10, applied=10 → 差值=0  ✅ 执行器没饱和
        饱和情况: computed=30, applied=15 → 差值=15 ⚠️ 电机想用力但被限制了！
⚠️ 限制：仅支持显式执行器
    显式执行器 (IdealPD):  applied_torque = Python 层手动算的 ✅ 可以用
    隐式执行器 (Implicit): applied_torque = PhysX 内部算的，Python 拿不到 ❌
'''


'''
惩罚动作的变化率——鼓励策略输出平滑连贯的动作，避免相邻两帧之间动作剧烈跳变。
公式
    penalty = Σ (a_current - a_previous)²
'''
def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    """使用L2平方内核来惩罚操作的变化速度。"""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
'''
    场景	    action 变化	        惩罚
    平滑操作	[0.1, 0.05, ...]	0.01
    突然猛转	[5.0, -3.0, ...]	34
为什么需要？
    没有这个奖励时，策略可能学会"每步随机输出大动作再纠正"——虽然在仿真中也能走，但动作震荡会导致：
        电机频繁加减速（耗能、发热）
        步态不自然、抖动
        Sim-to-Real 时真实电机无法跟上快速变化
'''

'''
    action_l2       →  Σ a²         "小动作加分"
    action_rate_l2  →  Σ (a - a′)²  "别骤变"
使用示例
    @configclass
    class RewardsCfg:
        small_actions = RewardTermCfg(func=rewards.action_l2,       weight=-0.01)
        smooth_actions = RewardTermCfg(func=rewards.action_rate_l2, weight=-0.01)
'''

'''
action_rate_l2 的姊妹——惩罚动作本身的大小，不仅是变化率。
'''
def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    """使用L2平方内核进行惩罚。"""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


"""
Contact sensor.
"""
"""接触传感器。
"""


'''
惩罚不该碰地的身体部位触地——如膝盖擦地、躯干蹭地。使用 ContactSensor 检测接触力是否超过阈值。
'''
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
'''
    net_forces_w_history 存了历史多帧的接触力，取最大值防止"瞬间轻擦"被忽略。
使用示例
    @configclass
    class RewardsCfg:
        no_knee_ground = RewardTermCfg(
            func=rewards.undesired_contacts,
            weight=-5.0,
            params={
                "threshold": 1.0,                      # 力超过 1N 就算接触
                "sensor_cfg": SceneEntityCfg(
                    "contact_sensor",
                    body_ids=[1, 2, 4, 5],             # 膝盖和手臂的身体 ID
                ),
            },
        )
'''


'''
undesired_contacts 的反面——惩罚该触地的脚却没碰到地。对于行走任务，脚在支撑相时必须有地面接触。
'''
def desired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize if none of the desired contacts are present."""
    """如果没有所需的接触者，请惩罚。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    zero_contact = (~contacts).all(dim=1)
    return 1.0 * zero_contact
'''
逻辑
    contacts = (历史力 > threshold)         # 每个身体是否有接触
    zero_contact = (~contacts).all()        # 所有指定身体都没接触？
    return 1.0 if zero_contact else 0       # 没接触就罚 1

    情况	                惩罚
    至少有一只脚着地	    0 ✅
    双脚都悬空	            1.0 ❌
使用示例
    @configclass
    class RewardsCfg:
        feet_on_ground = RewardTermCfg(
            func=rewards.desired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_ids=[3, 6]),  # 双脚
                "threshold": 1.0,
            },
        )
'''


'''
接触惩罚的最强版本——不像前两个只做有/无判断，这个按超幅力度成比例扣分。
'''
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
'''
    violation = max(‖contact_force‖, over_history) - threshold   # 超出阈值的力度
    penalty = Σ clip(violation, min=0)                            # 只罚正向越界

        接触力	        阈值 1N	    惩罚
        0 N（没碰）	    —	        0 ✅
        0.5 N（轻碰）	—	        0 ✅
        3 N（擦地）	    3-1=2	    2 ⚠️
        15 N（重摔）	15-1=14	    14 ❌
    轻擦不罚、重磕狠罚——比二值判断更精细。

使用示例
    @configclass
    class RewardsCfg:
        soft_contact_penalty = RewardTermCfg(
            func=rewards.contact_forces,
            weight=-0.1,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_ids=[1, 2]),
            },
        )
'''


"""
Velocity-tracking rewards.
"""
"""追踪速度的奖励。
"""


'''
励策略跟踪速度命令。用指数核而非 L2 惩罚，是最常用于行走任务的线速度跟踪奖励。
'''
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
'''
指数核 vs L2 惩罚
        # L2 做法: 惩罚误差
        penalty = error² * weight  →  误差越大扣分越多

        # 指数核: 奖励接近目标
        reward = exp(-error / σ²)  →  误差小给接近 1.0，误差大给接近 0
    误差	     L2 惩罚 (weight=-1)	指数核奖励
    0（完美）	   0	                1.0
    0.1	        -0.01	                0.85
    1.0	        -1.0	                0.0
    5.0	        -25.0	                0.0
    指数核的好处： 小幅误差不罚（有容忍度），大幅误差也不炸（卡在 0~1 之间），训练比 L2 更稳定。

std 控制容忍度
    std=0.25  →  误差 0.25 时 reward 降到 exp(-1)≈0.37（严格）
    std=1.0   →  误差 1.0 时 reward 降到 exp(-1)≈0.37（宽松）
使用示例
    @configclass
    class RewardsCfg:
        speed_tracking = RewardTermCfg(
            func=rewards.track_lin_vel_xy_exp,
            weight=1.0,                              # 正权重 = 奖励
            params={
                "std": 0.25,                         # 严格跟踪
                "command_name": "base_velocity",
            },
        )
'''


'''
track_lin_vel_xy_exp 的角速度版本——奖励策略跟踪yaw 角速度命令（原地转圈）。
'''
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
'''
与线速度版本的对比
    # 线速度: 2D 误差（vx, vy）
    error = (vx_des - vx_actual)² + (vy_des - vy_actual)²

    # 角速度: 1D 误差（ωz）
    error = (ωz_des - ωz_actual)²
同样用指数核 exp(-error/std²)，std 控制容忍度。
'''
