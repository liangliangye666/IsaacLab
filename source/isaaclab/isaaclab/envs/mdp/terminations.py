# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""
"""可用于激活某些终止的共同函数。

函数可以传递到:class:`isaaclab.managers.TerminationTermCfg`对象，以实现函数引入的终止。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""
"""MDP终止。
"""

'''
episode 超时终止。不关心机器人表现好坏，只判断"时间到了"
'''
def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    """当回合的长度超过最高回合的长度时，结束回合。"""
    return env.episode_length_buf >= env.max_episode_length
'''
使用示例
    @configclass
    class TerminationsCfg:
        time_out = TerminationTermCfg(
            func=terminations.time_out,
            time_out=True,           # ← 标记为截断，非失败
        )
'''


'''
基于命令重采样次数来终止 episode，而非基于固定时长。
    适用于"延迟奖励"场景——让 episode 在命令被重新采样了 N 次后自然结束。
'''
def command_resample(env: ManagerBasedRLEnv, command_name: str, num_resamples: int = 1) -> torch.Tensor:
    """Terminate the episode based on the total number of times commands have been re-sampled.

    This makes the maximum episode length fluid in nature as it depends on how the commands are
    sampled. It is useful in situations where delayed rewards are used :cite:`rudin2022advanced`.
    """
    """根据命令重新采样的总数，结束回合。

    这使得回合长度最大的流体在自然中，因为这取决于命令如何采样。
    在使用延迟奖励的情况下，它是有用的:`rudin2022advanced`。
    """
    '''
    command.time_left <= env.step_dt         # ① 命令马上要被重新采样
    command.command_counter == num_resamples # ② 累计重采样次数达到目标
    '''
    command: CommandTerm = env.command_manager.get_term(command_name)
    return torch.logical_and((command.time_left <= env.step_dt), (command.command_counter == num_resamples))
'''
使用示例
    @configclass
    class TerminationsCfg:
        # 命令被重采样 100 次后结束（弹性 episode）
        command_end = TerminationTermCfg(
            func=terminations.command_resample,
            time_out=True,                        # 标记为截断型
            params={
                "command_name": "base_velocity",
                "num_resamples": 100,
            },
        )
'''


"""
Root terminations.
"""
"""根源终结。
"""


'''
四足机器人行走任务中最经典的终止条件——机器人翻倒了就结束 episode。
'''
def bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    """当资产的导向远离所需的导向限制时，终止。

    通过检查投射重力向量和z轴之间的角度来计算。
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle    # —— 这就是身体的倾斜角！
'''
-projected_gravity_b[:, 2] 正好等于 cos(θ)，θ 是身体相对于竖直方向的倾斜角：
    机器人直立: projected_gravity = [0, 0, -1]
    → -(-1) = 1  → acos(1) = 0°     ← 没有倾斜 ✅

    前倾 30°:  projected_gravity = [0.5, 0, -0.87]
    → -(-0.87) = 0.87 → acos(0.87) ≈ 30° ⚠️ 还没倒

    侧倒 90°:  projected_gravity = [0, 1, 0]
    → -(0) = 0 → acos(0) = 90° = 1.57 rad ❌ 倒了！
使用示例
    @configclass
    class TerminationsCfg:
        fell_over = TerminationTermCfg(
            func=terminations.bad_orientation,
            time_out=False,                          # ← 真正的失败（不是超时）
            params={"limit_angle": 1.2},             # 倾斜超过 1.2 rad (约 69°) 判定摔倒
        )
'''


'''
机器人基座高度低于阈值 → 判定摔倒终止
'''
def root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    """在资产的根高度低于最低高度时终止。

    说明：
        目前仅支持平面地形，最低高度是世界框架的i.e.。
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height
'''
与 bad_orientation 互补
        终止条件	                    检测什么	        局限性
        bad_orientation	                倾斜角超过阈值	    机器人可能平躺着滑行→高度还不够低，没触发
        root_height_below_minimum	    高度过低	        机器人可能站得很直但膝盖完全弯曲→低但没倾斜
    两者配合覆盖所有摔法。
使用示例
    @configclass
    class TerminationsCfg:
        fell_over    = TerminationTermCfg(func=terminations.bad_orientation, time_out=False,
                                        params={"limit_angle": 1.2})
        too_low      = TerminationTermCfg(func=terminations.root_height_below_minimum, time_out=False,
                                        params={"minimum_height": 0.3})  # 低于 0.3m 判定摔倒
'''


"""
Joint terminations.
"""
"""联合终止。
"""


'''
任何关节超出软限制 → 终止 episode。
    这是 rewards.py 中 joint_pos_limits（扣分）的终极版本——不再是"扣点分"，而是直接被判定为失败。
'''
def joint_pos_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the soft joint limits."""
    """在资产的关节位置在软联合限度之外时，终止。"""
    '''
    逻辑
        upper = any(q > limit_max)     # 任一关节超过上限
        lower = any(q < limit_min)     # 任一关节低于下限
        terminate = upper | lower       # 超上或超下都终止
    '''
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)

    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > limits[..., 1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < limits[..., 0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)
'''
使用示例
    @configclass
    class TerminationsCfg:
        joint_limit = TerminationTermCfg(
            func=terminations.joint_pos_out_of_limit,
            time_out=False,
        )
'''


'''
joint_pos_out_of_limit 的手动限制版本。
    结构和逻辑完全相同，唯一的区别：不用资产自带的软限制，改用用户指定的 bounds = (min, max)。
'''
def joint_pos_out_of_manual_limit(
    env: ManagerBasedRLEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the configured bounds.

    Note:
        This function is similar to :func:`joint_pos_out_of_limit` but allows the user to specify the bounds manually.
    """
    """在资产的关节位置在配置边界之外时终止。

    说明：
        这个函数与:func:`joint_pos_out_of_limit`类似，但允许用户手动指定边界。
    """
    '''
    # 自动版: 用资产的 soft limits
        limits = asset.data.soft_joint_pos_limits

    # 手动版: 用户自己定
        limits = bounds  # (min, max)
    '''
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > bounds[1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < bounds[0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)
'''
使用示例
    @configclass
    class TerminationsCfg:
        narrow_limits = TerminationTermCfg(
            func=terminations.joint_pos_out_of_manual_limit,
            time_out=False,
            params={"bounds": (-1.0, 1.0)},   # 比物理极限更窄的自定义范围
        )
'''


'''
joint_pos_out_of_limit 的速度版本——关节转速超软限制 → 终止。
    |ω| > soft_vel_limit  →  终止
'''
def joint_vel_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside of the soft joint limits."""
    """当资产的关节速度超出柔软关节限制时，停止。"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    limits = asset.data.soft_joint_vel_limits
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > limits[:, asset_cfg.joint_ids], dim=1)
'''
使用示例
    # 关节转速超资产软限制 → 终止
    joint_speed_fail = TerminationTermCfg(
        func=terminations.joint_vel_out_of_limit,
        time_out=False,
    )
'''


'''
joint_vel_out_of_limit 的手动版本——用自定义 max_velocity 代替资产软限制。
    |ω| > max_velocity → 终止
'''
def joint_vel_out_of_manual_limit(
    env: ManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside the provided limits."""
    """当资产的关节速度超出所提供的限制时，终止。"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity, dim=1)
'''
速度限制只用单个 max_velocity（绝对值），因为速度方向不重要——不管是正转还是反转，转太快就是危险。
'''
'''
使用示例
    # 关节转速超过自定义上限 15 rad/s → 终止
    custom_speed_fail = TerminationTermCfg(
        func=terminations.joint_vel_out_of_manual_limit,
        time_out=False,
        params={"max_velocity": 15.0},
    )
'''


'''
检测执行器饱和并终止。
    这是 rewards.py 中 applied_torque_limits（扣分）的终止版本。
'''
def joint_effort_out_of_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when effort applied on the asset's joints are outside of the soft joint limits.

    In the actuators, the applied torque are the efforts applied on the joints. These are computed by clipping
    the computed torques to the joint limits. Hence, we check if the computed torques are equal to the applied
    torques. If they are not, it means that clipping has occurred.
    """
    """在资产的关节上施加的努力在软关节限制之外时，终止。

    在执行器中，应用的扭矩是对关节施加的努力。
    通过将计算的扭矩裁剪到关节极限来计算这些扭矩。
    因此，我们检查计算的扭矩是否等于应用的扭矩。
    如果没有，这意味着已经发生了裁剪。
    """
    '''
    computed_torque = PD 控制器想输出的力矩
    applied_torque  = 实际作用在关节上的力矩（被限制后裁剪）

    computed ≠ applied  →  饱和了 → 终止！
    '''
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if any joint effort is out of limit
    out_of_limits = ~torch.isclose(
        asset.data.computed_torque[:, asset_cfg.joint_ids], asset.data.applied_torque[:, asset_cfg.joint_ids]
    )
    return torch.any(out_of_limits, dim=1)
'''
使用示例
    # 电机想用力但被限制（饱和）→ 终止
    torque_saturation = TerminationTermCfg(
        func=terminations.joint_effort_out_of_limit,
        time_out=False,
    )
'''


"""
Contact sensor.
"""
"""接触传感器。
"""


'''
undesired_contacts（扣分）的终止版本——不该碰的身体碰到了 → 直接终止，不只是扣分。
    max(‖contact_force‖ over history) > threshold → 终止
'''
def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    """当传感器的接触力超过力门时，停止。"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
'''
使用示例
    # 膝盖/手臂接触地面 → 终止
    knees_hit_ground = TerminationTermCfg(
        func=terminations.illegal_contact,
        time_out=False,
        params={
            "threshold": 5.0,                              # 接触力超过 5N 判定触碰
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_ids=[1, 2, 4, 5],                      # 膝盖 + 手臂
            ),
        },
    )
'''
