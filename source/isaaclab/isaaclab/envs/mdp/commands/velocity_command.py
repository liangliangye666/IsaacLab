# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations
"""含有基于速度移动任务的命令生成器的子模块。"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import NormalVelocityCommandCfg, UniformVelocityCommandCfg

# import logger
logger = logging.getLogger(__name__)


class UniformVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """
    """命令生成器，从均分布中生成SE(2) 的速度命令。

    命令包括 x 和 y 方向的线性速度和z 轴周围的角性速度。
    在机器人的基架中提供。

    如果:attr:`cfg.heading_command`标志设置为True，则从标题错误计算的角度速度类似于对标题错误进行比例控制。
    目标标题采样均
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    从数学上来看，从头条命令计算角度速度如下:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})
    """

    cfg: UniformVelocityCommandCfg
    """The configuration of the command generator."""
    """命令生成器的配置。"""

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        """启动命令生成器。

        参数：
            cfg: 命令生成器的配置。
            env: 环境。

        异常：
            ValueError: 如果方向指令是活跃的，但方向范围没有提供。
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:    # 要用 heading 模式但没给范围
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:    # 填了 heading 范围但没开启
            logger.warning(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]    # 获取机器人句柄

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)  # 速度命令 [vx, vy, ωz]，在机器人机体坐标系中表示
        self.heading_target = torch.zeros(self.num_envs, device=self.device)    # 目标朝向角（世界系），只有 heading_command=True 时被赋值
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 为 True 的环境使用 heading 追踪模式计算角速度
        self.is_standing_env = torch.zeros_like(self.is_heading_env)    # 为 True 的环境速度强制归零（练习站立）
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)   # 线速度追踪误差，计算方式 ‖[vx_des, vy_des] - [vx_actual, vy_actual]‖ 累积
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)  # 角速度追踪误差

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        """返回命令生成器的字符串表示。"""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg
    '''
    输出示例
        UniformVelocityCommand:
            Command dimension: (3,)
            Resampling time range: (4.0, 8.0)
            Heading command: True
            Heading probability: 0.8
            Standing probability: 0.1
    '''

    """
    Properties
    """
    """产品
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        """在基架中所需的基速度命令。
        形状是 (num_envs， 3)。
        """
        return self.vel_command_b

    """
    Implementation specific functions.
    """
    """具体执行功能。
    """

    '''
    每一步调用一次，累积两个追踪指标：线速度误差和角速度误差。
    '''
    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        '''
        线速度追踪误差
            命令的前向+侧向速度   -    机器人实际前向+侧向速度, 欧几里得距离 √(Δvx² + Δvy²), 从命令被采样开始一直累加到命令更换
        '''
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
        '''
        角速度追踪误差
        '''
        '''
        resampling_time_range 是一个 (min, max) 元组，比如 (4.0, 8.0)，表示命令每隔 4~8 秒更换一次。这里取的是最大值 [1] = 8.0 秒。

        为什么要除以 max_command_step？ 
            这是在做一个巧妙的归一化。考虑两种场景：
                策略 A：每步追踪误差 = 0.1，命令持续 4 秒 → 累积误差 = 0.1 × (4.0/dt) 步
                策略 B：每步追踪误差 = 0.1，命令持续 8 秒 → 累积误差 = 0.1 × (8.0/dt) 步
            如果不归一化，策略 B 的累积误差永远是策略 A 的两倍——不是因为 B 追踪得差，只是因为命令持续更久。除以 max_command_step（最长可能的步数）后，两者在同一起跑线上：
                归一化累积误差 = 实际累积误差 / 最长可能的步数
                            = ∑(每步误差) / (max_command_time / dt)
            这是一个最大值归一化——把累积误差缩放到"相对于该命令最长持续时间的误差率"。归一化后的值可以直接在不同 resampling_time_range 配置之间公平比较。
        '''

    '''
    为指定的环境重新随机生成速度命令
    '''
    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
            '''
            # 例如: [0.3, 0.9, 0.1, 0.6] <= 0.8 → [True, False, True, True]
            '''
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    '''
    每步调用一次，在 _resample_command 之后执行。
        对已采样的速度命令做两种后处理：朝向追踪模式的角速度修正和站立环境的强制归零。
    '''
    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        """后处理速度命令。

        这项函数为站立环境设置速度指令为零，并在设置heading_command旗时计算方向的角速度。
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten() # 找出 is_heading_env=True 的环境索引
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
            '''
            P 控制器：朝向误差 → 角速度
                数学公式：
                    heading_error = wrap_to_pi(heading_target - heading_current)
                    ωz_des = clip(stiffness × heading_error, ωz_min, ωz_max)
                    这就是一个最简单的比例控制器（P 控制器）——没有积分项（I）也没有微分项（D），只有一个 P 增益。

                wrap_to_pi 的作用：
                    把角度差规范化到 [-π, π] 范围内。例如：
                        heading_target = 170° (≈ 2.97 rad)，heading_current = -170° (≈ -2.97 rad)
                        直接相减：2.97 - (-2.97) = 5.94 rad（错误！实际上只差 20°）
                        wrap_to_pi(5.94) → -0.34 rad (≈ -20°)（正确！）
                        然后 ωz = stiffness × (-0.34) → 机器人逆时针转 20°
                torch.clip 的作用：
                    即使 heading 误差很大（比如 180°），角速度也不会超过配置的 ang_vel_z 范围。
                    这防止了机器人猛烈旋转——就像一个"限速器"。
            '''
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()   # 找出 is_standing_env=True 的环境索引
        self.vel_command_b[standing_env_ids, :] = 0.0
        '''
        [:, :] 是整行赋值——把站立环境中 vel_command_b 的全部三个分量（vx, vy, ωz）都设为 0。
        无论之前采样了什么值，无论是不是 heading 模式，站立环境的速度命令彻底归零。
        '''

    '''
    控制 Isaac Sim 视口中两个速度箭头的显示/隐藏。
        这是一个 ON/OFF 开关——用户通过热键或界面按钮触发，父类 CommandTerm 调用这个方法，不涉及每帧更新。
        True = 开启可视化，False = 关闭可视化
    '''
    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    '''
    每渲染帧（不是物理步）被调用一次，在 Isaac Sim 视口中机器人上方绘制两个箭头：绿色表示命令目标速度，蓝色表示机器人实际速度。
    这是训练过程中最直观的调试工具——你一眼就能判断策略是否在正确地跟踪速度命令。
    '''
    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        '''
        与 _set_debug_vis_impl 的关系
            _set_debug_vis_impl(True)         ← 用户开启调试可视化
                │  创建标记对象（惰性初始化）
                │  设置 set_visibility(True)
                │  父类注册 _debug_vis_callback 到渲染循环
                ▼
            每渲染帧:
                _debug_vis_callback(event)    ← 【本方法】更新箭头位置和方向
                ▼
            _set_debug_vis_impl(False)        ← 用户关闭调试可视化
                │  设置 set_visibility(False)
                │  父类注销 _debug_vis_callback
        两者分别管"开关"和"更新"，职责完全解耦。
        _set_debug_vis_impl 不关心渲染什么内容，_debug_vis_callback 不关心什么时候被注册。
        '''

    """
    Internal helpers.
    """
    """内部助理。
    """

    '''
    把机体坐标系中的 2D 速度向量 [vx, vy] 转换成 Isaac Sim 视口中渲染箭头所需的缩放和四元数朝向。
        这是 _debug_vis_callback 的底层工具函数——被调用两次，一次生成绿色命令箭头，一次生成蓝色实际速度箭头。
    '''
    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        """将XY基速度命令转换为箭头方向旋转。"""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


class NormalVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from a normal distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The command is sampled from a normal distribution with mean and standard deviation specified in
    the configuration. With equal probability, the sign of the individual components is flipped.
    """
    """命令生成器，从正常分布中生成SE(2) 的速度命令。

    命令包括 x 和 y 方向的线性速度和z 轴周围的角性速度。
    在机器人的基架中提供。

    命令从正常分布中采集样本，中值和标准偏差在配置中指定。
    具有相同的可能性，单个组件的标志被翻转。
    """

    cfg: NormalVelocityCommandCfg
    """The command generator configuration."""
    """命令生成器配置。"""

    def __init__(self, cfg: NormalVelocityCommandCfg, env: ManagerBasedEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        """启动命令生成器。

        参数：
            cfg: 命令生成器配置。
            env: 环境。
        """
        super().__init__(cfg, env)
        # create buffers for zero commands envs
        self.is_zero_vel_x_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)
        '''
        为什么需要这三个新缓冲区？
            这是 NormalVelocityCommand 和 UniformVelocityCommand 在"停止指令"上最本质的差异：
                父类 UniformVelocityCommand：只有"全停"（all-or-nothing）
                    # 父类的缓冲区：
                    self.is_standing_env    # True → 所有三个速度分量归零 [0, 0, 0]
                子类 NormalVelocityCommand：可以"部分停止"（per-dimension）
                    # 父类的缓冲区（继承）：
                    self.is_standing_env        # True → [0, 0, 0] 全停

                    # 子类新增的三个缓冲区：
                    self.is_zero_vel_x_env      # True → vx = 0（只有前向速度归零）
                    self.is_zero_vel_y_env      # True → vy = 0（只有侧向速度归零）
                    self.is_zero_vel_yaw_env    # True → ωz = 0（只有角速度归零）
                三种停止模式的对比：
                    UniformVelocityCommand（父类）:
                    standing  → [0, 0, 0]                   只有一种停法：全停

                    NormalVelocityCommand（子类）:
                    standing  → [0, 0, 0]                   全停（继承）
                    zero_vx   → [0, vy, ωz]                 只停前向（侧移+转圈不受影响）
                    zero_vy   → [vx, 0, ωz]                 只停侧移（前进+转圈不受影响）
                    zero_yaw  → [vx, vy, 0]                 只停转圈（前进+侧移不受影响）
                    组合效果    → 如 vx=0 且 vy=0 → [0, 0, ωz]  原地转圈
        '''

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        """返回命令生成器的字符串表示。"""
        msg = "NormalVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg
    '''
    输出示例：
        NormalVelocityCommand:
            Command dimension: (3,)
            Resampling time range: (4.0, 8.0)
            Standing probability: 0.1
    '''

    '''
    覆写父类的 _resample_command，把采样方式从均匀分布改成正态（高斯）分布，并增加随机符号翻转和逐维度零值概率。
    这是 NormalVelocityCommand 和 UniformVelocityCommand 最核心的差异所在。
    '''
    def _resample_command(self, env_ids):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.normal_(mean=self.cfg.ranges.mean_vel[0], std=self.cfg.ranges.std_vel[0])
        self.vel_command_b[env_ids, 0] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.normal_(mean=self.cfg.ranges.mean_vel[1], std=self.cfg.ranges.std_vel[1])
        self.vel_command_b[env_ids, 1] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        # -- angular velocity - yaw direction
        self.vel_command_b[env_ids, 2] = r.normal_(mean=self.cfg.ranges.mean_vel[2], std=self.cfg.ranges.std_vel[2])
        self.vel_command_b[env_ids, 2] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)

        # update element wise zero velocity command
        # TODO what is zero prob ?
        self.is_zero_vel_x_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[0]
        self.is_zero_vel_y_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[1]
        self.is_zero_vel_yaw_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[2]

        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        '''
        与父类的核心差异
            父类 UniformVelocityCommand._resample_command:   子类 NormalVelocityCommand._resample_command:
            ═══════════════════════════════════════════════   ═══════════════════════════════════════════
            采样方式: 均匀分布                                  采样方式: 正态分布
            vx ~ U(min, max)                                   vx ~ N(mean, std)  ← 分布不同
            vy ~ U(min, max)                                   vy ~ N(mean, std)
            ωz ~ U(min, max)                                   ωz ~ N(mean, std)

            符号: 不翻转                                        符号: 随机翻转（±1 各 50%）
                                                                vx *= ±1  ← 新增
                                                                vy *= ±1
                                                                ωz *= ±1

            heading: 支持 heading_command=True              heading: 不支持（永远 False）

            停止: 只有全停 (is_standing_env)                  停止: 全停 + 逐维归零
                                                                is_zero_vel_x  ← 新增
                                                                is_zero_vel_y  ← 新增
                                                                is_zero_vel_yaw ← 新增
        '''

    '''
    覆写父类 _update_command，在正统采样结果上强制执行停止指令——两层：第一层全停（跟父类一样），第二层逐维归零（子类独有）。
    没有 heading 追踪逻辑（因为 heading_command 永远是 False）。
    '''
    def _update_command(self):
        """Sets velocity command to zero for standing envs."""
        """设置速度指令为站立的零envs。"""
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()  # TODO check if conversion is needed
        self.vel_command_b[standing_env_ids, :] = 0.0

        # Enforce zero velocity for individual elements
        # TODO: check if conversion is needed
        zero_vel_x_env_ids = self.is_zero_vel_x_env.nonzero(as_tuple=False).flatten()
        zero_vel_y_env_ids = self.is_zero_vel_y_env.nonzero(as_tuple=False).flatten()
        zero_vel_yaw_env_ids = self.is_zero_vel_yaw_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[zero_vel_x_env_ids, 0] = 0.0
        self.vel_command_b[zero_vel_y_env_ids, 1] = 0.0
        self.vel_command_b[zero_vel_yaw_env_ids, 2] = 0.0
        '''
        与父类的差异
            父类 UniformVelocityCommand._update_command:     子类 NormalVelocityCommand._update_command:
            ═══════════════════════════════════════════════   ════════════════════════════════════════════
            ① if heading_command:                            ① [不存在 — heading_command 永远是 False]
                计算 P 控制器 ωz

            ② standing → vel[:, :] = 0.0                     ② standing → vel[:, :] = 0.0  ← 相同

            ③ [不存在]                                       ③ zero_vel_x → vel[:, 0] = 0.0  ← 新增
                                                            ④ zero_vel_y → vel[:, 1] = 0.0  ← 新增
                                                            ⑤ zero_vel_yaw → vel[:, 2] = 0.0  ← 新增
        '''
