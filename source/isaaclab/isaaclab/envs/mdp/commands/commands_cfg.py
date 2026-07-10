# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .null_command import NullCommand
from .pose_2d_command import TerrainBasedPose2dCommand, UniformPose2dCommand
from .pose_command import UniformPoseCommand
from .velocity_command import NormalVelocityCommand, UniformVelocityCommand


@configclass
class NullCommandCfg(CommandTermCfg):
    """Configuration for the null command generator."""
    """为零命令生成器的配置。"""

    class_type: type = NullCommand

    def __post_init__(self):
        """Post initialization."""
        """在初始化后。"""
        # set the resampling time range to infinity to avoid resampling
        self.resampling_time_range = (math.inf, math.inf)


'''
速度从均匀分布采样
'''
@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""
    """统一速度命令生成器的配置。"""

    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    """在环境中产生的指令的资产名称。"""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """
    """不管使用方向命令还是角度速度命令。
    默认为 False。

    如果 True，角速度指令由标题错误计算，其中目标标题从所提供的范围均抽样。
    否则，从所提供的范围内均地采样角速度命令。
    """
    '''
    标志控制角速度的生成方式：
        模式 A：heading_command=False（默认）— 直接采样角速度
                # velocity_command.py:_resample_command
                self.vel_command_b[:, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
                # 角速度直接均匀采样，比如 [-1.0, 1.0] rad/s
            策略看到的是 [vx, vy, ωz]，其中 ωz 是一个随机值。策略需要自己学会"要往某个方向走的时候该给多大角速度"。

        模式 B：heading_command=True — 朝向追踪模式
            当你设为 True 时，角速度不再是随机采样，而是根据朝向误差自动计算：
                heading_error = wrap_to_pi(heading_target - heading_current)
                ωz_des = clip(heading_control_stiffness × heading_error, ang_vel_z_min, ang_vel_z_max)
            这条公式的本质是一个 P 控制器（比例控制器）。
                heading_control_stiffness 就是 P 增益：朝向误差 30°（≈0.52 rad），stiffness=1.0 → 角速度命令 = 0.52 rad/s，机器人会匀速转向。
                stiffness=2.0 → 角速度 = 1.04 rad/s，转向更快。
        通俗类比：
            模式 A 是"我告诉你转多快，你自己想办法走"。模式 B 是"我告诉你要去哪个方向，你自己看着转"。
            模式 B 更符合人类的直觉——我们说"往北走"，不会说"以 0.3 rad/s 旋转直到朝向朝北"。
            对于需要让机器人在随机方向上行走的任务，模式 B 是更好的选择。
    '''

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""
    """转换方向错误为角速度命令的尺度因素。
    默认到1.0。
    """
    '''
    P 控制器的增益系数
    '''

    '''
    heading_command=True 开了朝向追踪模式后，不是所有环境都必须用朝向追踪——所以有了下面两个参数。
    '''
    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""
    """必须保持静止的环境的概率。
    默认为0.0。
    """
    '''
    站立概率。
        训练时有一定比例的环境速度命令 = [0, 0, 0]。
    为什么需要这个？
        如果机器人一直在走，策略可能永远学不会"如何稳定站立"。混入一些"静止"样本，策略就学会了在停止指令下保持平衡。
    这也是一种课程学习的轻量替代：
        如果 rel_standing_envs=0.0，机器人只在环境 reset 时短暂静止；如果设为 0.2，有 20% 的环境每一步都在学习"站稳"。
    '''

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """
    """机器人遵循以标题为基础的角度速度指令的环境的样本概率 (其他的则遵循样本的角度速度指令)。
    默认到1.0。

    如果:attr:`heading_command`是True，则使用此参数。
    """
    '''
    朝向追踪的比例。
        只在 heading_command=True 时生效。
        设为 0.8 表示 80% 的环境用朝向追踪模式（从 heading 误差算角速度），20% 的环境还是用直接采样的角速度。
        这种混合训练让策略同时适应两种类型的角速度指令，增强泛化能力。
    '''

    '''
    速度范围
    '''
    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""
        """速度指令的分布范围均。"""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""
        """为线性x速度命令的范围 (以m/s)。"""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""
        """线性y速度指令的范围 (m/s)。"""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""
        """角度z速度命令的范围 (在rad/s)。"""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """
        """方向指令的距离 (rAD)。
        默认为 None。

        如果:attr:`~UniformVelocityCommandCfg.heading_command`是True，则使用此参数。
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""
    """速度指令的分布范围。"""

    '''
    可视化标记配置
        这是干什么的？
            在 Isaac Sim 视口中，你可以看到两个箭头悬浮在机器人上方：
                绿色箭头（goal_vel）：指向命令要求的速度方向，长度正比于速度大小
                蓝色箭头（current_vel）：指向机器人实际的速度方向
    '''
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""
    """目标速度可视化标记的配置。
    在GREEN_ARROW_X_MARKER_CFG中默认错误。
    """

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""
    """目前速度可视化标记的配置。
    在 BLUE_ARROW_X_MARKER_CFG中默认错误。
    """

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5) 箭头的缩放
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


'''
速度从正态分布采样
'''
@configclass
class NormalVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""
    """常规速度命令生成器的配置。"""

    class_type: type = NormalVelocityCommand
    heading_command: bool = False  # --> we don't use heading command for normal velocity command.
    '''
    正态分布速度命令不支持 heading 追踪模式。
    '''

    @configclass
    class Ranges:
        """Normal distribution ranges for the velocity commands."""
        """速度指令的正常分布范围。"""

        mean_vel: tuple[float, float, float] = MISSING
        """Mean velocity for the normal distribution (in m/s).

        The tuple contains the mean linear-x, linear-y, and angular-z velocity.
        """
        """通常分布的平均速度 (m/s)。

        元组包含平均线性x，线性y和角性z速度。       均值
        """

        std_vel: tuple[float, float, float] = MISSING
        """Standard deviation for the normal distribution (in m/s).

        The tuple contains the standard deviation linear-x, linear-y, and angular-z velocity.
        """
        """通常分布的标准偏差 (m/s)。

        元组包含标准偏差线性x，线性y和角性z速度。   标准差
        """

        zero_prob: tuple[float, float, float] = MISSING
        """Probability of zero velocity for the normal distribution.

        The tuple contains the probability of zero linear-x, linear-y, and angular-z velocity.
        """
        """对于正常分布的零速度概率。

        元组包含零线性x，线性y和角性z速度的概率。   零值概率
        """
        '''
        zero_prob 是一个三元组，每个分量独立控制对应速度维度被强制设为 0 的概率。
            这是每个维度独立的伯努利试验。比如 zero_prob=(0.3, 0.3, 0.5)：
                30% 的环境 vx = 0（只走侧移不走前后）
                30% 的环境 vy = 0（只走前后不走侧移）
                50% 的环境 ωz = 0（不转圈，纯直走）
            这和父类的 rel_standing_envs（全部三个分量同时清零）是互补的——rel_standing_envs 让机器人完全静止，zero_prob 让机器人练习"只用一个维度的运动"。
            两者都是隐式课程学习的手段：不需要手写课程表，靠概率采样自然创造出各种训练场景。
        '''

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""
    """速度指令的分布范围。"""
    '''
    典型配置示例：
        # 正态分布速度命令：均值前进 0.8 m/s，偶尔侧移，几乎不转圈
        NormalVelocityCommandCfg(
            asset_name="robot",
            rel_standing_envs=0.1,          # 10% 时间站立（继承自父类）
            ranges=NormalVelocityCommandCfg.Ranges(
                mean_vel=(0.8, 0.0, 0.0),  # 平均前向 0.8 m/s，不侧移不转圈
                std_vel=(0.3, 0.1, 0.1),   # 标准差: 前向波动大，侧向/转圈波动小
                zero_prob=(0.2, 0.3, 0.5), # 20% 环境无前向速度，30% 无侧移，50% 不转圈
            ),
        )
    产生的速度分布如下：
        vx:  N(0.8, 0.3) → 68% 落在 [0.5, 1.1] m/s
        vy:  N(0.0, 0.1) → 68% 落在 [-0.1, 0.1] m/s（基本直走）
        ωz:  N(0.0, 0.1) → 68% 落在 [-0.1, 0.1] rad/s（基本不转）

        然后再叠加:
        - 20% 环境 vx 强制为 0（偶尔练习静止或纯侧移）
        - 30% 环境 vy 强制为 0（偶尔练习纯直走不侧移）
        - 50% 环境 ωz 强制为 0（一半时间不转圈）
        - 10% 环境全部归零 → 学习站立
    '''


@configclass
class UniformPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""
    """配置为统一姿势命令生成器。"""

    class_type: type = UniformPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    """在环境中产生的指令的资产名称。"""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""
    """产品中指令生成的机构名称。"""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """
    """不管要么，要么，要么。
    默认为 False。

    如果是True，则通过确保真实部分是正的，使四分之一变得独特。
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""
        """对于姿势命令的均分布范围。"""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""
        """对 x 位置的范围 (在 m 中)。"""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""
        """为 y 位置的范围 (m)。"""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""
        """为z位置的范围 (m)。"""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""
        """滚动角的范围 (rAD)。"""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""
        """射程为角 (在rad)。"""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""
        """对角的范围 (在rad)。"""

    ranges: Ranges = MISSING
    """Ranges for the commands."""
    """距离是命令的。"""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""
    """目标位置的配置可视化标记。
    在 FRAME_MARKER_CFG 设置中，
    """

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""
    """目前姿势可视化标记的配置。
    在 FRAME_MARKER_CFG 设置中，
    """

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


@configclass
class UniformPose2dCommandCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""
    """统一的2D位置命令生成器的配置。"""

    class_type: type = UniformPose2dCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    """在环境中产生的指令的资产名称。"""

    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """
    """无论使用简单的标题还是不用。

    如果True，标题朝着目标位置方向。
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""
        """位置指令的分布范围均。"""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""
        """对 x 位置的范围 (在 m 中)。"""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""
        """为 y 位置的范围 (m)。"""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """
        """位置指令的方向范围 (rAD)。

        仅用于:attr:`simple_heading`是False。
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""
    """位置指令的分布范围。"""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    """The configuration for the goal pose visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""
    """目标位置的配置可视化标记。
    在GREEN_ARROW_X_MARKER_CFG中默认错误。
    """

    # Set the scale of the visualization markers to (0.2, 0.2, 0.8)
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)


@configclass
class TerrainBasedPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for the terrain-based position command generator."""
    """基于地形位置指令生成器的配置。"""

    class_type = TerrainBasedPose2dCommand

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""
        """位置指令的分布范围均。"""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """
        """位置指令的方向范围 (rAD)。

        仅用于:attr:`simple_heading`是False。
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the sampled commands."""
    """采样命令的分布范围。"""
