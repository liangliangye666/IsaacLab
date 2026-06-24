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

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""
    """转换方向错误为角速度命令的尺度因素。
    默认到1.0。
    """

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""
    """必须保持静止的环境的概率。
    默认为0.0。
    """

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """
    """机器人遵循以标题为基础的角度速度指令的环境的样本概率 (其他的则遵循样本的角度速度指令)。
    默认到1.0。

    如果:attr:`heading_command`是True，则使用此参数。
    """

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

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class NormalVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""
    """常规速度命令生成器的配置。"""

    class_type: type = NormalVelocityCommand
    heading_command: bool = False  # --> we don't use heading command for normal velocity command.

    @configclass
    class Ranges:
        """Normal distribution ranges for the velocity commands."""
        """速度指令的正常分布范围。"""

        mean_vel: tuple[float, float, float] = MISSING
        """Mean velocity for the normal distribution (in m/s).

        The tuple contains the mean linear-x, linear-y, and angular-z velocity.
        """
        """通常分布的平均速度 (m/s)。

        元组包含平均线性x，线性y和角性z速度。
        """

        std_vel: tuple[float, float, float] = MISSING
        """Standard deviation for the normal distribution (in m/s).

        The tuple contains the standard deviation linear-x, linear-y, and angular-z velocity.
        """
        """通常分布的标准偏差 (m/s)。

        元组包含标准偏差线性x，线性y和角性z速度。
        """

        zero_prob: tuple[float, float, float] = MISSING
        """Probability of zero velocity for the normal distribution.

        The tuple contains the probability of zero linear-x, linear-y, and angular-z velocity.
        """
        """对于正常分布的零速度概率。

        元组包含零线性x，线性y和角性z速度的概率。
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""
    """速度指令的分布范围。"""


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
