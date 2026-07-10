# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .articulation import Articulation


@configclass
class ArticulationCfg(AssetBaseCfg):
    """Configuration parameters for an articulation."""
    """关节的配置参数"""

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the articulation."""
        """关节的初始状态。"""

        # root velocity
        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        """在仿真世界框架中的根的线性速度。
        在 (0.0，0.0，0.0) 之前的默认设置。
        """
        '''
        机器人根节点在世界坐标系中的线速度 (m/s)
        '''

        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        """在仿真世界框架中的根的角速度。
        在 (0.0，0.0，0.0) 之前的默认设置。
        """
        '''
        机器人根节点在世界坐标系中的角速度 (rad/s)
        '''

        # joint state
        joint_pos: dict[str, float] = {".*": 0.0}
        """Joint positions of the joints. Defaults to 0.0 for all joints."""
        """关节的关节位置。
        所有关节的默认值为0.0。
        """
        '''
        各关节的初始角度 (rad) (正则表达式键，匹配关节名)
        匹配关节名称来指定初始角度:
            {".*": 0.0} 表示所有关节角度为 0（完全伸直状态）
            你可以指定特定关节：{".*HAA": 0.5, ".*KFE": -1.0} 只设置 HAA 和 KFE 关节
        '''

        joint_vel: dict[str, float] = {".*": 0.0}
        """Joint velocities of the joints. Defaults to 0.0 for all joints."""
        """关节的关节速度。
        所有关节的默认值为0.0。
        """
        '''
        各关节的初始角速度 (rad/s) (同样用正则表达式键)
        '''

    ##
    # Initialize configurations.
    ##

    class_type: type = Articulation

    articulation_root_prim_path: str | None = None
    """Path to the articulation root prim under the :attr:`prim_path`. Defaults to None, in which case the class
    will search for a prim with the USD ArticulationRootAPI on it.

    This path should be relative to the :attr:`prim_path` of the asset. If the asset is loaded from a USD file,
    this path should be relative to the root of the USD stage. For instance, if the loaded USD file at :attr:`prim_path`
    contains two articulations, one at `/robot1` and another at `/robot2`, and you want to use `robot2`,
    then you should set this to `/robot2`.

    The path must start with a slash (`/`).
    """
    """在:attr:`prim_path`下面的关节根prim的路径。
    在 None 上默认设置，在这种情况下，类别会搜索一个 prim，上面是 USD ArticulationRootAPI。

    这条路径应与资产的:attr:`prim_path`相对。
    如果资产从USD文件中加载，该路径应与USD阶段的根相对。
    例如，如果装载USD在:attr:`prim_path`包含两个关节，其中一个是`/robot1`另一个在`/robot2`你想使用`robot2`然后你应该把这个设置为`/robot2`。

    路径必须由一个切片 (`/`) 开始。
    """
    '''
    精确指定关节根
        作用
            指定关节的根 prim 相对于 prim_path 的路径。
            在 USD 格式中，一个文件可能包含多个关节结构（如 /robot1 和 /robot2），这个字段让你精确选择用哪一个。
        两种模式
            值	            行为
            None（默认）	自动搜索：在第一个环境中找到带 ArticulationRootAPI 的 prim，然后用正则匹配所有环境
            "/robot2"	    精确指定：直接用 prim_path + "/robot2"
        消费位置
            在 _initialize_impl 中（articulation.py:2157-2159）：
                    if self.cfg.articulation_root_prim_path is not None:
                        # The articulation root prim path is specified explicitly, so we can just use this.
                        root_prim_path_expr = self.cfg.prim_path + self.cfg.articulation_root_prim_path
        通俗类比
            articulation_root_prim_path 就像大仓库里的货架号。
            USD 文件是一个大仓库（可能放了多台机器人），prim_path 告诉你仓库在哪，articulation_root_prim_path 告诉你要取哪个货架上的机器人。
            不填的话，系统会自动找到第一个带"关节"标签的货架。
    '''

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the articulated object. Defaults to identity pose with zero velocity and zero joint state."""
    """关节对象的初始状态。
    在零速度和零联合状态下，身份默认存在。
    """

    soft_joint_pos_limit_factor: float = 1.0
    """Fraction specifying the range of joint position limits (parsed from the asset) to use. Defaults to 1.0.

    The soft joint position limits are scaled by this factor to specify a safety region within the simulated
    joint position limits. This isn't used by the simulation, but is useful for learning agents to prevent the joint
    positions from violating the limits, such as for termination conditions.

    The soft joint position limits are accessible through the :attr:`ArticulationData.soft_joint_pos_limits` attribute.
    """
    """占合资本的合资金额 (从资产中分析)
    默认到1.0。

    柔性关节位置限制由此因素扩大，以指定仿真的关节位置限制内的安全区域。
    这不是仿真所使用的，但对于学习代理来说是有用的，以防止关节位置违反限制，

    通过:attr:`ArticulationData.soft_joint_pos_limits`属性可访问柔性关节位置限制。
    """
    '''
    关节的安全"缓冲区"
        数学原理
            每个关节从 USD 文件中解析出的物理极限是 [lower, upper]，但策略不应该用到这个极限——碰到极限意味着关节卡死了，可能损坏机器人或产生不稳定的仿真。

            软极限的计算公式（articulation.py:1106-1109）：
                soft_joint_pos_limits[..., 0] = joint_pos_mean - 0.5 × joint_pos_range × soft_limit_factor
                soft_joint_pos_limits[..., 1] = joint_pos_mean + 0.5 × joint_pos_range × soft_limit_factor
            其中 joint_pos_mean = (lower + upper) / 2，joint_pos_range = upper - lower。

        图示
            关节物理极限 [-1.57, +1.57] rad（软极限因子 = 1.0）
            ├─────[───完全范围───]─────┤    软极限 = 物理极限
            -1.57                    +1.57

            关节软极限 [-1.256, +1.256] rad（软极限因子 = 0.8）
            ├──[──80% 范围──]──┤              安全缓冲区
            -1.57         -1.256    +1.256    +1.57
                        ↑ 策略可以安全活动 ↑
        通俗类比
            soft_joint_pos_limit_factor = 0.8 就像手机电量低于 20% 就报警——电池还能用到 0%，但为了安全，提醒你该充电了。
            同样，关节能转到极限角度，但策略被训练在 80% 范围以内活动，避免碰到硬件极限。
    '''

    actuators: dict[str, ActuatorBaseCfg] = MISSING
    """Actuators for the robot with corresponding joint names."""
    """机器人执行器具有相应的联合名称。"""
    '''
    作用
        指定机器人的执行器配置——把策略输出的目标角度转换为实际力矩的模型。这是 Isaac Lab 最具扩展性的设计之一。
    执行器按模型类型分为两种：
        类型	            基类	                            工作方式	                                        典型用途
        Explicit（显式）	如 DCMotorCfg、ActuatorNetLSTMCfg	策略输出力矩，执行器模型负责计算真实的电机响应	        四足机器人（需要模拟真实电机动力学）
        Implicit（隐式）	ImplicitActuatorCfg	                策略输出目标位置/速度，PhysX 内置 PD 控制器直接跟踪	    机械臂（简单、稳定）
    '''

    actuator_value_resolution_debug_print = False
    """Print the resolution of actuator final value when input cfg is different from USD value, Defaults to False
    """
    """在输入时打印动机最终值的分辨率cfg是不同于USD基本值，False
    """
    '''
    调试小开关
    作用
        当配置中的执行器参数（如 stiffness）与 USD 文件中解析出的值不一致时，是否打印差异信息。
        默认 False 表示静默处理——通常配置的值会覆盖 USD 中的值，这是预期行为。

        这是一个纯粹的调试工具，对仿真行为没有任何影响。只在排查"为什么执行器行为不对"时开启。
    '''
