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
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        """在仿真世界框架中的根的角速度。
        在 (0.0，0.0，0.0) 之前的默认设置。
        """

        # joint state
        joint_pos: dict[str, float] = {".*": 0.0}
        """Joint positions of the joints. Defaults to 0.0 for all joints."""
        """关节的关节位置。
        所有关节的默认值为0.0。
        """
        joint_vel: dict[str, float] = {".*": 0.0}
        """Joint velocities of the joints. Defaults to 0.0 for all joints."""
        """关节的关节速度。
        所有关节的默认值为0.0。
        """

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

    actuators: dict[str, ActuatorBaseCfg] = MISSING
    """Actuators for the robot with corresponding joint names."""
    """机器人执行器具有相应的联合名称。"""

    actuator_value_resolution_debug_print = False
    """Print the resolution of actuator final value when input cfg is different from USD value, Defaults to False
    """
    """在输入时打印动机最终值的分辨率cfg是不同于USD基本值，False
    """
