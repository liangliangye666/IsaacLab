# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_contrib.actuators import ThrusterCfg

from .multirotor import Multirotor


@configclass
class MultirotorCfg(ArticulationCfg):
    """Configuration parameters for a multirotor articulation.

    This configuration class extends :class:`~isaaclab.assets.ArticulationCfg` to add
    multirotor-specific parameters including thruster actuators, allocation matrices,
    and thruster-specific initial states.

    Unlike standard articulations that use joint actuators, multirotors are configured
    with :class:`~isaaclab_contrib.actuators.ThrusterCfg` actuators that model individual
    rotor/propeller dynamics.

    Key Configuration Parameters:
        - **actuators**: Dictionary mapping actuator names to :class:`~isaaclab_contrib.actuators.ThrusterCfg`
          configurations. Each configuration defines a group of thrusters with shared properties.
        - **allocation_matrix**: Maps individual thruster forces to 6D body wrenches. This matrix
          encodes the geometric configuration and should have shape (6, num_thrusters).
        - **thruster_force_direction**: Direction vector in body frame that thrusters push along.
          Typically (0, 0, 1) for upward-facing thrusters.
        - **rotor_directions**: Spin direction of each rotor (1 for CCW, -1 for CW). Used for
          computing reaction torques.

    Example:
        .. code-block:: python

            from isaaclab_contrib.assets import MultirotorCfg
            from isaaclab_contrib.actuators import ThrusterCfg
            import isaaclab.sim as sim_utils

            # Quadcopter configuration
            quadcopter_cfg = MultirotorCfg(
                prim_path="/World/envs/env_.*/Quadcopter",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="path/to/quadcopter.usd",
                ),
                init_state=MultirotorCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 1.0),  # Start 1m above ground
                    rps={".*": 110.0},  # All thrusters at 110 RPS (hover)
                ),
                actuators={
                    "thrusters": ThrusterCfg(
                        thruster_names_expr=["rotor_[0-3]"],
                        thrust_range=(0.0, 12.0),  # 0-12N per thruster
                        rise_time_constant=0.12,
                        fall_time_constant=0.25,
                    ),
                },
                allocation_matrix=[
                    [1.0, 1.0, 1.0, 1.0],  # Vertical thrust
                    [0.0, 0.0, 0.0, 0.0],  # Lateral force X
                    [0.0, 0.0, 0.0, 0.0],  # Lateral force Y
                    [0.0, 0.13, 0.0, -0.13],  # Roll torque
                    [-0.13, 0.0, 0.13, 0.0],  # Pitch torque
                    [0.01, -0.01, 0.01, -0.01],  # Yaw torque
                ],
                rotor_directions=[1, -1, 1, -1],  # Alternating CW/CCW
            )

    .. seealso::
        - :class:`~isaaclab.assets.ArticulationCfg`: Base articulation configuration
        - :class:`~isaaclab_contrib.actuators.ThrusterCfg`: Thruster actuator configuration
        - :class:`Multirotor`: Multirotor asset class
    """
    """多机动机关联的配置参数。

    该配置类扩大:class:`~isaaclab.assets.ArticulationCfg`，增加包括推进器执行器，分配矩阵和推进器特定的初始状态的多轮机特定参数。

    与使用关节动机的标准关节不同，多轮机配置
    with :class:`~isaaclab_contrib.actuators.ThrusterCfg` actuators that model individual
    轮子/螺旋动力。

    主要配置参数:
        - **动机**:字典将动机名称映射到:class:`~isaaclab_contrib.actuators.ThrusterCfg`配置.每个配置定义了一个具有共享属性的推进器组。
        - **allocation_matrix**:将单个推力力量映射到6D车身钥匙.这个矩阵编码了几何配置，应该有形状 (6，num_thrusters)。
        - **thruster_force_direction**:推进器推进的车身框架中的方向向量.通常是向上面推进器的 (0， 0， 1)。
        - **rotor_directions**:每个旋转器的旋转方向 (1 CCW， -1 CW).用于计算反应扭矩。

    示例：
        .. code-block:: python

            from isaaclab_contrib.assets import MultirotorCfg
            from isaaclab_contrib.actuators import ThrusterCfg
            import isaaclab.sim as sim_utils

            # Quadcopter configuration
            quadcopter_cfg = MultirotorCfg(
                prim_path="/World/envs/env_.*/Quadcopter",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="path/to/quadcopter.usd",
                ),
                init_state=MultirotorCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 1.0),  # Start 1m above ground
                    rps={".*": 110.0},  # All thrusters at 110 RPS (hover)
                ),
                actuators={
                    "thrusters": ThrusterCfg(
                        thruster_names_expr=["rotor_[0-3]"],
                        thrust_range=(0.0, 12.0),  # 0-12N per thruster
                        rise_time_constant=0.12,
                        fall_time_constant=0.25,
                    ),
                },
                allocation_matrix=[
                    [1.0, 1.0, 1.0, 1.0],  # Vertical thrust
                    [0.0, 0.0, 0.0, 0.0],  # Lateral force X
                    [0.0, 0.0, 0.0, 0.0],  # Lateral force Y
                    [0.0, 0.13, 0.0, -0.13],  # Roll torque
                    [-0.13, 0.0, 0.13, 0.0],  # Pitch torque
                    [0.01, -0.01, 0.01, -0.01],  # Yaw torque
                ],
                rotor_directions=[1, -1, 1, -1],  # Alternating CW/CCW
            )

    ..
    查看:
        - :class:`~isaaclab.assets.ArticulationCfg`:基础关节配置
        - :class:`~isaaclab_contrib.actuators.ThrusterCfg`:推进器执行器配置
        - :class:`Multirotor`:多动机资产类
    """

    class_type: type = Multirotor

    @configclass
    class InitialStateCfg(ArticulationCfg.InitialStateCfg):
        """Initial state of the multirotor articulation.

        This extends the base articulation initial state to include thruster-specific
        initial conditions. The thruster initial state is particularly important for
        multirotor stability, as it determines the starting thrust levels.

        For hovering multirotors, the initial RPS should be set to values that produce
        enough thrust to counteract gravity.
        """
        """多旋转关节的初始状态。

        这将扩大基座关节初始状态，包括推进器特定的初始条件。
        推进器的初始状态对于多轮机稳定性尤为重要，因为它决定了启动推力水平。

        对于浮动多轮机，初始RPS应设置为产生足够的推力来抵制重力的值。
        """

        # multirotor-specific initial state
        rps: dict[str, float] = {".*": 100.0}
        """Revolutions per second (RPS) of the thrusters. Default is 100 RPS.

        This can be specified as:

        - A dictionary mapping regex patterns to RPS values
        - A single wildcard pattern like ``{".*": 100.0}`` for uniform RPS
        - Explicit per-thruster values like ``{"rotor_0": 95.0, "rotor_1": 105.0}``

        The RPS values are used to initialize the thruster states and determine the
        default thrust targets when the multirotor is reset.

        Example:
            .. code-block:: python

                # Uniform RPS for all thrusters
                rps = {".*": 110.0}

                # Different RPS for different thruster groups
                rps = {"rotor_[0-1]": 105.0, "rotor_[2-3]": 115.0}

        Note:
            The actual thrust produced depends on the thruster model's thrust curve
            and other parameters in :class:`~isaaclab_contrib.actuators.ThrusterCfg`.
        """
        """推进器的每秒转换 (RPS)。
        默认是100RPS。

        这可以指定为:

        - 一个字典将regex模式映射到RPS值
        - 单个 wildcard 模式 ``{".*": 100.0}`` 对于统一 RPS
        - 显而易见的每驱动器值，如``{"rotor_0": 95.0， "rotor_1": 105.0}``

        在重置多驱动器时，使用RPS值来初始化推进器状态并确定默认推进目标。

        示例：
            .. code-block:: python

                # Uniform RPS for all thrusters
                rps = {".*": 110.0}

                # Different RPS for different thruster groups
                rps = {"rotor_[0-1]": 105.0, "rotor_[2-3]": 115.0}

        说明：
            产生的实际推力取决于推力模型的推力曲线和:class:`~isaaclab_contrib.actuators.ThrusterCfg`中的其他参数。
        """

    # multirotor-specific configuration
    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the multirotor object.

    This includes both the base articulation state (position, orientation, velocities)
    and multirotor-specific state (thruster RPS). See :class:`InitialStateCfg` for details.
    """
    """多旋转对象的初始状态。

    这包括基座关节状态 (位置，方向，速度) 和多轮机特定状态 (推进器RPS)。
    看看:class:`InitialStateCfg`为了详细信息。
    """

    actuators: dict[str, ThrusterCfg] = MISSING
    """Thruster actuators for the multirotor with corresponding thruster names.

    This dictionary maps actuator group names to their configurations. Each
    :class:`~isaaclab_contrib.actuators.ThrusterCfg` defines a group of thrusters
    with shared dynamic properties (rise/fall times, thrust limits, etc.).

    Example:
        .. code-block:: python

            actuators = {
                "thrusters": ThrusterCfg(
                    thruster_names_expr=["rotor_.*"],  # Regex to match thruster bodies
                    thrust_range=(0.0, 10.0),
                    rise_time_constant=0.1,
                    fall_time_constant=0.2,
                )
            }

    Note:
        Unlike standard articulations, multirotors should only contain thruster actuators.
        Mixing joint-based and thrust-based actuators is not currently supported.
    """
    """对多轮动机的推进动机，配备相应的推进动机名称。

    这本字典将执行器组名称映射到它们的配置。
    每个:class:`~isaaclab_contrib.actuators.ThrusterCfg`定义了一个推进器组
    with shared dynamic properties (rise/fall times, thrust limits, etc.).

    示例：
        .. code-block:: python

            actuators = {
                "thrusters": ThrusterCfg(
                    thruster_names_expr=["rotor_.*"],  # Regex to match thruster bodies
                    thrust_range=(0.0, 10.0),
                    rise_time_constant=0.1,
                    fall_time_constant=0.2,
                )
            }

    说明：
        与标准关节不同，多轮只应包含推进动机。
        目前不支持混合基于关节和基于推力的执行器。
    """

    # multirotor force application settings
    thruster_force_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Default force direction in body-local frame for thrusters. Default is ``(0.0, 0.0, 1.0)``,
    which is upward along the Z-axis.

    This 3D unit vector specifies the direction in which thrusters generate force
    in the multirotor's body frame. For standard configurations:

    - ``(0.0, 0.0, 1.0)``: Thrusters push upward (Z-axis, typical for quadcopters)
    - ``(0.0, 0.0, -1.0)``: Thrusters push downward
    - ``(1.0, 0.0, 0.0)``: Thrusters push forward (X-axis)

    This is used in conjunction with the allocation matrix to compute the wrench
    produced by each thruster.

    Default: ``(0.0, 0.0, 1.0)`` (upward along Z-axis)
    """
    """在推进器的车身局部框架中默认的力方向。
    默认是``(0.0， 0.0， 1.0)``，它沿着Z轴向上。

    这种3D单元向量指明了推进器在多轮机体框架中产生力量的方向。
    对于标准配置:

    - ``(0.0， 0.0， 1.0)``:推进器向上推 (Z轴，典型于四直升机)
    - ``(0.0， 0.0， -1.0)``:推进器向下推
    - ``(1.0， 0.0， 0.0)``:推进器向前推进 (X轴)

    这与分配矩阵一起用于计算每个推进器产生的钥匙。

    Default: ``(0.0， 0.0， 1.0)`` (沿着Z轴向上)
    """

    allocation_matrix: Sequence[Sequence[float]] | None = None
    """Allocation matrix for control allocation. Default is ``None``, which means that the thrusters
    are not used for control allocation.

    This matrix maps individual thruster forces to the 6D wrench (force + torque)
    applied to the multirotor's base link. It has shape ``(6, num_thrusters)``:

    - **Rows 0-2**: Force contributions in body frame (Fx, Fy, Fz)
    - **Rows 3-5**: Torque contributions in body frame (Tx, Ty, Tz)

    The allocation matrix encodes the geometric configuration of the multirotor,
    including thruster positions, orientations, and moment arms.

    Example for a quadcopter (4 thrusters in + configuration):
        .. code-block:: python

            allocation_matrix = [
                [1.0,  1.0,  1.0,  1.0],     # Total vertical thrust
                [0.0,  0.0,  0.0,  0.0],     # No lateral force
                [0.0,  0.0,  0.0,  0.0],     # No lateral force
                [0.0,  0.13, 0.0, -0.13],    # Roll moment (left/right)
                [-0.13, 0.0, 0.13, 0.0],     # Pitch moment (forward/back)
                [0.01,-0.01, 0.01,-0.01],    # Yaw moment (rotation)
            ]

    Note:
        If ``None``, forces must be applied through other means. For typical
        multirotor control, this should always be specified.
    """
    """控制分配的分配矩阵
    默认是``None``，这意味着推进器不用于控制分配。

    这一矩阵将单个推进力的图绘制到6D关 (力+扭矩) 应用到多轮机的基链上。
    它的形状是``(6， num_thrusters)``:

    - **行0-2**:体格框架中的力量贡献 (Fx， Fy， Fz)
    - **行3-5**:车身框架中的扭矩贡献 (Tx， Ty， Tz)

    配置矩阵编码了多轮机的几何配置，包括推进器的位置，方向和时机臂。

    四旋翼的例子 (4 推进器+配置):
        .. code-block:: python

            allocation_matrix = [
                [1.0,  1.0,  1.0,  1.0],     # Total vertical thrust
                [0.0,  0.0,  0.0,  0.0],     # No lateral force
                [0.0,  0.0,  0.0,  0.0],     # No lateral force
                [0.0,  0.13, 0.0, -0.13],    # Roll moment (left/right)
                [-0.13, 0.0, 0.13, 0.0],     # Pitch moment (forward/back)
                [0.01,-0.01, 0.01,-0.01],    # Yaw moment (rotation)
            ]

    说明：
        如果``None``，必须通过其他手段运用力量。
        对于典型的多轮机控制，这应该始终被指定。
    """

    rotor_directions: Sequence[int] | None = None
    """Sequence of rotor directions for each thruster. Default is ``None``, which means that the rotor directions
    are not specified.

    This specifies the spin direction of each rotor, which affects the reaction
    torques generated. Values should be:

    - ``1``: Counter-clockwise (CCW) rotation
    - ``-1``: Clockwise (CW) rotation

    For a quadcopter, a typical configuration is alternating directions to
    cancel reaction torques during hover: ``[1, -1, 1, -1]``.

    Example:
        .. code-block:: python

            # Quadcopter with alternating rotor directions
            rotor_directions = [1, -1, 1, -1]

            # Hexacopter
            rotor_directions = [1, -1, 1, -1, 1, -1]

    Note:
        The length must match the total number of thrusters defined in the
        actuators configuration, otherwise a ``ValueError`` will be raised
        during initialization.
    """
    """每个推进器的旋转方向序列。
    默认是``None``，这意味着旋转方向没有指定。

    这规定了每个旋转器的旋转方向，这影响了产生反应扭矩。
    值应为:

    - ``1``:逆时钟方向 (CCW) 的旋转
    - ``-1``:随时旋转 (CW)

    对于四旋翼来说，典型的配置是交换方向，在悬浮时取消反应扭矩:``[1， -1， 1， -1]``。

    示例：
        .. code-block:: python

            # Quadcopter with alternating rotor directions
            rotor_directions = [1, -1, 1, -1]

            # Hexacopter
            rotor_directions = [1, -1, 1, -1, 1, -1]

    说明：
        长度必须与执行器配置中定义的总推进器数量相匹配，否则在启动过程中将增加``ValueError``。
    """

    def __post_init__(self):
        """Post initialization validation."""
        """在初始化后验证。"""
        # Skip validation if actuators is MISSING
        if self.actuators is MISSING:
            return

        # Count the total number of thrusters from all actuator configs
        num_thrusters = 0
        for thruster_cfg in self.actuators.values():
            if hasattr(thruster_cfg, "thruster_names_expr") and thruster_cfg.thruster_names_expr is not None:
                num_thrusters += len(thruster_cfg.thruster_names_expr)

        # Validate rotor_directions matches number of thrusters
        if self.rotor_directions is not None:
            num_rotor_directions = len(self.rotor_directions)
            if num_thrusters != num_rotor_directions:
                raise ValueError(
                    f"Mismatch between number of thrusters ({num_thrusters}) and "
                    f"rotor_directions ({num_rotor_directions}). "
                    "They must have the same number of elements."
                )

        # Validate rps explicit entries match number of thrusters
        # Only validate if rps has explicit entries (not just a wildcard pattern)
        if hasattr(self.init_state, "rps") and self.init_state.rps is not None:
            rps_keys = list(self.init_state.rps.keys())
            # Check if rps uses a wildcard pattern (single key that's a regex)
            is_wildcard = len(rps_keys) == 1 and (rps_keys[0] == ".*" or rps_keys[0] == ".*:.*")

            if not is_wildcard and len(rps_keys) != num_thrusters:
                raise ValueError(
                    f"Mismatch between number of thrusters ({num_thrusters}) and "
                    f"rps entries ({len(rps_keys)}). "
                    "They must have the same number of elements when using explicit rps keys."
                )

        # Validate allocation_matrix second dimension matches number of thrusters
        if self.allocation_matrix is not None:
            if len(self.allocation_matrix) == 0:
                raise ValueError("Allocation matrix cannot be empty.")

            # Check that all rows have the same length
            num_cols = len(self.allocation_matrix[0])
            for i, row in enumerate(self.allocation_matrix):
                if len(row) != num_cols:
                    raise ValueError(
                        f"Allocation matrix row {i} has length {len(row)}, "
                        f"but expected {num_cols} (all rows must have the same length)."
                    )

            # Validate that the second dimension (columns) matches number of thrusters
            if num_cols != num_thrusters:
                raise ValueError(
                    f"Mismatch between number of thrusters ({num_thrusters}) and "
                    f"allocation matrix columns ({num_cols}). "
                    "The second dimension of the allocation matrix must match the number of thrusters."
                )
