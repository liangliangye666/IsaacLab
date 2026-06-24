# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets.articulation.articulation_data import ArticulationData


class MultirotorData(ArticulationData):
    """Data container for a multirotor articulation.

    This class extends the base :class:`~isaaclab.assets.ArticulationData` container to include
    multirotor-specific data such as thruster states, thrust targets, and computed forces.
    It provides access to all the state information needed for monitoring and controlling
    multirotor vehicles.

    The data container is automatically created and managed by the :class:`~isaaclab_contrib.assets.Multirotor`
    class. Users typically access this data through the :attr:`Multirotor.data` property.

    Note:
        All tensor attributes have shape ``(num_instances, num_thrusters)`` where
        ``num_instances`` is the number of environment instances and ``num_thrusters``
        is the total number of thrusters per multirotor.

    .. seealso::
        - :class:`~isaaclab.assets.ArticulationData`: Base articulation data container
        - :class:`~isaaclab_contrib.assets.Multirotor`: Multirotor asset class
    """
    """对于多轮机关节的数据容器。

    这类扩大:class:`~isaaclab.assets.ArticulationData`基容器，包括多轮机特定数据，如推进器状态，推进目标和计算力量。
    它提供了监测和控制多动力车辆所需的所有国家信息。

    数据容器由:class:`~isaaclab_contrib.assets.Multirotor`自动创建和管理
    class. Users typically access this data through the :attr:`Multirotor.data` property.

    说明：
        所有子属性都有``(num_instances，
        num_thrusters)``的形状，其中``num_instances``是环境实例数，``num_thrusters``是每个多轮机的推进器总数。

    ..
    查看:
        - :class:`~isaaclab.assets.ArticulationData`:基础关节数据容器
        - :class:`~isaaclab_contrib.assets.Multirotor`:多动机资产类
    """

    thruster_names: list[str] = None
    """List of thruster names in the multirotor.

    This list contains the ordered names of all thrusters, matching the order used
    for indexing in the thrust tensors. The names correspond to the USD body prim names
    matched by the thruster name expressions in the actuator configuration.

    Example:
        ``["rotor_0", "rotor_1", "rotor_2", "rotor_3"]`` for a quadcopter
    """
    """在多轮机中的推进器名称列表。

    本列表包含所有推进器的顺序名称，符合使用顺序
    for indexing in the thrust tensors. The names correspond to the USD body prim names
    在执行器配置中的推进器名称表达式相匹配。

    示例：
        ``["rotor_0"， "rotor_1"， "rotor_2"， "rotor_3"]``用于四旋翼
    """

    default_thruster_rps: torch.Tensor = None
    """Default thruster RPS (revolutions per second) state of all thrusters. Shape is (num_instances, num_thrusters).

    This quantity is configured through the :attr:`MultirotorCfg.init_state.rps` parameter
    and represents the baseline/hover RPS for each thruster. It is used to initialize
    thruster states during reset operations.

    For a hovering multirotor, these values should produce enough collective thrust
    to counteract gravity.

    Example:
        For a 1kg quadcopter with 4 thrusters, if each thruster produces 2.5N at 110 RPS,
        the default might be ``[[110.0, 110.0, 110.0, 110.0]]`` for hover.
    """
    """默认驱动器RPS (每秒转换) 所有驱动器的状态
    形状是 (num_instances，num_thrusters)。

    这种数量通过:attr:`MultirotorCfg.init_state.rps`参数进行配置，代表每个推进器的基线/飞行 RPS。
    它用于重置操作期间启动推进状态。

    对于浮动多轮机，这些值应产生足够的集体推力来抵制重力。

    示例：
        对于4驱动器的1kg四旋翼，如果每个驱动器在110RPS时产生2.5N，默认可能是 X升的``[[110.0， 110.0， 110.0， 110.0]]``。
    """

    thrust_target: torch.Tensor = None
    """Thrust targets commanded by the user or controller. Shape is ``(num_instances, num_thrusters)``

    This quantity contains the target thrust values set through the
    :meth:`~isaaclab_contrib.assets.Multirotor.set_thrust_target` method or by
    action terms in RL environments. These targets are processed by the thruster
    actuator models to compute actual applied thrusts.

    The units depend on the actuator model configuration (typically Newtons for
    force or RPS for rotational speed).
    """
    """用户或控制器命令的推力目标。
    形状是``(num_instances， num_thrusters)``

    这种数量包含通过:meth:`~isaaclab_contrib.assets.Multirotor.set_thrust_target`方法或在RL环境中的动作项设置的目标推力值。
    这些目标由驱动器动力模型处理，以计算实际应用的驱动。

    单元取决于执行器模型配置 (通常为力Newton或转速RPS)。
    """

    ##
    # Thruster commands
    ##

    computed_thrust: torch.Tensor = None
    """Computed thrust from the actuator model before clipping. Shape is (num_instances, num_thrusters).

    This quantity contains the thrust values computed by the thruster actuator models
    before any clipping or saturation is applied. It represents the "desired" thrust
    based on the actuator dynamics (rise/fall times) but may exceed physical limits.

    The difference between :attr:`computed_thrust` and :attr:`applied_thrust` indicates
    when the actuator is saturating at its limits.

    Example Use:
        Monitor actuator saturation by comparing computed vs applied thrust:

        .. code-block:: python

            saturation = multirotor.data.computed_thrust - multirotor.data.applied_thrust
            is_saturated = saturation.abs() > 1e-6
    """
    """在切断前从动机模型计算的推力。
    形状是 (num_instances，num_thrusters)。

    这种数量包含在任何裁剪或化之前由推动器动机模型计算的推力值。
    它代表基于动机动态 (上/下时间) 的"想要"推力，但可能超过物理限制。

    :attr:`computed_thrust`和:attr:`applied_thrust`之间的差异表明动机在其极限时和。

    实例使用:通过比较计算与应用的推力来监测执行器度:

        .. code-block:: python

            saturation = multirotor.data.computed_thrust - multirotor.data.applied_thrust
            is_saturated = saturation.abs() > 1e-6
    """

    applied_thrust: torch.Tensor = None
    """Applied thrust from the actuator model after clipping. Shape is (num_instances, num_thrusters).

    This quantity contains the final thrust values that are actually applied to the
    simulation after all actuator model processing, including:

    - Dynamic response (rise/fall time constants)
    - Clipping to thrust range limits
    - Any other actuator model constraints

    This is the "ground truth" thrust that affects the multirotor's motion in the
    physics simulation.
    """
    """切断后从动机模型中应用的推力。
    形状是 (num_instances，num_thrusters)。

    该数量包含在所有执行器模型处理后实际应用到仿真的最终推力值，包括:

    - 动态响应 (升/降时间常数)
    - 裁剪到推力范围限制
    - 任何其他驱动器模型限制

    这就是"基本真理"的推力，
    """
