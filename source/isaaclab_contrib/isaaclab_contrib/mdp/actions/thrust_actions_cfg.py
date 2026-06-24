# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import thrust_actions


@configclass
class ThrustActionCfg(ActionTermCfg):
    """Configuration for the thrust action term.

    This configuration class specifies how policy actions are transformed into thruster
    commands for multirotor control. It provides extensive customization of the action
    processing pipeline including scaling, offsetting, and clipping.

    The action term is designed to work with :class:`~isaaclab_contrib.assets.Multirotor`
    assets and uses their thruster configuration to determine which thrusters to control.

    Key Configuration Options:
        - **scale**: Multiplies raw actions to adjust command magnitude
        - **offset**: Adds a baseline value (e.g., hover thrust) to actions
        - **clip**: Constrains actions to safe operational ranges
        - **use_default_offset**: Automatically uses hover thrust as offset

    Example Configurations:
        **Normalized thrust control around hover**:

        .. code-block:: python

            thrust_action = ThrustActionCfg(
                asset_name="robot",
                scale=2.0,  # Actions in [-1,1] become [-2,2] N
                use_default_offset=True,  # Add hover thrust (e.g., 5N)
                clip={".*": (0.0, 10.0)},  # Final thrust in [0, 10] N
            )

        **Direct thrust control with per-thruster scaling**:

        .. code-block:: python

            thrust_action = ThrustActionCfg(
                asset_name="robot",
                scale={
                    "rotor_[0-1]": 8.0,  # Front rotors: stronger
                    "rotor_[2-3]": 7.0,  # Rear rotors: weaker
                },
                offset=0.0,
                use_default_offset=False,
            )

        **Differential thrust control**:

        .. code-block:: python

            thrust_action = ThrustActionCfg(
                asset_name="robot",
                scale=3.0,
                use_default_offset=True,  # Center around hover
                clip={".*": (-2.0, 8.0)},  # Allow +/-2N deviation
            )

    .. seealso::
        - :class:`~isaaclab_contrib.mdp.actions.ThrustAction`: Implementation of this action term
        - :class:`~isaaclab.managers.ActionTermCfg`: Base action term configuration
    """
    """推动动作项的配置

    这类配置类指定了如何将策略动作转化为多轮机控制的推进器命令。
    它提供了操作处理管道的广泛定制，包括扩展，抵消和裁剪。

    动作项旨在与:class:`~isaaclab_contrib.assets.Multirotor`资产工作，并使用它们的推进器配置来确定要控制哪些推进器。

    主要配置选项:
        - **规模**:乘以原始动作来调整命令大小
        - **抵消**:为动作增加基线值 (e.g.，悬浮力)
        - **clip**:限制动作到安全的运营范围
        - **use_default_offset**:自动使用浮动推力作为抵消

    举例 配置: **在悬浮周围的正常推力控制**:

        .. code-block:: python

            thrust_action = ThrustActionCfg(
                asset_name="robot",
                scale=2.0,  # Actions in [-1,1] become [-2,2] N
                use_default_offset=True,  # Add hover thrust (e.g., 5N)
                clip={".*": (0.0, 10.0)},  # Final thrust in [0, 10] N
            )

        **直接的推力控制，以每推力扩展**:

        .. code-block:: python

            thrust_action = ThrustActionCfg(
                asset_name="robot",
                scale={
                    "rotor_[0-1]": 8.0,  # Front rotors: stronger
                    "rotor_[2-3]": 7.0,  # Rear rotors: weaker
                },
                offset=0.0,
                use_default_offset=False,
            )

        **不同推力控制**:

        .. code-block:: python

            thrust_action = ThrustActionCfg(
                asset_name="robot",
                scale=3.0,
                use_default_offset=True,  # Center around hover
                clip={".*": (-2.0, 8.0)},  # Allow +/-2N deviation
            )

    ..
    查看:
        - :class:`~isaaclab_contrib.mdp.actions.ThrustAction`:执行本动作项
        - :class:`~isaaclab.managers.ActionTermCfg`:基本动作项配置
    """

    class_type: type[ActionTerm] = thrust_actions.ThrustAction

    asset_name: str = MISSING
    """Name or regex expression of the asset that the action will be mapped to.

    This should match the name given to the multirotor asset in the scene configuration.
    For example, if the robot is defined as ``robot = MultirotorCfg(...)``, then
    ``asset_name`` should be ``"robot"``.
    """
    """投资指数:投资指数

    这应与场景配置中的多轮机资产所给出的名称匹配。
    例如，如果机器人被定义为``robot = MultirotorCfg(...)``，那么``asset_name``应该是``"robot"``。
    """

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action. Default is ``1.0``, which means no scaling.

    This multiplies the raw action values to adjust the command magnitude. It can be:

    - A float: uniform scaling for all thrusters (e.g., ``2.0``)
    - A dict: per-thruster scaling using regex patterns (e.g., ``{"rotor_.*": 2.5}``)

    For normalized actions in [-1, 1], the scale determines the maximum deviation
    from the offset value.

    Example:
        .. code-block:: python

            # Uniform scaling
            scale = 5.0  # Actions of ±1 become ±5N

            # Per-thruster scaling
            scale = {
                "rotor_[0-1]": 8.0,   # Front rotors
                "rotor_[2-3]": 6.0,   # Rear rotors
            }
    """
    """动作的规模因素。
    默认是``1.0``，这意味着没有扩展。

    这使原始动作值乘以调整命令大小。
    它可能是:

    - 一个浮动器:所有推进器均的尺度 (e.g.，``2.0``)
    - 一个指令:使用regex模式 (e.g.，``{"rotor_.*": 2.5}``) 的每推进器规模化

    在 [-1， 1] 中的正常化操作中，尺度确定了最大偏差
    from the offset value.

    示例：
        .. code-block:: python

            # Uniform scaling
            scale = 5.0  # Actions of ±1 become ±5N

            # Per-thruster scaling
            scale = {
                "rotor_[0-1]": 8.0,   # Front rotors
                "rotor_[2-3]": 6.0,   # Rear rotors
            }
    """

    offset: float | dict[str, float] = 0.0
    """Offset factor for the action. Default is ``0.0``, which means no offset.

    This value is added to the scaled actions to establish a baseline thrust.
    It can be:

    - A float: uniform offset for all thrusters (e.g., ``5.0`` for 5N hover thrust)
    - A dict: per-thruster offset using regex patterns

    If :attr:`use_default_offset` is ``True``, this value is overwritten by the
    default thruster RPS from the multirotor configuration.

    Example:
        .. code-block:: python

            # Uniform offset (5N baseline thrust)
            offset = 5.0

            # Per-thruster offset
            offset = {
                "rotor_0": 5.2,
                "rotor_1": 4.8,
            }
    """
    """这种动作的抵消因素。
    默认是``0.0``，这意味着没有抵消。

    这一值应添加到规模化动作中，以确定基线推力。
    它可能是:

    - 一个浮动器:所有推进器均的偏移 (e.g.，``5.0``为5N浮动推力)
    - 一个命令:使用regex模式的每驱动器偏移

    If :attr:`use_default_offset`是``True``，这个值由
    默认驱动器RPS从多轮机配置。

    示例：
        .. code-block:: python

            # Uniform offset (5N baseline thrust)
            offset = 5.0

            # Per-thruster offset
            offset = {
                "rotor_0": 5.2,
                "rotor_1": 4.8,
            }
    """

    clip: dict[str, tuple[float, float]] | None = None
    """Clipping ranges for processed actions. Default is ``None``, which means no clipping.

    This constrains the final thrust commands to safe operational ranges after
    scaling and offset are applied. It must be specified as a dictionary mapping
    regex patterns to (min, max) tuples.

    Example:
        .. code-block:: python

            # Clip all thrusters to [0, 10] N
            clip = {".*": (0.0, 10.0)}

            # Different limits for different thrusters
            clip = {
                "rotor_[0-1]": (0.0, 12.0),  # Front rotors
                "rotor_[2-3]": (0.0, 8.0),   # Rear rotors
            }

    """
    """处理操作的裁剪范围。
    默认是``None``，这意味着没有裁剪。

    这限制了扩展和偏移应用后，最后的推力命令到安全的运行范围。
    它必须被指定为字典映射regex模式到 (min，max) tuples。

    示例：
        .. code-block:: python

            # Clip all thrusters to [0, 10] N
            clip = {".*": (0.0, 10.0)}

            # Different limits for different thrusters
            clip = {
                "rotor_[0-1]": (0.0, 12.0),  # Front rotors
                "rotor_[2-3]": (0.0, 8.0),   # Rear rotors
            }
    """

    preserve_order: bool = False
    """Whether to preserve the order of the asset names in the action output. Default is ``False``.

    If ``True``, the thruster ordering matches the regex pattern order exactly.
    If ``False``, ordering is determined by the USD scene traversal order.
    """
    """在动作输出中是否保留资产名称的顺序。
    默认是``False``。

    如果``True``，推进器的排序完全与Regex模式排序相匹配。
    如果 ``False``，排序由USD场景穿越排序决定。
    """

    use_default_offset: bool = True
    """Whether to use default thrust configured in the multirotor asset as offset. Default is ``True``.

    If ``True``, the :attr:`offset` value is overwritten with the default thruster
    RPS values from :attr:`MultirotorCfg.init_state.rps`. This is useful for
    controlling thrust as deviations from the hover state.

    If ``False``, the manually specified :attr:`offset` value is used.
    """
    """在多轮机资产中配置的默认推力是否作为抵消。
    默认是``True``。

    如果``True``，:attr:`offset`值将被从:attr:`MultirotorCfg.init_state.rps`的默认推进器RPS值覆盖。
    这对于控制推力而有用，因为它与悬浮状态的偏差。

    如果 ``False``，则使用手动指定的 :attr:`offset`值。
    """
