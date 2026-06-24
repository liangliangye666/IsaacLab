# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import torch

import isaaclab.utils.string as string_utils
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .actuator_base_cfg import ActuatorBaseCfg


class ActuatorBase(ABC):
    """Base class for actuator models over a collection of actuated joints in an articulation.

    Actuator models augment the simulated articulation joints with an external drive dynamics model.
    The model is used to convert the user-provided joint commands (positions, velocities and efforts)
    into the desired joint positions, velocities and efforts that are applied to the simulated articulation.

    The base class provides the interface for the actuator models. It is responsible for parsing the
    actuator parameters from the configuration and storing them as buffers. It also provides the
    interface for resetting the actuator state and computing the desired joint commands for the simulation.

    For each actuator model, a corresponding configuration class is provided. The configuration class
    is used to parse the actuator parameters from the configuration. It also specifies the joint names
    for which the actuator model is applied. These names can be specified as regular expressions, which
    are matched against the joint names in the articulation.

    To see how the class is used, check the :class:`isaaclab.assets.Articulation` class.
    """
    """动机模型的基类在关节中的动机关联的集合上。

    动机模型通过外部驱动动动力模型增强仿真的关节。
    该模型用于将用户提供的联合命令 (位置，速度和努力) 转换为用于仿真关节的所需关节位置，速度和努力。

    基本类为动机模型提供了接口。
    它负责从配置中分析执行器参数，并将它们作为缓冲器存储。
    它还提供了重置执行器状态的接口，并计算了仿真所需的联合命令。

    对于每个动机模型，提供相应的配置类。
    配置类用于从配置中分析执行器参数。
    它还规定了共同名称
    for which the actuator model is applied. These names can be specified as regular expressions, which
    它们与文本中的联合名称相匹配。

    查看:class:`isaaclab.assets.Articulation`类。
    """

    is_implicit_model: ClassVar[bool] = False
    """Flag indicating if the actuator is an implicit or explicit actuator model.

    If a class inherits from :class:`ImplicitActuator`, then this flag should be set to :obj:`True`.
    """
    """标志显示动机是否是隐含或明确的动机模型。

    如果一个类继承:class:`ImplicitActuator`，则该旗应设置为:obj:`True`。
    """

    computed_effort: torch.Tensor
    """The computed effort for the actuator group. Shape is (num_envs, num_joints)."""
    """执行器组的计算力。
    形状是 (num_envs，num_joints)。
    """

    applied_effort: torch.Tensor
    """The applied effort for the actuator group. Shape is (num_envs, num_joints).

    This is the effort obtained after clipping the :attr:`computed_effort` based on the
    actuator characteristics.
    """
    """执行器组所做的努力。
    形状是 (num_envs，num_joints)。

    这是在根据动机特性裁剪:attr:`computed_effort`后获得的努力。
    """

    effort_limit: torch.Tensor
    """The effort limit for the actuator group. Shape is (num_envs, num_joints).

    This limit is used differently depending on the actuator type:

    - **Explicit actuators**: Used for internal torque clipping within the actuator model
      (e.g., motor torque limits in DC motor models).
    - **Implicit actuators**: Same as :attr:`effort_limit_sim` (aliased for consistency).
    """
    """执行器组的努力限制。
    形状是 (num_envs，num_joints)。

    根据执行器类型，这种限制使用不同:

    - **显而易见的动力驱动器**:用于动力驱动器模型内部扭矩裁剪 (e.g.，DC电机模型中的电动扭矩限制)。
    - **隐含执行器**:与:attr:`effort_limit_sim`相同 (以一致性命名)。
    """

    effort_limit_sim: torch.Tensor
    """The effort limit for the actuator group in the simulation. Shape is (num_envs, num_joints).

    For implicit actuators, the :attr:`effort_limit` and :attr:`effort_limit_sim` are the same.

    - **Explicit actuators**: Typically set to a large value (1.0e9) to avoid double-clipping,
      since the actuator model already clips efforts using :attr:`effort_limit`.
    - **Implicit actuators**: Same as :attr:`effort_limit` (both values are synchronized).
    """
    """在仿真中执行器组的功耗限制。
    形状是 (num_envs，num_joints)。

    对于隐含动机，:attr:`effort_limit`和:attr:`effort_limit_sim`是相同的。

    - **明确执行器**:通常设置为大值 (1.0e9) 避免双切，因为执行器模型已经使用:attr:`effort_limit`来切断努力。
    - **隐含执行器**:与:attr:`effort_limit`相同 (两个值都同步)。
    """

    velocity_limit: torch.Tensor
    """The velocity limit for the actuator group. Shape is (num_envs, num_joints).

    For implicit actuators, the :attr:`velocity_limit` and :attr:`velocity_limit_sim` are the same.
    """
    """执行器组的速度限制。
    形状是 (num_envs，num_joints)。

    对于隐含动机，:attr:`velocity_limit`和:attr:`velocity_limit_sim`是相同的。
    """

    velocity_limit_sim: torch.Tensor
    """The velocity limit for the actuator group in the simulation. Shape is (num_envs, num_joints).

    For implicit actuators, the :attr:`velocity_limit` and :attr:`velocity_limit_sim` are the same.
    """
    """在仿真中执行器组的速度限制。
    形状是 (num_envs，num_joints)。

    对于隐含动机，:attr:`velocity_limit`和:attr:`velocity_limit_sim`是相同的。
    """

    stiffness: torch.Tensor
    """The stiffness (P gain) of the PD controller. Shape is (num_envs, num_joints)."""
    """PD控制器的硬度 (P增长)。
    形状是 (num_envs，num_joints)。
    """

    damping: torch.Tensor
    """The damping (D gain) of the PD controller. Shape is (num_envs, num_joints)."""
    """PD控制器的缩 (D增长)。
    形状是 (num_envs，num_joints)。
    """

    armature: torch.Tensor
    """The armature of the actuator joints. Shape is (num_envs, num_joints)."""
    """执行器关节的 armature。
    形状是 (num_envs，num_joints)。
    """

    friction: torch.Tensor
    """The joint static friction of the actuator joints. Shape is (num_envs, num_joints)."""
    """执行器关节的静态摩擦。
    形状是 (num_envs，num_joints)。
    """

    dynamic_friction: torch.Tensor
    """The joint dynamic friction of the actuator joints. Shape is (num_envs, num_joints)."""
    """执行器关节的动态摩擦。
    形状是 (num_envs，num_joints)。
    """

    viscous_friction: torch.Tensor
    """The joint viscous friction of the actuator joints. Shape is (num_envs, num_joints)."""
    """执行器关节的粘摩擦。
    形状是 (num_envs，num_joints)。
    """

    _DEFAULT_MAX_EFFORT_SIM: ClassVar[float] = 1.0e9
    """The default maximum effort for the actuator joints in the simulation. Defaults to 1.0e9.

    If the :attr:`ActuatorBaseCfg.effort_limit_sim` is not specified and the actuator is an explicit
    actuator, then this value is used.
    """
    """在仿真中执行器关节的默认最大功耗。
    在1.0e9上默认设置。

    如果没有指定:attr:`ActuatorBaseCfg.effort_limit_sim`，并且执行器是明确的执行器，则使用此值。
    """

    def __init__(
        self,
        cfg: ActuatorBaseCfg,
        joint_names: list[str],
        joint_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        dynamic_friction: torch.Tensor | float = 0.0,
        viscous_friction: torch.Tensor | float = 0.0,
        effort_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf,
    ):
        """Initialize the actuator.

        The actuator parameters are parsed from the configuration and stored as buffers. If the parameters
        are not specified in the configuration, then their values provided in the constructor are used.

        .. note::
            The values in the constructor are typically obtained through the USD values passed from the PhysX API calls
            corresponding to the joints in the actuator model; these values serve as default values if the parameters
            are not specified in the cfg.



        Args:
            cfg: The configuration of the actuator model.
            joint_names: The joint names in the articulation.
            joint_ids: The joint indices in the articulation. If :obj:`slice(None)`, then all
                the joints in the articulation are part of the group.
            num_envs: Number of articulations in the view.
            device: Device used for processing.
            stiffness: The default joint stiffness (P gain). Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            damping: The default joint damping (D gain). Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            armature: The default joint armature. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            friction: The default joint static friction. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            dynamic_friction: The default joint dynamic friction. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            viscous_friction: The default joint viscous friction. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            effort_limit: The default effort limit. Defaults to infinity.
                If a tensor, then the shape is (num_envs, num_joints).
            velocity_limit: The default velocity limit. Defaults to infinity.
                If a tensor, then the shape is (num_envs, num_joints).
        """
        """启动执行器。

        执行器参数从配置中分析并作为缓冲器存储。
        如果参数在配置中未指定，则使用在构造器中提供的值。

        .. 说明::
            在构造器中的值通常通过 PhysX API调用中传递的USD值来获得，这些值是执行器模型中的关节；如果参数不在cfg中指定，这些值将作为默认值。



        参数：
            cfg: 执行器模型的配置。
            joint_names: 关键词中的共同名称。
            joint_ids: 关节中的索引。
                       如果是:obj:`slice(None)`，那么关节中的所有关节都是集团的一部分。
            num_envs: 视图中的关节数量
            device: 用于处理的设备。
            stiffness: 默认关节硬度 (P增长)。
                       默认为0.0。
                       如果是子，则形状是 (num_envs，num_joints)。
            damping: 默认关节缩 (D增长)。
                     默认为0.0。
                     如果是子，则形状是 (num_envs，num_joints)。
            armature: 默认的关节 armature。
                      默认为0.0。
                      如果是子，则形状是 (num_envs，num_joints)。
            friction: 默认的关节静态摩擦。
                      默认为0.0。
                      如果是子，则形状是 (num_envs，num_joints)。
            dynamic_friction: 默认的关节动态摩擦。
                              默认为0.0。
                              如果是子，则形状是 (num_envs，num_joints)。
            viscous_friction: 默认关节粘性摩擦。
                              默认为0.0。
                              如果是子，则形状是 (num_envs，num_joints)。
            effort_limit: 默认的努力限制。
                          默认到无限。
                          如果是子，则形状是 (num_envs，num_joints)。
            velocity_limit: 默认速度限制。
                            默认到无限。
                            如果是子，则形状是 (num_envs，num_joints)。
        """
        # save parameters
        self.cfg = cfg
        self._num_envs = num_envs
        self._device = device
        self._joint_names = joint_names
        self._joint_indices = joint_ids
        self.joint_property_resolution_table: dict[str, list] = {}
        # For explicit models, we do not want to enforce the effort limit through the solver
        # (unless it is explicitly set)
        if not self.is_implicit_model and self.cfg.effort_limit_sim is None:
            self.cfg.effort_limit_sim = self._DEFAULT_MAX_EFFORT_SIM

        # resolve usd, actuator configuration values
        # case 1: if usd_value == actuator_cfg_value: all good,
        # case 2: if usd_value != actuator_cfg_value: we use actuator_cfg_value
        # case 3: if actuator_cfg_value is None: we use usd_value

        to_check = [
            ("velocity_limit_sim", velocity_limit),
            ("effort_limit_sim", effort_limit),
            ("stiffness", stiffness),
            ("damping", damping),
            ("armature", armature),
            ("friction", friction),
            ("dynamic_friction", dynamic_friction),
            ("viscous_friction", viscous_friction),
        ]
        for param_name, usd_val in to_check:
            cfg_val = getattr(self.cfg, param_name)
            setattr(self, param_name, self._parse_joint_parameter(cfg_val, usd_val))
            new_val = getattr(self, param_name)

            allclose = (
                torch.all(new_val == usd_val) if isinstance(usd_val, (float, int)) else torch.allclose(new_val, usd_val)
            )
            if cfg_val is None or not allclose:
                self._record_actuator_resolution(
                    cfg_val=getattr(self.cfg, param_name),
                    new_val=new_val[0],  # new val always has the shape of (num_envs, num_joints)
                    usd_val=usd_val,
                    joint_names=joint_names,
                    joint_ids=joint_ids,
                    actuator_param=param_name,
                )

        self.velocity_limit = self._parse_joint_parameter(self.cfg.velocity_limit, self.velocity_limit_sim)
        # Parse effort_limit with special default handling:
        # - If cfg.effort_limit is None, use the original USD value (effort_limit parameter from constructor)
        # - Otherwise, use effort_limit_sim as the default
        # Please refer to the documentation of the effort_limit and effort_limit_sim parameters for more details.
        effort_default = effort_limit if self.cfg.effort_limit is None else self.effort_limit_sim
        self.effort_limit = self._parse_joint_parameter(self.cfg.effort_limit, effort_default)

        # create commands buffers for allocation
        self.computed_effort = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        self.applied_effort = torch.zeros_like(self.computed_effort)

    def __str__(self) -> str:
        """Returns: A string representation of the actuator group."""
        """Returns: 执行器组的字符串表示。"""
        # resolve joint indices for printing
        joint_indices = self.joint_indices
        if joint_indices == slice(None):
            joint_indices = list(range(self.num_joints))
        # resolve model type (implicit or explicit)
        model_type = "implicit" if self.is_implicit_model else "explicit"

        return (
            f"<class {self.__class__.__name__}> object:\n"
            f"\tModel type            : {model_type}\n"
            f"\tNumber of joints      : {self.num_joints}\n"
            f"\tJoint names expression: {self.cfg.joint_names_expr}\n"
            f"\tJoint names           : {self.joint_names}\n"
            f"\tJoint indices         : {joint_indices}\n"
        )

    """
    Properties.
    """
    """属性。
    """

    @property
    def num_joints(self) -> int:
        """Number of actuators in the group."""
        """集团中的执行器数量"""
        return len(self._joint_names)

    @property
    def joint_names(self) -> list[str]:
        """Articulation's joint names that are part of the group."""
        """关节的共同名字是集团的一部分。"""
        return self._joint_names

    @property
    def joint_indices(self) -> slice | torch.Tensor:
        """Articulation's joint indices that are part of the group.

        Note:
            If :obj:`slice(None)` is returned, then the group contains all the joints in the articulation.
            We do this to avoid unnecessary indexing of the joints for performance reasons.
        """
        """关节的关节索引是组的一部分。

        说明：
            If :转换为obj:`slice(None)`，然后组包含关节中的所有关节。
            我们这样做是为了避免由于性能原因，
        """
        return self._joint_indices

    """
    Operations.
    """
    """操作。
    """

    @abstractmethod
    def reset(self, env_ids: Sequence[int]):
        """Reset the internals within the group.

        Args:
            env_ids: List of environment IDs to reset.
        """
        """在组内部重置。

        参数：
            env_ids: 设置环境 IDs的列表。
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Process the actuator group actions and compute the articulation actions.

        It computes the articulation actions based on the actuator model type

        Args:
            control_action: The joint action instance comprising of the desired joint positions, joint velocities
                and (feed-forward) joint efforts.
            joint_pos: The current joint positions of the joints in the group. Shape is (num_envs, num_joints).
            joint_vel: The current joint velocities of the joints in the group. Shape is (num_envs, num_joints).

        Returns:
            The computed desired joint positions, joint velocities and joint efforts.
        """
        """处理动机组操作和计算关节操作。

        它根据执行器模型类型计算关节动作

        参数：
            control_action: 联合动作实例，包括所需的关节位置，关节速度和 (向前) 联合努力。
            joint_pos: 组中关节的当前关节位置。
                       形状是 (num_envs，num_joints)。
            joint_vel: 组中的关节的当前关节速度。
                       形状是 (num_envs，num_joints)。

        返回：
            计算了所需的关节位置，关节速度和联合努力。
        """
        raise NotImplementedError

    """
    Helper functions.
    """
    """辅助函数。
    """

    def _record_actuator_resolution(self, cfg_val, new_val, usd_val, joint_names, joint_ids, actuator_param: str):
        if actuator_param not in self.joint_property_resolution_table:
            self.joint_property_resolution_table[actuator_param] = []
        table = self.joint_property_resolution_table[actuator_param]

        ids = joint_ids if isinstance(joint_ids, torch.Tensor) else list(range(len(joint_names)))
        for idx, name in enumerate(joint_names):
            cfg_val_log = "Not Specified" if cfg_val is None else float(new_val[idx])
            default_usd_val = usd_val if isinstance(usd_val, (float, int)) else float(usd_val[0][idx])
            applied_val_log = default_usd_val if cfg_val is None else float(new_val[idx])
            table.append([name, int(ids[idx]), default_usd_val, cfg_val_log, applied_val_log])

    def _parse_joint_parameter(
        self, cfg_value: float | dict[str, float] | None, default_value: float | torch.Tensor | None
    ) -> torch.Tensor:
        """Parse the joint parameter from the configuration.

        Args:
            cfg_value: The parameter value from the configuration. If None, then use the default value.
            default_value: The default value to use if the parameter is None. If it is also None,
                then an error is raised.

        Returns:
            The parsed parameter value.

        Raises:
            TypeError: If the parameter value is not of the expected type.
            TypeError: If the default value is not of the expected type.
            ValueError: If the parameter value is None and no default value is provided.
            ValueError: If the default value tensor is the wrong shape.
        """
        """从配置中分析联合参数。

        参数：
            cfg_value: 配置中的参数值。
                       如果 None，则使用默认值。
            default_value: 如果参数是None，则使用的默认值。
                           如果它也是None，则出现错误。

        返回：
            分析参数值。

        异常：
            TypeError: 如果参数值不符合预期类型。
            TypeError: 如果默认值不符合预期类型。
            ValueError: 如果参数值为 None，并且没有提供默认值。
            ValueError: 如果默认值子是错误的形状。
        """
        # create parameter buffer
        param = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        # parse the parameter
        if cfg_value is not None:
            if isinstance(cfg_value, (float, int)):
                # if float, then use the same value for all joints
                param[:] = float(cfg_value)
            elif isinstance(cfg_value, dict):
                # if dict, then parse the regular expression
                indices, _, values = string_utils.resolve_matching_names_values(cfg_value, self.joint_names)
                # note: need to specify type to be safe (e.g. values are ints, but we want floats)
                param[:, indices] = torch.tensor(values, dtype=torch.float, device=self._device)
            else:
                raise TypeError(
                    f"Invalid type for parameter value: {type(cfg_value)} for "
                    + f"actuator on joints {self.joint_names}. Expected float or dict."
                )
        elif default_value is not None:
            if isinstance(default_value, (float, int)):
                # if float, then use the same value for all joints
                param[:] = float(default_value)
            elif isinstance(default_value, torch.Tensor):
                # if tensor, then use the same tensor for all joints
                if default_value.shape == (self._num_envs, self.num_joints):
                    param = default_value.float()
                else:
                    raise ValueError(
                        "Invalid default value tensor shape.\n"
                        f"Got: {default_value.shape}\n"
                        f"Expected: {(self._num_envs, self.num_joints)}"
                    )
            else:
                raise TypeError(
                    f"Invalid type for default value: {type(default_value)} for "
                    + f"actuator on joints {self.joint_names}. Expected float or Tensor."
                )
        else:
            raise ValueError("The parameter value is None and no default value is provided.")

        return param

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Clip the desired torques based on the motor limits.

        Args:
            desired_torques: The desired torques to clip.

        Returns:
            The clipped torques.
        """
        """根据发动机限制，切断所需的扭矩。

        参数：
            desired_torques: 需要的扭矩来切断。

        返回：
            切断的扭矩。
        """
        return torch.clip(effort, min=-self.effort_limit, max=self.effort_limit)
