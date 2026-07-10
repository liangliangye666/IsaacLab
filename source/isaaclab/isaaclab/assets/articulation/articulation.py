# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from prettytable import PrettyTable

import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager
from pxr import PhysxSchema, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg, ImplicitActuator
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.version import get_isaac_sim_version
from isaaclab.utils.wrench_composer import WrenchComposer

from ..asset_base import AssetBase
from .articulation_data import ArticulationData

if TYPE_CHECKING:
    from .articulation_cfg import ArticulationCfg

# import logger
logger = logging.getLogger(__name__)


class Articulation(AssetBase):
    """An articulation asset class.

    An articulation is a collection of rigid bodies connected by joints. The joints can be either
    fixed or actuated. The joints can be of different types, such as revolute, prismatic, D-6, etc.
    However, the articulation class has currently been tested with revolute and prismatic joints.
    The class supports both floating-base and fixed-base articulations. The type of articulation
    is determined based on the root joint of the articulation. If the root joint is fixed, then
    the articulation is considered a fixed-base system. Otherwise, it is considered a floating-base
    system. This can be checked using the :attr:`Articulation.is_fixed_base` attribute.

    For an asset to be considered an articulation, the root prim of the asset must have the
    `USD ArticulationRootAPI`_. This API is used to define the sub-tree of the articulation using
    the reduced coordinate formulation. On playing the simulation, the physics engine parses the
    articulation root prim and creates the corresponding articulation in the physics engine. The
    articulation root prim can be specified using the :attr:`AssetBaseCfg.prim_path` attribute.

    The articulation class also provides the functionality to augment the simulation of an articulated
    system with custom actuator models. These models can either be explicit or implicit, as detailed in
    the :mod:`isaaclab.actuators` module. The actuator models are specified using the
    :attr:`ArticulationCfg.actuators` attribute. These are then parsed and used to initialize the
    corresponding actuator models, when the simulation is played.

    During the simulation step, the articulation class first applies the actuator models to compute
    the joint commands based on the user-specified targets. These joint commands are then applied
    into the simulation. The joint commands can be either position, velocity, or effort commands.
    As an example, the following snippet shows how this can be used for position commands:

    .. code-block:: python

        # an example instance of the articulation class
        my_articulation = Articulation(cfg)

        # set joint position targets
        my_articulation.set_joint_position_target(position)
        # propagate the actuator models and apply the computed commands into the simulation
        my_articulation.write_data_to_sim()

        # step the simulation using the simulation context
        sim_context.step()

        # update the articulation state, where dt is the simulation time step
        my_articulation.update(dt)

    .. _`USD ArticulationRootAPI`: https://openusd.org/dev/api/class_usd_physics_articulation_root_a_p_i.html

    """
    """一个关节资产类别。

    关节是通过关节连接的硬体集合。
    关节可以固定或激活。
    关节可以有不同类型，如转动， pris体，D-6，等。
    然而，关节阶级目前已经通过转动和结进行了测试。
    该类支持浮式和固定式关节。
    关节的类型根据关节的根关节来确定。
    如果根关节是固定的，则关节被认为是固定基础系统。
    否则，它将被视为浮动基座系统。
    这可以通过:attr:`Articulation.is_fixed_base`属性检查。

    为了使资产被视为关节，资产的根prim必须具有`USD ArticulationRootAPI`_。
    这种API用于使用缩小坐标配方定义关节的子树。
    在播放仿真时，物理引擎分析关节根prim，并在物理引擎中创建相应的关节。
    使用:attr:`AssetBaseCfg.prim_path`属性可以指定关节根 prim。

    关节类还提供了增强关节系统仿真的功能，使用定制动机模型。
    这些模型可以是明确的或隐含的，
    the :模块:`isaaclab.actuators`
         执行器模型使用
    一个":attr:`ArticulationCfg.actuators`"属性。
    然后分析并用于启动相应的动机模型，当仿真播放时。

    在仿真阶段，关节类首先应用动机模型来计算基于用户指定目标的联合命令。
    然后将这些联合命令应用到仿真中。
    联合命令可以是位置，速度或努力命令。
    作为一个例子，下面的摘录显示了如何使用这一点为位置命令:

    .. code-block:: python

        # an example instance of the articulation class
        my_articulation = Articulation(cfg)

        # set joint position targets
        my_articulation.set_joint_position_target(position)
        # propagate the actuator models and apply the computed commands into the simulation
        my_articulation.write_data_to_sim()

        # step the simulation using the simulation context
        sim_context.step()

        # update the articulation state, where dt is the simulation time step
        my_articulation.update(dt)

    .. _`USD ArticulationRootAPI`: https://openusd.org/dev/api/class_usd_physics_articulation_root_a_p_i.html
    """

    cfg: ArticulationCfg
    """Configuration instance for the articulations."""
    """关节的配置实例。"""

    actuators: dict[str, ActuatorBase]
    """Dictionary of actuator instances for the articulation.

    The keys are the actuator names and the values are the actuator instances. The actuator instances
    are initialized based on the actuator configurations specified in the :attr:`ArticulationCfg.actuators`
    attribute. They are used to compute the joint commands during the :meth:`write_data_to_sim` function.
    """
    """关节动机实例字典。

    关键是执行器名称，值是执行器实例。
    根据:attr:`ArticulationCfg.actuators`属性所指定的执行器配置，执行器实例启动。
    它们用于计算:meth:`write_data_to_sim`函数期间的联合命令。
    """

    def __init__(self, cfg: ArticulationCfg):
        """Initialize the articulation.

        Args:
            cfg: A configuration instance.
        """
        """开始关节。

        参数：
            cfg: 一个配置实例。
        """
        super().__init__(cfg)

    """
    Properties
    """
    """产品
    """

    @property
    def data(self) -> ArticulationData:
        return self._data

    @property
    def num_instances(self) -> int:     # 并行环境数量。所有数据 tensor 的第0维大小，示例值 4096
        return self.root_physx_view.count

    @property
    def is_fixed_base(self) -> bool:
        """Whether the articulation is a fixed-base or floating-base system."""
        """关节是固定基或浮基系统。"""
        return self.root_physx_view.shared_metatype.fixed_base
    '''
    是否固定底座。True = 机械臂基座固定在桌上，False = 四足/人形机器人可以自由移动
    '''

    @property
    def num_joints(self) -> int:
        """Number of joints in articulation."""
        """关节的数量"""
        return self.root_physx_view.shared_metatype.dof_count
    '''
    可驱动关节数量（自由度 DOF）
    '''

    @property
    def num_fixed_tendons(self) -> int:
        """Number of fixed tendons in articulation."""
        """关节的固定节数量"""
        return self.root_physx_view.max_fixed_tendons
    '''
    固定肌腱数量（一种柔体传动机制）
    '''

    @property
    def num_spatial_tendons(self) -> int:
        """Number of spatial tendons in articulation."""
        """关节中的空间的数量。"""
        return self.root_physx_view.max_spatial_tendons
    '''
    空间肌腱数量
    '''

    @property
    def num_bodies(self) -> int:
        """Number of bodies in articulation."""
        """关节体的数量"""
        return self.root_physx_view.shared_metatype.link_count
    '''
    连杆（刚体）数量，包括 base_link 和所有子连杆，示例值 13（四足 1+4×3）
    '''

    '''
    这些告诉你"每个关节/连杆叫什么名字"。名字的顺序和数据 tensor 的列顺序是一一对应的：
        joint_names[0] = "left_hip_joint"   ←→  joint_pos[:, 0]
        joint_names[1] = "left_knee_joint"  ←→  joint_pos[:, 1]
        ...
        body_names[0] = "base_link"         ←→  body_link_pose_w[:, 0, :]
        body_names[1] = "left_thigh_link"   ←→  body_link_pose_w[:, 1, :]
    这在调试和代码可读性上非常有用。比如你想找"左脚"的索引：
        foot_idx = robot.body_names.index("left_foot_link")
        foot_pos = robot.data.body_link_pos_w[:, foot_idx, :]
    '''

    @property
    def joint_names(self) -> list[str]:
        """Ordered names of joints in articulation."""
        """关节的名字在关节中排列。"""
        return self.root_physx_view.shared_metatype.dof_names

    @property
    def fixed_tendon_names(self) -> list[str]:
        """Ordered names of fixed tendons in articulation."""
        """按顺序命名的固定节。"""
        return self._fixed_tendon_names

    @property
    def spatial_tendon_names(self) -> list[str]:
        """Ordered names of spatial tendons in articulation."""
        """在关节中排列的空间肌肉名称。"""
        return self._spatial_tendon_names

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in articulation."""
        """列出了各个尸体的名字。"""
        return self.root_physx_view.shared_metatype.link_names

    @property
    def root_physx_view(self) -> physx.ArticulationView:
        """Articulation view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        """对资产的关节视图 (PhysX)。

        说明：
            用这种观点谨慎。
            它需要以特定的方式处理子。
        """
        return self._root_physx_view
    '''
    这是 PhysX Tensor API 的底层视图。
        警告标记 Note: Use this view with caution 是认真的——这个接口返回的 tensor 是 C++ 内存的直接映射，不像 ArticulationData 帮你做了 .clone() 和四元数格式转换。
        滥用它可能直接破坏仿真状态。
        除非你在写底层扩展，否则请用 data 属性代替。
    '''

    @property
    def instantaneous_wrench_composer(self) -> WrenchComposer:
        """Instantaneous wrench composer.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer are only valid for the current simulation step. At the end of the simulation step, the wrenches set
        to this object are discarded. This is useful to apply forces that change all the time, things like drag forces
        for instance.

        Note:
            Permanent wrenches are composed into the instantaneous wrench before the instantaneous wrenches are
            applied to the simulation.
        """
        """立刻的 w钥匙作曲家。

        返回一个:class:`~isaaclab.utils.wrench_composer.WrenchComposer`实例。
        添加或设置到此钥匙组件的关键仅适用于当前仿真步骤。
        在仿真步骤结束时，将对此物体设置的 w钥匙丢弃。
        这对于不断变化的力量来说是有用的。
        for instance.

        说明：
            在将瞬间的钥匙应用于仿真之前，永久的钥匙组成即时的钥匙。
        """
        return self._instantaneous_wrench_composer
    
    '''
    instantaneous_wrench_composer vs permanent_wrench_composer
        Wrench = 力（force）+ 力矩（torque）的组合向量 [Fx, Fy, Fz, τx, τy, τz]，形状 (N, B, 6)。
        这两个 composer 提供了给机器人施加外力的接口：
            ┌──────────────────────────────────────────────────────┐
            │              仿 真 步 (sim step)                      │
            │                                                      │
            │  ① permanent_wrench_composer    ← 持续力（如电机推力）│
            │         ↓                                            │
            │  ② instantaneous_wrench_composer ← 临时力（如空气阻力）│
            │         ↓                                            │
            │  ③ 合并后施加到 PhysX                                │
            │                                                      │
            │  ④ 清空 instantaneous_wrench_composer ← 仅本次有效    │
            │                                                      │
            │  ⑤ permanent_wrench_composer 保留 ← 下一步继续生效    │
            └──────────────────────────────────────────────────────┘
        类型	                         生命周期	                        典型用途
        permanent_wrench_composer	    持续存在，除非手动清除	            电机推力、恒定风力
        instantaneous_wrench_composer	只在本仿真步有效，下一步自动清零	    空气阻力（依赖实时速度）、碰撞冲量、一次性扰动
    '''

    @property
    def permanent_wrench_composer(self) -> WrenchComposer:
        """Permanent wrench composer.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer are persistent and are applied to the simulation at every step. This is useful to apply forces that
        are constant over a period of time, things like the thrust of a motor for instance.

        Note:
            Permanent wrenches are composed into the instantaneous wrench before the instantaneous wrenches are
            applied to the simulation.
        """
        """一个永久的 w钥匙作曲家。

        返回一个:class:`~isaaclab.utils.wrench_composer.WrenchComposer`实例。
        加入或设置到这个匙组件的关键是持久的，并且在每一步都应用于仿真。
        这对于在时间段内恒定的力量来说是有用的，例如电机的推力。

        说明：
            在将瞬间的钥匙应用于仿真之前，永久的钥匙组成即时的钥匙。
        """
        return self._permanent_wrench_composer
    '''
    使用示例
        # 获取机器人基本信息
        robot = Articulation(cfg)
        print(f"并行环境数: {robot.num_instances}")
        print(f"是否是固定底座: {robot.is_fixed_base}")
        print(f"关节数: {robot.num_joints}, 连杆数: {robot.num_bodies}")
        print(f"关节名: {robot.joint_names}")
        print(f"连杆名: {robot.body_names}")

        # 通过名称找索引
        hip_idx = robot.joint_names.index("left_hip_joint")
        hip_pos = robot.data.joint_pos[:, hip_idx]  # 读取左髋关节角度

        # 施加瞬时外力（比如推一下机器人身体）
        from isaaclab.utils.wrench_composer import WrenchComposer
        # 向所有环境的 base_link (body 0) 施加向上的力
        robot.instantaneous_wrench_composer.add(
            body_index=0,
            wrench=torch.tensor([0.0, 0.0, 100.0, 0.0, 0.0, 0.0]),  # z方向 100N
        )
        # 这一步结束时力会自动清零
    '''

    """
    Operations.
    """
    """操作。
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        '''
        类型注解 Sequence[int] | None 意味着你可以传三种值：
            None：复位所有并行环境
            [3, 7, 12]：只复位第 3、7、12 号环境
            slice(0, 100)：复位前 100 个环境
        '''
        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = slice(None)
        # reset actuators
        for actuator in self.actuators.values():
            actuator.reset(env_ids)
            '''
            复位执行器
                执行器（Actuator）是 Isaac Lab 中的"电机模型"。PD 执行器内部维护着控制器的历史误差（积分项 I），这些历史状态在 episode 重置时必须清零。
            '''
        # reset external wrenches.
        self._instantaneous_wrench_composer.reset(env_ids)
        self._permanent_wrench_composer.reset(env_ids)
        '''
        复位外力
            清除上一轮 episode 施加的所有外力（wind、push 等）
        '''

    '''
    负责把策略输出的动作命令真正"注入"到 PhysX 物理引擎中
        施加外力 (wrench)
        执行器模型计算 (PD 控制等)
        写入 PhysX (力矩/位置/速度目标)
    '''
    def write_data_to_sim(self):
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        """在仿真中写出外部钥匙和联合命令。

        如果存在任何明确的执行器，则执行器模型用于计算联合命令。
        否则，联合命令将直接设置在仿真中。

        说明：
            我们写出仿真的外部关键，因为这个函数在仿真步骤之前被调用。
            这确保在每个仿真步骤上使用外部钥匙。
        """
        # write external wrench     阶段 1：外力处理
        if self._instantaneous_wrench_composer.active or self._permanent_wrench_composer.active:
            if self._instantaneous_wrench_composer.active:
                # Compose instantaneous wrench with permanent wrench    # ① merge: 把永久力合入瞬时力
                self._instantaneous_wrench_composer.add_forces_and_torques(
                    forces=self._permanent_wrench_composer.composed_force,
                    torques=self._permanent_wrench_composer.composed_torque,
                    body_ids=self._ALL_BODY_INDICES_WP,
                    env_ids=self._ALL_INDICES_WP,
                )
                # Apply both instantaneous and permanent wrench to the simulation   # ② apply: 合并后的总力施加到 PhysX
                self.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=self._instantaneous_wrench_composer.composed_force_as_torch.view(-1, 3),
                    torque_data=self._instantaneous_wrench_composer.composed_torque_as_torch.view(-1, 3),
                    position_data=None,
                    indices=self._ALL_INDICES,
                    is_global=False,
                )
            else:   # 只有永久力，直接施加
                # Apply permanent wrench to the simulation
                self.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=self._permanent_wrench_composer.composed_force_as_torch.view(-1, 3),
                    torque_data=self._permanent_wrench_composer.composed_torque_as_torch.view(-1, 3),
                    position_data=None,
                    indices=self._ALL_INDICES,
                    is_global=False,
                )
        self._instantaneous_wrench_composer.reset() # ③ 清空瞬时力（永久力保留给下一步）
        '''
        关键常量：
            变量	                        含义
            self._ALL_INDICES	            torch.arange(N)，所有环境索引
            self._ALL_INDICES_WP	        同上，转为 Warp 格式
            self._ALL_BODY_INDICES_WP	    所有 body 索引，Warp 格式
        为什么用两个格式？
            WrenchComposer 内部用 Warp（GPU 加速的 Python 库）做高效计算，而 root_physx_view.apply_forces_and_torques_at_position 用 PyTorch tensor。
            所以 _ALL_INDICES 是 PyTorch 的，_ALL_INDICES_WP 是 Warp 的。
        .view(-1, 3) 的作用：
            WrenchComposer 内部存储的形状是 (N, B, 3)（N 个环境 × B 个 body × 3 维力/力矩），PhysX API 需要 (N*B, 3) 的展平形状。
            .view(-1, 3) 就是做这个展平。
        '''

        # apply actuator models     阶段 2：执行器模型
        self._apply_actuator_model()
        # write actions into simulation     阶段 3：写入 PhysX
        self.root_physx_view.set_dof_actuation_forces(self._joint_effort_target_sim, self._ALL_INDICES) # ③ 力矩目标写入 PhysX（所有执行器都用）
        # position and velocity targets only for implicit actuators
        if self._has_implicit_actuators:    # ④ 位置/速度目标——仅隐式执行器需要
            self.root_physx_view.set_dof_position_targets(self._joint_pos_target_sim, self._ALL_INDICES)
            self.root_physx_view.set_dof_velocity_targets(self._joint_vel_target_sim, self._ALL_INDICES)
        '''
        set_dof_actuation_forces：
            对所有执行器，把算好的力矩 τ 写入 PhysX。PD 控制器算出力矩，PhysX 用这个力矩驱动关节。
        set_dof_position_targets / set_dof_velocity_targets：
            只有隐式执行器需要。隐式执行器直接把用户目标透传给 PhysX 内置 PD，所以需要写位置和速度目标。
        '''

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """
    """搜索器
    """

    '''
    名字到索引的查找器
    输入/输出参数
        参数	            类型	                说明
        name_keys	        str | Sequence[str]	    正则表达式（字符串）或正则表达式列表。例如 ".*foot" 匹配所有以 "foot" 结尾的名字，["left.*", "right.*"] 分别匹配左、右侧
        preserve_order	    bool，默认 False	    True = 按 body_names 原始顺序排列结果；False = 按正则表达式匹配顺序排列
        返回值 [0]	        list[int]	            匹配到的连杆索引列表。可直接用于索引数据 tensor，如 body_link_pose_w[:, indices, :]
        返回值 [1]	        list[str]	            匹配到的连杆名称列表，和索引一一对应
    '''
    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        """根据名字键，在关节中找到尸体。

        请查看:meth:`isaaclab.utils.string_utils.resolve_matching_names`函数，了解更多关于名称匹配的信息。

        参数：
            name_keys: 一个正则表达式或一个与体名相匹配的正则表达式列表。
            preserve_order: 在输出中是否保留名称键的顺序。
                            默认为 False。

        返回：
            一个包含身体指标和名称的列表。
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)
    '''
    使用示例
        # 找所有脚（正则 .*FOOT 匹配 _FOOT 结尾）
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        # 结果: ([4, 7, 10, 13],
        #        ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"])
    '''

    def find_joints(
        self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find joints in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint names.
            joint_subset: A subset of joints to search for. Defaults to None, which means all joints
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the joint indices and names.
        """
        """根据名字键找到关节。

        请查看:func:`isaaclab.utils.string.resolve_matching_names`函数，了解更多关于名称匹配的信息。

        参数：
            name_keys: 一个正则表达式或一列正则表达式，以匹配共同名称。
            joint_subset: 关节的小组，要寻找。
                          在None上，意味着关节中的所有关节都被搜索。
            preserve_order: 在输出中是否保留名称键的顺序。
                            默认为 False。

        返回：
            一组列表包含联合索引和名称。
        """
        if joint_subset is None:
            joint_subset = self.joint_names
        # find joints
        return string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)
    '''
    和 find_bodies 的对比
        维度	    find_bodies	                        find_joints
        搜索范围	固定为 self.body_names（所有连杆）	    默认 self.joint_names，可缩小范围
        额外参数	无	                                joint_subset：限制搜索范围
        使用场景	找脚、找末端执行器	                    找手臂关节、找手指、找特定关节组
        底层引擎	resolve_matching_names	                同一个

    joint_subset 参数的设计
        这是 find_joints 独有的功能：在搜索时先限定一个子集。
                if joint_subset is None:
                    joint_subset = self.joint_names
        为什么需要 joint_subset？
            假设一个四足+手臂的复合机器人（如 Spot + 机械臂），它有 18 个关节。你想找"手臂的关节"，但手臂和腿的关节名可能都包含 joint：
                # 全部 18 个关节名：
                ["base_arm_joint1", "base_arm_joint2", "base_arm_joint3",
                "base_arm_joint4", "base_arm_joint5", "base_arm_joint6",
                "left_front_hip_joint", "left_front_knee_joint", ...]

                # 如果不用 joint_subset，搜 ".*joint.*" 会返回全部 18 个
                robot.find_joints(".*joint.*")  # → 全部 18 个

                # 先用 joint_subset 限定只搜手臂关节
                arm_joints = ["base_arm_joint1", "base_arm_joint2", ..., "base_arm_joint6"]
                robot.find_joints(".*joint.*", joint_subset=arm_joints)  # → 只返回 6 个手臂关节
            通俗类比：joint_subset 就像搜房时先"筛选区域"——你说"我只在朝阳区找三居室"（joint_subset = 朝阳区的房源列表），而不是在整个北京找。

    示例 1：精确找单个关节
        # 用配置中的名字精确匹配关节
        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        # 例如 cfg.cart_dof_name = "cart_joint"
        # 返回: ([0], ["cart_joint"])
    '''

    def find_fixed_tendons(
        self, name_keys: str | Sequence[str], tendon_subsets: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find fixed tendons in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint
                names with fixed tendons.
            tendon_subsets: A subset of joints with fixed tendons to search for. Defaults to None, which means
                all joints in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the tendon indices and names.
        """
        """在关节中找到固定的节点，根据名称键。

        请查看:func:`isaaclab.utils.string.resolve_matching_names`函数，了解更多关于名称匹配的信息。

        参数：
            name_keys: 一个正则表达式或一个正则表达式列表，以匹配固定肌的联合名称。
            tendon_subsets: 一组有固定节点的关节。
                            在None上，意味着关节中的所有关节都被搜索。
            preserve_order: 在输出中是否保留名称键的顺序。
                            默认为 False。

        返回：
            一个包含子指标和名称的列表。
        """
        if tendon_subsets is None:
            # tendons follow the joint names they are attached to
            tendon_subsets = self.fixed_tendon_names
        # find tendons
        return string_utils.resolve_matching_names(name_keys, tendon_subsets, preserve_order)

    def find_spatial_tendons(
        self, name_keys: str | Sequence[str], tendon_subsets: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find spatial tendons in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the tendon names.
            tendon_subsets: A subset of tendons to search for. Defaults to None, which means all tendons
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the tendon indices and names.
        """
        """根据名称键，找到关节中的空间。

        请查看:func:`isaaclab.utils.string.resolve_matching_names`函数，了解更多关于名称匹配的信息。

        参数：
            name_keys: 一个正则表达式或一系列正则表达式，
            tendon_subsets: 一组子要找。
                            默认为 None，这意味着关节中的所有都被搜索。
            preserve_order: 在输出中是否保留名称键的顺序。
                            默认为 False。

        返回：
            一个包含子指标和名称的列表。
        """
        if tendon_subsets is None:
            tendon_subsets = self.spatial_tendon_names
        # find tendons
        return string_utils.resolve_matching_names(name_keys, tendon_subsets, preserve_order)

    """
    Operations - State Writers.
    """
    """动作 - 国家秘书
    """
    '''
    root_state
        形状 (len(env_ids), 13)。
        顺序：[pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z,             前7个
            lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]   后6个
    '''

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选定的环境索引上的根状态。

        根状态包括卡特西亚位置，在 (w，x，y，z) 中的四元数方向以及线性和角的速度。
        所有数量都在仿真框架中。

        参数：
            root_state: 在仿真框架中的根状态。
                        形状是 (len(env_ids)，13。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_com_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置了选择的环境索引上质量状态的根中心。

        根状态包括卡特西亚位置，在 (w，x，y，z) 中的四元数方向以及线性和角的速度。
        所有数量都在仿真框架中。

        参数：
            root_state: 在仿真框架中的根状态。
                        形状是 (len(env_ids)，13。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        self.write_root_com_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_link_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选定的环境索引上根链状态。

        根状态包括卡特西亚位置，在 (w，x，y，z) 中的四元数方向以及线性和角的速度。
        所有数量都在仿真框架中。

        参数：
            root_state: 在仿真框架中的根状态。
                        形状是 (len(env_ids)，13。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_link_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选定的环境索引上的根姿势。

        根姿势包括在 (w，x，y，z) 中的卡特西亚位置和四元数方向。

        参数：
            root_pose: 在仿真框架中的根姿势。
                       形状是 (len(env_ids)， 7)。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        self.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

    def write_root_link_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选定的环境索引上根链接姿势。

        根姿势包括在 (w，x，y，z) 中的卡特西亚位置和四元数方向。

        参数：
            root_pose: 在仿真框架中的根姿势。
                       形状是 (len(env_ids)， 7)。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES

        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_link_pose_w[env_ids] = root_pose.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[env_ids, :7] = self._data.root_link_pose_w[env_ids]
        if self._data._root_state_w.data is not None:
            self._data.root_state_w[env_ids, :7] = self._data.root_link_pose_w[env_ids]

        # convert root quaternion from wxyz to xyzw
        root_poses_xyzw = self._data.root_link_pose_w.clone()
        root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")

        # Need to invalidate the buffer to trigger the update with the new state.
        self._data._body_link_pose_w.timestamp = -1.0
        self._data._body_com_pose_w.timestamp = -1.0
        self._data._body_state_w.timestamp = -1.0
        self._data._body_link_state_w.timestamp = -1.0
        self._data._body_com_state_w.timestamp = -1.0

        # set into simulation
        self.root_physx_view.set_root_transforms(root_poses_xyzw, indices=physx_env_ids)

    def write_root_com_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选择的环境索引上，

        根姿势包括在 (w，x，y，z) 中的卡特西亚位置和四元数方向。
        导向是惯性的主要轴的导向。

        参数：
            root_pose: 在仿真框架中，质量的根中心姿势。
                       形状是 (len(env_ids)， 7)。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        # resolve all indices
        if env_ids is None:
            local_env_ids = slice(env_ids)
        else:
            local_env_ids = env_ids

        # set into internal buffers
        self._data.root_com_pose_w[local_env_ids] = root_pose.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_com_state_w.data is not None:
            self._data.root_com_state_w[local_env_ids, :7] = self._data.root_com_pose_w[local_env_ids]

        # get CoM pose in link frame
        com_pos_b = self.data.body_com_pos_b[local_env_ids, 0, :]
        com_quat_b = self.data.body_com_quat_b[local_env_ids, 0, :]
        # transform input CoM pose to link frame
        root_link_pos, root_link_quat = math_utils.combine_frame_transforms(
            root_pose[..., :3],
            root_pose[..., 3:7],
            math_utils.quat_apply(math_utils.quat_inv(com_quat_b), -com_pos_b),
            math_utils.quat_inv(com_quat_b),
        )
        root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)

        # write transformed pose in link frame to sim
        self.write_root_link_pose_to_sim(root_pose=root_link_pose, env_ids=env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置质量速度的根中心，

        速度包括线性速度 (x，y，z) 和角速度 (x，y，z) 在这个顺序中。
        NOTE: 这设定了根的质量中心的速度，而不是根框架。

        参数：
            root_velocity: 在仿真世界框架中，
                           形状是 (len(env_ids)， 6。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        self.write_root_com_velocity_to_sim(root_velocity=root_velocity, env_ids=env_ids)

    def write_root_com_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置质量速度的根中心，

        速度包括线性速度 (x，y，z) 和角速度 (x，y，z) 在这个顺序中。
        NOTE: 这设定了根的质量中心的速度，而不是根框架。

        参数：
            root_velocity: 在仿真世界框架中，
                           形状是 (len(env_ids)， 6。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES

        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_com_vel_w[env_ids] = root_velocity.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_com_state_w.data is not None:
            self._data.root_com_state_w[env_ids, 7:] = self._data.root_com_vel_w[env_ids]
        if self._data._root_state_w.data is not None:
            self._data.root_state_w[env_ids, 7:] = self._data.root_com_vel_w[env_ids]
        # make the acceleration zero to prevent reporting old values
        self._data.body_acc_w[env_ids] = 0.0

        # set into simulation
        self.root_physx_view.set_root_velocities(self._data.root_com_vel_w, indices=physx_env_ids)

    def write_root_link_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选定的环境索引上根链速度。

        速度包括线性速度 (x，y，z) 和角速度 (x，y，z) 在这个顺序中。
        NOTE: 这设定了根框架的速度而不是根质中心。

        参数：
            root_velocity: 在仿真世界框架中的根框架速度。
                           形状是 (len(env_ids)， 6。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        # resolve all indices
        if env_ids is None:
            local_env_ids = slice(env_ids)
        else:
            local_env_ids = env_ids

        # set into internal buffers
        self._data.root_link_vel_w[local_env_ids] = root_velocity.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[local_env_ids, 7:] = self._data.root_link_vel_w[local_env_ids]

        # get CoM pose in link frame
        quat = self.data.root_link_quat_w[local_env_ids]
        com_pos_b = self.data.body_com_pos_b[local_env_ids, 0, :]
        # transform input velocity to center of mass frame
        root_com_velocity = root_velocity.clone()
        root_com_velocity[:, :3] += torch.linalg.cross(
            root_com_velocity[:, 3:], math_utils.quat_apply(quat, com_pos_b), dim=-1
        )

        # write transformed velocity in CoM frame to sim
        self.write_root_com_velocity_to_sim(root_velocity=root_com_velocity, env_ids=env_ids)

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
    ):
        """Write joint positions and velocities to the simulation.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)).
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        """在仿真中写出关节位置和速度。

        参数：
            position: 共同位置。
                      形状是 (len(env_ids)，len(joint_ids))。
            velocity: 关节速度。
                      形状是 (len(env_ids)，len(joint_ids))。
            joint_ids: 确定目标的共同索引。
                       在 None (所有关节) 上默认设置。
            env_ids: 环境索引设定目标。
                     在 None (所有环境) 中默认设置。
        """
        # set into simulation
        self.write_joint_position_to_sim(position, joint_ids=joint_ids, env_ids=env_ids)
        self.write_joint_velocity_to_sim(velocity, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_position_to_sim(
        self,
        position: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
    ):
        """Write joint positions to the simulation.

        Args:
            position: Joint positions. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        """在仿真中写下关节位置。

        参数：
            position: 共同位置。
                      形状是 (len(env_ids)，len(joint_ids))。
            joint_ids: 确定目标的共同索引。
                       在 None (所有关节) 上默认设置。
            env_ids: 环境索引设定目标。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_pos[env_ids, joint_ids] = position
        # Need to invalidate the buffer to trigger the update with the new root pose.
        self._data._body_com_vel_w.timestamp = -1.0
        self._data._body_link_vel_w.timestamp = -1.0
        self._data._body_com_pose_b.timestamp = -1.0
        self._data._body_com_pose_w.timestamp = -1.0
        self._data._body_link_pose_w.timestamp = -1.0

        self._data._body_state_w.timestamp = -1.0
        self._data._body_link_state_w.timestamp = -1.0
        self._data._body_com_state_w.timestamp = -1.0
        # set into simulation
        self.root_physx_view.set_dof_positions(self._data.joint_pos, indices=physx_env_ids)

    def write_joint_velocity_to_sim(
        self,
        velocity: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
    ):
        """Write joint velocities to the simulation.

        Args:
            velocity: Joint velocities. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        """在仿真中写出关节速度。

        参数：
            velocity: 关节速度。
                      形状是 (len(env_ids)，len(joint_ids))。
            joint_ids: 确定目标的共同索引。
                       在 None (所有关节) 上默认设置。
            env_ids: 环境索引设定目标。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_vel[env_ids, joint_ids] = velocity
        self._data._previous_joint_vel[env_ids, joint_ids] = velocity
        self._data.joint_acc[env_ids, joint_ids] = 0.0
        # set into simulation
        self.root_physx_view.set_dof_velocities(self._data.joint_vel, indices=physx_env_ids)

    """
    Operations - Simulation Parameters Writers.
    """
    """操作 - 仿真参数编写器。
    """

    def write_joint_stiffness_to_sim(
        self,
        stiffness: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint stiffness into the simulation.

        Args:
            stiffness: Joint stiffness. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the stiffness for. Defaults to None (all joints).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
        """
        """在仿真中写关节硬度。

        参数：
            stiffness: 关节硬化。
                       形状是 (len(env_ids)，len(joint_ids))。
            joint_ids: 固度的索引。
                       在 None (所有关节) 上默认设置。
            env_ids: 环境指标设定了度。
                     在 None (所有环境) 中默认设置。
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_stiffness[env_ids, joint_ids] = stiffness
        # set into simulation
        self.root_physx_view.set_dof_stiffnesses(self._data.joint_stiffness.cpu(), indices=physx_env_ids.cpu())

    def write_joint_damping_to_sim(
        self,
        damping: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint damping into the simulation.

        Args:
            damping: Joint damping. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the damping for. Defaults to None (all joints).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
        """
        """在仿真中输入关节缩。

        参数：
            damping: 关节缩。
                     形状是 (len(env_ids)，len(joint_ids))。
            joint_ids: 设置 joint缩的关节索引。
                       在 None (所有关节) 上默认设置。
            env_ids: 环境指标设置缩。
                     在 None (所有环境) 中默认设置。
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_damping[env_ids, joint_ids] = damping
        # set into simulation
        self.root_physx_view.set_dof_dampings(self._data.joint_damping.cpu(), indices=physx_env_ids.cpu())

    def write_joint_position_limit_to_sim(
        self,
        limits: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        warn_limit_violation: bool = True,
    ):
        """Write joint position limits into the simulation.

        Args:
            limits: Joint limits. Shape is (len(env_ids), len(joint_ids), 2).
            joint_ids: The joint indices to set the limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the limits for. Defaults to None (all environments).
            warn_limit_violation: Whether to use warning or info level logging when default joint positions
                exceed the new limits. Defaults to True.
        """
        """在仿真中写出关节位置限制。

        参数：
            limits: 共同的限制。
                    形状是 (len(env_ids)，len(joint_ids)，2)。
            joint_ids: 共同索引设定限制。
                       在 None (所有关节) 上默认设置。
            env_ids: 环境索引设定限制。
                     在 None (所有环境) 中默认设置。
            warn_limit_violation: 如果默认关联位置超过新的限制时，是否使用警告或信息水平记录。
                                  默认为 True。
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_pos_limits[env_ids, joint_ids] = limits
        # update default joint pos to stay within the new limits
        if torch.any(
            (self._data.default_joint_pos[env_ids, joint_ids] < limits[..., 0])
            | (self._data.default_joint_pos[env_ids, joint_ids] > limits[..., 1])
        ):
            self._data.default_joint_pos[env_ids, joint_ids] = torch.clamp(
                self._data.default_joint_pos[env_ids, joint_ids], limits[..., 0], limits[..., 1]
            )
            violation_message = (
                "Some default joint positions are outside of the range of the new joint limits. Default joint positions"
                " will be clamped to be within the new joint limits."
            )
            if warn_limit_violation:
                # warn level will show in console
                logger.warning(violation_message)
            else:
                # info level is only written to log file
                logger.info(violation_message)
        # set into simulation
        self.root_physx_view.set_dof_limits(self._data.joint_pos_limits.cpu(), indices=physx_env_ids.cpu())

        # compute the soft limits based on the joint limits
        # TODO: Optimize this computation for only selected joints
        # soft joint position limits (recommended not to be too close to limits).
        joint_pos_mean = (self._data.joint_pos_limits[..., 0] + self._data.joint_pos_limits[..., 1]) / 2
        joint_pos_range = self._data.joint_pos_limits[..., 1] - self._data.joint_pos_limits[..., 0]
        soft_limit_factor = self.cfg.soft_joint_pos_limit_factor
        # add to data
        self._data.soft_joint_pos_limits[..., 0] = joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor
        self._data.soft_joint_pos_limits[..., 1] = joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor

    def write_joint_velocity_limit_to_sim(
        self,
        limits: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint max velocity to the simulation.

        The velocity limit is used to constrain the joint velocities in the physics engine. The joint will only
        be able to reach this velocity if the joint's effort limit is sufficiently large. If the joint is moving
        faster than this velocity, the physics engine will actually try to brake the joint to reach this velocity.

        Args:
            limits: Joint max velocity. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the max velocity for. Defaults to None (all joints).
            env_ids: The environment indices to set the max velocity for. Defaults to None (all environments).
        """
        """在仿真中写出联合最大速度。

        速度限制用于限制物理引擎中的关节速度。
        关节只能达到这个速度，如果关节的力度极限足够大。
        如果关节速度超过这个速度，物理引擎实际上会试图制关节以达到这个速度。

        参数：
            limits: 联合最大速度。
                    形状是 (len(env_ids)，len(joint_ids))。
            joint_ids: 设置最大速度的联合索引。
                       在 None (所有关节) 上默认设置。
            env_ids: 环境索引设置最大速度。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # move tensor to cpu if needed
        if isinstance(limits, torch.Tensor):
            limits = limits.to(self.device)
        # set into internal buffers
        self._data.joint_vel_limits[env_ids, joint_ids] = limits
        # set into simulation
        self.root_physx_view.set_dof_max_velocities(self._data.joint_vel_limits.cpu(), indices=physx_env_ids.cpu())

    def write_joint_effort_limit_to_sim(
        self,
        limits: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint effort limits into the simulation.

        The effort limit is used to constrain the computed joint efforts in the physics engine. If the
        computed effort exceeds this limit, the physics engine will clip the effort to this value.

        Args:
            limits: Joint torque limits. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
        """
        """在仿真中写出联合努力限制。

        在物理引擎中的计算联合努力被限制。
        如果计算的功率超过这个限度，物理引擎将把功率缩小到这个值。

        参数：
            limits: 关节扭矩限制。
                    形状是 (len(env_ids)，len(joint_ids))。
            joint_ids: 组合索引设置组合扭矩限制。
                       在 None (所有关节) 上默认设置。
            env_ids: 环境指标设定关节力矩限制。
                     在 None (所有环境) 中默认设置。
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # move tensor to cpu if needed
        if isinstance(limits, torch.Tensor):
            limits = limits.to(self.device)
        # set into internal buffers
        self._data.joint_effort_limits[env_ids, joint_ids] = limits
        # set into simulation
        self.root_physx_view.set_dof_max_forces(self._data.joint_effort_limits.cpu(), indices=physx_env_ids.cpu())

    def write_joint_armature_to_sim(
        self,
        armature: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint armature into the simulation.

        The armature is directly added to the corresponding joint-space inertia. It helps improve the
        simulation stability by reducing the joint velocities.

        Args:
            armature: Joint armature. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the joint torque limits for. Defaults to None (all joints).
            env_ids: The environment indices to set the joint torque limits for. Defaults to None (all environments).
        """
        """在仿真中输入关节 armature。

        机器直接加入相应的关节空间惯性。
        通过减少关节速度，它有助于提高仿真稳定性。

        参数：
            armature: 关节 armature。
                      形状是 (len(env_ids)，len(joint_ids))。
            joint_ids: 组合索引设置组合扭矩限制。
                       在 None (所有关节) 上默认设置。
            env_ids: 环境指标设定关节力矩限制。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_armature[env_ids, joint_ids] = armature
        # set into simulation
        self.root_physx_view.set_dof_armatures(self._data.joint_armature.cpu(), indices=physx_env_ids.cpu())

    def write_joint_friction_coefficient_to_sim(
        self,
        joint_friction_coeff: torch.Tensor | float,
        joint_dynamic_friction_coeff: torch.Tensor | float | None = None,
        joint_viscous_friction_coeff: torch.Tensor | float | None = None,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        r"""Write joint friction coefficients into the simulation.

        For Isaac Sim versions below 5.0, only the static friction coefficient is set.
        This limits the resisting force or torque up to a maximum proportional to the transmitted
        spatial force: :math:`\|F_{resist}\| \leq \mu_s \, \|F_{spatial}\|`.

        For Isaac Sim versions 5.0 and above, the static, dynamic, and viscous friction coefficients
        are set. The model combines Coulomb (static & dynamic) friction with a viscous term:

        - Static friction :math:`\mu_s` defines the maximum effort that prevents motion at rest.
        - Dynamic friction :math:`\mu_d` applies once motion begins and remains constant during motion.
        - Viscous friction :math:`c_v` is a velocity-proportional resistive term.

        Args:
            joint_friction_coeff: Static friction coefficient :math:`\mu_s`.
                Shape is (len(env_ids), len(joint_ids)). Scalars are broadcast to all selections.
            joint_dynamic_friction_coeff: Dynamic (Coulomb) friction coefficient :math:`\mu_d`.
                Same shape as above. If None, the dynamic coefficient is not updated.
            joint_viscous_friction_coeff: Viscous friction coefficient :math:`c_v`.
                Same shape as above. If None, the viscous coefficient is not updated.
            joint_ids: The joint indices to set the friction coefficients for. Defaults to None (all joints).
            env_ids: The environment indices to set the friction coefficients for. Defaults to None (all environments).
        """
        """在仿真中写出关节摩擦系数。

        对于低于5.0的Isaac Sim版本，只有静态摩擦系数设置。
        这将阻力或扭矩限制在与传输的空间力最大比例:`\|F_{resist}\| \leq \mu_s \， \|F_{spatial}\|`。

        对于Isaac Sim版本5.0及以上，静态，动态和粘性摩擦系数设置。
        该模型将Coulomb (静态和动态) 的摩擦与粘性项结合在一起:

        - 静态摩擦:数学:`\mu_s` 定义了阻碍静止运动的最大努力。
        - 动态摩擦:数学:`\mu_d` 一旦运动开始，就适用于运动过程中。
        - 粘性摩擦:数学:`c_v`是速度比例的阻力项。

        参数：
            joint_friction_coeff: 静态摩擦系数:`\mu_s`
                                  形状是 (len(env_ids)，len(joint_ids))。
                                  度将播放到所有选择。
            joint_dynamic_friction_coeff: 动态摩擦系数:数学:`\mu_d`。
                                          同上面的形状。
                                          如果None，动态系数不会更新。
            joint_viscous_friction_coeff: 粘性摩擦系数:数学:`c_v`
                                          同上面的形状。
                                          如果None，粘度系数不会更新。
            joint_ids: 固定索引设置摩擦系数。
                       在 None (所有关节) 上默认设置。
            env_ids: 环境索引设置摩擦系数。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_friction_coeff[env_ids, joint_ids] = joint_friction_coeff

        # if dynamic or viscous friction coeffs are provided, set them too
        if joint_dynamic_friction_coeff is not None:
            self._data.joint_dynamic_friction_coeff[env_ids, joint_ids] = joint_dynamic_friction_coeff
        if joint_viscous_friction_coeff is not None:
            self._data.joint_viscous_friction_coeff[env_ids, joint_ids] = joint_viscous_friction_coeff

        # move the indices to cpu
        physx_envs_ids_cpu = physx_env_ids.cpu()

        # set into simulation
        if get_isaac_sim_version().major < 5:
            self.root_physx_view.set_dof_friction_coefficients(
                self._data.joint_friction_coeff.cpu(), indices=physx_envs_ids_cpu
            )
        else:
            friction_props = self.root_physx_view.get_dof_friction_properties()
            friction_props[physx_envs_ids_cpu, :, 0] = self._data.joint_friction_coeff[physx_envs_ids_cpu, :].cpu()

            # only set dynamic and viscous friction if provided
            if joint_dynamic_friction_coeff is not None:
                friction_props[physx_envs_ids_cpu, :, 1] = self._data.joint_dynamic_friction_coeff[
                    physx_envs_ids_cpu, :
                ].cpu()

            # only set viscous friction if provided
            if joint_viscous_friction_coeff is not None:
                friction_props[physx_envs_ids_cpu, :, 2] = self._data.joint_viscous_friction_coeff[
                    physx_envs_ids_cpu, :
                ].cpu()

            self.root_physx_view.set_dof_friction_properties(friction_props, indices=physx_envs_ids_cpu)

    def write_joint_dynamic_friction_coefficient_to_sim(
        self,
        joint_dynamic_friction_coeff: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        if get_isaac_sim_version().major < 5:
            logger.warning("Setting joint dynamic friction coefficients are not supported in Isaac Sim < 5.0")
            return
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_dynamic_friction_coeff[env_ids, joint_ids] = joint_dynamic_friction_coeff
        # set into simulation
        friction_props = self.root_physx_view.get_dof_friction_properties()
        friction_props[physx_env_ids.cpu(), :, 1] = self._data.joint_dynamic_friction_coeff[physx_env_ids, :].cpu()
        self.root_physx_view.set_dof_friction_properties(friction_props, indices=physx_env_ids.cpu())

    def write_joint_viscous_friction_coefficient_to_sim(
        self,
        joint_viscous_friction_coeff: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        if get_isaac_sim_version().major < 5:
            logger.warning("Setting joint viscous friction coefficients are not supported in Isaac Sim < 5.0")
            return
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set into internal buffers
        self._data.joint_viscous_friction_coeff[env_ids, joint_ids] = joint_viscous_friction_coeff
        # set into simulation
        friction_props = self.root_physx_view.get_dof_friction_properties()
        friction_props[physx_env_ids.cpu(), :, 2] = self._data.joint_viscous_friction_coeff[physx_env_ids, :].cpu()
        self.root_physx_view.set_dof_friction_properties(friction_props, indices=physx_env_ids.cpu())

    """
    Operations - Setters.
    """
    """运营 - 设置器。
    """

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        positions: torch.Tensor | None = None,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        is_global: bool = False,
    ):
        """Set external force and torque to apply on the asset's bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step. Optionally, set the position to apply the
        external wrench at (in the local link frame of the bodies).

        .. caution::
            If the function is called with empty forces and torques, then this function disables the application
            of external wrench to the simulation.

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=torch.zeros(0, 3), torques=torch.zeros(0, 3))

        .. note::
            This function does not apply the external wrench to the simulation. It only fills the buffers with
            the desired values. To apply the external wrench, call the :meth:`write_data_to_sim` function
            right before the simulation step.

        Args:
            forces: External forces in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            torques: External torques in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            positions: Positions to apply external wrench. Shape is (len(env_ids), len(body_ids), 3). Defaults to None.
            body_ids: Body indices to apply external wrench to. Defaults to None (all bodies).
            env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
            is_global: Whether to apply the external wrench in the global frame. Defaults to False. If set to False,
                the external wrench is applied in the link frame of the articulations' bodies.
        """
        """设置外部力和扭矩应在本地框架中的资产体上应用。

        在许多应用中，我们希望在一段时间内 (例如，在策略控制期间) 保持对硬体的外力稳定。
        这种功能使我们能够将外部力和扭矩存储在缓冲器中，然后在每一步都应用于仿真。
        选择地设置将外部钥匙应用到 (在机器的本地链接框中)。

        .. 谨慎::
            如果函数被用空力和扭矩调用，则该函数将外部钥匙被禁用在仿真中。

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=torch.zeros(0, 3), torques=torch.zeros(0, 3))

        .. 说明::
            这项函数不适用于仿真的外部关键。
            它只用所需的值填充缓冲器。
            在仿真步骤之前，请调用:meth:`write_data_to_sim`函数。

        参数：
            forces: 在身体的局部框架中，
                    形状是 (len(env_ids)，len(body_ids)，3)。
            torques: 身体的局部体内外部扭矩。
                     形状是 (len(env_ids)，len(body_ids)，3)。
            positions: 外部钥匙的位置。
                       形状是 (len(env_ids)，len(body_ids)，3)。
                       默认为 None。
            body_ids: 机体指标应用外部钥匙。
                      在None (所有机体) 上默认设置。
            env_ids: 环境索引应使用外部匙。
                     在 None 中默认设置 (所有实例)。
            is_global: 在全球框架中是否应使用外部 w钥匙。
                       默认为 False。
                       如果设置为False，则将外部钥匙应用在关节体的链接框架中。
        """
        logger.warning(
            "The function 'set_external_force_and_torque' will be deprecated in a future release. Please"
            " use 'permanent_wrench_composer.set_forces_and_torques' instead."
        )
        if forces is None and torques is None:
            logger.warning("No forces or torques provided. No permanent external wrench will be applied.")

        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES_WP
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        else:
            env_ids = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        # -- body_ids
        if body_ids is None:
            body_ids = self._ALL_BODY_INDICES_WP
        elif isinstance(body_ids, slice):
            body_ids = wp.from_torch(
                torch.arange(self.num_bodies, dtype=torch.int32, device=self.device)[body_ids], dtype=wp.int32
            )
        elif not isinstance(body_ids, torch.Tensor):
            body_ids = wp.array(body_ids, dtype=wp.int32, device=self.device)
        else:
            body_ids = wp.from_torch(body_ids.to(torch.int32), dtype=wp.int32)

        # Write to wrench composer
        self._permanent_wrench_composer.set_forces_and_torques(
            forces=wp.from_torch(forces, dtype=wp.vec3f) if forces is not None else None,
            torques=wp.from_torch(torques, dtype=wp.vec3f) if torques is not None else None,
            positions=wp.from_torch(positions, dtype=wp.vec3f) if positions is not None else None,
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=is_global,
        )

    def set_joint_position_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | slice | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set joint position targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint position targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        """在内部缓冲器中设置关节位置目标。

        这项功能不适用于仿真的共同目标。
        它只用所需的值填充缓冲器。
        为了应用共同目标，请调用:meth:`write_data_to_sim`函数。

        参数：
            target: 共同定位目标。
                    形状是 (len(env_ids)，len(joint_ids))。
            joint_ids: 确定目标的共同索引。
                       在 None (所有关节) 上默认设置。
            env_ids: 环境索引设定目标。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set targets
        self._data.joint_pos_target[env_ids, joint_ids] = target

    def set_joint_velocity_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | slice | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set joint velocity targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint velocity targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        """在内部缓冲器中设置关节速度目标。

        这项功能不适用于仿真的共同目标。
        它只用所需的值填充缓冲器。
        为了应用共同目标，请调用:meth:`write_data_to_sim`函数。

        参数：
            target: 共同的速度目标。
                    形状是 (len(env_ids)，len(joint_ids))。
            joint_ids: 确定目标的共同索引。
                       在 None (所有关节) 上默认设置。
            env_ids: 环境索引设定目标。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set targets
        self._data.joint_vel_target[env_ids, joint_ids] = target

    def set_joint_effort_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | slice | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set joint efforts into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint effort targets. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
            env_ids: The environment indices to set the targets for. Defaults to None (all environments).
        """
        """在内部缓冲器中设置共同努力。

        这项功能不适用于仿真的共同目标。
        它只用所需的值填充缓冲器。
        为了应用共同目标，请调用:meth:`write_data_to_sim`函数。

        参数：
            target: 共同努力目标。
                    形状是 (len(env_ids)，len(joint_ids))。
            joint_ids: 确定目标的共同索引。
                       在 None (所有关节) 上默认设置。
            env_ids: 环境索引设定目标。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        # broadcast env_ids if needed to allow double indexing
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        # set targets
        self._data.joint_effort_target[env_ids, joint_ids] = target

    """
    Operations - Tendons.
    """
    """医生: 部
    """

    def set_fixed_tendon_stiffness(
        self,
        stiffness: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the
        :meth:`write_fixed_tendon_properties_to_sim` method.

        Args:
            stiffness: Fixed tendon stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
        """
        """在内部缓冲器中设置固定门硬度。

        这种功能不适用于仿真的硬度。
        它只用所需的值填充缓冲器。
        要使用门硬性，请使用:meth:`write_fixed_tendon_properties_to_sim`方法。

        参数：
            stiffness: 固定的门硬。
                       形状是 (len(env_ids)，len(fixed_tendon_ids))。
            fixed_tendon_ids: 子指标设定了度。
                              None (所有固定节) 的默认情况。
            env_ids: 环境指标设定了度。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set stiffness
        self._data.fixed_tendon_stiffness[env_ids, fixed_tendon_ids] = stiffness

    def set_fixed_tendon_damping(
        self,
        damping: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            damping: Fixed tendon damping. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the damping for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the damping for. Defaults to None (all environments).
        """
        """设置固定的门入内部缓冲器。

        这项功能不适用于仿真门缩。
        它只用所需的值填充缓冲器。
        要使用门缩，请调用:meth:`write_fixed_tendon_properties_to_sim`函数。

        参数：
            damping: 固定节。
                     形状是 (len(env_ids)，len(fixed_tendon_ids))。
            fixed_tendon_ids: 子指标设置缩。
                              None (所有固定节) 的默认情况。
            env_ids: 环境指标设置缩。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set damping
        self._data.fixed_tendon_damping[env_ids, fixed_tendon_ids] = damping

    def set_fixed_tendon_limit_stiffness(
        self,
        limit_stiffness: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon limit stiffness efforts into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the
        :meth:`write_fixed_tendon_properties_to_sim` method.

        Args:
            limit_stiffness: Fixed tendon limit stiffness. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the limit stiffness for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limit stiffness for. Defaults to None (all environments).
        """
        """在内部缓冲器中设置固定门限制硬度努力。

        这项功能不适用于仿真的门限制硬度。
        它只用所需的值填充缓冲器。
        为了应用门极硬度，请调用:meth:`write_fixed_tendon_properties_to_sim`方法。

        参数：
            limit_stiffness: 固定门的硬度限制。
                             形状是 (len(env_ids)，len(fixed_tendon_ids))。
            fixed_tendon_ids: 子指标设定了限制的硬度。
                              None (所有固定节) 的默认情况。
            env_ids: 环境指标设定了 limit度限制。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set limit_stiffness
        self._data.fixed_tendon_limit_stiffness[env_ids, fixed_tendon_ids] = limit_stiffness

    def set_fixed_tendon_position_limit(
        self,
        limit: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon limit efforts into internal buffers.

        This function does not apply the tendon limit to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit, call the :meth:`write_fixed_tendon_properties_to_sim` function.

         Args:
             limit: Fixed tendon limit. Shape is (len(env_ids), len(fixed_tendon_ids)).
             fixed_tendon_ids: The tendon indices to set the limit for. Defaults to None (all fixed tendons).
             env_ids: The environment indices to set the limit for. Defaults to None (all environments).
        """
        """在内部缓冲器中设置固定门限制力。

        这项功能不适用于仿真的门限制。
        它只用所需的值填充缓冲器。
        要应用门极限，请调用:meth:`write_fixed_tendon_properties_to_sim`函数。

         参数：
             limit: 固定的门限制。
                    形状是 (len(env_ids)，len(fixed_tendon_ids))。
             fixed_tendon_ids: 子指标设定限制。
                               None (所有固定节) 的默认情况。
             env_ids: 环境索引设定限制。
                      在 None (所有环境) 中默认设置。
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set limit
        self._data.fixed_tendon_pos_limits[env_ids, fixed_tendon_ids] = limit

    def set_fixed_tendon_rest_length(
        self,
        rest_length: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon rest length efforts into internal buffers.

        This function does not apply the tendon rest length to the simulation. It only fills the buffers with
        the desired values. To apply the tendon rest length, call the
        :meth:`write_fixed_tendon_properties_to_sim` method.

        Args:
            rest_length: Fixed tendon rest length. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the rest length for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the rest length for. Defaults to None (all environments).
        """
        """在内部缓冲器中设置固定门休息长度。

        这项功能不适用于仿真的休息长度。
        它只用所需的值填充缓冲器。
        要使用部休息长度，请使用:meth:`write_fixed_tendon_properties_to_sim`方法。

        参数：
            rest_length: 固定的休息长度。
                         形状是 (len(env_ids)，len(fixed_tendon_ids))。
            fixed_tendon_ids: 子指标设定休息时间。
                              None (所有固定节) 的默认情况。
            env_ids: 环境指标设置休息时间。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set rest_length
        self._data.fixed_tendon_rest_length[env_ids, fixed_tendon_ids] = rest_length

    def set_fixed_tendon_offset(
        self,
        offset: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon offset efforts into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the :meth:`write_fixed_tendon_properties_to_sim` function.

        Args:
            offset: Fixed tendon offset. Shape is (len(env_ids), len(fixed_tendon_ids)).
            fixed_tendon_ids: The tendon indices to set the offset for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the offset for. Defaults to None (all environments).
        """
        """在内部缓冲器中设置固定部抵消力。

        这项功能不适用于仿真运动的 of位。
        它只用所需的值填充缓冲器。
        调用:meth:`write_fixed_tendon_properties_to_sim`函数来执行部偏移。

        参数：
            offset: 固定的位。
                    形状是 (len(env_ids)，len(fixed_tendon_ids))。
            fixed_tendon_ids: 子指标设定对冲。
                              None (所有固定节) 的默认情况。
            env_ids: 环境索引设置抵消。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)
        if env_ids != slice(None) and fixed_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set offset
        self._data.fixed_tendon_offset[env_ids, fixed_tendon_ids] = offset

    def write_fixed_tendon_properties_to_sim(
        self,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write fixed tendon properties into the simulation.

        Args:
            fixed_tendon_ids: The fixed tendon indices to set the limits for. Defaults to None (all fixed tendons).
            env_ids: The environment indices to set the limits for. Defaults to None (all environments).
        """
        """在仿真中写出固定的属性。

        参数：
            fixed_tendon_ids: 固定部索引设定限制。
                              None (所有固定节) 的默认情况。
            env_ids: 环境索引设定限制。
                     在 None (所有环境) 中默认设置。
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            physx_env_ids = self._ALL_INDICES
        if fixed_tendon_ids is None:
            fixed_tendon_ids = slice(None)

        # set into simulation
        self.root_physx_view.set_fixed_tendon_properties(
            self._data.fixed_tendon_stiffness,
            self._data.fixed_tendon_damping,
            self._data.fixed_tendon_limit_stiffness,
            self._data.fixed_tendon_pos_limits,
            self._data.fixed_tendon_rest_length,
            self._data.fixed_tendon_offset,
            indices=physx_env_ids,
        )

    def set_spatial_tendon_stiffness(
        self,
        stiffness: torch.Tensor,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set spatial tendon stiffness into internal buffers.

        This function does not apply the tendon stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon stiffness, call the
        :meth:`write_spatial_tendon_properties_to_sim` method.

        Args:
            stiffness: Spatial tendon stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the stiffness for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the stiffness for. Defaults to None (all environments).
        """
        """在内部缓冲器中设置空间硬性。

        这种功能不适用于仿真的硬度。
        它只用所需的值填充缓冲器。
        要使用门硬性，请使用:meth:`write_spatial_tendon_properties_to_sim`方法。

        参数：
            stiffness: 空间节硬化。
                       形状是 (len(env_ids)，len(spatial_tendon_ids))。
            spatial_tendon_ids: 子指标设定了度。
                                None (所有空间节) 的默认情况。
            env_ids: 环境指标设定了度。
                     在 None (所有环境) 中默认设置。
        """
        if get_isaac_sim_version().major < 5:
            logger.warning(
                "Spatial tendons are not supported in Isaac Sim < 5.0. Please update to Isaac Sim 5.0 or later."
            )
            return
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if spatial_tendon_ids is None:
            spatial_tendon_ids = slice(None)
        if env_ids != slice(None) and spatial_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set stiffness
        self._data.spatial_tendon_stiffness[env_ids, spatial_tendon_ids] = stiffness

    def set_spatial_tendon_damping(
        self,
        damping: torch.Tensor,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set spatial tendon damping into internal buffers.

        This function does not apply the tendon damping to the simulation. It only fills the buffers with
        the desired values. To apply the tendon damping, call the
        :meth:`write_spatial_tendon_properties_to_sim` method.

        Args:
            damping: Spatial tendon damping. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the damping for. Defaults to None,
                which means all spatial tendons.
            env_ids: The environment indices to set the damping for. Defaults to None, which means all environments.
        """
        """设置空间门入内部缓冲器。

        这项功能不适用于仿真门缩。
        它只用所需的值填充缓冲器。
        要使用门缩，请使用:meth:`write_spatial_tendon_properties_to_sim`方法。

        参数：
            damping: 空间节。
                     形状是 (len(env_ids)，len(spatial_tendon_ids))。
            spatial_tendon_ids: 子指标设置缩。
                                设置为None，这意味着所有的空间。
            env_ids: 环境指标设置缩。
                     默认为 None，这意味着所有环境。
        """
        if get_isaac_sim_version().major < 5:
            logger.warning(
                "Spatial tendons are not supported in Isaac Sim < 5.0. Please update to Isaac Sim 5.0 or later."
            )
            return
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if spatial_tendon_ids is None:
            spatial_tendon_ids = slice(None)
        if env_ids != slice(None) and spatial_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set damping
        self._data.spatial_tendon_damping[env_ids, spatial_tendon_ids] = damping

    def set_spatial_tendon_limit_stiffness(
        self,
        limit_stiffness: torch.Tensor,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set spatial tendon limit stiffness into internal buffers.

        This function does not apply the tendon limit stiffness to the simulation. It only fills the buffers with
        the desired values. To apply the tendon limit stiffness, call the
        :meth:`write_spatial_tendon_properties_to_sim` method.

        Args:
            limit_stiffness: Spatial tendon limit stiffness. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the limit stiffness for. Defaults to None,
                which means all spatial tendons.
            env_ids: The environment indices to set the limit stiffness for. Defaults to None (all environments).
        """
        """设置空间门限制硬度在内部缓冲器。

        这项功能不适用于仿真的门限制硬度。
        它只用所需的值填充缓冲器。
        为了应用门极硬度，请调用:meth:`write_spatial_tendon_properties_to_sim`方法。

        参数：
            limit_stiffness: 空间门限制了硬度。
                             形状是 (len(env_ids)，len(spatial_tendon_ids))。
            spatial_tendon_ids: 子指标设定了限制的硬度。
                                设置为None，这意味着所有的空间。
            env_ids: 环境指标设定了 limit度限制。
                     在 None (所有环境) 中默认设置。
        """
        if get_isaac_sim_version().major < 5:
            logger.warning(
                "Spatial tendons are not supported in Isaac Sim < 5.0. Please update to Isaac Sim 5.0 or later."
            )
            return
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if spatial_tendon_ids is None:
            spatial_tendon_ids = slice(None)
        if env_ids != slice(None) and spatial_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set limit stiffness
        self._data.spatial_tendon_limit_stiffness[env_ids, spatial_tendon_ids] = limit_stiffness

    def set_spatial_tendon_offset(
        self,
        offset: torch.Tensor,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set spatial tendon offset efforts into internal buffers.

        This function does not apply the tendon offset to the simulation. It only fills the buffers with
        the desired values. To apply the tendon offset, call the
        :meth:`write_spatial_tendon_properties_to_sim` method.

        Args:
            offset: Spatial tendon offset. Shape is (len(env_ids), len(spatial_tendon_ids)).
            spatial_tendon_ids: The tendon indices to set the offset for. Defaults to None (all spatial tendons).
            env_ids: The environment indices to set the offset for. Defaults to None (all environments).
        """
        """设置空间，抵消内部缓冲器的努力。

        这项功能不适用于仿真运动的 of位。
        它只用所需的值填充缓冲器。
        为了使用部偏移，请调用:meth:`write_spatial_tendon_properties_to_sim`方法。

        参数：
            offset: 空间部的移动。
                    形状是 (len(env_ids)，len(spatial_tendon_ids))。
            spatial_tendon_ids: 子指标设定对冲。
                                None (所有空间节) 的默认情况。
            env_ids: 环境索引设置抵消。
                     在 None (所有环境) 中默认设置。
        """
        if get_isaac_sim_version().major < 5:
            logger.warning(
                "Spatial tendons are not supported in Isaac Sim < 5.0. Please update to Isaac Sim 5.0 or later."
            )
            return
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if spatial_tendon_ids is None:
            spatial_tendon_ids = slice(None)
        if env_ids != slice(None) and spatial_tendon_ids != slice(None):
            env_ids = env_ids[:, None]
        # set offset
        self._data.spatial_tendon_offset[env_ids, spatial_tendon_ids] = offset

    def write_spatial_tendon_properties_to_sim(
        self,
        spatial_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write spatial tendon properties into the simulation.

        Args:
            spatial_tendon_ids: The spatial tendon indices to set the properties for. Defaults to None,
                which means all spatial tendons.
            env_ids: The environment indices to set the properties for. Defaults to None,
                which means all environments.
        """
        """在仿真中写出空间的属性。

        参数：
            spatial_tendon_ids: 空间子指标设定其特性。
                                设置为None，这意味着所有的空间。
            env_ids: 环境索引设置属性。
                     默认为 None，这意味着所有环境。
        """
        # resolve indices
        physx_env_ids = env_ids
        if env_ids is None:
            physx_env_ids = self._ALL_INDICES
        if spatial_tendon_ids is None:
            spatial_tendon_ids = slice(None)

        # set into simulation
        self.root_physx_view.set_spatial_tendon_properties(
            self._data.spatial_tendon_stiffness,
            self._data.spatial_tendon_damping,
            self._data.spatial_tendon_limit_stiffness,
            self._data.spatial_tendon_offset,
            indices=physx_env_ids,
        )

    """
    Internal helper.
    """
    """内部助理。
    """

    def _initialize_impl(self):
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        if self.cfg.articulation_root_prim_path is not None:
            # The articulation root prim path is specified explicitly, so we can just use this.
            root_prim_path_expr = self.cfg.prim_path + self.cfg.articulation_root_prim_path
        else:
            # No articulation root prim path was specified, so we need to search
            # for it. We search for this in the first environment and then
            # create a regex that matches all environments.
            first_env_matching_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
            if first_env_matching_prim is None:
                raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
            first_env_matching_prim_path = first_env_matching_prim.GetPath().pathString

            # Find all articulation root prims in the first environment.
            first_env_root_prims = sim_utils.get_all_matching_child_prims(
                first_env_matching_prim_path,
                predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                traverse_instance_prims=False,
            )
            if len(first_env_root_prims) == 0:
                raise RuntimeError(
                    f"Failed to find an articulation when resolving '{first_env_matching_prim_path}'."
                    " Please ensure that the prim has 'USD ArticulationRootAPI' applied."
                )
            if len(first_env_root_prims) > 1:
                raise RuntimeError(
                    f"Failed to find a single articulation when resolving '{first_env_matching_prim_path}'."
                    f" Found multiple '{first_env_root_prims}' under '{first_env_matching_prim_path}'."
                    " Please ensure that there is only one articulation in the prim path tree."
                )

            # Now we convert the found articulation root from the first
            # environment back into a regex that matches all environments.
            first_env_root_prim_path = first_env_root_prims[0].GetPath().pathString
            root_prim_path_relative_to_prim_path = first_env_root_prim_path[len(first_env_matching_prim_path) :]
            root_prim_path_expr = self.cfg.prim_path + root_prim_path_relative_to_prim_path

        # -- articulation
        self._root_physx_view = self._physics_sim_view.create_articulation_view(root_prim_path_expr.replace(".*", "*"))

        # check if the articulation was created
        if self._root_physx_view._backend is None:
            raise RuntimeError(f"Failed to create articulation at: {root_prim_path_expr}. Please check PhysX logs.")

        if get_isaac_sim_version().major < 5:
            logger.warning(
                "Spatial tendons are not supported in Isaac Sim < 5.0: patching spatial-tendon getter"
                " and setter to use dummy value"
            )
            self._root_physx_view.max_spatial_tendons = 0
            self._root_physx_view.get_spatial_tendon_stiffnesses = lambda: torch.empty(0, device=self.device)
            self._root_physx_view.get_spatial_tendon_dampings = lambda: torch.empty(0, device=self.device)
            self._root_physx_view.get_spatial_tendon_limit_stiffnesses = lambda: torch.empty(0, device=self.device)
            self._root_physx_view.get_spatial_tendon_offsets = lambda: torch.empty(0, device=self.device)
            self._root_physx_view.set_spatial_tendon_properties = lambda *args, **kwargs: logger.warning(
                "Spatial tendons are not supported in Isaac Sim < 5.0: Calling"
                " set_spatial_tendon_properties has no effect"
            )

        # log information about the articulation
        logger.info(f"Articulation initialized at: {self.cfg.prim_path} with root '{root_prim_path_expr}'.")
        logger.info(f"Is fixed root: {self.is_fixed_base}")
        logger.info(f"Number of bodies: {self.num_bodies}")
        logger.info(f"Body names: {self.body_names}")
        logger.info(f"Number of joints: {self.num_joints}")
        logger.info(f"Joint names: {self.joint_names}")
        logger.info(f"Number of fixed tendons: {self.num_fixed_tendons}")

        # container for data access
        self._data = ArticulationData(self.root_physx_view, self.device)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        self._process_actuators_cfg()
        self._process_tendons()
        # validate configuration
        self._validate_cfg()
        # update the robot data
        self.update(0.0)
        # log joint information
        self._log_articulation_info()

    def _create_buffers(self):
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        self._ALL_BODY_INDICES = torch.arange(self.num_bodies, dtype=torch.long, device=self.device)
        self._ALL_INDICES_WP = wp.from_torch(self._ALL_INDICES.to(torch.int32), dtype=wp.int32)
        self._ALL_BODY_INDICES_WP = wp.from_torch(self._ALL_BODY_INDICES.to(torch.int32), dtype=wp.int32)

        # external wrench composer
        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

        # asset named data
        self._data.joint_names = self.joint_names
        self._data.body_names = self.body_names
        # tendon names are set in _process_tendons function

        # -- joint properties
        self._data.default_joint_pos_limits = self.root_physx_view.get_dof_limits().to(self.device).clone()
        self._data.default_joint_stiffness = self.root_physx_view.get_dof_stiffnesses().to(self.device).clone()
        self._data.default_joint_damping = self.root_physx_view.get_dof_dampings().to(self.device).clone()
        self._data.default_joint_armature = self.root_physx_view.get_dof_armatures().to(self.device).clone()
        if get_isaac_sim_version().major < 5:
            self._data.default_joint_friction_coeff = (
                self.root_physx_view.get_dof_friction_coefficients().to(self.device).clone()
            )
            self._data.default_joint_dynamic_friction_coeff = torch.zeros_like(self._data.default_joint_friction_coeff)
            self._data.default_joint_viscous_friction_coeff = torch.zeros_like(self._data.default_joint_friction_coeff)
        else:
            friction_props = self.root_physx_view.get_dof_friction_properties()
            self._data.default_joint_friction_coeff = friction_props[:, :, 0].to(self.device).clone()
            self._data.default_joint_dynamic_friction_coeff = friction_props[:, :, 1].to(self.device).clone()
            self._data.default_joint_viscous_friction_coeff = friction_props[:, :, 2].to(self.device).clone()

        self._data.joint_pos_limits = self._data.default_joint_pos_limits.clone()
        self._data.joint_vel_limits = self.root_physx_view.get_dof_max_velocities().to(self.device).clone()
        self._data.joint_effort_limits = self.root_physx_view.get_dof_max_forces().to(self.device).clone()
        self._data.joint_stiffness = self._data.default_joint_stiffness.clone()
        self._data.joint_damping = self._data.default_joint_damping.clone()
        self._data.joint_armature = self._data.default_joint_armature.clone()
        self._data.joint_friction_coeff = self._data.default_joint_friction_coeff.clone()
        self._data.joint_dynamic_friction_coeff = self._data.default_joint_dynamic_friction_coeff.clone()
        self._data.joint_viscous_friction_coeff = self._data.default_joint_viscous_friction_coeff.clone()

        # -- body properties
        self._data.default_mass = self.root_physx_view.get_masses().clone()
        self._data.default_inertia = self.root_physx_view.get_inertias().clone()

        # -- joint commands (sent to the actuator from the user)
        self._data.joint_pos_target = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._data.joint_vel_target = torch.zeros_like(self._data.joint_pos_target)
        self._data.joint_effort_target = torch.zeros_like(self._data.joint_pos_target)
        # -- joint commands (sent to the simulation after actuator processing)
        self._joint_pos_target_sim = torch.zeros_like(self._data.joint_pos_target)
        self._joint_vel_target_sim = torch.zeros_like(self._data.joint_pos_target)
        self._joint_effort_target_sim = torch.zeros_like(self._data.joint_pos_target)

        # -- computed joint efforts from the actuator models
        self._data.computed_torque = torch.zeros_like(self._data.joint_pos_target)
        self._data.applied_torque = torch.zeros_like(self._data.joint_pos_target)

        # -- other data that are filled based on explicit actuator models
        self._data.soft_joint_vel_limits = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._data.gear_ratio = torch.ones(self.num_instances, self.num_joints, device=self.device)

        # soft joint position limits (recommended not to be too close to limits).
        joint_pos_mean = (self._data.joint_pos_limits[..., 0] + self._data.joint_pos_limits[..., 1]) / 2
        joint_pos_range = self._data.joint_pos_limits[..., 1] - self._data.joint_pos_limits[..., 0]
        soft_limit_factor = self.cfg.soft_joint_pos_limit_factor
        # add to data
        self._data.soft_joint_pos_limits = torch.zeros(self.num_instances, self.num_joints, 2, device=self.device)
        self._data.soft_joint_pos_limits[..., 0] = joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor
        self._data.soft_joint_pos_limits[..., 1] = joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        """配置参数后处理。"""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)
            + tuple(self.cfg.init_state.lin_vel)
            + tuple(self.cfg.init_state.ang_vel)
        )
        default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)

        # -- joint state
        self._data.default_joint_pos = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._data.default_joint_vel = torch.zeros_like(self._data.default_joint_pos)
        # joint pos
        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.init_state.joint_pos, self.joint_names
        )
        self._data.default_joint_pos[:, indices_list] = torch.tensor(values_list, device=self.device)
        # joint vel
        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.init_state.joint_vel, self.joint_names
        )
        self._data.default_joint_vel[:, indices_list] = torch.tensor(values_list, device=self.device)

    """
    Internal simulation callbacks.
    """
    """内部仿真回调。
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        """破坏场景元素。"""
        # call parent
        super()._invalidate_initialize_callback(event)
        self._root_physx_view = None

    """
    Internal helpers -- Actuators.
    """
    """内部辅助员-- 执行器。
    """

    def _process_actuators_cfg(self):
        """Process and apply articulation joint properties."""
        """处理和应用关节结合特性。"""
        # create actuators
        self.actuators = dict()
        # flag for implicit actuators
        # if this is false, we by-pass certain checks when doing actuator-related operations
        self._has_implicit_actuators = False

        # iterate over all actuator configurations
        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            # type annotation for type checkers
            actuator_cfg: ActuatorBaseCfg
            # create actuator group
            joint_ids, joint_names = self.find_joints(actuator_cfg.joint_names_expr)
            # check if any joints are found
            if len(joint_names) == 0:
                raise ValueError(
                    f"No joints found for actuator group: {actuator_name} with joint name expression:"
                    f" {actuator_cfg.joint_names_expr}."
                )
            # resolve joint indices
            # we pass a slice if all joints are selected to avoid indexing overhead
            if len(joint_names) == self.num_joints:
                joint_ids = slice(None)
            else:
                joint_ids = torch.tensor(joint_ids, device=self.device)
            # create actuator collection
            # note: for efficiency avoid indexing when over all indices
            actuator: ActuatorBase = actuator_cfg.class_type(
                cfg=actuator_cfg,
                joint_names=joint_names,
                joint_ids=joint_ids,
                num_envs=self.num_instances,
                device=self.device,
                stiffness=self._data.default_joint_stiffness[:, joint_ids],
                damping=self._data.default_joint_damping[:, joint_ids],
                armature=self._data.default_joint_armature[:, joint_ids],
                friction=self._data.default_joint_friction_coeff[:, joint_ids],
                dynamic_friction=self._data.default_joint_dynamic_friction_coeff[:, joint_ids],
                viscous_friction=self._data.default_joint_viscous_friction_coeff[:, joint_ids],
                effort_limit=self._data.joint_effort_limits[:, joint_ids].clone(),
                velocity_limit=self._data.joint_vel_limits[:, joint_ids],
            )
            # log information on actuator groups
            model_type = "implicit" if actuator.is_implicit_model else "explicit"
            logger.info(
                f"Actuator collection: {actuator_name} with model '{actuator_cfg.class_type.__name__}'"
                f" (type: {model_type}) and joint names: {joint_names} [{joint_ids}]."
            )
            # store actuator group
            self.actuators[actuator_name] = actuator
            # set the passed gains and limits into the simulation
            if isinstance(actuator, ImplicitActuator):
                self._has_implicit_actuators = True
                # the gains and limits are set into the simulation since actuator model is implicit
                self.write_joint_stiffness_to_sim(actuator.stiffness, joint_ids=actuator.joint_indices)
                self.write_joint_damping_to_sim(actuator.damping, joint_ids=actuator.joint_indices)
            else:
                # the gains and limits are processed by the actuator model
                # we set gains to zero, and torque limit to a high value in simulation to avoid any interference
                self.write_joint_stiffness_to_sim(0.0, joint_ids=actuator.joint_indices)
                self.write_joint_damping_to_sim(0.0, joint_ids=actuator.joint_indices)

            # Set common properties into the simulation
            self.write_joint_effort_limit_to_sim(actuator.effort_limit_sim, joint_ids=actuator.joint_indices)
            self.write_joint_velocity_limit_to_sim(actuator.velocity_limit_sim, joint_ids=actuator.joint_indices)
            self.write_joint_armature_to_sim(actuator.armature, joint_ids=actuator.joint_indices)
            self.write_joint_friction_coefficient_to_sim(actuator.friction, joint_ids=actuator.joint_indices)
            if get_isaac_sim_version().major >= 5:
                self.write_joint_dynamic_friction_coefficient_to_sim(
                    actuator.dynamic_friction, joint_ids=actuator.joint_indices
                )
                self.write_joint_viscous_friction_coefficient_to_sim(
                    actuator.viscous_friction, joint_ids=actuator.joint_indices
                )

            # Store the configured values from the actuator model
            # note: this is the value configured in the actuator model (for implicit and explicit actuators)
            self._data.default_joint_stiffness[:, actuator.joint_indices] = actuator.stiffness
            self._data.default_joint_damping[:, actuator.joint_indices] = actuator.damping
            self._data.default_joint_armature[:, actuator.joint_indices] = actuator.armature
            self._data.default_joint_friction_coeff[:, actuator.joint_indices] = actuator.friction
            if get_isaac_sim_version().major >= 5:
                self._data.default_joint_dynamic_friction_coeff[:, actuator.joint_indices] = actuator.dynamic_friction
                self._data.default_joint_viscous_friction_coeff[:, actuator.joint_indices] = actuator.viscous_friction

        # perform some sanity checks to ensure actuators are prepared correctly
        total_act_joints = sum(actuator.num_joints for actuator in self.actuators.values())
        if total_act_joints != (self.num_joints - self.num_fixed_tendons):
            logger.warning(
                "Not all actuators are configured! Total number of actuated joints not equal to number of"
                f" joints available: {total_act_joints} != {self.num_joints - self.num_fixed_tendons}."
            )

        if self.cfg.actuator_value_resolution_debug_print:
            t = PrettyTable(["Group", "Property", "Name", "ID", "USD Value", "ActutatorCfg Value", "Applied"])
            for actuator_group, actuator in self.actuators.items():
                group_count = 0
                for property, resolution_details in actuator.joint_property_resolution_table.items():
                    for prop_idx, resolution_detail in enumerate(resolution_details):
                        actuator_group_str = actuator_group if group_count == 0 else ""
                        property_str = property if prop_idx == 0 else ""
                        fmt = [f"{v:.2e}" if isinstance(v, float) else str(v) for v in resolution_detail]
                        t.add_row([actuator_group_str, property_str, *fmt])
                        group_count += 1
            logger.warning(f"\nActuatorCfg-USD Value Discrepancy Resolution (matching values are skipped): \n{t}")

    def _process_tendons(self):
        """Process fixed and spatial tendons."""
        """过程固定和空间。"""
        # create a list to store the fixed tendon names
        self._fixed_tendon_names = list()
        self._spatial_tendon_names = list()
        # parse fixed tendons properties if they exist
        if self.num_fixed_tendons > 0 or self.num_spatial_tendons > 0:
            joint_paths = self.root_physx_view.dof_paths[0]

            # iterate over all joints to find tendons attached to them
            for j in range(self.num_joints):
                usd_joint_path = joint_paths[j]
                # check whether joint has tendons - tendon name follows the joint name it is attached to
                joint = UsdPhysics.Joint.Get(self.stage, usd_joint_path)
                if joint.GetPrim().HasAPI(PhysxSchema.PhysxTendonAxisRootAPI):
                    joint_name = usd_joint_path.split("/")[-1]
                    self._fixed_tendon_names.append(joint_name)
                elif joint.GetPrim().HasAPI(PhysxSchema.PhysxTendonAttachmentRootAPI) or joint.GetPrim().HasAPI(
                    PhysxSchema.PhysxTendonAttachmentLeafAPI
                ):
                    joint_name = usd_joint_path.split("/")[-1]
                    self._spatial_tendon_names.append(joint_name)

            # store the fixed tendon names
            self._data.fixed_tendon_names = self._fixed_tendon_names
            self._data.spatial_tendon_names = self._spatial_tendon_names
            # store the current USD fixed tendon properties
            self._data.default_fixed_tendon_stiffness = self.root_physx_view.get_fixed_tendon_stiffnesses().clone()
            self._data.default_fixed_tendon_damping = self.root_physx_view.get_fixed_tendon_dampings().clone()
            self._data.default_fixed_tendon_limit_stiffness = (
                self.root_physx_view.get_fixed_tendon_limit_stiffnesses().clone()
            )
            self._data.default_fixed_tendon_pos_limits = self.root_physx_view.get_fixed_tendon_limits().clone()
            self._data.default_fixed_tendon_rest_length = self.root_physx_view.get_fixed_tendon_rest_lengths().clone()
            self._data.default_fixed_tendon_offset = self.root_physx_view.get_fixed_tendon_offsets().clone()
            self._data.default_spatial_tendon_stiffness = self.root_physx_view.get_spatial_tendon_stiffnesses().clone()
            self._data.default_spatial_tendon_damping = self.root_physx_view.get_spatial_tendon_dampings().clone()
            self._data.default_spatial_tendon_limit_stiffness = (
                self.root_physx_view.get_spatial_tendon_limit_stiffnesses().clone()
            )
            self._data.default_spatial_tendon_offset = self.root_physx_view.get_spatial_tendon_offsets().clone()

            # store a copy of the default values for the fixed tendons
            self._data.fixed_tendon_stiffness = self._data.default_fixed_tendon_stiffness.clone()
            self._data.fixed_tendon_damping = self._data.default_fixed_tendon_damping.clone()
            self._data.fixed_tendon_limit_stiffness = self._data.default_fixed_tendon_limit_stiffness.clone()
            self._data.fixed_tendon_pos_limits = self._data.default_fixed_tendon_pos_limits.clone()
            self._data.fixed_tendon_rest_length = self._data.default_fixed_tendon_rest_length.clone()
            self._data.fixed_tendon_offset = self._data.default_fixed_tendon_offset.clone()
            self._data.spatial_tendon_stiffness = self._data.default_spatial_tendon_stiffness.clone()
            self._data.spatial_tendon_damping = self._data.default_spatial_tendon_damping.clone()
            self._data.spatial_tendon_limit_stiffness = self._data.default_spatial_tendon_limit_stiffness.clone()
            self._data.spatial_tendon_offset = self._data.default_spatial_tendon_offset.clone()

    def _apply_actuator_model(self):
        """Processes joint commands for the articulation by forwarding them to the actuators.

        The actions are first processed using actuator models. Depending on the robot configuration,
        the actuator models compute the joint level simulation commands and sets them into the PhysX buffers.
        """
        """通过将它们转发到执行器来处理关节的联合命令。

        操作首先采用动机模型进行处理。
        根据机器人配置，执行器模型计算了联合级别仿真命令，并将其设置在PhysX缓冲器中。
        """
        # process actions per group
        for actuator in self.actuators.values():
            # prepare input for actuator model based on cached data
            # TODO : A tensor dict would be nice to do the indexing of all tensors together
            control_action = ArticulationActions(   # ① 从用户缓冲区读取策略输出的目标
                joint_positions=self._data.joint_pos_target[:, actuator.joint_indices],
                joint_velocities=self._data.joint_vel_target[:, actuator.joint_indices],
                joint_efforts=self._data.joint_effort_target[:, actuator.joint_indices],
                joint_indices=actuator.joint_indices,
            )
            # compute joint command from the actuator model
            control_action = actuator.compute(      # ② 通过执行器模型计算（例如 PD 控制器）
                control_action,
                joint_pos=self._data.joint_pos[:, actuator.joint_indices],  # 当前实际位置
                joint_vel=self._data.joint_vel[:, actuator.joint_indices],  # 当前实际速度
            )
            # update targets (these are set into the simulation)    # ③ 计算结果写入仿真缓冲区
            if control_action.joint_positions is not None:
                self._joint_pos_target_sim[:, actuator.joint_indices] = control_action.joint_positions
            if control_action.joint_velocities is not None:
                self._joint_vel_target_sim[:, actuator.joint_indices] = control_action.joint_velocities
            if control_action.joint_efforts is not None:
                self._joint_effort_target_sim[:, actuator.joint_indices] = control_action.joint_efforts
            # update state of the actuator model
            # -- torques
            self._data.computed_torque[:, actuator.joint_indices] = actuator.computed_effort
            self._data.applied_torque[:, actuator.joint_indices] = actuator.applied_effort
            # -- actuator data
            self._data.soft_joint_vel_limits[:, actuator.joint_indices] = actuator.velocity_limit
            # TODO: find a cleaner way to handle gear ratio. Only needed for variable gear ratio actuators.
            if hasattr(actuator, "gear_ratio"):
                self._data.gear_ratio[:, actuator.joint_indices] = actuator.gear_ratio

    """
    Internal helpers -- Debugging.
    """
    """内部助理 - - 调试。
    """

    def _validate_cfg(self):
        """Validate the configuration after processing.

        Note:
            This function should be called only after the configuration has been processed and the buffers have been
            created. Otherwise, some settings that are altered during processing may not be validated.
            For instance, the actuator models may change the joint max velocity limits.
        """
        """处理后验证配置。

        说明：
            在配置已处理并创建缓冲器后才应调用此函数。
            否则，在加工过程中改变的某些设置可能无法验证。
            例如，执行器模型可能会改变联合最大速度限制。
        """
        # check that the default values are within the limits
        joint_pos_limits = self.root_physx_view.get_dof_limits()[0].to(self.device)
        out_of_range = self._data.default_joint_pos[0] < joint_pos_limits[:, 0]
        out_of_range |= self._data.default_joint_pos[0] > joint_pos_limits[:, 1]
        violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)
        # throw error if any of the default joint positions are out of the limits
        if len(violated_indices) > 0:
            # prepare message for violated joints
            msg = "The following joints have default positions out of the limits: \n"
            for idx in violated_indices:
                joint_name = self.data.joint_names[idx]
                joint_limit = joint_pos_limits[idx]
                joint_pos = self.data.default_joint_pos[0, idx]
                # add to message
                msg += f"\t- '{joint_name}': {joint_pos:.3f} not in [{joint_limit[0]:.3f}, {joint_limit[1]:.3f}]\n"
            raise ValueError(msg)

        # check that the default joint velocities are within the limits
        joint_max_vel = self.root_physx_view.get_dof_max_velocities()[0].to(self.device)
        out_of_range = torch.abs(self._data.default_joint_vel[0]) > joint_max_vel
        violated_indices = torch.nonzero(out_of_range, as_tuple=False).squeeze(-1)
        if len(violated_indices) > 0:
            # prepare message for violated joints
            msg = "The following joints have default velocities out of the limits: \n"
            for idx in violated_indices:
                joint_name = self.data.joint_names[idx]
                joint_limit = [-joint_max_vel[idx], joint_max_vel[idx]]
                joint_vel = self.data.default_joint_vel[0, idx]
                # add to message
                msg += f"\t- '{joint_name}': {joint_vel:.3f} not in [{joint_limit[0]:.3f}, {joint_limit[1]:.3f}]\n"
            raise ValueError(msg)

    def _log_articulation_info(self):
        """Log information about the articulation.

        Note: We purposefully read the values from the simulator to ensure that the values are configured as expected.
        """
        """记录有关关节的信息。

        Note: 我们故意读取仿真器中的值，以确保值按预期配置。
        """

        # define custom formatters for large numbers and limit ranges
        def format_large_number(_, v: float) -> str:
            """Format large numbers using scientific notation."""
            """使用科学符号来格式化大数字。"""
            if abs(v) >= 1e3:
                return f"{v:.1e}"
            else:
                return f"{v:.3f}"

        def format_limits(_, v: tuple[float, float]) -> str:
            """Format limit ranges using scientific notation."""
            """使用科学标记来格式化限制范围。"""
            if abs(v[0]) >= 1e3 or abs(v[1]) >= 1e3:
                return f"[{v[0]:.1e}, {v[1]:.1e}]"
            else:
                return f"[{v[0]:.3f}, {v[1]:.3f}]"

        # read out all joint parameters from simulation
        # -- gains
        stiffnesses = self.root_physx_view.get_dof_stiffnesses()[0].tolist()
        dampings = self.root_physx_view.get_dof_dampings()[0].tolist()
        # -- properties
        armatures = self.root_physx_view.get_dof_armatures()[0].tolist()
        if get_isaac_sim_version().major < 5:
            static_frictions = self.root_physx_view.get_dof_friction_coefficients()[0].tolist()
        else:
            friction_props = self.root_physx_view.get_dof_friction_properties()
            static_frictions = friction_props[:, :, 0][0].tolist()
            dynamic_frictions = friction_props[:, :, 1][0].tolist()
            viscous_frictions = friction_props[:, :, 2][0].tolist()
        # -- limits
        position_limits = self.root_physx_view.get_dof_limits()[0].tolist()
        velocity_limits = self.root_physx_view.get_dof_max_velocities()[0].tolist()
        effort_limits = self.root_physx_view.get_dof_max_forces()[0].tolist()
        # create table for term information
        joint_table = PrettyTable()
        joint_table.title = f"Simulation Joint Information (Prim path: {self.cfg.prim_path})"
        # build field names based on Isaac Sim version
        field_names = ["Index", "Name", "Stiffness", "Damping", "Armature"]
        if get_isaac_sim_version().major < 5:
            field_names.append("Static Friction")
        else:
            field_names.extend(["Static Friction", "Dynamic Friction", "Viscous Friction"])
        field_names.extend(["Position Limits", "Velocity Limits", "Effort Limits"])
        joint_table.field_names = field_names

        # apply custom formatters to numeric columns
        joint_table.custom_format["Stiffness"] = format_large_number
        joint_table.custom_format["Damping"] = format_large_number
        joint_table.custom_format["Armature"] = format_large_number
        joint_table.custom_format["Static Friction"] = format_large_number
        if get_isaac_sim_version().major >= 5:
            joint_table.custom_format["Dynamic Friction"] = format_large_number
            joint_table.custom_format["Viscous Friction"] = format_large_number
        joint_table.custom_format["Position Limits"] = format_limits
        joint_table.custom_format["Velocity Limits"] = format_large_number
        joint_table.custom_format["Effort Limits"] = format_large_number

        # set alignment of table columns
        joint_table.align["Name"] = "l"
        # add info on each term
        for index, name in enumerate(self.joint_names):
            # build row data based on Isaac Sim version
            row_data = [index, name, stiffnesses[index], dampings[index], armatures[index]]
            if get_isaac_sim_version().major < 5:
                row_data.append(static_frictions[index])
            else:
                row_data.extend([static_frictions[index], dynamic_frictions[index], viscous_frictions[index]])
            row_data.extend([position_limits[index], velocity_limits[index], effort_limits[index]])
            # add row to table
            joint_table.add_row(row_data)
        # convert table to string
        logger.info(f"Simulation parameters for joints in {self.cfg.prim_path}:\n" + joint_table.get_string())

        # read out all fixed tendon parameters from simulation
        if self.num_fixed_tendons > 0:
            # -- gains
            ft_stiffnesses = self.root_physx_view.get_fixed_tendon_stiffnesses()[0].tolist()
            ft_dampings = self.root_physx_view.get_fixed_tendon_dampings()[0].tolist()
            # -- limits
            ft_limit_stiffnesses = self.root_physx_view.get_fixed_tendon_limit_stiffnesses()[0].tolist()
            ft_limits = self.root_physx_view.get_fixed_tendon_limits()[0].tolist()
            ft_rest_lengths = self.root_physx_view.get_fixed_tendon_rest_lengths()[0].tolist()
            ft_offsets = self.root_physx_view.get_fixed_tendon_offsets()[0].tolist()
            # create table for term information
            tendon_table = PrettyTable()
            tendon_table.title = f"Simulation Fixed Tendon Information (Prim path: {self.cfg.prim_path})"
            tendon_table.field_names = [
                "Index",
                "Stiffness",
                "Damping",
                "Limit Stiffness",
                "Limits",
                "Rest Length",
                "Offset",
            ]
            tendon_table.float_format = ".3"

            # apply custom formatters to tendon table columns
            tendon_table.custom_format["Stiffness"] = format_large_number
            tendon_table.custom_format["Damping"] = format_large_number
            tendon_table.custom_format["Limit Stiffness"] = format_large_number
            tendon_table.custom_format["Limits"] = format_limits
            tendon_table.custom_format["Rest Length"] = format_large_number
            tendon_table.custom_format["Offset"] = format_large_number

            # add info on each term
            for index in range(self.num_fixed_tendons):
                tendon_table.add_row(
                    [
                        index,
                        ft_stiffnesses[index],
                        ft_dampings[index],
                        ft_limit_stiffnesses[index],
                        ft_limits[index],
                        ft_rest_lengths[index],
                        ft_offsets[index],
                    ]
                )
            # convert table to string
            logger.info(
                f"Simulation parameters for fixed tendons in {self.cfg.prim_path}:\n" + tendon_table.get_string()
            )

        if self.num_spatial_tendons > 0:
            # -- gains
            st_stiffnesses = self.root_physx_view.get_spatial_tendon_stiffnesses()[0].tolist()
            st_dampings = self.root_physx_view.get_spatial_tendon_dampings()[0].tolist()
            # -- limits
            st_limit_stiffnesses = self.root_physx_view.get_spatial_tendon_limit_stiffnesses()[0].tolist()
            st_offsets = self.root_physx_view.get_spatial_tendon_offsets()[0].tolist()
            # create table for term information
            tendon_table = PrettyTable()
            tendon_table.title = f"Simulation Spatial Tendon Information (Prim path: {self.cfg.prim_path})"
            tendon_table.field_names = [
                "Index",
                "Stiffness",
                "Damping",
                "Limit Stiffness",
                "Offset",
            ]
            tendon_table.float_format = ".3"
            # add info on each term
            for index in range(self.num_spatial_tendons):
                tendon_table.add_row(
                    [
                        index,
                        st_stiffnesses[index],
                        st_dampings[index],
                        st_limit_stiffnesses[index],
                        st_offsets[index],
                    ]
                )
            # convert table to string
            logger.info(
                f"Simulation parameters for spatial tendons in {self.cfg.prim_path}:\n" + tendon_table.get_string()
            )

    """
    Deprecated methods.
    """
    """废弃的方法。
    """

    def write_joint_friction_to_sim(
        self,
        joint_friction: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Write joint friction coefficients into the simulation.

        .. deprecated:: 2.1.0
            Please use :meth:`write_joint_friction_coefficient_to_sim` instead.
        """
        """在仿真中写出关节摩擦系数。

        ..
        过时:: 2.1.0 请使用:meth:`write_joint_friction_coefficient_to_sim`。
        """
        logger.warning(
            "The function 'write_joint_friction_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_friction_coefficient_to_sim' instead."
        )
        self.write_joint_friction_coefficient_to_sim(joint_friction, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_limits_to_sim(
        self,
        limits: torch.Tensor | float,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        warn_limit_violation: bool = True,
    ):
        """Write joint limits into the simulation.

        .. deprecated:: 2.1.0
            Please use :meth:`write_joint_position_limit_to_sim` instead.
        """
        """在仿真中写出关节限制。

        ..
        过时:: 2.1.0 请使用:meth:`write_joint_position_limit_to_sim`。
        """
        logger.warning(
            "The function 'write_joint_limits_to_sim' will be deprecated in a future release. Please"
            " use 'write_joint_position_limit_to_sim' instead."
        )
        self.write_joint_position_limit_to_sim(
            limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=warn_limit_violation
        )

    def set_fixed_tendon_limit(
        self,
        limit: torch.Tensor,
        fixed_tendon_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set fixed tendon position limits into internal buffers.

        .. deprecated:: 2.1.0
            Please use :meth:`set_fixed_tendon_position_limit` instead.
        """
        """在内部缓冲器中设定固定位限制。

        ..
        过时:: 2.1.0 请使用:meth:`set_fixed_tendon_position_limit`。
        """
        logger.warning(
            "The function 'set_fixed_tendon_limit' will be deprecated in a future release. Please"
            " use 'set_fixed_tendon_position_limit' instead."
        )
        self.set_fixed_tendon_position_limit(limit, fixed_tendon_ids=fixed_tendon_ids, env_ids=env_ids)
