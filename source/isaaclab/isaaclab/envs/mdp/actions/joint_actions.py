# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

# import logger
logger = logging.getLogger(__name__)


class JointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """
    """联合动作的基层。

    这种操作项采用类似变化 (规模和抵消) 来预处理原始动作。
    这些变化可以配置用于关节的子组。

    在数学上，动作项定义为:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :数学:`\text{action}`是发送到关节的动力关节的动作，`\text{offset}`
    是对输入操作应用的偏移， :math:`\text{scaling}` 是对输入操作应用的规模， 和 :math:`\text{input action}` 是用户的输入操作。

    基于上述情况，这种动作转换确保输入和输出动作在相同的单位和尺寸中。
    然后，该动作项的子类可以将输出动作映射到关节的特定要求命令 (e.g.位置，速度等)。
    """

    cfg: actions_cfg.JointActionCfg     # 配置对象引用
    """The configuration of the action term."""
    """动作项的配置。"""
    _asset: Articulation                # 机器人关节体（场景中的 "robot"）
    """The articulation asset on which the action term is applied."""
    """动作项适用于的关节资产。"""
    '''
    场景中机器人的句柄
        来自 ActionTerm.__init__ 中通过 cfg.asset_name 从 scene[asset_name] 取到的实体对象。所有 PhysX 数据读写都通过它：
            self._asset.data.joint_pos       # 读取当前关节角度
            self._asset.set_joint_position_target(...)  # 写入目标位置
    '''

    _scale: torch.Tensor | float        # 缩放系数
    """The scaling factor applied to the input action."""
    """对输入操作所应用的扩展因素。"""
    _offset: torch.Tensor | float       # 偏移量
    """The offset applied to the input action."""
    """对输入操作所应用的抵消。"""
    _clip: torch.Tensor                 # 裁剪范围
    """The clip applied to the input action."""
    """在输入操作中应用的裁剪。"""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        '''
        正则匹配到索引 — 从字符串到 GPU 下标
                输入：joint_names=["L_.*hip.*", "L_.*knee.*"]（正则表达式列表）
                输出：_joint_ids=[3, 5, 7, 9, 11, 13]（关节体模型中的实际索引）
            _asset.find_joints() 是 Articulation 类的方法——它遍历机器人的所有关节名，用正则匹配出符合条件的关节，返回它们在 PhysX 数据缓冲区中的索引。
        '''

        # log the resolved joint names for debugging
        logger.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        # 输出: "Resolved joint names: ['L_hip_yaw', 'L_hip_roll', ...] [3, 5, 7, ...]"


        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)
            '''
            slice(None) 优化 — 全关节时的零成本索引
            问题：
                如果动作控制的是机器人的所有关节（如 joint_names=[".*"]），每一次 set_joint_position_target(actions, joint_ids=[0,1,2,...,11]) 都会触发一次索引查找。
            优化：
                slice(None) 是 Python 的"取全部"——等价于 [:]。PhysX API 看到 slice(None) 就知道"所有关节"，跳过索引表查找，直接操作整个缓冲区。
                    # 优化前：
                    self._asset.set_joint_position_target(actions, joint_ids=[0, 1, 2, ..., 11])
                    # 每物理步都遍历 12 个索引

                    # 优化后：
                    self._asset.set_joint_position_target(actions, joint_ids=slice(None))
                    # PhysX 直接写全部，不走索引遍历
            not self.cfg.preserve_order 是前提——只有不要求保持顺序时才能用 slice(None)（因为 slice(None) 的排列由 PhysX 内部决定，不保证和用户配置的顺序一致）。
            '''

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        '''
        张量	                形状	            内容	                                谁写的
        _raw_actions	        [N, action_dim]	    策略网络的原始输出	                    process_actions 第一步
        _processed_actions	    [N, action_dim]	    raw × scale + offset（+ clip）	      process_actions 后续
        '''

        '''
        scale / offset / clip 的三路解析
            三个字段共享完全相同的解析模式——标量 vs 字典：
        '''
        # parse scale
        if isinstance(cfg.scale, (float, int)): # 分支 A: 统一值
            self._scale = float(cfg.scale)      # 标量，广播时自动适用所有关节
        elif isinstance(cfg.scale, dict):       # 分支 B: 按正则分组
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)    # 先全部初始化为 1.0
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)       # 匹配到的关节覆盖
            '''
            分支 B：字典 — 按正则分组
                scale = {
                    ".*hip.*": 0.5,    # 髋关节：动作减半
                    ".*knee.*": 1.0,   # 膝关节：正常
                }
                resolve_matching_names_values 遍历字典的每个 key，用正则匹配 _joint_names 中的关节名，返回 index_list（匹配到的关节索引）和 value_list（对应的缩放值）。
                然后用索引赋值覆盖张量中的对应位置。
            '''
        else:   # 分支 C: 非法类型
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """
    """属性。
    """

    '''
    当前动作控制的关节数量
    '''
    @property
    def action_dim(self) -> int:
        return self._num_joints

    '''
    策略网络的原始输出
        ——还没经过 scale 和 offset 变换的数据。
        这对于调试和日志记录非常关键：你可以对比原始输出和最终发给物理引擎的指令，判断是策略乱输出还是缩放系数配错了。
    '''
    @property
    def raw_actions(self) -> torch.Tensor:  # 形状为 [N, action_dim] 的浮点张量，其中 N 是并行环境数量
        return self._raw_actions

    '''
    返回经过变换的、可以直接发给物理引擎的最终指令
    '''
    @property
    def processed_actions(self) -> torch.Tensor:    # 形状为 [N, action_dim] 的浮点张量，已经过仿射变换 raw × scale + offset
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the joint action.
        It adds the following information to the base descriptor:
        - joint_names: The names of the joints.
        - scale: The scale of the action term.
        - offset: The offset of the action term.
        - clip: The clip of the action term.

        Returns:
            The IO descriptor of the action term.
        """
        """动作项的IO描述符。

        该描述符用于描述联合动作的动作项。
        它将以下信息添加到基础描述符中:
        - joint_names关节的名称。
        - 规模:动作项的规模。
        - 抵消:动作项的抵消。
        - 动作项的裁剪。

        返回：
            动作项的IO描述符。
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "JointAction"
        self._IO_descriptor.joint_names = self._joint_names
        self._IO_descriptor.scale = self._scale
        # This seems to be always [4xNum_joints] IDK why. Need to check.
        if isinstance(self._offset, torch.Tensor):
            self._IO_descriptor.offset = self._offset[0].detach().cpu().numpy().tolist()
        else:
            self._IO_descriptor.offset = self._offset
        # FIXME: This is not correct. Add list support.
        if self.cfg.clip is not None:
            if isinstance(self._clip, torch.Tensor):
                self._IO_descriptor.clip = self._clip[0].detach().cpu().numpy().tolist()
            else:
                self._IO_descriptor.clip = self._clip
        else:
            self._IO_descriptor.clip = None
        return self._IO_descriptor
    '''
    # joint_actions.py:240-259 执行后的 _IO_descriptor.__dict__
        {
            # ===== 父类填充 =====
            "mdp_type": "Action",
            "name": "joint_position_action",
            "full_path": "isaaclab.envs.mdp.actions.joint_actions.JointPositionAction",
            "description": "Joint action term that applies the processed actions to the articulation's joints as position commands.",
            "export": True,

            # ===== 子类填充 =====
            "shape": (12,),                            # 12 维动作空间
            "dtype": "torch.float32",                  # 策略输出的数据类型
            "action_type": "JointAction",              # 动作类别标签
            "joint_names": [                           # 每个维度对应的关节名
                "L_hip_yaw", "L_hip_roll", "L_hip_pitch",
                "L_knee", "L_ankle",
                "R_hip_yaw", "R_hip_roll", "R_hip_pitch",
                "R_knee", "R_ankle",
                "L_wheel", "R_wheel"
            ],
            "scale": 1.0,                              # 标量：所有关节统一缩放

            # offset 是 Tensor → 转成 Python list
            "offset": [                                # 12 个关节的默认位置（弧度）
                0.0,  0.0, -0.3,    # 左髋 yaw/roll/pitch
                0.6, -0.3,           # 左膝/踝
                0.0,  0.0, -0.3,    # 右髋
                0.6, -0.3,           # 右膝/踝
                0.0,  0.0            # 左右轮
            ],

            # clip 是 Tensor[N,12,2] → 转成 Python list of [min, max]
            "clip": [
                [-0.5, 0.5], [-0.5, 0.5], [-1.0, 0.5],  # 髋关节限幅
                [-2.0, 2.0], [-2.0, 2.0],                # 膝关节
                [-0.5, 0.5], [-0.5, 0.5], [-1.0, 0.5],  # 右髋
                [-2.0, 2.0], [-2.0, 2.0],                # 右膝
                [-inf, inf], [-inf, inf]                   # 轮子不限幅
            ],

            "extras": {},
        }

    '''

    """
    Operations.
    """
    """操作。
    """

    '''
    动作缩放公式 : processed_action = raw_action × scale + offset
    '''
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
    '''
    函数作用
        process_actions 是策略输出到物理指令的翻译管道。它完成三件事：
            存档：把策略的原始输出存入 _raw_actions（方便后续调试和日志记录）
            仿射变换：raw × scale + offset，把策略的归一化输出映射到关节的实际物理范围
            裁剪：把超出安全范围的指令截断，防止关节发出危险动作
    '''

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

'''
策略输出关节目标位置
'''
class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""
    """联合动作项，将处理的动作应用于关节的关节作为位置命令。"""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""
    """动作项的配置。"""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
    '''
    它只把 processed_actions 写进 articulation 的 joint_pos_target 缓冲区。
    真正写入仿真发生在 [articulation.py (line 329)]的 write_data_to_sim()：先调用 _apply_actuator_model()，再写 PhysX。
        关键由 ArticulationCfg.actuators 里的 actuator 类型决定：
            ImplicitActuatorCfg
                position target 会被写成 PhysX 的 DOF position target。
                stiffness / damping 会写进 PhysX joint drive。
                PhysX 内部根据目标位置、当前关节位置、速度、刚度、阻尼、effort limit 等求解实际力矩。
                IsaacLab 里的 computed_torque/applied_torque 只是近似记录，因为 PhysX 不直接暴露真实隐式 drive 力矩。

            IdealPDActuatorCfg / DCMotorCfg 等显式 actuator
                IsaacLab 自己算：
                    tau = kp * (q_des - q) + kd * (qd_des - qd) + tau_ff
                然后按 actuator 的 effort limit 裁剪。
                最后把裁剪后的 effort/torque 通过 set_dof_actuation_forces() 写给 PhysX。
                这种情况下可以说“最后写入仿真的就是显式力矩/effort”。

        所以结论是：
            JointPositionAction 决定策略输出被解释为“目标位置”；
            ArticulationCfg.actuators[*].class_type 决定这个目标位置是交给 PhysX 隐式 PD drive，还是在 IsaacLab 里显式算成 torque。
            具体力矩大小还由 stiffness、damping、effort_limit/effort_limit_sim、velocity_limit_sim、当前关节状态、时间步和 PhysX solver 共同决定。
    '''


'''
策略输出相对当前关节位置的增量
    把策略网络的输出解释为相对于当前关节角度的增量（delta），而不是绝对角度。
    策略输出 0 表示"保持当前姿态不动"，输出正数表示"从当前位置往前转一点"。
'''
class RelativeJointPositionAction(JointAction):
    r"""Joint action term that applies the processed actions to the articulation's joints as relative position commands.

    Unlike :class:`JointPositionAction`, this action term applies the processed actions as relative position commands.
    This means that the processed actions are added to the current joint positions of the articulation's joints
    before being sent as position commands.

    This means that the action applied at every step is:

    .. math::

         \text{applied action} = \text{current joint positions} + \text{processed actions}

    where :math:`\text{current joint positions}` are the current joint positions of the articulation's joints.
    """
    """联合动作项，将处理的动作应用于关节的关节，作为相对位置命令。

    Unlike :类:`JointPositionAction`，本操作项将处理的操作作为相对位置命令。
    这意味着处理的操作在作为位置命令发送之前添加到关节关节的当前关节位置。

    这意味着每一步都需要采取以下措施:

    .. math::

         \text{applied action} = \text{current joint positions} + \text{processed actions}

    where :数学:`\text{current joint positions}`是关节关节的当前关节位置。
    """

    cfg: actions_cfg.RelativeJointPositionActionCfg
    """The configuration of the action term."""
    """动作项的配置。"""

    def __init__(self, cfg: actions_cfg.RelativeJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use zero offset for relative position
        if cfg.use_zero_offset:
            self._offset = 0.0

    def apply_actions(self):
        # add current joint positions to the processed actions
        current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        # set position targets
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)


'''
策略输出关节目标速度
    把策略网络的输出解释为关节的角速度目标（rad/s）。
    与位置控制不同，速度控制下策略不是在说"去哪里"，而是在说"转多快"——策略输出 0 表示"停住不动"，正数表示"正方向旋转"。
'''
class JointVelocityAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""
    """联合动作项，将处理的动作应用于关节的关节作为速度命令。"""

    cfg: actions_cfg.JointVelocityActionCfg
    """The configuration of the action term."""
    """动作项的配置。"""

    def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint velocity as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()

    def apply_actions(self):
        # set joint velocity targets
        self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)


'''
策略输出关节力矩
    策略网络的输出直接解释为关节力矩（N·m 或任意单位的扭矩），不做任何 PD 转换，直接写入 PhysX。
    它是唯一一种不依赖执行器模型做中间转换的动作类型。
'''
class JointEffortAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as effort commands."""
    """联合动作项，将处理的动作应用于关节作为努力命令。"""

    cfg: actions_cfg.JointEffortActionCfg
    """The configuration of the action term."""
    """动作项的配置。"""

    def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # set joint effort targets
        self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)
    '''
    四种动作的"抽象层级"
            越来越接近物理底层 ↓

            JointPositionAction        ← 策略说"去 0.5 弧度"      → PD 控制器算力矩
            RelativeJointPositionAction ← 策略说"再往前转 0.1"      → PD 控制器算力矩
            JointVelocityAction         ← 策略说"以 2 rad/s 旋转"   → PD 控制器算力矩
            JointEffortAction           ← 策略说"施加 50 N·m 力矩"  → 直接写力矩！
        前三种都需要执行器模型（隐式或显式 PD）把位置/速度目标翻译成力矩。JointEffortAction 跳过了这个翻译过程——策略直接对力矩负责。

    为什么力矩控制不需要特殊 offset？
        因为力矩的"零点"天然是 0——0 力矩意味着"不对关节施加任何力"，这和位置控制的复杂性完全不同：
            位置控制：策略输出 0 → 需要 offset 把关节引到合理姿态
            力矩控制：策略输出 0 → 关节受 0 力矩 → 自然保持当前状态（或受重力下落）
        不需要把"0 力矩"映射成任何特殊值，所以父类的 offset 解析（用户配置的 scale、offset、clip）直接用就好。
        process_actions 仍然生效：processed = raw × scale + offset，你可以用 scale 缩放力矩、用 offset 加恒定偏置力矩（比如抵消重力）。
    '''

    '''
    全局数据流（四种动作类型的共同路径）
        ┌──────────────────────────────────────────────────────────────┐
        │  ① 初始化阶段（__init__）                                      │
        │                                                              │
        │  用户配置 YAML                                                │
        │      │   joint_names / scale / offset / clip                  │
        │      ▼                                                       │
        │  JointAction.__init__()                                       │
        │      │   正则匹配关节名 → _joint_ids, _joint_names            │
        │      │   解析 scale / offset / clip                           │
        │      │   创建 _raw_actions[N,dim], _processed_actions[N,dim]  │
        │      ▼                                                       │
        │  子类 __init__() 覆盖 _offset:                                │
        │      JointPositionAction:    _offset = default_joint_pos      │
        │      RelativeJointPosition:  _offset = 0.0                   │
        │      JointVelocityAction:    _offset = default_joint_vel      │
        │      JointEffortAction:      不变（使用用户配置的 offset）      │
        └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  ② 每步执行阶段                                                │
        │                                                              │
        │  策略网络输出 actions [N, dim]                                │
        │      │                                                       │
        │      ▼                                                       │
        │  process_actions(actions)          ← JointAction 基类方法     │
        │      │                                                       │
        │      ├── _raw_actions[:] = actions   ← 存档原始输出           │
        │      ├── _processed_actions = raw × scale + offset  ← 变换   │
        │      └── torch.clamp(_processed_actions, clip)     ← 裁剪    │
        │      │                                                       │
        │      ▼                                                       │
        │  apply_actions()                    ← 子类覆写（核心差异）     │
        │      │                                                       │
        │      │  4 种不同的最终写入 ↓                                   │
        │      │                                                       │
        └──────┼───────────────────────────────────────────────────────┘
            │
            ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  ③ PhysX 写入阶段（write_data_to_sim）                        │
        │                                                              │
        │  set_joint_position_target / set_joint_velocity_target        │
        │  set_joint_effort_target                                     │
        │      │  写入 Articulation._data.joint_xxx_target              │
        │      ▼                                                       │
        │  _apply_actuator_model()  ← 执行器模型处理                    │
        │      │                                                       │
        │      ├── 隐式执行器: 位置/速度目标 → PhysX 内部 PD             │
        │      │              τ = PhysX内部计算（不可见）                 │
        │      │                                                       │
        │      └── 显式执行器: Python层 PD:                             │
        │                   τ = Kp*(q_des-q) + Kd*(q̇_des-q̇) + τ_ff    │
        │                   写入 _data.applied_torque                   │
        │                   清空位置/速度目标                             │
        │      │                                                       │
        │      ▼                                                       │
        │  set_dof_actuation_forces(τ)     ← 力矩写入 PhysX（始终执行）  │
        │  set_dof_position_targets(...)   ← 隐式执行器时写入            │
        │  set_dof_velocity_targets(...)   ← 隐式执行器时写入            │
        │      │                                                       │
        │      ▼                                                       │
        │  PhysX 物理步进                                                │
        │      │                                                       │
        │      ▼                                                       │
        │  新状态: joint_pos, joint_vel, ...  → 回到 ②                  │
        └──────────────────────────────────────────────────────────────┘

    四种动作的完整数据流（纵向对比）
        策略输出 actions [N, dim]
                │
                ▼
        ① process_actions (共同):
            processed = raw × scale + offset
                │
                ├── Position:          processed = raw×scale + default_joint_pos
                ├── RelativePosition:  processed = raw×scale + 0.0
                ├── Velocity:          processed = raw×scale + 0.0 (default_joint_vel)
                └── Effort:            processed = raw×scale + cfg.offset
                │
                ▼
        ② apply_actions (差异):
                │
        ┌──────┼──────────────────────────────────────────────┐
        │      │                                              │
        │ Position / RelativePosition:                        │
        │   写入 _data.joint_pos_target                       │
        │   → 隐式执行器: PhysX 内部 PD   → set_dof_actuation_forces │
        │   → 显式执行器: Python PD 算 τ → set_dof_actuation_forces │
        │                                                    │
        │ Velocity:                                          │
        │   写入 _data.joint_vel_target                       │
        │   → 隐式执行器: PhysX 内部 PD   → set_dof_actuation_forces │
        │   → 显式执行器: Python PD 算 τ → set_dof_actuation_forces │
        │                                                    │
        │ Effort:                                            │
        │   写入 _data.joint_effort_target                    │
        │   → 直接作为 τ（或作为前馈 τ_ff 叠加）              │
        │   → set_dof_actuation_forces                       │
        └────────────────────────────────────────────────────┘
                │
                ▼
        ③ PhysX 最终形式: 关节力矩 τ
                │
                ▼
        ④ 物理积分 → 新状态 (q, q̇, ...)

    四种动作的核心联系
        尽管四种动作看起来完全不同，但它们有三个共同的纽带：
        纽带 1：都经过相同的 process_actions 管道。
            不管你最终是位置、速度还是力矩控制，策略输出都会先经过 raw × scale + offset 的仿射变换。
            这保证了你可以在配置文件中统一调整动作的缩放和偏置。
        纽带 2：最终都以力矩形式作用于关节。
            位置控制和速度控制只是"中间指令"——它们必须通过执行器模型的 PD 控制器转化为力矩。
            力矩控制跳过这个中间步骤，但最终的物理效果完全一样：牛顿第二定律 τ = I × α，力矩产生角加速度。
        纽带 3：共享相同的关节索引和 safety clip 机制。
            四种动作类型使用同一套 _joint_ids 来确定控制哪些关节，同一套 _clip 来做安全限幅。
            你可以在同一台机器人上混合使用多种动作类型（比如腿用位置控制、轮子用速度控制），它们和平共处、互不干扰。
    '''
