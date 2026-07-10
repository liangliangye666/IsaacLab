# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import weakref

import torch

import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager

import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import TimestampedBuffer

# import logger
logger = logging.getLogger(__name__)


class ArticulationData:
    """Data container for an articulation.

    This class contains the data for an articulation in the simulation. The data includes the state of
    the root rigid body, the state of all the bodies in the articulation, and the joint state. The data is
    stored in the simulation world frame unless otherwise specified.

    An articulation is comprised of multiple rigid bodies or links. For a rigid body, there are two frames
    of reference that are used:

    - Actor frame: The frame of reference of the rigid body prim. This typically corresponds to the Xform prim
      with the rigid body schema.
    - Center of mass frame: The frame of reference of the center of mass of the rigid body.

    Depending on the settings, the two frames may not coincide with each other. In the robotics sense, the actor frame
    can be interpreted as the link frame.
    """
    """关节的数据容器。

    这类包含在仿真中的关节数据。
    数据包括根固体的状态，关节中的所有身体的状态和关节状态。
    除非另有说明，则数据存储在仿真世界框架中。

    一个关节由多个硬体或链接组成。
    对于硬体，使用的两个参考框架:

    - 演员框架:硬体prim的参考框架.这通常与X形式prim相符
      with the rigid body schema.
    - 质量框架的中心:是硬体质量中心的参考框架。

    根据设置，两个框架可能不匹配。
    在机器人意义上，演员框架可以被解释为链接框架。
    """

    '''
    输入参数
    root_physx_view：
        PhysX 的 ArticulationView 对象，这是 Isaac Sim 提供的 C++/Python 接口，用于批量读写关节体的物理状态（位置、速度、施加力矩等）。
        它是通往 PhysX 引擎的唯一通道。
        root_physx_view 是 PhysX 的 ArticulationView，可以理解为一组机器人的底层物理视图。
            在强化学习里通常不是只仿真一个机器人，而是并行仿真很多个机器人，例如：
                env_0 里一个 L5A
                env_1 里一个 L5A
                env_2 里一个 L5A
                ...
                env_4095 里一个 L5A

            这些同类型机器人会被一个 ArticulationView 批量管理。所以后面很多数据的形状都是：
                (num_envs, xxx)
    device：
        计算设备，如 "cuda:0" 或 "cpu"。所有内部 tensor 都创建在这个设备上。
    '''
    def __init__(self, root_physx_view: physx.ArticulationView, device: str):
        """Initializes the articulation data.

        Args:
            root_physx_view: The root articulation view.
            device: The device used for processing.
        """
        """启动关节数据。

        参数：
            root_physx_view: 根关节的观点。
            device: 用于加工的装置。
        """
        # Set the parameters
        self.device = device
        # Set the root articulation view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_physx_view: physx.ArticulationView = weakref.proxy(root_physx_view)  # 用弱引用防止循环引用和内存泄漏
        '''
        root_physx_view 是 PhysX Tensor API 的 ArticulationView，可以批量读取机器人状态，比如根位姿、连杆速度、关节位置、关节速度等。
        它通常覆盖所有并行环境里的同一种机器人实例。
        弱引用原因
            因为在 Isaac Lab 里经常有这种关系：
                Articulation 机器人对象
                    └── 持有 ArticulationData 数据对象
                ArticulationData 数据对象
                    └── 又需要访问 PhysX view / robot 相关对象
            如果大家互相强引用，就可能形成循环引用：
                A 引用 B
                B 又引用 A
            这样 Python 垃圾回收时可能不容易及时释放，导致显存、内存占用不释放。

            所以这里用弱引用，含义是：
                我可以访问 root_physx_view，
                但我不负责延长它的生命周期。
        '''

        # Set initial time stamp
        self._sim_timestamp = 0.0
        '''
        初始化全局时钟
            这是所有 17 个 TimestampedBuffer 的统一时钟。
            每个 buffer 通过比较自己的 timestamp 和这个 _sim_timestamp 来判断数据是否过期。
        '''

        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()   # 拿到整个 PhysX 仿真世界的管理视图
        gravity = self._physics_sim_view.get_gravity()
        '''
        取得全局物理仿真视图，然后读取当前世界重力。比如常见情况是 (0, 0, -9.81)。z 轴向下
        '''
        # Convert to direction vector
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        '''
        把重力向量转成 torch 张量，并归一化成“方向”。也就是说，-9.81 的大小被去掉，只保留方向，例如大致变成 (0, 0, -1)。
        '''

        # Initialize constants
        # 生成每个机器人实例一份重力方向,后面用于计算机器人机体坐标系下的重力方向：
            # projected_gravity_b = quat_apply_inverse(root_link_quat_w, GRAVITY_VEC_W) 把世界坐标系里的重力方向，转换到机器人机体坐标系下
        self.GRAVITY_VEC_W = gravity_dir.repeat(self._root_physx_view.count, 1)
        # 表示机器人 base 坐标系里的“前方”方向，也就是 body frame 的 x 轴。后面用于计算朝向角：
            # forward_w = quat_apply(root_link_quat_w, FORWARD_VEC_B)
            # heading_w = atan2(forward_w[:, 1], forward_w[:, 0])       算出机器人在世界坐标系里的朝向角
        self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_physx_view.count, 1)
        '''
        重力方向 + 前进方向
            这两个向量是提前算好、不变的量：
                常量	           形状	        含义	                                    用途
                GRAVITY_VEC_W	(N, 3)	    世界坐标系中的重力方向（归一化）	            计算机器人的上下朝向、倾斜角;
                FORWARD_VEC_B	(N, 3)	    机器人自身坐标系的正前方 (body frame x 轴)	    计算速度方向是否为"向前"
            repeat() 的作用：
                gravity_dir 只是一个 (3,) 的向量，.repeat(N, 1) 把它广播成 (N, 3)——每个环境一份副本。
                这样后续做向量运算时不需要 broadcasting，更高效。
            一般机器人自身坐标系里：
                x 轴 = 前方
                y 轴 = 左右
                z 轴 = 上下
                所以：
                    (1.0, 0.0, 0.0)
                    表示机器人自己的正前方。
        '''

        # Initialize history for finite differencing
        self._previous_joint_vel = self._root_physx_view.get_dof_velocities().clone()
        '''
        get_dof_velocities() 在初始化时从 PhysX 拉取当前速度，作为第 0 帧的"上一帧"值。后续每帧更新时（articulation_data.py:1184）：
            self._previous_joint_vel[:] = self.joint_vel
        把当前速度存下来，供下一帧做差分。
        '''

        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame  连杆帧（相对于世界系）  这些是 link / actor frame 相对于世界系的数据。w 表示 world frame。
        self._root_link_pose_w = TimestampedBuffer()
        self._root_link_vel_w = TimestampedBuffer()
        self._body_link_pose_w = TimestampedBuffer()
        self._body_link_vel_w = TimestampedBuffer()
        # -- com frame w.r.t. link frame    质心帧（坐标系转换相关）    这是每个 body 的质心 frame 相对于自身 link frame 的位姿。b 这里表示 body/link 本体系。
        self._body_com_pose_b = TimestampedBuffer()
        # -- com frame w.r.t. world frame   质心帧（相对于世界系）  这些是质心 COM frame 相对于世界系的数据。PhysX 物理计算更关心 COM，而 USD/link 可视化和关节结构更常用 link frame。
        self._root_com_pose_w = TimestampedBuffer()
        self._root_com_vel_w = TimestampedBuffer()
        self._body_com_pose_w = TimestampedBuffer()
        self._body_com_vel_w = TimestampedBuffer()
        self._body_com_acc_w = TimestampedBuffer()
        # -- combined state (these are cached as they concatenate)  组合状态（缓存拼接结果）
        self._root_state_w = TimestampedBuffer()
        self._root_link_state_w = TimestampedBuffer()
        self._root_com_state_w = TimestampedBuffer()
        self._body_state_w = TimestampedBuffer()
        self._body_link_state_w = TimestampedBuffer()
        self._body_com_state_w = TimestampedBuffer()
        # -- joint state    关节状态    这些是关节状态：关节位置、关节速度、差分得到的关节加速度，以及关节传递到 body 的 wrench。
        self._joint_pos = TimestampedBuffer()
        self._joint_vel = TimestampedBuffer()
        self._joint_acc = TimestampedBuffer()
        self._body_incoming_joint_wrench_b = TimestampedBuffer()
        '''
        link frame 是什么？
                link frame 可以理解为机器人 URDF/USD 里定义的连杆坐标系。
            比如：
                base_link
                left_thigh_link
                left_calf_link
                left_wheel_link
                right_thigh_link
                ...
            这些 link 坐标系更多和机器人模型结构、可视化、关节树有关。
            代码里：
                self._root_link_pose_w
                self._root_link_vel_w
                self._body_link_pose_w
                self._body_link_vel_w
            表示的是 link frame 相对于 world frame 的位姿、速度。
            其中：
                pose = 位置 + 姿态
                vel = 线速度 + 角速度
                w = world frame
        com frame 是什么？
            com 是 center of mass，质心。
            com frame 是以刚体质心为参考的坐标系。
            物理仿真里，动力学计算更关心质心，因为刚体受力、惯性、动量等都围绕质心计算。
            所以代码里还有：
                self._root_com_pose_w
                self._root_com_vel_w
                self._body_com_pose_w
                self._body_com_vel_w
                self._body_com_acc_w
            这些是质心坐标系相对于世界坐标系的状态。
        link frame 和 com frame 不一定重合
            一个连杆的 link frame 可能定义在关节连接点，也可能定义在几何中心附近。
            但质心位置由质量分布决定。
            所以：
                link frame 原点 ≠ com frame 原点
            这就是为什么代码里要同时缓存 link 状态和 com 状态。
        root 和 body 的区别
            root 是什么？
                root 表示整个 articulation 的根刚体。
                对于浮动基机器人，比如轮腿机器人，root 通常就是：
                    base_link / trunk / pelvis
                所以：
                    self._root_link_pose_w
                    self._root_com_pose_w
                表示机器人根刚体的状态。
            body 是什么？
                body 表示 articulation 里面所有刚体。
                例如你的轮腿机器人可能有：
                    base
                    left_hip_link
                    left_thigh
                    left_calf
                    left_wheel
                    right_hip_link
                    right_thigh
                    right_calf
                    right_wheel
                所以：
                    self._body_link_pose_w
                    self._body_com_pose_w
                一般形状可能是：
                    (num_envs, num_bodies, 7)
                其中 7 通常表示：
                    x, y, z, qw, qx, qy, qz
            后缀 _w 和 _b 的含义
                一般含义是：
                    _w = world frame，世界坐标系
                    _b = body frame，本体坐标系 / 局部坐标系
                例如：
                    _root_link_pose_w
                表示：
                    root link 在世界坐标系下的 pose
                    _body_com_pose_b
                表示：
                    body 的质心坐标系相对于自身 body/link 坐标系的 pose
        组合状态缓存
            比如单独的 pose 可能是：
                [x, y, z, qw, qx, qy, qz]
            单独的 velocity 可能是：
                [vx, vy, vz, wx, wy, wz]
            组合 state 可能就是把它们拼起来：
                [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            所以 shape 可能是：
                (num_envs, 13)
            对于所有 body，则可能是：
                (num_envs, num_bodies, 13)
        关节状态缓存
            _joint_pos
                关节位置。
                对于旋转关节，就是角度，例如：
                    hip_roll 角度
                    hip_pitch 角度
                    knee 角度
                    wheel 角度
                shape 一般是：
                    (num_envs, num_joints)
            _joint_vel
                关节速度。
                对于旋转关节，就是角速度。
                shape 一般也是：
                    (num_envs, num_joints)
            _joint_acc
                关节加速度。
                通常由速度差分得到：
                    joint_acc = (joint_vel - previous_joint_vel) / dt
                它和前面的：
                    self._previous_joint_vel
                配合使用。
            _body_incoming_joint_wrench_b
                可以拆开理解：
                    body = 刚体
                    incoming_joint = 传入该刚体的关节
                    wrench = 力 + 力矩
                    b = body frame
                wrench 一般是 6 维：
                    [force_x, force_y, force_z, torque_x, torque_y, torque_z]
                所以它表示：通过关节传递到某个 body 上的力和力矩，并且是在 body 坐标系下表达。
                这个量在动力学分析、受力分析、调试关节受力时会用到。
        '''

    def update(self, dt: float):
        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the joint acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.joint_acc
        '''
        这里强制触发self.joint_acc计算，主要目的是强制每个物理步更新joint_vel/joint_acc/_previous_joint_vel这三个buffer
        如果依靠用户惰性触发的话，某个周期没有访问进行更新：
            物理步 1: _sim_timestamp = 0.005
            → 没人访问 joint_acc → _previous_joint_vel 停留在初始化时的旧值

            物理步 2: _sim_timestamp = 0.010
            → 没人访问 joint_acc → _previous_joint_vel 还是旧值

            物理步 3: _sim_timestamp = 0.015
            → 终于有人访问 joint_acc
            → 从 PhysX 读取当前 joint_vel
            → 加速度 = (vel_now - vel_init) / (0.015 - 0.0)
            → ❌ 算出来的是 3 个物理步的平均加速度，不是瞬时加速度！
            → 而且 _previous_joint_vel 被更新成了 vel_now
            → 两个物理步的速度历史永远丢失了
        '''

    ##
    # Names.
    ##

    '''
    这 4 个字段结构一致、作用相同，只是对应不同的关节组件：
        身体部件名称      ┌── body_names: list[str]
                        │     例: ["base", "LF_HIP", "LF_THIGH", "LF_SHANK", ...]
                        │
        关节名称          ├── joint_names: list[str]    ← 最重要
                        │     例: ["LF_HAA", "LF_HFE", "LF_KFE", ...]
                        │
        固定肌腱名称      ├── fixed_tendon_names: list[str]
                        │     例: ["tendon_0", "tendon_1", ...]
                        │
        空间肌腱名称      └── spatial_tendon_names: list[str]
                            例: ["spatial_tendon_0", ...]
    作用：Tensor 列索引 → 可读名称的映射
        这些字段解决了仿真中一个基本问题：PhysX 返回的数据是数值 tensor，但人类需要知道"这个数字是哪个关节的"。
            PhysX 返回的关节位置:
            tensor([[0.0, 0.4, -0.8, 0.0,-0.4, 0.8, ...
                    ↑    ↑     ↑    ↑    ↑    ↑
                    列0  列1   列2  列3  列4  列5  ← 这些数字是哪个关节？

            joint_names:
                    ["LF_HAA","LF_HFE","LF_KFE","LH_HAA","LH_HFE","LH_KFE",...]
                        ↑       ↑       ↑       ↑       ↑       ↑
                    列0→   列1→    列2→    列3→    列4→    列5→
        这四个名字列表就是列索引到可读名称的字典。
    '''

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""
    """在仿真视图中分析的顺序中，"""

    joint_names: list[str] = None
    """Joint names in the order parsed by the simulation view."""
    """在仿真视图中解析的顺序中，"""

    fixed_tendon_names: list[str] = None
    """Fixed tendon names in the order parsed by the simulation view."""
    """按仿真视图分析的顺序确定位名字。"""

    spatial_tendon_names: list[str] = None
    """Spatial tendon names in the order parsed by the simulation view."""
    """在仿真视图中解析的顺序中，空间位名称。"""

    ##
    # Defaults - Initial state.
    ##

    '''
    default_root_state: torch.Tensor   形状 (num_instances, 13)
    ├── [0:3]   pos       默认位置 (x, y, z)      ← 来自 cfg.init_state.pos
    ├── [3:7]   rot       默认朝向 (w, x, y, z)   ← 来自 cfg.init_state.rot
    ├── [7:10]  lin_vel   默认线速度                ← 来自 cfg.init_state.lin_vel
    └── [10:13] ang_vel   默认角速度                ← 来自 cfg.init_state.ang_vel
    '''
    default_root_state: torch.Tensor = None
    """Default root state ``[pos, quat, lin_vel, ang_vel]`` in the local environment frame.
    Shape is (num_instances, 13).

    The position and quaternion are of the articulation root's actor frame. Meanwhile, the linear and angular
    velocities are of its center of mass frame.

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """
    """在本地环境框架中的默认根状态``[pos， quat， lin_vel， ang_vel]``。
    形状是 (num_instances， 13)。

    位置和四元数是关节根的演员框架。
    与此同时，线性和角度速度是它的质量框架中心。

    这种数量通过:attr:`isaaclab.assets.ArticulationCfg.init_state`参数进行配置。
    """

    '''
    default_joint_pos: torch.Tensor    形状 (num_instances, num_joints)
        每个关节的默认角度                           ← 来自 cfg.init_state.joint_pos
    '''
    default_joint_pos: torch.Tensor = None
    """Default joint positions of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """
    """所有关节的默认关节位置。
    形状是 (num_instances，num_joints)。

    这种数量通过:attr:`isaaclab.assets.ArticulationCfg.init_state`参数进行配置。
    """

    '''
    default_joint_vel: torch.Tensor    形状 (num_instances, num_joints)
        每个关节的默认角速度                         ← 来自 cfg.init_state.joint_vel
    '''
    default_joint_vel: torch.Tensor = None
    """Default joint velocities of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the :attr:`isaaclab.assets.ArticulationCfg.init_state` parameter.
    """
    """所有关节的默认关节速度。
    形状是 (num_instances，num_joints)。

    这种数量通过:attr:`isaaclab.assets.ArticulationCfg.init_state`参数进行配置。
    """

    ##
    # Defaults - Physical properties.
    ##
    '''
    ┌── 刚体属性（2 个）────────────────────────────────
    │   default_mass                         每个连杆的质量
    │   default_inertia                      每个连杆的惯性张量 (9 维)
    │
    ├── 关节执行器属性（7 个）────────────────────────
    │   default_joint_stiffness              PD 控制器的刚度 (Kp)
    │   default_joint_damping                PD 控制器的阻尼 (Kd)
    │   default_joint_armature               关节电枢惯量
    │   default_joint_friction_coeff         静摩擦系数
    │   default_joint_dynamic_friction_coeff 动摩擦系数
    │   default_joint_viscous_friction_coeff 粘性摩擦系数
    │   default_joint_pos_limits             关节位置极限 [lower, upper]
    │
    ├── 固定肌腱属性（5 个）─────────────────────────
    │   default_fixed_tendon_stiffness       肌腱刚度
    │   default_fixed_tendon_damping         肌腱阻尼
    │   default_fixed_tendon_limit_stiffness 肌腱极限刚度
    │   default_fixed_tendon_rest_length     肌腱松弛长度
    │   default_fixed_tendon_offset          肌腱偏移量
    │   default_fixed_tendon_pos_limits      肌腱位置极限
    │
    └── 空间肌腱属性（4 个）─────────────────────────
        default_spatial_tendon_stiffness
        default_spatial_tendon_damping
        default_spatial_tendon_limit_stiffness
        default_spatial_tendon_offset

    default_* 的值从哪来？
        ┌── USD 文件（机器人模型）────────── 绝大多数字段
        │   如: mass, inertia, pos_limits, tendon_*
        │   通过 root_physx_view.get_xxx() 从 PhysX 读取
        │
        ├── 执行器配置（actuator cfg）──────── stiffness, damping, armature, friction
        │   如: actuator_cfg.stiffness = 40.0
        │   优先级高于 USD 中的值
        │
        └── 位置限制因子 ────────────────── soft_joint_pos_limits
            由 joint_pos_limits × soft_limit_factor 计算得出

    为什么要保留两份？
        因为执行器模型可能会覆盖这些值。
        例如，显式执行器（ExplicitActuator）会把 stiffness 设为 0（它自己管理力矩计算，不让 PhysX 的 PD 干扰）。
        如果你想"恢复出厂设置"，需要知道原始值是什么。default_* 就是这份备份。
    '''

    default_mass: torch.Tensor = None
    """Default mass for all the bodies in the articulation. Shape is (num_instances, num_bodies).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """在关节中所有身体的默认质量。
    形状是 (num_instances，num_bodies)。

    在初始化时，这个数量从USD方案中解析。
    """

    '''
    惯性张量（9 维）
    '''
    default_inertia: torch.Tensor = None
    """Default inertia for all the bodies in the articulation. Shape is (num_instances, num_bodies, 9).

    The inertia tensor should be given with respect to the center of mass, expressed in the articulation links'
    actor frame. The values are stored in the order
    :math:`[I_{xx}, I_{yx}, I_{zx}, I_{xy}, I_{yy}, I_{zy}, I_{xz}, I_{yz}, I_{zz}]`. However, due to the
    symmetry of inertia tensors, row- and column-major orders are equivalent.

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """在关节中的所有体体的默认惯性。
    形状是 (num_instances，num_bodies， 9)。

    按 mass结链的演员框架表达的质量中心应给出惯性子。
    值按顺序存储
    :math:`[I_{xx}， I_{yx}， I_{zx}， I_{xy}， I_{yy}， I_{zy}， I_{xz}， I_{yz}， I_{zz}]`然而，由于
    惰性子，排列和列大序列的对称性等等。

    在初始化时，这个数量从USD方案中解析。
    """

    '''
    PD 控制器的 Kp 和 Kd
    '''
    default_joint_stiffness: torch.Tensor = None
    """Default joint stiffness of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.stiffness`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.

    .. attention::
        The default stiffness is the value configured by the user or the value parsed from the USD schema.
        It should not be confused with :attr:`joint_stiffness`, which is the value set into the simulation.
    """
    """所有关节的默认关节硬度。
    形状是 (num_instances，num_joints)。

    这种数量通过执行器模型的:attr:`isaaclab.actuators.ActuatorBaseCfg.stiffness`参数进行配置。
    如果参数的值是None，则在初始化时使用从USD方案解析的值。

    .. 注意::
        默认硬度是用户配置的值或从USD方案中解析的值。
        它不应与:attr:`joint_stiffness`混，这是仿真中设置的值。
    """

    default_joint_damping: torch.Tensor = None
    """Default joint damping of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.damping`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.

    .. attention::
        The default stiffness is the value configured by the user or the value parsed from the USD schema.
        It should not be confused with :attr:`joint_damping`, which is the value set into the simulation.
    """
    """所有关节的默认关节缩。
    形状是 (num_instances，num_joints)。

    这种数量通过执行器模型的:attr:`isaaclab.actuators.ActuatorBaseCfg.damping`参数进行配置。
    如果参数的值是None，则在初始化时使用从USD方案解析的值。

    .. 注意::
        默认硬度是用户配置的值或从USD方案中解析的值。
        它不应与:attr:`joint_damping`混，这是仿真中设置的值。
    """

    '''
    armature 模拟了电机转子的额外转动惯量。
        真实电机的转子有质量，加速时需要额外力矩来克服转子本身的惯性。这个参数让仿真更接近真实电机响应。
    '''
    default_joint_armature: torch.Tensor = None
    """Default joint armature of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.armature`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.
    """
    """所有关节的默认关节 armature。
    形状是 (num_instances，num_joints)。

    这种数量通过执行器模型的:attr:`isaaclab.actuators.ActuatorBaseCfg.armature`参数进行配置。
    如果参数的值是None，则在初始化时使用从USD方案解析的值。
    """

    default_joint_friction_coeff: torch.Tensor = None
    """Default joint static friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's :attr:`isaaclab.actuators.ActuatorBaseCfg.friction`
    parameter. If the parameter's value is None, the value parsed from the USD schema, at the time of initialization,
    is used.

    Note:
        In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
        it is modeled as an effort (torque or force).
    """
    """所有关节的默认静态摩擦系数。
    形状是 (num_instances，num_joints)。

    这种数量通过执行器模型的:attr:`isaaclab.actuators.ActuatorBaseCfg.friction`参数进行配置。
    如果参数的值是None，则在初始化时使用从USD方案解析的值。

    说明：
        在Isaac Sim 4.5中，这个参数是作为一个系数的模型。
        在Isaac Sim 5.0及后版本中，它被仿真为功率 (扭矩或力)。
    """

    default_joint_dynamic_friction_coeff: torch.Tensor = None
    """Default joint dynamic friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's
    :attr:`isaaclab.actuators.ActuatorBaseCfg.dynamic_friction` parameter. If the parameter's value is None,
    the value parsed from the USD schema, at the time of initialization, is used.

    Note:
        In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
        it is modeled as an effort (torque or force).
    """
    """所有关节的默认关节动态摩擦系数
    形状是 (num_instances，num_joints)。

    这种数量通过执行器模型的:attr:`isaaclab.actuators.ActuatorBaseCfg.dynamic_friction`参数进行配置。
    如果参数的值是None，则在初始化时使用从USD方案解析的值。

    说明：
        在Isaac Sim 4.5中，这个参数是作为一个系数的模型。
        在Isaac Sim 5.0及后版本中，它被仿真为功率 (扭矩或力)。
    """

    default_joint_viscous_friction_coeff: torch.Tensor = None
    """Default joint viscous friction coefficient of all joints. Shape is (num_instances, num_joints).

    This quantity is configured through the actuator model's
    :attr:`isaaclab.actuators.ActuatorBaseCfg.viscous_friction` parameter. If the parameter's value is None,
    the value parsed from the USD schema, at the time of initialization, is used.
    """
    """所有关节的默认粘性摩擦系数。
    形状是 (num_instances，num_joints)。

    这种数量通过执行器模型的:attr:`isaaclab.actuators.ActuatorBaseCfg.viscous_friction`参数进行配置。
    如果参数的值是None，则在初始化时使用从USD方案解析的值。
    """

    default_joint_pos_limits: torch.Tensor = None
    """Default joint position limits of all joints. Shape is (num_instances, num_joints, 2).

    The limits are in the order :math:`[lower, upper]`. They are parsed from the USD schema at the
    time of initialization.
    """
    """所有关节的默认关节位置限制。
    形状是 (num_instances，num_joints，2)。

    极限是以数学为`[lower， upper]`的顺序。
    在启动时，它们从USD方案中解析。
    """

    default_fixed_tendon_stiffness: torch.Tensor = None
    """Default tendon stiffness of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """所有固定的子的默认硬性。
    形状是 (num_instances，num_fixed_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_fixed_tendon_damping: torch.Tensor = None
    """Default tendon damping of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """所有固定的fa门默认缩。
    形状是 (num_instances，num_fixed_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_fixed_tendon_limit_stiffness: torch.Tensor = None
    """Default tendon limit stiffness of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """默认的门限制了所有固定门的硬性。
    形状是 (num_instances，num_fixed_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_fixed_tendon_rest_length: torch.Tensor = None
    """Default tendon rest length of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """所有固定的 rest门休息长度是默认的。
    形状是 (num_instances，num_fixed_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_fixed_tendon_offset: torch.Tensor = None
    """Default tendon offset of all fixed tendons. Shape is (num_instances, num_fixed_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """部的缺陷，
    形状是 (num_instances，num_fixed_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_fixed_tendon_pos_limits: torch.Tensor = None
    """Default tendon position limits of all fixed tendons. Shape is (num_instances, num_fixed_tendons, 2).

    The position limits are in the order :math:`[lower, upper]`. They are parsed from the USD schema at the time of
    initialization.
    """
    """所有固定的 tend门位置限制。
    形状是 (num_instances，num_fixed_tendons，2)。

    位置限制是数学:`[lower， upper]`的顺序。
    在启动时，它们从USD方案中解析。
    """

    default_spatial_tendon_stiffness: torch.Tensor = None
    """Default tendon stiffness of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """所有空间肌的默认硬性。
    形状是 (num_instances，num_spatial_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_spatial_tendon_damping: torch.Tensor = None
    """Default tendon damping of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """所有空间肌的默认节。
    形状是 (num_instances，num_spatial_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_spatial_tendon_limit_stiffness: torch.Tensor = None
    """Default tendon limit stiffness of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """默认的门限制了所有空间门的硬性。
    形状是 (num_instances，num_spatial_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    default_spatial_tendon_offset: torch.Tensor = None
    """Default tendon offset of all spatial tendons. Shape is (num_instances, num_spatial_tendons).

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """所有空间的默认 of位。
    形状是 (num_instances，num_spatial_tendons)。

    在初始化时，这个数量从USD方案中解析。
    """

    ##
    # Joint commands -- Set into simulation.
    ##
    '''
        joint_pos_target:    torch.Tensor  (N, num_joints)  策略输出的目标关节角度
        joint_vel_target:    torch.Tensor  (N, num_joints)  策略输出的目标关节角速度
        joint_effort_target: torch.Tensor  (N, num_joints)  策略输出的目标关节力矩
    它们是策略输出和 PhysX 之间的中转站——数据流中的"用户缓冲区"。
    两层缓冲区架构：
        策略输出 actions
                │
                ▼
        action_manager.process_actions()
                │
                ▼
        set_joint_position_target(targets)       ← 第 1 层："用户缓冲区"
            └──→ data.joint_pos_target          （Python 侧，随时可读写）
                    │
                    ▼
        _apply_actuator_model()                  ← 执行器模型处理
            └──→ data.joint_pos_target ──→ actuator.compute() ──→ _joint_pos_target_sim
                                                                ↑ 第 2 层："仿真缓冲区"
                    │
                    ▼
        write_data_to_sim()
            └──→ _joint_pos_target_sim ──→ root_physx_view.set_dof_xxx() ──→ PhysX
    '''

    joint_pos_target: torch.Tensor = None
    """Joint position targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """
    """用户命令的关节位置目标。
    形状是 (num_instances，num_joints)。

    对于隐含的执行器模型，目标直接被设置在仿真中。
    对于明确的执行器模型，目标用于计算关节力矩 (见:attr:`applied_torque`)，然后设置在仿真中。
    """

    joint_vel_target: torch.Tensor = None
    """Joint velocity targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """
    """用户命令的关节速度目标。
    形状是 (num_instances，num_joints)。

    对于隐含的执行器模型，目标直接被设置在仿真中。
    对于明确的执行器模型，目标用于计算关节力矩 (见:attr:`applied_torque`)，然后设置在仿真中。
    """

    joint_effort_target: torch.Tensor = None
    """Joint effort targets commanded by the user. Shape is (num_instances, num_joints).

    For an implicit actuator model, the targets are directly set into the simulation.
    For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
    which are then set into the simulation.
    """
    """用户命令的联合努力目标。
    形状是 (num_instances，num_joints)。

    对于隐含的执行器模型，目标直接被设置在仿真中。
    对于明确的执行器模型，目标用于计算关节力矩 (见:attr:`applied_torque`)，然后设置在仿真中。
    """

    ##
    # Joint commands -- Explicit actuators.
    ##

    '''
        computed_torque ──→ 执行器模型输出的"原始力矩"
                                │
                            _clip_effort()
                                │
                                ▼
        applied_torque  ──→ 裁剪后的"实际施加力矩"（真正写入 PhysX）
    这是一对镜像字段，形状相同 (N, num_joints)，但值不同——applied_torque 是 computed_torque 经过电机限制裁剪后的结果。

    二、为什么需要两个？
        实际场景：策略输出了一个很大的力矩指令，执行器模型算出需要 150 N·m，但电机最大只能输出 80 N·m。
            computed_torque = 150  ← "我想要这么多"
            applied_torque  = 80   ← "实际上只能给这么多"
            差距 = 150 - 80 = 70  ← 如果把这个差值作为惩罚项，策略会学着不超限
        如果不暴露 computed_torque，策略永远不知道自己的指令被"截断"了——它看到 applied_torque=80，以为这就是它要的，无法学习到"不要输出过大指令"。
    '''

    computed_torque: torch.Tensor = None
    """Joint torques computed from the actuator model (before clipping). Shape is (num_instances, num_joints).

    This quantity is the raw torque output from the actuator mode, before any clipping is applied.
    It is exposed for users who want to inspect the computations inside the actuator model.
    For instance, to penalize the learning agent for a difference between the computed and applied torques.
    """
    """从动机模型计算的关节扭矩 (在切断之前)。
    形状是 (num_instances，num_joints)。

    在裁剪之前，该量是从动机模式中输出的原始扭矩。
    对于想要检查执行器模型内部计算的用户来说，
    例如，为计算和应用扭矩之间的差异处罚学习代理。
    """

    applied_torque: torch.Tensor = None
    """Joint torques applied from the actuator model (after clipping). Shape is (num_instances, num_joints).

    These torques are set into the simulation, after clipping the :attr:`computed_torque` based on the
    actuator model.
    """
    """从执行器模型上应用的关节扭矩 (裁剪后)。
    形状是 (num_instances，num_joints)。

    这些扭矩按动力驱动器模型切断:attr:`computed_torque`后设置在仿真中。
    """

    ##
    # Joint properties.
    ##
    '''
    这是 9 个可变的运行时关节参数，与前文讲的 default_* 版本一一对应：
        default_* (只读出厂值)               joint_* (可修改的运行时值)
        ─────────────────────────          ─────────────────────────
        default_joint_stiffness      ←→    joint_stiffness
        default_joint_damping        ←→    joint_damping
        default_joint_armature       ←→    joint_armature
        default_joint_friction_coeff ←→    joint_friction_coeff
        default_joint_dynamic_friction_coeff ←→ joint_dynamic_friction_coeff
        default_joint_viscous_friction_coeff ←→ joint_viscous_friction_coeff
        default_joint_pos_limits     ←→    joint_pos_limits
            (无对应 default)       ←→    joint_vel_limits
            (无对应 default)       ←→    joint_effort_limits
    核心设计：default_* vs joint_* 的"备份-工作"镜像
        这是我之前讲解 default_* 时提到的双重缓冲区设计。现在两边都出现了，可以完整对比：
            _initialize_impl 时：
                default_joint_stiffness = 从 USD/配置读取  ← 只读备份
                joint_stiffness         = default.clone()  ← 运行时工作副本

            运行时（显式执行器）：
                joint_stiffness[:, joint_ids] = 0.0  ← 修改工作副本
                # default_joint_stiffness 保持原值不变

            需要恢复时：
                joint_stiffness = default_joint_stiffness.clone()  ← 从备份恢复
    显式执行器为什么把 stiffness/damping 设为 0？
        显式执行器自己管理力矩计算，不能让 PhysX 的 PD 控制器也同时出力。
            隐式执行器（Implicit）:
                策略输出目标角度 → PhysX PD 控制器 → 力矩 = stiffness×(目标-当前) + damping×...
                                                                ↑           ↑
                                                            joint_stiffness  joint_damping
                                                            设为配置值       设为配置值

            显式执行器（Explicit）:
                策略输出目标角度 → LSTM/MLP 执行器模型 → 力矩 = network(目标, 当前状态)
                PhysX 的 PD 必须禁用:
                    joint_stiffness = 0
                    joint_damping   = 0         ← 这样 PhysX 不会额外施加 PD 力矩
    关节的物理极限，形状区别值得注意：
            字段	                形状	        含义
            joint_pos_limits	    (N, J, 2)	[0] = lower, [1] = upper
            joint_vel_limits	    (N, J)	    最大速度（正负对称，如 ±7.5 rad/s）
            joint_effort_limits	    (N, J)	    最大力矩（正负对称，如 ±80 N·m）
        位置极限是 (N, J, 2) 因为它有上下两个值，速度和力矩只有最大值（假设正负对称）。
        注意 joint_vel_limits 和 joint_effort_limits 没有对应的 default_* 版本——它们从 USD 解析出来后就保持不变，不需要备份-恢复机制。
    '''

    joint_stiffness: torch.Tensor = None
    """Joint stiffness provided to the simulation. Shape is (num_instances, num_joints).

    In the case of explicit actuators, the value for the corresponding joints is zero.
    """
    """在仿真过程中提供关节硬度。
    形状是 (num_instances，num_joints)。

    在明确执行器的情况下，对相应的关节的值为零。
    """

    joint_damping: torch.Tensor = None
    """Joint damping provided to the simulation. Shape is (num_instances, num_joints)

    In the case of explicit actuators, the value for the corresponding joints is zero.
    """
    """在仿真中提供了关节缩。
    形状是 (num_instances，num_joints)

    在明确执行器的情况下，对相应的关节的值为零。
    """

    joint_armature: torch.Tensor = None
    """Joint armature provided to the simulation. Shape is (num_instances, num_joints)."""
    """在仿真中提供的联合 armature。
    形状是 (num_instances，num_joints)。
    """

    joint_friction_coeff: torch.Tensor = None
    """Joint static friction coefficient provided to the simulation. Shape is (num_instances, num_joints).

    Note: In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
    it is modeled as an effort (torque or force).
    """
    """为仿真提供的联合静态摩擦系数。
    形状是 (num_instances，num_joints)。

    Note: 在Isaac Sim 4.5中，这个参数是作为一个系数的模型。
          在Isaac Sim 5.0及后版本中，
    它是以力力 (扭矩或力) 模型的。
    """

    joint_dynamic_friction_coeff: torch.Tensor = None
    """Joint dynamic friction coefficient provided to the simulation. Shape is (num_instances, num_joints).

    Note: In Isaac Sim 4.5, this parameter is modeled as a coefficient. In Isaac Sim 5.0 and later,
    it is modeled as an effort (torque or force).
    """
    """为仿真提供联合动态摩擦系数。
    形状是 (num_instances，num_joints)。

    Note: 在Isaac Sim 4.5中，这个参数是作为一个系数的模型。
          在Isaac Sim 5.0及后版本中，
    它是以力力 (扭矩或力) 模型的。
    """

    joint_viscous_friction_coeff: torch.Tensor = None
    """Joint viscous friction coefficient provided to the simulation. Shape is (num_instances, num_joints)."""
    """在仿真过程中提供的粘性摩擦系数。
    形状是 (num_instances，num_joints)。
    """

    joint_pos_limits: torch.Tensor = None
    """Joint position limits provided to the simulation. Shape is (num_instances, num_joints, 2).

    The limits are in the order :math:`[lower, upper]`.
    """
    """对仿真提供的关节位置限制。
    形状是 (num_instances，num_joints，2)。

    极限是以数学为`[lower， upper]`的顺序。
    """

    joint_vel_limits: torch.Tensor = None
    """Joint maximum velocity provided to the simulation. Shape is (num_instances, num_joints)."""
    """为仿真提供的联合最大速度。
    形状是 (num_instances，num_joints)。
    """

    joint_effort_limits: torch.Tensor = None
    """Joint maximum effort provided to the simulation. Shape is (num_instances, num_joints)."""
    """为仿真提供的最大共同努力。
    形状是 (num_instances，num_joints)。
    """

    ##
    # Joint properties - Custom.
    ##
    '''
    soft_joint_pos_limits — 关节的"安全活动区"
        核心思想：
            物理极限: [lower, upper]        ← PhysX 允许的范围
            软极限:   [soft_lower, soft_upper] ← 策略被训练在
                                                这个范围内活动

            当 factor=1.0: 软极限 = 物理极限（零缓冲区）
            当 factor=0.8: 软极限 = 物理极限的中间 80%（两端各留 10% 安全缓冲）
    soft_joint_vel_limits — 来自执行器模型的动态速度限制
        与 joint_vel_limits 的区别
                    joint_vel_limits	soft_joint_vel_limits
            来源	PhysX / USD 文件	    执行器模型
            可变性	固定（解析后不变）	    动态变化（执行器模型可能每步更新）
            典型值	7.5 rad/s	        可能 ≠ 7.5，取决于当前力矩、齿轮比等
        注释中特别提到变齿轮比执行器——当齿轮比变化时，电机的有效速度限制也跟着变。soft_joint_vel_limits 从执行器模型动态获取这个值。

        填充来源（articulation.py:2563）
                self._data.soft_joint_vel_limits[:, actuator.joint_indices] = actuator.velocity_limit
            每个物理步的 _apply_actuator_model 中，执行器模型更新自己的 velocity_limit，然后写入 data。
    gear_ratio — 减速比
        作用
            把电机的旋转运动转换为关节的线性/旋转运动。减速比定义了：
                关节力矩 = 电机力矩 × gear_ratio
                关节速度 = 电机速度 / gear_ratio
        对于大多数机器人，gear_ratio 是固定值（默认 1.0，初始化时设）。但对于变减速比执行器，这个值会每步更新。
            填充来源（articulation.py:2565-2566）
                if hasattr(actuator, "gear_ratio"):
                    self._data.gear_ratio[:, actuator.joint_indices] = actuator.gear_ratio
    '''

    soft_joint_pos_limits: torch.Tensor = None
    r"""Soft joint positions limits for all joints. Shape is (num_instances, num_joints, 2).

    The limits are in the order :math:`[lower, upper]`.The soft joint position limits are computed as
    a sub-region of the :attr:`joint_pos_limits` based on the
    :attr:`~isaaclab.assets.ArticulationCfg.soft_joint_pos_limit_factor` parameter.

    Consider the joint position limits :math:`[lower, upper]` and the soft joint position limits
    :math:`[soft_lower, soft_upper]`. The soft joint position limits are computed as:

    .. math::

        soft\_lower = (lower + upper) / 2 - factor * (upper - lower) / 2
        soft\_upper = (lower + upper) / 2 + factor * (upper - lower) / 2

    The soft joint position limits help specify a safety region around the joint limits. It isn't used by the
    simulation, but is useful for learning agents to prevent the joint positions from violating the limits.
    """
    """所有关节的柔软位置限制。
    形状是 (num_instances，num_joints，2)。

    极限是以数学顺序进行的:`[lower， upper]`软结合位置限制是计算为:attr:`joint_pos_limits`基于:attr:`~isaaclab.assets.Articulatio
    nCfg.soft_joint_pos_limit_factor`参数

    考虑关节位置限制:数学:`[lower， upper]`和软关节位置限制
    :math:`[soft_lower， soft_upper]`软关节位置限制计算为:

    .. math::

        soft\_lower = (lower + upper) / 2 - factor * (upper - lower) / 2
        soft\_upper = (lower + upper) / 2 + factor * (upper - lower) / 2

    柔性关节位置限制有助于确定关节限制周围的安全区域。
    它不是在仿真中使用的，但对于学习代理来说是有用的，
    """

    soft_joint_vel_limits: torch.Tensor = None
    """Soft joint velocity limits for all joints. Shape is (num_instances, num_joints).

    These are obtained from the actuator model. It may differ from :attr:`joint_vel_limits` if the actuator model
    has a variable velocity limit model. For instance, in a variable gear ratio actuator model.
    """
    """所有关节的软关节速度限制。
    形状是 (num_instances，num_joints)。

    这些来自动机模型。
    如果执行器模型具有可变速度限制模型，它可能与:attr:`joint_vel_limits`不同。
    例如，在变速率驱动器模型中。
    """

    gear_ratio: torch.Tensor = None
    """Gear ratio for relating motor torques to applied Joint torques. Shape is (num_instances, num_joints)."""
    """适用于运动扭矩和应用的关节力矩的交换比。
    形状是 (num_instances，num_joints)。
    """

    ##
    # Fixed tendon properties.
    ##
    '''
    这是 6 个固定肌腱（Fixed Tendon）的运行时参数，与 default_fixed_tendon_* 构成镜像：
        default_fixed_tendon_* (只读出厂值)       fixed_tendon_* (可修改运行时值)
        ────────────────────────────────         ─────────────────────────────
        default_fixed_tendon_stiffness      ←→   fixed_tendon_stiffness
        default_fixed_tendon_damping        ←→   fixed_tendon_damping
        default_fixed_tendon_limit_stiffness ←→  fixed_tendon_limit_stiffness
        default_fixed_tendon_rest_length    ←→   fixed_tendon_rest_length
        default_fixed_tendon_offset         ←→   fixed_tendon_offset
        default_fixed_tendon_pos_limits     ←→   fixed_tendon_pos_limits
    什么是固定肌腱？
        在 PhysX 中，固定肌腱（Fixed Tendon） 是一种特殊的约束，用于模拟绳索、钢缆、腱鞘等柔性连接。它与关节不同——它不是铰链或球窝，而是一根"绳子"：
            一个固定肌腱的例子：并联机构中连接两个关节的钢索

            电机 ────[钢索]──── 远处关节
                    ↑
                tendon（肌腱）
                电机转动 → 钢索拉紧/放松 → 远处关节运动
        空间肌腱（Spatial Tendon）是更复杂的版本，钢索可以穿过多个滑轮/锚点。
        对大多数机器人训练来说，肌腱用得很少——主要用于仿生机器人（如肌肉-肌腱驱动）或特殊并联机构。
    6 个参数的含义
        字段	        物理含义	                                            类比
        stiffness	    肌腱弹性系数，拉力 = stiffness × (当前长度 - 松弛长度)	    弹簧的 K 值
        damping	        肌腱阻尼系数，阻力 = damping × 长度变化速度	                减震器
        limit_stiffness	达到极限位置后的额外刚度（防止拉断）	                    弹簧触底后的缓冲垫
        rest_length	    肌腱的"自然长度"（不受力时的长度）	                        弹簧的自由长度
        offset	        肌腱的初始偏移量（调节初始张力）	                        预紧力调整螺丝
        pos_limits	    [lower, upper] 肌腱伸缩的物理极限	                      弹簧的最大压缩/拉伸
    通俗类比
        把固定肌腱想象成一根带弹簧的自行车刹车线：
            字段	        类比
            stiffness	    刹车线内弹簧的硬度
            damping	        刹车线护套的摩擦阻尼
            limit_stiffness	刹车捏到底后的"硬止档"
            rest_length	    不捏刹车时线的自由长度
            offset	        微调螺丝的位置（调节线的初始张力）
            pos_limits	    刹车把能捏多深
    '''

    fixed_tendon_stiffness: torch.Tensor = None
    """Fixed tendon stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
    """在仿真过程中提供固定的门硬度。
    形状是 (num_instances，num_fixed_tendons)。
    """

    fixed_tendon_damping: torch.Tensor = None
    """Fixed tendon damping provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
    """在仿真过程中提供固定门。
    形状是 (num_instances，num_fixed_tendons)。
    """

    fixed_tendon_limit_stiffness: torch.Tensor = None
    """Fixed tendon limit stiffness provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
    """在仿真中提供固定门限制硬度。
    形状是 (num_instances，num_fixed_tendons)。
    """

    fixed_tendon_rest_length: torch.Tensor = None
    """Fixed tendon rest length provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
    """在仿真过程中提供固定的肌休息长度。
    形状是 (num_instances，num_fixed_tendons)。
    """

    fixed_tendon_offset: torch.Tensor = None
    """Fixed tendon offset provided to the simulation. Shape is (num_instances, num_fixed_tendons)."""
    """在仿真过程中提供固定的肌肉偏移。
    形状是 (num_instances，num_fixed_tendons)。
    """

    fixed_tendon_pos_limits: torch.Tensor = None
    """Fixed tendon position limits provided to the simulation. Shape is (num_instances, num_fixed_tendons, 2)."""
    """在仿真过程中提供固定位限制。
    形状是 (num_instances，num_fixed_tendons，2)。
    """

    ##
    # Spatial tendon properties.
    ##
    '''
    4 个空间肌腱（Spatial Tendon）的运行时参数，与 default_spatial_tendon_* 构成镜像：
        default_spatial_tendon_* (只读出厂值)       spatial_tendon_* (可修改运行时值)
        ────────────────────────────────           ─────────────────────────────
        default_spatial_tendon_stiffness      ←→   spatial_tendon_stiffness
        default_spatial_tendon_damping        ←→   spatial_tendon_damping
        default_spatial_tendon_limit_stiffness ←→  spatial_tendon_limit_stiffness
        default_spatial_tendon_offset         ←→   spatial_tendon_offset
        与固定肌腱的关键区别：只有 4 个字段，缺少 rest_length 和 pos_limits。这是因为空间肌腱的"长度"由穿过多个锚点的钢索空间路径决定，不是一个简单的标量参数。

    空间肌腱 vs 固定肌腱
            固定肌腱（Fixed Tendon）:
                电机 ───[一根直钢索]─── 关节
                简单：长度 = |电机锚点 - 关节锚点|

            空间肌腱（Spatial Tendon）:
                电机 ───[锚点1]───[锚点2]───[锚点3]─── 关节
                                ↑
                        钢索弯曲穿过多个滑轮
                复杂：长度 = 折线路径的总长度，由空间坐标决定
        因为空间肌腱的"长度"由穿过所有锚点的 3D 路径动态决定，所以没有固定的 rest_length 和 pos_limits。其他 4 个弹性参数仍然适用。
    4 个参数的含义
        字段	        物理含义
        stiffness	    钢索的弹性系数
        damping	        钢索运动的阻尼
        limit_stiffness	钢索拉到极限后的额外刚度
        offset	        钢索的初始预紧力偏移
    通俗类比
        把空间肌腱想象成自行车的前后变速线：
                        固定肌腱	            空间肌腱
            类比	    前刹车线（直拉）	    后变速线（穿过车架多个卡扣，弯曲路径）
            长度	    固定长度	            由卡扣位置决定，无法单独调
            可调参数	张力、阻尼、极限、预紧力	相同（除了长度相关的）
    '''

    spatial_tendon_stiffness: torch.Tensor = None
    """Spatial tendon stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
    """在仿真过程中提供了空间硬性。
    形状是 (num_instances，num_spatial_tendons)。
    """

    spatial_tendon_damping: torch.Tensor = None
    """Spatial tendon damping provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
    """在仿真中提供了空间门。
    形状是 (num_instances，num_spatial_tendons)。
    """

    spatial_tendon_limit_stiffness: torch.Tensor = None
    """Spatial tendon limit stiffness provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
    """在仿真中提供空间门限制硬度。
    形状是 (num_instances，num_spatial_tendons)。
    """

    spatial_tendon_offset: torch.Tensor = None
    """Spatial tendon offset provided to the simulation. Shape is (num_instances, num_spatial_tendons)."""
    """在仿真过程中提供空间位。
    形状是 (num_instances，num_spatial_tendons)。
    """

    ##
    # Root state properties.
    ##

    '''
    root_link_pose_w → torch.Tensor  形状 (num_instances, 7)
        返回值：
            [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]，机器人根连杆在世界坐标系中的位姿。
        副作用：
            可能从 PhysX 读取数据并缓存到 _root_link_pose_w 缓冲区。
    '''
    @property
    def root_link_pose_w(self) -> torch.Tensor:
        """Root link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the articulation root's actor frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中，根链接呈现``[pos， quat]``。
        形状是 (num_instances， 7)。

        这个数量是关节根的演员框架相对于世界的姿势。
        导向提供 (w， x， y， z) 格式。
        """
        if self._root_link_pose_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_physx_view.get_root_transforms().clone()
            pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._root_link_pose_w.data = pose
            self._root_link_pose_w.timestamp = self._sim_timestamp

        return self._root_link_pose_w.data
    '''
    get_root_transforms() 返回什么？
    PhysX 的 ArticulationView.get_root_transforms() 返回 [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]——四元数是 (x, y, z, w) 格式（标量在后）。

    但 Isaac Lab 内部统一使用 (w, x, y, z) 格式（标量在前）。所以需要转换：
    '''

    '''
    root_link_vel_w → torch.Tensor  形状 (num_instances, 6)
        返回值：
            [lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]，根连杆在世界坐标系中的速度。
        核心计算：
            物理引擎只给质心速度，需要通过几何变换转为连杆速度。
    '''
    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's actor frame
        relative to the world.
        """
        """在仿真世界框架中的根链速度``[lin_vel， ang_vel]``。
        形状是 (num_instances， 6)。

        这个数量包含关节根的演员框架相对于世界的线性和角速度。
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            # read the CoM velocity
            vel = self.root_com_vel_w.clone()   # 读取质心速度
            # adjust linear velocity to link from center of mass
            vel[:, :3] += torch.linalg.cross(   # 物理变换 v_link=v_COM+ω×r
                vel[:, 3:], math_utils.quat_apply(self.root_link_quat_w, -self.body_com_pos_b[:, 0]), dim=-1
            )
            '''
            分步解析：
                vel[:, :3]               ← 当前是质心线速度
                vel[:, 3:]               ← 质心角速度 ω（绕世界坐标系的旋转轴）

                -self.body_com_pos_b[:, 0]  ← 从 link 到 COM 的偏移向量 r
                                            负号: body_com_pos_b 是 COM 在 body 系的位置
                                            取反 = link 在 COM 系的位置
                                            [:, 0] 是根连杆（body 0）

                quat_apply(root_link_quat_w, r_body)  ← 把 body 系中的 r 转到世界系
                                                        因为 ω 是世界系的，叉积必须同坐标系

                torch.linalg.cross(ω, r_world)        ← ω × r = 旋转引起的额外线速度

                vel[:, :3] += ω × r                   ← 质心速度 + 旋转补偿 = 连杆速度
            '''
            # set the buffer data and timestamp
            self._root_link_vel_w.data = vel
            self._root_link_vel_w.timestamp = self._sim_timestamp

        return self._root_link_vel_w.data
    '''
    物理背景：为什么不能直接从 PhysX 读取？
        PhysX 返回的根速度是根质心（Root COM）的速度。但 root_link_vel_w 要返回的是根连杆（Root Link）——即 Xform prim 所在位置——的速度。
        关键物理公式：
                v_link=v_COM+ω×r
            其中 r 是从 link 指向 COM 的向量，ω 是角速度。当机器人在旋转时，link frame 的位置因为离 COM 有一段距离，会产生额外的线速度。
    为什么 PhysX 不直接给 link 速度？
        PhysX 的关节 API 以质心为参考计算动力学，因为牛顿-欧拉方程在质心处最简洁。
        Isaac Lab 封装了从 COM 到 link 的坐标变换，让你用更直观的"机器人 Xform 所在位置的速度"。
    '''

    '''
    root_com_pose_w → torch.Tensor  形状 (num_instances, 7)
        返回值：[pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]，根质心（Root Center of Mass）在世界坐标系中的位姿。
    '''
    @property
    def root_com_pose_w(self) -> torch.Tensor:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the articulation root's center of mass frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中，质量中心 ``[pos， quat]``。
        形状是 (num_instances， 7)。

        这个数量是关节根的质量框架中心相对于世界的姿势。
        导向提供 (w， x， y， z) 格式。
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            pos, quat = math_utils.combine_frame_transforms(
                self.root_link_pos_w, self.root_link_quat_w, self.body_com_pos_b[:, 0], self.body_com_quat_b[:, 0]
            )
            # set the buffer data and timestamp
            self._root_com_pose_w.data = torch.cat((pos, quat), dim=-1)
            self._root_com_pose_w.timestamp = self._sim_timestamp

        return self._root_com_pose_w.data
    '''
    物理原理：从 Link 到 COM 的坐标变换
        这恰好是 root_link_pose_w 的"另一边"：
        root_link_pose_w     →  连杆帧在世界系中的位姿（"机器人外壳在哪"）
        root_com_pose_w      →  质心帧在世界系中的位姿（"机器人质量中心在哪"）
        数学公式：
            T_world→COM=T_world→link×T_link→COM
                其中 T_link→COM  是质心相对于连杆的偏移——这个值是静态的（不随时间变化），在 USD 文件中定义。
    '''

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the articulation root's center of mass frame
        relative to the world.
        """
        """大量速度的根中心``[lin_vel， ang_vel]``在仿真世界框架中。
        形状是 (num_instances， 6)。

        这种数量包含与世界相对的关节根质量框架中心的线性和角速度。
        """
        if self._root_com_vel_w.timestamp < self._sim_timestamp:
            self._root_com_vel_w.data = self._root_physx_view.get_root_velocities()
            self._root_com_vel_w.timestamp = self._sim_timestamp

        return self._root_com_vel_w.data
    '''
    什么是"根质心速度"？
        先拆开看这几个词：
            根（Root）：机器人最顶层的那条运动链的起点。比如四足机器人的身体底座，或者是机械臂的基座。URDF 里通常叫 base_link。
            质心（COM, Center of Mass）：刚体的质量分布中心。物理引擎计算动力学时，所有力、动量、惯性都在质心处处理最简洁。
            世界坐标系（w, world frame）：仿真场景的全局参考系，原点在地面某处。
        注意区分两个容易混淆的概念：
            root_link_vel_w：根连杆（link frame） 在世界系的速度。连杆坐标系是你在 URDF/USD 里定义的关节连接点，和可视化模型一致。
            root_com_vel_w：根质心（COM frame） 在世界系的速度。质心坐标系是物理引擎内部计算的，和真实物理惯性一致。
    '''

    '''
    root_state_w 是一个组合属性——它本身不去 PhysX 读取数据，而是把两个已有的属性拼在一起，形成一个"一站式"的机器人根状态接口。
    '''
    @property
    def root_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and quaternion are of the articulation root's actor frame relative to the world. Meanwhile,
        the linear and angular velocities are of the articulation root's center of mass frame.
        """
        """在仿真世界框架中的根状态``[pos， quat， lin_vel， ang_vel]``。
        形状是 (num_instances， 13)。

        位置和四元数是关节根与世界相对的演员框架。
        与此同时，线性和角的速度是关节根的质量框架中心。
        """
        if self._root_state_w.timestamp < self._sim_timestamp:
            self._root_state_w.data = torch.cat((self.root_link_pose_w, self.root_com_vel_w), dim=-1)
            self._root_state_w.timestamp = self._sim_timestamp

        return self._root_state_w.data
    '''
    这里把两个数据沿最后一维拼接：
            来源属性	            内容	                                                            形状	    参考系	                    说明
            root_link_pose_w	[pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]	            (N, 7)	    Actor frame（连杆坐标系）	位置 + 姿态（四元数 wxyz 格式）
            root_com_vel_w	    [lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]	(N, 6)	    COM frame（质心坐标系）	    线速度 + 角速度
            cat 结果	            —	                                                            (N, 13)	    混合！	                    7 + 6 = 13 维状态向量
        dim=-1 表示在最后一维上拼接。两个张量都是 (N, K) 的形状（K 分别是 7 和 6），沿 dim=-1（即 dim=1）拼接后变成 (N, 13)。
    '''

    '''
    root_link_state_w 的特点是：位姿和速度都在同一个参考系下（link frame / actor frame），所有13维数据描述的都是同一个物理点——连杆坐标系原点——的运动状态。
    '''
    @property
    def root_link_state_w(self):
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the articulation root's actor frame relative to the
        world.
        """
        """在仿真世界框架中的根状态``[pos， quat， lin_vel， ang_vel]``。
        形状是 (num_instances， 13)。

        位置，四元数和线性/角的速度是关节根的演员框架相对于世界。
        """
        if self._root_link_state_w.timestamp < self._sim_timestamp:
            self._root_link_state_w.data = torch.cat((self.root_link_pose_w, self.root_link_vel_w), dim=-1)
            self._root_link_state_w.timestamp = self._sim_timestamp

        return self._root_link_state_w.data

    @property
    def root_com_state_w(self):
        """Root center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the articulation root link's center of mass frame
        relative to the world. Center of mass frame is assumed to be the same orientation as the link rather than the
        orientation of the principle inertia.
        """
        """在仿真世界框架中的质量状态``[pos， quat， lin_vel， ang_vel]``的根中心。
        形状是 (num_instances， 13)。

        位置，四元数和线性/角速度是关节根链的质量框架中心相对于世界。
        质量框架的中心被认为是与链接相同的方向，而不是惯性原则的方向。
        """
        if self._root_com_state_w.timestamp < self._sim_timestamp:
            self._root_com_state_w.data = torch.cat((self.root_com_pose_w, self.root_com_vel_w), dim=-1)
            self._root_com_state_w.timestamp = self._sim_timestamp

        return self._root_com_state_w.data

    ##
    # Body state properties.
    ##

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        """Body link pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the articulation links' actor frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中，身体链接呈现``[pos， quat]``。
        形状是 (num_instances，num_bodies， 7)。

        这个数量是关节链的演员框架相对于世界的姿势。
        导向提供 (w， x， y， z) 格式。
        """
        if self._body_link_pose_w.timestamp < self._sim_timestamp:
            # perform forward kinematics (shouldn't cause overhead if it happened already)
            self._physics_sim_view.update_articulations_kinematic()
            # read data from simulation
            poses = self._root_physx_view.get_link_transforms().clone()
            poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._body_link_pose_w.data = poses
            self._body_link_pose_w.timestamp = self._sim_timestamp

        return self._body_link_pose_w.data

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' actor frame
        relative to the world.
        """
        """在仿真世界框架中，身体链接速度``[lin_vel， ang_vel]``。
        形状是 (num_instances，num_bodies， 6)。

        这个数量包含关节链的演员框架相对于世界的线性和角速度。
        """
        if self._body_link_vel_w.timestamp < self._sim_timestamp:
            # read data from simulation
            velocities = self.body_com_vel_w.clone()
            # adjust linear velocity to link from center of mass
            velocities[..., :3] += torch.linalg.cross(
                velocities[..., 3:], math_utils.quat_apply(self.body_link_quat_w, -self.body_com_pos_b), dim=-1
            )
            # set the buffer data and timestamp
            self._body_link_vel_w.data = velocities
            self._body_link_vel_w.timestamp = self._sim_timestamp

        return self._body_link_vel_w.data

    @property
    def body_com_pose_w(self) -> torch.Tensor:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 7).

        This quantity is the pose of the center of mass frame of the articulation links relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中，体质中心 ``[pos， quat]``。
        形状是 (num_instances，num_bodies， 7)。

        这个数量是对世界相对的关节链的质量框架中心的姿势。
        导向提供 (w， x， y， z) 格式。
        """
        if self._body_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            pos, quat = math_utils.combine_frame_transforms(
                self.body_link_pos_w, self.body_link_quat_w, self.body_com_pos_b, self.body_com_quat_b
            )
            # set the buffer data and timestamp
            self._body_com_pose_w.data = torch.cat((pos, quat), dim=-1)
            self._body_com_pose_w.timestamp = self._sim_timestamp

        return self._body_com_pose_w.data

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 6).

        This quantity contains the linear and angular velocities of the articulation links' center of mass frame
        relative to the world.
        """
        """在仿真世界框架中体积速度``[lin_vel， ang_vel]``的中心。
        形状是 (num_instances，num_bodies， 6)。

        这个数量包含与世界相对的质量框架的关节链的直线和角速度。
        """
        if self._body_com_vel_w.timestamp < self._sim_timestamp:
            self._body_com_vel_w.data = self._root_physx_view.get_link_velocities()
            self._body_com_vel_w.timestamp = self._sim_timestamp

        return self._body_com_vel_w.data

    @property
    def body_state_w(self):
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position and quaternion are of all the articulation links' actor frame. Meanwhile, the linear and angular
        velocities are of the articulation links's center of mass frame.
        """
        """在仿真世界框架中所有物体的状态 `[pos， quat， lin_vel， ang_vel]`。
        形状是 (num_instances，num_bodies， 13)。

        位置和四元数是所有关节链的演员框架。
        与此同时，线性和角的速度是关节链的质量框架中心。
        """
        if self._body_state_w.timestamp < self._sim_timestamp:
            self._body_state_w.data = torch.cat((self.body_link_pose_w, self.body_com_vel_w), dim=-1)
            self._body_state_w.timestamp = self._sim_timestamp

        return self._body_state_w.data

    @property
    def body_link_state_w(self):
        """State of all bodies' link frame`[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        """
        """在仿真世界框架中，所有机体的链接框架`[pos， quat， lin_vel， ang_vel]`状态。
        形状是 (num_instances，num_bodies， 13)。

        位置，四元数和线性/角速度是身体与世界相对的链接框架。
        """
        if self._body_link_state_w.timestamp < self._sim_timestamp:
            self._body_link_state_w.data = torch.cat((self.body_link_pose_w, self.body_link_vel_w), dim=-1)
            self._body_link_state_w.timestamp = self._sim_timestamp

        return self._body_link_state_w.data

    @property
    def body_com_state_w(self):
        """State of all bodies center of mass `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia.
        """
        """在仿真世界框架中所有物体的质量中心`[pos， quat， lin_vel， ang_vel]`的状态。
        形状是 (num_instances，num_bodies， 13)。

        位置，四元数和线性/角的速度是身体与世界相对的质量框架中心。
        质量框架的中心被认为是与链接相同的方向，而不是惯性原则的方向。
        """
        if self._body_com_state_w.timestamp < self._sim_timestamp:
            self._body_com_state_w.data = torch.cat((self.body_com_pose_w, self.body_com_vel_w), dim=-1)
            self._body_com_state_w.timestamp = self._sim_timestamp

        return self._body_com_state_w.data

    @property
    def body_com_acc_w(self):
        """Acceleration of all bodies center of mass ``[lin_acc, ang_acc]``.
        Shape is (num_instances, num_bodies, 6).

        All values are relative to the world.
        """
        """所有物体的加速重量中心``[lin_acc， ang_acc]``。
        形状是 (num_instances，num_bodies， 6)。

        所有的价值观都与世界相对。
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._body_com_acc_w.data = self._root_physx_view.get_link_accelerations()
            self._body_com_acc_w.timestamp = self._sim_timestamp

        return self._body_com_acc_w.data

    @property
    def body_com_pose_b(self) -> torch.Tensor:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.
        Shape is (num_instances, 1, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (w, x, y, z) format.
        """
        """所有体体在各自体的链接框架中，质量中心 ``[pos， quat]``。
        形状是 (num_instances， 1， 7)。

        这个数量是硬体质量框架中心的姿势与身体的链接框架相比。
        导向提供 (w， x， y， z) 格式。
        """
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_physx_view.get_coms().to(self.device)
            pose[..., 3:7] = math_utils.convert_quat(pose[..., 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._body_com_pose_b.data = pose
            self._body_com_pose_b.timestamp = self._sim_timestamp

        return self._body_com_pose_b.data

    @property
    def body_incoming_joint_wrench_b(self) -> torch.Tensor:
        """Joint reaction wrench applied from body parent to child body in parent body frame.

        Shape is (num_instances, num_bodies, 6). All body reaction wrenches are provided including the root body to the
        world of an articulation.

        For more information on joint wrenches, please check the`PhysX documentation`_ and the underlying
        `PhysX Tensor API`_.

        .. _`PhysX documentation`: https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force
        .. _`PhysX Tensor API`: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force
        """
        """从父母身体到孩子身体的联合反应钥匙在父母身体框架中应用。

        形状是 (num_instances，num_bodies， 6)。
        所有的身体反应钥匙都提供，包括根体，

        有关关键的更多信息，请查看`PhysX documentation`_和底层`PhysX Tensor API`_。

        .. _`PhysX documentation`: https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force
        .. _`PhysX Tensor API`: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_incoming_joint_force
        """

        if self._body_incoming_joint_wrench_b.timestamp < self._sim_timestamp:
            self._body_incoming_joint_wrench_b.data = self._root_physx_view.get_link_incoming_joint_force()
            self._body_incoming_joint_wrench_b.time_stamp = self._sim_timestamp
        return self._body_incoming_joint_wrench_b.data

    ##
    # Joint state properties.
    ##

    @property
    def joint_pos(self):
        """Joint positions of all joints. Shape is (num_instances, num_joints)."""
        """所有关节的关节位置。
        形状是 (num_instances，num_joints)。
        """
        if self._joint_pos.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_pos.data = self._root_physx_view.get_dof_positions()
            self._joint_pos.timestamp = self._sim_timestamp
        return self._joint_pos.data

    @property
    def joint_vel(self):
        """Joint velocities of all joints. Shape is (num_instances, num_joints)."""
        """所有关节的关节速度。
        形状是 (num_instances，num_joints)。
        """
        if self._joint_vel.timestamp < self._sim_timestamp:
            # read data from simulation and set the buffer data and timestamp
            self._joint_vel.data = self._root_physx_view.get_dof_velocities()
            self._joint_vel.timestamp = self._sim_timestamp
        return self._joint_vel.data

    @property
    def joint_acc(self):
        """Joint acceleration of all joints. Shape is (num_instances, num_joints)."""
        """所有关节的联合加速。
        形状是 (num_instances，num_joints)。
        """
        '''
            从 PhysX 读取最新关节速度（通过 self.joint_vel）
            用有限差分计算加速度：(当前速度 - 上一帧速度) / dt
            更新 _previous_joint_vel：把当前速度存为"上一帧速度"，供下一次差分使用
        '''
        if self._joint_acc.timestamp < self._sim_timestamp:
            # note: we use finite differencing to compute acceleration
            time_elapsed = self._sim_timestamp - self._joint_acc.timestamp
            self._joint_acc.data = (self.joint_vel - self._previous_joint_vel) / time_elapsed
            self._joint_acc.timestamp = self._sim_timestamp
            # update the previous joint velocity
            self._previous_joint_vel[:] = self.joint_vel
        return self._joint_acc.data

    ##
    # Derived Properties.
    ##

    @property
    def projected_gravity_b(self):
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        """在基架上投射重力方向。
        形状是 (num_instances， 3)。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        """基架的 Yaw方向 (在半径中)。
        形状是 (num_instances，)。

        说明：
            这个数量是通过假设基架的前向方向沿着x方向计算的，i.e.:数学:`(1， 0， 0)`。
        """
        forward_w = math_utils.quat_apply(self.root_link_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        """根链的线性速度在基架中。
        形状是 (num_instances， 3)。

        这个数量是关节根的演员框架与其演员框架的线性速度。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's actor frame with respect to the
        its actor frame.
        """
        """根链角速度在基础世界框架。
        形状是 (num_instances， 3)。

        这个数量是关节根的演员框架与其演员框架的角速度。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        """在基架中，质量线性速度的根中心。
        形状是 (num_instances， 3)。

        这个数量是关节根的质量框架中心与其演员框架的线性速度。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the articulation root's center of mass frame with respect to the
        its actor frame.
        """
        """基层世界框架中的质量角速度的根中心。
        形状是 (num_instances， 3)。

        这个数量是关节根的质量框架中心与其演员框架的角速度。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_ang_vel_w)

    ##
    # Sliced properties.
    ##

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        """在仿真世界框架中的根链位置。
        形状是 (num_instances， 3)。

        这种数量是根固体与世界相对的演员框架的位置。
        """
        return self.root_link_pose_w[:, :3]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        """在仿真世界框架中，根链的导向 (w，x，y，z)。
        形状是 (num_instances， 4)。

        这种数量是根固体的演员框架的方向。
        """
        return self.root_link_pose_w[:, 3:7]

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        """在仿真世界框架中的根线性速度。
        形状是 (num_instances， 3)。

        这个数量是根固体的演员框架相对于世界的线性速度。
        """
        return self.root_link_vel_w[:, :3]

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        """在仿真世界框架中的根链角速度。
        形状是 (num_instances， 3)。

        这个数量是根固体与世界相对的演员框架的角速度。
        """
        return self.root_link_vel_w[:, 3:6]

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        """在仿真世界框架中，
        形状是 (num_instances， 3)。

        这种数量是根固体与世界相对的演员框架的位置。
        """
        return self.root_com_pose_w[:, :3]

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        """在仿真世界框架中的质量导向的根中心 (w，x，y，z)。
        形状是 (num_instances， 4)。

        这种数量是根固体与世界相对的演员框架的方向。
        """
        return self.root_com_pose_w[:, 3:7]

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        """在仿真世界框架中的质量线性速度的根中心。
        形状是 (num_instances， 3)。

        这个数量是根固体质量框架中心与世界相对的线性速度。
        """
        return self.root_com_vel_w[:, :3]

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        """在仿真世界框架中的质量角速度的根中心。
        形状是 (num_instances， 3)。

        这个数量是根固体质量框架中心与世界相对的角速度。
        """
        return self.root_com_vel_w[:, 3:6]

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame relative to the world.
        """
        """仿真世界框架中的所有物体的位置。
        形状是 (num_instances，num_bodies， 3)。

        这种数量是关节体的演员框架与世界相对的位置。
        """
        return self.body_link_pose_w[..., :3]

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame relative to the world.
        """
        """在仿真世界框架中的所有物体的导向 (w，x，y，z)。
        形状是 (num_instances，num_bodies， 4)。

        这种数量是关节体的演员框架与世界相对的方向。
        """
        return self.body_link_pose_w[..., 3:7]

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame relative to the world.
        """
        """在仿真世界框架中的所有物体的线性速度。
        形状是 (num_instances，num_bodies， 3)。

        这个数量是关节体质量框架中心的线性速度相对于世界。
        """
        return self.body_link_vel_w[..., :3]

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame relative to the world.
        """
        """在仿真世界框架中的所有物体的角速度。
        形状是 (num_instances，num_bodies， 3)。

        这个数量是关节体质量框架中心与世界相对的角速度。
        """
        return self.body_link_vel_w[..., 3:6]

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the position of the articulation bodies' actor frame.
        """
        """仿真世界框架中的所有物体的位置。
        形状是 (num_instances，num_bodies， 3)。

        这种数量是关节体的演员框架的位置。
        """
        return self.body_com_pose_w[..., :3]

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all bodies in simulation world frame.
        Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the articulation bodies' actor frame.
        """
        """在仿真世界框架中所有物体的惯性基本轴的导向 (w，x，y，z)。
        形状是 (num_instances，num_bodies， 4)。

        这种数量是关节体的演员框架的方向。
        """
        return self.body_com_pose_w[..., 3:7]

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear velocity of the articulation bodies' center of mass frame.
        """
        """在仿真世界框架中的所有物体的线性速度。
        形状是 (num_instances，num_bodies， 3)。

        这个数量是关节体质量框架中心的线性速度。
        """
        return self.body_com_vel_w[..., :3]

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular velocity of the articulation bodies' center of mass frame.
        """
        """在仿真世界框架中的所有物体的角速度。
        形状是 (num_instances，num_bodies， 3)。

        这个数量是关节体质量框架中心的角速度。
        """
        return self.body_com_vel_w[..., 3:6]

    @property
    def body_com_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the linear acceleration of the articulation bodies' center of mass frame.
        """
        """在仿真世界框架中的所有物体的线性加速。
        形状是 (num_instances，num_bodies， 3)。

        这种数量是关节体质量框架中心的线性加速。
        """
        return self.body_com_acc_w[..., :3]

    @property
    def body_com_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, num_bodies, 3).

        This quantity is the angular acceleration of the articulation bodies' center of mass frame.
        """
        """在仿真世界框架中的所有物体的角加速。
        形状是 (num_instances，num_bodies， 3)。

        这种数量是关节体质量框架中心的角加速。
        """
        return self.body_com_acc_w[..., 3:6]

    @property
    def body_com_pos_b(self) -> torch.Tensor:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, num_bodies, 3).

        This quantity is the center of mass location relative to its body'slink frame.
        """
        """所有物体在各自的链接框架中的质量位置中心。
        形状是 (num_instances，num_bodies， 3)。

        这个数量是相对于其身体的斜体位置的中心。
        """
        return self.body_com_pose_b[..., :3]

    @property
    def body_com_quat_b(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, num_bodies, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        """所有物体在各自的链接框架中的惯性轴的方向 (w，x，y，z)。
        形状是 (num_instances，num_bodies， 4)。

        这种数量是对其身体的链接框架的惯性轴的方向。
        """
        return self.body_com_pose_b[..., 3:7]

    ##
    # Backward compatibility.
    ##
    '''
    向后兼容别名（直接跳过，约15个属性）
        这些是旧 API 名的别名，指向新名字。比如 root_pose_w → root_link_pose_w，root_vel_w → root_com_vel_w。
        永远不要用这些别名，它们只是为了兼容老代码保留的。
    '''

    @property
    def root_pose_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_pose_w`."""
        """像:attr:`root_link_pose_w`一样。"""
        return self.root_link_pose_w

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_pos_w`."""
        """像:attr:`root_link_pos_w`一样。"""
        return self.root_link_pos_w

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_quat_w`."""
        """像:attr:`root_link_quat_w`一样。"""
        return self.root_link_quat_w

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_vel_w`."""
        """像:attr:`root_com_vel_w`一样。"""
        return self.root_com_vel_w

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_lin_vel_w`."""
        """像:attr:`root_com_lin_vel_w`一样。"""
        return self.root_com_lin_vel_w

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_ang_vel_w`."""
        """像:attr:`root_com_ang_vel_w`一样。"""
        return self.root_com_ang_vel_w

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Same as :attr:`root_com_lin_vel_b`."""
        """像:attr:`root_com_lin_vel_b`一样。"""
        return self.root_com_lin_vel_b

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Same as :attr:`root_com_ang_vel_b`."""
        """像:attr:`root_com_ang_vel_b`一样。"""
        return self.root_com_ang_vel_b

    @property
    def body_pose_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_pose_w`."""
        """像:attr:`body_link_pose_w`一样。"""
        return self.body_link_pose_w

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_pos_w`."""
        """像:attr:`body_link_pos_w`一样。"""
        return self.body_link_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_quat_w`."""
        """像:attr:`body_link_quat_w`一样。"""
        return self.body_link_quat_w

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_vel_w`."""
        """像:attr:`body_com_vel_w`一样。"""
        return self.body_com_vel_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_lin_vel_w`."""
        """像:attr:`body_com_lin_vel_w`一样。"""
        return self.body_com_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_ang_vel_w`."""
        """像:attr:`body_com_ang_vel_w`一样。"""
        return self.body_com_ang_vel_w

    @property
    def body_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_acc_w`."""
        """像:attr:`body_com_acc_w`一样。"""
        return self.body_com_acc_w

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_lin_acc_w`."""
        """像:attr:`body_com_lin_acc_w`一样。"""
        return self.body_com_lin_acc_w

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_ang_acc_w`."""
        """像:attr:`body_com_ang_acc_w`一样。"""
        return self.body_com_ang_acc_w

    @property
    def com_pos_b(self) -> torch.Tensor:
        """Same as :attr:`body_com_pos_b`."""
        """像:attr:`body_com_pos_b`一样。"""
        return self.body_com_pos_b

    @property
    def com_quat_b(self) -> torch.Tensor:
        """Same as :attr:`body_com_quat_b`."""
        """像:attr:`body_com_quat_b`一样。"""
        return self.body_com_quat_b

    @property
    def joint_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_pos_limits` instead."""
        """废弃的财产。
        请使用:attr:`joint_pos_limits`。
        """
        logger.warning(
            "The `joint_limits` property will be deprecated in a future release. Please use `joint_pos_limits` instead."
        )
        return self.joint_pos_limits

    @property
    def default_joint_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_joint_pos_limits` instead."""
        """废弃的财产。
        请使用:attr:`default_joint_pos_limits`。
        """
        logger.warning(
            "The `default_joint_limits` property will be deprecated in a future release. Please use"
            " `default_joint_pos_limits` instead."
        )
        return self.default_joint_pos_limits

    @property
    def joint_velocity_limits(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_vel_limits` instead."""
        """废弃的财产。
        请使用:attr:`joint_vel_limits`。
        """
        logger.warning(
            "The `joint_velocity_limits` property will be deprecated in a future release. Please use"
            " `joint_vel_limits` instead."
        )
        return self.joint_vel_limits

    @property
    def joint_friction(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`joint_friction_coeff` instead."""
        """废弃的财产。
        请使用:attr:`joint_friction_coeff`。
        """
        logger.warning(
            "The `joint_friction` property will be deprecated in a future release. Please use"
            " `joint_friction_coeff` instead."
        )
        return self.joint_friction_coeff

    @property
    def default_joint_friction(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_joint_friction_coeff` instead."""
        """废弃的财产。
        请使用:attr:`default_joint_friction_coeff`。
        """
        logger.warning(
            "The `default_joint_friction` property will be deprecated in a future release. Please use"
            " `default_joint_friction_coeff` instead."
        )
        return self.default_joint_friction_coeff

    @property
    def fixed_tendon_limit(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`fixed_tendon_pos_limits` instead."""
        """废弃的财产。
        请使用:attr:`fixed_tendon_pos_limits`。
        """
        logger.warning(
            "The `fixed_tendon_limit` property will be deprecated in a future release. Please use"
            " `fixed_tendon_pos_limits` instead."
        )
        return self.fixed_tendon_pos_limits

    @property
    def default_fixed_tendon_limit(self) -> torch.Tensor:
        """Deprecated property. Please use :attr:`default_fixed_tendon_pos_limits` instead."""
        """废弃的财产。
        请使用:attr:`default_fixed_tendon_pos_limits`。
        """
        logger.warning(
            "The `default_fixed_tendon_limit` property will be deprecated in a future release. Please use"
            " `default_fixed_tendon_pos_limits` instead."
        )
        return self.default_fixed_tendon_pos_limits
