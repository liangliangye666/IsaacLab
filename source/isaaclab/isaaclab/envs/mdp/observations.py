# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""
"""可以用来创建观测项的共同函数。

函数可以传递到:class:`isaaclab.managers.ObservationTermCfg`对象，以实现函数引入的观测。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)

"""
Root state.
"""
"""根源状态。
"""

'''
@generic_io_descriptor(
    units="m",                              # 物理单位：米
    axes=["Z"],                             # 轴标签：Z 轴
    observation_type="RootState",            # 观测类别：根状态
    on_inspect=[record_shape, record_dtype]  # 自动记录形状和类型
)
@generic_io_descriptor(...)
    是一个装饰器（decorator）。它把 base_pos_z 函数"包装"了一下——在函数被调用时，自动记录输出的元数据（形状、类型），用于后续的模型导出和部署。

on_inspect 钩子的作用
        on_inspect=[record_shape, record_dtype]
    这两个是钩子函数，在函数被调用且 inspect=True 时自动执行：
        # 正常调用（每步训练）:
        base_pos_z(env)                          # → 正常返回 [N, 1]，不触发钩子

        # inspect 调用（导出描述符时）:
        base_pos_z(env, inspect=True)            # → 返回 [N, 1]
                                                # → 同时触发 record_shape → descriptor.shape = (1,)
                                                # → 同时触发 record_dtype → descriptor.dtype = "torch.float32"
'''

'''
返回机器人基座的高度（世界坐标系 Z 坐标）
    输出： torch.Tensor，形状 [N, 1]，N 个环境各自的基础高度（米）
'''
@generic_io_descriptor(units="m", axes=["Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype])
def base_pos_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in the simulation world frame."""
    """在仿真世界框架中的根高度。"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)    # unsqueeze(-1) 在最后一维增加一个维度：[N] → [N, 1]。
'''
使用示例
    @configclass
    class ObservationsCfg:
        """策略观测：机器人基座高度"""

        base_height = ObservationTermCfg(
            func=observations.base_pos_z,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
    效果： 策略的观测向量中多了 1 维——机器人当前离地面多高。

env 是由 ObservationManager 自动注入的，你在配置中不需要也不能手动传。
    管理器自动注入了什么
        回顾 EventManager.apply() 的调用方式（观测管理器同理）：
            # 管理器内部调用观测函数:
            term_cfg.func(self._env, **term_cfg.params)
            #             ↑ 自动注入      ↑ 你配置的参数
        所以函数签名中的前两个参数是被管理器自动填的：
            def base_lin_vel(env, asset_cfg):
                            ↑          ↑
                        管理器自动填    你在 params 中填

            def base_ang_vel(env, asset_cfg):
                            ↑          ↑
                        管理器自动填    你在 params 中填
        正确 vs 错误
            # ✅ 正确：只写函数自己的参数
            ObservationTermCfg(
                func=observations.base_lin_vel,
                params={"asset_cfg": SceneEntityCfg("robot")},  # 只写 asset_cfg
            )

            # ❌ 错误：不要把 env 放进 params
            ObservationTermCfg(
                func=observations.base_lin_vel,
                params={"env": env, "asset_cfg": ...},  # ← 这样会出错
            )
'''

'''
返回机器人基座的线速度（在机体坐标系中）
    输出： torch.Tensor，形状 [N, 3] — [vx, vy, vz]，单位 m/s
'''
@generic_io_descriptor(
    units="m/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    """在资产的根框架中的根线性速度。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b
'''
使用示例
    @configclass
    class ObservationsCfg:
        """观测：机器人速度 + 命令速度"""

        # 机器人实际速度（机体坐标系）
        base_velocity = ObservationTermCfg(
            func=observations.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # 命令速度（也来自机体坐标系，可以直接对比）
        # 这里通常还有 commands 观测，两者坐标系一致方便对比
'''


'''
返回机体坐标系下的角速度 [ωx, ωy, ωz]，单位 rad/s
    输出： [N, 3]
'''
@generic_io_descriptor(
    units="rad/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    """在资产的根框架中的根角速度。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b
'''
使用示例
    @configclass
    class ObservationsCfg:
        base_ang_vel = ObservationTermCfg(
            func=observations.base_ang_vel,
            # env 由管理器自动注入，不需要写
        )
'''


'''
返回重力方向在机体坐标系中的投影。
用一个 3 维向量隐式地告诉策略"机器人现在往哪边歪"。
这是 Isaac Lab 行走任务中最重要的观测之一——比欧拉角更好，因为它没有角度表示的不连续问题。
'''
@generic_io_descriptor(
    units="m/s^2", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    """在资产的根框架上的重力投影。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b
'''
数学原理
    projected_gravity_b 的计算（articulation_data.py:1192-1197）：
        projected_gravity_b = quat_apply_inverse(root_quat_w, GRAVITY_VEC_W)
    把世界坐标系中的重力向量 [0, 0, -9.81]，反向旋转到机体坐标系中：
        世界系: 重力始终指向 [0, 0, -1]（正下方，归一化后）

        机器人直立:
        projected_gravity = [0, 0, -1]    → "我感觉重力在我脚下" ✅

        机器人前倾 30°:
        projected_gravity = [0.5, 0, -0.87]  → "我感觉重力偏前了" ⚠️

        机器人完全倒立:
        projected_gravity = [0, 0, 1]     → "我感觉重力在头顶" ❌（要摔了！）

        机器人侧倾 45°:
        projected_gravity = [0, 0.7, -0.7]  → "我感觉重力偏侧面了" ⚠️
使用示例
    @configclass
    class ObservationsCfg:
        """行走任务的标准观测四件套"""

        base_height     = ObservationTermCfg(func=observations.base_pos_z)      # 1维
        base_lin_vel    = ObservationTermCfg(func=observations.base_lin_vel)    # 3维
        base_ang_vel    = ObservationTermCfg(func=observations.base_ang_vel)    # 3维
        gravity         = ObservationTermCfg(func=observations.projected_gravity) # 3维

    # 总计 10 维基础观测 → 策略的核心"本体感知"
'''


'''
返回机器人在环境本地坐标系（而非绝对世界坐标系）中的位置
'''
@generic_io_descriptor(
    units="m", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    """资产根位置在环境框架中"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins
'''
使用示例
    @configclass
    class ObservationsCfg:
        """用于导航任务的观测 — 需要知道当前位置"""

        robot_position = ObservationTermCfg(
            func=observations.root_pos_w,
        )

    # 对于到达目标位置的任务，策略对比:
    #   error = target_pos - robot_position  → 判断是否到达
'''


'''
返回机器人的朝向，以四元数 [w, x, y, z] 表示。告诉策略"我面朝哪个方向"。
    make_quat_unique： bool，默认 False。True 时强制四元数的实部 w ≥ 0
'''
@generic_io_descriptor(
    units="unit", axes=["W", "X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_quat_w(
    env: ManagerBasedEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    """环境框架中的资产根导向 (w，x，y，z)。

    If :attr:`make_quat_unique`是True，然后返回的四元数通过确保
    四元数具有非负的真实组成部分。
    因为``q``和``-q``都代表着相同的方向。
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    # make the quaternion real-part positive if configured
    return math_utils.quat_unique(quat) if make_quat_unique else quat
'''
核心概念：四元数的双覆盖问题
        q  = [ 0.707,  0.707,  0,  0]  → 绕 X 轴旋转 90°
        -q = [-0.707, -0.707,  0,  0]  → 绕 X 轴旋转 90°（同一个朝向！）
    q 和 -q 代表完全相同的空间朝向，但作为观测向量，两者天差地别。
    如果 PhysX 在连续两帧中返回了 q 然后 -q，策略会看到观测从 [0.7, 0.7, 0, 0] 跳到 [-0.7, -0.7, 0, 0]——这是欧几里得距离 2.0 的巨大跳变，但实际上机器人的朝向根本没变！

    make_quat_unique=True 解决这个问题：
        quat_unique 检查实部 w，如果 w < 0，就把整个四元数取反（-q）。这样就保证了 w ≥ 0，消除了符号跳变。
为什么q 和 -q 表示完全相同的空间旋转？
    因为四元数旋转一个向量时，用的是这个形式：
        v' = q * v * q^-1
使用示例
    @configclass
    class ObservationsCfg:
        """朝向观测 — 不唯一化（默认）"""

        orientation = ObservationTermCfg(
            func=observations.root_quat_w,
            # make_quat_unique=False（默认）→ 可能跳变，但更快
        )

        # 或
        orientation_unique = ObservationTermCfg(
            func=observations.root_quat_w,
            params={"make_quat_unique": True},  # 消除符号跳变
        )
'''


'''
返回机器人在世界坐标系中的线速度
'''
@generic_io_descriptor(
    units="m/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    """在环境框架中的资产根线性速度。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w
'''
使用示例
    @configclass
    class ObservationsCfg:
        # 行走任务用机体坐标系（配合速度命令）
        body_vel = ObservationTermCfg(func=observations.base_lin_vel)

        # 导航任务用世界坐标系（配合全局目标位置）
        world_vel = ObservationTermCfg(func=observations.root_lin_vel_w)
'''


'''
返回机器人在世界坐标系中的角速度
'''
@generic_io_descriptor(
    units="rad/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the environment frame."""
    """在环境框架中的资产根角速度。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w
'''
使用示例
    @configclass
    class ObservationsCfg:
        world_ang_vel = ObservationTermCfg(func=observations.root_ang_vel_w)
'''


"""
Body state
"""
"""身体状态
"""

'''
返回关节体中每个身体（连杆）的位姿，并全部展平成一维。
    和之前只返回根状态不同——根是"机器人整体在哪"，body 是"机器人的各个部件（腿、手臂、头）分别在哪"。
输出形状
    [N, 7 × num_bodies]
    例如有 4 个身体 (body_ids=[0,3,5,7]) → [N, 28]

    展平前: [N, 4, 7]    每个身体 [x, y, z, qw, qx, qy, qz]
    展平后: [N, 28]       [x0,y0,z0,qw0,qx0,qy0,qz0, x1,y1,z1,...]
'''
@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def body_pose_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The flattened body poses of the asset w.r.t the env.scene.origin.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The poses of bodies in articulation [num_env, 7 * num_bodies]. Pose order is [x,y,z,qw,qx,qy,qz].
        Output is stacked horizontally per body.
    """
    """的体格设置的资产w.r.tXenv.scene.origin。

    Note: 只有在:attr:`asset_cfg.body_ids`中配置的尸体才能恢复姿势。

    参数：
        env: 环境。
        asset_cfg: 在SceneEntity这种观测与相关。

    返回：
        关节体的姿势 [num_env， 7 * num_bodies]。
        位置顺序是 [x，y，z，qw，qx，qy，qz]。
        每个机体的输出水平堆叠。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    pose = asset.data.body_pose_w[:, asset_cfg.body_ids, :7]
    if isinstance(asset_cfg.body_ids, (slice, int)):
        pose = pose.clone()  # if slice or int, make a copy to avoid modifying original data
    pose[..., :3] = pose[..., :3] - env.scene.env_origins.unsqueeze(1)
    return pose.reshape(env.num_envs, -1)
'''
使用示例
    @configclass
    class ObservationsCfg:
        """观测所有身体的位姿 — 用于全身控制任务"""

        body_poses = ObservationTermCfg(
            func=observations.body_pose_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_ids=[0, 3, 6],    # 只观测 3 个身体: 躯干 + 左脚 + 右脚
                ),
            },
        )
    # 输出: [N, 21]  = 3 个身体 × 7 维

        # 或：观测全体
        all_body_poses = ObservationTermCfg(
            func=observations.body_pose_w,
        )
    # 输出: [N, 7×num_bodies]  = 全部连杆的位姿
'''


'''
projected_gravity（我们之前讲的"整体重力方向"）的身体级版本——为每个身体（连杆）独立计算重力方向。
    根级别只告诉策略"机器人整体歪没歪"，身体级别告诉策略"每条腿、每只手分别往哪歪"。
'''
@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def body_projected_gravity_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The direction of gravity projected on to bodies of an Articulation.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The Articulation associated with this observation.

    Returns:
        The unit vector direction of gravity projected onto body_name's frame. Gravity projection vector order is
        [x,y,z]. Output is stacked horizontally per body.
    """
    """引力向关节体投射。

    Note: 只有在:attr:`asset_cfg.body_ids`中配置的尸体才能恢复姿势。

    参数：
        env: 环境。
        asset_cfg: 关节与这一观测有关。

    返回：
        引力的单位向量方向投射到body_name的框架上。
        引力投射向量顺序是 [x，y，z]。
        每个机体的输出水平堆叠。
    """
    '''
    与 projected_gravity 的对比
        # projected_gravity（根级别）:
        asset.data.projected_gravity_b                    → [N, 3]
        # 对根身体做一次四元数逆旋转 → "整个机器人觉得重力在哪"

        # body_projected_gravity_b（身体级别）:
        body_quat = asset.data.body_quat_w[:, body_ids]    → [N, B, 4]  B 个身体的朝向
        quat_apply_inverse(body_quat, GRAVITY_VEC_W)       → [N, B, 3]  每个身体的重力投影
        .view(N, -1)                                        → [N, 3×B]   展平
        核心差异：
            根级别的 projected_gravity_b 是 ArticulationData 的属性（预计算好的），身体级别需要自己手动算——取每个身体的四元数，逐个做逆旋转。
    '''
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    gravity_dir = asset.data.GRAVITY_VEC_W.unsqueeze(1)
    return math_utils.quat_apply_inverse(body_quat, gravity_dir).view(env.num_envs, -1)
'''
① body_quat [N, B, 4]：
    每个身体在世界坐标系中的朝向四元数
② gravity_dir [N, 1, 3]：
    世界坐标系重力方向 [0, 0, -9.81]，unsqueeze(1) 为广播扩充维度
③ quat_apply_inverse(q, v)：
    把世界向量 v 用四元数 q 的逆旋转 → "这个向量在身体坐标系中看起来向哪"。结果 [N, B, 3] 展平为 [N, 3*B]

使用示例
    @configclass
    class ObservationsCfg:
        """全身重力感知 — 每条腿的倾斜程度"""

        body_gravity = ObservationTermCfg(
            func=observations.body_projected_gravity_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_ids=[3, 4, 5, 6],   # 只有四条腿
                ),
            },
        )
    # 输出: [N, 12]  = 4 条腿 × 3 维
'''


"""
Joint state.
"""
"""关节状态
"""


'''
返回关节体中指定关节的当前角度。
    输出： torch.Tensor，形状 [N, J]（J = 选中的关节数），单位 rad
'''
@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad"
)
def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    """资产的共同位置。

    Note: 只有在:attr:`asset_cfg.joint_ids`中配置的关节才能返回位置。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]
'''
使用示例
    @configclass
    class ObservationsCfg:
        """标准关节角度观测"""

        joint_positions = ObservationTermCfg(
            func=observations.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_ids=[0, 1, 2, 3, 4, 5],   # 只观测 6 个腿部关节
                ),
            },
        )
    # 输出: [N, 6]  = 6 个关节的当前角度 (rad)

        # 或：全部关节
        all_joints = ObservationTermCfg(
            func=observations.joint_pos,
        )
    # 输出: [N, num_joints]
'''


'''
joint_pos 的相对版本——不是返回绝对关节角度，而是返回相对于默认姿态的偏移。
    策略看到的是"我偏离了标准站姿多少"，而不是"我现在的绝对角度是多少"。
'''
@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_pos_offsets],
    units="rad",
)
def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    """资产w.r.t的共同位置
    默认的关节位置。

    Note: 只有在:attr:`asset_cfg.joint_ids`中配置的关节才能返回位置。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
'''
使用示例
    @configclass
    class ObservationsCfg:
        """相对关节角度 — 零中心观测"""

        joint_offsets = ObservationTermCfg(
            func=observations.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_ids=[0,1,2,3]),
            },
        )
    # 默认站姿 → 输出 [0, 0, 0, 0]
    # 微蹲     → 输出 [0, 0.1, 0, -0.05]  ← 每条腿各自偏离默认多少
'''


'''
把关节角度归一化到 [-1, 1] 范围。
    -1 = 关节在下限，0 = 关节在中间，1 = 关节在上限。
    这是强化学习中处理关节角度最常用的观测形式——神经网络处理 [-1, 1] 的归一化值远比处理 [-2.0, 3.5] 的原始弧度值稳定。
'''
@generic_io_descriptor(observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape])
def joint_pos_limit_normalized(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    """资产的共同地位与资产的共同限制正常化。

    Note: 只有在:attr:`asset_cfg.joint_ids`中配置的关节才能恢复正常位置。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1],
    )
'''
数学公式
    scale_transform(x, lower, upper) = 2 × (x - lower) / (upper - lower) - 1
    膝关节示例: lower = -2.0, upper = 2.0
        x = -2.0  →  normalized = -1.0    (完全伸直)
        x =  0.0  →  normalized =  0.0    (中间位置)
        x =  2.0  →  normalized =  1.0    (完全弯曲)
        x =  0.6  →  normalized =  0.3    (微弯，默认站姿)
为什么用 soft_joint_pos_limits 而非 joint_pos_limits？
    soft limits = 物理 limits 的 90% 范围（保留了 10% 的安全余量）。
    if 关节实际到达了物理极限但不在 soft 范围内，归一化值可能略微超出 [-1, 1]——但这也给策略提供了"我已经快到极限了"的信号。
使用示例
    @configclass
    class ObservationsCfg:
        normalized_joints = ObservationTermCfg(
            func=observations.joint_pos_limit_normalized,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_ids=[0,1,2,3]),
            },
        )
    # 输出: [N, 4] 均为 [-1, 1] 范围
'''


'''
返回关节角速度
'''
@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad/s"
)
def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    """资产的关节速度。

    Note: 只有在:attr:`asset_cfg.joint_ids`中配置的关节才能恢复速度。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]
'''
使用示例
    @configclass
    class ObservationsCfg:
        """标准本体感知：位置 + 速度"""

        joint_positions = ObservationTermCfg(func=observations.joint_pos)
        joint_velocities = ObservationTermCfg(func=observations.joint_vel)
    # 输出: [N, 2J] — 每个关节的位置 + 速度
'''


'''
返回关节角速度相对于默认角速度的偏移
    由于 default_joint_vel 通常是全零，joint_vel_rel 和 joint_vel 在数值上相同。
    但语义上，_rel 版本强调了"这是相对于默认状态的偏移"——提供了零中心的观测，即使将来改了默认速度也不会影响代码行为。
'''
@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_vel_offsets],
    units="rad/s",
)
def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    """资产 w.r.t的关节速度。
    默认的关节速度。

    Note: 只有在:attr:`asset_cfg.joint_ids`中配置的关节才能恢复速度。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


'''
回每个关节实际承受的力矩（扭矩）
'''
@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="N.m"
)
def joint_effort(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint applied effort of the robot.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their effort returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The joint effort (N or N-m) for joint_names in asset_cfg, shape is [num_env,num_joints].
    """
    """机器人的共同应用。

    NOTE: 只有在:attr:`asset_cfg.joint_ids`中配置的关节才能回复他们的努力。

    参数：
        env: 环境。
        asset_cfg: 在SceneEntity这种观测与相关。

    返回：
        在 asset_cfg，形状中的joint_names的联合努力 (N或N-m) 是 [num_env，num_joints]。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]    # asset.data.applied_torque 是执行器模型裁剪后的实际力矩
'''
使用示例
    @configclass
    class ObservationsCfg:
        joint_efforts = ObservationTermCfg(func=observations.joint_effort)

    # 常用于力矩正则化 — 在奖励函数中惩罚过大扭矩:
    #   torque_penalty = -sum(abs(joint_effort))
    #   → 鼓励策略使用节能的步态
'''


"""
Sensors.
"""
"""传感器。
"""


'''
通过 RayCaster（射线投射器） 测量地面高度。
    RayCaster 从机器人向下发射射线，测量每条射线到地面的距离，从而构建一个"地形高度图"。
'''
def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    """从给定的传感器w.r.t的高度扫描。
    传感器的框架。

    在返回的值中，所提供的抵消 (默认到0.5) 将减去。
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
'''
= 传感器高度   - 射线命中点Z  - 传感器安装偏移
ray_hits_w[..., 2] 是所有射线命中点的世界 Z 坐标。传感器安装高度 - 命中点 Z - offset → 机器人脚下地面的相对高度。

使用示例
    @configclass
    class ObservationsCfg:
        """地面高度感知 — 用于越野行走"""

        terrain_scan = ObservationTermCfg(
            func=observations.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.5,
            },
        )
'''


'''
返回每个身体在关节处承受的力（3D）和力矩（3D）——总共 6 维 per body。
    这是"关节反作用力"——当机器人的腿蹬地时，膝关节会把力传递给大腿，大腿再传给躯干。
    这个观测让策略"感受"到这些传递链中的力。
'''
def body_incoming_wrench(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """
    """在仿真世界框架中的关节物体上的空间钥匙。

    这是一个6D关 (力和扭矩) 通过接入的联合力对车身链接进行应用。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # obtain the link incoming forces in world frame
    body_incoming_joint_wrench_b = asset.data.body_incoming_joint_wrench_b[:, asset_cfg.body_ids]
    return body_incoming_joint_wrench_b.view(env.num_envs, -1)
'''
使用示例
    @configclass
    class ObservationsCfg:
        """感知腿部的受力——用于接触检测和力控"""

        foot_forces = ObservationTermCfg(
            func=observations.body_incoming_wrench,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_ids=[5, 6],        # 只观测两只脚
                ),
            },
        )
    # 输出: [N, 12]  = 2 只脚 × 6 维
'''

'''
IMU 传感器四件套
    imu_orientation       → [N, 4]  四元数朝向
    imu_projected_gravity → [N, 3]  重力方向
    imu_ang_vel           → [N, 3]  角速度
    imu_lin_acc           → [N, 3]  线加速度
'''

'''
读取 IMU 传感器（惯性测量单元）的朝向四元数。
    和 root_quat_w（读关节体的根朝向）不同——这是从独立的 IMU 传感器对象中读取的数据，模拟了真实机器人上物理 IMU 的输出。
'''
def imu_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor orientation in the simulation world frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        Orientation in the world frame in (w, x, y, z) quaternion form. Shape is (num_envs, 4).
    """
    """在仿真世界框架中的Imu传感器导向。

    参数：
        env: 环境。
        asset_cfg: SceneEntity与IMU传感器相关。
                   在 SceneEntityCfg (("imu") 中默认存在。

    返回：
        在世界框架中的导向 (w，x，y，z) 四角形。
        形状是 (num_envs， 4)。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the orientation quaternion
    return asset.data.quat_w
'''
与 root_quat_w 的关键区别
                        root_quat_w	                imu_orientation
    数据源	            Articulation 根状态	            Imu 传感器对象
    默认资产名	        "robot"	                        "imu"
    物理意义	        机器人基座的朝向	                IMU 芯片的朝向
    真实机器人能读到吗	  不一定（取决于是否有根状态传感器）	是（IMU 是标配硬件）

    IMU 可能和机器人基座有安装偏移——在实际部署中可以模拟这个偏差，让策略对传感器安装误差鲁棒。
通俗类比：
    root_quat_w 像"上帝视角知道的朝向"（仿真内部的真值），imu_orientation 像"手机里陀螺仪读到的朝向"（传感器数据）。
    对于 Sim-to-Real 迁移，应该训练策略依赖 IMU 传感器数据（真实机器人有 IMU），而不是依赖仿真内部状态（真实机器人没有"上帝视角"）。
使用示例
    @configclass
    class ObservationsCfg:
        imu_quat = ObservationTermCfg(func=observations.imu_orientation)
'''


'''
projected_gravity 的 IMU 传感器版本——从 IMU 读取重力方向，而非从关节体根状态读取。
'''
def imu_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor orientation w.r.t the env.scene.origin.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an Imu sensor.

    Returns:
        Gravity projected on imu_frame, shape of torch.tensor is (num_env,3).
    """
    """传感器导向w.r.t env.scene.origin。

    参数：
        env: 环境。
        asset_cfg: SceneEntity与Imu传感器相关。

    返回：
        在imu_frame上投射的重力，torch.tensor的形状是 (num_env，3)。
    """

    asset: Imu = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b
'''
通俗类比：
    projected_gravity 是仿真告诉你的"上帝视角的重力方向"，imu_projected_gravity 是 IMU 芯片实际测量的重力方向。
    两者数学上应该相同，但在 Sim-to-Real 场景中，真实机器人只有 IMU 数据可用。用 IMU 版本训练的策略部署时不需要适配——因为它在仿真中就已经习惯读 IMU 数据了。
'''


'''
base_ang_vel 的 IMU 传感器版本——从 IMU 读取角速度。
'''
def imu_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor angular velocity w.r.t. environment origin expressed in the sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The angular velocity (rad/s) in the sensor frame. Shape is (num_envs, 3).
    """
    """图像传感器角速度w.r.t。
    在传感器框架中表达的环境来源。

    参数：
        env: 环境。
        asset_cfg: SceneEntity与IMU传感器相关。
                   在 SceneEntityCfg (("imu") 中默认存在。

    返回：
        传感器框架中的角速度 (rad/s)。
        形状是 (num_envs， 3)。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the angular velocity
    return asset.data.ang_vel_b


'''
IMU 中的加速度计测量传感器坐标系下的线性加速度。
    base_lin_vel 是速度，imu_lin_acc 是加速度——两者是积分关系（加速度积分得到速度）。真实 IMU 输出加速度而非速度。策略需要学会从加速度推断速度的变化趋势。
'''
def imu_lin_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor linear acceleration w.r.t. the environment origin expressed in sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The linear acceleration (m/s^2) in the sensor frame. Shape is (num_envs, 3).
    """
    """电感器线性加速w.r.t。
    在传感器框架中表达的环境来源。

    参数：
        env: 环境。
        asset_cfg: SceneEntity与IMU传感器相关。
                   在 SceneEntityCfg (("imu") 中默认存在。

    返回：
        传感器框架中的线性加速 (m/s^2)。
        形状是 (num_envs， 3)。
    """
    asset: Imu = env.scene[asset_cfg.name]
    return asset.data.lin_acc_b


'''
从相机传感器中读取图像数据。
    这是 Isaac Lab 中视觉策略的基础——策略从图像中看世界，而不是靠手工设计的物理量（如关节角度、速度）。
    相比之前的 1D 观测，这里返回的是 4D 张量 [N, H, W, C]（批处理 × 高度 × 宽度 × 通道）。
'''
def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    """摄像头传感器的特定数据类型的图像。

    如果标志:attr:`normalize`是True，则根据图像的
    data-types:

    - "rgb":将图像缩小到 (0， 1) 并以当前图像批量的平均值减去。
    - "深度"或"distance_to_camera"或"distance_to_plane":以零取代无限值。

    参数：
        env: 摄像机的环境。
        sensor_cfg: 需要的传感器。
                    默认的SceneEntityCfg("tiled_camera")。
        data_type: 从所需的相机中抽取的数据类型。
                   默认的"rgb"。
        convert_perspective_to_orthogonal: 是否直角化视角深度图像。
                                           只有数据类型是"distance_to_camera"时才使用。
                                           默认为 False。
        normalize: 是否将图像正常化。
                   这取决于选择的数据类型。
                   默认为 True。

    返回：
        在最后一步制作的图像
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth/normals image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0
        elif "normals" in data_type:
            images = (images + 1.0) * 0.5

    return images.clone()
'''
通俗类比：
    之前所有的观测函数都像机器人身体里的"感官数字"——关节角度像数字角度计、IMU 像数字陀螺仪。
    image 函数像是给机器人装了个"眼睛"——直接把相机画面喂给策略。
    视觉策略用卷积神经网络（CNN）处理这些图像，从中提取有用信息（物体在哪、地面多高、障碍物在哪），而不是依赖手工设计的物理特征。这是从传统控制走向深度强化学习的关键一步。
使用示例
    @configclass
    class ObservationsCfg:
        camera_rgb = ObservationTermCfg(
            func=observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "rgb",
                "normalize": True,
            },
        )

        camera_depth = ObservationTermCfg(
            func=observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_camera",
                "normalize": True,
            },
        )
    # RGB 输出: [N, H, W, 3]
    # Depth 输出: [N, H, W, 1]
'''


class image_features(ManagerTermBase):
    """Extracted image features from a pre-trained frozen encoder.

    This term uses models from the model zoo in PyTorch and extracts features from the images.

    It calls the :func:`image` function to get the images and then processes them using the model zoo.

    A user can provide their own model zoo configuration to use different models for feature extraction.
    The model zoo configuration should be a dictionary that maps different model names to a dictionary
    that defines the model, preprocess and inference functions. The dictionary should have the following
    entries:

    - "model": A callable that returns the model when invoked without arguments.
    - "reset": A callable that resets the model. This is useful when the model has a state that needs to be reset.
    - "inference": A callable that, when given the model and the images, returns the extracted features.

    If the model zoo configuration is not provided, the default model zoo configurations are used. The default
    model zoo configurations include the models from Theia :cite:`shang2024theia` and ResNet :cite:`he2016deep`.
    These models are loaded from `Hugging-Face transformers <https://huggingface.co/docs/transformers/index>`_ and
    `PyTorch torchvision <https://pytorch.org/vision/stable/models.html>`_ respectively.

    Args:
        sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The sensor data type. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        model_zoo_cfg: A user-defined dictionary that maps different model names to their respective configurations.
            Defaults to None. If None, the default model zoo configurations are used.
        model_name: The name of the model to use for inference. Defaults to "resnet18".
        model_device: The device to store and infer the model on. This is useful when offloading the computation
            from the environment simulation device. Defaults to the environment device.
        inference_kwargs: Additional keyword arguments to pass to the inference function. Defaults to None,
            which means no additional arguments are passed.

    Returns:
        The extracted features tensor. Shape is (num_envs, feature_dim).

    Raises:
        ValueError: When the model name is not found in the provided model zoo configuration.
        ValueError: When the model name is not found in the default model zoo configuration.
    """
    """从预训练的冷编码器中提取了图像的特征。

    这个项使用PyTorch中的模型动物园模型，并从图像中提取特征。

    它调用:func:`image`函数来获取图像，

    用户可以提供自己的模型动物园配置，以使用不同的模型进行特征提取。
    模型动物园配置应是一个字典，将不同的模型名称映射到定义模型，预处理和推断函数的字典。
    字典应该包含以下内容:
    entries:

    - "模型":无论证地调用时返回模型的可调用器。
    - "重置":一个调用式，重置模型。 这在模型需要重置的状态时是有用的。
    - "推理":一个调用器，在给模型和图像时，返回取出的特征。

    如果没有提供模型动物园配置，则使用默认模型动物园配置。
    默认模型动物园配置包括来自Theia的模型:cite:`shang2024theia`和ResNet:cite:`he2016deep`。
    这些模型由`Hugging-Face transformers <https://huggingface.co/docs/transformers/index>`并且`PyTorch
    torchvision <https://pytorch.org/vision/stable/models.html>`它们是

    参数：
        sensor_cfg: 传感器配置到投票。
                    默认的SceneEntityCfg("tiled_camera")。
        data_type: 传感器数据类型。
                   默认的"rgb"。
        convert_perspective_to_orthogonal: 是否直角化视角深度图像。
                                           只有数据类型是"distance_to_camera"时才使用。
                                           默认为 False。
        model_zoo_cfg: 用户定义的字典，将不同的模型名称映射到各自的配置。
                       默认为 None。
                       如果 None，则使用默认模型动物园配置。
        model_name: 用于推断的模型名称。
                    在"resnet18"上默认设置。
        model_device: 存储和推断模型的设备。
                      这在卸载计算时有用。
            from the environment simulation device. Defaults to the environment device.
        inference_kwargs: 其他关键词参数将转移到推理函数。
                          在 None 中，默认情况下，这意味着没有其他参数被通过。

    返回：
        提取的特征是子。
        形状是 (num_envs，feature_dim)。

    异常：
        ValueError: 当模型名称不在提供的模型动物园配置中时。
        ValueError: 当模型名称不在默认模型动物园配置中时。
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # extract parameters from the configuration
        self.model_zoo_cfg: dict = cfg.params.get("model_zoo_cfg")  # type: ignore
        self.model_name: str = cfg.params.get("model_name", "resnet18")  # type: ignore
        self.model_device: str = cfg.params.get("model_device", env.device)  # type: ignore

        # List of Theia models - These are configured through `_prepare_theia_transformer_model` function
        default_theia_models = [
            "theia-tiny-patch16-224-cddsv",
            "theia-tiny-patch16-224-cdiv",
            "theia-small-patch16-224-cdiv",
            "theia-base-patch16-224-cdiv",
            "theia-small-patch16-224-cddsv",
            "theia-base-patch16-224-cddsv",
        ]
        # List of ResNet models - These are configured through `_prepare_resnet_model` function
        default_resnet_models = ["resnet18", "resnet34", "resnet50", "resnet101"]

        # Check if model name is specified in the model zoo configuration
        if self.model_zoo_cfg is not None and self.model_name not in self.model_zoo_cfg:
            raise ValueError(
                f"Model name '{self.model_name}' not found in the provided model zoo configuration."
                " Please add the model to the model zoo configuration or use a different model name."
                f" Available models in the provided list: {list(self.model_zoo_cfg.keys())}."
                "\nHint: If you want to use a default model, consider using one of the following models:"
                f" {default_theia_models + default_resnet_models}. In this case, you can remove the"
                " 'model_zoo_cfg' parameter from the observation term configuration."
            )
        if self.model_zoo_cfg is None:
            if self.model_name in default_theia_models:
                model_config = self._prepare_theia_transformer_model(self.model_name, self.model_device)
            elif self.model_name in default_resnet_models:
                model_config = self._prepare_resnet_model(self.model_name, self.model_device)
            else:
                raise ValueError(
                    f"Model name '{self.model_name}' not found in the default model zoo configuration."
                    f" Available models: {default_theia_models + default_resnet_models}."
                )
        else:
            model_config = self.model_zoo_cfg[self.model_name]

        # Retrieve the model, preprocess and inference functions
        self._model = model_config["model"]()
        self._reset_fn = model_config.get("reset")
        self._inference_fn = model_config["inference"]

    def reset(self, env_ids: torch.Tensor | None = None):
        # reset the model if a reset function is provided
        # this might be useful when the model has a state that needs to be reset
        # for example: video transformers
        if self._reset_fn is not None:
            self._reset_fn(self._model, env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "resnet18",
        model_device: str | None = None,
        inference_kwargs: dict | None = None,
    ) -> torch.Tensor:
        # obtain the images from the sensor
        image_data = image(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=False,  # we pre-process based on model
        )
        # store the device of the image
        image_device = image_data.device
        # forward the images through the model
        features = self._inference_fn(self._model, image_data, **(inference_kwargs or {}))

        # move the features back to the image device
        return features.detach().to(image_device)

    """
    Helper functions.
    """
    """辅助函数。
    """

    def _prepare_theia_transformer_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the Theia transformer model for inference.

        Args:
            model_name: The name of the Theia transformer model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        """准备Theia变压器模型进行推断。

        参数：
            model_name: 修亚变压器模型的名称。
            model_device: 存储和推断模型的设备。

        返回：
            包含模型和推断函数的字典。
        """
        from transformers import AutoModel

        def _load_model() -> torch.nn.Module:
            """Load the Theia transformer model."""
            """装载了Theia变压器模型。"""
            model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the Theia transformer model.

            Args:
                model: The Theia transformer model.
                images: The preprocessed image tensor. Shape is (num_envs, height, width, channel).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            """引入了Theia变压器模型。

            参数：
                model: 这种变压器模型。
                images: 预处理的图像子。
                        形状是 (num_envs，高度，宽度，道)。

            返回：
                提取的特征是子。
                形状是 (num_envs，feature_dim)。
            """
            # Move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # Normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # Taken from Transformers; inference converted to be GPU only
            features = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
            return features.last_hidden_state[:, 1:]

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}

    def _prepare_resnet_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the ResNet model for inference.

        Args:
            model_name: The name of the ResNet model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        """准备ResNet模型进行推断。

        参数：
            model_name: 准备ResNet模型的名称。
            model_device: 存储和推断模型的设备。

        返回：
            包含模型和推断函数的字典。
        """
        from torchvision import models

        def _load_model() -> torch.nn.Module:
            """Load the ResNet model."""
            """装载ResNet模型。"""
            # map the model name to the weights
            resnet_weights = {
                "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
                "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
                "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
                "resnet101": "ResNet101_Weights.IMAGENET1K_V1",
            }

            # load the model
            model = getattr(models, model_name)(weights=resnet_weights[model_name]).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the ResNet model.

            Args:
                model: The ResNet model.
                images: The preprocessed image tensor. Shape is (num_envs, channel, height, width).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            """引进ResNet模型。

            参数：
                model: 这就是ResNet模型。
                images: 预处理的图像子。
                        形状是 (num_envs，道，高度，宽度)。

            返回：
                提取的特征是子。
                形状是 (num_envs，feature_dim)。
            """
            # move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # forward the image through the model
            return model(image_proc)

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}


"""
Actions.
"""
"""动作。
"""


'''
返回上一帧策略输出的动作。
    这是唯一一个不从仿真物理状态或传感器读取数据的观测——它读的是策略自己上一次的输出。
    给策略提供"我刚才做了什么"的记忆。
'''
@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    """最后一次向环境输入动作。

    要求采取动作的动作项名称。
    如果 None，则返回整个动作张量。
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions
'''
为什么需要这个？
    策略网络通常是无状态的（前馈网络）——每一帧独立做决策，不看历史。如果加上 last_action，策略可以看到"上一帧我给了什么命令"，从而：
        产生平滑连续的动作（不跳变）
        隐式地感知动态（从"上一帧动作 + 当前观测"推断状态变化）
        近似一阶低通滤波器效果（新动作 = 基于旧动作微调）
两种返回模式
        # 整个动作向量:
        last_action() → [N, action_dim]  所有动作项拼接

        # 特定动作项:
        last_action("joint_pos") → [N, J]  只有关节位置动作的原始输出
    raw_actions 是策略的原始输出（未经 process_actions 的 scale+offset 变换），和策略直接对接。
使用示例
    @configclass
    class ObservationsCfg:
        """策略可以用上一帧的动作辅助决策"""

        prev_action = ObservationTermCfg(
            func=observations.last_action,
        )
    # 输出: [N, action_dim]  上一帧策略输出
'''

"""
Commands.
"""
"""命令。
"""


'''
把 CommandManager 生成的当前命令喂给策略——让策略知道"我现在该做什么"。
'''
@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def generated_commands(env: ManagerBasedRLEnv, command_name: str | None = None) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    """在指令管理器中从指令项中生成的命令。"""
    return env.command_manager.get_command(command_name)
'''
使用示例
    @configclass
    class ObservationsCfg:
        velocity_command = ObservationTermCfg(
            func=observations.generated_commands,
            params={"command_name": "base_velocity"},
        )
    # 输出: [N, 3]  [vx_des, vy_des, ωz_des]
'''

"""
Time.
"""
"""时间。
"""

'''
返回当前 episode 已经运行了多久（秒）。
'''
def current_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The current time in the episode (in seconds)."""
    """回合中的当前时间 (秒钟)。"""
    return env.episode_length_buf.unsqueeze(1) * env.step_dt


'''
current_time_s 的互补版本——返回 episode 还剩多少秒
'''
def remaining_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The maximum time remaining in the episode (in seconds)."""
    """回合剩余的最大时间 (秒钟)。"""
    return env.max_episode_length_s - env.episode_length_buf.unsqueeze(1) * env.step_dt
