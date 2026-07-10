# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""
"""共同的函数，可用于实现不同的事件。

事件包括任何与仿真状态的改变相关的东西。
这包括改变物理材料，应用外部力量，并重置资产状态。

函数可以传递到:class:`isaaclab.managers.EventTermCfg`对象，以实现函数引入的事件。
"""

'''
在训练过程中随机改变环境的各种属性，训练出对物理/视觉变化鲁棒的策略，最终实现 Sim-to-Real 迁移。
全部事件类型一览
    ┌── 静态物理属性 ────────────────────────────────────────┐
    │ randomize_rigid_body_scale           → USD 几何尺寸缩放    │
    │ randomize_rigid_body_material        → 表面物理材质       │
    │ randomize_rigid_body_mass            → 质量 + 惯性张量    │
    │ randomize_rigid_body_com             → 质心偏移          │
    │ randomize_rigid_body_collider_offsets → 碰撞检测参数      │
    │ randomize_physics_scene_gravity       → 全局重力          │
    │ randomize_actuator_gains              → PD Kp/Kd          │
    │ randomize_joint_parameters            → 关节底层物理      │
    │ randomize_fixed_tendon_parameters     → 肌腱物理参数      │
    │                                                          │
    ├── reset 状态 ────────────────────────────────────────────┤
    │ reset_root_state_uniform              → 位置+欧拉角+速度  │
    │ reset_root_state_with_random_orientation → 位置+SO(3)+速度│
    │ reset_root_state_from_terrain          → 地形出生点+朝向   │
    │ reset_joints_by_scale                  → 关节角度等比缩放 │
    │ reset_joints_by_offset                 → 关节角度加减偏移 │
    │ reset_nodal_state_uniform              → 形变体节点状态    │
    │ reset_scene_to_default                 → 全场恢复默认      │
    │                                                          │
    ├── 动态扰动 ──────────────────────────────────────────────┤
    │ apply_external_force_torque            → 持续外力+扭矩    │
    │ push_by_setting_velocity               → 瞬时速度冲击     │
    │                                                          │
    ├── 视觉随机化 ────────────────────────────────────────────┤
    │ randomize_visual_texture_material      → 纹理贴图         │
    │ randomize_visual_color                 → 漫反射 RGB 颜色  │
    │                                                          │
    ├── 内部辅助（不直接在 YAML 中使用） ───────────────────────│
    │ _randomize_prop_by_op                  → 通用采样+操作引擎 │
    │ _validate_scale_range                  → "scale"参数校验  │
    └──────────────────────────────────────────────────────────┘

两种实现方式：类 vs 函数
    函数型（14 个）
            # 定义：直接是一个普通函数
            def reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg):
                ...

            # 配置文件：
            EventTermCfg(func=events.reset_root_state_uniform, ...)
        特点：无状态、不需要预计算、每次调用时执行所有逻辑。

    类型（7 个，继承 ManagerTermBase）
            # 定义：一个类，有 __init__ 和 __call__
            class randomize_rigid_body_material(ManagerTermBase):
                def __init__(self, cfg, env):
                    ...  # 预计算材料桶、缓存属性

                def __call__(self, env, env_ids, ...):
                    ...  # 随机分配材质

            # 配置文件：
            EventTermCfg(func=events.randomize_rigid_body_material, ...)
        特点：有状态、需要在 __init__ 中预计算（如采样材料桶、解析关节索引），然后在 __call__ 中快速执行。

    为什么调用方式看起来一样？
        秘密在 ManagerBase._process_term_cfg_at_play（manager_base.py）：
            # 如果 func 是类 → 实例化，用实例替换类
            if inspect.isclass(term_cfg.func):
                term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)
        替换前： term_cfg.func = randomize_rigid_body_material（类本身）
        替换后： term_cfg.func = <randomize_rigid_body_material object at 0x...>（实例）

        之后 EventManager.apply() 统一用一行代码调用，不管原来是什么类型：
                # event_manager.py:384, 393
                term_cfg.func(self._env, env_ids, **term_cfg.params)
            如果 func 是函数 → 直接调用函数
            如果 func 是实例 → 调用 instance.__call__(env, env_ids, **params)
        因为类实例实现了 __call__，表现和函数一模一样。

    函数 vs 类的选择标准
        选择函数当...	                    选择类当...
        不需要保存状态	                    需要缓存预计算结果（材料桶、索引映射）
        每次调用逻辑简单	                需要区分"初始化"和"每次执行"
        不需要在 __init__ 中做昂贵操作	    __init__ 做昂贵操作，__call__ 轻量
        例子：reset_root_state_uniform	    例子：randomize_rigid_body_material
'''

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Literal

import torch

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions, get_isaac_sim_version

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# import logger
logger = logging.getLogger(__name__)


'''
函数作用
    在仿真启动之前，随机修改刚体资产的 USD 缩放属性（xformOp:scale），从而创造出尺寸各不相同的训练场景。
    这是 Domain Randomization（域随机化）的一种——通过在训练中暴露各种物体尺寸，让策略学会适应不同大小的目标，提高 Sim-to-Real 泛化能力。
重要约束：
    由于这个函数修改 USD 属性，而物理引擎只在仿真启动时解析一次这些属性，所以只能在仿真开始前使用（"usd" 事件模式）。
    仿真运行中调用会导致不可预测行为。
'''
def randomize_rigid_body_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
):
    """Randomize the scale of a rigid body asset in the USD stage.

    This function modifies the "xformOp:scale" property of all the prims corresponding to the asset.

    It takes a tuple or dictionary for the scale ranges. If it is a tuple, then the scaling along
    individual axis is performed equally. If it is a dictionary, the scaling is independent across each dimension.
    The keys of the dictionary are ``x``, ``y``, and ``z``. The values are tuples of the form ``(min, max)``.

    If the dictionary does not contain a key, the range is set to one for that axis.

    Relative child path can be used to randomize the scale of a specific child prim of the asset.
    For example, if the asset at prim path expression ``/World/envs/env_.*/Object`` has a child
    with the path ``/World/envs/env_.*/Object/mesh``, then the relative child path should be ``mesh`` or
    ``/mesh``.

    .. attention::
        Since this function modifies USD properties that are parsed by the physics engine once the simulation
        starts, the term should only be used before the simulation starts playing. This corresponds to the
        event mode named "usd". Using it at simulation time, may lead to unpredictable behaviors.

    .. note::
        When randomizing the scale of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """
    """在USD阶段，随机定制硬体资产的规模。

    这项函数修改了所有对资产的prims的"xformOp:scale"属性。

    这需要一个图普或字典来测量范围。
    如果它是元组，则沿着单个轴进行的扩展均。
    如果是字典，那么每个维度的尺度是独立的。
    字典的键是``x``，``y``和``z``。
    这些值是表格``(min， max)``的双倍。

    如果字典没有关键，则设置该轴的范围为一个。

    可以使用相对儿童路径来随机定制特定儿童prim的资产规模。
    例如，如果prim路径表达式``/World/envs/env_.*/Object``的资产有一个孩子
    with the path ``/World/envs/env_.*/Object/mesh``, then the relative child path should be ``mesh`` or
    ``/mesh``。

    .. 注意::
        由于这个函数在仿真启动后修改了物理引擎分析的USD属性，所以该项应仅在仿真开始播放之前使用。
        这与"usd"命名的事件模式相符。
        在仿真时使用它，可能导致不可预测的行为。

    .. 说明::
        在随机化个人资产规模时，请确保设置:attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics`到False。
        这确保物理解析器将单独分析个别资产属性。
    """
    '''
    输入/输出参数
        env：
            ManagerBasedEnv，环境实例，用于获取场景、环境数量、仿真状态
        env_ids：
            torch.Tensor | None，需要随机化的环境索引列表。None 表示所有环境
        scale_range：
            tuple[float, float] | dict[str, tuple[float, float]]，缩放范围：
                元组 (0.5, 1.5)：所有轴统一缩放，在 [0.5, 1.5] 之间均匀采样
                字典 {"x": (0.5, 1.0), "y": (1.0, 2.0), "z": (0.8, 1.2)}：每个轴独立采样
        asset_cfg：
            SceneEntityCfg，资产配置，通过 asset_cfg.name 从场景中提取对应的资产对象
        relative_child_path：
            str | None，默认为 None。如果指定，则只随机化资产的某个子节点而非整个资产。如 "mesh" 表示只缩放网格体，不影响碰撞体

    输出： 无返回值。副作用：直接修改 USD 文件中的 prim 属性
    
    例子：
        @configclass
        class EventCfg:
            """创造扁的、高的、宽的各种形状"""

            randomize_object_shape = EventTermCfg(
                func=events.randomize_rigid_body_scale,
                mode="prestartup",
                params={
                    "asset_cfg": SceneEntityCfg("table"),
                    "scale_range": {
                        "x": (0.8, 1.5),    # 长度：0.8~1.5 倍 → 有的桌子长，有的短
                        "y": (0.3, 0.6),    # 宽度：0.3~0.6 倍 → 有的桌子窄，有的宽
                        "z": (0.8, 1.2),    # 高度：0.8~1.2 倍 → 有的桌子矮，有的高
                    },
                },
            )
    '''
    # check if sim is running
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if isinstance(asset, Articulation):
        raise ValueError(
            "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
            " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
            " each version of the articulation and using multi-asset spawning. For more details, refer to:"
            " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
        )

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # acquire stage
    stage = get_current_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    # sample scale values
    if isinstance(scale_range, dict):
        range_list = [scale_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device="cpu")
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
    else:
        rand_samples = math_utils.sample_uniform(*scale_range, (len(env_ids), 1), device="cpu")
        rand_samples = rand_samples.repeat(1, 3)
    # convert to list for the for loop
    rand_samples = rand_samples.tolist()

    # apply the randomization to the parent if no relative child path is provided
    # this might be useful if user wants to randomize a particular mesh in the prim hierarchy
    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    # use sdf changeblock for faster processing of USD properties
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            # path to prim to randomize
            prim_path = prim_paths[env_id] + relative_child_path
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # get the attribute to randomize
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            # if the scale attribute does not exist, create it
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            # set the new scale
            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            # ensure the operation is done in the right ordering if we created the scale attribute.
            # otherwise, we assume the scale attribute is already in the right order.
            # note: by default isaac sim follows this ordering for the transform stack so any asset
            #   created through it will have the correct ordering
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])


'''
材料随机化需要一次性预采样一组"材料桶"，之后每次调用时从桶里随机分配
    这个类解决的问题
        PhysX 限制场景中最多只能有 64000 个独立物理材质。
        如果有 4096 个环境、每个资产有 10 个碰撞形状——不能为每个环境单独采样一组材质（会超过限制）。
        解决方案是：预采样 num_buckets 个材料桶（如 64 个），然后在 __call__ 时随机给每个形状分配桶 ID。
        所有环境共享同一套桶，但每个形状分配到哪个桶是随机的。
'''
class randomize_rigid_body_material(ManagerTermBase):
    """Randomize the physics materials on all geometries of the asset.

    This function creates a set of physics materials with random static friction, dynamic friction, and restitution
    values. The number of materials is specified by ``num_buckets``. The materials are generated by sampling
    uniform random values from the given ranges.

    The material properties are then assigned to the geometries of the asset. The assignment is done by
    creating a random integer tensor of shape  (num_instances, max_num_shapes) where ``num_instances``
    is the number of assets spawned and ``max_num_shapes`` is the maximum number of shapes in the asset (over
    all bodies). The integer values are used as indices to select the material properties from the
    material buckets.

    If the flag ``make_consistent`` is set to ``True``, the dynamic friction is set to be less than or equal to
    the static friction. This obeys the physics constraint on friction values. However, it may not always be
    essential for the application. Thus, the flag is set to ``False`` by default.

    .. attention::
        This function uses CPU tensors to assign the material properties. It is recommended to use this function
        only during the initialization of the environment. Otherwise, it may lead to a significant performance
        overhead.

    .. note::
        PhysX only allows 64000 unique physics materials in the scene. If the number of materials exceeds this
        limit, the simulation will crash. Due to this reason, we sample the materials only once during initialization.
        Afterwards, these materials are randomly assigned to the geometries of the asset.
    """
    """随机对所有物体的几何进行物理材料。

    这个函数创建了一个随机静态摩擦，动态摩擦和恢复值的物理材料集。
    材料数量由``num_buckets``指定。
    这些材料由从给定的范围中抽取统一的随机值来生成。

    然后将材料属性分配给资产的几何。
    通过创建一个随机整数形状张量 (num_instances，max_num_shapes)
    来进行分配，其中``num_instances``是产生的资产数量，``max_num_shapes``是资产中最大数量的形状 (在所有体上)。
    整数值作为索引用于从材料桶中选择材料属性。

    如果标志``make_consistent``设置为``True``，则动态摩擦设置为不到或等于静态摩擦。
    这符合摩擦值的物理约束。
    然而，对于申请可能并不总是必要的。
    因此，旗默认设置为``False``。

    .. 注意::
        这个函数使用CPU子分配材料属性。
        建议仅在环境初始化期间使用此功能。
        否则可能会导致显著的业绩总费。

    .. 说明::
        在场景，PhysX只允许64000个独特的物理材料。
        如果材料数量超过这个限度，仿真将崩。
        由于这个原因，我们在初始化过程中只采样材料一次。
        后，这些材料被随机分配到资产的几何。
    """
    '''
    __init__（初始化时一次）:
    ├── 获取资产对象
    ├── 解析碰撞形状数量（如果是指定身体的 Articulation）
    ├── 预采样 num_buckets 个材料桶 → self.material_buckets [B, 3]
    └── 可选：强制动摩擦 ≤ 静摩擦

    __call__（每次事件触发时）:
        ├── 对每个环境的每个形状，随机分配一个桶 ID
        │     bucket_ids = randint(0, num_buckets, [N, total_shapes])
        │     material_samples = self.material_buckets[bucket_ids]  ← 从桶里取
        ├── 读 PhysX 材料缓冲区
        │     materials = root_physx_view.get_material_properties()
        ├── 写入新材料
        │     materials[env_ids, ...] = material_samples
        └── 写回 PhysX

    例子：
        @configclass
        class EventCfg:
            """reset 时随机化物体表面摩擦力"""

            randomize_friction = EventTermCfg(
                func=events.randomize_rigid_body_material,
                mode="reset",                              # 每次 reset 时重新随机
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "static_friction_range": (0.3, 1.5),   # 静摩擦：0.3~1.5
                    "dynamic_friction_range": (0.2, 1.0),  # 动摩擦：0.2~1.0
                    "restitution_range": (0.0, 0.3),       # 弹性：0.0~0.3
                    "num_buckets": 32,                     # 32 种材料桶
                },
            )
        效果： 4096 个并行环境中，物体的静摩擦、动摩擦、弹性每次 reset 时随机改变。有的物体很滑（摩擦力 0.3），有的很粘（摩擦力 1.5）。
        策略必须学会适应不同的摩擦力——低摩擦时轻拿轻放，高摩擦时可以更大力。
    '''

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        """开始这个词。

        参数：
            cfg: 事件时间的配置。
            env: 环境情况。

        异常：
            ValueError: 如果资产不是RigidObject或 Articulation。
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # obtain number of shapes per body (needed for indexing the material properties correctly)
        # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
        #  per body. We use the physics simulation view to obtain the number of shapes per body.
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            # ensure the parsing is correct
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            # in this case, we don't need to do special indexing
            self.num_shapes_per_body = None

        # obtain parameters for sampling friction and restitution values
        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(cfg.params.get("num_buckets", 1))

        # sample material properties from the given ranges
        # note: we only sample the materials once during initialization
        #   afterwards these are randomly assigned to the geometries of the asset
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")

        # ensure dynamic friction is always less than static friction
        make_consistent = cfg.params.get("make_consistent", False)
        if make_consistent:
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # randomly assign material IDs to the geometries
        total_num_shapes = self.asset.root_physx_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        # retrieve material buffer from the physics simulation
        materials = self.asset.root_physx_view.get_material_properties()

        # update material buffer with new samples
        if self.num_shapes_per_body is not None:
            # sample material properties from the given ranges
            for body_id in self.asset_cfg.body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes x 3
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            # assign all the materials
            materials[env_ids] = material_samples[:]

        # apply to simulation
        self.asset.root_physx_view.set_material_properties(materials, env_ids)


'''
随机化资产（刚体或关节点）的质量和惯性张量。
    支持三种修改模式：加减、缩放、设为绝对值。
    支持三种概率分布：均匀、对数均匀、高斯。
    和 randomize_rigid_body_material 一样，它是类型事件项（继承 ManagerTermBase），因为质量随机化需要读取默认值作为基准。
'''
class randomize_rigid_body_mass(ManagerTermBase):
    """Randomize the mass of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random
    values from the given distribution parameters and adds, scales, or sets the values into the physics
    simulation based on the operation.

    If the :attr:`recompute_inertia` flag is set to :obj:`True`, the function recomputes the inertia tensor
    of the bodies after setting the mass. This is useful when the mass is changed significantly, as the
    inertia tensor depends on the mass. It assumes the body is a uniform density object. If the body is not
    a uniform density object, the inertia tensor may not be accurate.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    """通过添加，扩展或设置随机值来随机定位体积。

    这种函数允许随机对资产体质量进行排序。
    函数从给定的分布参数中抽取随机值，并根据操作添加，量度或设置值在物理仿真中。

    如果:attr:`recompute_inertia`标志设置为:obj:`True`，则在设置质量后，函数重新计算体体的惯性子。
    这在质量显著变化时是有用的，因为惯性子取决于质量。
    它假设身体是一个均密度的物体。
    如果身体不是一个均密度的对象，则惰性子可能不准确。

    .. 提示::
        这个函数使用CPU子分配体质量。
        建议仅在环境初始化期间使用此功能。
    """
    '''
    输入/输出参数
        __init__ 的 cfg.params
            键	                            含义
            "asset_cfg"	                    必须，资产配置
            "operation"	                    "add" / "scale" / "abs"
            "mass_distribution_params"	    分布参数 (min, max) 或 (mean, std)
            "min_mass"	                    可选，质量下限，默认 1e-6
        __call__ 的参数
            参数	                    默认值	            含义
            mass_distribution_params	—	            分布参数
            operation	                —	            "add" / "scale" / "abs"
            distribution	            "uniform"	    "uniform" / "log_uniform" / "gaussian"
            recompute_inertia	        True	        是否按比例重算惯性
            min_mass	                1e-6	        质量下限（防止 0 或负值）
        三种操作模式
            原始默认质量 m0（如 2.0 kg）
            随机采样值 r（如 0.5）

            "add":    m_new = m0 + r      → 2.0 + 0.5 = 2.5 kg   加减偏差
            "scale":  m_new = m0 × r      → 2.0 × 0.5 = 1.0 kg   等比缩放
            "abs":    m_new = r            → m_new = 0.5 kg         直接设为绝对值

    示例 1：加减偏差 — 轻微质量扰动
        @configclass
        class EventCfg:
            """每次 reset 随机加减 ±0.5kg 的质量扰动"""

            randomize_mass = EventTermCfg(
                func=events.randomize_rigid_body_mass,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "operation": "add",                    # 在原质量上加减
                    "mass_distribution_params": (-0.5, 0.5),  # 随机偏移 ±0.5kg
                    "distribution": "uniform",
                },
            )
        效果： 物体默认质量 2.0kg → 随机变成 1.5~2.5kg。
            适合做轻微的域随机化，策略不需要重新学习"抓取"的基本动作，只需要适应略有变化的重量感。
    '''

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        """开始这个词。

        参数：
            cfg: 事件时间的配置。
            env: 环境情况。

        异常：
            TypeError: 如果`params`不是两个数字的。
            ValueError: 如果操作不支持。
            ValueError: 如果下限是负值或零值，如果不允许。
            ValueError: 如果上限小于下限。
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "mass_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["mass_distribution_params"], "mass_distribution_params", allow_zero=False
                )
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_rigid_body_mass' does not support operation:"
                f" '{cfg.params['operation']}'."
            )
        if cfg.params.get("min_mass") is not None:
            if cfg.params.get("min_mass") < 1e-6:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_mass' does not support 'min_mass' less than 1e-6 to avoid"
                    " physics errors."
                )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
        min_mass: float = 1e-6,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # resolve body indices
        if self.asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        # get the current masses of the bodies (num_assets, num_bodies)
        masses = self.asset.root_physx_view.get_masses()

        # apply randomization on default values
        # this is to make sure when calling the function multiple times, the randomization is applied on the
        # default values and not the previously randomized values
        masses[env_ids[:, None], body_ids] = self.asset.data.default_mass[env_ids[:, None], body_ids].clone()

        # sample from the given range
        # note: we modify the masses in-place for all environments
        #   however, the setter takes care that only the masses of the specified environments are modified
        masses = _randomize_prop_by_op(
            masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
        )
        masses = torch.clamp(masses, min=min_mass)  # ensure masses are positive

        # set the mass into the physics simulation
        self.asset.root_physx_view.set_masses(masses, env_ids)

        # recompute inertia tensors if needed
        if recompute_inertia:
            # compute the ratios of the new masses to the initial masses
            ratios = masses[env_ids[:, None], body_ids] / self.asset.data.default_mass[env_ids[:, None], body_ids]
            # scale the inertia tensors by the the ratios
            # since mass randomization is done on default values, we can use the default inertia tensors
            inertias = self.asset.root_physx_view.get_inertias()
            if isinstance(self.asset, Articulation):
                # inertia has shape: (num_envs, num_bodies, 9) for articulation
                inertias[env_ids[:, None], body_ids] = (
                    self.asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
                )
            else:
                # inertia has shape: (num_envs, 9) for rigid object
                inertias[env_ids] = self.asset.data.default_inertia[env_ids] * ratios
            # set the inertia tensors into the physics simulation
            self.asset.root_physx_view.set_inertias(inertias, env_ids)

'''
随机偏移刚体的质心位置（Center of Mass）。
    质心是物体"重量集中点"——把质心往后挪，物体就变得"头轻脚重"；往前挪就"头重脚轻"。
    这是域随机化的另一种形式：让策略面对物理上"重心不对"的物体，提高鲁棒性。
输入/输出参数
    env：
        ManagerBasedEnv，环境实例
    env_ids：
        torch.Tensor | None，需要随机化的环境索引。None = 所有环境
    com_range：
        dict[str, tuple[float, float]]，每个轴的质量中心偏移范围：
            {"x": (-0.05, 0.05), "y": (-0.03, 0.03), "z": (-0.1, 0.1)}
            缺失的键默认为 (0.0, 0.0)——不偏移
            asset_cfg： SceneEntityCfg，资产配置。
                        ⚠️ 只支持 Articulation（关节体），不支持纯 RigidObject（代码第 648 行直接强制类型为 Articulation）

    输出： 无返回值。副作用：通过 PhysX API 写入新的质心位置
'''
def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    """通过从给定的范围中抽取的随机值添加，随机定制硬体质量中心 (CoM)。

    .. 说明::
        这个函数使用CPU子分配CoM。
        建议仅在环境初始化期间使用此功能。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[env_ids[:, None], body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)
    '''
    示例 1：轻微质心扰动
        @configclass
        class EventCfg:
            """每次 reset 轻微随机化物体质心"""

            randomize_com = EventTermCfg(
                func=events.randomize_rigid_body_com,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "com_range": {
                        "x": (-0.02, 0.02),   # X 轴 ±2cm
                        "y": (-0.02, 0.02),   # Y 轴 ±2cm
                        "z": (-0.05, 0.05),   # Z 轴 ±5cm（上下偏移更明显）
                    },
                },
            )
        效果： 物体的质量中心在原位置附近 ±2~5cm 浮动。策略学会了适应"重心不太对"的物体——这对抓取任务很有用：真实世界中物体的重心不一定在几何中心。
    '''

'''
随机化 PhysX 碰撞检测的两个关键偏移参数：rest offset 和 contact offset。
    这两个参数控制碰撞的"敏感度"——物体快碰到时，物理引擎在多大距离外就开始产生接触力。
    这是一个相对冷门但强大的域随机化手段：改变碰撞检测的精度，让策略适应不同的物理仿真参数，有利于 Sim-to-Real。
输入/输出参数
    env：
        ManagerBasedEnv
    env_ids：
        torch.Tensor | None
    asset_cfg：
        SceneEntityCfg，支持 RigidObject 和 Articulation
    rest_offset_distribution_params：
        tuple[float, float] | None，rest offset 的分布参数。None = 不改
    contact_offset_distribution_params：
        tuple[float, float] | None，contact offset 的分布参数。None = 不改
    distribution：
        "uniform" / "log_uniform" / "gaussian"，默认 "uniform"

    输出： 无返回值。副作用：写入 PhysX 碰撞参数
'''
def randomize_rigid_body_collider_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    rest_offset_distribution_params: tuple[float, float] | None = None,
    contact_offset_distribution_params: tuple[float, float] | None = None,
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the collider parameters of rigid bodies in an asset by adding, scaling, or setting random values.

    This function allows randomizing the collider parameters of the asset, such as rest and contact offsets.
    These correspond to the physics engine collider properties that affect the collision checking.

    The function samples random values from the given distribution parameters and applies the operation to
    the collider properties. It then sets the values into the physics simulation. If the distribution parameters
    are not provided for a particular property, the function does not modify the property.

    Currently, the distribution parameters are applied as absolute values.

    .. tip::
        This function uses CPU tensors to assign the collision properties. It is recommended to use this function
        only during the initialization of the environment.
    """
    """通过添加，扩展或设置随机值来随机定位在资产中固体的碰撞参数。

    这种函数允许随机化资产的碰撞器参数，例如休息和接触抵消。
    这些与对撞检查影响的物理发动机碰撞器性能相匹配。

    函数从给定的分布参数中抽取随机值，并将操作应用于碰撞器属性。
    然后它将值设置在物理仿真中。
    如果分配参数不为特定属性提供，函数不会改变属性。

    目前，分布参数被应用为绝对值。

    .. 提示::
        这个函数使用CPU子来分配碰撞属性。
        建议仅在环境初始化期间使用此功能。
    """
    '''
    物理背景：rest offset 和 contact offset
        ┌─────────┐                    ┌─────────┐
        │  物体 A  │          ← 间隙 →  │  物体 B  │
        └─────────┘                    └─────────┘
            │                              │
            │←── contact_offset ────────→│  在这个距离内，PhysX 开始生成接触点
            │←─ rest_offset ──→│              在这个距离时，接触力达到稳定值（零穿透）
            │← 穿透 →│                         小于 rest_offset = 实际穿透
        通俗理解：
            contact offset：
                碰撞"预警距离"——物体 A 和 B 的间距小于这个值时，PhysX 开始算接触力。
                大的 contact offset 意味着"还没碰到就感到阻力"（像磁铁一样的缓冲）
            rest offset：
                碰撞"稳定位置"——接触力在该距离处达到平衡（通常为负值或 0，表示轻微穿透或刚好接触）。
                改变 rest offset 相当于改变"物体的表面有多软"
        两者关系：contact_offset > rest_offset。contact offset 越大，碰撞检测越早启动，碰撞力越平滑——但计算量也更大。
    '''
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")

    # sample collider properties from the given ranges and set into the physics simulation
    # -- rest offsets
    if rest_offset_distribution_params is not None:
        rest_offset = asset.root_physx_view.get_rest_offsets().clone()
        rest_offset = _randomize_prop_by_op(
            rest_offset,
            rest_offset_distribution_params,
            None,
            slice(None),
            operation="abs",
            distribution=distribution,
        )
        asset.root_physx_view.set_rest_offsets(rest_offset, env_ids.cpu())
    # -- contact offsets
    if contact_offset_distribution_params is not None:
        contact_offset = asset.root_physx_view.get_contact_offsets().clone()
        contact_offset = _randomize_prop_by_op(
            contact_offset,
            contact_offset_distribution_params,
            None,
            slice(None),
            operation="abs",
            distribution=distribution,
        )
        asset.root_physx_view.set_contact_offsets(contact_offset, env_ids.cpu())
    '''
    通俗类比：
        前四个函数改的是物体的"物理属性"——质量、重心、摩擦、大小。
        randomize_rigid_body_collider_offsets 改的是物理引擎的"触觉灵敏度"——就像你戴厚手套 vs 薄手套触摸物体，物体本身没变，但你的感觉变了。
        厚手套（大 contact offset）让你提前感知到物体但感觉模糊，薄手套（小 contact offset）让你直到碰到才感知但感觉精确。
        训练策略时混合厚薄手套，部署时无论物理引擎用什么参数都能适应。
    '''

'''
随机化整个物理场景的重力向量。
    和之前所有随机化函数最大的区别：重力是所有环境共享的——你没法让 4096 个环境各有不同的重力方向。PhysX 只有一个全局重力参数。
    这也是一个函数型事件项（不是类），因为不需要预计算或缓存状态。
输入/输出参数
    env：
        ManagerBasedEnv
    env_ids：
        ⚠️ 参数列表中接受，但实际上没用到——因为重力是全局的，改一次所有环境都生效
    gravity_distribution_params：
        tuple[list[float], list[float]]
            格式：([min_x, min_y, min_z], [max_x, max_y, max_z])
            每个分量独立采样。比如 ([-1.0, -10.0, -10.0], [1.0, -8.0, -8.0]) — X 在 ±1.0 随机，Y 和 Z 在 -10~-8 随机
    operation：
        "add" / "scale" / "abs"
            "add"：默认重力 + 随机偏移 → 重力方向小幅抖动
            "scale"：默认重力 × 随机系数 → 重力"强弱"改变，方向不变
            "abs"：直接设为随机值 → 重力方向可能完全翻转
    distribution：
        "uniform" / "log_uniform" / "gaussian"，默认 "uniform"

    输出： 无返回值。副作用：写入全局 PhysX 重力参数
'''
def randomize_physics_scene_gravity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    gravity_distribution_params: tuple[list[float], list[float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize gravity by adding, scaling, or setting random values.

    This function allows randomizing gravity of the physics scene. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the
    operation.

    The distribution parameters are lists of two elements each, representing the lower and upper bounds of the
    distribution for the x, y, and z components of the gravity vector. The function samples random values for each
    component independently.

    .. attention::
        This function applied the same gravity for all the environments.

    .. tip::
        This function uses CPU tensors to assign gravity.
    """
    """通过添加，扩展或设置随机值来随机调整重力。

    这个函数允许随机化物理场景的重力。
    函数从给定的分布参数中抽取随机值，并根据操作添加，量度或设置值在物理仿真中。

    分布参数是两个元素的列表，每个元素代表重力向量的x，y和z组件的分布的下和上边界。
    函数独立对每个组件进行随机测量。

    .. 注意::
        这种函数对所有环境都应用相同的重力。

    .. 提示::
        这个函数使用CPU子分配重力。
    """
    # get the current gravity
    gravity = torch.tensor(env.sim.cfg.gravity, device="cpu").unsqueeze(0)
    dist_param_0 = torch.tensor(gravity_distribution_params[0], device="cpu")
    dist_param_1 = torch.tensor(gravity_distribution_params[1], device="cpu")
    gravity = _randomize_prop_by_op(
        gravity,
        (dist_param_0, dist_param_1),
        None,
        slice(None),
        operation=operation,
        distribution=distribution,
    )
    # unbatch the gravity tensor into a list
    gravity = gravity[0].tolist()

    # set the gravity into the physics simulation
    physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view
    physics_sim_view.set_gravity(carb.Float3(*gravity))
    '''
    示例 1：仿真前随机化重力强度
        @configclass
        class EventCfg:
            """仿真启动前随机重力大小，训练期间保持不变"""

            randomize_gravity = EventTermCfg(
                func=events.randomize_physics_scene_gravity,
                mode="prestartup",                    # ← 只能 prestartup 或 startup！
                params={
                    "gravity_distribution_params": (
                        [0.0, 0.0, -9.0],             # [min_x, min_y, min_z]
                        [0.0, 0.0, -11.0],            # [max_x, max_y, max_z]
                    ),
                    "operation": "abs",               # 直接设绝对值
                    "distribution": "uniform",
                },
            )
        效果： 仿真前重力 Z 分量在 -9.0~-11.0 m/s² 之间随机（地球重量 ~ 略重），XY 方向为 0。
            训练期间重力不变——策略学会在某个特定的重力环境下行走。下次重新训练时又是另一个重力。
    '''

'''
随机化关节执行器的 PD 控制器参数：刚度（stiffness, Kp） 和 阻尼（damping, Kd）。
    这是对先前学过的 PD 控制器（actuator_pd.py 中 τ = Kp×(q_des-q) + Kd×(q̇_des-q̇)）的参数直接做域随机化。
输入/输出参数
    __init__ 的 cfg.params
        键	                                含义
        "asset_cfg"	                        必须，资产配置
        "operation"	                        "add" / "scale" / "abs"
        "stiffness_distribution_params"	    可选，刚度分布参数
        "damping_distribution_params"	    可选，阻尼分布参数
    __call__ 的参数
        参数	                            默认值	        含义
        stiffness_distribution_params	    None	    不填 = 不改刚度
        damping_distribution_params	        None	    不填 = 改阻尼
        operation	                        "abs"	    默认直接设绝对值
        distribution	                    "uniform"	均匀分布
'''
class randomize_actuator_gains(ManagerTermBase):
    """Randomize the actuator gains in an articulation by adding, scaling, or setting random values.

    This function allows randomizing the actuator stiffness and damping gains.

    The function samples random values from the given distribution parameters and applies the operation to
    the joint properties. It then sets the values into the actuator models. If the distribution parameters
    are not provided for a particular property, the function does not modify the property.

    .. tip::
        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
        In such cases, it is recommended to use this function only during the initialization of the environment.
    """
    """通过添加，扩展或设置随机值来随机调整动机在关节中的收益。

    这种函数允许随机化执行器的硬度和缩增长。

    函数从给定的分布参数中抽取随机值，并将操作应用于联合属性。
    然后它将值设置在执行器模型中。
    如果分配参数不为特定属性提供，函数不会改变属性。

    .. 提示::
        对于隐含执行器，该函数使用CPU子将执行器的收益分配到仿真中。
        在这种情况下，建议仅在环境启动期间使用此功能。
    """
    '''
    为什么这个很重要？
        Sim-to-Real 的核心差距之一就是电机特性——仿真中的 Kp/Kd 和真实电机的响应特性不完全一致。
        随机化执行器增益让策略学会适应"不同力度的电机"，迁移到真实硬件时不容易因为电机特性差异而失败。
    '''

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        """开始这个词。

        参数：
            cfg: 事件时间的配置。
            env: 环境情况。

        异常：
            TypeError: 如果`params`不是两个数字的。
            ValueError: 如果操作不支持。
            ValueError: 如果下限是负值或零值，如果不允许。
            ValueError: 如果上限小于下限。
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["stiffness_distribution_params"], "stiffness_distribution_params", allow_zero=False
                )
            if "damping_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["damping_distribution_params"], "damping_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_actuator_gains' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # Resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        def randomize(data: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
            return _randomize_prop_by_op(
                data, params, dim_0_ids=None, dim_1_ids=actuator_indices, operation=operation, distribution=distribution
            )

        # Loop through actuators and randomize gains
        for actuator in self.asset.actuators.values():
            if isinstance(self.asset_cfg.joint_ids, slice):
                # we take all the joints of the actuator
                actuator_indices = slice(None)
                if isinstance(actuator.joint_indices, slice):
                    global_indices = slice(None)
                elif isinstance(actuator.joint_indices, torch.Tensor):
                    global_indices = actuator.joint_indices.to(self.asset.device)
                else:
                    raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")
            elif isinstance(actuator.joint_indices, slice):
                # we take the joints defined in the asset config
                global_indices = actuator_indices = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
            else:
                # we take the intersection of the actuator joints and the asset config joints
                actuator_joint_indices = actuator.joint_indices
                asset_joint_ids = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
                # the indices of the joints in the actuator that have to be randomized
                actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
                if len(actuator_indices) == 0:
                    continue
                # maps actuator indices that have to be randomized to global joint indices
                global_indices = actuator_joint_indices[actuator_indices]
            # Randomize stiffness
            if stiffness_distribution_params is not None:
                stiffness = actuator.stiffness[env_ids].clone()
                stiffness[:, actuator_indices] = self.asset.data.default_joint_stiffness[env_ids][
                    :, global_indices
                ].clone()
                randomize(stiffness, stiffness_distribution_params)
                actuator.stiffness[env_ids] = stiffness
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_stiffness_to_sim(
                        stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids
                    )
            # Randomize damping
            if damping_distribution_params is not None:
                damping = actuator.damping[env_ids].clone()
                damping[:, actuator_indices] = self.asset.data.default_joint_damping[env_ids][:, global_indices].clone()
                randomize(damping, damping_distribution_params)
                actuator.damping[env_ids] = damping
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)
            '''
            隐式 vs 显式执行器的差异：
                显式执行器（IdealPDActuator）——Kp/Kd 存在 Python 层，修改 actuator.stiffness 就行。
                隐式执行器（ImplicitActuator）——Kp/Kd 在 PhysX C++ 层，必须额外调用 write_joint_stiffness_to_sim 才能生效。
                如果不写这一步，隐式执行器的增益在仿真中不会改变。
            '''
    '''
    示例 1：柔和的电机 — 弱刚度随机化
        @configclass
        class EventCfg:
            """每次 reset 时电机刚度在默认值的 ±20% 范围内浮动"""

            randomize_stiffness = EventTermCfg(
                func=events.randomize_actuator_gains,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "stiffness_distribution_params": (-10.0, 10.0),  # 加减 ±10 N·m/rad
                    "operation": "add",                               # 加减偏差模式
                    "distribution": "uniform",
                },
            )
        效果： 电机刚度在默认值 ±10 范围内均匀浮动。"add" 模式适合小幅扰动——不会把一台强电机变成弱电机，只是让它的"力度感"有所差异。
    '''

'''
随机化 PhysX 关节的三个底层属性：摩擦系数、电枢（armature） 和 位置限制。
这是"随机化关节物理特性"的一站式函数——一个类覆盖了三类关节参数的随机化，每个参数独立可控（填了就改，不填不改）。
'''
class randomize_joint_parameters(ManagerTermBase):
    """Randomize the simulated joint parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the joint parameters of the asset. These correspond to the physics engine
    joint properties that affect the joint behavior. The properties include the joint friction coefficient, armature,
    and joint position limits.

    The function samples random values from the given distribution parameters and applies the operation to the
    joint properties. It then sets the values into the physics simulation. If the distribution parameters are
    not provided for a particular property, the function does not modify the property.

    .. tip::
        This function uses CPU tensors to assign the joint properties. It is recommended to use this function
        only during the initialization of the environment.
    """
    """通过添加，扩展或设置随机值来随机定制一个关节的仿真关节参数。

    这种函数允许随机化资产的联合参数。
    这些与物理引擎关节特性相符，影响关节行为。
    这些特性包括联合摩擦系数， armature和关节位置限制。

    函数从给定的分布参数中抽取随机值，并将操作应用于联合属性。
    然后它将值设置在物理仿真中。
    如果分配参数不为特定属性提供，函数不会改变属性。

    .. 提示::
        这个函数使用CPU子分配联合属性。
        建议仅在环境初始化期间使用此功能。
    """
    '''
    三个随机化目标
        参数	            物理含义	    随机化效果
        friction	        关节摩擦系数	关节转动的"阻力感"——高摩擦像生锈的轴承
        armature	        电枢惯性	    电机的"等效转动惯量"——影响加速响应
        position limits	    关节角度限制	关节能转多大角度——上下限可以随机放宽/收窄
    三种摩擦类型：
        静态摩擦 (static):  关节从静止开始转动需要克服的力
                            → 类似"启动扭矩阈值"
        动摩擦 (dynamic):   关节已经在转动时，持续的摩擦阻力
                            → 类似"滑动摩擦力"，通常 ≤ 静态摩擦
        粘性摩擦 (viscous): 与转速成正比的阻力
                            → τ_friction = c × ω，转速越快阻力越大
        torch.minimum(dynamic, friction_coeff) — 确保动摩擦 ≤ 静摩擦。 物理约束：推动一个静止的物体比保持它在运动中所需的力更大。
    Armature 是什么？
        不是"盔甲"——是电机术语，表示电枢的等效转动惯量。在 PhysX 中，armature 被加到关节的转动惯量上，使得电机加速/减速变慢。
        增加 armature 等价于"电机转子上绑了一个更大的飞轮"——需要更大扭矩才能加速。
        物理公式： τ_effective = τ_motor - armature × α，其中 α 是角加速度。armature 越大，同样的电机扭矩产生的加速度越小。
    '''
    '''
    示例 1：关节生锈了 — 高摩擦随机化
        @configclass
        class EventCfg:
            """每次 reset 随机化关节摩擦（模拟不同程度的老化/生锈）"""

            randomize_joint_friction = EventTermCfg(
                func=events.randomize_joint_parameters,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "friction_distribution_params": (0.0, 2.0),   # 0（无摩擦）~ 2.0（生锈）
                    "operation": "abs",
                    "distribution": "uniform",
                },
            )
        效果： 关节摩擦在 0~2.0 随机——有些环境关节顺滑，有些像"生锈了"一样需要克服大的阻力才能转动。策略学会了在不同的"关节顺滑度"下完成动作。
    '''

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        """开始这个词。

        参数：
            cfg: 事件时间的配置。
            env: 环境情况。

        异常：
            TypeError: 如果`params`不是两个数字的。
            ValueError: 如果操作不支持。
            ValueError: 如果下限是负值或零值，如果不允许。
            ValueError: 如果上限小于下限。
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "friction_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["friction_distribution_params"], "friction_distribution_params")
            if "armature_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["armature_distribution_params"], "armature_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_fixed_tendon_parameters' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        # resolve joint indices
        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)  # for optimization purposes
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids_for_slice = env_ids[:, None]
        else:
            env_ids_for_slice = env_ids

        # sample joint properties from the given ranges and set into the physics simulation
        # joint friction coefficient
        if friction_distribution_params is not None:
            friction_coeff = _randomize_prop_by_op(
                self.asset.data.default_joint_friction_coeff.clone(),
                friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )

            # ensure the friction coefficient is non-negative
            friction_coeff = torch.clamp(friction_coeff, min=0.0)

            # Always set static friction (indexed once)
            static_friction_coeff = friction_coeff[env_ids_for_slice, joint_ids]

            # if isaacsim version is lower than 5.0.0 we can set only the static friction coefficient
            if get_isaac_sim_version().major >= 5:
                # Randomize raw tensors
                dynamic_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_dynamic_friction_coeff.clone(),
                    friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                viscous_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_viscous_friction_coeff.clone(),
                    friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

                # Clamp to non-negative
                dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
                viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)

                # Ensure dynamic ≤ static (same shape before indexing)
                dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)

                # Index once at the end
                dynamic_friction_coeff = dynamic_friction_coeff[env_ids_for_slice, joint_ids]
                viscous_friction_coeff = viscous_friction_coeff[env_ids_for_slice, joint_ids]
            else:
                # For versions < 5.0.0, we do not set these values
                dynamic_friction_coeff = None
                viscous_friction_coeff = None

            # Single write call for all versions
            self.asset.write_joint_friction_coefficient_to_sim(
                joint_friction_coeff=static_friction_coeff,
                joint_dynamic_friction_coeff=dynamic_friction_coeff,
                joint_viscous_friction_coeff=viscous_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )

        # joint armature
        if armature_distribution_params is not None:
            armature = _randomize_prop_by_op(
                self.asset.data.default_joint_armature.clone(),
                armature_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_armature_to_sim(
                armature[env_ids_for_slice, joint_ids], joint_ids=joint_ids, env_ids=env_ids
            )

        # joint position limits
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            joint_pos_limits = self.asset.data.default_joint_pos_limits.clone()
            # -- randomize the lower limits
            if lower_limit_distribution_params is not None:
                joint_pos_limits[..., 0] = _randomize_prop_by_op(
                    joint_pos_limits[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
            # -- randomize the upper limits
            if upper_limit_distribution_params is not None:
                joint_pos_limits[..., 1] = _randomize_prop_by_op(
                    joint_pos_limits[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

            # extract the position limits for the concerned joints
            joint_pos_limits = joint_pos_limits[env_ids_for_slice, joint_ids]
            if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'randomize_joint_parameters' is setting lower joint limits that are greater"
                    " than upper joint limits. Please check the distribution parameters for the joint position limits."
                )
            # set the position limits into the physics simulation
            self.asset.write_joint_position_limit_to_sim(
                joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
            )


'''
随机化 PhysX 中**固定肌腱（Fixed Tendon）**的各项物理参数。
    肌腱是连接两个刚体的"弹性绳"/"弹簧"——在机器人仿真中用于模拟缆绳、韧带、柔性连接等。
    这个类支持六种肌腱属性的独立随机化。
'''
class randomize_fixed_tendon_parameters(ManagerTermBase):
    """Randomize the simulated fixed tendon parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the fixed tendon parameters of the asset.
    These correspond to the physics engine tendon properties that affect the joint behavior.

    The function samples random values from the given distribution parameters and applies the operation to
    the tendon properties. It then sets the values into the physics simulation. If the distribution parameters
    are not provided for a particular property, the function does not modify the property.
    """
    """通过添加，扩展或设置随机值来随机调整一个关节的仿真固定子参数。

    这种函数允许随机化资产的固定部参数。
    这些与物理引擎的特性相匹配，

    函数从给定的分布参数中抽取随机值，并将操作应用于子属性。
    然后它将值设置在物理仿真中。
    如果分配参数不为特定属性提供，函数不会改变属性。
    """
    '''
    什么是 Fixed Tendon？
        在 PhysX 中，fixed tendon 建模了两点之间的柔性连接：
        ┌─────────┐                              ┌─────────┐
        │  刚体 A  │  ←── 肌腱（弹簧/缆绳）──→  │  刚体 B  │
        └─────────┘    stiffness, damping       └─────────┘
                            rest_length
                            limits [min, max]
        肌腱可以限制两个物体的相对运动——像一根橡皮筋（弹性）连接它们。当两个物体的距离 ≠ rest_length 时，肌腱产生一个恢复力。

    六个可随机化的参数
        参数	            默认值来源	                                物理含义
        stiffness	        default_fixed_tendon_stiffness	        肌腱的"弹簧硬度"——偏离 rest_length 时产生的力
        damping	            default_fixed_tendon_damping	        肌腱伸缩的"阻力"——速度越快阻力越大
        limit_stiffness	    default_fixed_tendon_limit_stiffness	超出位置限制后的额外硬度
        position limits	    default_fixed_tendon_pos_limits	        肌腱的 [下限, 上限] 伸缩范围
        rest_length	        default_fixed_tendon_rest_length	    肌腱的"自然长度"——恢复力为零的长度
        offset	            default_fixed_tendon_offset	            初始偏移量

    '''

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        """开始这个词。

        参数：
            cfg: 事件时间的配置。
            env: 环境情况。

        异常：
            TypeError: 如果`params`不是两个数字的。
            ValueError: 如果操作不支持。
            ValueError: 如果下限是负值或零值，如果不允许。
            ValueError: 如果上限小于下限。
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["stiffness_distribution_params"], "stiffness_distribution_params", allow_zero=False
                )
            if "damping_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["damping_distribution_params"], "damping_distribution_params")
            if "limit_stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["limit_stiffness_distribution_params"], "limit_stiffness_distribution_params"
                )
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_fixed_tendon_parameters' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        limit_stiffness_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        rest_length_distribution_params: tuple[float, float] | None = None,
        offset_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        # resolve joint indices
        if self.asset_cfg.fixed_tendon_ids == slice(None):
            tendon_ids = slice(None)  # for optimization purposes
        else:
            tendon_ids = torch.tensor(self.asset_cfg.fixed_tendon_ids, dtype=torch.int, device=self.asset.device)

        # sample tendon properties from the given ranges and set into the physics simulation
        # stiffness
        if stiffness_distribution_params is not None:
            stiffness = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_stiffness.clone(),
                stiffness_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_stiffness(stiffness[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

        # damping
        if damping_distribution_params is not None:
            damping = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_damping.clone(),
                damping_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_damping(damping[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

        # limit stiffness
        if limit_stiffness_distribution_params is not None:
            limit_stiffness = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_limit_stiffness.clone(),
                limit_stiffness_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_limit_stiffness(
                limit_stiffness[env_ids[:, None], tendon_ids], tendon_ids, env_ids
            )

        # position limits
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            limit = self.asset.data.default_fixed_tendon_pos_limits.clone()
            # -- lower limit
            if lower_limit_distribution_params is not None:
                limit[..., 0] = _randomize_prop_by_op(
                    limit[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    tendon_ids,
                    operation=operation,
                    distribution=distribution,
                )
            # -- upper limit
            if upper_limit_distribution_params is not None:
                limit[..., 1] = _randomize_prop_by_op(
                    limit[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    tendon_ids,
                    operation=operation,
                    distribution=distribution,
                )

            # check if the limits are valid
            tendon_limits = limit[env_ids[:, None], tendon_ids]
            if (tendon_limits[..., 0] > tendon_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'randomize_fixed_tendon_parameters' is setting lower tendon limits that are"
                    " greater than upper tendon limits."
                )
            self.asset.set_fixed_tendon_position_limit(tendon_limits, tendon_ids, env_ids)

        # rest length
        if rest_length_distribution_params is not None:
            rest_length = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_rest_length.clone(),
                rest_length_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_rest_length(rest_length[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

        # offset
        if offset_distribution_params is not None:
            offset = _randomize_prop_by_op(
                self.asset.data.default_fixed_tendon_offset.clone(),
                offset_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_offset(offset[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

        # write the fixed tendon properties into the simulation
        self.asset.write_fixed_tendon_properties_to_sim(tendon_ids, env_ids)


'''
对刚体的各个身体施加持续性的随机外力和扭矩。
    和之前所有的"改属性"随机化不同——这是直接对机器人施加物理扰动。
    外力不会只生效一瞬间，而是通过 permanent_wrench_composer 持续作用，直到被下一次事件覆盖或 reset 清零。

输入/输出参数
    env：
        ManagerBasedEnv
    env_ids：
        torch.Tensor，需要施加力的环境索引
    force_range：
        tuple[float, float]，力的大小范围（牛顿），每个轴独立采样
        例如 (-10.0, 10.0) — 力在 X/Y/Z 三个方向都是 -10~10N
    torque_range：
        tuple[float, float]，扭矩的大小范围（牛·米），每个轴独立采样
    asset_cfg：
        SceneEntityCfg，默认 SceneEntityCfg("robot")

    输出： 无返回值。副作用：写入 permanent_wrench_composer——外力在后续每步物理仿真中持续生效
'''
def apply_external_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    """随机定制对身体的外部力和扭矩。

    这个函数创建了从给定的范围抽取的随机力量和扭矩的集合。
    动力和扭矩数量等于体体数量乘以环境数量。
    通过调用``asset.set_external_force_and_torque``，将力和扭矩应用于机体。
    只有在环境中调用``asset.write_data_to_sim()``时才会应用力和扭矩。
    """
    '''
    示例 ：恒定的侧面风 — 单次施加
        @configclass
        class EventCfg:
            """仿真启动后施加恒定侧向力（模拟侧风）"""

            constant_side_wind = EventTermCfg(
                func=events.apply_external_force_torque,
                mode="startup",                         # 启动后一次
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_ids=[0]),  # 只对根身体（躯干）
                    "force_range": (5.0, 5.0),          # 固定 5N（min=max）
                    "torque_range": (0.0, 0.0),         # 无扭矩
                },
            )
        效果： force_range=(5.0, 5.0) — min=max，所以不是随机，是恒定的 5N。整个训练期间机器人被推向一个方向，策略要学会产生一个抵消力。
    通俗类比：
        这个函数就像训练时的"随机推搡教练"——在你练习走路时，教练时不时从各个方向推你一把。
        推的方向、力度每次都不同（uniform 采样）。
        推完手不松（permanent_wrench_composer — 力持续作用），你得自己用力站稳。
        练多了，真实的推搡也不怕。
    '''

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )


'''
通过直接覆写根速度来实现"推一把"的效果。
    和 apply_external_force_torque（通过力间接改变速度）不同，这个函数直接改速度值——像是在仿真世界中说"机器人现在以 v 的速度运动"，跳过了牛顿定律的 F=ma 过程。
输入/输出参数
    env：
        ManagerBasedEnv
    env_ids：
        torch.Tensor，需要推的环境索引
    velocity_range：
        dict[str, tuple[float, float]]，六个自由度的速度范围字典：
            键："x", "y", "z"（线速度，m/s），"roll", "pitch", "yaw"（角速度，rad/s）
            值：(min, max) 元组
            缺失的键默认为 (0.0, 0.0) — 该轴速度不变
    asset_cfg：
        SceneEntityCfg，默认 SceneEntityCfg("robot")

    输出： 无返回值。副作用：直接覆写 PhysX 中的根速度
'''
def push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    """通过将根速度设置为给定的范围内的随机值来推动资产。

    这会产生类似于随机冲动推动资产的效果，
    它从给定的范围中取出根速度样本，并将速度设置在物理仿真中。

    函数为每个轴和旋转的速度范围采用字典。
    字典的键是``x``，``y``，``z``，``roll``，``pitch``和``yaw``。
    这些值是表格``(min， max)``的双倍。
    如果字典中没有关键，速度为该轴设置为零。
    """
    '''
    示例 1：reset 时给机器人一个初始速度
        @configclass
        class EventCfg:
            """每次 reset 时机器人有一个随机的初始运动"""

            random_initial_velocity = EventTermCfg(
                func=events.push_by_setting_velocity,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "velocity_range": {
                        "x": (-0.5, 1.0),         # 前向 -0.5~1.0 m/s
                        "y": (-0.3, 0.3),         # 侧向 ±0.3 m/s
                        "yaw": (-0.5, 0.5),       # 旋转 ±0.5 rad/s
                        # z, roll, pitch 不填 → 默认为 0（不跳、不翻滚）
                    },
                },
            )
        效果： 每次 reset 时机器人被"丢出去"——有一个随机的初始线速度和角速度。
            策略必须从"已经在运动"的状态下瞬间学会控制，而不是每次都从静止开始。
            这防止了策略过度依赖"从静止启动"的特殊条件。
    通俗类比：
        apply_external_force_torque 像背后有人持续推你——力不松、你一直加速。
        push_by_setting_velocity 像你被传送带甩了出去——速度瞬间改变，然后靠你自己的动力学演化。
        前者训练"抗持续干扰"，后者训练"从非静止状态恢复"。
    '''
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


'''
一次性 reset 资产的全部根状态——位置、朝向、线速度、角速度。
    这是之前 push_by_setting_velocity（只改速度）的"完整版"——把位置和速度一起随机化。
    通常用在 "reset" 模式中，替代默认的"回到原点"逻辑。

输入/输出参数
    env：
        ManagerBasedEnv
    env_ids：
        torch.Tensor
    pose_range：
        dict[str, tuple[float, float]]，位置和朝向的随机范围：
            键："x", "y", "z"（位置偏移，米），"roll", "pitch", "yaw"（朝向偏移，弧度）
            位置是加到默认位置上的偏移，朝向是绕当前朝向的旋转增量
            缺失键默认 (0.0, 0.0)
    velocity_range：
        dict[str, tuple[float, float]]，线速度和角速度范围：
            键："x", "y", "z"（线速度），"roll", "pitch", "yaw"（角速度）
            缺失键默认 (0.0, 0.0)
    asset_cfg：
        SceneEntityCfg，默认 SceneEntityCfg("robot")

    输出： 无返回值。副作用：直接覆写 PhysX 中的根位姿和根速度
'''
def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    """在给定的范围内，将资产根状态重置为随机位置和速度。

    这种函数随机化了资产的根位置和速度。

    * 它从给定的范围中采样根位置，然后将它们添加到默认根位置，然后将它们设置在物理仿真中。
    * 它从给定的范围中取样根向并将它们放在物理仿真中。
    * 它从给定的范围中取出根速度样本，并将它们放在物理仿真中。

    函数为每一个轴和旋转的姿势和速度范围。
    字典的键是``x``，``y``，``z``，``roll``，``pitch``和``yaw``。
    这些值是表格``(min， max)``的双倍。
    如果字典没有关键，则该轴的位置或速度设置为零。
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()
    '''
    default_root_state 的形状是 [N, 13]，包含：
        索引     含义
        0:3      默认位置 [x, y, z]（在世界系中相对 env_origin 的偏移）
        3:7      默认朝向 [w, x, y, z]（四元数）
        7:10     默认线速度 [vx, vy, vz]
        10:13    默认角速度 [ωx, ωy, ωz]
    '''

    # poses # 采样 6 维随机偏移 [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # 位置 = 默认位置 + 环境原点偏移 + 随机偏移
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    '''
    最终位置 = default_root_position + env_origin + random_offset
           ↑                       ↑              ↑
      USD定义的出生点         这个环境的原点      随机偏移
     (如 [0, 0, 0.5])       (如 [2, 3, 0])    (如 [0.1, -0.2, 0.3])
        env.scene.env_origins[env_ids] 是每个独立环境在地图中的坐标原点。
        在多环境并行训练中，环境 0 在 (0,0,0)，环境 1 在 (2,0,0)（间隔 2 米排列）。
        加上 env_origin 保证机器人出现在"自己的地盘"而不是所有环境重叠在一个点。
    '''
    # 朝向 = 默认朝向 × 随机旋转增量（绕 XYZ 轴的欧拉角）
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    '''
    朝向计算：
        quat_mul(default_quat, delta_quat) — 先旋转 delta（随机小角度），再旋转 default（初始朝向）。
        效果是"在默认朝向的基础上，随机偏一点"。
    '''
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    '''
    示例 1：reset 时随机出生位置
        @configclass
        class EventCfg:
            """每次 reset 机器人出生位置和朝向随机"""

            randomize_spawn = EventTermCfg(
                func=events.reset_root_state_uniform,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "pose_range": {
                        "x": (-1.0, 1.0),         # X 方向 ±1m
                        "y": (-1.0, 1.0),         # Y 方向 ±1m
                        "yaw": (-0.5, 0.5),       # 朝向 ±0.5 rad（约 ±28°）
                        # z, roll, pitch 不填 → 0
                    },
                    "velocity_range": {
                        "x": (-0.2, 0.2),         # 初始向前/向后速度
                        # 其他不填 → 0
                    },
                },
            )
        效果： 机器人每次 reset 出现在默认位置 ±1m 的方形区域内，朝向在 ±28° 之间随机，有一个小幅初始前向速度。
            策略不能假设"机器人每次从同一个位置出发"——它必须学会从任意的起始位置和朝向完成行走任务。
    '''


'''
和 reset_root_state_uniform 几乎完全一样——reset 资产的根位置和速度。
    但有一个关键差异：朝向不是在有限范围内随机，而是在整个 SO(3) 空间（所有可能的 3D 旋转）中均匀采样。
    reset_root_state_uniform：
        效果: 在默认朝向基础上，绕 X/Y/Z 各偏一点 → 朝向仍接近默认值
    reset_root_state_with_random_orientation：
        效果: 完全忽略默认朝向，从 SO(3) 均匀采样 → 可能朝任何方向
'''
def reset_root_state_with_random_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root position and velocities sampled randomly within the given ranges
    and the asset root orientation sampled randomly from the SO(3).

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation uniformly from the SO(3) and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of position ranges for each axis. The keys of the dictionary are ``x``,
      ``y``, and ``z``. The orientation is sampled uniformly from the SO(3).
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.
    """
    """在给定的范围内随机抽取的资产根位置和速度，以及从SO(3中随机抽取的资产根导向重置。

    这种函数随机化了资产的根位置和速度。

    * 它从给定的范围中采样根位置，然后将它们添加到默认根位置，然后将它们设置在物理仿真中。
    * 它从SO(3) 中均地采样根导向，并将它们放入物理仿真中。
    * 它从给定的范围中取出根速度样本，并将它们放在物理仿真中。

    函数为每个轴和旋转采用位置和速度范围的字典:

    * :attr:`pose_range` - 每个轴的位置范围字典.字典的键是``x``，``y``和``z``.从SO[3]中均地采样方向。
    * :attr:`velocity_range` - 每个轴和旋转的速度范围字典.字典的键是``x``，``y``，``z``，``roll``，``pitch``和``yaw``。

    这些值是表格``(min， max)``的双倍。
    如果字典没有特定的键，则该轴的位置设置为零。
    """
    '''
    什么是 SO(3)？
        SO(3) = 所有可能的 3D 旋转构成的集合。在 SO(3) 上均匀采样意味着：朝上、朝下、朝左、朝右、斜着、倒立——每种方向的概率完全相同。
        reset_root_state_uniform 的朝向:
            yaw ∈ [-0.5, 0.5] rad → 只在 ±28° 内偏转
            机器人始终接近"站立"姿态
        reset_root_state_with_random_orientation 的朝向:
            完全随机 360° 球面均匀采样
            机器人可能 upside down（倒立）、sideways（侧躺）、任何姿态！
    为什么要单独一个函数？
        欧拉角的"死区"问题。
        如果你用 reset_root_state_uniform 把 yaw 范围设到 (-3.14, 3.14)，表面上覆盖了所有角度，但实际上——欧拉角的采样不是均匀的：某些朝向（尤其是接近 ±90° pitch 时）会因为 gimbal lock 而出现采样偏置。
        random_orientation 用数值方法直接在 SO(3) 上均匀采样，保证了真正的无偏分布。
    '''
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples
    orientations = math_utils.random_orientation(len(env_ids), device=asset.device)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    '''
    示例 1：最激进的位置随机化 — 机器人可能四脚朝天
        @configclass
        class EventCfg:
            """reset 时机器人出现在随机位置、完全随机朝向"""

            randomize_spawn_so3 = EventTermCfg(
                func=events.reset_root_state_with_random_orientation,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "pose_range": {
                        "x": (-1.0, 1.0),
                        "y": (-1.0, 1.0),
                        "z": (0.0, 0.5),           # 可能悬空 0~0.5m 掉下来
                    },
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-1.0, 0.0),
                        "roll": (-2.0, 2.0),
                        "pitch": (-2.0, 2.0),
                        "yaw": (-2.0, 2.0),
                    },
                },
            )
        效果： 机器人可能以任何姿态出生——站立、侧躺、甚至倒立。同时有速度和旋转。
            策略必须学会从任何初始姿态恢复站立——这对四足机器人的"摔倒后爬起来"任务至关重要。
            用 reset_root_state_uniform 无法产生倒立姿态，用这个函数可以。
    两个 reset 函数的选择指南
        用 reset_root_state_uniform 当:
        ✓ 朝向只需要在有限范围内随机（±30°、±90°）
        ✓ 默认朝向有意义（机器人应该接近站立）
        ✓ 训练行走、简单操作任务
        ✓ 配置直观（用欧拉角写范围）

        用 reset_root_state_with_random_orientation 当:
        ✓ 朝向需要完全随机（任何姿态都有可能）
        ✓ 训练"摔倒恢复"、"从任意姿态爬起来"
        ✓ 需要真正的均匀分布（不能有 gimbal lock 偏置）
        ✓ 不在乎默认朝向是什么
    '''

'''
从地形的有效平坦区域中随机选取出生位置。
    和前两个 reset 函数最大的区别：位置不是随机偏移，而是从地形预计算的 "init_pos" 平坦区中挑选。
输入/输出参数
    env：
        ManagerBasedEnv，必须包含 terrain 场景实体
    env_ids：
        torch.Tensor
    pose_range：
        dict[str, tuple[float, float]]，只有朝向范围——"roll", "pitch", "yaw"。位置不在这里指定（从地形来）
    velocity_range：
        dict[str, tuple[float, float]]，六个自由度的速度范围
    asset_cfg：
        SceneEntityCfg，默认 SceneEntityCfg("robot")

    输出： 无返回值。副作用：覆写 PhysX 中的根位姿和速度
'''
def reset_root_state_from_terrain(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state by sampling a random valid pose from the terrain.

    This function samples a random valid pose(based on flat patches) from the terrain and sets the root state
    of the asset to this position. The function also samples random velocities from the given ranges and sets them
    into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of pose ranges for each axis. The keys of the dictionary are ``roll``,
      ``pitch``, and ``yaw``. The position is sampled from the flat patches of the terrain.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.

    Note:
        The function expects the terrain to have valid flat patches under the key "init_pos". The flat patches
        are used to sample the random pose for the robot.

    Raises:
        ValueError: If the terrain does not have valid flat patches under the key "init_pos".
    """
    """通过从地形中抽取随机有效姿势来重置资产根状态。

    这个函数从地形中采样一个随机有效的姿势，基于平坦的补丁，并将资产的根状态设置在这个位置。
    函数还从给定的范围中抽取随机速度，并将它们放在物理仿真中。

    函数为每个轴和旋转采用位置和速度范围的字典:

    * :attr:`pose_range` - 每个轴的姿势范围的字典.字典的键是``roll``，``pitch``和``yaw``.从地形的平坦片段中取样位置。
    * :attr:`velocity_range` - 每个轴和旋转的速度范围字典.字典的键是``x``，``y``，``z``，``roll``，``pitch``和``yaw``。

    这些值是表格``(min， max)``的双倍。
    如果字典没有特定的键，则该轴的位置设置为零。

    说明：
        函数预计地形在"init_pos"键下有有效的平面补丁。
        机器人使用平坦的贴片来抽取随机姿势。

    异常：
        ValueError: 如果地形在"init_pos"键下没有有效的平面补丁。
    """
    '''
    三种 reset 函数的对比
        函数	                                        位置来源	        朝向来源                适合场景
        reset_root_state_uniform	                默认位置 + 随机偏移	    默认朝向 + 欧拉角增量       标准训练、操作
        reset_root_state_with_random_orientation	默认位置 + 随机偏移	    SO(3) 均匀随机            摔倒恢复
        reset_root_state_from_terrain	            地形平坦区中随机选点	欧拉角直接设绝对值          越野行走、导航
        什么是 terrain.flat_patches["init_pos"]？
            地形在创建时会预计算所有"足够平坦、适合机器人站立的区域"，存在 flat_patches 字典中。"init_pos" 是专门用于"机器人出生点"的平坦区域子集。
            地形数据结构:
                terrain.flat_patches["init_pos"]   →  形状 [terrain_levels, terrain_types, num_patches, 3]
                                                        每个 patch 是 [x, y, z] 世界坐标
                terrain.terrain_levels[env_ids]    →  每个环境的地形难度等级
                terrain.terrain_types[env_ids]     →  每个环境的地形类型
            采样逻辑： 根据环境的 terrain_level 和 terrain_type，从对应行列选中一个平坦位置。这保证了不同难度等级和地形类型各有自己的出生点集合。
    '''
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # obtain all flat patches corresponding to the valid poses
    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # sample random valid poses
    ids = torch.randint(0, valid_positions.shape[2], size=(len(env_ids),), device=env.device)
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    positions += asset.data.default_root_state[env_ids, :3]

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = asset.data.default_root_state[env_ids, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    '''
    示例 1：越野训练 — 机器人在地形不同位置出生
        @configclass
        class EventCfg:
            """每次 reset 从地形的任意平坦区域随机出生"""

            randomize_terrain_spawn = EventTermCfg(
                func=events.reset_root_state_from_terrain,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "pose_range": {
                        "yaw": (-3.14, 3.14),         # 面朝任意方向
                        # roll, pitch 不填 → 0（不会倾斜出生）
                    },
                    "velocity_range": {
                        "x": (-0.5, 0.5),             # 初始小幅速度
                    },
                },
            )
        效果： 机器人在地图上所有"init_pos"标记的平坦区域中随机出生——有时在山脚，有时在山顶。
            面朝方向完全随机（通过 pose_range.yaw 而非 reset_root_state_with_random_orientation），但始终保持直立（roll=pitch=0）。
            策略学会了从地图上各种不同的位置出发走路。
    '''


'''
通过缩放默认关节角度来随机化关节的初始状态。
    不是加减偏移，而是乘一个随机系数——比如默认膝关节角度是 0.6 rad，position_range=(0.5, 1.2) → 随机系数 0.8 → 新角度 = 0.6 × 0.8 = 0.48 rad。
    本质上是"默认姿态的等比缩放版本"。
输入/输出参数
    env：
        ManagerBasedEnv
    env_ids：
        torch.Tensor
    position_range：
        tuple[float, float]，位置缩放系数范围。例如 (0.5, 1.5) — 关节角度在默认值的 50%~150% 之间随机
    velocity_range：
        tuple[float, float]，速度缩放系数范围
    asset_cfg：
        SceneEntityCfg，默认 SceneEntityCfg("robot")。⚠️ 只支持 Articulation

    输出： 无返回值。副作用：覆写 PhysX 中的关节位置和速度
'''
def reset_joints_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    """通过按给定的范围调整默认位置和速度来重置机器人关节。

    这种函数从给定的范围中抽取随机值，并通过这些值量度调度默认的关节位置和速度。
    然后将规模值设置在物理仿真中。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


'''
和 reset_joints_by_scale 是姊妹函数——代码结构 99% 相同，唯一的差异在随机化那一行：加法偏移而非乘法缩放。
'''
def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    """按给定的范围重置机器人关节，

    这个函数从给定的范围中抽取随机值，并通过这些值来偏差默认的关节位置和速度。
    然后将偏见值设置在物理仿真中。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    '''
    scale vs offset 选择指南
        用 reset_joints_by_scale 当:
        ✓ 想保持姿态的整体形状（等比例缩放到不同幅度）
        ✓ 默认角度都非零（大角度关节变化大，小角度变化小 — 合理）
        ✓ 训练行走任务的初始姿态随机化
        ✗ 默认值为 0 的关节需要随机化 → 不行

        用 reset_joints_by_offset 当:
        ✓ 想每个关节独立随机偏移（姿态可以很"野"）
        ✓ 有默认角度为 0 的关节需要随机化
        ✓ 训练抗扰动的恢复能力
        ✓ 制造更大的姿态多样性
    两个关节 reset 函数的完整关系
                        reset_joints_by_scale       reset_joints_by_offset
                        ─────────────────────       ──────────────────────
        随机化方式         默认值 × 随机系数           默认值 + 随机偏移
        数学公式           q = q₀ × r                 q = q₀ + Δ
        关节间关系         保持比例（协调）             各关节独立（乱序）
        0 角度关节         不会变                      会变
        姿态效果           保留"站姿"的形态             可以产生各种奇怪姿态
        类比               整体调低/调高                每个关节自由抖动
    通俗类比：
        scale 像是对机器人说"保持站姿，但要更弯一点"或"保持站姿，但更直一点"——全身关节同比例缩放。
        offset 像是"你可以随便挪动每个关节"——左膝多弯一点，右髋少弯一点，脚踝可能完全伸直。
        scale 保留了协调性，offset 更激进、更多样。
        对于标准行走训练，先用 scale 做域随机化；对于"摔倒后站起来"这类需要极端姿态的任务，用 offset。
    '''

'''
随机化可变形物体（DeformableObject）的所有节点状态——每个顶点的位置和速度。
    和之前所有针对刚体/关节体的 reset 函数不同，这里操作的是形变体的内部顶点网格，就像一块布、一个软球、一堆颗粒。
'''
def reset_nodal_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset nodal state to a random position and velocity uniformly within the given ranges.

    This function randomizes the nodal position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default nodal position, before setting
      them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis. The keys of the
    dictionary are ``x``, ``y``, ``z``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    """在给定的范围内，将资产节点状态重置为随机位置和速度。

    这种函数随机化了资产的节点位置和速度。

    * 它从给定的范围中取样根位置，然后将它们添加到默认节点位置，然后将它们设置在物理仿真中。
    * 它从给定的范围中取出根速度样本，并将它们放在物理仿真中。

    函数为每个轴取定位和速度范围的字典。
    字典的键是``x``，``y``，``z``。
    这些值是表格``(min， max)``的双倍。
    如果字典没有关键，则该轴的位置或速度设置为零。
    """
    # extract the used quantities (to enable type-hinting)
    asset: DeformableObject = env.scene[asset_cfg.name]
    # get default root state
    nodal_state = asset.data.default_nodal_state_w[env_ids].clone()

    # position
    range_list = [position_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 1, 3), device=asset.device)

    nodal_state[..., :3] += rand_samples

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 1, 3), device=asset.device)

    nodal_state[..., 3:] += rand_samples

    # set into the physics simulation
    asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


'''
场景的"恢复出厂设置"——把所有资产（刚体、关节点、形变体）全部复位到它们的默认状态。
    这是 Isaac Lab 中最全面的 reset 函数，通常作为"清场"步骤在 reset 流程的最前面调用，然后再让其他随机化事件在默认值的基础上做抖动。

输入/输出参数
    env：
        ManagerBasedEnv
    env_ids：
        torch.Tensor
    reset_joint_targets：
        bool，默认 False
            False：只重置物理状态（位置、速度），不调整动作目标
            True：额外把关节的目标位置/速度也复位——相当于"不仅机器人回到站姿，还告诉 PD 控制器'你的新目标就是站姿本身'"
    输出： 无返回值。副作用：覆写 PhysX 中所有资产的根状态、关节状态、节点状态
'''
def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor, reset_joint_targets: bool = False):
    """Reset the scene to the default state specified in the scene configuration.

    If :attr:`reset_joint_targets` is True, the joint position and velocity targets of the articulations are
    also reset to their default values. This might be useful for some cases to clear out any previously set targets.
    However, this is not the default behavior as based on our experience, it is not always desired to reset
    targets to default values, especially when the targets should be handled by action terms and not event terms.
    """
    """将场景重置到场景配置中指定的默认状态。

    If :attr:`reset_joint_targets`是True，关节的关节位置和速度目标是
    也将其重置为默认值。
    在某些情况下，这可能有助于清除之前设定的目标。
    然而，这是不是默认的行为， 根据我们的经验， 并非总是希望重置目标到默认值，
    """
    '''
    为什么默认不重置 joint targets？
        目标位置应该由动作模块（JointPositionAction.apply_actions）管理，而不是由事件模块（event terms）管理。
        如果事件模块把目标位置复位了，动作模块还没来得及写入新目标——瞬间会出现"关节要去默认姿态，但下一帧动作模块让它去策略指定的姿态"的冲突。
        动作模块和事件模块各管各的，边界清晰。
    '''
    # rigid bodies  第 1 部分：重置刚体 遍历场景中所有刚体（如桌子、方盒、球），逐个复位。
    for rigid_object in env.scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    # articulations 第 2 部分：重置关节点
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        # reset joint targets if required
        if reset_joint_targets:
            articulation_asset.set_joint_position_target(default_joint_pos, env_ids=env_ids)
            articulation_asset.set_joint_velocity_target(default_joint_vel, env_ids=env_ids)
        '''
        关节点需要两步：先复位根状态（整体位置），再复位关节状态（内部关节角度）。
        这和 reset_root_state_uniform + reset_joints_by_offset 的双层结构对应。
        '''
    # deformable objects    第 3 部分：重置形变体   形变体不需要加 env_origins——它的默认节点状态已经是世界坐标。
    for deformable_object in env.scene.deformable_objects.values():
        # obtain default and set into the physics simulation
        nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)
    '''
    通俗类比：
        reset_scene_to_default 就像是"打烊后把餐厅恢复原样"——桌子回到原位（刚体），椅子收好（关节点），桌上的橡胶桌布铺平（形变体）。第二天开门时，餐厅是干净的默认状态。
        然后厨师和服务员（reset_root_state_uniform 等随机化事件）再根据自己的需要调整——把桌布拉歪一点、椅子挪开一点，创造每天的"训练多样性"。
    '''


'''
通过 NVIDIA Replicator API 随机化资产的视觉纹理——把物体的外观随机替换为不同图案（如木纹、金属、石头等）。
    和之前所有改"物理属性"的随机化不同，这个类只改变外观，不改变物理——视觉域随机化（Visual Domain Randomization），专门用于训练对视觉变化鲁棒的策略。
'''
class randomize_visual_texture_material(ManagerTermBase):
    """Randomize the visual texture of bodies on an asset using Replicator API.

    This function randomizes the visual texture of the bodies of the asset using the Replicator API.
    The function samples random textures from the given texture paths and applies them to the bodies
    of the asset. The textures are projected onto the bodies and rotated by the given angles.

    .. note::
        The function assumes that the asset follows the prim naming convention as:
        "{asset_prim_path}/{body_name}/visuals" where the body name is the name of the body to
        which the texture is applied. This is the default prim ordering when importing assets
        from the asset converters in Isaac Lab.

    .. note::
        When randomizing the texture of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """
    """使用复制器API来随机定制身体的视觉纹理。

    这种函数使用复制器API来随机定制物体的视觉纹理。
    函数从给定的纹理路径中抽取随机纹理，并将其应用到资产体中。
    这些纹理被投射到体体上，并由给定的角度旋转。

    .. 说明::
        函数假设该资产遵循prim命名规则为:"{asset_prim_path}/{body_name}/视觉" (体名是实质应用于体名)。
        在进口资产时，这是默认prim订单
        from the asset converters in Isaac Lab.

    .. 说明::
        在随机化单个资产的纹理时，请确保设置:attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics`为False。
        这确保物理解析器将单独分析个别资产属性。
    """
    '''
    通俗类比：
        之前的随机化函数像"换了物体的物理参数"——把桌子变重、变滑、变大。
        randomize_visual_texture_material 像"给桌子贴了不同的墙纸"——物理上还是同一张桌子，但看起来可能是木头的、金属的、石头的。
        策略被训练成"不看外表只看形状"的专家，在真实世界中不管物体长什么样都能操作。
    '''

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        """开始这个词。

        参数：
            cfg: 事件时间的配置。
            env: 环境情况。
        """
        super().__init__(cfg, env)

        # check to make sure replicate_physics is set to False, else raise error
        # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
        #   and the event manager doesn't check in that case.
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual texture material with scene replication enabled."
                " For stable USD-level randomization, please disable scene replication"
                " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # enable replicator extension if not already enabled
        enable_extension("omni.replicator.core")

        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")

        # obtain the asset entity
        asset = env.scene[asset_cfg.name]

        # join all bodies in the asset
        body_names = asset_cfg.body_names
        if isinstance(body_names, str):
            body_names_regex = body_names
        elif isinstance(body_names, list):
            body_names_regex = "|".join(body_names)
        else:
            body_names_regex = ".*"

        # create the affected prim path
        # Check if the pattern with '/visuals' yields results when matching `body_names_regex`.
        # If not, fall back to a broader pattern without '/visuals'.
        asset_main_prim_path = asset.cfg.prim_path
        pattern_with_visuals = f"{asset_main_prim_path}/{body_names_regex}/visuals"
        # Use sim_utils to check if any prims currently match this pattern
        matching_prims = sim_utils.find_matching_prim_paths(pattern_with_visuals)
        if matching_prims:
            # If matches are found, use the pattern with /visuals
            prim_path = pattern_with_visuals
        else:
            # If no matches found, fall back to the broader pattern without /visuals
            # This pattern (e.g., /World/envs/env_.*/Table/.*) should match visual prims
            # whether they end in /visuals or have other structures.
            prim_path = f"{asset_main_prim_path}/.*"
            logging.info(
                f"Pattern '{pattern_with_visuals}' found no prims. Falling back to '{prim_path}' for texture"
                " randomization."
            )

        # extract the replicator version
        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            texture_paths = cfg.params.get("texture_paths")
            event_name = cfg.params.get("event_name")
            texture_rotation = cfg.params.get("texture_rotation", (0.0, 0.0))

            # convert from radians to degrees
            texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

            # Create the omni-graph node for the randomization term
            def rep_texture_randomization():
                prims_group = rep.get.prims(path_pattern=prim_path)

                with prims_group:
                    rep.randomizer.texture(
                        textures=texture_paths,
                        project_uvw=True,
                        texture_rotate=rep.distribution.uniform(*texture_rotation),
                    )
                return prims_group.node

            # Register the event to the replicator
            with rep.trigger.on_custom_event(event_name=event_name):
                rep_texture_randomization()
        else:
            # acquire stage
            stage = get_current_stage()
            prims_group = rep.functional.get.prims(path_pattern=prim_path, stage=stage)

            num_prims = len(prims_group)
            # rng that randomizes the texture and rotation
            self.texture_rng = rep.rng.ReplicatorRNG()

            # Create the material first and bind it to the prims
            for i, prim in enumerate(prims_group):
                # Disable instancble
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)

            # TODO: Should we specify the value when creating the material?
            self.material_prims = rep.functional.create_batch.material(
                mdl="OmniPBR.mdl", bind_prims=prims_group, count=num_prims, project_uvw=True
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str],
        texture_rotation: tuple[float, float] = (0.0, 0.0),
    ):
        # note: This triggers the nodes for all the environments.
        #   We need to investigate how to make it happen only for a subset based on env_ids.
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # extract the replicator version
        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            rep.utils.send_og_event(event_name)
        else:
            # read parameters from the configuration
            texture_paths = texture_paths if texture_paths else self._cfg.params.get("texture_paths")
            texture_rotation = (
                texture_rotation if texture_rotation else self._cfg.params.get("texture_rotation", (0.0, 0.0))
            )

            # convert from radians to degrees
            texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

            num_prims = len(self.material_prims)
            random_textures = self.texture_rng.generator.choice(texture_paths, size=num_prims)
            random_rotations = self.texture_rng.generator.uniform(
                texture_rotation[0], texture_rotation[1], size=num_prims
            )

            # modify the material properties
            rep.functional.modify.attribute(self.material_prims, "diffuse_texture", random_textures)
            rep.functional.modify.attribute(self.material_prims, "texture_rotate", random_rotations)


'''
通过 NVIDIA Replicator API 随机化资产的漫反射颜色（diffuse_color_constant）。
    和 randomize_visual_texture_material（改纹理贴图）是姊妹类——一个改"图案"，一个改"纯色"。
    两者都只改变视觉外观，不改变物理属性。
'''
class randomize_visual_color(ManagerTermBase):
    """Randomize the visual color of bodies on an asset using Replicator API.

    This function randomizes the visual color of the bodies of the asset using the Replicator API.
    The function samples random colors from the given colors and applies them to the bodies
    of the asset.

    The function assumes that the asset follows the prim naming convention as:
    "{asset_prim_path}/{mesh_name}" where the mesh name is the name of the mesh to
    which the color is applied. For instance, if the asset has a prim path "/World/asset"
    and a mesh named "body_0/mesh", the prim path for the mesh would be
    "/World/asset/body_0/mesh".

    The colors can be specified as a list of tuples of the form ``(r, g, b)`` or as a dictionary
    with the keys ``r``, ``g``, ``b`` and values as tuples of the form ``(low, high)``.
    If a dictionary is used, the function will sample random colors from the given ranges.

    .. note::
        When randomizing the color of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """
    """使用复制器API来随机调整物体的视觉颜色。

    这种函数随机化使用复制器API来对物体的视觉颜色。
    函数从给定的颜色中抽取随机颜色，并将它们应用到资产的体体上。

    函数假设该资产遵循prim命名惯例为:"{asset_prim_path}/{mesh_name}"，其中网格名称是使用颜色的网格名称。
    例如，如果资产有一个prim路径"/世界/资产"和一个称为"body_0/网格"的网格，则该网格的prim路径将是"/世界/资产/body_0/网格"。

    颜色可以指定为 ``(r， g， b)`` 形式的双重列表或字典
    with the keys ``r``, ``g``, ``b`` and values as tuples of the form ``(low, high)``.
    如果使用字典，函数将从给定的范围中抽取随机颜色。

    .. 说明::
        在随机对个体资产的颜色时，请确保设置:attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics`到False。
        这确保物理解析器将单独分析个别资产属性。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the randomization term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        """开始随机化项。

        参数：
            cfg: 事件时间的配置。
            env: 环境情况。
        """
        super().__init__(cfg, env)

        # enable replicator extension if not already enabled
        enable_extension("omni.replicator.core")
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        mesh_name: str = cfg.params.get("mesh_name", "")  # type: ignore

        # check to make sure replicate_physics is set to False, else raise error
        # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
        #   and the event manager doesn't check in that case.
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual color with scene replication enabled."
                " For stable USD-level randomization, please disable scene replication"
                " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # obtain the asset entity
        asset = env.scene[asset_cfg.name]

        # create the affected prim path
        if not mesh_name.startswith("/"):
            mesh_name = "/" + mesh_name
        mesh_prim_path = f"{asset.cfg.prim_path}{mesh_name}"
        # TODO: Need to make it work for multiple meshes.

        # extract the replicator version
        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            colors = cfg.params.get("colors")
            event_name = cfg.params.get("event_name")

            # parse the colors into replicator format
            if isinstance(colors, dict):
                # (r, g, b) - low, high --> (low_r, low_g, low_b) and (high_r, high_g, high_b)
                color_low = [colors[key][0] for key in ["r", "g", "b"]]
                color_high = [colors[key][1] for key in ["r", "g", "b"]]
                colors = rep.distribution.uniform(color_low, color_high)
            else:
                colors = list(colors)

            # Create the omni-graph node for the randomization term
            def rep_color_randomization():
                prims_group = rep.get.prims(path_pattern=mesh_prim_path)
                with prims_group:
                    rep.randomizer.color(colors=colors)

                return prims_group.node

            # Register the event to the replicator
            with rep.trigger.on_custom_event(event_name=event_name):
                rep_color_randomization()
        else:
            stage = get_current_stage()
            prims_group = rep.functional.get.prims(path_pattern=mesh_prim_path, stage=stage)

            num_prims = len(prims_group)
            self.color_rng = rep.rng.ReplicatorRNG()

            # Create the material first and bind it to the prims
            for i, prim in enumerate(prims_group):
                # Disable instancble
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)

            # TODO: Should we specify the value when creating the material?
            self.material_prims = rep.functional.create_batch.material(
                mdl="OmniPBR.mdl", bind_prims=prims_group, count=num_prims, project_uvw=True
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
        mesh_name: str = "",
    ):
        # note: This triggers the nodes for all the environments.
        #   We need to investigate how to make it happen only for a subset based on env_ids.

        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            rep.utils.send_og_event(event_name)
        else:
            colors = colors if colors else self._cfg.params.get("colors")

            # parse the colors into replicator format
            if isinstance(colors, dict):
                # (r, g, b) - low, high --> (low_r, low_g, low_b) and (high_r, high_g, high_b)
                color_low = [colors[key][0] for key in ["r", "g", "b"]]
                color_high = [colors[key][1] for key in ["r", "g", "b"]]
                colors = [color_low, color_high]
            else:
                colors = list(colors)

            num_prims = len(self.material_prims)
            random_colors = self.color_rng.generator.uniform(colors[0], colors[1], size=(num_prims, 3))

            rep.functional.modify.attribute(self.material_prims, "diffuse_color_constant", random_colors)


"""
Internal helper functions.
"""
"""内部助理功能。
"""

'''
内部辅助函数——被 6 个事件类/函数复用。
    它将"分布采样"和"算术操作"组合成一个通用引擎，支持 3 种操作 × 3 种分布 = 9 种组合。
输入/输出参数
    data：
        torch.Tensor，形状 [N, M]（二维），待随机化的数据
    distribution_parameters：
        (param0, param1)：
            uniform：(min, max) — 均匀分布范围
            log_uniform：(min, max) — 对数均匀分布范围
            gaussian：(mean, std) — 正态分布参数
    dim_0_ids：
        torch.Tensor | None，第一维（行）的索引。None = 所有行
    dim_1_ids：
        torch.Tensor | slice，第二维（列）的索引。slice(None) = 所有列
    operation：
        "add" / "scale" / "abs"
    distribution：
        "uniform" / "log_uniform" / "gaussian"

    输出： 修改后的 data（就地修改，也返回引用）
'''
def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    """根据给定的操作和分布进行数据随机化。

    参数：
        data: 随机定位的数据子。
              形状是 (dim_0，dim_1)。
        distribution_parameters: 分配给样本值的参数。
        dim_0_ids: 随机化第一维度的索引。
        dim_1_ids: 第二维度的索引进行随机化。
        operation: 在数据上执行的操作。
                   选项:"添加"，"规模"，"abs"。
        distribution: 随机值的分布。
                      选项:"统一""，log_uniform"。

    返回：
        随机化后的数据子。
        形状是 (dim_0，dim_1)。

    异常：
        NotImplementedError: 如果操作或分配不支持。
    """
    '''
    通俗类比：
        _randomize_prop_by_op 就像一个"万能调料机"
        ——你告诉它"往这个盘子的这格加什么料"（data[dim_0_ids, dim_1_ids]）、"加多少"（distribution_parameters）、"怎么加——是撒在上面（add）、搅拌均匀（scale）、还是全部换掉（abs）"——它就能自动处理。
        6 个事件类就像是 6 个不同的厨师，各自有不同的食材（质量、刚度、重力），但都用同一台调料机来完成"随机化调味"这一步。
        没有调料机的话，每个厨师都要自己配调料——代码会又长又容易出错。
    '''
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data

'''
校验 "scale" 操作中用到的 (low, high) 分布参数是否合法。
    events.py 中最后的内部辅助函数，被 5 个事件类的 __init__ 调用来做参数验证。
    这是一个纯粹的"防御性编程"函数——捕获配置错误，给出清晰的错误信息。
输入/输出参数
    params：
        tuple[float, float] | None，(low, high) 范围。None 表示该属性不被随机化（跳过校验）
    name：
        str，参数名称，用于错误信息（如 "mass_distribution_params"）
    allow_negative：
        bool，默认 False — 大多数缩放参数不允许负值
    allow_zero：
        bool，默认 True — 大多数缩放参数允许零（但有些不允许，见下表）
    *（强制关键字参数）：
        * 表示 allow_negative 和 allow_zero 必须用关键字形式传递（不能按位置），防止调用时参数顺序搞混。_validate_scale_range(params, name, True, False) 会被语法检测拒绝——必须写成 _validate_scale_range(params, name, allow_negative=True, allow_zero=False)。

    输出： 无返回值（或 None）。不正确就抛异常
'''
def _validate_scale_range(
    params: tuple[float, float] | None,
    name: str,
    *,
    allow_negative: bool = False,
    allow_zero: bool = True,
) -> None:
    """
    Validates a (low, high) tuple used in scale-based randomization.

    This function ensures the tuple follows expected rules when applying a 'scale'
    operation. It performs type and value checks, optionally allowing negative or
    zero lower bounds.

    Args:
        params (tuple[float, float] | None): The (low, high) range to validate. If None,
            validation is skipped.
        name (str): The name of the parameter being validated, used for error messages.
        allow_negative (bool, optional): If True, allows the lower bound to be negative.
            Defaults to False.
        allow_zero (bool, optional): If True, allows the lower bound to be zero.
            Defaults to True.

    Raises:
        TypeError: If `params` is not a tuple of two numbers.
        ValueError: If the lower bound is negative or zero when not allowed.
        ValueError: If the upper bound is less than the lower bound.

    Example:
        _validate_scale_range((0.5, 1.5), "mass_scale")
    """
    """在基于规模的随机化中使用的 (低，高) 元组进行验证。

    这种函数确保在应用"规模"操作时，元组遵循预期规则。
    它执行类型和值检查，可选择允许负或零的下限。

    参数：
        params (tuple[float, float] | None): 验证的 (低，高) 范围。
                                             如果None，验证将被跳过。
        name (str): 验证参数名称，用于错误信息。
        allow_negative (bool, optional): 如果 True，则允许下边界为负。
                                         默认为 False。
        allow_zero (bool, optional): 如果True，则允许下边界为零。
                                     默认为 True。

    异常：
        TypeError: 如果`params`不是两个数字的。
        ValueError: 如果下限是负值或零值，如果不允许。
        ValueError: 如果上限小于下限。

    示例：
        根据"_validate_scale_range"的规定，mass_scale")
    """
    '''
    通俗类比：
    _validate_scale_range 就像工地上的"安全护栏"——你在把一车砖（随机化参数）倒进搅拌机（_randomize_prop_by_op）之前，护栏检查砖块是否大小合适、有没有混入钢筋头（负数或零）。
    如果发现不合适的砖，护栏直接拦住并告诉你"第三块砖太小了"（lower bound must be > 0）。
    有了护栏，搅拌机不会因为吃错料而卡住（物理引擎崩溃）。
    '''
    if params is None:  # caller didn’t request randomisation for this field
        return
    low, high = params
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise TypeError(f"{name}: expected (low, high) to be a tuple of numbers, got {params}.")
    if not allow_negative and not allow_zero and low <= 0:
        raise ValueError(f"{name}: lower bound must be > 0 when using the 'scale' operation (got {low}).")
    if not allow_negative and allow_zero and low < 0:
        raise ValueError(f"{name}: lower bound must be ≥ 0 when using the 'scale' operation (got {low}).")
    if high < low:
        raise ValueError(f"{name}: upper bound ({high}) must be ≥ lower bound ({low}).")
