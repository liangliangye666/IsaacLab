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

from __future__ import annotations
"""共同的函数，可用于实现不同的事件。

事件包括任何与仿真状态的改变相关的东西。
这包括改变物理材料，应用外部力量，并重置资产状态。

函数可以传递到:class:`isaaclab.managers.EventTermCfg`对象，以实现函数引入的事件。
"""

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

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


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
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    # articulations
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
    # deformable objects
    for deformable_object in env.scene.deformable_objects.values():
        # obtain default and set into the physics simulation
        nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


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
