# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: Usd.Stage | None
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import omni.physx.scripts.utils as physx_utils
from omni.physx.scripts import deformableUtils as deformable_utils
from pxr import PhysxSchema, Usd, UsdPhysics

from isaaclab.sim.utils.stage import get_current_stage

from ..utils import (
    apply_nested,
    find_global_fixed_joint_prim,
    get_all_matching_child_prims,
    safe_set_attribute_on_usd_schema,
)
from . import schemas_cfg

# import logger
logger = logging.getLogger(__name__)


"""
Constants.
"""
"""总数。
"""

# Mapping from string names to USD/PhysX tokens for mesh collision approximation
# Refer to omniverse documentation
# https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html#mesh-geometry-colliders
# for available tokens.
MESH_APPROXIMATION_TOKENS = {
    "boundingCube": UsdPhysics.Tokens.boundingCube,
    "boundingSphere": UsdPhysics.Tokens.boundingSphere,
    "convexDecomposition": UsdPhysics.Tokens.convexDecomposition,
    "convexHull": UsdPhysics.Tokens.convexHull,
    "none": UsdPhysics.Tokens.none,
    "meshSimplification": UsdPhysics.Tokens.meshSimplification,
    "sdf": PhysxSchema.Tokens.sdf,
}


PHYSX_MESH_COLLISION_CFGS = [
    schemas_cfg.ConvexDecompositionPropertiesCfg,
    schemas_cfg.ConvexHullPropertiesCfg,
    schemas_cfg.TriangleMeshPropertiesCfg,
    schemas_cfg.TriangleMeshSimplificationPropertiesCfg,
    schemas_cfg.SDFMeshPropertiesCfg,
]

USD_MESH_COLLISION_CFGS = [
    schemas_cfg.BoundingCubePropertiesCfg,
    schemas_cfg.BoundingSpherePropertiesCfg,
    schemas_cfg.ConvexDecompositionPropertiesCfg,
    schemas_cfg.ConvexHullPropertiesCfg,
    schemas_cfg.TriangleMeshSimplificationPropertiesCfg,
]


"""
Articulation root properties.
"""
"""关节根的特性。
"""


def define_articulation_root_properties(
    prim_path: str, cfg: schemas_cfg.ArticulationRootPropertiesCfg, stage: Usd.Stage | None = None
):
    """Apply the articulation root schema on the input prim and set its properties.

    See :func:`modify_articulation_root_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the articulation root schema.
        cfg: The configuration for the articulation root.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
        TypeError: When the prim already has conflicting API schemas.
    """
    """将关节根图应用于输入 prim，并设置其属性。

    See :功能:`modify_articulation_root_properties` 详细说明如何设置属性。

    参数：
        prim_path: 运用关节根方案的prim路径。
        cfg: 关节根的配置。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    异常：
        ValueError: 当prim路径不有效时。
        TypeError: 当prim已经有冲突的API方案。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get articulation USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has articulation applied on it
    if not UsdPhysics.ArticulationRootAPI(prim):
        UsdPhysics.ArticulationRootAPI.Apply(prim)
    # set articulation root properties
    modify_articulation_root_properties(prim_path, cfg, stage)


@apply_nested
def modify_articulation_root_properties(
    prim_path: str, cfg: schemas_cfg.ArticulationRootPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for an articulation root prim.

    The `articulation root`_ marks the root of an articulation tree. For floating articulations, this should be on
    the root body. For fixed articulations, this API can be on a direct or indirect parent of the root joint
    which is fixed to the world.

    The schema comprises of attributes that belong to the `ArticulationRootAPI`_ and `PhysxArticulationAPI`_.
    schemas. The latter contains the PhysX parameters for the articulation root.

    The properties are applied to the articulation root prim. The common properties (such as solver position
    and velocity iteration counts, sleep threshold, stabilization threshold) take precedence over those specified
    in the rigid body schemas for all the rigid bodies in the articulation.

    .. caution::
        When the attribute :attr:`schemas_cfg.ArticulationRootPropertiesCfg.fix_root_link` is set to True,
        a fixed joint is created between the root link and the world frame (if it does not already exist). However,
        to deal with physics parser limitations, the articulation root schema needs to be applied to the parent of
        the root link.

    .. note::
        This function is decorated with :func:`apply_nested` that set the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _articulation root: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Articulations.html
    .. _ArticulationRootAPI: https://openusd.org/dev/api/class_usd_physics_articulation_root_a_p_i.html
    .. _PhysxArticulationAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_articulation_a_p_i.html

    Args:
        prim_path: The prim path to the articulation root.
        cfg: The configuration for the articulation root.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        NotImplementedError: When the root prim is not a rigid body and a fixed joint is to be created.
    """
    """修改对关节根 prim 的PhysX参数。

    在`articulation root`标志着关节树的根。
    对于浮动关节，这应该是根部。
    对于固定关节，这个API可以在固定到世界的根关节的直接或间接母体上。

    该方案包括属于`ArticulationRootAPI`_和`PhysxArticulationAPI`_的属性。
    计划。
    后者包含关节根的PhysX参数。

    这些特性应用于关节根prim。
    常见的特性 (如溶剂位置和速度回复数量，睡眠门，稳定门) 在关节中的所有硬体上优先于硬体方案中规定的特性。

    .. 谨慎::
        当属性:attr:`schemas_cfg.ArticulationRootPropertiesCfg.fix_root_link`设置为True时，根链接和世界框架之间会创建一个固定的关节
        (如果它还没有存在)。
        然而，为了应对物理解析器的局限性，关节根图需要应用于根链的母体。

    .. 说明::
        这个函数是用:func:`apply_nested`装饰的，这些函数在输入prim路径下设置了所有prims的属性 (这些函数上应用了方案)。

    .. _articulation root: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Articulations.html
    .. _ArticulationRootAPI: https://openusd.org/dev/api/class_usd_physics_articulation_root_a_p_i.html
    .. _PhysxArticulationAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_articulation_a_p_i.html

    参数：
        prim_path: 关节根的prim路径。
        cfg: 关节根的配置。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    返回：
        如果设置性能成功，True，否则False。

    异常：
        NotImplementedError: 当根prim不是硬体，并且要创建固定关节时。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get articulation USD prim
    articulation_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has articulation applied on it
    if not UsdPhysics.ArticulationRootAPI(articulation_prim):
        return False
    # retrieve the articulation api
    physx_articulation_api = PhysxSchema.PhysxArticulationAPI(articulation_prim)
    if not physx_articulation_api:
        physx_articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(articulation_prim)

    # convert to dict
    cfg = cfg.to_dict()
    # extract non-USD properties
    fix_root_link = cfg.pop("fix_root_link", None)

    # set into physx api
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_articulation_api, attr_name, value, camel_case=True)

    # fix root link based on input
    # we do the fixed joint processing later to not interfere with setting other properties
    if fix_root_link is not None:
        # check if a global fixed joint exists under the root prim
        existing_fixed_joint_prim = find_global_fixed_joint_prim(prim_path)

        # if we found a fixed joint, enable/disable it based on the input
        # otherwise, create a fixed joint between the world and the root link
        if existing_fixed_joint_prim is not None:
            logger.info(
                f"Found an existing fixed joint for the articulation: '{prim_path}'. Setting it to: {fix_root_link}."
            )
            existing_fixed_joint_prim.GetJointEnabledAttr().Set(fix_root_link)
        elif fix_root_link:
            logger.info(f"Creating a fixed joint for the articulation: '{prim_path}'.")

            # note: we have to assume that the root prim is a rigid body,
            #   i.e. we don't handle the case where the root prim is not a rigid body but has articulation api on it
            # Currently, there is no obvious way to get first rigid body link identified by the PhysX parser
            if not articulation_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise NotImplementedError(
                    f"The articulation prim '{prim_path}' does not have the RigidBodyAPI applied."
                    " To create a fixed joint, we need to determine the first rigid body link in"
                    " the articulation tree. However, this is not implemented yet."
                )

            # create a fixed joint between the root link and the world frame
            physx_utils.createJoint(stage=stage, joint_type="Fixed", from_prim=None, to_prim=articulation_prim)

            # Having a fixed joint on a rigid body is not treated as "fixed base articulation".
            # instead, it is treated as a part of the maximal coordinate tree.
            # Moving the articulation root to the parent solves this issue. This is a limitation of the PhysX parser.
            # get parent prim
            parent_prim = articulation_prim.GetParent()
            # apply api to parent
            UsdPhysics.ArticulationRootAPI.Apply(parent_prim)
            PhysxSchema.PhysxArticulationAPI.Apply(parent_prim)

            # copy the attributes
            # -- usd attributes
            usd_articulation_api = UsdPhysics.ArticulationRootAPI(articulation_prim)
            for attr_name in usd_articulation_api.GetSchemaAttributeNames():
                attr = articulation_prim.GetAttribute(attr_name)
                parent_prim.GetAttribute(attr_name).Set(attr.Get())
            # -- physx attributes
            physx_articulation_api = PhysxSchema.PhysxArticulationAPI(articulation_prim)
            for attr_name in physx_articulation_api.GetSchemaAttributeNames():
                attr = articulation_prim.GetAttribute(attr_name)
                parent_prim.GetAttribute(attr_name).Set(attr.Get())

            # remove api from root
            articulation_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            articulation_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

    # success
    return True


"""
Rigid body properties.
"""
"""固体特性。
"""


def define_rigid_body_properties(
    prim_path: str, cfg: schemas_cfg.RigidBodyPropertiesCfg, stage: Usd.Stage | None = None
):
    """Apply the rigid body schema on the input prim and set its properties.

    See :func:`modify_rigid_body_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the rigid body schema.
        cfg: The configuration for the rigid body.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
        TypeError: When the prim already has conflicting API schemas.
    """
    """运用硬体方案在输入 prim 上，设置其属性。

    See :功能:`modify_rigid_body_properties` 详细说明如何设置属性。

    参数：
        prim_path: 运用硬体方案的prim路径。
        cfg: 硬体的配置。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    异常：
        ValueError: 当prim路径不有效时。
        TypeError: 当prim已经有冲突的API方案。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has rigid body applied on it
    if not UsdPhysics.RigidBodyAPI(prim):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    # set rigid body properties
    modify_rigid_body_properties(prim_path, cfg, stage)


@apply_nested
def modify_rigid_body_properties(
    prim_path: str, cfg: schemas_cfg.RigidBodyPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for a rigid body prim.

    A `rigid body`_ is a single body that can be simulated by PhysX. It can be either dynamic or kinematic.
    A dynamic body responds to forces and collisions. A `kinematic body`_ can be moved by the user, but does not
    respond to forces. They are similar to having static bodies that can be moved around.

    The schema comprises of attributes that belong to the `RigidBodyAPI`_ and `PhysxRigidBodyAPI`_.
    schemas. The latter contains the PhysX parameters for the rigid body.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _rigid body: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyOverview.html
    .. _kinematic body: https://openusd.org/release/wp_rigid_body_physics.html#kinematic-bodies
    .. _RigidBodyAPI: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    .. _PhysxRigidBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_rigid_body_a_p_i.html

    Args:
        prim_path: The prim path to the rigid body.
        cfg: The configuration for the rigid body.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    """修改PhysX参数，以适用于硬体prim。

    `rigid body`_是一个单体，可以通过PhysX仿真。
    它可以是动态或动态。
    一个动态的身体应对力量和碰撞。
    用户可以移动`kinematic body`_，但不响应力。
    它们类似于可以移动的静态体。

    该方案包括属于`RigidBodyAPI`_和`PhysxRigidBodyAPI`_的属性。
    计划。
    后者包含了硬体的PhysX参数。

    .. 说明::
        这个函数是用:func:`apply_nested`装饰的，它设置了所有prims的属性 (它们上应用了方案) 在输入prim路径下。

    .. _rigid body: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyOverview.html
    .. _kinematic body: https://openusd.org/release/wp_rigid_body_physics.html#kinematic-bodies
    .. _RigidBodyAPI: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    .. _PhysxRigidBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_rigid_body_a_p_i.html

    参数：
        prim_path: 走向硬体的prim路径。
        cfg: 硬体的配置。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    返回：
        如果设置性能成功，True，否则False。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get rigid-body USD prim
    rigid_body_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has rigid-body applied on it
    if not UsdPhysics.RigidBodyAPI(rigid_body_prim):
        return False
    # retrieve the USD rigid-body api
    usd_rigid_body_api = UsdPhysics.RigidBodyAPI(rigid_body_prim)
    # retrieve the physx rigid-body api
    physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(rigid_body_prim)
    if not physx_rigid_body_api:
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_body_prim)

    # convert to dict
    cfg = cfg.to_dict()
    # set into USD API
    for attr_name in ["rigid_body_enabled", "kinematic_enabled"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_rigid_body_api, attr_name, value, camel_case=True)
    # set into PhysX API
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_rigid_body_api, attr_name, value, camel_case=True)
    # success
    return True


"""
Collision properties.
"""
"""碰撞性能。
"""


def define_collision_properties(
    prim_path: str, cfg: schemas_cfg.CollisionPropertiesCfg, stage: Usd.Stage | None = None
):
    """Apply the collision schema on the input prim and set its properties.

    See :func:`modify_collision_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the rigid body schema.
        cfg: The configuration for the collider.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
    """
    """应对输入 prim 的碰撞方案，并设置其属性。

    See :功能:`modify_collision_properties` 详细说明如何设置属性。

    参数：
        prim_path: 运用硬体方案的prim路径。
        cfg: 碰撞机的配置。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    异常：
        ValueError: 当prim路径不有效时。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has collision applied on it
    if not UsdPhysics.CollisionAPI(prim):
        UsdPhysics.CollisionAPI.Apply(prim)
    # set collision properties
    modify_collision_properties(prim_path, cfg, stage)


@apply_nested
def modify_collision_properties(
    prim_path: str, cfg: schemas_cfg.CollisionPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX properties of collider prim.

    These properties are based on the `UsdPhysics.CollisionAPI`_ and `PhysxSchema.PhysxCollisionAPI`_ schemas.
    For more information on the properties, please refer to the official documentation.

    Tuning these parameters influence the contact behavior of the rigid body. For more information on
    tune them and their effect on the simulation, please refer to the
    `PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html>`__.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _UsdPhysics.CollisionAPI: https://openusd.org/dev/api/class_usd_physics_collision_a_p_i.html
    .. _PhysxSchema.PhysxCollisionAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_collision_a_p_i.html

    Args:
        prim_path: The prim path of parent.
        cfg: The configuration for the collider.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    """修改碰撞机prim的PhysX性能。

    这些属性基于`UsdPhysics.CollisionAPI`_和`PhysxSchema.PhysxCollisionAPI`_方案。
    更多关于房地产的信息请参阅官方文件。

    调整这些参数影响了硬体的接触行为。
    更多关于调整它们及其对仿真的影响的信息请参阅`PhysX documentation
    <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html>`__。

    .. 说明::
        这个函数是用:func:`apply_nested`装饰的，它设置了所有prims的属性 (它们上应用了方案) 在输入prim路径下。

    .. _UsdPhysics.CollisionAPI: https://openusd.org/dev/api/class_usd_physics_collision_a_p_i.html
    .. _PhysxSchema.PhysxCollisionAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_collision_a_p_i.html

    参数：
        prim_path: 父母的prim路径。
        cfg: 碰撞机的配置。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    返回：
        如果设置性能成功，True，否则False。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    collider_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has collision applied on it
    if not UsdPhysics.CollisionAPI(collider_prim):
        return False
    # retrieve the USD collision api
    usd_collision_api = UsdPhysics.CollisionAPI(collider_prim)
    # retrieve the collision api
    physx_collision_api = PhysxSchema.PhysxCollisionAPI(collider_prim)
    if not physx_collision_api:
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collider_prim)

    # convert to dict
    cfg = cfg.to_dict()
    # set into USD API
    for attr_name in ["collision_enabled"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_collision_api, attr_name, value, camel_case=True)
    # set into PhysX API
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_collision_api, attr_name, value, camel_case=True)
    # success
    return True


"""
Mass properties.
"""
"""大量物质。
"""


def define_mass_properties(prim_path: str, cfg: schemas_cfg.MassPropertiesCfg, stage: Usd.Stage | None = None):
    """Apply the mass schema on the input prim and set its properties.

    See :func:`modify_mass_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the rigid body schema.
        cfg: The configuration for the mass properties.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
    """
    """将质量方案应用于输入 prim，并设置其属性。

    See :功能:`modify_mass_properties` 详细说明如何设置属性。

    参数：
        prim_path: 运用硬体方案的prim路径。
        cfg: 对质量特性的配置。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    异常：
        ValueError: 当prim路径不有效时。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has mass applied on it
    if not UsdPhysics.MassAPI(prim):
        UsdPhysics.MassAPI.Apply(prim)
    # set mass properties
    modify_mass_properties(prim_path, cfg, stage)


@apply_nested
def modify_mass_properties(prim_path: str, cfg: schemas_cfg.MassPropertiesCfg, stage: Usd.Stage | None = None) -> bool:
    """Set properties for the mass of a rigid body prim.

    These properties are based on the `UsdPhysics.MassAPI` schema. If the mass is not defined, the density is used
    to compute the mass. However, in that case, a collision approximation of the rigid body is used to
    compute the density. For more information on the properties, please refer to the
    `documentation <https://openusd.org/release/wp_rigid_body_physics.html#body-mass-properties>`__.

    .. caution::

        The mass of an object can be specified in multiple ways and have several conflicting settings
        that are resolved based on precedence. Please make sure to understand the precedence rules
        before using this property.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. UsdPhysics.MassAPI: https://openusd.org/dev/api/class_usd_physics_mass_a_p_i.html

    Args:
        prim_path: The prim path of the rigid body.
        cfg: The configuration for the mass properties.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    """固体体质量prim的特性设定。

    这些属性基于`UsdPhysics.MassAPI`方案。
    如果质量未定义，则使用密度来计算质量。
    然而，在这种情况下，用于计算密度使用硬体的碰撞接近。
    更多关于房地产的信息请参阅`documentation
    <https://openusd.org/release/wp_rigid_body_physics.html#body-mass-properties>`__。

    .. 谨慎::

        对象的质量可以以多种方式指定，并且有多种矛盾的设置，这些设置根据优先级得到解决。
        在使用此产品之前，请确保您了解优先权规则。

    .. 说明::
        这个函数是用:func:`apply_nested`装饰的，它设置了所有prims的属性 (它们上应用了方案) 在输入prim路径下。

    ..
    UsdPhysics.MassAPI: https://openusd.org/dev/api/class_usd_physics_mass_a_p_i.html

    参数：
        prim_path: 硬体的prim路径。
        cfg: 对质量特性的配置。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    返回：
        如果设置性能成功，True，否则False。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    rigid_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has mass API applied on it
    if not UsdPhysics.MassAPI(rigid_prim):
        return False
    # retrieve the USD mass api
    usd_physics_mass_api = UsdPhysics.MassAPI(rigid_prim)

    # convert to dict
    cfg = cfg.to_dict()
    # set into USD API
    for attr_name in ["mass", "density"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_physics_mass_api, attr_name, value, camel_case=True)
    # success
    return True


"""
Contact sensor.
"""
"""接触传感器。
"""


def activate_contact_sensors(prim_path: str, threshold: float = 0.0, stage: Usd.Stage = None):
    """Activate the contact sensor on all rigid bodies under a specified prim path.

    This function adds the PhysX contact report API to all rigid bodies under the specified prim path.
    It also sets the force threshold beyond which the contact sensor reports the contact. The contact
    reporting API can only be added to rigid bodies.

    Args:
        prim_path: The prim path under which to search and prepare contact sensors.
        threshold: The threshold for the contact sensor. Defaults to 0.0.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: If the input prim path is not valid.
        ValueError: If there are no rigid bodies under the prim path.
    """
    """在指定的prim路径下激活所有硬体的接触传感器。

    该函数将PhysX接触报告API添加到指定prim路径下的所有硬体中。
    它还设定了接触传感器报告接触的强力门。
    接触报告API只可添加到硬体中。

    参数：
        prim_path: 检查和准备接触传感器的prim路径。
        threshold: 接触传感器的门。
                   默认为0.0。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    异常：
        ValueError: 如果输入prim路径不有效。
        ValueError: 如果prim路径下面没有硬体。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get prim
    prim: Usd.Prim = stage.GetPrimAtPath(prim_path)
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # iterate over all children
    num_contact_sensors = 0
    all_prims = [prim]
    while len(all_prims) > 0:
        # get current prim
        child_prim = all_prims.pop(0)
        # check if prim is a rigid body
        # nested rigid bodies are not allowed by SDK so we can safely assume that
        # if a prim has a rigid body API, it is a rigid body and we don't need to
        # check its children
        if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # set sleep threshold to zero
            rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPrimPath())
            rb.CreateSleepThresholdAttr().Set(0.0)
            # add contact report API with threshold of zero
            if not child_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                logger.debug(f"Adding contact report API to prim: '{child_prim.GetPrimPath()}'")
                cr_api = PhysxSchema.PhysxContactReportAPI.Apply(child_prim)
            else:
                logger.debug(f"Contact report API already exists on prim: '{child_prim.GetPrimPath()}'")
                cr_api = PhysxSchema.PhysxContactReportAPI.Get(stage, child_prim.GetPrimPath())
            # set threshold to zero
            cr_api.CreateThresholdAttr().Set(threshold)
            # increment number of contact sensors
            num_contact_sensors += 1
        else:
            # add all children to tree
            all_prims += child_prim.GetChildren()
    # check if no contact sensors were found
    if num_contact_sensors == 0:
        raise ValueError(
            f"No contact sensors added to the prim: '{prim_path}'. This means that no rigid bodies"
            " are present under this prim. Please check the prim path."
        )
    # success
    return True


"""
Joint drive properties.
"""
"""联合驱动性能。
"""


@apply_nested
def modify_joint_drive_properties(
    prim_path: str, cfg: schemas_cfg.JointDrivePropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for a joint prim.

    This function checks if the input prim is a prismatic or revolute joint and applies the joint drive schema
    on it. If the joint is a tendon (i.e., it has the `PhysxTendonAxisAPI`_ schema applied on it), then the joint
    drive schema is not applied.

    Based on the configuration, this method modifies the properties of the joint drive. These properties are
    based on the `UsdPhysics.DriveAPI`_ schema. For more information on the properties, please refer to the
    official documentation.

    .. caution::

        We highly recommend modifying joint properties of articulations through the functionalities in the
        :mod:`isaaclab.actuators` module. The methods here are for setting simulation low-level
        properties only.

    .. _UsdPhysics.DriveAPI: https://openusd.org/dev/api/class_usd_physics_drive_a_p_i.html
    .. _PhysxTendonAxisAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_a_p_i.html

    Args:
        prim_path: The prim path where to apply the joint drive schema.
        cfg: The configuration for the joint drive.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        ValueError: If the input prim path is not valid.
    """
    """修改PhysX参数，用于联合prim。

    这种函数检查输入 prim 是不是一个 revolu体或转换组合，并对其应用了联合驱动方案。
    如果关节是一个子 (i.e.，它上应用`PhysxTendonAxisAPI`_图案)，则不应用关节驱动图案。

    根据配置，这种方法修改了联合驱动的性能。
    这些属性基于`UsdPhysics.DriveAPI`_方案。
    更多关于房地产的信息请参阅官方文件。

    .. 谨慎::

        我们强烈建议通过:mod:`isaaclab.actuators`模块的功能来修改关节的关节特性。
        这里的方法仅用于设置仿真低级属性。

    .. _UsdPhysics.DriveAPI: https://openusd.org/dev/api/class_usd_physics_drive_a_p_i.html
    .. _PhysxTendonAxisAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_a_p_i.html

    参数：
        prim_path: 运行联合驱动方案的prim路径。
        cfg: 联合驱动器的配置。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    返回：
        如果设置性能成功，True，否则False。

    异常：
        ValueError: 如果输入prim路径不有效。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    # check if prim has joint drive applied on it
    if prim.IsA(UsdPhysics.RevoluteJoint):
        drive_api_name = "angular"
    elif prim.IsA(UsdPhysics.PrismaticJoint):
        drive_api_name = "linear"
    else:
        return False
    # check that prim is not a tendon child prim
    # note: root prim is what "controls" the tendon so we still want to apply the drive to it
    if prim.HasAPI(PhysxSchema.PhysxTendonAxisAPI) and not prim.HasAPI(PhysxSchema.PhysxTendonAxisRootAPI):
        return False

    # check if prim has joint drive applied on it
    usd_drive_api = UsdPhysics.DriveAPI(prim, drive_api_name)
    if not usd_drive_api:
        usd_drive_api = UsdPhysics.DriveAPI.Apply(prim, drive_api_name)
    # check if prim has Physx joint drive applied on it
    physx_joint_api = PhysxSchema.PhysxJointAPI(prim)
    if not physx_joint_api:
        physx_joint_api = PhysxSchema.PhysxJointAPI.Apply(prim)

    # mapping from configuration name to USD attribute name
    cfg_to_usd_map = {
        "max_velocity": "max_joint_velocity",
        "max_effort": "max_force",
        "drive_type": "type",
    }
    # convert to dict
    cfg = cfg.to_dict()

    # check if linear drive
    is_linear_drive = prim.IsA(UsdPhysics.PrismaticJoint)
    # convert values for angular drives from radians to degrees units
    if not is_linear_drive:
        if cfg["max_velocity"] is not None:
            # rad / s --> deg / s
            cfg["max_velocity"] = cfg["max_velocity"] * 180.0 / math.pi
        if cfg["stiffness"] is not None:
            # N-m/rad --> N-m/deg
            cfg["stiffness"] = cfg["stiffness"] * math.pi / 180.0
        if cfg["damping"] is not None:
            # N-m-s/rad --> N-m-s/deg
            cfg["damping"] = cfg["damping"] * math.pi / 180.0

    # set into PhysX API
    for attr_name in ["max_velocity"]:
        value = cfg.pop(attr_name, None)
        attr_name = cfg_to_usd_map[attr_name]
        safe_set_attribute_on_usd_schema(physx_joint_api, attr_name, value, camel_case=True)
    # set into USD API
    for attr_name, attr_value in cfg.items():
        attr_name = cfg_to_usd_map.get(attr_name, attr_name)
        safe_set_attribute_on_usd_schema(usd_drive_api, attr_name, attr_value, camel_case=True)

    return True


"""
Fixed tendon properties.
"""
"""固定的肌肉特性。
"""


@apply_nested
def modify_fixed_tendon_properties(
    prim_path: str, cfg: schemas_cfg.FixedTendonPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for a fixed tendon attachment prim.

    A `fixed tendon`_ can be used to link multiple degrees of freedom of articulation joints
    through length and limit constraints. For instance, it can be used to set up an equality constraint
    between a driven and passive revolute joints.

    The schema comprises of attributes that belong to the `PhysxTendonAxisRootAPI`_ schema.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _fixed tendon: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxArticulationFixedTendon.html
    .. _PhysxTendonAxisRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_root_a_p_i.html

    Args:
        prim_path: The prim path to the tendon attachment.
        cfg: The configuration for the tendon attachment.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        ValueError: If the input prim path is not valid.
    """
    """修改固定部附带prim的PhysX参数。

    通过长度和限制限制，可以使用`fixed tendon`_来连接关节自由的多个程度。
    例如，它可以用来设置驱动和被动转动关节之间的平等约束。

    该方案包含属于`PhysxTendonAxisRootAPI`_方案的属性。

    .. 说明::
        这个函数是用:func:`apply_nested`装饰的，它设置了所有prims的属性 (它们上应用了方案) 在输入prim路径下。

    .. _fixed tendon: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxArticulationFixedTendon.html
    .. _PhysxTendonAxisRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_root_a_p_i.html

    参数：
        prim_path: X子附带的prim路径。
        cfg: 部的配置。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    返回：
        如果设置性能成功，True，否则False。

    异常：
        ValueError: 如果输入prim路径不有效。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    tendon_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has fixed tendon applied on it
    has_root_fixed_tendon = tendon_prim.HasAPI(PhysxSchema.PhysxTendonAxisRootAPI)
    if not has_root_fixed_tendon:
        return False

    # resolve all available instances of the schema since it is multi-instance
    for schema_name in tendon_prim.GetAppliedSchemas():
        # only consider the fixed tendon schema
        if "PhysxTendonAxisRootAPI" not in schema_name:
            continue
        # retrieve the USD tendon api
        instance_name = schema_name.split(":")[-1]
        physx_tendon_axis_api = PhysxSchema.PhysxTendonAxisRootAPI(tendon_prim, instance_name)

        # convert to dict
        cfg = cfg.to_dict()
        # set into PhysX API
        for attr_name, value in cfg.items():
            safe_set_attribute_on_usd_schema(physx_tendon_axis_api, attr_name, value, camel_case=True)
    # success
    return True


"""
Spatial tendon properties.
"""
"""空间的特性。
"""


@apply_nested
def modify_spatial_tendon_properties(
    prim_path: str, cfg: schemas_cfg.SpatialTendonPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for a spatial tendon attachment prim.

    A `spatial tendon`_ can be used to link multiple degrees of freedom of articulation joints
    through length and limit constraints. For instance, it can be used to set up an equality constraint
    between a driven and passive revolute joints.

    The schema comprises of attributes that belong to the `PhysxTendonAxisRootAPI`_ schema.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _spatial tendon: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxArticulationSpatialTendon.html
    .. _PhysxTendonAxisRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_root_a_p_i.html
    .. _PhysxTendonAttachmentRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_attachment_root_a_p_i.html
    .. _PhysxTendonAttachmentLeafAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_attachment_leaf_a_p_i.html

    Args:
        prim_path: The prim path to the tendon attachment.
        cfg: The configuration for the tendon attachment.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        ValueError: If the input prim path is not valid.
    """
    """修改PhysX参数，用于空间门附带prim。

    通过长度和限制限制，可以使用`spatial tendon`_来连接关节自由的多个程度。
    例如，它可以用来设置驱动和被动转动关节之间的平等约束。

    该方案包含属于`PhysxTendonAxisRootAPI`_方案的属性。

    .. 说明::
        这个函数是用:func:`apply_nested`装饰的，它设置了所有prims的属性 (它们上应用了方案) 在输入prim路径下。

    .. _spatial tendon: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxArticulationSpatialTendon.html
    .. _PhysxTendonAxisRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_root_a_p_i.html
    .. _PhysxTendonAttachmentRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_attachment_root_a_p_i.html
    .. _PhysxTendonAttachmentLeafAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_attachment_leaf_a_p_i.html

    参数：
        prim_path: X子附带的prim路径。
        cfg: 部的配置。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    返回：
        如果设置性能成功，True，否则False。

    异常：
        ValueError: 如果输入prim路径不有效。
    """
    # obtain stage
    if stage is None:
        stage = get_current_stage()
    # get USD prim
    tendon_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has spatial tendon applied on it
    has_spatial_tendon = tendon_prim.HasAPI(PhysxSchema.PhysxTendonAttachmentRootAPI) or tendon_prim.HasAPI(
        PhysxSchema.PhysxTendonAttachmentLeafAPI
    )
    if not has_spatial_tendon:
        return False

    # resolve all available instances of the schema since it is multi-instance
    for schema_name in tendon_prim.GetAppliedSchemas():
        # only consider the spatial tendon schema
        # retrieve the USD tendon api
        if "PhysxTendonAttachmentRootAPI" in schema_name:
            instance_name = schema_name.split(":")[-1]
            physx_tendon_spatial_api = PhysxSchema.PhysxTendonAttachmentRootAPI(tendon_prim, instance_name)
        elif "PhysxTendonAttachmentLeafAPI" in schema_name:
            instance_name = schema_name.split(":")[-1]
            physx_tendon_spatial_api = PhysxSchema.PhysxTendonAttachmentLeafAPI(tendon_prim, instance_name)
        else:
            continue
        # convert to dict
        cfg = cfg.to_dict()
        # set into PhysX API
        for attr_name, value in cfg.items():
            safe_set_attribute_on_usd_schema(physx_tendon_spatial_api, attr_name, value, camel_case=True)
    # success
    return True


"""
Deformable body properties.
"""
"""可变形的身体特性。
"""


def define_deformable_body_properties(
    prim_path: str, cfg: schemas_cfg.DeformableBodyPropertiesCfg, stage: Usd.Stage | None = None
):
    """Apply the deformable body schema on the input prim and set its properties.

    See :func:`modify_deformable_body_properties` for more details on how the properties are set.

    .. note::
        If the input prim is not a mesh, this function will traverse the prim and find the first mesh
        under it. If no mesh or multiple meshes are found, an error is raised. This is because the deformable
        body schema can only be applied to a single mesh.

    Args:
        prim_path: The prim path where to apply the deformable body schema.
        cfg: The configuration for the deformable body.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
        ValueError: When the prim has no mesh or multiple meshes.
    """
    """将可变化体格式应用于输入 prim，并设置其属性。

    See :功能:`modify_deformable_body_properties` 详细说明如何设置属性。

    .. 说明::
        如果输入prim不是网格，则这个函数将穿过prim并找到其下面的第一个网格。
        如果没有网格或多个网格，则会出现错误。
        这就是因为可变化体格式只能应用于单个网格。

    参数：
        prim_path: 运行可变体方案的prim路径。
        cfg: 变形机身的配置。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    异常：
        ValueError: 当prim路径不有效时。
        ValueError: 当prim没有网格或多个网格时。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    # traverse the prim and get the mesh
    matching_prims = get_all_matching_child_prims(prim_path, lambda p: p.GetTypeName() == "Mesh")
    # check if the mesh is valid
    if len(matching_prims) == 0:
        raise ValueError(f"Could not find any mesh in '{prim_path}'. Please check asset.")
    if len(matching_prims) > 1:
        # get list of all meshes found
        mesh_paths = [p.GetPrimPath() for p in matching_prims]
        raise ValueError(
            f"Found multiple meshes in '{prim_path}': {mesh_paths}."
            " Deformable body schema can only be applied to one mesh."
        )

    # get deformable-body USD prim
    mesh_prim = matching_prims[0]
    # check if prim has deformable-body applied on it
    if not PhysxSchema.PhysxDeformableBodyAPI(mesh_prim):
        PhysxSchema.PhysxDeformableBodyAPI.Apply(mesh_prim)
    # set deformable body properties
    modify_deformable_body_properties(mesh_prim.GetPrimPath(), cfg, stage)


@apply_nested
def modify_deformable_body_properties(
    prim_path: str, cfg: schemas_cfg.DeformableBodyPropertiesCfg, stage: Usd.Stage | None = None
):
    """Modify PhysX parameters for a deformable body prim.

    A `deformable body`_ is a single body that can be simulated by PhysX. Unlike rigid bodies, deformable bodies
    support relative motion of the nodes in the mesh. Consequently, they can be used to simulate deformations
    under applied forces.

    PhysX soft body simulation employs Finite Element Analysis (FEA) to simulate the deformations of the mesh.
    It uses two tetrahedral meshes to represent the deformable body:

    1. **Simulation mesh**: This mesh is used for the simulation and is the one that is deformed by the solver.
    2. **Collision mesh**: This mesh only needs to match the surface of the simulation mesh and is used for
       collision detection.

    For most applications, we assume that the above two meshes are computed from the "render mesh" of the deformable
    body. The render mesh is the mesh that is visible in the scene and is used for rendering purposes. It is composed
    of triangles and is the one that is used to compute the above meshes based on PhysX cookings.

    The schema comprises of attributes that belong to the `PhysxDeformableBodyAPI`_. schemas containing the PhysX
    parameters for the deformable body.

    .. caution::
        The deformable body schema is still under development by the Omniverse team. The current implementation
        works with the PhysX schemas shipped with Isaac Sim 4.0.0 onwards. It may change in future releases.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _deformable body: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html
    .. _PhysxDeformableBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_deformable_a_p_i.html

    Args:
        prim_path: The prim path to the deformable body.
        cfg: The configuration for the deformable body.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    """修改可变形体prim的PhysX参数。

    一个`deformable body`_是一个单体可以通过PhysX仿真。
    与硬体不同，可变形体支持网格中的节点的相对运动。
    因此，它们可以用来仿真在应用的力量下发生变形。

    PhysX软体仿真采用有限元素分析 (FEA) 来仿真网格的变形。
    它使用两个四面形网格来代表可变形的身体:

    1. **仿真网**:该网用于仿真，是溶剂扭曲的网。
    2. **碰撞网**:该网只需要与仿真网的表面相匹配，用于碰撞检测。

    在大多数应用中，我们假设上述两个网格是从可变化体的"发射网格"计算的。
    渲染网是场景可见的网，用于渲染目的。
    它由三角形组成，是基于PhysX料计算上述网格的。

    该方案包含属于`PhysxDeformableBodyAPI`_的属性。
    包含可变化机体的PhysX参数的方案。

    .. 谨慎::
        变形的身体方案仍在全宇宙团队开发中。
        目前的实施与以赛克·西姆4.0.0后的PhysX方案合作。
        在未来的版本中可能会发生变化。

    .. 说明::
        这个函数是用:func:`apply_nested`装饰的，它设置了所有prims的属性 (它们上应用了方案) 在输入prim路径下。

    .. _deformable body: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html
    .. _PhysxDeformableBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_deformable_a_p_i.html

    参数：
        prim_path: 形体的prim路径。
        cfg: 变形机身的配置。
        stage: 在哪里找到prim。
               在 None 上默认设置，此时使用当前阶段。

    返回：
        如果设置性能成功，True，否则False。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get deformable-body USD prim
    deformable_body_prim = stage.GetPrimAtPath(prim_path)

    # check if the prim is valid and has the deformable-body API
    if not deformable_body_prim.IsValid() or not PhysxSchema.PhysxDeformableBodyAPI(deformable_body_prim):
        return False

    # retrieve the physx deformable-body api
    physx_deformable_body_api = PhysxSchema.PhysxDeformableBodyAPI(deformable_body_prim)
    # retrieve the physx deformable api
    physx_deformable_api = PhysxSchema.PhysxDeformableAPI(physx_deformable_body_api)

    # convert to dict
    cfg = cfg.to_dict()
    # set into deformable body API
    attr_kwargs = {
        attr_name: cfg.pop(attr_name)
        for attr_name in [
            "kinematic_enabled",
            "collision_simplification",
            "collision_simplification_remeshing",
            "collision_simplification_remeshing_resolution",
            "collision_simplification_target_triangle_count",
            "collision_simplification_force_conforming",
            "simulation_hexahedral_resolution",
            "solver_position_iteration_count",
            "vertex_velocity_damping",
            "sleep_damping",
            "sleep_threshold",
            "settling_threshold",
            "self_collision",
            "self_collision_filter_distance",
        ]
    }
    status = deformable_utils.add_physx_deformable_body(stage, prim_path=prim_path, **attr_kwargs)
    # check if the deformable body was successfully added
    if not status:
        return False

    # obtain the PhysX collision API (this is set when the deformable body is added)
    physx_collision_api = PhysxSchema.PhysxCollisionAPI(deformable_body_prim)

    # set into PhysX API
    for attr_name, value in cfg.items():
        if attr_name in ["rest_offset", "contact_offset"]:
            safe_set_attribute_on_usd_schema(physx_collision_api, attr_name, value, camel_case=True)
        else:
            safe_set_attribute_on_usd_schema(physx_deformable_api, attr_name, value, camel_case=True)

    # success
    return True


"""
Collision mesh properties.
"""
"""碰撞网格性能
"""


def extract_mesh_collision_api_and_attrs(
    cfg: schemas_cfg.MeshCollisionPropertiesCfg,
) -> tuple[Callable, dict[str, Any]]:
    """Extract the mesh collision API function and custom attributes from the configuration.

    Args:
        cfg: The configuration for the mesh collision properties.

    Returns:
        A tuple containing the API function to use and a dictionary of custom attributes.

    Raises:
        ValueError: When neither USD nor PhysX API can be determined to be used.
    """
    """从配置中提取网格碰撞API函数和定制属性。

    参数：
        cfg: 网格碰撞性能的配置。

    返回：
        包含使用的API函数和定制属性字典的图布。

    异常：
        ValueError: 当不能确定使用USD或PhysX API。
    """
    # We use the number of user set attributes outside of the API function
    # to determine which API to use in ambiguous cases, so collect them here
    custom_attrs = {
        key: value
        for key, value in cfg.to_dict().items()
        if value is not None and key not in ["usd_func", "physx_func", "mesh_approximation_name"]
    }

    use_usd_api = False
    use_phsyx_api = False

    # We have some custom attributes and allow them
    if len(custom_attrs) > 0 and type(cfg) in PHYSX_MESH_COLLISION_CFGS:
        use_phsyx_api = True
    # We have no custom attributes
    elif len(custom_attrs) == 0:
        if type(cfg) in USD_MESH_COLLISION_CFGS:
            # Use the USD API
            use_usd_api = True
        else:
            # Use the PhysX API
            use_phsyx_api = True

    elif len(custom_attrs) > 0 and type(cfg) in USD_MESH_COLLISION_CFGS:
        raise ValueError("Args are specified but the USD Mesh API doesn't support them!")

    if use_usd_api:
        # Use USD API for corresponding attributes
        # For mesh collision approximation attribute, we set it explicitly in `modify_mesh_collision_properties``
        api_func = cfg.usd_func
    elif use_phsyx_api:
        api_func = cfg.physx_func
    else:
        raise ValueError("Either USD or PhysX API should be used for modifying mesh collision attributes!")

    return api_func, custom_attrs


def define_mesh_collision_properties(
    prim_path: str, cfg: schemas_cfg.MeshCollisionPropertiesCfg, stage: Usd.Stage | None = None
):
    """Apply the mesh collision schema on the input prim and set its properties.
    See :func:`modify_collision_mesh_properties` for more details on how the properties are set.
    Args:
        prim_path : The prim path where to apply the mesh collision schema.
        cfg : The configuration for the mesh collision properties.
        stage : The stage where to find the prim. Defaults to None, in which case the
            current stage is used.
    Raises:
        ValueError: When the prim path is not valid.
    """
    """将网格碰撞方案应用于输入prim，并设置其属性。
    See :功能:`modify_collision_mesh_properties` 详细说明如何设置属性。
    参数：
        prim_path : 应用网格碰撞方案的prim路径。
        cfg : 网格碰撞性能的配置。
        stage : 在哪里找到prim。
                在 None 上默认设置，此时使用当前阶段。
    异常：
        ValueError: 当prim路径不有效时。
    """
    # obtain stage
    if stage is None:
        stage = get_current_stage()
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    api_func, _ = extract_mesh_collision_api_and_attrs(cfg=cfg)

    # Only enable if not already enabled
    if not api_func(prim):
        api_func.Apply(prim)

    modify_mesh_collision_properties(prim_path=prim_path, cfg=cfg, stage=stage)


@apply_nested
def modify_mesh_collision_properties(
    prim_path: str, cfg: schemas_cfg.MeshCollisionPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Set properties for the mesh collision of a prim.
    These properties are based on either the `Phsyx the `UsdPhysics.MeshCollisionAPI` schema.
    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.
    .. UsdPhysics.MeshCollisionAPI: https://openusd.org/release/api/class_usd_physics_mesh_collision_a_p_i.html
    Args:
        prim_path : The prim path of the rigid body. This prim should be a Mesh prim.
        cfg : The configuration for the mesh collision properties.
        stage : The stage where to find the prim. Defaults to None, in which case the
            current stage is used.
    Returns:
        True if the properties were successfully set, False otherwise.
    Raises:
        ValueError: When the mesh approximation name is invalid.
    """
    """设置prim的网格碰撞特性。
    这些特性基于`Phsyx the `UsdPhysics.MeshCollisionAPI`方案。
    .. 说明::
        这个函数是用:func:`apply_nested`装饰的，它设置了所有prims的属性 (它们上应用了方案) 在输入prim路径下。
        ..
        UsdPhysics.MeshCollisionAPI:
        https://openusd.org/release/api/class_usd_physics_mesh_collision_a_p_i.html
    参数：
        prim_path : 硬体的prim路径。
                    这个prim应该是 Mesh prim。
        cfg : 网格碰撞性能的配置。
        stage : 在哪里找到prim。
                在 None 上默认设置，此时使用当前阶段。
    返回：
        如果设置性能成功，True，否则False。
    异常：
        ValueError: 如果网格近似名称是无效的。
    """
    # obtain stage
    if stage is None:
        stage = get_current_stage()
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)

    # we need MeshCollisionAPI to set mesh collision approximation attribute
    if not UsdPhysics.MeshCollisionAPI(prim):
        UsdPhysics.MeshCollisionAPI.Apply(prim)
    # convert mesh approximation string to token
    approximation_name = cfg.mesh_approximation_name
    if approximation_name not in MESH_APPROXIMATION_TOKENS:
        raise ValueError(
            f"Invalid mesh approximation name: '{approximation_name}'. "
            f"Valid options are: {list(MESH_APPROXIMATION_TOKENS.keys())}"
        )
    approximation_token = MESH_APPROXIMATION_TOKENS[approximation_name]
    safe_set_attribute_on_usd_schema(
        UsdPhysics.MeshCollisionAPI(prim), "Approximation", approximation_token, camel_case=False
    )

    api_func, custom_attrs = extract_mesh_collision_api_and_attrs(cfg=cfg)

    # retrieve the mesh collision API
    mesh_collision_api = api_func(prim)

    # set custom attributes into mesh collision API
    for attr_name, value in custom_attrs.items():
        # Only "Attribute" attr should be in format "boundingSphere", so set camel_case to be False
        if attr_name == "Attribute":
            camel_case = False
        else:
            camel_case = True
        safe_set_attribute_on_usd_schema(mesh_collision_api, attr_name, value, camel_case=camel_case)

    # success
    return True
