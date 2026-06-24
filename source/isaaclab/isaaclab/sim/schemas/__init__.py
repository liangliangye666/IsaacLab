# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for schemas used in Omniverse.

We wrap the USD schemas for PhysX and USD Physics in a more convenient API for setting the parameters from
Python. This is done so that configuration objects can define the schema properties to set and make it easier
to tune the physics parameters without requiring to open Omniverse Kit and manually set the parameters into
the respective USD attributes.

.. caution::

    Schema properties cannot be applied on prims that are prototypes as they are read-only prims. This
    particularly affects instanced assets where some of the prims (usually the visual and collision meshes)
    are prototypes so that the instancing can be done efficiently.

    In such cases, it is assumed that the prototypes have sim-ready properties on them that don't need to be modified.
    Trying to set properties into prototypes will throw a warning saying that the prim is a prototype and the
    properties cannot be set.

The schemas are defined in the following links:

* `UsdPhysics schema <https://openusd.org/dev/api/usd_physics_page_front.html>`_
* `PhysxSchema schema <https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/index.html>`_

Locally, the schemas are defined in the following files:

* ``_isaac_sim/extsPhysics/omni.usd.schema.physics/plugins/UsdPhysics/resources/UsdPhysics/schema.usda``
* ``_isaac_sim/extsPhysics/omni.usd.schema.physx/plugins/PhysxSchema/resources/generatedSchema.usda``

"""
"""包含Omniverse中使用的方案的公用工具的子模块。

我们将PhysX和USD物理的USD方案包装成一个更方便的API，
这样，配置对象可以定义设置的方案属性，并使物理参数更容易调节，而不需要打开Omniverse Kit，并手动设置参数到各自的USD属性中。

.. 谨慎::

    prims是原型的方案属性不能应用于，因为它们只能读取prims。
    这特别影响了一些prims (通常是视觉和碰撞网格) 的实例资产，以便实例化能够高效地进行。

    在这种情况下，假设原型具有无需修改的sim准备性质。
    试图将属性设置在原型中会发出警告说prim是原型，

方案在以下链接中定义:

* `UsdPhysics schema <https://openusd.org/dev/api/usd_physics_page_front.html>`_
* `PhysxSchema schema
  <https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/index.html>`_

在本地范围内，方案在以下文件中定义:

* ``_isaac_sim/extsPhysics/omni.usd.schema.physics/plugins/UsdPhysics/resources/UsdPhysics/schema.us
  da``
* ``_isaac_sim/extsPhysics/omni.usd.schema.physx/plugins/PhysxSchema/resources/generatedSchema.usda`
  `
"""

from .schemas import (
    MESH_APPROXIMATION_TOKENS,
    PHYSX_MESH_COLLISION_CFGS,
    USD_MESH_COLLISION_CFGS,
    activate_contact_sensors,
    define_articulation_root_properties,
    define_collision_properties,
    define_deformable_body_properties,
    define_mass_properties,
    define_mesh_collision_properties,
    define_rigid_body_properties,
    modify_articulation_root_properties,
    modify_collision_properties,
    modify_deformable_body_properties,
    modify_fixed_tendon_properties,
    modify_joint_drive_properties,
    modify_mass_properties,
    modify_mesh_collision_properties,
    modify_rigid_body_properties,
    modify_spatial_tendon_properties,
)
from .schemas_cfg import (
    ArticulationRootPropertiesCfg,
    BoundingCubePropertiesCfg,
    BoundingSpherePropertiesCfg,
    CollisionPropertiesCfg,
    ConvexDecompositionPropertiesCfg,
    ConvexHullPropertiesCfg,
    DeformableBodyPropertiesCfg,
    FixedTendonPropertiesCfg,
    JointDrivePropertiesCfg,
    MassPropertiesCfg,
    MeshCollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
    SDFMeshPropertiesCfg,
    SpatialTendonPropertiesCfg,
    TriangleMeshPropertiesCfg,
    TriangleMeshSimplificationPropertiesCfg,
)

__all__ = [
    # articulation root
    "ArticulationRootPropertiesCfg",
    "define_articulation_root_properties",
    "modify_articulation_root_properties",
    # rigid bodies
    "RigidBodyPropertiesCfg",
    "define_rigid_body_properties",
    "modify_rigid_body_properties",
    "activate_contact_sensors",
    # colliders
    "CollisionPropertiesCfg",
    "define_collision_properties",
    "modify_collision_properties",
    # deformables
    "DeformableBodyPropertiesCfg",
    "define_deformable_body_properties",
    "modify_deformable_body_properties",
    # joints
    "JointDrivePropertiesCfg",
    "modify_joint_drive_properties",
    # mass
    "MassPropertiesCfg",
    "define_mass_properties",
    "modify_mass_properties",
    # mesh colliders
    "MeshCollisionPropertiesCfg",
    "define_mesh_collision_properties",
    "modify_mesh_collision_properties",
    # bounding cube
    "BoundingCubePropertiesCfg",
    # bounding sphere
    "BoundingSpherePropertiesCfg",
    # convex decomposition
    "ConvexDecompositionPropertiesCfg",
    # convex hull
    "ConvexHullPropertiesCfg",
    # sdf mesh
    "SDFMeshPropertiesCfg",
    # triangle mesh
    "TriangleMeshPropertiesCfg",
    # triangle mesh simplification
    "TriangleMeshSimplificationPropertiesCfg",
    # tendons
    "FixedTendonPropertiesCfg",
    "SpatialTendonPropertiesCfg",
    "modify_fixed_tendon_properties",
    "modify_spatial_tendon_properties",
    # Constants for configs that use PhysX vs USD API
    "PHYSX_MESH_COLLISION_CFGS",
    "USD_MESH_COLLISION_CFGS",
    "MESH_APPROXIMATION_TOKENS",
]
