# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

from isaaclab.sim.utils import clone, safe_set_attribute_on_usd_schema
from isaaclab.sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from . import physics_materials_cfg


@clone
def spawn_rigid_body_material(prim_path: str, cfg: physics_materials_cfg.RigidBodyMaterialCfg) -> Usd.Prim:
    """Create material with rigid-body physics properties.

    Rigid body materials are used to define the physical properties to meshes of a rigid body. These
    include the friction, restitution, and their respective combination modes. For more information on
    rigid body material, please refer to the `documentation on PxMaterial <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxBaseMaterial.html>`_.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration for the physics material.

    Returns:
        The spawned rigid body material prim.

    Raises:
        ValueError:  When a prim already exists at the specified prim path and is not a material.
    """
    """创建具有固体物理特性的材料。

    硬体材料用于定义硬体的物理特性。
    这些包括摩擦，恢复和它们各自的组合方式。
    关于硬体材料的更多信息请参阅`documentation on PxMaterial
    <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxBaseMaterial.html>`_。

    .. 说明::
        这个函数是用 :func:`clone` 装饰的，解决了 prim 路径的路径列表
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    参数：
        prim_path: 在 prim 路径或模式中产生资产。
                   如果prim路径是regex模式，那么所有匹配的prim路径都会产生资产。
        cfg: 物理材料的配置。

    返回：
        产生的硬体材料prim。

    异常：
        ValueError:  如果prim在指定prim路径上已经存在，并且不是材料。
    """
    # get stage handle
    stage = get_current_stage()

    # create material prim if no prim exists
    if not stage.GetPrimAtPath(prim_path).IsValid():
        _ = UsdShade.Material.Define(stage, prim_path)

    # obtain prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim is a material
    if not prim.IsA(UsdShade.Material):
        raise ValueError(f"A prim already exists at path: '{prim_path}' but is not a material.")
    # retrieve the USD rigid-body api
    usd_physics_material_api = UsdPhysics.MaterialAPI(prim)
    if not usd_physics_material_api:
        usd_physics_material_api = UsdPhysics.MaterialAPI.Apply(prim)
    # retrieve the collision api
    physx_material_api = PhysxSchema.PhysxMaterialAPI(prim)
    if not physx_material_api:
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)

    # convert to dict
    cfg = cfg.to_dict()
    del cfg["func"]
    # set into USD API
    for attr_name in ["static_friction", "dynamic_friction", "restitution"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_physics_material_api, attr_name, value, camel_case=True)
    # set into PhysX API
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_material_api, attr_name, value, camel_case=True)
    # return the prim
    return prim


@clone
def spawn_deformable_body_material(prim_path: str, cfg: physics_materials_cfg.DeformableBodyMaterialCfg) -> Usd.Prim:
    """Create material with deformable-body physics properties.

    Deformable body materials are used to define the physical properties to meshes of a deformable body. These
    include the friction and deformable body properties. For more information on deformable body material,
    please refer to the documentation on `PxFEMSoftBodyMaterial`_.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration for the physics material.

    Returns:
        The spawned deformable body material prim.

    Raises:
        ValueError:  When a prim already exists at the specified prim path and is not a material.

    .. _PxFEMSoftBodyMaterial: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxFEMSoftBodyMaterialModel.html
    """
    """创建具有可变体物理特性的材料。

    可变形体材料用于定义变形体的物理特性。
    这些包括摩擦和可变化的体质。
    对于可变形体材料的更多信息，请参阅`PxFEMSoftBodyMaterial`_的文档。

    .. 说明::
        这个函数是用 :func:`clone` 装饰的，解决了 prim 路径的路径列表
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    参数：
        prim_path: 在 prim 路径或模式中产生资产。
                   如果prim路径是regex模式，那么所有匹配的prim路径都会产生资产。
        cfg: 物理材料的配置。

    返回：
        产生的可变体材质prim。

    异常：
        ValueError:  如果prim在指定prim路径上已经存在，并且不是材料。

    .. _PxFEMSoftBodyMaterial: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxFEMSoftBodyMaterialModel.html
    """
    # get stage handle
    stage = get_current_stage()

    # create material prim if no prim exists
    if not stage.GetPrimAtPath(prim_path).IsValid():
        _ = UsdShade.Material.Define(stage, prim_path)

    # obtain prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim is a material
    if not prim.IsA(UsdShade.Material):
        raise ValueError(f"A prim already exists at path: '{prim_path}' but is not a material.")
    # retrieve the deformable-body api
    physx_deformable_body_material_api = PhysxSchema.PhysxDeformableBodyMaterialAPI(prim)
    if not physx_deformable_body_material_api:
        physx_deformable_body_material_api = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(prim)

    # convert to dict
    cfg = cfg.to_dict()
    del cfg["func"]
    # set into PhysX API
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_deformable_body_material_api, attr_name, value, camel_case=True)
    # return the prim
    return prim
