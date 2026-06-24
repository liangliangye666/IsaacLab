# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
import trimesh.transformations

from pxr import Usd, UsdPhysics

from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage

from ..materials import DeformableBodyMaterialCfg, RigidBodyMaterialCfg

if TYPE_CHECKING:
    from . import meshes_cfg


@clone
def spawn_mesh_sphere(
    prim_path: str,
    cfg: meshes_cfg.MeshSphereCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh sphere prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    """创建一个USD-Mesh球 prim与给出的属性。

    .. 说明::
        这个函数是用 :func:`clone` 装饰的，解决了 prim 路径的路径列表
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    参数：
        prim_path: 在 prim 路径或模式中产生资产。
                   如果prim路径是regex模式，那么所有匹配的prim路径都会产生资产。
        cfg: 设置实例。
        translation: 适用于prim w.r.t的翻译。
                     它的母prim。
                     默认为 None，在这种情况下，它设置为源。
        orientation: 在 (w， x， y， z) 中适用于prim w.r.t的方向。
                     它的母prim。
                     默认设置为None，在这种情况下，设置为身份。
        **kwargs: 其他关键词参数，比如``clone_in_fabric``。

    返回：
        创建了prim。

    异常：
        ValueError: 如果已在给定的路径上存在prim。
    """
    # create a trimesh sphere
    sphere = trimesh.creation.uv_sphere(radius=cfg.radius)

    # obtain stage handle
    stage = get_current_stage()
    # spawn the sphere as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, sphere, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_cuboid(
    prim_path: str,
    cfg: meshes_cfg.MeshCuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh cuboid prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    """创建一个USD-Mesh立方体prim，

    .. 说明::
        这个函数是用 :func:`clone` 装饰的，解决了 prim 路径的路径列表
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    参数：
        prim_path: 在 prim 路径或模式中产生资产。
                   如果prim路径是regex模式，那么所有匹配的prim路径都会产生资产。
        cfg: 设置实例。
        translation: 适用于prim w.r.t的翻译。
                     它的母prim。
                     默认为 None，在这种情况下，它设置为源。
        orientation: 在 (w， x， y， z) 中适用于prim w.r.t的方向。
                     它的母prim。
                     默认设置为None，在这种情况下，设置为身份。
        **kwargs: 其他关键词参数，比如``clone_in_fabric``。

    返回：
        创建了prim。

    异常：
        ValueError: 如果已在给定的路径上存在prim。
    """
    # create a trimesh box
    box = trimesh.creation.box(cfg.size)

    # obtain stage handle
    stage = get_current_stage()
    # spawn the cuboid as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, box, translation, orientation, None, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_cylinder(
    prim_path: str,
    cfg: meshes_cfg.MeshCylinderCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh cylinder prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    """创建一个USD-Mesh prim，

    .. 说明::
        这个函数是用 :func:`clone` 装饰的，解决了 prim 路径的路径列表
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    参数：
        prim_path: 在 prim 路径或模式中产生资产。
                   如果prim路径是regex模式，那么所有匹配的prim路径都会产生资产。
        cfg: 设置实例。
        translation: 适用于prim w.r.t的翻译。
                     它的母prim。
                     默认为 None，在这种情况下，它设置为源。
        orientation: 在 (w， x， y， z) 中适用于prim w.r.t的方向。
                     它的母prim。
                     默认设置为None，在这种情况下，设置为身份。
        **kwargs: 其他关键词参数，比如``clone_in_fabric``。

    返回：
        创建了prim。

    异常：
        ValueError: 如果已在给定的路径上存在prim。
    """
    # align axis from "Z" to input by rotating the cylinder
    axis = cfg.axis.upper()
    if axis == "X":
        transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    elif axis == "Y":
        transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    else:
        transform = None
    # create a trimesh cylinder
    cylinder = trimesh.creation.cylinder(radius=cfg.radius, height=cfg.height, transform=transform)

    # obtain stage handle
    stage = get_current_stage()
    # spawn the cylinder as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, cylinder, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_capsule(
    prim_path: str,
    cfg: meshes_cfg.MeshCapsuleCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh capsule prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    """创建一个USD-Mesh囊prim，

    .. 说明::
        这个函数是用 :func:`clone` 装饰的，解决了 prim 路径的路径列表
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    参数：
        prim_path: 在 prim 路径或模式中产生资产。
                   如果prim路径是regex模式，那么所有匹配的prim路径都会产生资产。
        cfg: 设置实例。
        translation: 适用于prim w.r.t的翻译。
                     它的母prim。
                     默认为 None，在这种情况下，它设置为源。
        orientation: 在 (w， x， y， z) 中适用于prim w.r.t的方向。
                     它的母prim。
                     默认设置为None，在这种情况下，设置为身份。
        **kwargs: 其他关键词参数，比如``clone_in_fabric``。

    返回：
        创建了prim。

    异常：
        ValueError: 如果已在给定的路径上存在prim。
    """
    # align axis from "Z" to input by rotating the cylinder
    axis = cfg.axis.upper()
    if axis == "X":
        transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    elif axis == "Y":
        transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    else:
        transform = None
    # create a trimesh capsule
    capsule = trimesh.creation.capsule(radius=cfg.radius, height=cfg.height, transform=transform)

    # obtain stage handle
    stage = get_current_stage()
    # spawn capsule if it doesn't exist.
    _spawn_mesh_geom_from_mesh(prim_path, cfg, capsule, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_mesh_cone(
    prim_path: str,
    cfg: meshes_cfg.MeshConeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh cone prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    """创建一个USD-Mesh角形prim与给出的属性。

    .. 说明::
        这个函数是用 :func:`clone` 装饰的，解决了 prim 路径的路径列表
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    参数：
        prim_path: 在 prim 路径或模式中产生资产。
                   如果prim路径是regex模式，那么所有匹配的prim路径都会产生资产。
        cfg: 设置实例。
        translation: 适用于prim w.r.t的翻译。
                     它的母prim。
                     默认为 None，在这种情况下，它设置为源。
        orientation: 在 (w， x， y， z) 中适用于prim w.r.t的方向。
                     它的母prim。
                     默认设置为None，在这种情况下，设置为身份。
        **kwargs: 其他关键词参数，比如``clone_in_fabric``。

    返回：
        创建了prim。

    异常：
        ValueError: 如果已在给定的路径上存在prim。
    """
    # align axis from "Z" to input by rotating the cylinder
    axis = cfg.axis.upper()
    if axis == "X":
        transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    elif axis == "Y":
        transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    else:
        transform = None
    # create a trimesh cone
    cone = trimesh.creation.cone(radius=cfg.radius, height=cfg.height, transform=transform)

    # obtain stage handle
    stage = get_current_stage()
    # spawn cone if it doesn't exist.
    _spawn_mesh_geom_from_mesh(prim_path, cfg, cone, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


"""
Helper functions.
"""
"""辅助函数。
"""


def _spawn_mesh_geom_from_mesh(
    prim_path: str,
    cfg: meshes_cfg.MeshCfg,
    mesh: trimesh.Trimesh,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
    stage: Usd.Stage | None = None,
    **kwargs,
):
    """Create a `USDGeomMesh`_ prim from the given mesh.

    This function is similar to :func:`shapes._spawn_geom_from_prim_type` but spawns the prim from a given mesh.
    In case of the mesh, it is spawned as a USDGeomMesh prim with the given vertices and faces.

    There is a difference in how the properties are applied to the prim based on the type of object:

    - Deformable body properties: The properties are applied to the mesh prim: ``{prim_path}/geometry/mesh``.
    - Collision properties: The properties are applied to the mesh prim: ``{prim_path}/geometry/mesh``.
    - Rigid body properties: The properties are applied to the parent prim: ``{prim_path}``.

    Args:
        prim_path: The prim path to spawn the asset at.
        cfg: The config containing the properties to apply.
        mesh: The mesh to spawn the prim from.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        scale: The scale to apply to the prim. Defaults to None, in which case this is set to identity.
        stage: The stage to spawn the asset at. Defaults to None, in which case the current stage is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Raises:
        ValueError: If a prim already exists at the given path.
        ValueError: If both deformable and rigid properties are used.
        ValueError: If both deformable and collision properties are used.
        ValueError: If the physics material is not of the correct type. Deformable properties require a deformable
            physics material, and rigid properties require a rigid physics material.

    .. _USDGeomMesh: https://openusd.org/dev/api/class_usd_geom_mesh.html
    """
    """从给定的网格中创建一个`USDGeomMesh`_prim。

    这个函数与:func:`shapes._spawn_geom_from_prim_type`类似，但从给定的网格中产生prim。
    在网格的情况下，它以给定的顶点和面孔为USDGeomMesh prim。

    根据对象类型，对prim的性能是不同的:

    - 可变形的车身特性:这些特性应应用于网格prim:``{prim_path}/geometry/mesh``。
    - 碰撞性质:这些特性应用于网格prim:``{prim_path}/geometry/mesh``。
    - 固体特性:这些特性应用于母体prim:``{prim_path}``。

    参数：
        prim_path: 在prim的路径中产生资产。
        cfg: 包含适用属性的配置。
        mesh: 让prim产生网格。
        translation: 适用于prim w.r.t的翻译。
                     它的母prim。
                     默认为 None，在这种情况下，它设置为源。
        orientation: 在 (w， x， y， z) 中适用于prim w.r.t的方向。
                     它的母prim。
                     默认设置为None，在这种情况下，设置为身份。
        scale: 适用于prim的尺度。
               默认设置为None，在这种情况下，设置为身份。
        stage: 在这个阶段，我们可以产生资产。
               在 None 上默认设置，此时使用当前阶段。
        **kwargs: 其他关键词参数，比如``clone_in_fabric``。

    异常：
        ValueError: 如果已在给定的路径上存在prim。
        ValueError: 如果使用可变化和刚性特性。
        ValueError: 如果使用可变性和碰撞性质。
        ValueError: 如果物理材料不是正确的类型。
                    变形性质需要变形性物理材料，而刚性质需要刚性物理材料。

    .. _USDGeomMesh: https://openusd.org/dev/api/class_usd_geom_mesh.html
    """
    # obtain stage handle
    stage = stage if stage is not None else get_current_stage()

    # spawn geometry if it doesn't exist.
    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    # check that invalid schema types are not used
    if cfg.deformable_props is not None and cfg.rigid_props is not None:
        raise ValueError("Cannot use both deformable and rigid properties at the same time.")
    if cfg.deformable_props is not None and cfg.collision_props is not None:
        raise ValueError("Cannot use both deformable and collision properties at the same time.")
    # check material types are correct
    if cfg.deformable_props is not None and cfg.physics_material is not None:
        if not isinstance(cfg.physics_material, DeformableBodyMaterialCfg):
            raise ValueError("Deformable properties require a deformable physics material.")
    if cfg.rigid_props is not None and cfg.physics_material is not None:
        if not isinstance(cfg.physics_material, RigidBodyMaterialCfg):
            raise ValueError("Rigid properties require a rigid physics material.")

    # create all the paths we need for clarity
    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    # create the mesh prim
    mesh_prim = create_prim(
        mesh_prim_path,
        prim_type="Mesh",
        scale=scale,
        attributes={
            "points": mesh.vertices,
            "faceVertexIndices": mesh.faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(mesh.faces)),
            "subdivisionScheme": "bilinear",
        },
        stage=stage,
    )

    # note: in case of deformable objects, we need to apply the deformable properties to the mesh prim.
    #   this is different from rigid objects where we apply the properties to the parent prim.
    if cfg.deformable_props is not None:
        # apply mass properties
        if cfg.mass_props is not None:
            schemas.define_mass_properties(mesh_prim_path, cfg.mass_props, stage=stage)
        # apply deformable body properties
        schemas.define_deformable_body_properties(mesh_prim_path, cfg.deformable_props, stage=stage)
    elif cfg.collision_props is not None:
        # decide on type of collision approximation based on the mesh
        if cfg.__class__.__name__ == "MeshSphereCfg":
            collision_approximation = "boundingSphere"
        elif cfg.__class__.__name__ == "MeshCuboidCfg":
            collision_approximation = "boundingCube"
        else:
            # for: MeshCylinderCfg, MeshCapsuleCfg, MeshConeCfg
            collision_approximation = "convexHull"
        # apply collision approximation to mesh
        # note: for primitives, we use the convex hull approximation -- this should be sufficient for most cases.
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision_api.GetApproximationAttr().Set(collision_approximation)
        # apply collision properties
        schemas.define_collision_properties(mesh_prim_path, cfg.collision_props, stage=stage)

    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(mesh_prim_path, material_path, stage=stage)

    # apply physics material
    if cfg.physics_material is not None:
        if not cfg.physics_material_path.startswith("/"):
            material_path = f"{geom_prim_path}/{cfg.physics_material_path}"
        else:
            material_path = cfg.physics_material_path
        # create material
        cfg.physics_material.func(material_path, cfg.physics_material)
        # apply material
        bind_physics_material(mesh_prim_path, material_path, stage=stage)

    # note: we apply the rigid properties to the parent prim in case of rigid objects.
    if cfg.rigid_props is not None:
        # apply mass properties
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
        # apply rigid properties
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
