# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import Usd

from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage

if TYPE_CHECKING:
    from . import shapes_cfg


@clone
def spawn_sphere(
    prim_path: str,
    cfg: shapes_cfg.SphereCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USDGeom-based sphere prim with the given attributes.

    For more information, see `USDGeomSphere <https://openusd.org/dev/api/class_usd_geom_sphere.html>`_.

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
    """创建一个基于USDGeom的 prim球体。

    查看更多信息`USDGeomSphere <https://openusd.org/dev/api/class_usd_geom_sphere.html>`_。

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
    # obtain stage handle
    stage = get_current_stage()
    # spawn sphere if it doesn't exist.
    attributes = {"radius": cfg.radius}
    _spawn_geom_from_prim_type(prim_path, cfg, "Sphere", attributes, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_cuboid(
    prim_path: str,
    cfg: shapes_cfg.CuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USDGeom-based cuboid prim with the given attributes.

    For more information, see `USDGeomCube <https://openusd.org/dev/api/class_usd_geom_cube.html>`_.

    Note:
        Since USD only supports cubes, we set the size of the cube to the minimum of the given size and
        scale the cube accordingly.

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
        If a prim already exists at the given path.
    """
    """创建一个基于USDGeom的立方体prim，

    查看更多信息`USDGeomCube <https://openusd.org/dev/api/class_usd_geom_cube.html>`_。

    说明：
        由于USD只支持立方体，所以我们将立方体的尺寸设置为所给定的最小尺寸，

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
        如果已在给定的路径上存在prim。
    """
    # obtain stage handle
    stage = get_current_stage()
    # resolve the scale
    size = min(cfg.size)
    scale = [dim / size for dim in cfg.size]
    # spawn cuboid if it doesn't exist.
    attributes = {"size": size}
    _spawn_geom_from_prim_type(prim_path, cfg, "Cube", attributes, translation, orientation, scale, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_cylinder(
    prim_path: str,
    cfg: shapes_cfg.CylinderCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USDGeom-based cylinder prim with the given attributes.

    For more information, see `USDGeomCylinder <https://openusd.org/dev/api/class_usd_geom_cylinder.html>`_.

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
    """创建一个基于USDGeom的 prim，

    查看更多信息`USDGeomCylinder <https://openusd.org/dev/api/class_usd_geom_cylinder.html>`_。

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
    # obtain stage handle
    stage = get_current_stage()
    # spawn cylinder if it doesn't exist.
    attributes = {"radius": cfg.radius, "height": cfg.height, "axis": cfg.axis.upper()}
    _spawn_geom_from_prim_type(prim_path, cfg, "Cylinder", attributes, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_capsule(
    prim_path: str,
    cfg: shapes_cfg.CapsuleCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USDGeom-based capsule prim with the given attributes.

    For more information, see `USDGeomCapsule <https://openusd.org/dev/api/class_usd_geom_capsule.html>`_.

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
    """创建一个基于USDGeom的prim囊，

    查看更多信息`USDGeomCapsule <https://openusd.org/dev/api/class_usd_geom_capsule.html>`_。

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
    # obtain stage handle
    stage = get_current_stage()
    # spawn capsule if it doesn't exist.
    attributes = {"radius": cfg.radius, "height": cfg.height, "axis": cfg.axis.upper()}
    _spawn_geom_from_prim_type(prim_path, cfg, "Capsule", attributes, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_cone(
    prim_path: str,
    cfg: shapes_cfg.ConeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USDGeom-based cone prim with the given attributes.

    For more information, see `USDGeomCone <https://openusd.org/dev/api/class_usd_geom_cone.html>`_.

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
    """创建一个基于USDGeom的角形prim，

    查看更多信息`USDGeomCone <https://openusd.org/dev/api/class_usd_geom_cone.html>`_。

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
    # obtain stage handle
    stage = get_current_stage()
    # spawn cone if it doesn't exist.
    attributes = {"radius": cfg.radius, "height": cfg.height, "axis": cfg.axis.upper()}
    _spawn_geom_from_prim_type(prim_path, cfg, "Cone", attributes, translation, orientation, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


"""
Helper functions.
"""
"""辅助函数。
"""


def _spawn_geom_from_prim_type(
    prim_path: str,
    cfg: shapes_cfg.ShapeCfg,
    prim_type: str,
    attributes: dict,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
    stage: Usd.Stage | None = None,
):
    """Create a USDGeom-based prim with the given attributes.

    To make the asset instanceable, we must follow a certain structure dictated by how USD scene-graph
    instancing and physics work. The rigid body component must be added to each instance and not the
    referenced asset (i.e. the prototype prim itself). This is because the rigid body component defines
    properties that are specific to each instance and cannot be shared under the referenced asset. For
    more information, please check the `documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/physics/simulation_fundamentals.html#rigid-body>`_.

    Due to the above, we follow the following structure:

    * ``{prim_path}`` - The root prim that is an Xform with the rigid body and mass APIs if configured.
    * ``{prim_path}/geometry`` - The prim that contains the mesh and optionally the materials if configured.
      If instancing is enabled, this prim will be an instanceable reference to the prototype prim.

    Args:
        prim_path: The prim path to spawn the asset at.
        cfg: The config containing the properties to apply.
        prim_type: The type of prim to create.
        attributes: The attributes to apply to the prim.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        scale: The scale to apply to the prim. Defaults to None, in which case this is set to identity.
        stage: The stage to spawn the asset at. Defaults to None, in which case the current stage is used.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    """创建一个基于USDGeom的prim，

    为了使资产可以实例化，我们必须遵循一个特定的结构， 根据USD场景图实例化和物理的运行。
    每个实例必须添加硬体组件，而不是引用的资产 (i.e.原型prim本身)。
    这就是因为硬体组件定义为每个实例的特征，不能在引用资产下共享。
    更多信息请查看`documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/physics/simulation_fundamen
    tals.html#rigid-body>`_。

    由于上述情况，我们遵循以下结构:

    * ``{prim_path}`` - 根 prim 是一个X形式，如果配置，则具有硬体和质量 APIs。
    * ``{prim_path}/geometry`` - 包含网格和选项材料的prim，如果配置.如果启用实例化，这个prim将是原型prim的实例化参考。

    参数：
        prim_path: 在prim的路径中产生资产。
        cfg: 包含适用属性的配置。
        prim_type: 创建的prim类型。
        attributes: 适用于prim的属性。
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

    异常：
        ValueError: 如果已在给定的路径上存在prim。
    """
    # obtain stage handle
    stage = stage if stage is not None else get_current_stage()

    # spawn geometry if it doesn't exist.
    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    # create all the paths we need for clarity
    geom_prim_path = prim_path + "/geometry"
    mesh_prim_path = geom_prim_path + "/mesh"

    # create the geometry prim
    create_prim(mesh_prim_path, prim_type, scale=scale, attributes=attributes, stage=stage)
    # apply collision properties
    if cfg.collision_props is not None:
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

    # note: we apply rigid properties in the end to later make the instanceable prim
    # apply mass properties
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    # apply rigid body properties
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
