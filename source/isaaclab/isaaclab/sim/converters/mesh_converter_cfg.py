# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sim.converters.asset_converter_base_cfg import AssetConverterBaseCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils import configclass


@configclass
class MeshConverterCfg(AssetConverterBaseCfg):
    """The configuration class for MeshConverter."""
    """MeshConverter的配置类。"""

    mass_props: schemas_cfg.MassPropertiesCfg = None
    """Mass properties to apply to the USD. Defaults to None.

    Note:
        If None, then no mass properties will be added.
    """
    """适用于USD的质量特性。
    默认为 None。

    说明：
        如果 None，则不会添加质量属性。
    """

    rigid_props: schemas_cfg.RigidBodyPropertiesCfg = None
    """Rigid body properties to apply to the USD. Defaults to None.

    Note:
        If None, then no rigid body properties will be added.
    """
    """适用于USD的硬体特性。
    默认为 None。

    说明：
        如果是None，则不会添加任何固体特性。
    """

    collision_props: schemas_cfg.CollisionPropertiesCfg = None
    """Collision properties to apply to the USD. Defaults to None.

    Note:
        If None, then no collision properties will be added.
    """
    """适用于USD的碰撞性能。
    默认为 None。

    说明：
        如果 None，则不会添加碰撞属性。
    """
    mesh_collision_props: schemas_cfg.MeshCollisionPropertiesCfg = None
    """Mesh approximation properties to apply to all collision meshes in the USD.
    Note:
        If None, then no mesh approximation properties will be added.
    """
    """适用于USD中的所有碰撞网格的网格接近性质。
    说明：
        如果是None，则不会添加网格近似性质。
    """

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The translation of the mesh to the origin. Defaults to (0.0, 0.0, 0.0)."""
    """网格转换到原始。
    在 (0.0，0.0，0.0) 之前的默认设置。
    """

    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """The rotation of the mesh in quaternion format (w, x, y, z). Defaults to (1.0, 0.0, 0.0, 0.0)."""
    """在四元数格式 (w， x， y， z) 中的网格旋转。
    在 (1.0，0.0，0.0，0.0，0.0的默认情况下。
    """

    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """The scale of the mesh. Defaults to (1.0, 1.0, 1.0)."""
    """网格的尺度。
    设置为 (1.0， 1.0， 1.0)。
    """
