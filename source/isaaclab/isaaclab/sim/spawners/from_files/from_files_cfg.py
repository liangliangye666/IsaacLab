# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.sim import converters, schemas
from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg, SpawnerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import from_files


@configclass
class FileCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Configuration parameters for spawning an asset from a file.

    This class is a base class for spawning assets from files. It includes the common parameters
    for spawning assets from files, such as the path to the file and the function to use for spawning
    the asset.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.
    """
    """配置参数用于从文件中产产资产。

    这类是从文件中产生的基础类。
    它包括共同的参数
    for spawning assets from files, such as the path to the file and the function to use for spawning
    资产。

    说明：
        默认情况下，所有属性设置为None。
        这意味着将不会添加或修改任何特性prim在产卵时默认可用的属性之外prim。

        如果它们设置为值，那么产生的prim的属性将以嵌套的方式进行修改。
        这是在指定的属性称呼相应的函数。
    """

    scale: tuple[float, float, float] | None = None
    """Scale of the asset. Defaults to None, in which case the scale is not modified."""
    """资产规模。
    在 None 时的默认情况，在这种情况下，尺度不会被修改。
    """

    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None
    """Properties to apply to the articulation root."""
    """适用于关节根的属性。"""

    fixed_tendons_props: schemas.FixedTendonPropertiesCfg | None = None
    """Properties to apply to the fixed tendons (if any)."""
    """适用于固定肌的特性 (如果有的话)。"""

    spatial_tendons_props: schemas.SpatialTendonPropertiesCfg | None = None
    """Properties to apply to the spatial tendons (if any)."""
    """适用于空间肌的特性 (如果有的话)。"""

    joint_drive_props: schemas.JointDrivePropertiesCfg | None = None
    """Properties to apply to a joint.

    .. note::
        The joint drive properties set the USD attributes of all the joint drives in the asset.
        We recommend using this attribute sparingly and only when necessary. Instead, please use the
        :attr:`~isaaclab.assets.ArticulationCfg.actuators` parameter to set the joint drive properties
        for specific joints in an articulation.
    """
    """适用于联合的属性。

    .. 说明::
        合动驱动的属性设置了资产中的所有合动驱动的USD属性。
        我们建议使用这种属性，只有在必要时。
        请使用:attr:`~isaaclab.assets.ArticulationCfg.actuators`参数来设置联合驱动性能
        for specific joints in an articulation.
    """

    visual_material_path: str = "material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """
    """在prim中使用的视觉材料的路径。
    "材料"的默认设置。

    如果路径是相对的，那么它将相对于prim的路径。
    如果`visual_material`不是None，则会忽略这个参数。
    """

    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties to override the visual material properties in the URDF file.

    Note:
        If None, then no visual material will be added.
    """
    """视觉材料属性以取代URDF文件中的视觉材料属性。

    说明：
        如果是None，则不会添加任何视觉材料。
    """


@configclass
class UsdFileCfg(FileCfg):
    """USD file to spawn asset from.

    USD files are imported directly into the scene. However, given their complexity, there are various different
    operations that can be performed on them. For example, selecting variants, applying materials, or modifying
    existing properties.

    To prevent the explosion of configuration parameters, the available operations are limited to the most common
    ones. These include:

    - **Selecting variants**: This is done by specifying the :attr:`variants` parameter.
    - **Creating and applying materials**: This is done by specifying the :attr:`visual_material` parameter.
    - **Modifying existing properties**: This is done by specifying the respective properties in the configuration
      class. For instance, to modify the scale of the imported prim, set the :attr:`scale` parameter.

    See :meth:`spawn_from_usd` for more information.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.
    """
    """在 USD 文件中，

    USD文件直接进口到场景。
    然而，由于它们复杂，
    例如，选择变体，应用材料或修改现有的特性。

    为了防止配置参数爆炸，可用的操作仅限于最常见的操作。
    这些包括:

    - **选择变体**:通过指定:attr:`variants`参数来完成。
    - **创建和应用材料**:通过指定:attr:`visual_material`参数完成。
    - **修改现有的属性**:通过在配置中指定各自的属性
      class. For instance, to modify the scale of the imported prim, set the :attr:`scale` parameter.

    See :麻:`spawn_from_usd`更多信息。

    .. 说明::
        配置参数包括各种属性。
        如果不是`None`，则这些属性在产生的prim上以嵌套的方式进行修改。

        如果它们设置为值，那么产生的prim的属性将以嵌套的方式进行修改。
        这是在指定的属性称呼相应的函数。
    """

    func: Callable = from_files.spawn_from_usd

    usd_path: str = MISSING
    """Path to the USD file to spawn asset from."""
    """进入USD文件的路径。"""

    variants: object | dict[str, str] | None = None
    """Variants to select from in the input USD file. Defaults to None, in which case no variants are applied.

    This can either be a configclass object, in which case each attribute is used as a variant set name and
    its specified value, or a dictionary mapping between the two. Please check the
    :meth:`~isaaclab.sim.utils.select_usd_variants` function for more information.
    """
    """在输入 USD 文件中选择的变体。
    对于None的默认设置，在这种情况下，没有应用变体。

    这可以是一个 configclass 对象，在这种情况下，每个属性都被用作变量集名称及其指定值，或者是两者之间的字典映射。
    请查看:meth:`~isaaclab.sim.utils.select_usd_variants`函数，以获取更多信息。
    """


@configclass
class UrdfFileCfg(FileCfg, converters.UrdfConverterCfg):
    """URDF file to spawn asset from.

    It uses the :class:`UrdfConverter` class to create a USD file from URDF and spawns the imported
    USD file. Similar to the :class:`UsdFileCfg`, the generated USD file can be modified by specifying
    the respective properties in the configuration class.

    See :meth:`spawn_from_urdf` for more information.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.

    """
    """在 URDF 文件中，

    它使用:class:`UrdfConverter`类来从URDF创建USD文件，并生成进口的USD文件。
    与:class:`UsdFileCfg`类似，生成的USD文件可以通过指定配置类中的各自属性来修改。

    See :麻:`spawn_from_urdf`更多信息。

    .. 说明::
        配置参数包括各种属性。
        如果不是`None`，则这些属性在产生的prim上以嵌套的方式进行修改。

        如果它们设置为值，那么产生的prim的属性将以嵌套的方式进行修改。
        这是在指定的属性称呼相应的函数。
    """

    func: Callable = from_files.spawn_from_urdf


@configclass
class MjcfFileCfg(FileCfg, converters.MjcfConverterCfg):
    """MJCF file to spawn asset from.

    It uses the :class:`MjcfConverter` class to create a USD file from MJCF and spawns the imported
    USD file. Similar to the :class:`UsdFileCfg`, the generated USD file can be modified by specifying
    the respective properties in the configuration class.

    See :meth:`spawn_from_mjcf` for more information.

    .. note::
        The configuration parameters include various properties. If not `None`, these properties
        are modified on the spawned prim in a nested manner.

        If they are set to a value, then the properties are modified on the spawned prim in a nested manner.
        This is done by calling the respective function with the specified properties.

    """
    """在 MJCF 文件中，

    它使用:class:`MjcfConverter`类来从MJCF创建USD文件，并生成进口的USD文件。
    与:class:`UsdFileCfg`类似，生成的USD文件可以通过指定配置类中的各自属性来修改。

    See :麻:`spawn_from_mjcf`更多信息。

    .. 说明::
        配置参数包括各种属性。
        如果不是`None`，则这些属性在产生的prim上以嵌套的方式进行修改。

        如果它们设置为值，那么产生的prim的属性将以嵌套的方式进行修改。
        这是在指定的属性称呼相应的函数。
    """

    func: Callable = from_files.spawn_from_mjcf


"""
Spawning ground plane.
"""
"""在地上飞机。
"""


@configclass
class UsdFileWithCompliantContactCfg(UsdFileCfg):
    """Configuration for spawning a USD asset with compliant contact physics material.

    This class extends :class:`UsdFileCfg` to support applying compliant contact properties
    (stiffness and damping) to specific prims in the spawned asset. It uses the
    :meth:`spawn_from_usd_with_compliant_contact_material` function to perform the spawning and
    material application.
    """
    """配置用于使用符合标准的接触物理材料生成USD资产。

    该类扩展:class:`UsdFileCfg`以支持在产生的资产中对特定prims应用符合的接触特性 (硬度和缩)。
    它使用:meth:`spawn_from_usd_with_compliant_contact_material`函数进行繁殖和材料应用。
    """

    func: Callable = from_files.spawn_from_usd_with_compliant_contact_material

    compliant_contact_stiffness: float | None = None
    """Stiffness of the compliant contact. Defaults to None.

    This parameter is the same as
    :attr:`~isaaclab.sim.spawners.materials.RigidBodyMaterialCfg.compliant_contact_stiffness`.
    """
    """合规接触的硬度。
    默认为 None。

    这一参数与:attr:`~isaaclab.sim.spawners.materials.RigidBodyMaterialCfg.compliant_contact_stiffness`相同。
    """

    compliant_contact_damping: float | None = None
    """Damping of the compliant contact. Defaults to None.

    This parameter is the same as
    :attr:`isaaclab.sim.spawners.materials.RigidBodyMaterialCfg.compliant_contact_damping`.
    """
    """适应性接触的化。
    默认为 None。

    这一参数与:attr:`isaaclab.sim.spawners.materials.RigidBodyMaterialCfg.compliant_contact_damping`相同。
    """

    physics_material_prim_path: str | list[str] | None = None
    """Path to the prim or prims to apply the physics material to. Defaults to None, in which case the
    physics material is not applied.

    If the path is relative, then it will be relative to the prim's path.
    If None, then the physics material will not be applied.
    """
    """在 prim或 prims上运行物理材料。
    在 None 中，默认情况下，不使用物理材料。

    如果路径是相对的，那么它将相对于prim的路径。
    如果是None，那么物理材料不会被应用。
    """


@configclass
class GroundPlaneCfg(SpawnerCfg):
    """Create a ground plane prim.

    This uses the USD for the standard grid-world ground plane from Isaac Sim by default.
    """
    """创建一个地面飞机prim。

    这里默认情况下使用了来自Isaac Sim的标准格式世界地面飞机的USD。
    """

    func: Callable = from_files.spawn_ground_plane

    usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd"
    """Path to the USD file to spawn asset from. Defaults to the grid-world ground plane."""
    """进入USD文件的路径。
    基格世界地面飞机的默认状态。
    """

    color: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    """The color of the ground plane. Defaults to (0.0, 0.0, 0.0).

    If None, then the color remains unchanged.
    """
    """地平面的颜色。
    在 (0.0，0.0，0.0) 之前的默认设置。

    如果是None，那么颜色保持不变。
    """

    size: tuple[float, float] = (100.0, 100.0)
    """The size of the ground plane. Defaults to 100 m x 100 m."""
    """地面飞机的尺寸。
    在100m x100m的默认状态下。
    """

    physics_material: materials.RigidBodyMaterialCfg = materials.RigidBodyMaterialCfg()
    """Physics material properties. Defaults to the default rigid body material."""
    """物理材料的特性。
    默认硬体材料的默认故障。
    """
