# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg, SpawnerCfg
from isaaclab.utils import configclass

from . import wrappers


@configclass
class MultiAssetSpawnerCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):
    """Configuration parameters for loading multiple assets from their individual configurations.

    Specifying values for any properties at the configuration level will override the settings of
    individual assets' configuration. For instance if the attribute
    :attr:`MultiAssetSpawnerCfg.mass_props` is specified, its value will overwrite the values of the
    mass properties in each configuration inside :attr:`assets_cfg` (wherever applicable).
    This is done to simplify configuring similar properties globally. By default, all properties are set to None.

    The following is an exception to the above:

    * :attr:`visible`: This parameter is ignored. Its value for the individual assets is used.
    * :attr:`semantic_tags`: If specified, it will be appended to each individual asset's semantic tags.

    """
    """配置参数用于从其单个配置中加载多个资产。

    在配置级别上指定任何属性的值将取消单个资产配置的设置。
    例如，如果指定:attr:`MultiAssetSpawnerCfg.mass_props`属性，它的值将重写:attr:`assets_cfg`内部每个配置中的质量属性值 (如适用)。
    为了简化全球范围内配置类似属性。
    默认情况下，所有属性设置为None。

    下面的情况是例外的:

    * :attr:`visible`:这个参数被忽略。
    * :attr:`semantic_tags`:如果指定，将添加到各个资产的语义标签。
    """

    func = wrappers.spawn_multi_asset

    assets_cfg: list[SpawnerCfg] = MISSING
    """List of asset configurations to spawn."""
    """产品配置列表"""

    random_choice: bool = True
    """Whether to randomly select an asset configuration. Default is True.

    If False, the asset configurations are spawned in the order they are provided in the list.
    If True, a random asset configuration is selected for each spawn.
    """
    """是否随机选择资产配置。
    默认是True。

    如果 False，资产配置以列表中提供的顺序生成。
    如果True，则为每个产物选择一个随机的资产配置。
    """


@configclass
class MultiUsdFileCfg(UsdFileCfg):
    """Configuration parameters for loading multiple USD files.

    Specifying values for any properties at the configuration level is applied to all the assets
    imported from their USD files.

    .. tip::
        It is recommended that all the USD based assets follow a similar prim-hierarchy.

    """
    """装载多个USD文件的配置参数。

    在配置级别上指定任何属性的值应用于所有从USD文件中进口的资产。

    .. 提示::
        建议所有基于USD的资产都遵循类似的prim等级。
    """

    func = wrappers.spawn_multi_usd_file

    usd_path: str | list[str] = MISSING
    """Path or a list of paths to the USD files to spawn asset from."""
    """路径或USD文件的路径列表"""

    random_choice: bool = True
    """Whether to randomly select an asset configuration. Default is True.

    If False, the asset configurations are spawned in the order they are provided in the list.
    If True, a random asset configuration is selected for each spawn.
    """
    """是否随机选择资产配置。
    默认是True。

    如果 False，资产配置以列表中提供的顺序生成。
    如果True，则为每个产物选择一个随机的资产配置。
    """
