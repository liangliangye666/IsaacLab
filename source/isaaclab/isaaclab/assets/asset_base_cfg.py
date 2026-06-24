# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.sim import SpawnerCfg
from isaaclab.utils import configclass

from .asset_base import AssetBase


@configclass
class AssetBaseCfg:
    """The base configuration class for an asset's parameters.

    Please see the :class:`AssetBase` class for more information on the asset class.
    """
    """资产参数的基础配置类。

    请查看:class:`AssetBase`类，了解资产类别的更多信息。
    """

    @configclass
    class InitialStateCfg:
        """Initial state of the asset.

        This defines the default initial state of the asset when it is spawned into the simulation, as
        well as the default state when the simulation is reset.

        After parsing the initial state, the asset class stores this information in the :attr:`data`
        attribute of the asset class. This can then be accessed by the user to modify the state of the asset
        during the simulation, for example, at resets.
        """
        """资产的初始状态。

        根据该指标，在仿真中产生的资产的默认初始状态，以及在仿真重置时的默认状态。

        在分析初始状态后，资产类别将这些信息存储在资产类别的:attr:`data`属性中。
        然后，用户可以访问该数据，以在仿真过程中修改资产状态，例如在重置时。
        """

        # root position
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Position of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        """仿真世界框架中的根位置。
        在 (0.0，0.0，0.0) 之前的默认设置。
        """
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) of the root in simulation world frame.
        Defaults to (1.0, 0.0, 0.0, 0.0).
        """
        """在仿真世界框架中根的四元数旋转 (w，x，y，z)。
        在 (1.0，0.0，0.0，0.0，0.0的默认情况下。
        """

    class_type: type[AssetBase] = None
    """The associated asset class. Defaults to None, which means that the asset will be spawned
    but cannot be interacted with via the asset class.

    The class should inherit from :class:`isaaclab.assets.asset_base.AssetBase`.
    """
    """相关资产类别。
    对None的默认值，这意味着资产将产生，但无法通过资产类别进行交互。

    这类应该继承:class:`isaaclab.assets.asset_base.AssetBase`。
    """

    prim_path: str = MISSING
    """Prim path (or expression) to the asset.

    .. note::
        The expression can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Robot`` will be replaced with ``/World/envs/env_.*/Robot``.
    """
    """资产的原始路径 (或表达)。

    .. 说明::
        这个表达式可以包含环境命名空间regex ``{ENV_REGEX_NS}``，将被环境命名空间取代。

        Example: ``{ENV_REGEX_NS}/Robot``将被 ``/World/envs/env_.*/Robot`` 取代。
    """

    spawn: SpawnerCfg | None = None
    """Spawn configuration for the asset. Defaults to None.

    If None, then no prims are spawned by the asset class. Instead, it is assumed that the
    asset is already present in the scene.
    """
    """产品的产品配置。
    默认为 None。

    如果None，则没有prims由资产类别产生。
    相反，假设该资产已经存在场景。
    """

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose."""
    """硬体的初始状态。
    认同状态的默认问题。
    """

    collision_group: Literal[0, -1] = 0
    """Collision group of the asset. Defaults to ``0``.

    * ``-1``: global collision group (collides with all assets in the scene).
    * ``0``: local collision group (collides with other assets in the same environment).
    """
    """资产的碰撞组。
    在``0``上默认。

    * ``-1``:全球碰撞组 (与场景的所有资产发生碰撞)。
    * ``0``:局部碰撞组 (与同一个环境中的其他资产碰撞)。
    """

    debug_vis: bool = False
    """Whether to enable debug visualization for the asset. Defaults to ``False``."""
    """是否启用对资产的调试可视化。
    在``False``上默认。
    """
