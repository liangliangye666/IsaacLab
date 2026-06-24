# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import omni.kit.commands

from .asset_converter_base import AssetConverterBase
from .mjcf_converter_cfg import MjcfConverterCfg

if TYPE_CHECKING:
    import isaacsim.asset.importer.mjcf


class MjcfConverter(AssetConverterBase):
    """Converter for a MJCF description file to a USD file.

    This class wraps around the `isaacsim.asset.importer.mjcf`_ extension to provide a lazy implementation
    for MJCF to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    .. caution::
        The current lazy conversion implementation does not automatically trigger USD generation if
        only the mesh files used by the MJCF are modified. To force generation, either set
        :obj:`AssetConverterBaseCfg.force_usd_conversion` to True or delete the output directory.

    .. note::
        From Isaac Sim 4.5 onwards, the extension name changed from ``omni.importer.mjcf`` to
        ``isaacsim.asset.importer.mjcf``. This converter class now uses the latest extension from Isaac Sim.

    .. _isaacsim.asset.importer.mjcf:  https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_mjcf.html
    """
    """转换MJCF描述文件为USD文件。

    这个类包围`isaacsim.asset.importer.mjcf`_扩展以提供惰的实现
    for MJCF to USD conversion. It stores the output USD file in an instanceable format since that is
    在所有学习相关应用中通常使用。

    .. 谨慎::
        如果只修改MJCF所使用的网格文件，目前的惰转换实现不会自动触发USD生成。
        要强制生成，要么设置:obj:`AssetConverterBaseCfg.force_usd_conversion`为True，要么删除输出目录。

    .. 说明::
        从Isaac Sim4.5开始，扩展名称从``omni.importer.mjcf``变为``isaacsim.asset.importer.mjcf``。
        这类转换器现在使用了Isaac Sim最新的扩展。

    .. _isaacsim.asset.importer.mjcf:  https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_mjcf.html
    """

    cfg: MjcfConverterCfg
    """The configuration instance for MJCF to USD conversion."""
    """为MJCF转换到USD的配置实例。"""

    def __init__(self, cfg: MjcfConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for URDF to USD conversion.
        """
        """开始课程。

        参数：
            cfg: 为URDF转换到USD的配置实例。
        """
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """
    """具体实施方法。
    """

    def _convert_asset(self, cfg: MjcfConverterCfg):
        """Calls underlying Omniverse command to convert MJCF to USD.

        Args:
            cfg: The configuration instance for MJCF to USD conversion.
        """
        """调用底层的全宇宙命令将MJCF转换为USD。

        参数：
            cfg: 为MJCF转换到USD的配置实例。
        """
        import_config = self._get_mjcf_import_config()
        file_basename, _ = os.path.basename(cfg.asset_path).split(".")
        omni.kit.commands.execute(
            "MJCFCreateAsset",
            mjcf_path=cfg.asset_path,
            import_config=import_config,
            dest_path=self.usd_path,
            prim_path=f"/{file_basename}",
        )

    def _get_mjcf_import_config(self) -> isaacsim.asset.importer.mjcf._mjcf.ImportConfig:
        """Returns the import configuration for MJCF to USD conversion.

        Returns:
            The constructed ``ImportConfig`` object containing the desired settings.
        """
        """返回MJCF到USD转换的进口配置。

        返回：
            包含所需设置的构建``ImportConfig``对象。
        """

        _, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")

        # set the unit scaling factor, 1.0 means meters, 100.0 means cm
        # import_config.set_distance_scale(1.0)
        # set imported robot as default prim
        # import_config.set_make_default_prim(True)
        # add a physics scene to the stage on import if none exists
        # import_config.set_create_physics_scene(False)
        # set flag to parse <site> tag
        import_config.set_import_sites(True)

        # -- instancing settings
        # meshes will be placed in a separate usd file
        import_config.set_make_instanceable(self.cfg.make_instanceable)
        import_config.set_instanceable_usd_path(self.usd_instanceable_meshes_path)

        # -- asset settings
        # default density used for links, use 0 to auto-compute
        import_config.set_density(self.cfg.link_density)
        # import inertia tensor from urdf, if it is not specified in urdf it will import as identity
        import_config.set_import_inertia_tensor(self.cfg.import_inertia_tensor)

        # -- physics settings
        # create fix joint for base link
        import_config.set_fix_base(self.cfg.fix_base)
        # self collisions between links in the articulation
        import_config.set_self_collision(self.cfg.self_collision)

        return import_config
