# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class AssetConverterBaseCfg:
    """The base configuration class for asset converters."""
    """资产转换器的基本配置类。"""

    asset_path: str = MISSING
    """The absolute path to the asset file to convert into USD."""
    """转换为USD的资产文件的绝对路径。"""

    usd_dir: str | None = None
    """The output directory path to store the generated USD file. Defaults to None.

    If None, it is resolved as ``/tmp/IsaacLab/usd_{date}_{time}_{random}``, where
    the parameters in braces are runtime generated.
    """
    """输出目录路径存储生成的USD文件。
    默认为 None。

    如果None，则被解决为``/tmp/IsaacLab/usd_{date}_{time}_{random}``，在支中的参数生成运行时间。
    """

    usd_file_name: str | None = None
    """The name of the generated usd file. Defaults to None.

    If None, it is resolved from the asset file name. For example, if the asset file
    name is ``"my_asset.urdf"``, then the generated USD file name is ``"my_asset.usd"``.

    If the providing file name does not end with ".usd" or ".usda", then the extension
    ".usd" is appended to the file name.
    """
    """生成的USD文件的名称。
    默认为 None。

    如果 None，则从资产文件名称中解决。
    例如，如果资产文件名称是``"my_asset.urdf"``，则生成的USD文件名称是``"my_asset.usd"``。

    如果提供文件名没有以 ".usd"或 ".usda"结束，则扩展 ".usd"将添加到文件名。
    """

    force_usd_conversion: bool = False
    """Force the conversion of the asset file to usd. Defaults to False.

    If True, then the USD file is always generated. It will overwrite the existing USD file if it exists.
    """
    """强制将资产文件转换为美元。
    默认为 False。

    如果是True，则总是生成USD文件。
    如果存在，它将重写现有USD文件。
    """

    make_instanceable: bool = True
    """Make the generated USD file instanceable. Defaults to True.

    Note:
        Instancing helps reduce the memory footprint of the asset when multiple copies of the asset are
        used in the scene. For more information, please check the USD documentation on
        `scene-graph instancing <https://openusd.org/dev/api/_usd__page__scenegraph_instancing.html>`_.
    """
    """让生成的USD文件可以实例化。
    默认为 True。

    说明：
        当场景中使用多个副本时，即时化帮助减少资产的内存足迹。
        更多信息请查看USD关于`scene-graph instancing
        <https://openusd.org/dev/api/_usd__page__scenegraph_instancing.html>`_。
    """
