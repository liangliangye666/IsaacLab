# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.sim.converters.asset_converter_base_cfg import AssetConverterBaseCfg
from isaaclab.utils import configclass


@configclass
class MjcfConverterCfg(AssetConverterBaseCfg):
    """The configuration class for MjcfConverter."""
    """MjcfConverter的配置类。"""

    link_density = 0.0
    """Default density used for links. Defaults to 0.

    This setting is only effective if ``"inertial"`` properties are missing in the MJCF.
    """
    """在链接中使用的默认密度
    默认为0。

    如果MJCF中缺少``"inertial"``属性，则此设置仅有效。
    """

    import_inertia_tensor: bool = True
    """Import the inertia tensor from mjcf. Defaults to True.

    If the ``"inertial"`` tag is missing, then it is imported as an identity.
    """
    """从 mjcf 进口惯性子。
    默认为 True。

    如果``"inertial"``标签缺失，则将其作为身份进口。
    """

    fix_base: bool = MISSING
    """Create a fix joint to the root/base link. Defaults to True."""
    """创建根/基链接的固定关联。
    默认为 True。
    """

    import_sites: bool = True
    """Import the sites from the MJCF. Defaults to True."""
    """进口来自MJCF的网站。
    默认为 True。
    """

    self_collision: bool = False
    """Activate self-collisions between links of the articulation. Defaults to False."""
    """激活关节链之间的自我碰撞。
    默认为 False。
    """
