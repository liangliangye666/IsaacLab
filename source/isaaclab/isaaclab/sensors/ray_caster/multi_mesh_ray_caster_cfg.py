# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the ray-cast sensor."""
"""射线传感器的配置。"""

from dataclasses import MISSING

from isaaclab.utils import configclass

from .multi_mesh_ray_caster import MultiMeshRayCaster
from .ray_caster_cfg import RayCasterCfg


@configclass
class MultiMeshRayCasterCfg(RayCasterCfg):
    """Configuration for the multi-mesh ray-cast sensor."""
    """多网射线传感器的配置。"""

    @configclass
    class RaycastTargetCfg:
        """Configuration for different ray-cast targets."""
        """为不同的射线目标配置。"""

        prim_expr: str = MISSING
        """The regex to specify the target prim to ray cast against."""
        """目标prim的射线。"""

        is_shared: bool = False
        """Whether the target prim is assumed to be the same mesh across all environments. Defaults to False.

        If True, only the first mesh is read and then reused for all environments, rather than re-parsed.
        This provides a startup performance boost when there are many environments that all use the same asset.

        .. note::
            If :attr:`MultiMeshRayCasterCfg.reference_meshes` is False, this flag has no effect.
        """
        """目标prim在所有环境中是否被认为是相同的网格。
        默认为 False。

        如果True，只有第一个网格才能读取，然后再用于所有环境，而不是重新解析。
        在许多环境中，所有使用相同的资产时，

        .. 说明::
            If :吸引:`MultiMeshRayCasterCfg.reference_meshes`是False这旗子没有效果。
        """

        merge_prim_meshes: bool = True
        """Whether to merge the parsed meshes for a prim that contains multiple meshes. Defaults to True.

        This will create a new mesh that combines all meshes in the parsed prim. The raycast hits mesh IDs
        will then refer to the single merged mesh.
        """
        """是否将解析的网格合并为包含多个网格的prim。
        默认为 True。

        这将创建一个新的网格， 将分析的prim中的所有网格结合在一起。
        射线射击网格IDs然后将引用单个合并网格。
        """

        track_mesh_transforms: bool = True
        """Whether the mesh transformations should be tracked. Defaults to True.

        .. note::
            Not tracking the mesh transformations is recommended when the meshes are static to increase performance.
        """
        """是否应该跟踪网格变化。
        默认为 True。

        .. 说明::
            如果网格是静态的，建议不要跟踪网格转换。
        """

    class_type: type = MultiMeshRayCaster

    mesh_prim_paths: list[str | RaycastTargetCfg] = MISSING
    """The list of mesh primitive paths to ray cast against.

    If an entry is a string, it is internally converted to :class:`RaycastTargetCfg` with
    :attr:`~RaycastTargetCfg.track_mesh_transforms` disabled. These settings ensure backwards compatibility
    with the default raycaster.
    """
    """列表对射线的原始网路。

    如果输入是一个字符串，则内部将其转换为:class:`RaycastTargetCfg`，并且已禁用:attr:`~RaycastTargetCfg.track_mesh_transforms`。
    这些设置确保了反向兼容性
    with the default raycaster.
    """

    update_mesh_ids: bool = False
    """Whether to update the mesh ids of the ray hits in the :attr:`data` container."""
    """在:attr:`data`容器中是否更新射线撞击的网格ID。"""

    reference_meshes: bool = True
    """Whether to reference duplicated meshes instead of loading each one separately into memory.
    Defaults to True.

    When enabled, the raycaster parses all meshes in all environments, but reuses references
    for duplicates instead of storing multiple copies. This reduces memory footprint.
    """
    """不管是指重复的网格，而不是单独加载到内存中。
    默认为 True。

    在启用时，射线播放器在所有环境中分析所有网格，但重复使用参考
    for duplicates instead of storing multiple copies. This reduces memory footprint.
    """
