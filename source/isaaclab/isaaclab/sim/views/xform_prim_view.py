# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

import carb
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.utils.warp import fabric as fabric_utils

logger = logging.getLogger(__name__)


class XformPrimView:
    """Optimized batched interface for reading and writing transforms of multiple USD prims.

    This class provides efficient batch operations for getting and setting poses (position and orientation)
    of multiple prims at once using torch tensors. It is designed for scenarios where you need to manipulate
    many prims simultaneously, such as in multi-agent simulations or large-scale procedural generation.

    The class supports both world-space and local-space pose operations:

    - **World poses**: Positions and orientations in the global world frame
    - **Local poses**: Positions and orientations relative to each prim's parent

    When Fabric is enabled, the class leverages NVIDIA's Fabric API for GPU-accelerated batch operations:

    - Uses `omni:fabric:worldMatrix` and `omni:fabric:localMatrix` attributes for all Boundable prims
    - Performs batch matrix decomposition/composition using Warp kernels on GPU
    - Achieves performance comparable to Isaac Sim's XFormPrim implementation
    - Works for both physics-enabled and non-physics prims (cameras, meshes, etc.).
      Note: renderers typically consume USD-authored camera transforms.

    .. warning::
        **Fabric requires CUDA**: Fabric is only supported with on CUDA devices.
        Warp's CPU backend for fabric-array writes has known issues, so attempting to use
        Fabric with CPU device (``device="cpu"``) will raise a ValueError at initialization.

    .. note::
        **Fabric Support:**

        When Fabric is enabled, this view ensures prims have the required Fabric hierarchy
        attributes (``omni:fabric:localMatrix`` and ``omni:fabric:worldMatrix``). On first Fabric
        read, USD-authored transforms initialize Fabric state. Fabric writes can optionally
        be mirrored back to USD via :attr:`sync_usd_on_fabric_write`.

        For more information, see the `Fabric Hierarchy documentation`_.

        .. _Fabric Hierarchy documentation: https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/fabric_hierarchy.html

    .. note::
        **Performance Considerations:**

        * Tensor operations are performed on the specified device (CPU/CUDA)
        * USD write operations use ``Sdf.ChangeBlock`` for batched updates
        * Fabric operations use GPU-accelerated Warp kernels for maximum performance
        * For maximum performance, minimize get/set operations within tight loops

    .. note::
        **Transform Requirements:**

        All prims in the view must be Xformable and have standardized transform operations:
        ``[translate, orient, scale]``. Non-standard prims will raise a ValueError during
        initialization if :attr:`validate_xform_ops` is True. Please use the function
        :func:`isaaclab.sim.utils.standardize_xform_ops` to prepare prims before using this view.

    .. warning::
        This class operates at the USD default time code. Any animation or time-sampled data
        will not be affected by write operations. For animated transforms, you need to handle
        time-sampled keyframes separately.
    """
    """优化进行阅读和编写的多个USDXprims转换。

    该类提供了有效的批量操作，以使用火器同时获得和设置多个prims的姿势 (位置和方向)。
    它是为需要同时操纵许多prims的场景而设计的，例如多代理仿真或大规模程序生成。

    这类支持世界空间和本地空间姿势操作:

    - **世界姿势**:全球世界框架中的位置和方向
    - **本地姿势**:对每个prim的母体的位置和方向

    当启用Fabric时，该类将NVIDIA的Fabric API用于GPU加速批次操作:

    - 使用`omni:fabric:worldMatrix`和`omni:fabric:localMatrix`属性用于所有Boundable prims
    - 通过GPU上的Warp核进行批量矩阵分解/组合
    - 与Isaac Sim的XFormPrim实现相比的性能
    - 适用于物理和非物理prims (摄像头，网格等)。
      Note: 渲染器通常使用USD授权的相机转换。

    .. 警告::
        **织物需要CUDA**:织物仅支持CUDA设备。
        华普的CPU后端用于织物阵列写作有已知问题，因此尝试使用CPU设备 (``device="cpu"``) 的 Fabric将在初始化时提升ValueError。

    .. 说明::
        **工厂支持:**

        当启用Fabric时，这种视图确保prims具有所需的Fabric等级属性 (``omni:fabric:localMatrix``和``omni:fabric:worldMatrix``)。
        在第一个Fabric阅读时，USD授权的转换启动Fabric状态。
        布料写字可通过:attr:`sync_usd_on_fabric_write`可反射到USD。

        更多信息请参见`Fabric Hierarchy documentation`_。

        .. _Fabric Hierarchy documentation: https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/fabric_hierarchy.html

    .. 说明::
        **性能考虑因素:**

        * 在指定的设备上进行压操作 (CPU/CUDA)
        * USD写操作使用``Sdf.ChangeBlock``批次更新
        * 织物操作使用GPU加速的Warp核以实现最大性能
        * 为了达到最大性能，在紧密循环中尽量减少取/设置操作

    .. 说明::
        **转换要求:**

        视图中的所有prims都必须是Xformable，并且具有标准化转换操作:``[translate， orient， scale]``。
        如果:attr:`validate_xform_ops`是True，不标准的prims将在初始化过程中提升ValueError。
        请使用 :func:`isaaclab.sim.utils.standardize_xform_ops` 函数来准备 prims 在使用此视图之前。

    .. 警告::
        这类操作在USD默认时间代码。
        任何动画或时间样本数据都不会受到写作操作的影响。
        对于动画转换，你需要单独处理时间样本的键框。
    """

    def __init__(
        self,
        prim_path: str,
        device: str = "cpu",
        validate_xform_ops: bool = True,
        sync_usd_on_fabric_write: bool = False,
        stage: Usd.Stage | None = None,
    ):
        """Initialize the view with matching prims.

        This method searches the USD stage for all prims matching the provided path pattern,
        validates that they are Xformable with standard transform operations, and stores
        references for efficient batch operations.

        We generally recommend to validate the xform operations, as it ensures that the prims are in a consistent state
        and have the standard transform operations (translate, orient, scale in that order).
        However, if you are sure that the prims are in a consistent state, you can set this to False to improve
        performance. This can save around 45-50% of the time taken to initialize the view.

        Args:
            prim_path: USD prim path pattern to match prims. Supports wildcards (``*``) and
                regex patterns (e.g., ``"/World/Env_.*/Robot"``). See
                :func:`isaaclab.sim.utils.find_matching_prims` for pattern syntax.
            device: Device to place the tensors on. Can be ``"cpu"`` or CUDA devices like
                ``"cuda:0"``. Defaults to ``"cpu"``.
            validate_xform_ops: Whether to validate that the prims have standard xform operations.
                Defaults to True.
            sync_usd_on_fabric_write: Whether to mirror Fabric transform writes back to USD.
                When True, transform updates are synchronized to USD so that USD data readers (e.g., rendering
                cameras) can observe these changes. Defaults to False for better performance.
            stage: USD stage to search for prims. Defaults to None, in which case the current active stage
                from the simulation context is used.

        Raises:
            ValueError: If any matched prim is not Xformable or doesn't have standardized
                transform operations (translate, orient, scale in that order).
        """
        """通过匹配prims启动视图。

        这种方法搜索USD阶段所有符合提供的路径模式的prims，验证它们是Xformable的标准转换操作，并存储有效批量操作的参考。

        我们一般建议验证xform操作，因为它确保prims处于一致状态，并且具有标准的转换操作 (翻译，定向，按顺序进行扩展)。
        但是，如果你确定prims处于一致状态，你可以设置False以提高性能。
        这可以节省45-50%的视图初始化时间。

        参数：
            prim_path: USD prim路径模式与prims相匹配。
                       支持野生卡 (``*``) 和regex模式 (e.g.，``"/World/Env_.*/Robot"``)。
                       查看:func:`isaaclab.sim.utils.find_matching_prims`的模式语法。
            device: 设置光器的设备。
                    可能是``"cpu"``或CUDA设备，比如``"cuda:0"``。
                    在``"cpu"``上默认。
            validate_xform_ops: 是否验证prims具有标准的x形式操作。
                                默认为 True。
            sync_usd_on_fabric_write: 织物转换是否写回USD。
                                      在True时，转换更新将同步到USD，以便USD数据读者 (e.g.，渲染摄像头) 可以观测这些变化。
                                      设置为False，以提高性能。
            stage: 在USD阶段寻找prims。
                   在 None 时的默认状态，在这种情况下，当前的活跃阶段
                from the simulation context is used.

        异常：
            ValueError: 如果任何匹配的prim不是Xformable或没有标准化转换操作 (翻译，定向，按照这个顺序缩放)。
        """
        # Store configuration
        self._prim_path = prim_path
        self._device = device

        # Find and validate matching prims
        stage = sim_utils.get_current_stage() if stage is None else stage
        self._prims: list[Usd.Prim] = sim_utils.find_matching_prims(prim_path, stage=stage)

        # Validate all prims have standard xform operations
        if validate_xform_ops:
            for prim in self._prims:
                if not sim_utils.validate_standard_xform_ops(prim):
                    raise ValueError(
                        f"Prim at path '{prim.GetPath().pathString}' is not a xformable prim with standard transform"
                        f" operations [translate, orient, scale]. Received type: '{prim.GetTypeName()}'."
                        " Use sim_utils.standardize_xform_ops() to prepare the prim."
                    )

        # Determine if Fabric is supported on the device
        self._use_fabric = carb.settings.get_settings().get("/physics/fabricEnabled")
        logger.debug(f"Using Fabric for the XFormPrimView over '{self._prim_path}' on device '{self._device}'.")

        # Check for unsupported Fabric + CPU combination
        if self._use_fabric and self._device == "cpu":
            logger.warning(
                "Fabric mode with Warp fabric-array operations is not supported on CPU devices. "
                "While Fabric itself can run on both CPU and GPU, our batch Warp kernels for "
                "fabric-array operations require CUDA and are not reliable on the CPU backend. "
                "To ensure stability, Fabric is being disabled and execution will fall back "
                "to standard USD operations on the CPU. This may impact performance."
            )
            self._use_fabric = False

        # Create indices buffer
        # Since we iterate over the indices, we need to use range instead of torch tensor
        self._ALL_INDICES = list(range(len(self._prims)))

        # Some prims (e.g., Cameras) require USD-authored transforms for rendering.
        # When enabled, mirror Fabric pose writes to USD for those prims.
        self._sync_usd_on_fabric_write = sync_usd_on_fabric_write

        # Fabric batch infrastructure (initialized lazily on first use)
        self._fabric_initialized = False
        self._fabric_usd_sync_done = False
        self._fabric_selection = None
        self._fabric_to_view: wp.array | None = None
        self._view_to_fabric: wp.array | None = None
        self._default_view_indices: wp.array | None = None
        self._fabric_hierarchy = None
        # Create a valid USD attribute name: namespace:name
        # Use "isaaclab" namespace to identify our custom attributes
        self._view_index_attr = f"isaaclab:view_index:{abs(hash(self))}"

    """
    Properties.
    """
    """属性。
    """

    @property
    def count(self) -> int:
        """Number of prims in this view."""
        """在这个视图中，prims的数量。"""
        return len(self._prims)

    @property
    def device(self) -> str:
        """Device where tensors are allocated (cpu or cuda)."""
        """配分紧器的装置 (CPU或Cuda)。"""
        return self._device

    @property
    def prims(self) -> list[Usd.Prim]:
        """List of USD prims being managed by this view."""
        """通过此视图管理的USD prims列表。"""
        return self._prims

    @property
    def prim_paths(self) -> list[str]:
        """List of prim paths (as strings) for all prims being managed by this view.

        This property converts each prim to its path string representation. The conversion is
        performed lazily on first access and cached for subsequent accesses.

        Note:
            For most use cases, prefer using :attr:`prims` directly as it provides direct access
            to the USD prim objects without the conversion overhead. This property is mainly useful
            for logging, debugging, or when string paths are explicitly required.
        """
        """所有由此视图管理的prims的prim路径列表 (作为字符串)。

        这种属性将每个prim转换为其路径字符串表示。
        在第一次访问时，转换是缓慢的，并为后续访问进行缓存。

        说明：
            对于大多数使用情况，更喜欢直接使用:attr:`prims`，因为它提供了直接访问USD prim对象，而没有转换通用费用。
            这个特性主要是有用的
            for logging, debugging, or when string paths are explicitly required.
        """
        # we cache it the first time it is accessed.
        # we don't compute it in constructor because it is expensive and we don't need it most of the time.
        # users should usually deal with prims directly as they typically need to access the prims directly.
        if not hasattr(self, "_prim_paths"):
            self._prim_paths = [prim.GetPath().pathString for prim in self._prims]
        return self._prim_paths

    """
    Operations - Setters.
    """
    """运营 - 设置器。
    """

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Set world-space poses for prims in the view.

        This method sets the position and/or orientation of each prim in world space.

        - When Fabric is enabled, the function writes directly to Fabric's ``omni:fabric:worldMatrix``
          attribute using GPU-accelerated batch operations.
        - When Fabric is disabled, the function converts to local space and writes to USD's ``xformOp:translate``
          and ``xformOp:orient`` attributes.

        Args:
            positions: World-space positions as a tensor of shape (M, 3) where M is the number of prims
                to set (either all prims if indices is None, or the number of indices provided).
                Defaults to None, in which case positions are not modified.
            orientations: World-space orientations as quaternions (w, x, y, z) with shape (M, 4).
                Defaults to None, in which case orientations are not modified.
            indices: Indices of prims to set poses for. Defaults to None, in which case poses are set
                for all prims in the view.

        Raises:
            ValueError: If positions shape is not (M, 3) or orientations shape is not (M, 4).
            ValueError: If the number of poses doesn't match the number of indices provided.
        """
        """在视图中设置世界空间姿势为prims。

        这种方法确定了每个prim在世界空间中的位置和/或方向。

        - 当Fabric被启用时，该函数直接使用GPU加速批量操作写入Fabric的``omni:fabric:worldMatrix``属性。
        - 当禁用Fabric时，该函数将转换为本地空间，并写入USD的``xformOp:translate``和``xformOp:orient``属性。

        参数：
            positions: 世界空间作为形状张量 (M，3) 位置，M是设置的prims的数量 (如果索引是None，则所有prims，或者提供的索引数)。
                       在 None 时的默认位置没有修改。
            orientations: 世界空间导向为四角形 (w，x，y，z) 与形状 (M，4)。
                          在 None 中默认设置，在这种情况下，方向没有修改。
            indices: 设置姿势的prims索引。
                     在 None 时的默认设置，此时设置姿势
                for all prims in the view.

        异常：
            ValueError: 如果位置的形状不是 (M， 3) 或方向的形状不是 (M， 4)。
            ValueError: 如果姿势的数量不匹配给出的索引。
        """
        if self._use_fabric:
            self._set_world_poses_fabric(positions, orientations, indices)
        else:
            self._set_world_poses_usd(positions, orientations, indices)

    def set_local_poses(
        self,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Set local-space poses for prims in the view.

        This method sets the position and/or orientation of each prim in local space (relative to
        their parent prims).

        The function writes directly to USD's ``xformOp:translate`` and ``xformOp:orient`` attributes.

        Note:
            Even in Fabric mode, local pose operations use USD. This behavior is based on Isaac Sim's design
            where Fabric is only used for world pose operations.

            Rationale:
                - Local pose writes need correct parent-child hierarchy relationships
                - USD maintains these relationships correctly and efficiently
                - Fabric is optimized for world pose operations, not local hierarchies

        Args:
            translations: Local-space translations as a tensor of shape (M, 3) where M is the number of prims
                to set (either all prims if indices is None, or the number of indices provided).
                Defaults to None, in which case translations are not modified.
            orientations: Local-space orientations as quaternions (w, x, y, z) with shape (M, 4).
                Defaults to None, in which case orientations are not modified.
            indices: Indices of prims to set poses for. Defaults to None, in which case poses are set
                for all prims in the view.

        Raises:
            ValueError: If translations shape is not (M, 3) or orientations shape is not (M, 4).
            ValueError: If the number of poses doesn't match the number of indices provided.
        """
        """在视图中设置prims的本地空间姿势。

        这种方法设定每个prim在本地空间中的位置和/或方向 (相对于其母prims)。

        该函数直接写到USD的``xformOp:translate``和``xformOp:orient``属性。

        说明：
            即使在织模式下，当地的姿势操作使用USD。
            这种行为是基于艾萨克·西姆的设计，

            Rationale:
                - 当地姿势写需要正确的父母-孩子等级关系
                - USD正确有效地保持这些关系
                - 布料被优化为世界姿势操作，而不是当地层次

        参数：
            translations: 局域转换为形状张量 (M，3) ，其中M是设置的prims的数量 (如果索引是None，则所有prims，或者提供的索引数)。
                          在None中默认，在这种情况下，翻译没有修改。
            orientations: 地方空间定向为四元数 (w，x，y，z) 形状 (M，4)。
                          在 None 中默认设置，在这种情况下，方向没有修改。
            indices: 设置姿势的prims索引。
                     在 None 时的默认设置，此时设置姿势
                for all prims in the view.

        异常：
            ValueError: 如果翻译的形状不是 (M， 3) 或方向的形状不是 (M， 4)。
            ValueError: 如果姿势的数量不匹配给出的索引。
        """
        if self._use_fabric:
            self._set_local_poses_fabric(translations, orientations, indices)
        else:
            self._set_local_poses_usd(translations, orientations, indices)

    def set_scales(self, scales: torch.Tensor, indices: Sequence[int] | None = None):
        """Set scales for prims in the view.

        This method sets the scale of each prim in the view.

        - When Fabric is enabled, the function updates scales in Fabric matrices using GPU-accelerated batch operations.
        - When Fabric is disabled, the function writes to USD's ``xformOp:scale`` attributes.

        Args:
            scales: Scales as a tensor of shape (M, 3) where M is the number of prims
                to set (either all prims if indices is None, or the number of indices provided).
            indices: Indices of prims to set scales for. Defaults to None, in which case scales are set
                for all prims in the view.

        Raises:
            ValueError: If scales shape is not (M, 3).
        """
        """在视图中设置prims的尺度。

        这种方法设置了视图中的每个prim的尺度。

        - 当启用Fabric时，该函数会使用GPU加速批次操作更新Fabric矩阵中的规模。
        - 当 Fabric 被禁用时，该函数会写到USD的``xformOp:scale``属性。

        参数：
            scales: 尺度作为形状张量 (M，3) ，其中M是设置的prims的数量 (如果索引是None，则所有prims，或者提供的索引数)。
            indices: 设定尺度的prims索引。
                     默认对 None的设置，此时设置了秤
                for all prims in the view.

        异常：
            ValueError: 如果尺度的形状不 (M， 3)。
        """
        if self._use_fabric:
            self._set_scales_fabric(scales, indices)
        else:
            self._set_scales_usd(scales, indices)

    def set_visibility(self, visibility: torch.Tensor, indices: Sequence[int] | None = None):
        """Set visibility for prims in the view.

        This method sets the visibility of each prim in the view.

        Args:
            visibility: Visibility as a boolean tensor of shape (M,) where M is the
                number of prims to set (either all prims if indices is None, or the number of indices provided).
            indices: Indices of prims to set visibility for. Defaults to None, in which case visibility is set
                for all prims in the view.

        Raises:
            ValueError: If visibility shape is not (M,).
        """
        """在视图中设置prims的可见性。

        这种方法设定了视图中的每个prim的可见性。

        参数：
            visibility: 可视性作为形状的布尔式子 (M，) 在M是设置的prims的数量 (如果索引是None，则所有prims，或者提供的索引数)。
            indices: 设置可见度的prims索引。
                     默认对 None的设置，在这种情况下设置可见性
                for all prims in the view.

        异常：
            ValueError: 如果可见性形状不是 (M，)。
        """
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Validate inputs
        if visibility.shape != (len(indices_list),):
            raise ValueError(f"Expected visibility shape ({len(indices_list)},), got {visibility.shape}.")

        # Set visibility for each prim
        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                # Convert prim to imageable
                imageable = UsdGeom.Imageable(self._prims[prim_idx])
                # Set visibility
                if visibility[idx]:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()

    """
    Operations - Getters.
    """
    """运营 - 盖特斯。
    """

    def get_world_poses(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get world-space poses for prims in the view.

        This method retrieves the position and orientation of each prim in world space by computing
        the full transform hierarchy from the prim to the world root.

        - When Fabric is enabled, the function uses Fabric batch operations with Warp kernels.
        - When Fabric is disabled, the function uses USD XformCache.

        Note:
            Scale and skew are ignored. The returned poses contain only translation and rotation.

        Args:
            indices: Indices of prims to get poses for. Defaults to None, in which case poses are retrieved
                for all prims in the view.

        Returns:
            A tuple of (positions, orientations) where:

            - positions: Torch tensor of shape (M, 3) containing world-space positions (x, y, z),
              where M is the number of prims queried.
            - orientations: Torch tensor of shape (M, 4) containing world-space quaternions (w, x, y, z)
        """
        """让我们看到prims的世界空间姿势。

        这种方法通过计算从prim到世界根的全部转换等级来检索每个prim的位置和方向。

        - 当Fabric启用时，该函数使用Warp内核的Fabric批量操作。
        - 当Tubric被禁用时，该函数使用USD XformCache。

        说明：
            度和偏差被忽视。
            返回的姿势只包含翻译和旋转。

        参数：
            indices: 得到prims的索引来做姿势。
                     在 None 时的默认状态，在这种情况下，取回姿势
                for all prims in the view.

        返回：
            一个 (位置，方向) 的元组，其中:

            - 位置:含有世界空间位置 (x，y，z) 的形状焦点子 (M，3) ，其中M是查询的prims数。
            - 导向:含有世界空间四元数 (w， x， y， z) 的形状火 (M，4)
        """
        if self._use_fabric:
            return self._get_world_poses_fabric(indices)
        else:
            return self._get_world_poses_usd(indices)

    def get_local_poses(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get local-space poses for prims in the view.

        This method retrieves the position and orientation of each prim in local space (relative to
        their parent prims). It reads directly from USD's ``xformOp:translate`` and ``xformOp:orient`` attributes.

        Note:
            Even in Fabric mode, local pose operations use USD. This behavior is based on Isaac Sim's design
            where Fabric is only used for world pose operations.

            Rationale:
                - Local pose reads need correct parent-child hierarchy relationships
                - USD maintains these relationships correctly and efficiently
                - Fabric is optimized for world pose operations, not local hierarchies

        Note:
            Scale is ignored. The returned poses contain only translation and rotation.

        Args:
            indices: Indices of prims to get poses for. Defaults to None, in which case poses are retrieved
                for all prims in the view.

        Returns:
            A tuple of (translations, orientations) where:

            - translations: Torch tensor of shape (M, 3) containing local-space translations (x, y, z),
              where M is the number of prims queried.
            - orientations: Torch tensor of shape (M, 4) containing local-space quaternions (w, x, y, z)
        """
        """让prims在视野中做地方空间姿势。

        这种方法检索每个prim在本地空间中的位置和方向 (相对于其母prims)。
        它直接来自USD的``xformOp:translate``和``xformOp:orient``属性。

        说明：
            即使在织模式下，当地的姿势操作使用USD。
            这种行为是基于艾萨克·西姆的设计，

            Rationale:
                - 当地姿势阅读需要正确的父母-孩子等级关系
                - USD正确有效地保持这些关系
                - 布料被优化为世界姿势操作，而不是当地层次

        说明：
            规模被忽视。
            返回的姿势只包含翻译和旋转。

        参数：
            indices: 得到prims的索引来做姿势。
                     在 None 时的默认状态，在这种情况下，取回姿势
                for all prims in the view.

        返回：
            一个 (翻译，方向) 的元组，其中:

            - 翻译:含有本地空间翻译 (x，y，z) 的形状火 (M，3) ，其中M是查询的prims数。
            - 导向:含有本地空间四元数 (w，x，y，z) 的形状火 (M，4)
        """
        if self._use_fabric:
            return self._get_local_poses_fabric(indices)
        else:
            return self._get_local_poses_usd(indices)

    def get_scales(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """Get scales for prims in the view.

        This method retrieves the scale of each prim in the view.

        - When Fabric is enabled, the function extracts scales from Fabric matrices using batch operations with
          Warp kernels.
        - When Fabric is disabled, the function reads from USD's ``xformOp:scale`` attributes.

        Args:
            indices: Indices of prims to get scales for. Defaults to None, in which case scales are retrieved
                for all prims in the view.

        Returns:
            A tensor of shape (M, 3) containing the scales of each prim, where M is the number of prims queried.
        """
        """在视图中得到prims的尺度。

        这种方法检索视图中的每个prim的规模。

        - 当Fabric启用时，该函数将使用Warp核进行批量操作，从Fabric矩阵中提取规模。
        - 当禁用Fabric时，该函数从USD的``xformOp:scale``属性中读取。

        参数：
            indices: 的索引prims为了获得秤。
                     在 None 中的默认值，此时取取了秤
                for all prims in the view.

        返回：
            包含每个prim的尺度的形状张量 (M，3) ，其中M是查询的prims数。
        """
        if self._use_fabric:
            return self._get_scales_fabric(indices)
        else:
            return self._get_scales_usd(indices)

    def get_visibility(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """Get visibility for prims in the view.

        This method retrieves the visibility of each prim in the view.

        Args:
            indices: Indices of prims to get visibility for. Defaults to None, in which case visibility is retrieved
                for all prims in the view.

        Returns:
            A tensor of shape (M,) containing the visibility of each prim, where M is the number of prims queried.
            The tensor is of type bool.
        """
        """让prims在视野中看到。

        这种方法检索视图中的每个prim的可见性。

        参数：
            indices: 的索引prims为了获得可见性。
                     在 None 时的默认设置，此情况下可视性被检索
                for all prims in the view.

        返回：
            包含每个prim的可见性 (M，) 的形状数，其中M是查询的prims数。
            子是 bool类型的。
        """
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            # Convert to list if it is a tensor array
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Create buffers
        visibility = torch.zeros(len(indices_list), dtype=torch.bool, device=self._device)

        for idx, prim_idx in enumerate(indices_list):
            # Get prim
            imageable = UsdGeom.Imageable(self._prims[prim_idx])
            # Get visibility
            visibility[idx] = imageable.ComputeVisibility() != UsdGeom.Tokens.invisible

        return visibility

    """
    Internal Functions - USD.
    """
    """内部功能 - USD
    """

    def _set_world_poses_usd(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Set world poses to USD."""
        """设置世界姿势为USD。"""
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            # Convert to list if it is a tensor array
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Validate inputs
        if positions is not None:
            if positions.shape != (len(indices_list), 3):
                raise ValueError(
                    f"Expected positions shape ({len(indices_list)}, 3), got {positions.shape}. "
                    "Number of positions must match the number of prims in the view."
                )
            positions_array = Vt.Vec3dArray.FromNumpy(positions.cpu().numpy())
        else:
            positions_array = None
        if orientations is not None:
            if orientations.shape != (len(indices_list), 4):
                raise ValueError(
                    f"Expected orientations shape ({len(indices_list)}, 4), got {orientations.shape}. "
                    "Number of orientations must match the number of prims in the view."
                )
            # Vt expects quaternions in xyzw order
            orientations_array = Vt.QuatdArray.FromNumpy(math_utils.convert_quat(orientations, to="xyzw").cpu().numpy())
        else:
            orientations_array = None

        # Create xform cache instance
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        # Set poses for each prim
        # We use Sdf.ChangeBlock to minimize notification overhead.
        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                # Get prim
                prim = self._prims[prim_idx]
                # Get parent prim for local space conversion
                parent_prim = prim.GetParent()

                # Determine what to set
                world_pos = positions_array[idx] if positions_array is not None else None
                world_quat = orientations_array[idx] if orientations_array is not None else None

                # Convert world pose to local if we have a valid parent
                # Note: We don't use :func:`isaaclab.sim.utils.transforms.convert_world_pose_to_local`
                #   here since it isn't optimized for batch operations.
                if parent_prim.IsValid() and parent_prim.GetPath() != Sdf.Path.absoluteRootPath:
                    # Get current world pose if we're only setting one component
                    if positions_array is None or orientations_array is None:
                        # get prim xform
                        prim_tf = xform_cache.GetLocalToWorldTransform(prim)
                        # sanitize quaternion
                        # this is needed, otherwise the quaternion might be non-normalized
                        prim_tf.Orthonormalize()
                        # populate desired world transform
                        if world_pos is not None:
                            prim_tf.SetTranslateOnly(world_pos)
                        if world_quat is not None:
                            prim_tf.SetRotateOnly(world_quat)
                    else:
                        # Both position and orientation are provided, create new transform
                        prim_tf = Gf.Matrix4d()
                        prim_tf.SetTranslateOnly(world_pos)
                        prim_tf.SetRotateOnly(world_quat)

                    # Convert to local space
                    parent_world_tf = xform_cache.GetLocalToWorldTransform(parent_prim)
                    local_tf = prim_tf * parent_world_tf.GetInverse()
                    local_pos = local_tf.ExtractTranslation()
                    local_quat = local_tf.ExtractRotationQuat()
                else:
                    # No parent or parent is root, world == local
                    local_pos = world_pos
                    local_quat = world_quat

                # Get or create the standard transform operations
                if local_pos is not None:
                    prim.GetAttribute("xformOp:translate").Set(local_pos)
                if local_quat is not None:
                    prim.GetAttribute("xformOp:orient").Set(local_quat)

    def _set_local_poses_usd(
        self,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Set local poses to USD."""
        """设置本地姿势为USD。"""
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Validate inputs
        if translations is not None:
            if translations.shape != (len(indices_list), 3):
                raise ValueError(f"Expected translations shape ({len(indices_list)}, 3), got {translations.shape}.")
            translations_array = Vt.Vec3dArray.FromNumpy(translations.cpu().numpy())
        else:
            translations_array = None
        if orientations is not None:
            if orientations.shape != (len(indices_list), 4):
                raise ValueError(f"Expected orientations shape ({len(indices_list)}, 4), got {orientations.shape}.")
            orientations_array = Vt.QuatdArray.FromNumpy(math_utils.convert_quat(orientations, to="xyzw").cpu().numpy())
        else:
            orientations_array = None

        # Set local poses
        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                prim = self._prims[prim_idx]
                if translations_array is not None:
                    prim.GetAttribute("xformOp:translate").Set(translations_array[idx])
                if orientations_array is not None:
                    prim.GetAttribute("xformOp:orient").Set(orientations_array[idx])

    def _set_scales_usd(self, scales: torch.Tensor, indices: Sequence[int] | None = None):
        """Set scales to USD."""
        """设置尺度为USD。"""
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Validate inputs
        if scales.shape != (len(indices_list), 3):
            raise ValueError(f"Expected scales shape ({len(indices_list)}, 3), got {scales.shape}.")

        scales_array = Vt.Vec3dArray.FromNumpy(scales.cpu().numpy())
        # Set scales for each prim
        with Sdf.ChangeBlock():
            for idx, prim_idx in enumerate(indices_list):
                prim = self._prims[prim_idx]
                prim.GetAttribute("xformOp:scale").Set(scales_array[idx])

    def _get_world_poses_usd(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get world poses from USD."""
        """从USD那里得到世界姿势。"""
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            # Convert to list if it is a tensor array
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Create buffers
        positions = Vt.Vec3dArray(len(indices_list))
        orientations = Vt.QuatdArray(len(indices_list))
        # Create xform cache instance
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        # Note: We don't use :func:`isaaclab.sim.utils.transforms.resolve_prim_pose`
        #   here since it isn't optimized for batch operations.
        for idx, prim_idx in enumerate(indices_list):
            # Get prim
            prim = self._prims[prim_idx]
            # get prim xform
            prim_tf = xform_cache.GetLocalToWorldTransform(prim)
            # sanitize quaternion
            # this is needed, otherwise the quaternion might be non-normalized
            prim_tf.Orthonormalize()
            # extract position and orientation
            positions[idx] = prim_tf.ExtractTranslation()
            orientations[idx] = prim_tf.ExtractRotationQuat()

        # move to torch tensors
        positions = torch.tensor(np.array(positions), dtype=torch.float32, device=self._device)
        orientations = torch.tensor(np.array(orientations), dtype=torch.float32, device=self._device)
        # underlying data is in xyzw order, convert to wxyz order
        orientations = math_utils.convert_quat(orientations, to="wxyz")

        return positions, orientations  # type: ignore

    def _get_local_poses_usd(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get local poses from USD."""
        """从USD那里得到当地姿势。"""
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Create buffers
        translations = Vt.Vec3dArray(len(indices_list))
        orientations = Vt.QuatdArray(len(indices_list))

        # Create a fresh XformCache to avoid stale cached values
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        for idx, prim_idx in enumerate(indices_list):
            prim = self._prims[prim_idx]
            prim_tf = xform_cache.GetLocalTransformation(prim)[0]
            prim_tf.Orthonormalize()
            translations[idx] = prim_tf.ExtractTranslation()
            orientations[idx] = prim_tf.ExtractRotationQuat()

        translations = torch.tensor(np.array(translations), dtype=torch.float32, device=self._device)
        orientations = torch.tensor(np.array(orientations), dtype=torch.float32, device=self._device)
        orientations = math_utils.convert_quat(orientations, to="wxyz")

        return translations, orientations  # type: ignore

    def _get_scales_usd(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """Get scales from USD."""
        """拿出USD的秤。"""
        # Resolve indices
        if indices is None or indices == slice(None):
            indices_list = self._ALL_INDICES
        else:
            indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

        # Create buffers
        scales = Vt.Vec3dArray(len(indices_list))

        for idx, prim_idx in enumerate(indices_list):
            prim = self._prims[prim_idx]
            scales[idx] = prim.GetAttribute("xformOp:scale").Get()

        # Convert to tensor
        return torch.tensor(np.array(scales), dtype=torch.float32, device=self._device)

    """
    Internal Functions - Fabric.
    """
    """内部功能 - 织物
    """

    def _set_world_poses_fabric(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Set world poses using Fabric GPU batch operations.

        Writes directly to Fabric's ``omni:fabric:worldMatrix`` attribute using Warp kernels.
        Changes are propagated through Fabric's hierarchy system but remain GPU-resident.

        For workflows mixing Fabric world pose writes with USD local pose queries, note
        that local poses read from USD's xformOp:* attributes, which may not immediately
        reflect Fabric changes. For best performance and consistency, use Fabric methods
        exclusively (get_world_poses/set_world_poses with Fabric enabled).
        """
        """设置世界姿势，使用"织品GPU"批量操作。

        使用Warp内核直接写入Fabric的``omni:fabric:worldMatrix``属性。
        变化通过Fabric的等级系统传播，但仍然是GPU居民。

        对于混合Fabric世界姿势与USD本地姿势查询的工作流，请注意本地姿势从USD的xformOp:*属性中读取，这可能不会立即反映Fabric的变化。
        为了获得最佳性能和一致性，只使用"织"方法 (使用"织"的get_world_poses/set_world_poses)。
        """
        # Lazy initialization
        if not self._fabric_initialized:
            self._initialize_fabric()

        # Resolve indices (treat slice(None) as None for consistency with USD path)
        indices_wp = self._resolve_indices_wp(indices)

        count = indices_wp.shape[0]

        # Convert torch to warp (if provided), use dummy arrays for None to avoid Warp kernel issues
        if positions is not None:
            positions_wp = wp.from_torch(positions)
        else:
            positions_wp = wp.zeros((0, 3), dtype=wp.float32).to(self._device)

        if orientations is not None:
            orientations_wp = wp.from_torch(orientations)
        else:
            orientations_wp = wp.zeros((0, 4), dtype=wp.float32).to(self._device)

        # Dummy array for scales (not modifying)
        scales_wp = wp.zeros((0, 3), dtype=wp.float32).to(self._device)

        # Use cached fabricarray for world matrices
        world_matrices = self._fabric_world_matrices

        # Batch compose matrices with a single kernel launch
        wp.launch(
            kernel=fabric_utils.compose_fabric_transformation_matrix_from_warp_arrays,
            dim=count,
            inputs=[
                world_matrices,
                positions_wp,
                orientations_wp,
                scales_wp,  # dummy array instead of None
                False,  # broadcast_positions
                False,  # broadcast_orientations
                False,  # broadcast_scales
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._device,
        )

        # Synchronize to ensure kernel completes
        wp.synchronize()

        # Update world transforms within Fabric hierarchy
        self._fabric_hierarchy.update_world_xforms()
        # Fabric now has authoritative data; skip future USD syncs
        self._fabric_usd_sync_done = True
        # Mirror to USD for renderer-facing prims when enabled.
        if self._sync_usd_on_fabric_write:
            self._set_world_poses_usd(positions, orientations, indices)

        # Fabric writes are GPU-resident; local pose operations still use USD.

    def _set_local_poses_fabric(
        self,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ):
        """Set local poses using USD (matches Isaac Sim's design).

        Note: Even in Fabric mode, local pose operations use USD.
        This is Isaac Sim's design: the ``usd=False`` parameter only affects world poses.

        Rationale:
        - Local pose writes need correct parent-child hierarchy relationships
        - USD maintains these relationships correctly and efficiently
        - Fabric is optimized for world pose operations, not local hierarchies
        """
        """使用USD设置本地姿势 (与艾萨克·西姆的设计相匹配)。

        Note: 即使在织模式下，当地的姿势操作使用USD。
        这就是Isaac Sim的设计:``usd=False``参数只影响世界姿势。

        Rationale:
        - 当地姿势写需要正确的父母-孩子等级关系
        - USD正确有效地保持这些关系
        - 布料被优化为世界姿势操作，而不是当地层次
        """
        self._set_local_poses_usd(translations, orientations, indices)

    def _set_scales_fabric(self, scales: torch.Tensor, indices: Sequence[int] | None = None):
        """Set scales using Fabric GPU batch operations."""
        """使用"织品GPU"批量操作设置秤。"""
        # Lazy initialization
        if not self._fabric_initialized:
            self._initialize_fabric()

        # Resolve indices (treat slice(None) as None for consistency with USD path)
        indices_wp = self._resolve_indices_wp(indices)

        count = indices_wp.shape[0]

        # Convert torch to warp
        scales_wp = wp.from_torch(scales)

        # Dummy arrays for positions and orientations (not modifying)
        positions_wp = wp.zeros((0, 3), dtype=wp.float32).to(self._device)
        orientations_wp = wp.zeros((0, 4), dtype=wp.float32).to(self._device)

        # Use cached fabricarray for world matrices
        world_matrices = self._fabric_world_matrices

        # Batch compose matrices on GPU with a single kernel launch
        wp.launch(
            kernel=fabric_utils.compose_fabric_transformation_matrix_from_warp_arrays,
            dim=count,
            inputs=[
                world_matrices,
                positions_wp,  # dummy array instead of None
                orientations_wp,  # dummy array instead of None
                scales_wp,
                False,  # broadcast_positions
                False,  # broadcast_orientations
                False,  # broadcast_scales
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._device,
        )

        # Synchronize to ensure kernel completes before syncing
        wp.synchronize()

        # Update world transforms to propagate changes
        self._fabric_hierarchy.update_world_xforms()
        # Fabric now has authoritative data; skip future USD syncs
        self._fabric_usd_sync_done = True
        # Mirror to USD for renderer-facing prims when enabled.
        if self._sync_usd_on_fabric_write:
            self._set_scales_usd(scales, indices)

    def _get_world_poses_fabric(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get world poses from Fabric using GPU batch operations."""
        """通过GPU批次操作从织品中获取世界姿势。"""
        # Lazy initialization of Fabric infrastructure
        if not self._fabric_initialized:
            self._initialize_fabric()
        # Sync once from USD to ensure reads see the latest authored transforms
        if not self._fabric_usd_sync_done:
            self._sync_fabric_from_usd_once()

        # Resolve indices (treat slice(None) as None for consistency with USD path)
        indices_wp = self._resolve_indices_wp(indices)

        count = indices_wp.shape[0]

        # Use pre-allocated buffers for full reads, allocate only for partial reads
        use_cached_buffers = indices is None or indices == slice(None)
        if use_cached_buffers:
            # Full read: Use cached buffers (zero allocation overhead!)
            positions_wp = self._fabric_positions_buffer
            orientations_wp = self._fabric_orientations_buffer
            scales_wp = self._fabric_dummy_buffer
        else:
            # Partial read: Need to allocate buffers of appropriate size
            positions_wp = wp.zeros((count, 3), dtype=wp.float32).to(self._device)
            orientations_wp = wp.zeros((count, 4), dtype=wp.float32).to(self._device)
            scales_wp = self._fabric_dummy_buffer  # Always use dummy for scales

        # Use cached fabricarray for world matrices
        # This eliminates the 0.06-0.30ms variability from creating fabricarray each call
        world_matrices = self._fabric_world_matrices

        # Launch GPU kernel to decompose matrices in parallel
        wp.launch(
            kernel=fabric_utils.decompose_fabric_transformation_matrix_to_warp_arrays,
            dim=count,
            inputs=[
                world_matrices,
                positions_wp,
                orientations_wp,
                scales_wp,  # dummy array instead of None
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._device,
        )

        # Return tensors: zero-copy for cached buffers, conversion for partial reads
        if use_cached_buffers:
            # Zero-copy! The Warp kernel wrote directly into the PyTorch tensors
            # We just need to synchronize to ensure the kernel is done
            wp.synchronize()
            return self._fabric_positions_torch, self._fabric_orientations_torch
        else:
            # Partial read: Need to convert from Warp to torch
            positions = wp.to_torch(positions_wp)
            orientations = wp.to_torch(orientations_wp)
            return positions, orientations

    def _get_local_poses_fabric(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get local poses using USD (matches Isaac Sim's design).

        Note:
            Even in Fabric mode, local pose operations use USD's XformCache.
            This is Isaac Sim's design: the ``usd=False`` parameter only affects world poses.

        Rationale:
            - Local pose computation requires parent transforms which may not be in the view
            - USD's XformCache provides efficient hierarchy-aware local transform queries
            - Fabric is optimized for world pose operations, not local hierarchies
        """
        """通过USD来获得本地姿势。

        说明：
            即使在织模式下，当地的姿势操作使用USD的XformCache。
            这就是Isaac Sim的设计:``usd=False``参数只影响世界姿势。

        Rationale:
            - 局部姿势计算需要母体转换，这些转换可能不在视图中
            - USD的XformCache提供了高效的层次意识到本地转换查询
            - 布料被优化为世界姿势操作，而不是当地层次
        """
        return self._get_local_poses_usd(indices)

    def _get_scales_fabric(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """Get scales from Fabric using GPU batch operations."""
        """通过GPU批量操作从织品中获取秤。"""
        # Lazy initialization
        if not self._fabric_initialized:
            self._initialize_fabric()
        # Sync once from USD to ensure reads see the latest authored transforms
        if not self._fabric_usd_sync_done:
            self._sync_fabric_from_usd_once()

        # Resolve indices (treat slice(None) as None for consistency with USD path)
        indices_wp = self._resolve_indices_wp(indices)

        count = indices_wp.shape[0]

        # Use pre-allocated buffers for full reads, allocate only for partial reads
        use_cached_buffers = indices is None or indices == slice(None)
        if use_cached_buffers:
            # Full read: Use cached buffers (zero allocation overhead!)
            scales_wp = self._fabric_scales_buffer
        else:
            # Partial read: Need to allocate buffer of appropriate size
            scales_wp = wp.zeros((count, 3), dtype=wp.float32).to(self._device)

        # Always use dummy buffers for positions and orientations (not needed for scales)
        positions_wp = self._fabric_dummy_buffer
        orientations_wp = self._fabric_dummy_buffer

        # Use cached fabricarray for world matrices
        world_matrices = self._fabric_world_matrices

        # Launch GPU kernel to decompose matrices in parallel
        wp.launch(
            kernel=fabric_utils.decompose_fabric_transformation_matrix_to_warp_arrays,
            dim=count,
            inputs=[
                world_matrices,
                positions_wp,  # dummy array instead of None
                orientations_wp,  # dummy array instead of None
                scales_wp,
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._device,
        )

        # Return tensor: zero-copy for cached buffers, conversion for partial reads
        if use_cached_buffers:
            # Zero-copy! The Warp kernel wrote directly into the PyTorch tensor
            wp.synchronize()
            return self._fabric_scales_torch
        else:
            # Partial read: Need to convert from Warp to torch
            return wp.to_torch(scales_wp)

    """
    Internal Functions - Initialization.
    """
    """内部功能 - 启动。
    """

    def _initialize_fabric(self) -> None:
        """Initialize Fabric batch infrastructure for GPU-accelerated pose queries.

        This method ensures all prims have the required Fabric hierarchy attributes
        (``omni:fabric:localMatrix`` and ``omni:fabric:worldMatrix``) and creates the necessary
        infrastructure for batch GPU operations using Warp.

        Based on the Fabric Hierarchy documentation, when Fabric Scene Delegate is enabled,
        all boundable prims should have these attributes. This method ensures they exist
        and are properly synchronized with USD.
        """
        """启动GPU加速姿势查询的织品批量基础设施。

        这种方法确保所有prims都具有所需的织层次属性
        (``omni:fabric:localMatrix``和``omni:fabric:worldMatrix``)，并为使用Warp的批量GPU操作创建了必要的基础设施。

        根据" Hier布层次结构"文件，在启用"布场景代表"时，所有可划分的prims都应该具有这些属性。
        这种方法确保它们存在并与USD进行适当的同步。
        """
        import usdrt
        from usdrt import Rt

        # Get USDRT (Fabric) stage
        stage_id = sim_utils.get_current_stage_id()
        fabric_stage = usdrt.Usd.Stage.Attach(stage_id)

        # Step 1: Ensure all prims have Fabric hierarchy attributes
        # According to the documentation, these attributes are created automatically
        # when Fabric Scene Delegate is enabled, but we ensure they exist
        for i in range(self.count):
            rt_prim = fabric_stage.GetPrimAtPath(self.prim_paths[i])
            rt_xformable = Rt.Xformable(rt_prim)

            # Create Fabric hierarchy world matrix attribute if it doesn't exist
            has_attr = (
                rt_xformable.HasFabricHierarchyWorldMatrixAttr()
                if hasattr(rt_xformable, "HasFabricHierarchyWorldMatrixAttr")
                else False
            )
            if not has_attr:
                rt_xformable.CreateFabricHierarchyWorldMatrixAttr()

            # Best-effort USD->Fabric sync; authoritative initialization happens on first read.
            rt_xformable.SetWorldXformFromUsd()

            # Create view index attribute for batch operations
            rt_prim.CreateAttribute(self._view_index_attr, usdrt.Sdf.ValueTypeNames.UInt, custom=True)
            rt_prim.GetAttribute(self._view_index_attr).Set(i)

        # After syncing all prims, update the Fabric hierarchy to ensure world matrices are computed
        self._fabric_hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            fabric_stage.GetFabricId(), fabric_stage.GetStageIdAsStageId()
        )
        self._fabric_hierarchy.update_world_xforms()

        # Step 2: Create index arrays for batch operations
        self._default_view_indices = wp.zeros((self.count,), dtype=wp.uint32).to(self._device)
        wp.launch(
            kernel=fabric_utils.arange_k,
            dim=self.count,
            inputs=[self._default_view_indices],
            device=self._device,
        )
        wp.synchronize()  # Ensure indices are ready

        # Step 3: Create Fabric selection with attribute filtering
        # SelectPrims expects device format like "cuda:0" not "cuda"
        #
        # KNOWN ISSUE: SelectPrims may return prims in a different order than self._prims
        # (which comes from USD's find_matching_prims). We create a bidirectional mapping
        # (_view_to_fabric and _fabric_to_view) to handle this ordering difference.
        # This works correctly for full-view operations but partial indexing still has issues.
        fabric_device = self._device
        if self._device == "cuda":
            logger.warning("Fabric device is not specified, defaulting to 'cuda:0'.")
            fabric_device = "cuda:0"

        self._fabric_selection = fabric_stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.UInt, self._view_index_attr, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.ReadWrite),
            ],
            device=fabric_device,
        )

        # Step 4: Create bidirectional mapping between view and fabric indices
        self._view_to_fabric = wp.zeros((self.count,), dtype=wp.uint32).to(self._device)
        self._fabric_to_view = wp.fabricarray(self._fabric_selection, self._view_index_attr)

        wp.launch(
            kernel=fabric_utils.set_view_to_fabric_array,
            dim=self._fabric_to_view.shape[0],
            inputs=[self._fabric_to_view, self._view_to_fabric],
            device=self._device,
        )
        # Synchronize to ensure mapping is ready before any operations
        wp.synchronize()

        # Pre-allocate reusable output buffers for read operations
        self._fabric_positions_torch = torch.zeros((self.count, 3), dtype=torch.float32, device=self._device)
        self._fabric_orientations_torch = torch.zeros((self.count, 4), dtype=torch.float32, device=self._device)
        self._fabric_scales_torch = torch.zeros((self.count, 3), dtype=torch.float32, device=self._device)

        # Create Warp views of the PyTorch tensors
        self._fabric_positions_buffer = wp.from_torch(self._fabric_positions_torch, dtype=wp.float32)
        self._fabric_orientations_buffer = wp.from_torch(self._fabric_orientations_torch, dtype=wp.float32)
        self._fabric_scales_buffer = wp.from_torch(self._fabric_scales_torch, dtype=wp.float32)

        # Dummy array for unused outputs (always empty)
        self._fabric_dummy_buffer = wp.zeros((0, 3), dtype=wp.float32).to(self._device)

        # Cache fabricarray for world matrices to avoid recreation overhead
        # Refs: https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usdrt_prim_selection.html
        #       https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/scenegraph_use.html
        self._fabric_world_matrices = wp.fabricarray(self._fabric_selection, "omni:fabric:worldMatrix")

        # Cache Fabric stage to avoid expensive get_current_stage() calls
        self._fabric_stage = fabric_stage

        self._fabric_initialized = True
        # Force a one-time USD->Fabric sync on first read to pick up any USD edits
        # made after the view was constructed.
        self._fabric_usd_sync_done = False

    def _sync_fabric_from_usd_once(self) -> None:
        """Sync Fabric world matrices from USD once, on the first read."""
        """在第一次阅读时，同步USD的世界矩阵。"""
        # Ensure Fabric is initialized
        if not self._fabric_initialized:
            self._initialize_fabric()

        # Ensure authored USD transforms are flushed before reading into Fabric.
        sim_utils.update_stage()

        # Read authoritative transforms from USD and write once into Fabric.
        positions_usd, orientations_usd = self._get_world_poses_usd()
        scales_usd = self._get_scales_usd()

        prev_sync = self._sync_usd_on_fabric_write
        self._sync_usd_on_fabric_write = False
        self._set_world_poses_fabric(positions_usd, orientations_usd)
        self._set_scales_fabric(scales_usd)
        self._sync_usd_on_fabric_write = prev_sync

        self._fabric_usd_sync_done = True

    def _resolve_indices_wp(self, indices: Sequence[int] | None) -> wp.array:
        """Resolve view indices as a Warp array."""
        """解决视图索引作为一个变形阵列。"""
        if indices is None or indices == slice(None):
            if self._default_view_indices is None:
                raise RuntimeError("Fabric indices are not initialized.")
            return self._default_view_indices
        indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
        return wp.array(indices_list, dtype=wp.uint32).to(self._device)
