# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for working with USD transform (xform) operations.

This module provides utilities for manipulating USD transform operations (xform ops) on prims.
Transform operations in USD define how geometry is positioned, oriented, and scaled in 3D space.

The utilities in this module help standardize transform stacks, clear operations, and manipulate
transforms in a consistent way across different USD assets.
"""

from __future__ import annotations
"""用于处理USD转换 (xform) 操作的工具。

该模块为prims操作USD转换操作 (xform ops) 提供工具。
在USD中转换操作定义了几何在3D空间中如何定位，定向和扩展。

在此模块中的工具帮助标准化转换堆，清晰操作，并以一致的方式对不同USD资产进行转换操作。
"""

import logging

from pxr import Gf, Sdf, Usd, UsdGeom

# import logger
logger = logging.getLogger(__name__)

_INVALID_XFORM_OPS = [
    "xformOp:rotateX",
    "xformOp:rotateXZY",
    "xformOp:rotateY",
    "xformOp:rotateYXZ",
    "xformOp:rotateYZX",
    "xformOp:rotateZ",
    "xformOp:rotateZYX",
    "xformOp:rotateZXY",
    "xformOp:rotateXYZ",
    "xformOp:transform",
]
"""List of invalid xform ops that should be removed."""
"""必须删除的无效的xform操作列表。"""


def standardize_xform_ops(
    prim: Usd.Prim,
    translation: tuple[float, ...] | None = None,
    orientation: tuple[float, ...] | None = None,
    scale: tuple[float, ...] | None = None,
) -> bool:
    """Standardize the transform operation stack on a USD prim to a canonical form.

    This function converts a prim's transform stack to use the standard USD transform operation
    order: [translate, orient, scale]. The function performs the following operations:

    1. Validates that the prim is Xformable
    2. Captures the current local transform (translation, rotation, scale)
    3. Resolves and bakes unit scale conversions (xformOp:scale:unitsResolve)
    4. Creates or reuses standard transform operations (translate, orient, scale)
    5. Sets the transform operation order to [translate, orient, scale]
    6. Applies the preserved or user-specified transform values

    The entire modification is performed within an ``Sdf.ChangeBlock`` for optimal performance
    when processing multiple prims.

    .. note::
        **Standard Transform Order:** The function enforces the USD best practice order:
        ``xformOp:translate``, ``xformOp:orient``, ``xformOp:scale``. This order is
        compatible with most USD tools and workflows, and uses quaternions for rotation
        (avoiding gimbal lock issues).

    .. note::
        **Pose Preservation:** By default, the function preserves the prim's local transform
        (relative to its parent). The world-space position of the prim remains unchanged
        unless explicit ``translation``, ``orientation``, or ``scale`` values are provided.

    .. warning::
        **Animation Data Loss:** This function only preserves transform values at the default
        time code (``Usd.TimeCode.Default()``). Any animation or time-sampled transform data
        will be lost. Use this function during asset import or preparation, not on animated prims.

    .. warning::
        **Unit Scale Resolution:** If the prim has a ``xformOp:scale:unitsResolve`` attribute
        (common in imported assets with unit mismatches), it will be baked into the scale
        and removed. For example, a scale of (1, 1, 1) with unitsResolve of (100, 100, 100)
        becomes a final scale of (100, 100, 100).

    Args:
        prim: The USD prim to standardize. Must be a valid prim that supports the
            UsdGeom.Xformable schema (e.g., Xform, Mesh, Cube, etc.). Material and
            Shader prims are not Xformable and will return False.
        translation: Optional translation vector (x, y, z) in local space. If provided,
            overrides the prim's current translation. If None, preserves the current
            local translation. Defaults to None.
        orientation: Optional orientation quaternion (w, x, y, z) in local space. If provided,
            overrides the prim's current orientation. If None, preserves the current
            local orientation. Defaults to None.
        scale: Optional scale vector (x, y, z). If provided, overrides the prim's current scale.
            If None, preserves the current scale (after unit resolution) or uses (1, 1, 1)
            if no scale exists. Defaults to None.

    Returns:
        bool: True if the transform operations were successfully standardized. False if the
            prim is not Xformable (e.g., Material, Shader prims). The function will log an
            error message when returning False.

    Raises:
        ValueError: If the prim is not valid (i.e., does not exist or is an invalid prim).

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # Standardize a prim with non-standard transform operations
        >>> prim = stage.GetPrimAtPath("/World/ImportedAsset")
        >>> result = sim_utils.standardize_xform_ops(prim)
        >>> if result:
        ...     print("Transform stack standardized successfully")
        >>> # The prim now uses: [translate, orient, scale] in that order
        >>>
        >>> # Standardize and set new transform values
        >>> sim_utils.standardize_xform_ops(
        ...     prim,
        ...     translation=(1.0, 2.0, 3.0),
        ...     orientation=(1.0, 0.0, 0.0, 0.0),  # identity rotation (w, x, y, z)
        ...     scale=(2.0, 2.0, 2.0),
        ... )
        >>>
        >>> # Batch processing for performance
        >>> prims_to_standardize = [stage.GetPrimAtPath(p) for p in prim_paths]
        >>> for prim in prims_to_standardize:
        ...     sim_utils.standardize_xform_ops(prim)  # Each call uses Sdf.ChangeBlock
    """
    """在USDprim上，将转换操作堆标准化成正文形式。

    这个函数将prim的转换堆转换为使用标准USD转换操作
    order: [翻译，方向，规模]。
           该函数执行以下操作:

    1. 验证prim是Xformable的
    2. 捕捉当前的本地转换 (翻译，旋转，规模)
    3. 解决和烤单位规模转换 (xformOp:scale:unitsResolve)
    4. 创建或重复使用标准转换操作 (翻译，定向，规模)
    5. 设置转换操作顺序为 [翻译，方向，规模]
    6. 应用保存或用户指定的转换值

    在处理多个prims时，整个修改都在``Sdf.ChangeBlock``内进行，以实现最佳性能。

    .. 说明::
        **标准转换命令:** 该函数执行USD最佳实践顺序:``xformOp:translate``，``xformOp:orient``，``xformOp:scale``。
        这种顺序与大多数USD工具和工作流兼容，并使用四元数进行旋转 (避免关键锁问题)。

    .. 说明::
        **位置保存:**默认情况下，该函数保留prim的本地转换 (相对于其母体)。
        除非提供明确的``translation``，``orientation``或``scale``值外，prim的世界空间位置保持不变。

    .. 警告::
        **动画数据丢失:** 这个函数仅保留默认时间代码 (``Usd.TimeCode.Default()``) 的转换值。
        任何动画或时间样本转换数据都会丢失。
        在资产进口或准备过程中使用此功能，而不是在动画prims上。

    .. 警告::
        **单位尺度分辨率:** 如果prim具有``xformOp:scale:unitsResolve``属性 (在进口资产中常见的单位不匹配)，则将其入尺度并移除。
        例如，一个 (1， 1， 1) 的尺度，以 (100， 100， 100 的单位Resolve) 成为最终的尺度 (100， 100， 100)。

    参数：
        prim: 在USD prim标准化。
              必须是有效的prim，支持UsdGeom.Xformable方案 (e.g.，Xform， Mesh， Cube等)。
              材料和Shader prims是不 Xformable，将返回False。
        translation: 在本地空间中可选的翻译向量 (x，y，z)。
                     如果提供，则取消prim的当前翻译。
                     如果是None，则保存当前的本地翻译。
                     默认为 None。
        orientation: 在本地空间中可选的导向四角形 (w，x，y，z)。
                     如果提供，则取消prim的当前方向。
                     如果 None，保持当前的本地方向。
                     默认为 None。
        scale: 选择性规模向量 (x，y，z)。
               如果提供，则取消prim的当前规模。
               如果None，则保持当前规模 (单元分辨率后) 或使用 (1， 1， 1)
            if no scale exists. Defaults to None.

    返回：
        bool: True如果转换操作成功标准化。
              False如果prim不是可 Xformable (e.g.， 材料，遮光器prims)。
              函数将在返回False时记录错误信息。

    异常：
        ValueError: 如果prim不有效 (i.e.，不存在或是无效的prim)。

    示例：
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # Standardize a prim with non-standard transform operations
        >>> prim = stage.GetPrimAtPath("/World/ImportedAsset")
        >>> result = sim_utils.standardize_xform_ops(prim)
        >>> if result:
        ...     print("Transform stack standardized successfully")
        >>> # The prim now uses: [translate, orient, scale] in that order
        >>>
        >>> # Standardize and set new transform values
        >>> sim_utils.standardize_xform_ops(
        ...     prim,
        ...     translation=(1.0, 2.0, 3.0),
        ...     orientation=(1.0, 0.0, 0.0, 0.0),  # identity rotation (w, x, y, z)
        ...     scale=(2.0, 2.0, 2.0),
        ... )
        >>>
        >>> # Batch processing for performance
        >>> prims_to_standardize = [stage.GetPrimAtPath(p) for p in prim_paths]
        >>> for prim in prims_to_standardize:
        ...     sim_utils.standardize_xform_ops(prim)  # Each call uses Sdf.ChangeBlock
    """
    # Validate prim
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath()}' is not valid.")

    # Check if prim is an Xformable
    if not prim.IsA(UsdGeom.Xformable):
        logger.error(
            f"Prim at path '{prim.GetPath().pathString}' is of type '{prim.GetTypeName()}', "
            "which is not an Xformable. Transform operations will not be standardized. "
            "This is expected for material, shader, and scope prims."
        )
        return False

    # Create xformable interface
    xformable = UsdGeom.Xformable(prim)
    # Get current property names
    prop_names = prim.GetPropertyNames()

    # Obtain current local transformations
    tf = Gf.Transform(xformable.GetLocalTransformation())
    xform_pos = Gf.Vec3d(tf.GetTranslation())
    xform_quat = Gf.Quatd(tf.GetRotation().GetQuat())
    xform_scale = Gf.Vec3d(tf.GetScale())

    if translation is not None:
        xform_pos = Gf.Vec3d(*translation)
    if orientation is not None:
        xform_quat = Gf.Quatd(*orientation)

    # Handle scale resolution
    if scale is not None:
        # User provided scale
        xform_scale = Gf.Vec3d(scale)
    elif "xformOp:scale" in prop_names:
        # Handle unit resolution for scale if present
        # This occurs when assets are imported with different unit scales
        # Reference: Omniverse Metrics Assembler
        if "xformOp:scale:unitsResolve" in prop_names:
            units_resolve = prim.GetAttribute("xformOp:scale:unitsResolve").Get()
            for i in range(3):
                xform_scale[i] = xform_scale[i] * units_resolve[i]
    else:
        # No scale exists, use default uniform scale
        xform_scale = Gf.Vec3d(1.0, 1.0, 1.0)

    # Verify if xform stack is reset
    has_reset = xformable.GetResetXformStack()
    # Batch the operations
    with Sdf.ChangeBlock():
        # Clear the existing transform operation order
        for prop_name in prop_names:
            if prop_name in _INVALID_XFORM_OPS:
                prim.RemoveProperty(prop_name)

        # Remove unitsResolve attribute if present (already handled in scale resolution above)
        if "xformOp:scale:unitsResolve" in prop_names:
            prim.RemoveProperty("xformOp:scale:unitsResolve")

        # Set up or retrieve scale operation
        xform_op_scale = UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))
        if not xform_op_scale:
            xform_op_scale = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")

        # Set up or retrieve translate operation
        xform_op_translate = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))
        if not xform_op_translate:
            xform_op_translate = xformable.AddXformOp(
                UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, ""
            )

        # Set up or retrieve orient (quaternion rotation) operation
        xform_op_orient = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))
        if not xform_op_orient:
            xform_op_orient = xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")

        # Handle different floating point precisions
        # Existing Xform operations might have floating or double precision.
        # We need to cast the data to the correct type to avoid setting the wrong type.
        xform_ops = [xform_op_translate, xform_op_orient, xform_op_scale]
        xform_values = [xform_pos, xform_quat, xform_scale]
        for xform_op, value in zip(xform_ops, xform_values):
            # Get current value to determine precision type
            current_value = xform_op.Get()
            # Cast to existing type to preserve precision (float/double)
            xform_op.Set(type(current_value)(value) if current_value is not None else value)

        # Set the transform operation order: translate -> orient -> scale
        # This is the standard USD convention and ensures consistent behavior
        xformable.SetXformOpOrder([xform_op_translate, xform_op_orient, xform_op_scale], has_reset)

    return True


def validate_standard_xform_ops(prim: Usd.Prim) -> bool:
    """Validate if the transform operations on a prim are standardized.

    This function checks if the transform operations on a prim are standardized to the canonical form:
    [translate, orient, scale].

    Args:
        prim: The USD prim to validate.
    """
    """如果prim上的转换操作是标准化的，则验证。

    这个函数检查prim上的转换操作是否被标准化到正文形式: [翻译，方向，尺度]。

    参数：
        prim: 验证的USDprim。
    """
    # check if prim is valid
    if not prim.IsValid():
        logger.error(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
        return False
    # check if prim is an xformable
    if not prim.IsA(UsdGeom.Xformable):
        logger.error(f"Prim at path '{prim.GetPath().pathString}' is not an xformable.")
        return False
    # get the xformable interface
    xformable = UsdGeom.Xformable(prim)
    # get the xform operation order
    xform_op_order = xformable.GetOrderedXformOps()
    xform_op_order = [op.GetOpName() for op in xform_op_order]
    # check if the xform operation order is the canonical form
    if xform_op_order != ["xformOp:translate", "xformOp:orient", "xformOp:scale"]:
        msg = f"Xform operation order for prim at path '{prim.GetPath().pathString}' is not the canonical form."
        msg += f" Received order: {xform_op_order}"
        msg += " Expected order: ['xformOp:translate', 'xformOp:orient', 'xformOp:scale']"
        logger.error(msg)
        return False
    return True


def resolve_prim_pose(
    prim: Usd.Prim, ref_prim: Usd.Prim | None = None
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Resolve the pose of a prim with respect to another prim.

    Note:
        This function ignores scale and skew by orthonormalizing the transformation
        matrix at the final step. However, if any ancestor prim in the hierarchy
        has non-uniform scale, that scale will still affect the resulting position
        and orientation of the prim (because it's baked into the transform before
        scale removal).

        In other words: scale **is not removed hierarchically**. If you need
        completely scale-free poses, you must walk the transform chain and strip
        scale at each level. Please open an issue if you need this functionality.

    Args:
        prim: The USD prim to resolve the pose for.
        ref_prim: The USD prim to compute the pose with respect to.
            Defaults to None, in which case the world frame is used.

    Returns:
        A tuple containing the position (as a 3D vector) and the quaternion orientation
        in the (w, x, y, z) format.

    Raises:
        ValueError: If the prim or ref prim is not valid.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>> from pxr import Usd, UsdGeom
        >>>
        >>> # Get prim
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/ImportedAsset")
        >>>
        >>> # Resolve pose
        >>> pos, quat = sim_utils.resolve_prim_pose(prim)
        >>> print(f"Position: {pos}")
        >>> print(f"Orientation: {quat}")
        >>>
        >>> # Resolve pose with respect to another prim
        >>> ref_prim = stage.GetPrimAtPath("/World/Reference")
        >>> pos, quat = sim_utils.resolve_prim_pose(prim, ref_prim)
        >>> print(f"Position: {pos}")
        >>> print(f"Orientation: {quat}")
    """
    """解决一个prim与另一个prim的姿势。

    说明：
        这种函数通过在最后一步实现转换矩阵的正规化，忽略了规模和偏差。
        然而，如果任何prim的祖先在等级中具有非均的尺度，那个尺度仍然会影响prim的结果位置和方向 (因为它在尺度移除之前被烤成变化)。

        换句话说:尺度**没有层次取消**。
        如果你需要完全无尺度的姿势，你必须在每个级别上走过转换链和排放尺度。
        如果您需要该函数，请打开一个问题。

    参数：
        prim: 在USD prim为了解决这个问题。
        ref_prim: 对于USDprim来计算姿势。
                  默认为 None，在这种情况下使用世界框架。

    返回：
        包含位置 (作为3D向量) 和四元数方向的元组 (w， x， y， z) 格式。

    异常：
        ValueError: 如果prim或refprim不有效。

    示例：
        >>> import isaaclab.sim as sim_utils
        >>> from pxr import Usd, UsdGeom
        >>>
        >>> # Get prim
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/ImportedAsset")
        >>>
        >>> # Resolve pose
        >>> pos, quat = sim_utils.resolve_prim_pose(prim)
        >>> print(f"Position: {pos}")
        >>> print(f"Orientation: {quat}")
        >>>
        >>> # Resolve pose with respect to another prim
        >>> ref_prim = stage.GetPrimAtPath("/World/Reference")
        >>> pos, quat = sim_utils.resolve_prim_pose(prim, ref_prim)
        >>> print(f"Position: {pos}")
        >>> print(f"Orientation: {quat}")
    """
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
    # get prim xform
    xform = UsdGeom.Xformable(prim)
    prim_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # sanitize quaternion
    # this is needed, otherwise the quaternion might be non-normalized
    prim_tf.Orthonormalize()

    if ref_prim is not None:
        # if reference prim is the root, we can skip the computation
        if ref_prim.GetPath() != Sdf.Path.absoluteRootPath:
            # get ref prim xform
            ref_xform = UsdGeom.Xformable(ref_prim)
            ref_tf = ref_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            # make sure ref tf is orthonormal
            ref_tf.Orthonormalize()
            # compute relative transform to get prim in ref frame
            prim_tf = prim_tf * ref_tf.GetInverse()

    # extract position and orientation
    prim_pos = [*prim_tf.ExtractTranslation()]
    prim_quat = [prim_tf.ExtractRotationQuat().real, *prim_tf.ExtractRotationQuat().imaginary]
    return tuple(prim_pos), tuple(prim_quat)


def resolve_prim_scale(prim: Usd.Prim) -> tuple[float, float, float]:
    """Resolve the scale of a prim in the world frame.

    At an attribute level, a USD prim's scale is a scaling transformation applied to the prim with
    respect to its parent prim. This function resolves the scale of the prim in the world frame,
    by computing the local to world transform of the prim. This is equivalent to traversing up
    the prim hierarchy and accounting for the rotations and scales of the prims.

    For instance, if a prim has a scale of (1, 2, 3) and it is a child of a prim with a scale of (4, 5, 6),
    then the scale of the prim in the world frame is (4, 10, 18).

    Args:
        prim: The USD prim to resolve the scale for.

    Returns:
        The scale of the prim in the x, y, and z directions in the world frame.

    Raises:
        ValueError: If the prim is not valid.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>> from pxr import Usd, UsdGeom
        >>>
        >>> # Get prim
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/ImportedAsset")
        >>>
        >>> # Resolve scale
        >>> scale = sim_utils.resolve_prim_scale(prim)
        >>> print(f"Scale: {scale}")
    """
    """在世界框架中解决prim的尺度。

    在属性层面上，USD prim的尺度是对prim的扩展变化，与其母prim相比。
    这个函数通过计算prim的局部到世界转换来解决了世界框架中的prim的规模。
    这相当于穿越prim等级，并计算prims的旋转和尺度。

    例如，如果一个prim有一个 (1， 2， 3) 的尺度，它是一个prim的子女，一个尺度为 (4， 5， 6)，那么世界框架中的prim的尺度是 (4， 10， 18)。

    参数：
        prim: 在USD prim为了解决这个问题。

    返回：
        在世界框架中的x，y和z方向中的prim的尺度。

    异常：
        ValueError: 如果prim不有效。

    示例：
        >>> import isaaclab.sim as sim_utils
        >>> from pxr import Usd, UsdGeom
        >>>
        >>> # Get prim
        >>> stage = sim_utils.get_current_stage()
        >>> prim = stage.GetPrimAtPath("/World/ImportedAsset")
        >>>
        >>> # Resolve scale
        >>> scale = sim_utils.resolve_prim_scale(prim)
        >>> print(f"Scale: {scale}")
    """
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim.GetPath().pathString}' is not valid.")
    # compute local to world transform
    xform = UsdGeom.Xformable(prim)
    world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # extract scale
    return tuple([*(v.GetLength() for v in world_transform.ExtractRotationMatrix())])


def convert_world_pose_to_local(
    position: tuple[float, ...],
    orientation: tuple[float, ...] | None,
    ref_prim: Usd.Prim,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float] | None]:
    """Convert a world-space pose to local-space pose relative to a reference prim.

    This function takes a position and orientation in world space and converts them to local space
    relative to the given reference prim. This is useful when creating or positioning prims where you
    know the desired world position but need to set local transform attributes relative to another prim.

    The conversion uses the standard USD transformation math:
    ``local_transform = world_transform * inverse(ref_world_transform)``

    .. note::
        If the reference prim is the root prim ("/"), the position and orientation are returned
        unchanged, as they are already effectively in local/world space.

    Args:
        position: The world-space position as (x, y, z).
        orientation: The world-space orientation as quaternion (w, x, y, z). If None, only position is converted
            and None is returned for orientation.
        ref_prim: The reference USD prim to compute the local transform relative to. If this is
            the root prim ("/"), the world pose is returned unchanged.

    Returns:
        A tuple of (local_translation, local_orientation) where:

        - local_translation is a tuple of (x, y, z) in local space relative to ref_prim
        - local_orientation is a tuple of (w, x, y, z) in local space relative to ref_prim,
          or None if no orientation was provided

    Raises:
        ValueError: If the reference prim is not a valid USD prim.

    Example:
        >>> import isaaclab.sim as sim_utils
        >>> from pxr import Usd, UsdGeom
        >>>
        >>> # Get reference prim
        >>> stage = sim_utils.get_current_stage()
        >>> ref_prim = stage.GetPrimAtPath("/World/Reference")
        >>>
        >>> # Convert world pose to local (relative to ref_prim)
        >>> world_pos = (10.0, 5.0, 0.0)
        >>> world_quat = (1.0, 0.0, 0.0, 0.0)  # identity rotation
        >>> local_pos, local_quat = sim_utils.convert_world_pose_to_local(world_pos, world_quat, ref_prim)
        >>> print(f"Local position: {local_pos}")
        >>> print(f"Local orientation: {local_quat}")
    """
    """将世界空间姿势转换为与参考prim相对的本地空间姿势。

    这种函数在世界空间中占据位置和方向，并将它们转换为与给定的参考prim相比的本地空间。
    这在创建或定位prims时有用，你知道所需的世界位置，但需要与另一个prim相比设置本地转换属性。

    转换使用标准USD转换数学:``local_transform = world_transform * inverse(ref_world_transform)``

    .. 说明::
        如果参考prim是根prim ("/")，位置和方向将保持不变，因为它们已经有效地在本地/世界空间中。

    参数：
        position: 世界空间位置为 (x，y，z)。
        orientation: 作为四元数的世界空间导向 (w，x，y，z)。
                     如果None，只有位置转换，None返回方向。
        ref_prim: 参考USDprim计算相对的本地转换。
                  如果这是根prim ("/")，世界姿势将保持不变。

    返回：
        一个 (local_translation，local_orientation) 的元组，其中:

        - local_translation是局部空间中的 (x，y，z) 的乘法，相对于 ref_prim
        - local_orientation是局部空间中的 (w， x， y， z) 乘以 ref_prim，或 None，如果没有提供指导

    异常：
        ValueError: 如果参考prim不是有效的USD prim。

    示例：
        >>> import isaaclab.sim as sim_utils
        >>> from pxr import Usd, UsdGeom
        >>>
        >>> # Get reference prim
        >>> stage = sim_utils.get_current_stage()
        >>> ref_prim = stage.GetPrimAtPath("/World/Reference")
        >>>
        >>> # Convert world pose to local (relative to ref_prim)
        >>> world_pos = (10.0, 5.0, 0.0)
        >>> world_quat = (1.0, 0.0, 0.0, 0.0)  # identity rotation
        >>> local_pos, local_quat = sim_utils.convert_world_pose_to_local(world_pos, world_quat, ref_prim)
        >>> print(f"Local position: {local_pos}")
        >>> print(f"Local orientation: {local_quat}")
    """
    # Check if prim is valid
    if not ref_prim.IsValid():
        raise ValueError(f"Reference prim at path '{ref_prim.GetPath().pathString}' is not valid.")

    # If reference prim is the root, return world pose as-is
    if ref_prim.GetPath() == Sdf.Path.absoluteRootPath:
        return position, orientation  # type: ignore

    # Check if reference prim is a valid xformable
    ref_xformable = UsdGeom.Xformable(ref_prim)
    # Get reference prim's world transform
    ref_world_tf = ref_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # Create world transform for the desired position and orientation
    desired_world_tf = Gf.Matrix4d()
    desired_world_tf.SetTranslateOnly(Gf.Vec3d(*position))

    if orientation is not None:
        # Set rotation from quaternion (w, x, y, z)
        quat = Gf.Quatd(*orientation)
        desired_world_tf.SetRotateOnly(quat)

    # Convert world transform to local: local = world * inv(ref_world)
    ref_world_tf_inv = ref_world_tf.GetInverse()
    local_tf = desired_world_tf * ref_world_tf_inv

    # Extract local translation and orientation
    local_transform = Gf.Transform(local_tf)
    local_translation = tuple(local_transform.GetTranslation())

    local_orientation = None
    if orientation is not None:
        quat_result = local_transform.GetRotation().GetQuat()
        local_orientation = (quat_result.GetReal(), *quat_result.GetImaginary())

    return local_translation, local_orientation
