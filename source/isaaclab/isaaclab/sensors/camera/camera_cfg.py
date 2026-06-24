# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.sim import FisheyeCameraCfg, PinholeCameraCfg
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .camera import Camera


@configclass
class CameraCfg(SensorBaseCfg):
    """Configuration for a camera sensor."""
    """摄像头传感器的配置。"""

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""
        """传感器框架的偏移姿势与传感器的母体框架。"""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        """翻译w.r.t
        它们的母体。
        在 (0.0，0.0，0.0) 之前的默认设置。
        """

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""
        """四元数旋转 (w，x，y，z) w.r.t。
        它们的母体。
        在 (1.0，0.0，0.0，0.0，0.0的默认情况下。
        """

        convention: Literal["opengl", "ros", "world"] = "ros"
        """The convention in which the frame offset is applied. Defaults to "ros".

        - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y`` - Offset is applied in the OpenGL (Usd.Camera)
          convention.
        - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y`` - Offset is applied in the ROS convention.
        - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z`` - Offset is applied in the World Frame convention.

        """
        """框架抵消的公约
        默认的"ros"。

        - 在OpenGL (Usd.Camera) 公约中，应用``"opengl"`` - 前轴:``-Z`` - 上轴:``+Y`` - 抵消。
        - 在ROS公约中，``"ros"`` - 前轴:``+Z`` - 上轴:``-Y`` - 抵消是应用的。
        - 在"世界框架"公约中，应用``"world"`` - 前轴:``+X`` - 上轴:``+Z`` - 抵消。
        """

    class_type: type = Camera

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity.

    Note:
        The parent frame is the frame the sensor attaches to. For example, the parent frame of a
        camera at path ``/World/envs/env_0/Robot/Camera`` is ``/World/envs/env_0/Robot``.
    """
    """传感器框架的偏移姿势与传感器的母体框架。
    默认身份。

    说明：
        传感器附着的框架。
        例如，路径``/World/envs/env_0/Robot/Camera``的相机的母框是``/World/envs/env_0/Robot``。
    """

    spawn: PinholeCameraCfg | FisheyeCameraCfg | None = MISSING
    """Spawn configuration for the asset.

    If None, then the prim is not spawned by the asset. Instead, it is assumed that the
    asset is already present in the scene.
    """
    """产品的产品配置。

    如果None，那么prim不是由资产产产生的。
    相反，假设该资产已经存在场景。
    """

    depth_clipping_behavior: Literal["max", "zero", "none"] = "none"
    """Clipping behavior for the camera for values exceed the maximum value. Defaults to "none".

    - ``"max"``: Values are clipped to the maximum value.
    - ``"zero"``: Values are clipped to zero.
    - ``"none``: No clipping is applied. Values will be returned as ``inf``.
    """
    """摄像机的裁剪行为，以查取值超过最大值。
    默认调整为"没有"。

    - ``"max"``:值被裁剪到最大值。
    - ``"zero"``:值被切断到零。
    - ``"none``:没有裁剪.值将被返回为``inf``。
    """

    data_types: list[str] = ["rgb"]
    """List of sensor names/types to enable for the camera. Defaults to ["rgb"].

    Please refer to the :class:`Camera` class for a list of available data types.
    """
    """传感器名字/类型列表
    在 ["rgb"中默认设置。

    请参阅:class:`Camera`类，以查看可用的数据类型列表。
    """

    width: int = MISSING
    """Width of the image in pixels."""
    """像素的宽度。"""

    height: int = MISSING
    """Height of the image in pixels."""
    """像素的高度。"""

    update_latest_camera_pose: bool = False
    """Whether to update the latest camera pose when fetching the camera's data. Defaults to False.

    If True, the latest camera pose is updated in the camera's data which will slow down performance
    due to the use of :class:`XformPrimView`.
    If False, the pose of the camera during initialization is returned.
    """
    """在获取相机数据时是否更新最新的相机姿势。
    默认为 False。

    如果True，相机数据中更新了最新的相机姿势，这将由于使用:class:`XformPrimView`而减慢性能。
    如果 False，将在启动时的相机姿势返回。
    """

    semantic_filter: str | list[str] = "*:*"
    """A string or a list specifying a semantic filter predicate. Defaults to ``"*:*"``.

    If a string, it should be a disjunctive normal form of (semantic type, labels). For examples:

    * ``"typeA : labelA & !labelB | labelC , typeB: labelA ; typeC: labelE"``:
      All prims with semantic type "typeA" and label "labelA" but not "labelB" or with label "labelC".
      Also, all prims with semantic type "typeB" and label "labelA", or with semantic type "typeC" and label "labelE".
    * ``"typeA : * ; * : labelA"``: All prims with semantic type "typeA" or with label "labelA"

    If a list of strings, each string should be a semantic type. The segmentation for prims with
    semantics of the specified types will be retrieved. For example, if the list is ["class"], only
    the segmentation for prims with semantics of type "class" will be retrieved.

    .. seealso::

        For more information on the semantics filter, see the documentation on `Replicator Semantics Schema Editor`_.

    .. _Replicator Semantics Schema Editor: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html#semantics-filtering
    """
    """一个字符串或列表指定一个语义过器预示。
    在``"*:*"``上默认。

    如果是一个字符串，它应该是分离式的正常形式 (语义类型，标签)。
    例如:

    * ``"typeA : labelA & !labelB | labelC ， typeB: labelA ； typeC:
      labelE"``所有:prims有"A型"和"A型"标签，但不是"B型"或"C型"标签.prims有语义类型"B型"和标签"A型"，或有语义类型"C型"和标签"E型"。
    * ``"typeA : * ； * : labelA"``:所有具有语义类型"typeA"或标签"labelA"的prims

    如果列表列表，每个字符串应该是语义类型。
    对于prims的细分，将采集指定类型的语义。
    例如，如果列表是 ["类"]，只会获取prims的语义类型"类"的细分。

    ..
    查看:

        有关语义过器的更多信息，请参见`Replicator Semantics Schema Editor`_的文档。

    .. _Replicator Semantics Schema Editor: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html#semantics-filtering
    """

    colorize_semantic_segmentation: bool = True
    """Whether to colorize the semantic segmentation images. Defaults to True.

    If True, semantic segmentation is converted to an image where semantic IDs are mapped to colors
    and returned as a ``uint8`` 4-channel array. If False, the output is returned as a ``int32`` array.
    """
    """是否将语义细分图像进行配色。
    默认为 True。

    如果是True，语义细分将转换为图像，其中语义IDs被映射到颜色，并作为``uint8`` 4通道阵列返回。
    如果False，输出将作为``int32``阵列返回。
    """

    colorize_instance_id_segmentation: bool = True
    """Whether to colorize the instance ID segmentation images. Defaults to True.

    If True, instance id segmentation is converted to an image where instance IDs are mapped to colors.
    and returned as a ``uint8`` 4-channel array. If False, the output is returned as a ``int32`` array.
    """
    """是否将实例 ID 分段图像进行配色。
    默认为 True。

    如果True，实例 id 分段将转换为图像，实例IDs将映射到颜色。
    作为一个``uint8``4道阵列。
    如果False，输出将作为``int32``阵列返回。
    """

    colorize_instance_segmentation: bool = True
    """Whether to colorize the instance ID segmentation images. Defaults to True.

    If True, instance segmentation is converted to an image where instance IDs are mapped to colors.
    and returned as a ``uint8`` 4-channel array. If False, the output is returned as a ``int32`` array.
    """
    """是否将实例 ID 分段图像进行配色。
    默认为 True。

    如果True，实例细分将转换为图像，其中实例IDs被映射为颜色。
    作为一个``uint8``4道阵列。
    如果False，输出将作为``int32``阵列返回。
    """

    semantic_segmentation_mapping: dict = {}
    """Dictionary mapping semantics to specific colours

    Eg.

    .. code-block:: python

        {
            "class:cube_1": (255, 36, 66, 255),
            "class:cube_2": (255, 184, 48, 255),
            "class:cube_3": (55, 255, 139, 255),
            "class:table": (255, 237, 218, 255),
            "class:ground": (100, 100, 100, 255),
            "class:robot": (61, 178, 255, 255),
        }

    """
    """字典对特定颜色进行语义映射

    现在，我知道。

    .. code-block:: python

        {
            "class:cube_1": (255, 36, 66, 255),
            "class:cube_2": (255, 184, 48, 255),
            "class:cube_3": (55, 255, 139, 255),
            "class:table": (255, 237, 218, 255),
            "class:ground": (100, 100, 100, 255),
            "class:robot": (61, 178, 255, 255),
        }
    """
