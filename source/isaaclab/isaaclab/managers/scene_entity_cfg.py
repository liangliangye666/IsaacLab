# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations
"""对于不同管理器来说，配置项。"""

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
    from isaaclab.scene import InteractiveScene


@configclass
class SceneEntityCfg:
    """Configuration for a scene entity that is used by the manager's term.

    This class is used to specify the name of the scene entity that is queried from the
    :class:`InteractiveScene` and passed to the manager's term function.
    """
    """管理器使用的场景实体配置

    这个类用于指定从:class:`InteractiveScene`中查询到管理器的项函数的场景实体名称。
    """

    name: str = MISSING
    """The name of the scene entity.

    This is the name defined in the scene configuration file. See the :class:`InteractiveSceneCfg`
    class for more details.
    """
    """场景实体的名称。

    这就是场景配置文件中定义的名称。
    看看:class:`InteractiveSceneCfg`
    class for more details.
    """

    joint_names: str | list[str] | None = None
    """The names of the joints from the scene entity. Defaults to None.

    The names can be either joint names or a regular expression matching the joint names.

    These are converted to joint indices on initialization of the manager and passed to the term
    function as a list of joint indices under :attr:`joint_ids`.
    """
    """场景实体的关节名称。
    默认为 None。

    这些名字可以是共同名字或与共同名字相匹配的普通表达式。

    管理器初始化后将这些转换为联合索引，并将其转移到:attr:`joint_ids`下的联合索引列表。
    """

    joint_ids: list[int] | slice = slice(None)
    """The indices of the joints from the asset required by the term. Defaults to slice(None), which means
    all the joints in the asset (if present).

    If :attr:`joint_names` is specified, this is filled in automatically on initialization of the
    manager.
    """
    """按期所要求的资产的结合索引。
    缺陷切割 ((None)，即资产中的所有关节 (如果存在)。

    If :吸引:`joint_names`在启动时，该数据自动填写。
    管理器。
    """

    fixed_tendon_names: str | list[str] | None = None
    """The names of the fixed tendons from the scene entity. Defaults to None.

    The names can be either joint names or a regular expression matching the joint names.

    These are converted to fixed tendon indices on initialization of the manager and passed to the term
    function as a list of fixed tendon indices under :attr:`fixed_tendon_ids`.
    """
    """场景实体的固定节点名称。
    默认为 None。

    这些名字可以是共同名字或与共同名字相匹配的普通表达式。

    在管理器初始化时，这些将转换为固定索引，并将其转移到项函数，作为:attr:`fixed_tendon_ids`下的固定索引列表。
    """

    fixed_tendon_ids: list[int] | slice = slice(None)
    """The indices of the fixed tendons from the asset required by the term. Defaults to slice(None), which means
    all the fixed tendons in the asset (if present).

    If :attr:`fixed_tendon_names` is specified, this is filled in automatically on initialization of the
    manager.
    """
    """按该项所要求的资产的固定的索引。
    缺陷切割 ((None)，即所有固定的 asset在资产中 (如果存在)。

    If :吸引:`fixed_tendon_names`在启动时，该数据自动填写。
    管理器。
    """

    body_names: str | list[str] | None = None
    """The names of the bodies from the asset required by the term. Defaults to None.

    The names can be either body names or a regular expression matching the body names.

    These are converted to body indices on initialization of the manager and passed to the term
    function as a list of body indices under :attr:`body_ids`.
    """
    """在该项所要求的资产中所述机构名称。
    默认为 None。

    这些名字可以是身体名字或与身体名字相匹配的普通表达式。

    管理器初始化后将这些转换为体索引，并将其转换为:attr:`body_ids`下的体索引列表。
    """

    body_ids: list[int] | slice = slice(None)
    """The indices of the bodies from the asset required by the term. Defaults to slice(None), which means
    all the bodies in the asset.

    If :attr:`body_names` is specified, this is filled in automatically on initialization of the
    manager.
    """
    """根据该项所要求的资产的机构索引。
    缺陷切割 ((None)，这意味着资产中的所有物体。

    If :吸引:`body_names`在启动时，该数据自动填写。
    管理器。
    """

    object_collection_names: str | list[str] | None = None
    """The names of the objects in the rigid object collection required by the term. Defaults to None.

    The names can be either names or a regular expression matching the object names in the collection.

    These are converted to object indices on initialization of the manager and passed to the term
    function as a list of object indices under :attr:`object_collection_ids`.
    """
    """按这个项所要求的硬体集合中的物体名称。
    默认为 None。

    名称可以是名称或是与集合中的物体名称相匹配的正则表达式。

    在管理器初始化时，这些将转换为对象索引，并将其传递到:attr:`object_collection_ids`下的项函数，作为对象索引列表。
    """

    object_collection_ids: list[int] | slice = slice(None)
    """The indices of the objects from the rigid object collection required by the term. Defaults to slice(None),
    which means all the objects in the collection.

    If :attr:`object_collection_names` is specified, this is filled in automatically on initialization of the manager.
    """
    """该项所要求的来自硬体集合的物体索引。
    默认的切片 ((None)，这意味着集合中的所有对象。

    If :attr:`object_collection_names`是指定的，在管理器初始化时自动填写。
    """

    preserve_order: bool = False
    """Whether to preserve indices ordering to match with that in the specified joint, body, or object collection names.
    Defaults to False.

    If False, the ordering of the indices are sorted in ascending order (i.e. the ordering in the entity's joints,
    bodies, or object in the object collection). Otherwise, the indices are preserved in the order of the specified
    joint, body, or object collection names.

    For more details, see the :meth:`isaaclab.utils.string.resolve_matching_names` function.

    .. note::
        This attribute is only used when :attr:`joint_names`, :attr:`body_names`, or :attr:`object_collection_names`
        are specified.

    """
    """保存与指定关节，体体或物体集合名称的索引一致的索引。
    默认为 False。

    如果False，索引的排序是以上升顺序排序的 (i.e.是对象集合中的实体的关节，体或对象的排序)。
    其他情况下，索引按指定关节，体或物体集合名称顺序保存。

    详细见:meth:`isaaclab.utils.string.resolve_matching_names`函数。

    .. 说明::
        这种属性仅用于指定:attr:`joint_names`，:attr:`body_names`或:attr:`object_collection_names`时。
    """

    def resolve(self, scene: InteractiveScene):
        """Resolves the scene entity and converts the joint and body names to indices.

        This function examines the scene entity from the :class:`InteractiveScene` and resolves the indices
        and names of the joints and bodies. It is an expensive operation as it resolves regular expressions
        and should be called only once.

        Args:
            scene: The interactive scene instance.

        Raises:
            ValueError: If the scene entity is not found.
            ValueError: If both ``joint_names`` and ``joint_ids`` are specified and are not consistent.
            ValueError: If both ``fixed_tendon_names`` and ``fixed_tendon_ids`` are specified and are not consistent.
            ValueError: If both ``body_names`` and ``body_ids`` are specified and are not consistent.
            ValueError: If both ``object_collection_names`` and ``object_collection_ids`` are specified and
                are not consistent.
        """
        """解决场景实体，并将关节和体名转换为索引。

        这项函数从:class:`InteractiveScene`中检查场景实体，并解决关节和身体的索引和名称。
        这是一个昂贵的操作，因为它解决了正则表达式，

        参数：
            scene: 交互场景实例。

        异常：
            ValueError: 如果未找到场景实体。
            ValueError: 如果``joint_names``和``joint_ids``都指定，并且不一致。
            ValueError: 如果``fixed_tendon_names``和``fixed_tendon_ids``都指定，并且不一致。
            ValueError: 如果``body_names``和``body_ids``都指定，并且不一致。
            ValueError: 如果``object_collection_names``和``object_collection_ids``都指定，并且不一致。
        """
        # check if the entity is valid
        if self.name not in scene.keys():
            raise ValueError(f"The scene entity '{self.name}' does not exist. Available entities: {scene.keys()}.")

        # convert joint names to indices based on regex
        self._resolve_joint_names(scene)

        # convert fixed tendon names to indices based on regex
        self._resolve_fixed_tendon_names(scene)

        # convert body names to indices based on regex
        self._resolve_body_names(scene)

        # convert object collection names to indices based on regex
        self._resolve_object_collection_names(scene)

    '''
    关节名与索引的双向翻译器
        它处理一种非常实际的配置场景——你的配置里可能只写了关节名（"slider_to_cart"），也可能只写了索引（[0, 1]），也可能两个都写了。
        这个函数负责：校验 + 补全缺失的一方。

        进入 _resolve_joint_names
            │
            ├── 情况 A: joint_names ≠ None 且 joint_ids ≠ slice(None)
            │     用户同时指定了名字和索引 → 校验一致性
            │
            ├── 情况 B: joint_names ≠ None (joint_ids 是默认的 slice(None))
            │     用户只给了名字 → 查 PhysX，补上 joint_ids
            │
            └── 情况 C: joint_ids ≠ slice(None) (joint_names 是 None)
                用户只给了索引 → 查 PhysX，补上 joint_names
    '''
    def _resolve_joint_names(self, scene: InteractiveScene):
        # convert joint names to indices based on regex
        if self.joint_names is not None or self.joint_ids != slice(None):
            entity: Articulation = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.joint_names is not None and self.joint_ids != slice(None):
                if isinstance(self.joint_names, str):
                    self.joint_names = [self.joint_names]
                if isinstance(self.joint_ids, int):
                    self.joint_ids = [self.joint_ids]
                joint_ids, _ = entity.find_joints(self.joint_names, preserve_order=self.preserve_order)
                joint_names = [entity.joint_names[i] for i in self.joint_ids]
                if joint_ids != self.joint_ids or joint_names != self.joint_names:
                    raise ValueError(
                        "Both 'joint_names' and 'joint_ids' are specified, and are not consistent."
                        f"\n\tfrom joint names: {self.joint_names} [{joint_ids}]"
                        f"\n\tfrom joint ids: {joint_names} [{self.joint_ids}]"
                        "\nHint: Use either 'joint_names' or 'joint_ids' to avoid confusion."
                    )
            # -- from joint names to joint indices
            elif self.joint_names is not None:
                if isinstance(self.joint_names, str):
                    self.joint_names = [self.joint_names]
                self.joint_ids, _ = entity.find_joints(self.joint_names, preserve_order=self.preserve_order)
                # performance optimization (slice offers faster indexing than list of indices)
                # only all joint in the entity order are selected
                if len(self.joint_ids) == entity.num_joints and self.joint_names == entity.joint_names:
                    self.joint_ids = slice(None)
            # -- from joint indices to joint names
            elif self.joint_ids != slice(None):
                if isinstance(self.joint_ids, int):
                    self.joint_ids = [self.joint_ids]
                self.joint_names = [entity.joint_names[i] for i in self.joint_ids]

    '''
    与关节镜像的肌腱解析器（如仿生韧带）
    这个方法和刚刚讲的 _resolve_joint_names 结构完全一致，唯一区别是把"关节（joint）"换成了"固定肌腱（fixed tendon）"。

    一、固定肌腱（Fixed Tendon）是什么？
        在机器人物理仿真中，固定肌腱模拟的是机器人身上固定长度的被动连接结构——比如真实的肌腱、钢缆、同步带。
        它有长度约束，但不能被主动控制，不具备驱动器（电机）的功能。

        特性	        关节 (Joint)	            固定肌腱 (Fixed Tendon)
        有无驱动力矩	有（电机驱动）	                无（纯被动约束）
        作用	        连接两个刚体，允许相对运动	    保持两个点之间的固定距离
        举例	        膝关节、肘关节	                仿生肌腱、四足机器人的弹性韧带
        通俗理解：关节像一个可以主动转动的铰链，肌腱像一根不能拉长缩短的绳子。
    '''
    def _resolve_fixed_tendon_names(self, scene: InteractiveScene):
        # convert tendon names to indices based on regex
        if self.fixed_tendon_names is not None or self.fixed_tendon_ids != slice(None):
            entity: Articulation = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.fixed_tendon_names is not None and self.fixed_tendon_ids != slice(None):
                if isinstance(self.fixed_tendon_names, str):
                    self.fixed_tendon_names = [self.fixed_tendon_names]
                if isinstance(self.fixed_tendon_ids, int):
                    self.fixed_tendon_ids = [self.fixed_tendon_ids]
                fixed_tendon_ids, _ = entity.find_fixed_tendons(
                    self.fixed_tendon_names, preserve_order=self.preserve_order
                )
                fixed_tendon_names = [entity.fixed_tendon_names[i] for i in self.fixed_tendon_ids]
                if fixed_tendon_ids != self.fixed_tendon_ids or fixed_tendon_names != self.fixed_tendon_names:
                    raise ValueError(
                        "Both 'fixed_tendon_names' and 'fixed_tendon_ids' are specified, and are not consistent."
                        f"\n\tfrom joint names: {self.fixed_tendon_names} [{fixed_tendon_ids}]"
                        f"\n\tfrom joint ids: {fixed_tendon_names} [{self.fixed_tendon_ids}]"
                        "\nHint: Use either 'fixed_tendon_names' or 'fixed_tendon_ids' to avoid confusion."
                    )
            # -- from fixed tendon names to fixed tendon indices
            elif self.fixed_tendon_names is not None:
                if isinstance(self.fixed_tendon_names, str):
                    self.fixed_tendon_names = [self.fixed_tendon_names]
                self.fixed_tendon_ids, _ = entity.find_fixed_tendons(
                    self.fixed_tendon_names, preserve_order=self.preserve_order
                )
                # performance optimization (slice offers faster indexing than list of indices)
                # only all fixed tendon in the entity order are selected
                if (
                    len(self.fixed_tendon_ids) == entity.num_fixed_tendons
                    and self.fixed_tendon_names == entity.fixed_tendon_names
                ):
                    self.fixed_tendon_ids = slice(None)
            # -- from fixed tendon indices to fixed tendon names
            elif self.fixed_tendon_ids != slice(None):
                if isinstance(self.fixed_tendon_ids, int):
                    self.fixed_tendon_ids = [self.fixed_tendon_ids]
                self.fixed_tendon_names = [entity.fixed_tendon_names[i] for i in self.fixed_tendon_ids]

    '''
    刚体名称与索引的双向翻译器
    结构和前两个（关节、肌腱）完全一致，但这里操作的是刚体（Body）——机器人身上每个独立的物理链路。

    一、刚体（Body）是什么？
        一个机器人的物理结构由多个刚体组成，它们通过关节连接：
            四足机器人 Anymal 的刚体结构：
            ┌────────────────┐
            │  base（机身）    │  ← body_names[0] = "base"
            │  ┌──┴──┐       │
            │  │     │       │
            │ LF    RF       │  ← body_names[1] = "LF_HIP", [2] = "RF_HIP"
            │ │     │        │
            │LF_THIGH ...    │  ← body_names[3] = "LF_THIGH", ...
            └────────────────┘
        每个刚体是一个物理计算单元——PhysX 对每个刚体单独计算碰撞、惯性、受力。你需要通过刚体名来：
            读取某个身体部件的位置/速度（如"前左脚的 z 坐标"）
            判断接触（如"右脚是否碰到地面了"）
            施加外部力（如"推一下机身的质心"）
    '''
    def _resolve_body_names(self, scene: InteractiveScene):
        # convert body names to indices based on regex
        if self.body_names is not None or self.body_ids != slice(None):
            entity: RigidObject = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.body_names is not None and self.body_ids != slice(None):
                if isinstance(self.body_names, str):
                    self.body_names = [self.body_names]
                if isinstance(self.body_ids, int):
                    self.body_ids = [self.body_ids]
                body_ids, _ = entity.find_bodies(self.body_names, preserve_order=self.preserve_order)
                body_names = [entity.body_names[i] for i in self.body_ids]
                if body_ids != self.body_ids or body_names != self.body_names:
                    raise ValueError(
                        "Both 'body_names' and 'body_ids' are specified, and are not consistent."
                        f"\n\tfrom body names: {self.body_names} [{body_ids}]"
                        f"\n\tfrom body ids: {body_names} [{self.body_ids}]"
                        "\nHint: Use either 'body_names' or 'body_ids' to avoid confusion."
                    )
            # -- from body names to body indices
            elif self.body_names is not None:
                if isinstance(self.body_names, str):
                    self.body_names = [self.body_names]
                self.body_ids, _ = entity.find_bodies(self.body_names, preserve_order=self.preserve_order)
                # performance optimization (slice offers faster indexing than list of indices)
                # only all bodies in the entity order are selected
                if len(self.body_ids) == entity.num_bodies and self.body_names == entity.body_names:
                    self.body_ids = slice(None)
            # -- from body indices to body names
            elif self.body_ids != slice(None):
                if isinstance(self.body_ids, int):
                    self.body_ids = [self.body_ids]
                self.body_names = [entity.body_names[i] for i in self.body_ids]

    '''
    物体集合的名称解析器
    这是 SceneEntityCfg 中第四个名称解析方法。结构和前三个 95% 一致，但有两个关键差异值得关注。

    一、物体集合（RigidObjectCollection）是什么？
    前三个方法操作的都是单个机器人（Articulation 或 RigidObject）的内部结构（关节、肌腱、刚体）。但这个方法操作的是一种完全不同的实体：
        entity: RigidObjectCollection = scene[self.name]
        RigidObjectCollection 是多个独立刚体的集合——比如桌子上散落的 10 个方块，或者场景中的 5 个目标物体。它们之间没有关节连接，各自独立存在。

        概念	    类型	                举例
        关节体	    Articulation	        一个 Franka 机械臂（关节连接的刚体链）
        单个刚体	RigidObject	            桌面上的一个方块
        物体集合	RigidObjectCollection	桌面上的 10 个方块（批量管理）
    使用场景：在抓取任务中，你需要同时管理多个可抓取物体。RigidObjectCollection 让你对一堆方块统一操作：
        # 场景中的物体集合
        SceneEntityCfg("object", object_collection_names=["cube_0", "cube_1", "cube_3"])
        # 解析后: object_collection_ids = [0, 1, 3]
    '''
    def _resolve_object_collection_names(self, scene: InteractiveScene):
        # convert object names to indices based on regex
        if self.object_collection_names is not None or self.object_collection_ids != slice(None):
            entity: RigidObjectCollection = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.object_collection_names is not None and self.object_collection_ids != slice(None):
                if isinstance(self.object_collection_names, str):
                    self.object_collection_names = [self.object_collection_names]
                if isinstance(self.object_collection_ids, int):
                    self.object_collection_ids = [self.object_collection_ids]
                object_ids, _ = entity.find_objects(self.object_collection_names, preserve_order=self.preserve_order)
                object_names = [entity.object_names[i] for i in self.object_collection_ids]
                if object_ids != self.object_collection_ids or object_names != self.object_collection_names:
                    raise ValueError(
                        "Both 'object_collection_names' and 'object_collection_ids' are specified, and are not"
                        " consistent.\n\tfrom object collection names:"
                        f" {self.object_collection_names} [{object_ids}]\n\tfrom object collection ids:"
                        f" {object_names} [{self.object_collection_ids}]\nHint: Use either 'object_collection_names' or"
                        " 'object_collection_ids' to avoid confusion."
                    )
            # -- from object names to object indices
            elif self.object_collection_names is not None:
                if isinstance(self.object_collection_names, str):
                    self.object_collection_names = [self.object_collection_names]
                self.object_collection_ids, _ = entity.find_objects(
                    self.object_collection_names, preserve_order=self.preserve_order
                )
            # -- from object indices to object names
            elif self.object_collection_ids != slice(None):
                if isinstance(self.object_collection_ids, int):
                    self.object_collection_ids = [self.object_collection_ids]
                self.object_collection_names = [entity.object_names[i] for i in self.object_collection_ids]
