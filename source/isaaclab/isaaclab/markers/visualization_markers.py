# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A class to coordinate groups of visual markers (such as spheres, frames or arrows)
using `UsdGeom.PointInstancer`_ class.

The class :class:`VisualizationMarkers` is used to create a group of visual markers and
visualize them in the viewport. The markers are represented as :class:`UsdGeom.PointInstancer` prims
in the USD stage. The markers are created as prototypes in the :class:`UsdGeom.PointInstancer` prim
and are instanced in the :class:`UsdGeom.PointInstancer` prim. The markers can be visualized by
passing the indices of the marker prototypes and their translations, orientations and scales.
The marker prototypes can be configured with the :class:`VisualizationMarkersCfg` class.

.. _UsdGeom.PointInstancer: https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html
"""

# needed to import for allowing type-hinting: np.ndarray | torch.Tensor | None
from __future__ import annotations
"""用`UsdGeom.PointInstancer`_类进行视觉标记组组协调 (如球体，框架或箭头)。

类:class:`VisualizationMarkers`用于创建一组视觉标记，并在视角中可视化它们。
在USD阶段，标记表示为:class:`UsdGeom.PointInstancer` prims。
标记在:class:`UsdGeom.PointInstancer` prim中创建为原型，并在:class:`UsdGeom.PointInstancer` prim中实例化。
通过通过标记原型的指标及其翻译，方向和尺度来可视化标记。
标记原型可以与:class:`VisualizationMarkersCfg`类配置。

.. _UsdGeom.PointInstancer: https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html
"""

import logging
from dataclasses import MISSING

import numpy as np
import torch

import omni.physx.scripts.utils as physx_utils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, Vt

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners import SpawnerCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import convert_quat

# import logger
logger = logging.getLogger(__name__)


@configclass
class VisualizationMarkersCfg:
    """A class to configure a :class:`VisualizationMarkers`."""
    """一个配置:class:`VisualizationMarkers`的课程。"""

    prim_path: str = MISSING
    """The prim path where the :class:`UsdGeom.PointInstancer` will be created."""
    """在 prim 路径中，将创建 :class:`UsdGeom.PointInstancer`。"""

    markers: dict[str, SpawnerCfg] = MISSING
    """The dictionary of marker configurations.

    The key is the name of the marker, and the value is the configuration of the marker.
    The key is used to identify the marker in the class.
    """
    """标记配置字典。

    关键是标记名称，值是标记配置。
    键用于识别该类中的标记。
    """


class VisualizationMarkers:
    """A class to coordinate groups of visual markers (loaded from USD).

    This class allows visualization of different UI markers in the scene, such as points and frames.
    The class wraps around the `UsdGeom.PointInstancer`_ for efficient handling of objects
    in the stage via instancing the created marker prototype prims.

    A marker prototype prim is a reusable template prim used for defining variations of objects
    in the scene. For example, a sphere prim can be used as a marker prototype prim to create
    multiple sphere prims in the scene at different locations. Thus, prototype prims are useful
    for creating multiple instances of the same prim in the scene.

    The class parses the configuration to create different the marker prototypes into the stage. Each marker
    prototype prim is created as a child of the :class:`UsdGeom.PointInstancer` prim. The prim path for the
    marker prim is resolved using the key of the marker in the :attr:`VisualizationMarkersCfg.markers`
    dictionary. The marker prototypes are created using the :meth:`isaaclab.sim.utils.prims.create_prim`
    function, and then instanced using :class:`UsdGeom.PointInstancer` prim to allow creating multiple
    instances of the marker prims.

    Switching between different marker prototypes is possible by calling the :meth:`visualize` method with
    the prototype indices corresponding to the marker prototype. The prototype indices are based on the order
    in the :attr:`VisualizationMarkersCfg.markers` dictionary. For example, if the dictionary has two markers,
    "marker1" and "marker2", then their prototype indices are 0 and 1 respectively. The prototype indices
    can be passed as a list or array of integers.

    Usage:
        The following snippet shows how to create 24 sphere markers with a radius of 1.0 at random translations
        within the range [-1.0, 1.0]. The first 12 markers will be colored red and the rest will be colored green.

        .. code-block:: python

            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

            # Create the markers configuration
            # This creates two marker prototypes, "marker1" and "marker2" which are spheres with a radius of 1.0.
            # The color of "marker1" is red and the color of "marker2" is green.
            cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/testMarkers",
                markers={
                    "marker1": sim_utils.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "marker2": VisualizationMarkersCfg.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
            # Create the markers instance
            # This will create a UsdGeom.PointInstancer prim at the given path along with the marker prototypes.
            marker = VisualizationMarkers(cfg)

            # Set position of the marker
            # -- randomly sample translations between -1.0 and 1.0
            marker_translations = np.random.uniform(-1.0, 1.0, (24, 3))
            # -- this will create 24 markers at the given translations
            # note: the markers will all be `marker1` since the marker indices are not given
            marker.visualize(translations=marker_translations)

            # alter the markers based on their prototypes indices
            # first 12 markers will be marker1 and the rest will be marker2
            # 0 -> marker1, 1 -> marker2
            marker_indices = [0] * 12 + [1] * 12
            # this will change the marker prototypes at the given indices
            # note: the translations of the markers will not be changed from the previous call
            #  since the translations are not given.
            marker.visualize(marker_indices=marker_indices)

            # alter the markers based on their prototypes indices and translations
            marker.visualize(marker_indices=marker_indices, translations=marker_translations)

    .. _UsdGeom.PointInstancer: https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html

    """
    """一个对视觉标记组进行协调的类 (从USD上载)。

    这个类允许在场景中可视化不同的UI标记，如点和框架。
    该类包裹在`UsdGeom.PointInstancer`_周围，以通过创建的标记原型prims实现阶段对象的有效处理。

    标记原型prim是可重复使用的模板prim用于定义场景中的对象变化。
    例如，一个球体prim可以作为标记原型prim用于在不同地点在场景中创建多个球体prims。
    因此，prims原型是有用的
    for creating multiple instances of the same prim in the scene.

    该类分析配置，以创建不同的标记原型。
    每个标记原型prim都是:class:`UsdGeom.PointInstancer` prim的产物。
    用:attr:`VisualizationMarkersCfg.markers`字典中的标记键来解决标记prim的prim路径。
    标记原型是使用 :meth:`isaaclab.sim.utils.prims.create_prim` 函数创建的，然后使用 :class:`UsdGeom.PointInstancer` prim
    实例化以允许创建多个标记 prims 实例。

    通过将:meth:`visualize`方法与与标记原型相符的原型指标调用来切换不同的标记原型来实现。
    原型指标基于:attr:`VisualizationMarkersCfg.markers`字典中的顺序。
    例如，如果字典有两个标记"，标记者1"和"标记者2"，则它们的原型索引分别为0和1。
    原型索引可以作为整数列表或数组传递。

    Usage: 下面的摘录显示如何在 [-1.0， 1.0]范围内随机翻译时创建24个半径球标记。
           第12个标记将是红色，其余的将是绿色。

        .. code-block:: python

            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

            # Create the markers configuration
            # This creates two marker prototypes, "marker1" and "marker2" which are spheres with a radius of 1.0.
            # The color of "marker1" is red and the color of "marker2" is green.
            cfg = VisualizationMarkersCfg(
                prim_path="/World/Visuals/testMarkers",
                markers={
                    "marker1": sim_utils.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "marker2": VisualizationMarkersCfg.SphereCfg(
                        radius=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
            # Create the markers instance
            # This will create a UsdGeom.PointInstancer prim at the given path along with the marker prototypes.
            marker = VisualizationMarkers(cfg)

            # Set position of the marker
            # -- randomly sample translations between -1.0 and 1.0
            marker_translations = np.random.uniform(-1.0, 1.0, (24, 3))
            # -- this will create 24 markers at the given translations
            # note: the markers will all be `marker1` since the marker indices are not given
            marker.visualize(translations=marker_translations)

            # alter the markers based on their prototypes indices
            # first 12 markers will be marker1 and the rest will be marker2
            # 0 -> marker1, 1 -> marker2
            marker_indices = [0] * 12 + [1] * 12
            # this will change the marker prototypes at the given indices
            # note: the translations of the markers will not be changed from the previous call
            #  since the translations are not given.
            marker.visualize(marker_indices=marker_indices)

            # alter the markers based on their prototypes indices and translations
            marker.visualize(marker_indices=marker_indices, translations=marker_translations)

    .. _UsdGeom.PointInstancer: https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html
    """

    def __init__(self, cfg: VisualizationMarkersCfg):
        """Initialize the class.

        When the class is initialized, the :class:`UsdGeom.PointInstancer` is created into the stage
        and the marker prims are registered into it.

        .. note::
            If a prim already exists at the given path, the function will find the next free path
            and create the :class:`UsdGeom.PointInstancer` prim there.

        Args:
            cfg: The configuration for the markers.

        Raises:
            ValueError: When no markers are provided in the :obj:`cfg`.
        """
        """启动课程。

        当类被初始化时，:class:`UsdGeom.PointInstancer`被创建到阶段，并注册prims标记。

        .. 说明::
            如果一个prim已经存在给定的路径，函数将找到下一个自由路径，并在那里创建:class:`UsdGeom.PointInstancer` prim。

        参数：
            cfg: 标记的配置。

        异常：
            ValueError: 在:obj:`cfg`中没有标记。
        """
        # get next free path for the prim
        prim_path = sim_utils.get_next_free_prim_path(cfg.prim_path)
        # create a new prim
        self.stage = sim_utils.get_current_stage()
        self._instancer_manager = UsdGeom.PointInstancer.Define(self.stage, prim_path)
        # store inputs
        self.prim_path = prim_path
        self.cfg = cfg
        # check if any markers is provided
        if len(self.cfg.markers) == 0:
            raise ValueError(f"The `cfg.markers` cannot be empty. Received: {self.cfg.markers}")

        # create a child prim for the marker
        self._add_markers_prototypes(self.cfg.markers)
        # Note: We need to do this the first time to initialize the instancer.
        #   Otherwise, the instancer will not be "created" and the function `GetInstanceIndices()` will fail.
        self._instancer_manager.GetProtoIndicesAttr().Set(list(range(self.num_prototypes)))
        self._instancer_manager.GetPositionsAttr().Set([Gf.Vec3f(0.0)] * self.num_prototypes)
        self._count = self.num_prototypes

    def __str__(self) -> str:
        """Return: A string representation of the class."""
        """Return: 一个类的字符串表示。"""
        msg = f"VisualizationMarkers(prim_path={self.prim_path})"
        msg += f"\n\tCount: {self.count}"
        msg += f"\n\tNumber of prototypes: {self.num_prototypes}"
        msg += "\n\tMarkers Prototypes:"
        for index, (name, marker) in enumerate(self.cfg.markers.items()):
            msg += f"\n\t\t[Index: {index}]: {name}: {marker.to_dict()}"
        return msg

    """
    Properties.
    """
    """属性。
    """

    @property
    def num_prototypes(self) -> int:
        """The number of marker prototypes available."""
        """有的标记原型数量。"""
        return len(self.cfg.markers)

    @property
    def count(self) -> int:
        """The total number of marker instances."""
        """标记例的总数"""
        # TODO: Update this when the USD API is available (Isaac Sim 2023.1)
        # return self._instancer_manager.GetInstanceCount()
        return self._count

    """
    Operations.
    """
    """操作。
    """

    def set_visibility(self, visible: bool):
        """Sets the visibility of the markers.

        The method does this through the USD API.

        Args:
            visible: flag to set the visibility.
        """
        """设定标记的可见性。

        这种方法通过USDAPI来实现。

        参数：
            visible: 标志设置可见性。
        """
        imageable = UsdGeom.Imageable(self._instancer_manager)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def is_visible(self) -> bool:
        """Checks the visibility of the markers.

        Returns:
            True if the markers are visible, False otherwise.
        """
        """检查标记的可见性。

        返回：
            如果标记可见，则True，否则False。
        """
        return self._instancer_manager.GetVisibilityAttr().Get() != UsdGeom.Tokens.invisible

    def visualize(
        self,
        translations: np.ndarray | torch.Tensor | None = None,
        orientations: np.ndarray | torch.Tensor | None = None,
        scales: np.ndarray | torch.Tensor | None = None,
        marker_indices: list[int] | np.ndarray | torch.Tensor | None = None,
    ):
        """Update markers in the viewport.

        .. note::
            If the prim `PointInstancer` is hidden in the stage, the function will simply return
            without updating the markers. This helps in unnecessary computation when the markers
            are not visible.

        Whenever updating the markers, the input arrays must have the same number of elements
        in the first dimension. If the number of elements is different, the `UsdGeom.PointInstancer`
        will raise an error complaining about the mismatch.

        Additionally, the function supports dynamic update of the markers. This means that the
        number of markers can change between calls. For example, if you have 24 points that you
        want to visualize, you can pass 24 translations, orientations, and scales. If you want to
        visualize only 12 points, you can pass 12 translations, orientations, and scales. The
        function will automatically update the number of markers in the scene.

        The function will also update the marker prototypes based on their prototype indices. For instance,
        if you have two marker prototypes, and you pass the following marker indices: [0, 1, 0, 1], the function
        will update the first and third markers with the first prototype, and the second and fourth markers
        with the second prototype. This is useful when you want to visualize different markers in the same
        scene. The list of marker indices must have the same number of elements as the translations, orientations,
        or scales. If the number of elements is different, the function will raise an error.

        .. caution::
            This function will update all the markers instanced from the prototypes. That means
            if you have 24 markers, you will need to pass 24 translations, orientations, and scales.

            If you want to update only a subset of the markers, you will need to handle the indices
            yourself and pass the complete arrays to this function.

        Args:
            translations: Translations w.r.t. parent prim frame. Shape is (M, 3).
                Defaults to None, which means left unchanged.
            orientations: Quaternion orientations (w, x, y, z) w.r.t. parent prim frame. Shape is (M, 4).
                Defaults to None, which means left unchanged.
            scales: Scale applied before any rotation is applied. Shape is (M, 3).
                Defaults to None, which means left unchanged.
            marker_indices: Decides which marker prototype to visualize. Shape is (M).
                Defaults to None, which means left unchanged provided that the total number of markers
                is the same as the previous call. If the number of markers is different, the function
                will update the number of markers in the scene.

        Raises:
            ValueError: When input arrays do not follow the expected shapes.
            ValueError: When the function is called with all None arguments.
        """
        """更新视角标记。

        .. 说明::
            如果prim `PointInstancer`隐藏在阶段， 函数将简单地返回，
            这有助于不必要的计算，

        在更新标记时，输入阵列必须在第一维度中具有相同数量的元素。
        如果元素的数量不同，`UsdGeom.PointInstancer`将出现一个错误，抱怨不匹配。

        此外，该函数支持动态更新标记。
        这意味着电话间的标记数量可以变化。
        例如，如果你想想象24个点，你可以通过24个翻译，指向和尺度。
        如果你想想想象出12个点，你可以通过12个翻译，指向和尺度。
        功能将自动更新场景标记数量。

        该函数还将根据其原型索引更新标记原型。
        例如，
        if you have two marker prototypes, and you pass the following marker indices: [0, 1, 0, 1], the function
        将第一个和第三个标记更新到第一个原型，第二个和第四个标记
        with the second prototype. This is useful when you want to visualize different markers in the same
        在场景。
        标记索引列表必须与翻译，方向或尺度相同的元素数量。
        如果元素数量不同，函数会产生错误。

        .. 谨慎::
            这项功能将更新从原型中获得的所有标记。
            这意味着
            if you have 24 markers, you will need to pass 24 translations, orientations, and scales.

            如果您想更新只有一个子集的标记，您需要自己处理索引，并将整个阵列传递到这个函数。

        参数：
            translations: 翻译w.r.t。
                          产物 prim 框架。
                          形状是 (M， 3)。
                          默认为 None，这意味着没有改变。
            orientations: 四元数方向 (w， x， y， z) w.r.t。
                          产物 prim 框架。
                          形状是 (M， 4)。
                          默认为 None，这意味着没有改变。
            scales: 在任何旋转之前应用的尺度。
                    形状是 (M， 3)。
                    默认为 None，这意味着没有改变。
            marker_indices: 决定哪个标记原型可视化。
                            形状是 (M)。
                            默认对None的默认，这意味着没有变化，只要标记的总数与上次调用相同。
                            如果标记数量不同，函数将更新场景标记数量。

        异常：
            ValueError: 当输入阵列不按照预期的形状。
            ValueError: 当函数调用所有None参数时。
        """
        # check if it is visible (if not then let's not waste time)
        if not self.is_visible():
            return
        # check if we have any markers to visualize
        num_markers = 0
        # resolve inputs
        # -- position
        if translations is not None:
            if isinstance(translations, torch.Tensor):
                translations = translations.detach().cpu().numpy()
            # check that shape is correct
            if translations.shape[1] != 3 or len(translations.shape) != 2:
                raise ValueError(f"Expected `translations` to have shape (M, 3). Received: {translations.shape}.")
            # apply translations
            self._instancer_manager.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(translations))
            # update number of markers
            num_markers = translations.shape[0]
        # -- orientation
        if orientations is not None:
            if isinstance(orientations, torch.Tensor):
                orientations = orientations.detach().cpu().numpy()
            # check that shape is correct
            if orientations.shape[1] != 4 or len(orientations.shape) != 2:
                raise ValueError(f"Expected `orientations` to have shape (M, 4). Received: {orientations.shape}.")
            # roll orientations from (w, x, y, z) to (x, y, z, w)
            # internally USD expects (x, y, z, w)
            orientations = convert_quat(orientations, to="xyzw")
            # apply orientations
            self._instancer_manager.GetOrientationsAttr().Set(Vt.QuathArray.FromNumpy(orientations))
            # update number of markers
            num_markers = orientations.shape[0]
        # -- scales
        if scales is not None:
            if isinstance(scales, torch.Tensor):
                scales = scales.detach().cpu().numpy()
            # check that shape is correct
            if scales.shape[1] != 3 or len(scales.shape) != 2:
                raise ValueError(f"Expected `scales` to have shape (M, 3). Received: {scales.shape}.")
            # apply scales
            self._instancer_manager.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(scales))
            # update number of markers
            num_markers = scales.shape[0]
        # -- status
        if marker_indices is not None or num_markers != self._count:
            # apply marker indices
            if marker_indices is not None:
                if isinstance(marker_indices, torch.Tensor):
                    marker_indices = marker_indices.detach().cpu().numpy()
                elif isinstance(marker_indices, list):
                    marker_indices = np.array(marker_indices)
                # check that shape is correct
                if len(marker_indices.shape) != 1:
                    raise ValueError(f"Expected `marker_indices` to have shape (M,). Received: {marker_indices.shape}.")
                # apply proto indices
                self._instancer_manager.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(marker_indices))
                # update number of markers
                num_markers = marker_indices.shape[0]
            else:
                # check that number of markers is not zero
                if num_markers == 0:
                    raise ValueError("Number of markers cannot be zero! Hint: The function was called with no inputs?")
                # set all markers to be the first prototype
                self._instancer_manager.GetProtoIndicesAttr().Set([0] * num_markers)
        # set number of markers
        self._count = num_markers

    """
    Helper functions.
    """
    """辅助函数。
    """

    def _add_markers_prototypes(self, markers_cfg: dict[str, sim_utils.SpawnerCfg]):
        """Adds markers prototypes to the scene and sets the markers instancer to use them."""
        """添加标记原型到场景，并设置标记器即时使用。"""
        # add markers based on config
        for name, cfg in markers_cfg.items():
            # resolve prim path
            marker_prim_path = f"{self.prim_path}/{name}"
            # create a child prim for the marker
            marker_prim = cfg.func(prim_path=marker_prim_path, cfg=cfg)
            # make the asset uninstanceable (in case it is)
            # point instancer defines its own prototypes so if an asset is already instanced, this doesn't work.
            self._process_prototype_prim(marker_prim)
            # add child reference to point instancer
            self._instancer_manager.GetPrototypesRel().AddTarget(marker_prim_path)
        # check that we loaded all the prototypes
        prototypes = self._instancer_manager.GetPrototypesRel().GetTargets()
        if len(prototypes) != len(markers_cfg):
            raise RuntimeError(
                f"Failed to load all the prototypes. Expected: {len(markers_cfg)}. Received: {len(prototypes)}."
            )

    def _process_prototype_prim(self, prim: Usd.Prim):
        """Process a prim and its descendants to make them suitable for defining prototypes.

        Point instancer defines its own prototypes so if an asset is already instanced, this doesn't work.
        This function checks if the prim at the specified prim path and its descendants are instanced.
        If so, it makes the respective prim uninstanceable by disabling instancing on the prim.

        Additionally, it makes the prim invisible to secondary rays. This is useful when we do not want
        to see the marker prims on camera images.

        Args:
            prim: The prim to check.
        """
        """处理prim及其后代，使它们适合定义原型。

        如果一个资产已经被实例化了，这就不起作用。
        这项函数检查了指定的prim路径上的prim及其后代是否被实例化。
        如果是这样，它将对应的prim不可实现，通过在prim上禁用实例化。

        此外，它使prim对二次射线不可见。
        这很有用，如果我们不想在相机图像上看到prims标记。

        参数：
            prim: 在prim为了查看。
        """
        # check if prim is valid
        if not prim.IsValid():
            raise ValueError(f"Prim at path '{prim.GetPrimAtPath()}' is not valid.")
        # iterate over all prims under prim-path
        all_prims = [prim]
        while len(all_prims) > 0:
            # get current prim
            child_prim = all_prims.pop(0)
            # check if it is physics body -> if so, remove it
            if child_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                child_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                child_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                child_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                child_prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
            if child_prim.IsA(UsdPhysics.Joint):
                child_prim.GetAttribute("physics:jointEnabled").Set(False)
            # check if prim is instanced -> if so, make it uninstanceable
            if child_prim.IsInstance():
                child_prim.SetInstanceable(False)
            # check if prim is a mesh -> if so, make it invisible to secondary rays
            if child_prim.IsA(UsdGeom.Gprim):
                # invisible to secondary rays such as depth images
                sim_utils.change_prim_property(
                    prop_path=f"{child_prim.GetPrimPath().pathString}.primvars:invisibleToSecondaryRays",
                    value=True,
                    stage=prim.GetStage(),
                    type_to_create_if_not_exist=Sdf.ValueTypeNames.Bool,
                )
            # add children to list
            all_prims += child_prim.GetChildren()

        # remove any physics on the markers because they are only for visualization!
        physx_utils.removeRigidBodySubtree(prim)
