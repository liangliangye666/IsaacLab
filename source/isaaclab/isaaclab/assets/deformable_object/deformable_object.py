# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager
from pxr import PhysxSchema, UsdShade

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers

from ..asset_base import AssetBase
from .deformable_object_data import DeformableObjectData

if TYPE_CHECKING:
    from .deformable_object_cfg import DeformableObjectCfg

# import logger
logger = logging.getLogger(__name__)


class DeformableObject(AssetBase):
    """A deformable object asset class.

    Deformable objects are assets that can be deformed in the simulation. They are typically used for
    soft bodies, such as stuffed animals and food items.

    Unlike rigid object assets, deformable objects have a more complex structure and require additional
    handling for simulation. The simulation of deformable objects follows a finite element approach, where
    the object is discretized into a mesh of nodes and elements. The nodes are connected by elements, which
    define the material properties of the object. The nodes can be moved and deformed, and the elements
    respond to these changes.

    The state of a deformable object comprises of its nodal positions and velocities, and not the object's root
    position and orientation. The nodal positions and velocities are in the simulation frame.

    Soft bodies can be `partially kinematic`_, where some nodes are driven by kinematic targets, and the rest are
    simulated. The kinematic targets are the desired positions of the nodes, and the simulation drives the nodes
    towards these targets. This is useful for partial control of the object, such as moving a stuffed animal's
    head while the rest of the body is simulated.

    .. attention::
        This class is experimental and subject to change due to changes on the underlying PhysX API on which
        it depends. We will try to maintain backward compatibility as much as possible but some changes may be
        necessary.

    .. _partially kinematic: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html#kinematic-soft-bodies
    """
    """变形物体资产类别。

    可变形的物体是可以在仿真中变形的资产。
    它们通常用于软体，如料动物和食物。

    与硬体物体资产不同，可变化的物体结构更复杂，需要额外的处理来仿真。
    可变形物体的仿真遵循有限元素方法，该物体被分离为节点和元素的网格。
    这些节点由元素连接，它们定义对象的物质性质。
    节点可以移动和变形，元素对这些变化作出反应。

    变形物体的状态包括其节点位置和速度，而不是物体的根位置和方向。
    结点位置和速度在仿真框架中。

    软体可以是`partially kinematic`_，其中一些节点由动态目标驱动，其余则是仿真的。
    动态目标是节点所需的位置，仿真驱动节点向这些目标。
    这对于对象的部分控制是有用的，例如在仿真身体的其他部分时移动头。

    .. 注意::
        这类是实验性的，由于其依赖的底层PhysX API的变化而可能发生变化。
        我们将尽可能努力保持后退兼容性，

    .. _partially kinematic: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html#kinematic-soft-bodies
    """

    cfg: DeformableObjectCfg
    """Configuration instance for the deformable object."""
    """对可变化的对象的配置实例。"""

    def __init__(self, cfg: DeformableObjectCfg):
        """Initialize the deformable object.

        Args:
            cfg: A configuration instance.
        """
        """启动可变的对象。

        参数：
            cfg: 一个配置实例。
        """
        super().__init__(cfg)

    """
    Properties
    """
    """产品
    """

    @property
    def data(self) -> DeformableObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single deformable body.
        """
        """资产中的尸体数量。

        这总是1因为每个对象都是一个可变的体。
        """
        return 1

    @property
    def root_physx_view(self) -> physx.SoftBodyView:
        """Deformable body view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        """对资产的可变体视图 (PhysX)。

        说明：
            用这种观点谨慎。
            它需要以特定的方式处理子。
        """
        return self._root_physx_view

    @property
    def material_physx_view(self) -> physx.SoftBodyMaterialView | None:
        """Deformable material view for the asset (PhysX).

        This view is optional and may not be available if the material is not bound to the deformable body.
        If the material is not available, then the material properties will be set to default values.

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        """资产可变形材料视图 (PhysX)。

        如果材料没有与可变化体相结合，则这种视图是可选的，并且可能无法使用。
        如果材料不存在，则材料属性将设置为默认值。

        说明：
            用这种观点谨慎。
            它需要以特定的方式处理子。
        """
        return self._material_physx_view

    @property
    def max_sim_elements_per_body(self) -> int:
        """The maximum number of simulation mesh elements per deformable body."""
        """每个可变体的仿真网格元素的最大数量。"""
        return self.root_physx_view.max_sim_elements_per_body

    @property
    def max_collision_elements_per_body(self) -> int:
        """The maximum number of collision mesh elements per deformable body."""
        """每个可变体的碰撞网元素的最大数量。"""
        return self.root_physx_view.max_elements_per_body

    @property
    def max_sim_vertices_per_body(self) -> int:
        """The maximum number of simulation mesh vertices per deformable body."""
        """每个可变体的仿真网顶数量最大。"""
        return self.root_physx_view.max_sim_vertices_per_body

    @property
    def max_collision_vertices_per_body(self) -> int:
        """The maximum number of collision mesh vertices per deformable body."""
        """每个可变体的碰撞网顶数最大。"""
        return self.root_physx_view.max_vertices_per_body

    """
    Operations.
    """
    """操作。
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # Think: Should we reset the kinematic targets when resetting the object?
        #  This is not done in the current implementation. We assume users will reset the kinematic targets.
        pass

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Write to simulation.
    """
    """操作 - 写入仿真。
    """

    def write_nodal_state_to_sim(self, nodal_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the nodal state over selected environment indices into the simulation.

        The nodal state comprises of the nodal positions and velocities. Since these are nodes, the velocity only has
        a translational component. All the quantities are in the simulation frame.

        Args:
            nodal_state: Nodal state in simulation frame.
                Shape is (len(env_ids), max_sim_vertices_per_body, 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置了选择的环境索引上的节点状态。

        节点状态包括节点位置和速度。
        由于这些是节点，速度只有一个转换成分。
        所有数量都在仿真框架中。

        参数：
            nodal_state: 在仿真框架中的点状态。
                         形状是 (len(env_ids)，max_sim_vertices_per_body，6)。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        # set into simulation
        self.write_nodal_pos_to_sim(nodal_state[..., :3], env_ids=env_ids)
        self.write_nodal_velocity_to_sim(nodal_state[..., 3:], env_ids=env_ids)

    def write_nodal_pos_to_sim(self, nodal_pos: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the nodal positions over selected environment indices into the simulation.

        The nodal position comprises of individual nodal positions of the simulation mesh for the deformable body.
        The positions are in the simulation frame.

        Args:
            nodal_pos: Nodal positions in simulation frame.
                Shape is (len(env_ids), max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选择的环境索引上的节点位置。

        结点位置包括可变体仿真网格的单个结点位置。
        位置在仿真框架中。

        参数：
            nodal_pos: 仿真框架中的点位置。
                       形状是 (len(env_ids)，max_sim_vertices_per_body，3)。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.nodal_pos_w[env_ids] = nodal_pos.clone()
        # set into simulation
        self.root_physx_view.set_sim_nodal_positions(self._data.nodal_pos_w, indices=physx_env_ids)

    def write_nodal_velocity_to_sim(self, nodal_vel: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the nodal velocity over selected environment indices into the simulation.

        The nodal velocity comprises of individual nodal velocities of the simulation mesh for the deformable
        body. Since these are nodes, the velocity only has a translational component. The velocities are in the
        simulation frame.

        Args:
            nodal_vel: Nodal velocities in simulation frame.
                Shape is (len(env_ids), max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选择的环境索引上的节点速度。

        结点速度包括可变体仿真网的单个结点速度。
        由于这些是节点，速度只有一个转换成分。
        速度在仿真框架中。

        参数：
            nodal_vel: 在仿真框架中的节点速度。
                       形状是 (len(env_ids)，max_sim_vertices_per_body，3)。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.nodal_vel_w[env_ids] = nodal_vel.clone()
        # set into simulation
        self.root_physx_view.set_sim_nodal_velocities(self._data.nodal_vel_w, indices=physx_env_ids)

    def write_nodal_kinematic_target_to_sim(self, targets: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the kinematic targets of the simulation mesh for the deformable bodies indicated by the indices.

        The kinematic targets comprise of individual nodal positions of the simulation mesh for the deformable body
        and a flag indicating whether the node is kinematically driven or not. The positions are in the simulation
        frame.

        Note:
            The flag is set to 0.0 for kinematically driven nodes and 1.0 for free nodes.

        Args:
            targets: The kinematic targets comprising of nodal positions and flags.
                Shape is (len(env_ids), max_sim_vertices_per_body, 4).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """设置仿真网的动态目标，用于索引所指定的可变体。

        动态目标包括可变体仿真网格的单个节点位置和一个标志表明节点是否动态驱动。
        位置在仿真框架中。

        说明：
            旗为动态驱动节点设置为0.0和自由节点设置为1.0。

        参数：
            targets: 包含节点位置和标志的动态目标。
                     形状是 (len(env_ids)，max_sim_vertices_per_body，4)。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # store into internal buffers
        self._data.nodal_kinematic_target[env_ids] = targets.clone()
        # set into simulation
        self.root_physx_view.set_sim_kinematic_targets(self._data.nodal_kinematic_target, indices=physx_env_ids)

    """
    Operations - Helper.
    """
    """运营 - 助理。
    """

    def transform_nodal_pos(
        self, nodal_pos: torch.tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Transform the nodal positions based on the pose transformation.

        This function computes the transformation of the nodal positions based on the pose transformation.
        It multiplies the nodal positions with the rotation matrix of the pose and adds the translation.
        Internally, it calls the :meth:`isaaclab.utils.math.transform_points` function.

        Args:
            nodal_pos: The nodal positions in the simulation frame. Shape is (N, max_sim_vertices_per_body, 3).
            pos: The position transformation. Shape is (N, 3).
                Defaults to None, in which case the position is assumed to be zero.
            quat: The orientation transformation as quaternion (w, x, y, z). Shape is (N, 4).
                Defaults to None, in which case the orientation is assumed to be identity.

        Returns:
            The transformed nodal positions. Shape is (N, max_sim_vertices_per_body, 3).
        """
        """根据姿势转变，转换节点位置。

        这个函数根据姿势转换计算了节点位置的转变。
        它将节点位置乘以姿势的旋转矩阵并添加翻译。
        在内部，它调用:meth:`isaaclab.utils.math.transform_points`函数。

        参数：
            nodal_pos: 在仿真框架中的节点位置。
                       形状是 (N，max_sim_vertices_per_body，3)。
            pos: 位置转变。
                 形状是 (N， 3)。
                 在 None 时的默认值，在这种情况下，假设位置为零。
            quat: 方向转变为四元数 (w，x，y，z)。
                  形状是 (N， 4)。
                  在 None 时的默认状态下，在这种情况下，取向被认为是身份。

        返回：
            转变的节点位置。
            形状是 (N，max_sim_vertices_per_body，3)。
        """
        # offset the nodal positions to center them around the origin
        mean_nodal_pos = nodal_pos.mean(dim=1, keepdim=True)
        nodal_pos = nodal_pos - mean_nodal_pos
        # transform the nodal positions based on the pose around the origin
        return math_utils.transform_points(nodal_pos, pos, quat) + mean_nodal_pos

    """
    Internal helper.
    """
    """内部助理。
    """

    def _initialize_impl(self):
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        # find deformable root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI),
            traverse_instance_prims=False,
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a deformable body when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim has 'PhysxSchema.PhysxDeformableBodyAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single deformable body when resolving '{self.cfg.prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one deformable body in the prim path tree."
            )
        # we only need the first one from the list
        root_prim = root_prims[0]

        # find deformable material prims
        material_prim = None
        # obtain material prim from the root prim
        # note: here we assume that all the root prims have their material prims at similar paths
        #   and we only need to find the first one. This may not be the case for all scenarios.
        #   However, the checks in that case get cumbersome and are not included here.
        if root_prim.HasAPI(UsdShade.MaterialBindingAPI):
            # check the materials that are bound with the purpose 'physics'
            material_paths = UsdShade.MaterialBindingAPI(root_prim).GetDirectBindingRel("physics").GetTargets()
            # iterate through targets and find the deformable body material
            if len(material_paths) > 0:
                for mat_path in material_paths:
                    mat_prim = root_prim.GetStage().GetPrimAtPath(mat_path)
                    if mat_prim.HasAPI(PhysxSchema.PhysxDeformableBodyMaterialAPI):
                        material_prim = mat_prim
                        break
        if material_prim is None:
            logger.info(
                f"Failed to find a deformable material binding for '{root_prim.GetPath().pathString}'."
                " The material properties will be set to default values and are not modifiable at runtime."
                " If you want to modify the material properties, please ensure that the material is bound"
                " to the deformable body."
            )

        # resolve root path back into regex expression
        # -- root prim expression
        root_prim_path = root_prim.GetPath().pathString
        root_prim_path_expr = self.cfg.prim_path + root_prim_path[len(template_prim_path) :]
        # -- object view
        self._root_physx_view = self._physics_sim_view.create_soft_body_view(root_prim_path_expr.replace(".*", "*"))

        # Return if the asset is not found
        if self._root_physx_view._backend is None:
            raise RuntimeError(f"Failed to create deformable body at: {self.cfg.prim_path}. Please check PhysX logs.")

        # resolve material path back into regex expression
        if material_prim is not None:
            # -- material prim expression
            material_prim_path = material_prim.GetPath().pathString
            # check if the material prim is under the template prim
            # if not then we are assuming that the single material prim is used for all the deformable bodies
            if template_prim_path in material_prim_path:
                material_prim_path_expr = self.cfg.prim_path + material_prim_path[len(template_prim_path) :]
            else:
                material_prim_path_expr = material_prim_path
            # -- material view
            self._material_physx_view = self._physics_sim_view.create_soft_body_material_view(
                material_prim_path_expr.replace(".*", "*")
            )
        else:
            self._material_physx_view = None

        # log information about the deformable body
        logger.info(f"Deformable body initialized at: {root_prim_path_expr}")
        logger.info(f"Number of instances: {self.num_instances}")
        logger.info(f"Number of bodies: {self.num_bodies}")
        if self._material_physx_view is not None:
            logger.info(f"Deformable material initialized at: {material_prim_path_expr}")
            logger.info(f"Number of instances: {self._material_physx_view.count}")
        else:
            logger.info("No deformable material found. Material properties will be set to default values.")

        # container for data access
        self._data = DeformableObjectData(self.root_physx_view, self.device)

        # create buffers
        self._create_buffers()
        # update the deformable body data
        self.update(0.0)

        # Initialize debug visualization handle
        if self._debug_vis_handle is None:
            # set initial state of debug visualization
            self.set_debug_vis(self.cfg.debug_vis)

    def _create_buffers(self):
        """Create buffers for storing data."""
        """创建存储数据的缓冲器。"""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)

        # default state
        # we use the initial nodal positions at spawn time as the default state
        # note: these are all in the simulation frame
        nodal_positions = self.root_physx_view.get_sim_nodal_positions()
        nodal_velocities = torch.zeros_like(nodal_positions)
        self._data.default_nodal_state_w = torch.cat((nodal_positions, nodal_velocities), dim=-1)

        # kinematic targets
        self._data.nodal_kinematic_target = self.root_physx_view.get_sim_kinematic_targets()
        # set all nodes as non-kinematic targets by default
        self._data.nodal_kinematic_target[..., -1] = 1.0

    """
    Internal simulation callbacks.
    """
    """内部仿真回调。
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check where to visualize
        targets_enabled = self.data.nodal_kinematic_target[:, :, 3] == 0.0
        num_enabled = int(torch.sum(targets_enabled).item())
        # get positions if any targets are enabled
        if num_enabled == 0:
            # create a marker below the ground
            positions = torch.tensor([[0.0, 0.0, -10.0]], device=self.device)
        else:
            positions = self.data.nodal_kinematic_target[targets_enabled][..., :3]
        # show target visualizer
        self.target_visualizer.visualize(positions)

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        """破坏场景元素。"""
        # call parent
        super()._invalidate_initialize_callback(event)
        self._root_physx_view = None
