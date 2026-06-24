# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.camera import Camera, TiledCamera
from isaaclab.sensors.sensor_base import SensorBase

from .visuotactile_render import GelsightRender
from .visuotactile_sensor_data import VisuoTactileSensorData

if TYPE_CHECKING:
    from .visuotactile_sensor_cfg import VisuoTactileSensorCfg

import trimesh

logger = logging.getLogger(__name__)


class VisuoTactileSensor(SensorBase):
    r"""A tactile sensor for both camera-based and force field tactile sensing.

    This sensor provides:
    1. Camera-based tactile sensing: depth images from tactile surface
    2. Force field tactile sensing: Penalty-based normal and shear forces using SDF queries

    The sensor can be configured to use either or both sensing modalities.

    **Computation Pipeline:**
        Camera-based sensing computes depth differences from a nominal (no-contact) baseline and
        processes them through the tac-sl GelSight renderer to produce realistic tactile images.

        Force field sensing queries Signed Distance Fields (SDF) to compute penetration depths,
        then applies penalty-based spring-damper models
        (:math:`F_n = k_n \cdot \text{depth}`, :math:`F_t = \min(k_t \cdot \|v_t\|, \mu \cdot F_n)`)
        to compute normal and shear forces at discrete tactile points.

    **Example Usage:**
        For a complete working example, see: ``scripts/demos/sensors/tacsl/tacsl_example.py``

    **Current Limitations:**
        - SDF collision meshes must be pre-computed and objects specified before simulation starts
        - Force field computation requires specific rigid body and mesh configurations
        - No support for dynamic addition/removal of interacting objects during runtime

    Configuration Requirements:
        The following requirements must be satisfied for proper sensor operation:

        **Camera Tactile Imaging**
            If ``enable_camera_tactile=True``, a valid ``camera_cfg`` (TiledCameraCfg) must be
            provided with appropriate camera parameters.

        **Force Field Computation**
            If ``enable_force_field=True``, the following parameters are required:

            * ``contact_object_prim_path_expr`` - Prim path expression to find the contact object prim

        **SDF Computation**
            When force field computation is enabled, penalty-based normal and shear forces are
            computed using Signed Distance Field (SDF) queries. To achieve GPU acceleration:

            * Interacting objects should have pre-computed SDF collision meshes
            * An SDFView must be defined during initialization, therefore interacting objects
              should be specified before simulation.

    """
    """触觉传感器用于基于摄像机的触觉传感器和力场触觉传感器。

    这种传感器提供:
    1. 基于摄像头的触觉传感:从触觉表面的深度图像
    2. 动力场触觉传感:使用SDF查询的罚款正常和切割力

    传感器可以配置以使用任何一种或两种传感方式。

    **计算管道:**基于摄像头的传感器计算了名义 (无接触) 基线的深度差异，并通过 tac-sl GelSight 渲染器处理它们以产生现实的触觉图像。

        强势场感测查询 签署距离场 (SDF) 来计算透深度，然后应用基于罚款的弹道损伤模型 (:math:`F_n = k_n \cdot \text{depth}`， :math:`F_t =
        \min(k_t \cdot \|v_t\|， \mu \cdot F_n)`) 来计算在离散触觉点的正常和切割力。

    **例子使用:** 查看完整的工作例子:``scripts/demos/sensors/tacsl/tacsl_example.py``

    **目前的限制:**
        - 在仿真开始之前，必须预先计算SDF碰撞网和指定物体
        - 电力场计算需要特定的固体和网格配置
        - 在运行时间内没有支持动态添加/移除交互对象

    配置要求:为了正常运行传感器，必须满足以下要求:

        **摄像头触觉成像**如果``enable_camera_tactile=True``，必须提供合适的摄像头参数的有效``camera_cfg`` (TiledCameraCfg)。

        **实力场计算**如果``enable_force_field=True``，需要以下参数:

            * ``contact_object_prim_path_expr`` - 寻找接触对象 prim的基本路径表达

        **SDF计算**当启用强力场计算时，使用签署距离场 (SDF) 查询计算基于处罚的正常和切割力。
        为了实现GPU加速:

            * 交互对象应具有预先计算的SDF碰撞网
            * 在启动过程中必须定义SDFView，因此在仿真之前应指定相互作用的对象。
    """

    cfg: VisuoTactileSensorCfg
    """The configuration parameters."""
    """配置参数。"""

    def __init__(self, cfg: VisuoTactileSensorCfg):
        """Initializes the tactile sensor object.

        Args:
            cfg: The configuration parameters.
        """
        """启动触觉传感器对象。

        参数：
            cfg: 配置参数。
        """

        # Create empty variables for storing output data
        self._data: VisuoTactileSensorData = VisuoTactileSensorData()

        # Camera-based tactile sensing
        self._camera_sensor: Camera | TiledCamera | None = None
        self._nominal_tactile: dict | None = None

        # Force field tactile sensing
        self._tactile_pos_local: torch.Tensor | None = None
        self._tactile_quat_local: torch.Tensor | None = None
        self._sdf_object: Any | None = None

        # COMs for velocity correction
        self._elastomer_com_b: torch.Tensor | None = None
        self._contact_object_com_b: torch.Tensor | None = None

        # Physics views
        self._physics_sim_view = None
        self._elastomer_body_view = None
        self._elastomer_tip_view = None
        self._contact_object_body_view = None

        # Visualization
        self._tactile_visualizer: VisualizationMarkers | None = None

        # Tactile points count
        self.num_tactile_points: int = 0

        # Now call parent class constructor
        super().__init__(cfg)

    def __del__(self):
        """Unsubscribes from callbacks and detach from the replicator registry."""
        """退出回调和脱离复制器注册表。"""
        if self._camera_sensor is not None:
            self._camera_sensor.__del__()
        # unsubscribe from callbacks
        super().__del__()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        """Returns: 包含有关实例的信息。"""
        return (
            f"Tactile sensor @ '{self.cfg.prim_path}': \n"
            f"\trender config     : {self.cfg.render_cfg.base_data_path}/{self.cfg.render_cfg.sensor_data_dir_name}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tcamera enabled    : {self.cfg.enable_camera_tactile}\n"
            f"\tforce field enabled: {self.cfg.enable_force_field}\n"
            f"\tnum instances     : {self.num_instances}\n"
        )

    """
    Properties
    """
    """产品
    """

    @property
    def num_instances(self) -> int:
        return self._num_envs

    @property
    def data(self) -> VisuoTactileSensorData:
        # Update sensors if needed
        self._update_outdated_buffers()
        # Return the data
        return self._data

    """
    Operations
    """
    """运营
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the sensor internals."""
        """调整传感器内部。"""
        # reset the timestamps
        super().reset(env_ids)

        # Reset camera sensor if enabled
        if self._camera_sensor:
            self._camera_sensor.reset(env_ids)

    """
    Implementation
    """
    """实施
    """

    def _initialize_impl(self):
        """Initializes the sensor-related handles and internal buffers."""
        """启动与传感器相关的句柄和内部缓冲器。"""
        super()._initialize_impl()

        # Obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # Initialize camera-based tactile sensing
        if self.cfg.enable_camera_tactile:
            self._initialize_camera_tactile()

        # Initialize force field tactile sensing
        if self.cfg.enable_force_field:
            self._initialize_force_field()

        # Initialize visualization
        if self.cfg.debug_vis:
            self._initialize_visualization()

    def get_initial_render(self) -> dict | None:
        """Get the initial tactile sensor render for baseline comparison.

        This method captures the initial state of the tactile sensor when no contact
        is occurring. This baseline is used for computing relative changes during
        tactile interactions.

        .. warning::
            It is the user's responsibility to ensure that the sensor is in a "no contact" state
            when this method is called. If the sensor is in contact with an object, the baseline
            will be incorrect, leading to erroneous tactile readings.

        Returns:
            dict | None: Dictionary containing initial render data with sensor output keys
                        and corresponding tensor values. Returns None if camera tactile
                        sensing is disabled.

        Raises:
            RuntimeError: If camera sensor is not initialized or initial render fails.
        """
        """得到初始触觉传感器进行比较。

        这种方法在没有接触的情况下捕获触觉传感器的初始状态。
        这一基线用于触觉相互作用期间计算相对变化。

        .. 警告::
            当调用这种方法时，用户的责任是确保传感器处于"无接触"状态。
            如果传感器与物体接触，基线将是错误的，导致错误的触觉读数。

        返回：
            给你一个命令.None:含有传感器输出键和相应的光值的初始 data渲染数据的字典。
            如果已禁用摄像头触觉传感器，则返回None。

        异常：
            RuntimeError: 如果摄像头传感器没有启动或初始渲染失败。
        """
        if not self.cfg.enable_camera_tactile:
            return None

        self._camera_sensor.update(dt=0.0)

        # get the initial render
        initial_render = self._camera_sensor.data.output
        if initial_render is None:
            raise RuntimeError("Initial render is None")

        # Store the initial nominal tactile data
        self._nominal_tactile = dict()
        for key, value in initial_render.items():
            self._nominal_tactile[key] = value.clone()

        return self._nominal_tactile

    def _initialize_camera_tactile(self):
        """Initialize camera-based tactile sensing."""
        """启动基于相机的触觉传感。"""
        if self.cfg.camera_cfg is None:
            raise ValueError("Camera configuration is None. Please provide a valid camera configuration.")
        # check image size is consistent with the render config
        if (
            self.cfg.camera_cfg.height != self.cfg.render_cfg.image_height
            or self.cfg.camera_cfg.width != self.cfg.render_cfg.image_width
        ):
            raise ValueError(
                "Camera configuration image size is not consistent with the render config. Camera size:"
                f" {self.cfg.camera_cfg.height}x{self.cfg.camera_cfg.width}, Render config:"
                f" {self.cfg.render_cfg.image_height}x{self.cfg.render_cfg.image_width}"
            )
        # check data types
        if not all(data_type in ["distance_to_image_plane", "depth"] for data_type in self.cfg.camera_cfg.data_types):
            raise ValueError(
                f"Camera configuration data types are not supported. Data types: {self.cfg.camera_cfg.data_types}"
            )
        if self.cfg.camera_cfg.update_period != self.cfg.update_period:
            logger.warning(
                f"Camera configuration update period ({self.cfg.camera_cfg.update_period}) is not equal to sensor"
                f" update period ({self.cfg.update_period}), changing camera update period to match sensor update"
                " period"
            )
            self.cfg.camera_cfg.update_period = self.cfg.update_period

        # gelsightRender
        self._tactile_rgb_render = GelsightRender(self.cfg.render_cfg, device=self.device)

        # Create camera sensor
        self._camera_sensor = TiledCamera(self.cfg.camera_cfg)

        # Initialize camera
        if not self._camera_sensor.is_initialized:
            self._camera_sensor._initialize_impl()
            self._camera_sensor._is_initialized = True

        # Initialize camera buffers
        self._data.tactile_rgb_image = torch.zeros(
            (self._num_envs, self.cfg.camera_cfg.height, self.cfg.camera_cfg.width, 3), device=self._device
        )
        self._data.tactile_depth_image = torch.zeros(
            (self._num_envs, self.cfg.camera_cfg.height, self.cfg.camera_cfg.width, 1), device=self._device
        )

        logger.info("Camera-based tactile sensing initialized.")

    def _initialize_force_field(self):
        """Initialize force field tactile sensing components.

        This method sets up all components required for force field based tactile sensing:

        1. Creates PhysX views for elastomer and contact object rigid bodies
        2. Generates tactile sensing points on the elastomer surface using mesh geometry
        3. Initializes SDF (Signed Distance Field) for collision detection
        4. Creates data buffers for storing force field measurements

        The tactile points are generated by ray-casting onto the elastomer mesh surface
        to create a grid of sensing points that will be used for force computation.

        """
        """启动动动力场触觉传感组件。

        这种方法设置了基于力场触觉传感所需的所有组件:

        1. 创建对弹性体和接触物体的PhysX视图
        2. 通过网格几何学生成弹性质表面上的触觉感觉点
        3. 启动SDF (标记距离场) 进行碰撞检测
        4. 创建数据缓冲器用于存储力场测量

        触觉点通过射线在弹性网表面产生，以创建一个用于力计算的感觉点网格。
        """

        # Generate tactile points on elastomer surface
        self._generate_tactile_points(
            num_divs=list(self.cfg.tactile_array_size),
            margin=getattr(self.cfg, "tactile_margin", 0.003),
            visualize=self.cfg.trimesh_vis_tactile_points,
        )

        self._create_physx_views()

        # Initialize force field data buffers
        self._initialize_force_field_buffers()
        logger.info("Force field tactile sensing initialized.")

    def _create_physx_views(self) -> None:
        """Create PhysX views for contact object and elastomer bodies.

        This method sets up the necessary PhysX views for force field computation:
        1. Creates rigid body view for elastomer
        2. If contact object prim path expression is not None, then:
            a. Finds and validates the object prim and its collision mesh
            b. Creates SDF view for collision detection
            c. Creates rigid body view for object

        """
        """创建接触物体和弹性体的PhysX视图。

        这种方法为力场计算设置了必要的PhysX视图:
        1. 产生体视图
        2. 如果接触对象prim路径表达式不是None，那么: a。 找到和验证对象prim及其碰撞网格 b。 创建对撞检测的SDF视图 c。 创建对象的硬体视图
        """
        elastomer_pattern = self._parent_prims[0].GetPath().pathString.replace("env_0", "env_*")
        self._elastomer_body_view = self._physics_sim_view.create_rigid_body_view([elastomer_pattern])
        # Get elastomer COM for velocity correction
        self._elastomer_com_b = self._elastomer_body_view.get_coms().to(self._device).split([3, 4], dim=-1)[0]

        if self.cfg.contact_object_prim_path_expr is None:
            return

        contact_object_mesh, contact_object_rigid_body = self._find_contact_object_components()
        # Create SDF view for collision detection
        num_query_points = self.cfg.tactile_array_size[0] * self.cfg.tactile_array_size[1]
        mesh_path_pattern = contact_object_mesh.GetPath().pathString.replace("env_0", "env_*")
        self._contact_object_sdf_view = self._physics_sim_view.create_sdf_shape_view(
            mesh_path_pattern, num_query_points
        )

        # Create rigid body views for contact object and elastomer
        body_path_pattern = contact_object_rigid_body.GetPath().pathString.replace("env_0", "env_*")
        self._contact_object_body_view = self._physics_sim_view.create_rigid_body_view([body_path_pattern])
        # Get contact object COM for velocity correction
        self._contact_object_com_b = self._contact_object_body_view.get_coms().to(self._device).split([3, 4], dim=-1)[0]

    def _find_contact_object_components(self) -> tuple[Any, Any]:
        """Find and validate contact object SDF mesh and its parent rigid body.

        This method searches for the contact object prim using the configured filter pattern,
        then locates the first SDF collision mesh within that prim hierarchy and
        identifies its parent rigid body for physics simulation.

        Returns:
            Tuple of (contact_object_mesh, contact_object_rigid_body)
            Returns None if contact object components are not found.

        Note:
            Only SDF meshes are supported for optimal force field computation performance.
            If no SDF mesh is found, the method will log a warning and return None.
        """
        """找到和验证接触物体SDF网格及其母体体。

        这种方法通过配置过器模式搜索接触对象prim，然后在 prim 层次内找到第一个SDF碰撞网格，并识别其母体硬体用于物理仿真。

        返回：
            两倍 (contact_object_mesh， contact_object_rigid_body) 退款None如果没有找到接触物体组件。

        说明：
            只有SDF网格才能实现最佳的力场计算性能。
            如果没有发现SDF网格，该方法将记录警告并返回None。
        """
        # Find the contact object prim using the configured pattern
        contact_object_prim = sim_utils.find_first_matching_prim(self.cfg.contact_object_prim_path_expr)
        if contact_object_prim is None:
            raise RuntimeError(
                f"No contact object prim found matching pattern: {self.cfg.contact_object_prim_path_expr}"
            )

        def is_sdf_mesh(prim: Usd.Prim) -> bool:
            """Check if a mesh prim is configured for SDF approximation."""
            """检查是否设置prim网格为SDF接近。"""
            return (
                prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "sdf"
            )

        # Find the SDF mesh within the contact object
        contact_object_mesh = sim_utils.get_first_matching_child_prim(
            contact_object_prim.GetPath(), predicate=is_sdf_mesh
        )
        if contact_object_mesh is None:
            raise RuntimeError(
                f"No SDF mesh found under contact object at path: {contact_object_prim.GetPath().pathString}"
            )

        def find_parent_rigid_body(prim: Usd.Prim) -> Usd.Prim | None:
            """Find the first parent prim with RigidBodyAPI."""
            """找到第一个母 prim与 RigidBodyAPI。"""
            current_prim = prim
            while current_prim and current_prim.IsValid():
                if current_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    return current_prim
                current_prim = current_prim.GetParent()
                if current_prim.GetPath() == "/":
                    break
            return None

        # Find the rigid body parent of the SDF mesh
        contact_object_rigid_body = find_parent_rigid_body(contact_object_mesh)
        if contact_object_rigid_body is None:
            raise RuntimeError(
                f"No contact object rigid body found for mesh at path: {contact_object_mesh.GetPath().pathString}"
            )

        return contact_object_mesh, contact_object_rigid_body

    def _generate_tactile_points(self, num_divs: list, margin: float, visualize: bool):
        """Generate tactile sensing points from elastomer mesh geometry.

        This method creates a grid of tactile sensing points on the elastomer surface
        by ray-casting onto the mesh geometry. Visual meshes are used for smoother point sampling.

        Args:
            num_divs: Number of divisions [rows, cols] for the tactile grid.
            margin: Margin distance from mesh edges in meters.
            visualize: Whether to show the generated points in trimesh visualization.

        """
        """通过弹体网格几何学生成触觉感觉点。

        这种方法通过射线投射到网格几何学上，在弹体表面创建了触觉感觉点的网格。
        视觉网格用于更平滑的点样本。

        参数：
            num_divs: 触觉网的分区数。
            margin: 从网边缘的边缘距离在米。
            visualize: 如何显示生成的点在trimesh可视化中。
        """

        # Get the elastomer prim path
        elastomer_prim_path = self._parent_prims[0].GetPath().pathString

        def is_visual_mesh(prim) -> bool:
            """Check if a mesh prim has visual properties (visual mesh, not collision mesh)."""
            """检查prim网是否具有视觉性质 (视觉网，不是碰撞网)。"""
            return prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI)

        elastomer_mesh_prim = sim_utils.get_first_matching_child_prim(elastomer_prim_path, predicate=is_visual_mesh)
        if elastomer_mesh_prim is None:
            raise RuntimeError(f"No visual mesh found under elastomer at path: {elastomer_prim_path}")

        logger.info(f"Generating tactile points from USD mesh: {elastomer_mesh_prim.GetPath().pathString}")

        # Extract mesh data
        usd_mesh = UsdGeom.Mesh(elastomer_mesh_prim)
        points = np.asarray(usd_mesh.GetPointsAttr().Get())
        face_indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get())

        # Simple triangulation
        faces = face_indices.reshape(-1, 3)

        # Create bounds
        mesh_bounds = np.array([points.min(axis=0), points.max(axis=0)])

        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=points, faces=faces)

        # Generate grid on elastomer
        elastomer_dims = np.diff(mesh_bounds, axis=0).squeeze()
        slim_axis = np.argmin(elastomer_dims)  # Determine flat axis of elastomer

        # Determine tip direction using dome geometry
        # For dome-shaped elastomers, the center of mass is shifted toward the dome (contact) side
        mesh_center_of_mass = mesh.center_mass[slim_axis]
        bounding_box_center = (mesh_bounds[0, slim_axis] + mesh_bounds[1, slim_axis]) / 2.0

        tip_direction_sign = 1.0 if mesh_center_of_mass > bounding_box_center else -1.0

        # Determine gap between adjacent tactile points
        axis_idxs = list(range(3))
        axis_idxs.remove(int(slim_axis))  # Remove slim idx
        div_sz = (elastomer_dims[axis_idxs] - margin * 2.0) / (np.array(num_divs) + 1)
        tactile_points_dx = min(div_sz)

        # Sample points on the flat plane
        planar_grid_points = []
        center = (mesh_bounds[0] + mesh_bounds[1]) / 2.0
        idx = 0
        for axis_i in range(3):
            if axis_i == slim_axis:
                # On the slim axis, place a point far away so ray is pointing at the elastomer tip
                planar_grid_points.append([tip_direction_sign])
            else:
                axis_grid_points = np.linspace(
                    center[axis_i] - tactile_points_dx * (num_divs[idx] + 1.0) / 2.0,
                    center[axis_i] + tactile_points_dx * (num_divs[idx] + 1.0) / 2.0,
                    num_divs[idx] + 2,
                )
                planar_grid_points.append(axis_grid_points[1:-1])  # Leave out the extreme corners
                idx += 1

        grid_corners = itertools.product(planar_grid_points[0], planar_grid_points[1], planar_grid_points[2])
        grid_corners = np.array(list(grid_corners))

        # Project ray in positive y direction on the mesh
        mesh_data = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        ray_dir = np.array([0, 0, 0])
        ray_dir[slim_axis] = -tip_direction_sign  # Ray points towards elastomer (opposite of tip direction)

        # Handle the ray intersection result
        index_tri, index_ray, locations = mesh_data.intersects_id(
            grid_corners, np.tile([ray_dir], (grid_corners.shape[0], 1)), return_locations=True, multiple_hits=False
        )

        if visualize:
            query_pointcloud = trimesh.PointCloud(locations, colors=(0.0, 0.0, 1.0))
            trimesh.Scene([mesh, query_pointcloud]).show()

        # Sort and store tactile points
        tactile_points = locations[index_ray.argsort()]
        # in the frame of the elastomer
        self._tactile_pos_local = torch.tensor(tactile_points, dtype=torch.float32, device=self._device)
        self.num_tactile_points = self._tactile_pos_local.shape[0]
        if self.num_tactile_points != self.cfg.tactile_array_size[0] * self.cfg.tactile_array_size[1]:
            raise RuntimeError(
                f"Number of tactile points does not match expected: {self.num_tactile_points} !="
                f" {self.cfg.tactile_array_size[0] * self.cfg.tactile_array_size[1]}"
            )

        # Assume tactile frame rotation are all the same
        rotation = torch.tensor([0, 0, -torch.pi], device=self._device)
        self._tactile_quat_local = (
            math_utils.quat_from_euler_xyz(rotation[0], rotation[1], rotation[2])
            .unsqueeze(0)
            .repeat(len(tactile_points), 1)
        )

        logger.info(f"Generated {len(tactile_points)} tactile points from USD mesh using ray casting")

    def _initialize_force_field_buffers(self):
        """Initialize data buffers for force field sensing."""
        """启动数据缓冲器用于强力场传感。"""
        num_pts = self.num_tactile_points

        # Initialize force field data tensors
        self._data.tactile_points_pos_w = torch.zeros((self._num_envs, num_pts, 3), device=self._device)
        self._data.tactile_points_quat_w = torch.zeros((self._num_envs, num_pts, 4), device=self._device)
        self._data.penetration_depth = torch.zeros((self._num_envs, num_pts), device=self._device)
        self._data.tactile_normal_force = torch.zeros((self._num_envs, num_pts), device=self._device)
        self._data.tactile_shear_force = torch.zeros((self._num_envs, num_pts, 2), device=self._device)
        # Pre-compute expanded tactile point tensors to avoid repeated unsqueeze/expand operations
        self._tactile_pos_expanded = self._tactile_pos_local.unsqueeze(0).expand(self._num_envs, -1, -1)
        self._tactile_quat_expanded = self._tactile_quat_local.unsqueeze(0).expand(self._num_envs, -1, -1)

    def _initialize_visualization(self):
        """Initialize visualization markers for tactile points."""
        """启动触觉点的可视化标记。"""
        if self.cfg.visualizer_cfg:
            self._visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data.

        This method updates both camera-based and force field tactile sensing data
        for the specified environments.

        Args:
            env_ids: Sequence of environment indices to update. If length equals
                    total number of environments, all environments are updated.
        """
        """填充传感器数据的缓冲器。

        这种方法更新了基于摄像头和力场触觉传感数据
        for the specified environments.

        参数：
            env_ids: 更新环境索引的序列。
                     如果长度等于环境的总数，则会更新所有环境。
        """
        # Convert to proper indices for internal methods
        if len(env_ids) == self._num_envs:
            internal_env_ids = slice(None)
        else:
            internal_env_ids = env_ids

        # Update camera-based tactile data
        if self.cfg.enable_camera_tactile:
            self._update_camera_tactile(internal_env_ids)

        # Update force field tactile data
        if self.cfg.enable_force_field:
            self._update_force_field(internal_env_ids)

    def _update_camera_tactile(self, env_ids: Sequence[int] | slice):
        """Update camera-based tactile sensing data.

        This method updates the camera sensor and processes the depth information
        to compute tactile measurements. It computes the difference from the nominal
        (no-contact) state and renders it using the GelSight tactile renderer.

        Args:
            env_ids: Environment indices or slice to update. Can be a sequence of
                    integers or a slice object for batch processing.
        """
        """更新基于摄像头的触觉传感数据。

        这种方法会更新摄像头传感器，并处理深度信息来计算触觉测量。
        它计算了与名义 (无接触) 状态的差异，并使用GelSight触觉渲染器进行渲染。

        参数：
            env_ids: 环境索引或切片更新。
                     可以是整数序列或用于批量处理的切片对象。
        """
        if self._nominal_tactile is None:
            raise RuntimeError("Nominal tactile is not set. Please call get_initial_render() first.")
        # Update camera sensor
        self._camera_sensor.update(self._sim_physics_dt)

        # Get camera data
        camera_data = self._camera_sensor.data

        # Check for either distance_to_image_plane or depth (they are equivalent)
        depth_key = None
        if "distance_to_image_plane" in camera_data.output:
            depth_key = "distance_to_image_plane"
        elif "depth" in camera_data.output:
            depth_key = "depth"

        if depth_key:
            self._data.tactile_depth_image[env_ids] = camera_data.output[depth_key][env_ids].clone()
            diff = self._nominal_tactile[depth_key][env_ids] - self._data.tactile_depth_image[env_ids]
            self._data.tactile_rgb_image[env_ids] = self._tactile_rgb_render.render(diff.squeeze(-1))

    #########################################################################################
    # Force field tactile sensing
    #########################################################################################

    def _update_force_field(self, env_ids: Sequence[int] | slice):
        """Update force field tactile sensing data.

        This method computes penalty-based tactile forces using Signed Distance Field (SDF)
        queries. It transforms tactile points to contact object local coordinates, queries the SDF of the
        contact object for collision detection, and computes normal and shear forces based on
        penetration depth and relative velocities.

        Args:
            env_ids: Environment indices or slice to update. Can be a sequence of
                    integers or a slice object for batch processing.

        Note:
            Requires both elastomer and contact object body views to be initialized. Returns
            early if tactile points or body views are not available.
        """
        """更新动力场触觉传感数据。

        这种方法通过签署距离场 (SDF) 查询计算基于惩罚的触动力。
        它将触觉点转换为接触物体本地坐标，查询接触物体的SDF用于碰撞检测，并根据透深度和相对速度计算正常和切割力。

        参数：
            env_ids: 环境索引或切片更新。
                     可以是整数序列或用于批量处理的切片对象。

        说明：
            需要启动弹性体和接触物体体视图。
            如果没有触觉点或身体视觉，
        """
        # Step 1: Get elastomer pose and precompute pose components
        elastomer_pos_w, elastomer_quat_w = self._elastomer_body_view.get_transforms().split([3, 4], dim=-1)
        elastomer_quat_w = math_utils.convert_quat(elastomer_quat_w, to="wxyz")

        # Transform tactile points to world coordinates, used for visualization
        self._transform_tactile_points_to_world(elastomer_pos_w, elastomer_quat_w)

        # earlly return if contact object body view is not available
        # this could happen if the contact object is not specified when tactile_points are required for visualization
        if self._contact_object_body_view is None:
            return

        # Step 2: Transform tactile points to contact object local frame for SDF queries
        contact_object_pos_w, contact_object_quat_w = self._contact_object_body_view.get_transforms().split(
            [3, 4], dim=-1
        )
        contact_object_quat_w = math_utils.convert_quat(contact_object_quat_w, to="wxyz")

        world_tactile_points = self._data.tactile_points_pos_w
        points_contact_object_local, contact_object_quat_inv = self._transform_points_to_contact_object_local(
            world_tactile_points, contact_object_pos_w, contact_object_quat_w
        )

        # Step 3: Query SDF for collision detection
        sdf_values_and_gradients = self._contact_object_sdf_view.get_sdf_and_gradients(points_contact_object_local)
        sdf_values = sdf_values_and_gradients[..., -1]  # Last component is SDF value
        sdf_gradients = sdf_values_and_gradients[..., :-1]  # First 3 components are gradients

        # Step 4: Compute tactile forces from SDF data
        self._compute_tactile_forces_from_sdf(
            points_contact_object_local,
            sdf_values,
            sdf_gradients,
            contact_object_pos_w,
            contact_object_quat_w,
            elastomer_quat_w,
            env_ids,
        )

    def _transform_tactile_points_to_world(self, pos_w: torch.Tensor, quat_w: torch.Tensor):
        """Transform tactile points from local to world coordinates.

        Args:
            pos_w: Elastomer positions in world frame. Shape: (num_envs, 3)
            quat_w: Elastomer quaternions in world frame. Shape: (num_envs, 4)
        """
        """从本地坐标转换为世界坐标。

        参数：
            pos_w: 在世界框架中，
                   形状: (num_envs， 3)
            quat_w: 在世界框架中。
                    形状: (num_envs， 4)
        """
        num_pts = self.num_tactile_points

        quat_expanded = quat_w.unsqueeze(1).expand(-1, num_pts, -1)
        pos_expanded = pos_w.unsqueeze(1).expand(-1, num_pts, -1)

        # Apply transformation
        tactile_pos_w = math_utils.quat_apply(quat_expanded, self._tactile_pos_expanded) + pos_expanded
        tactile_quat_w = math_utils.quat_mul(quat_expanded, self._tactile_quat_expanded)

        # Store in data
        self._data.tactile_points_pos_w = tactile_pos_w
        self._data.tactile_points_quat_w = tactile_quat_w

    def _transform_points_to_contact_object_local(
        self, world_points: torch.Tensor, contact_object_pos_w: torch.Tensor, contact_object_quat_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimized version: Transform world coordinates to contact object local frame.

        Args:
            world_points: Points in world coordinates. Shape: (num_envs, num_points, 3)
            contact_object_pos_w: Contact object positions in world frame. Shape: (num_envs, 3)
            contact_object_quat_w: Contact object quaternions in world frame. Shape: (num_envs, 4)

        Returns:
            Points in contact object local coordinates and inverse quaternions
        """
        """优化版本:将世界坐标转换为接触物体本地框架。

        参数：
            world_points: 在世界坐标中的点。
                          形状: (num_envs，num_points，3)
            contact_object_pos_w: 在世界框架中接触物体的位置。
                                  形状: (num_envs， 3)
            contact_object_quat_w: 在世界框架中接触物体四元数。
                                   形状: (num_envs， 4)

        返回：
            接触物体中的点，本地坐标和逆四角
        """
        # Get inverse transformation (per environment)
        # wxyz in torch
        contact_object_quat_inv, contact_object_pos_inv = torch_utils.tf_inverse(
            contact_object_quat_w, contact_object_pos_w
        )
        num_pts = self.num_tactile_points

        contact_object_quat_expanded = contact_object_quat_inv.unsqueeze(1).expand(-1, num_pts, 4)
        contact_object_pos_expanded = contact_object_pos_inv.unsqueeze(1).expand(-1, num_pts, 3)

        # Apply transformation
        points_sdf = torch_utils.tf_apply(contact_object_quat_expanded, contact_object_pos_expanded, world_points)

        return points_sdf, contact_object_quat_inv

    def _get_tactile_points_velocities(
        self, linvel_world: torch.Tensor, angvel_world: torch.Tensor, quat_world: torch.Tensor
    ) -> torch.Tensor:
        """Optimized version: Compute tactile point velocities from precomputed velocities.

        Args:
            linvel_world: Elastomer linear velocities. Shape: (num_envs, 3)
            angvel_world: Elastomer angular velocities. Shape: (num_envs, 3)
            quat_world: Elastomer quaternions. Shape: (num_envs, 4)

        Returns:
            Tactile point velocities in world frame. Shape: (num_envs, num_points, 3)
        """
        """优化版本:从预计算速度计算触点速度。

        参数：
            linvel_world: 电阻的线性速度。
                          形状: (num_envs， 3)
            angvel_world: 电阻的角度速度
                          形状: (num_envs， 3)
            quat_world: 黄四季。
                        形状: (num_envs， 4)

        返回：
            在世界框架中的触觉点速度。
            形状: (num_envs，num_points，3)
        """
        num_pts = self.num_tactile_points

        # Pre-expand all required tensors once
        quat_expanded = quat_world.unsqueeze(1).expand(-1, num_pts, 4)
        tactile_pos_expanded = self._tactile_pos_expanded

        # Transform local positions to world frame relative vectors
        tactile_pos_world_relative = math_utils.quat_apply(quat_expanded, tactile_pos_expanded)

        # Compute velocity due to angular motion: ω × r
        angvel_expanded = angvel_world.unsqueeze(1).expand(-1, num_pts, 3)
        angular_velocity_contribution = torch.cross(angvel_expanded, tactile_pos_world_relative, dim=-1)

        # Add linear velocity contribution
        linvel_expanded = linvel_world.unsqueeze(1).expand(-1, num_pts, 3)
        tactile_velocity_world = angular_velocity_contribution + linvel_expanded

        return tactile_velocity_world

    def _compute_tactile_forces_from_sdf(
        self,
        points_contact_object_local: torch.Tensor,
        sdf_values: torch.Tensor,
        sdf_gradients: torch.Tensor,
        contact_object_pos_w: torch.Tensor,
        contact_object_quat_w: torch.Tensor,
        elastomer_quat_w: torch.Tensor,
        env_ids: Sequence[int] | slice,
    ) -> None:
        """Optimized version: Compute tactile forces from SDF values using precomputed parameters.

        This method now operates directly on the pre-allocated data tensors to avoid
        unnecessary memory allocation and copying.

        Args:
            points_contact_object_local: Points in contact object local frame
            sdf_values: SDF values (negative means penetration)
            sdf_gradients: SDF gradients (surface normals)
            contact_object_pos_w: Contact object positions in world frame
            contact_object_quat_w: Contact object quaternions in world frame
            elastomer_quat_w: Elastomer quaternions
            env_ids: Environment indices being updated

        """
        """优化版本:使用预先计算的参数从SDF值计算触动力。

        这种方法现在直接运行预先分配的数据器，以避免不必要的存储器分配和复制。

        参数：
            points_contact_object_local: 接触对象本地框架中的点
            sdf_values: SDF值 (负值的透值)
            sdf_gradients: SDF梯度 (表面正常)
            contact_object_pos_w: 在世界框架中接触物体的位置
            contact_object_quat_w: 在世界框架中接触物体四元数
            elastomer_quat_w: 子子
            env_ids: 环境索引更新
        """
        depth = self._data.penetration_depth[env_ids]
        tactile_normal_force = self._data.tactile_normal_force[env_ids]
        tactile_shear_force = self._data.tactile_shear_force[env_ids]

        # Clear the output tensors
        tactile_normal_force.zero_()
        tactile_shear_force.zero_()
        depth.zero_()

        # Convert SDF values to penetration depth (positive for penetration)
        depth[:] = torch.clamp(-sdf_values[env_ids], min=0.0)  # Negative SDF means inside (penetrating)

        # Get collision mask for points that are penetrating
        collision_mask = depth > 0.0

        # Use pre-allocated tensors instead of creating new ones
        num_pts = self.num_tactile_points

        if collision_mask.any() or self.cfg.visualize_sdf_closest_pts:
            # Get contact object and elastomer velocities (com velocities)
            contact_object_velocities = self._contact_object_body_view.get_velocities()
            contact_object_linvel_w_com = contact_object_velocities[env_ids, :3]
            contact_object_angvel_w = contact_object_velocities[env_ids, 3:]

            elastomer_velocities = self._elastomer_body_view.get_velocities()
            elastomer_linvel_w_com = elastomer_velocities[env_ids, :3]
            elastomer_angvel_w = elastomer_velocities[env_ids, 3:]

            # Contact object adjustment
            contact_object_com_w_offset = math_utils.quat_apply(
                contact_object_quat_w[env_ids], self._contact_object_com_b[env_ids]
            )
            contact_object_linvel_w = contact_object_linvel_w_com - torch.cross(
                contact_object_angvel_w, contact_object_com_w_offset, dim=-1
            )
            # v_origin = v_com - w x (com_world_offset) where com_world_offset = quat_apply(quat, com_b)
            elastomer_com_w_offset = math_utils.quat_apply(elastomer_quat_w[env_ids], self._elastomer_com_b[env_ids])
            elastomer_linvel_w = elastomer_linvel_w_com - torch.cross(
                elastomer_angvel_w, elastomer_com_w_offset, dim=-1
            )

            # Normalize gradients to get surface normals in local frame
            normals_local = torch.nn.functional.normalize(sdf_gradients[env_ids], dim=-1)

            # Transform normals to world frame (rotate by contact object orientation) - use precomputed quaternions
            contact_object_quat_expanded = contact_object_quat_w[env_ids].unsqueeze(1).expand(-1, num_pts, 4)

            # Apply quaternion transformation
            normals_world = math_utils.quat_apply(contact_object_quat_expanded, normals_local)

            # Compute normal contact force: F_n = k_n * depth
            fc_norm = self.cfg.normal_contact_stiffness * depth
            fc_world = fc_norm.unsqueeze(-1) * normals_world

            # Get tactile point velocities using precomputed velocities
            tactile_velocity_world = self._get_tactile_points_velocities(
                elastomer_linvel_w, elastomer_angvel_w, elastomer_quat_w[env_ids]
            )

            # Use precomputed contact object velocities
            closest_points_sdf = points_contact_object_local[env_ids] + depth.unsqueeze(-1) * normals_local

            if self.cfg.visualize_sdf_closest_pts:
                debug_closest_points_sdf = (
                    points_contact_object_local[env_ids] - sdf_values[env_ids].unsqueeze(-1) * normals_local
                )
                self.debug_closest_points_wolrd = math_utils.quat_apply(
                    contact_object_quat_expanded, debug_closest_points_sdf
                ) + contact_object_pos_w[env_ids].unsqueeze(1).expand(-1, num_pts, 3)

            contact_object_linvel_expanded = contact_object_linvel_w.unsqueeze(1).expand(-1, num_pts, 3)
            contact_object_angvel_expanded = contact_object_angvel_w.unsqueeze(1).expand(-1, num_pts, 3)
            closest_points_vel_world = (
                torch.linalg.cross(
                    contact_object_angvel_expanded,
                    math_utils.quat_apply(contact_object_quat_expanded, closest_points_sdf),
                )
                + contact_object_linvel_expanded
            )

            # Compute relative velocity at contact points
            relative_velocity_world = tactile_velocity_world - closest_points_vel_world

            # Compute tangential velocity (perpendicular to normal)
            vt_world = relative_velocity_world - normals_world * torch.sum(
                normals_world * relative_velocity_world, dim=-1, keepdim=True
            )
            vt_norm = torch.norm(vt_world, dim=-1)

            # Compute friction force: F_t = min(k_t * |v_t|, mu * F_n)
            ft_static_norm = self.cfg.tangential_stiffness * vt_norm
            ft_dynamic_norm = self.cfg.friction_coefficient * fc_norm
            ft_norm = torch.minimum(ft_static_norm, ft_dynamic_norm)

            # Apply friction force opposite to tangential velocity
            ft_world = -ft_norm.unsqueeze(-1) * vt_world / (vt_norm.unsqueeze(-1).clamp(min=1e-9))

            # Total tactile force in world frame
            tactile_force_world = fc_world + ft_world

            # Transform forces to tactile frame
            tactile_force_tactile = math_utils.quat_apply_inverse(
                self._data.tactile_points_quat_w[env_ids], tactile_force_world
            )

            # Extract normal and shear components
            # Assume tactile frame has Z as normal direction
            tactile_normal_force[:] = tactile_force_tactile[..., 2]  # Z component
            tactile_shear_force[:] = tactile_force_tactile[..., :2]  # X,Y components

    #########################################################################################
    # Debug visualization
    #########################################################################################

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects."""
        """设置调试可视化到可视化对象。"""
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if self._tactile_visualizer is None:
                self._tactile_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self._tactile_visualizer.set_visibility(True)
        else:
            if self._tactile_visualizer:
                self._tactile_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback for debug visualization of tactile sensor data.

        This method is called during each simulation step when debug visualization is enabled.
        It visualizes tactile sensing points as 3D markers in the simulation viewport to help
        with debugging and understanding sensor behavior.

        The method handles two visualization modes:

        1. **Standard mode**: Visualizes ``tactile_points_pos_w`` - the world positions of
            tactile sensing points on the sensor surface
        2. **SDF debug mode**: When ``cfg.visualize_sdf_closest_pts`` is True, visualizes
            ``debug_closest_points_wolrd`` - the closest surface points computed during
            SDF-based force calculations
        """
        """调用触觉传感器数据的错误可视化。

        在每个仿真步骤中调用这种方法，
        它可视化触觉感知点作为仿真视角中的3D标记，以帮助
        with debugging and understanding sensor behavior.

        该方法处理两个可视化模式:

        1. **标准模式**:可视化``tactile_points_pos_w`` - 传感器表面触觉感知点的世界位置
        2. **SDF调试模式**:当``cfg.visualize_sdf_closest_pts``是True时，可视化``debug_closest_points_wolrd`` -
           在基于SDF的力量计算中计算的最接近表面点
        """
        # Safety check - return if not properly initialized
        if not hasattr(self, "_tactile_visualizer") or self._tactile_visualizer is None:
            return
        vis_points = None

        if self.cfg.visualize_sdf_closest_pts and hasattr(self, "debug_closest_points_wolrd"):
            vis_points = self.debug_closest_points_wolrd
        else:
            vis_points = self._data.tactile_points_pos_w

        if vis_points is None or vis_points.numel() == 0:
            return

        viz_points = vis_points.view(-1, 3)  # Shape: (num_envs * num_points, 3)

        indices = torch.zeros(viz_points.shape[0], dtype=torch.long, device=self._device)

        marker_scales = torch.ones(viz_points.shape[0], 3, device=self._device)

        # Visualize tactile points
        self._tactile_visualizer.visualize(translations=viz_points, marker_indices=indices, scales=marker_scales)
