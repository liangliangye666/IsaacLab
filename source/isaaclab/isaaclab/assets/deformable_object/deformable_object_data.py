# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import weakref

import torch

import omni.physics.tensors.impl.api as physx

import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import TimestampedBuffer


class DeformableObjectData:
    """Data container for a deformable object.

    This class contains the data for a deformable object in the simulation. The data includes the nodal states of
    the root deformable body in the object. The data is stored in the simulation world frame unless otherwise specified.

    A deformable object in PhysX uses two tetrahedral meshes to represent the object:

    1. **Simulation mesh**: This mesh is used for the simulation and is the one that is deformed by the solver.
    2. **Collision mesh**: This mesh only needs to match the surface of the simulation mesh and is used for
       collision detection.

    The APIs exposed provides the data for both the simulation and collision meshes. These are specified
    by the `sim` and `collision` prefixes in the property names.

    The data is lazily updated, meaning that the data is only updated when it is accessed. This is useful
    when the data is expensive to compute or retrieve. The data is updated when the timestamp of the buffer
    is older than the current simulation timestamp. The timestamp is updated whenever the data is updated.
    """
    """对可变化的物体的数据容器。

    这类包含仿真中可变化的对象的数据。
    数据包括对象中的根可变体的节点状态。
    除非另有说明，则数据存储在仿真世界框架中。

    在 PhysX 中，可变形的对象使用两个四面形网格来表示对象:

    1. **仿真网**:该网用于仿真，是溶剂扭曲的网。
    2. **碰撞网**:该网只需要与仿真网的表面相匹配，用于碰撞检测。

    暴露的APIs提供了仿真和碰撞网格的数据。
    这些在物业名称中的`sim`和`collision`前置符所指定。

    数据更新缓慢，这意味着数据只有在获取时才会更新。
    如果数据计算或检索昂贵，这很有用。
    当缓冲器的时间标比当前仿真时间标更老时，数据会更新。
    每次数据更新时，时刻标签都会更新。
    """

    def __init__(self, root_physx_view: physx.SoftBodyView, device: str):
        """Initializes the deformable object data.

        Args:
            root_physx_view: The root deformable body view of the object.
            device: The device used for processing.
        """
        """启动可变化的对象数据。

        参数：
            root_physx_view: 基因可变化的体体视图。
            device: 用于加工的装置。
        """
        # Set the parameters
        self.device = device
        # Set the root deformable body view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_physx_view: physx.SoftBodyView = weakref.proxy(root_physx_view)

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # Initialize the lazy buffers.
        # -- node state in simulation world frame
        self._nodal_pos_w = TimestampedBuffer()
        self._nodal_vel_w = TimestampedBuffer()
        self._nodal_state_w = TimestampedBuffer()
        # -- mesh element-wise rotations
        self._sim_element_quat_w = TimestampedBuffer()
        self._collision_element_quat_w = TimestampedBuffer()
        # -- mesh element-wise deformation gradients
        self._sim_element_deform_gradient_w = TimestampedBuffer()
        self._collision_element_deform_gradient_w = TimestampedBuffer()
        # -- mesh element-wise stresses
        self._sim_element_stress_w = TimestampedBuffer()
        self._collision_element_stress_w = TimestampedBuffer()

    def update(self, dt: float):
        """Updates the data for the deformable object.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        """更新可变化的对象的数据。

        参数：
            dt: 更新的时间。
                这一定是积极的价值。
        """
        # update the simulation timestamp
        self._sim_timestamp += dt

    ##
    # Defaults.
    ##

    default_nodal_state_w: torch.Tensor = None
    """Default nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.
    Shape is (num_instances, max_sim_vertices_per_body, 6).
    """
    """在仿真世界框架中的默认节点状态``[nodal_pos， nodal_vel]``。
    形状是 (num_instances，max_sim_vertices_per_body， 6)。
    """

    ##
    # Kinematic commands
    ##

    nodal_kinematic_target: torch.Tensor = None
    """Simulation mesh kinematic targets for the deformable bodies.
    Shape is (num_instances, max_sim_vertices_per_body, 4).

    The kinematic targets are used to drive the simulation mesh vertices to the target positions.
    The targets are stored as (x, y, z, is_not_kinematic) where "is_not_kinematic" is a binary
    flag indicating whether the vertex is kinematic or not. The flag is set to 0 for kinematic vertices
    and 1 for non-kinematic vertices.
    """
    """对可变体的仿真网动目标。
    形状是 (num_instances，max_sim_vertices_per_body， 4)。

    动态目标用于将仿真网顶向目标位置驱动。
    目标存储为 (x， y， z， is_not_kinematic)，其中"is_not_kinematic"是一个二进制标志，表明顶点是否动态。
    标志为动态顶点设置为0和非动态顶点设置为1。
    """

    ##
    # Properties.
    ##

    @property
    def nodal_pos_w(self):
        """Nodal positions in simulation world frame. Shape is (num_instances, max_sim_vertices_per_body, 3)."""
        """在仿真世界框架中。
        形状是 (num_instances，max_sim_vertices_per_body， 3)。
        """
        if self._nodal_pos_w.timestamp < self._sim_timestamp:
            self._nodal_pos_w.data = self._root_physx_view.get_sim_nodal_positions()
            self._nodal_pos_w.timestamp = self._sim_timestamp
        return self._nodal_pos_w.data

    @property
    def nodal_vel_w(self):
        """Nodal velocities in simulation world frame. Shape is (num_instances, max_sim_vertices_per_body, 3)."""
        """在仿真世界框架中，
        形状是 (num_instances，max_sim_vertices_per_body， 3)。
        """
        if self._nodal_vel_w.timestamp < self._sim_timestamp:
            self._nodal_vel_w.data = self._root_physx_view.get_sim_nodal_velocities()
            self._nodal_vel_w.timestamp = self._sim_timestamp
        return self._nodal_vel_w.data

    @property
    def nodal_state_w(self):
        """Nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.
        Shape is (num_instances, max_sim_vertices_per_body, 6).
        """
        """结核状态``[nodal_pos， nodal_vel]``在仿真世界框架中。
        形状是 (num_instances，max_sim_vertices_per_body， 6)。
        """
        if self._nodal_state_w.timestamp < self._sim_timestamp:
            self._nodal_state_w.data = torch.cat((self.nodal_pos_w, self.nodal_vel_w), dim=-1)
            self._nodal_state_w.timestamp = self._sim_timestamp
        return self._nodal_state_w.data

    @property
    def sim_element_quat_w(self):
        """Simulation mesh element-wise rotations as quaternions for the deformable bodies in simulation world frame.
        Shape is (num_instances, max_sim_elements_per_body, 4).

        The rotations are stored as quaternions in the order (w, x, y, z).
        """
        """在仿真世界框架中，可变形体的四元数以仿真网格元素进行旋转。
        形状是 (num_instances，max_sim_elements_per_body， 4)。

        旋转按顺序 (w，x，y，z) 存储为四元数。
        """
        if self._sim_element_quat_w.timestamp < self._sim_timestamp:
            # convert from xyzw to wxyz
            quats = self._root_physx_view.get_sim_element_rotations().view(self._root_physx_view.count, -1, 4)
            quats = math_utils.convert_quat(quats, to="wxyz")
            # set the buffer data and timestamp
            self._sim_element_quat_w.data = quats
            self._sim_element_quat_w.timestamp = self._sim_timestamp
        return self._sim_element_quat_w.data

    @property
    def collision_element_quat_w(self):
        """Collision mesh element-wise rotations as quaternions for the deformable bodies in simulation world frame.
        Shape is (num_instances, max_collision_elements_per_body, 4).

        The rotations are stored as quaternions in the order (w, x, y, z).
        """
        """在仿真世界框架中，可变形体的四元数作为碰撞网元素的旋转。
        形状是 (num_instances，max_collision_elements_per_body， 4)。

        旋转按顺序 (w，x，y，z) 存储为四元数。
        """
        if self._collision_element_quat_w.timestamp < self._sim_timestamp:
            # convert from xyzw to wxyz
            quats = self._root_physx_view.get_element_rotations().view(self._root_physx_view.count, -1, 4)
            quats = math_utils.convert_quat(quats, to="wxyz")
            # set the buffer data and timestamp
            self._collision_element_quat_w.data = quats
            self._collision_element_quat_w.timestamp = self._sim_timestamp
        return self._collision_element_quat_w.data

    @property
    def sim_element_deform_gradient_w(self):
        """Simulation mesh element-wise second-order deformation gradient tensors for the deformable bodies
        in simulation world frame. Shape is (num_instances, max_sim_elements_per_body, 3, 3).
        """
        """在仿真世界框架中可变化体的仿真网格元素的第二级变形梯度紧缩器。
        形状是 (num_instances，max_sim_elements_per_body，3，3)。
        """
        if self._sim_element_deform_gradient_w.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            self._sim_element_deform_gradient_w.data = (
                self._root_physx_view.get_sim_element_deformation_gradients().view(
                    self._root_physx_view.count, -1, 3, 3
                )
            )
            self._sim_element_deform_gradient_w.timestamp = self._sim_timestamp
        return self._sim_element_deform_gradient_w.data

    @property
    def collision_element_deform_gradient_w(self):
        """Collision mesh element-wise second-order deformation gradient tensors for the deformable bodies
        in simulation world frame. Shape is (num_instances, max_collision_elements_per_body, 3, 3).
        """
        """在仿真世界框架中可变化体的碰撞网元素的第二级变形梯度子。
        形状是 (num_instances，max_collision_elements_per_body，3，3)。
        """
        if self._collision_element_deform_gradient_w.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            self._collision_element_deform_gradient_w.data = (
                self._root_physx_view.get_element_deformation_gradients().view(self._root_physx_view.count, -1, 3, 3)
            )
            self._collision_element_deform_gradient_w.timestamp = self._sim_timestamp
        return self._collision_element_deform_gradient_w.data

    @property
    def sim_element_stress_w(self):
        """Simulation mesh element-wise second-order Cauchy stress tensors for the deformable bodies
        in simulation world frame. Shape is (num_instances, max_sim_elements_per_body, 3, 3).
        """
        """在仿真世界框架中，可变化体的二级考希压力器。
        形状是 (num_instances，max_sim_elements_per_body，3，3)。
        """
        if self._sim_element_stress_w.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            self._sim_element_stress_w.data = self._root_physx_view.get_sim_element_stresses().view(
                self._root_physx_view.count, -1, 3, 3
            )
            self._sim_element_stress_w.timestamp = self._sim_timestamp
        return self._sim_element_stress_w.data

    @property
    def collision_element_stress_w(self):
        """Collision mesh element-wise second-order Cauchy stress tensors for the deformable bodies
        in simulation world frame. Shape is (num_instances, max_collision_elements_per_body, 3, 3).
        """
        """在仿真世界框架中，可变体的二级考希压力器。
        形状是 (num_instances，max_collision_elements_per_body，3，3)。
        """
        if self._collision_element_stress_w.timestamp < self._sim_timestamp:
            # set the buffer data and timestamp
            self._collision_element_stress_w.data = self._root_physx_view.get_element_stresses().view(
                self._root_physx_view.count, -1, 3, 3
            )
            self._collision_element_stress_w.timestamp = self._sim_timestamp
        return self._collision_element_stress_w.data

    ##
    # Derived properties.
    ##

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position from nodal positions of the simulation mesh for the deformable bodies in simulation
        world frame. Shape is (num_instances, 3).

        This quantity is computed as the mean of the nodal positions.
        """
        """在仿真世界框架中可变形体的仿真网格从节点位置的根位置。
        形状是 (num_instances， 3)。

        这种数量被计算为节点位置的平均值。
        """
        return self.nodal_pos_w.mean(dim=1)

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity from vertex velocities for the deformable bodies in simulation world frame.
        Shape is (num_instances, 3).

        This quantity is computed as the mean of the nodal velocities.
        """
        """在仿真世界框架中的可变体的顶点速度的根速度。
        形状是 (num_instances， 3)。

        这种数量是节点速度的平均值。
        """
        return self.nodal_vel_w.mean(dim=1)
