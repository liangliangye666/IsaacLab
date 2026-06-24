# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import weakref

import torch

import omni.physics.tensors.impl.api as physx

import isaaclab.utils.math as math_utils
from isaaclab.sim.utils.stage import get_current_stage_id
from isaaclab.utils.buffers import TimestampedBuffer


class RigidObjectCollectionData:
    """Data container for a rigid object collection.

    This class contains the data for a rigid object collection in the simulation. The data includes the state of
    all the bodies in the collection. The data is stored in the simulation world frame unless otherwise specified.
    The data is in the order ``(num_instances, num_objects, data_size)``, where data_size is the size of the data.

    For a rigid body, there are two frames of reference that are used:

    - Actor frame: The frame of reference of the rigid body prim. This typically corresponds to the Xform prim
      with the rigid body schema.
    - Center of mass frame: The frame of reference of the center of mass of the rigid body.

    Depending on the settings of the simulation, the actor frame and the center of mass frame may be the same.
    This needs to be taken into account when interpreting the data.

    The data is lazily updated, meaning that the data is only updated when it is accessed. This is useful
    when the data is expensive to compute or retrieve. The data is updated when the timestamp of the buffer
    is older than the current simulation timestamp. The timestamp is updated whenever the data is updated.
    """
    """硬体收集数据容器。

    这类包含在仿真中的硬体集合数据。
    这些数据包括收集中的所有机构的状态。
    除非另有说明，则数据存储在仿真世界框架中。
    数据在 ``(num_instances， num_objects， data_size)`` 顺序中， data_size 是数据的大小。

    对于硬体，使用的两个参考框架:

    - 演员框架:硬体prim的参考框架.这通常与X形式prim相符
      with the rigid body schema.
    - 质量框架的中心:是硬体质量中心的参考框架。

    根据仿真的设置，演员框架和质量框架的中心可能是相同的。
    在解释数据时需要考虑到这一点。

    数据更新缓慢，这意味着数据只有在获取时才会更新。
    如果数据计算或检索昂贵，这很有用。
    当缓冲器的时间标比当前仿真时间标更老时，数据会更新。
    每次数据更新时，时刻标签都会更新。
    """

    def __init__(self, root_physx_view: physx.RigidBodyView, num_objects: int, device: str):
        """Initializes the data.

        Args:
            root_physx_view: The root rigid body view.
            num_objects: The number of objects in the collection.
            device: The device used for processing.
        """
        """启动数据。

        参数：
            root_physx_view: 根固体视图。
            num_objects: 集合中的物品数量
            device: 用于加工的装置。
        """
        # Set the parameters
        self.device = device
        self.num_objects = num_objects
        # Set the root rigid body view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_physx_view: physx.RigidBodyView = weakref.proxy(root_physx_view)
        self.num_instances = self._root_physx_view.count // self.num_objects

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # Obtain global physics sim view
        stage_id = get_current_stage_id()
        physics_sim_view = physx.create_simulation_view("torch", stage_id)
        physics_sim_view.set_subspace_roots("/")
        gravity = physics_sim_view.get_gravity()
        # Convert to direction vector
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)

        # Initialize constants
        self.GRAVITY_VEC_W = gravity_dir.repeat(self.num_instances, self.num_objects, 1)
        self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(
            self.num_instances, self.num_objects, 1
        )

        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._object_link_pose_w = TimestampedBuffer()
        self._object_link_vel_w = TimestampedBuffer()
        # -- com frame w.r.t. link frame
        self._object_com_pose_b = TimestampedBuffer()
        # -- com frame w.r.t. world frame
        self._object_com_pose_w = TimestampedBuffer()
        self._object_com_vel_w = TimestampedBuffer()
        self._object_com_acc_w = TimestampedBuffer()
        # -- combined state(these are cached as they concatenate)
        self._object_state_w = TimestampedBuffer()
        self._object_link_state_w = TimestampedBuffer()
        self._object_com_state_w = TimestampedBuffer()

    def update(self, dt: float):
        """Updates the data for the rigid object collection.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        """更新硬体集合数据。

        参数：
            dt: 更新的时间。
                这一定是积极的价值。
        """
        # update the simulation timestamp
        self._sim_timestamp += dt

    ##
    # Names.
    ##

    object_names: list[str] = None
    """Object names in the order parsed by the simulation view."""
    """仿真视图分析的顺序中对象名称。"""

    ##
    # Defaults.
    ##

    default_object_state: torch.Tensor = None
    """Default object state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame.
    Shape is (num_instances, num_objects, 13).

    The position and quaternion are of each object's rigid body's actor frame. Meanwhile, the linear and
    angular velocities are of the center of mass frame.
    """
    """在本地环境框架中默认对象状态 ``[pos， quat， lin_vel， ang_vel]``。
    形状是 (num_instances，num_objects， 13)。

    每个物体的硬体的演员框架的位置和四元数。
    与此同时，线性和角度速度是质量框架的中心。
    """

    default_mass: torch.Tensor = None
    """Default object mass read from the simulation. Shape is (num_instances, num_objects, 1)."""
    """从仿真中读取的默认对象质量。
    形状是 (num_instances，num_objects， 1)。
    """

    default_inertia: torch.Tensor = None
    """Default object inertia tensor read from the simulation. Shape is (num_instances, num_objects, 9).

    The inertia tensor should be given with respect to the center of mass, expressed in the rigid body's actor frame.
    The values are stored in the order :math:`[I_{xx}, I_{yx}, I_{zx}, I_{xy}, I_{yy}, I_{zy}, I_{xz}, I_{yz}, I_{zz}]`.
    However, due to the symmetry of inertia tensors, row- and column-major orders are equivalent.

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """在仿真中读取的默认对象惯性子。
    形状是 (num_instances，num_objects， 9)。

    should惰性子应与质量中心相比，表达在硬体的演员框架中。
    值存储的顺序:数学:`[I_{xx}， I_{yx}， I_{zx}， I_{xy}， I_{yy}， I_{zy}， I_{xz}， I_{yz}， I_{zz}]`。
    然而，由于惯性子的对称性，排列和柱子主要顺序是等等。

    在初始化时，这个数量从USD方案中解析。
    """

    ##
    # Root state properties.
    ##

    @property
    def object_link_pose_w(self):
        """Object link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, num_objects, 7).

        The position and orientation are of the rigid body's actor frame.
        """
        """在仿真世界框架中，对象链接呈现``[pos， quat]``。
        形状是 (num_instances，num_objects， 7)。

        位置和方向是硬体的演员框架。
        """
        if self._object_link_pose_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._reshape_view_to_data(self._root_physx_view.get_transforms().clone())
            pose[..., 3:7] = math_utils.convert_quat(pose[..., 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._object_link_pose_w.data = pose
            self._object_link_pose_w.timestamp = self._sim_timestamp

        return self._object_link_pose_w.data

    @property
    def object_link_vel_w(self):
        """Object link velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_objects, 6).

        The linear and angular velocities are of the rigid body's actor frame.
        """
        """在仿真世界框架中的物体链接速度``[lin_vel， ang_vel]``。
        形状是 (num_instances，num_objects， 6)。

        线性和角的速度是硬体的演员框架。
        """
        if self._object_link_vel_w.timestamp < self._sim_timestamp:
            # read data from simulation
            velocity = self.object_com_vel_w.clone()
            # adjust linear velocity to link from center of mass
            velocity[..., :3] += torch.linalg.cross(
                velocity[..., 3:], math_utils.quat_apply(self.object_link_quat_w, -self.object_com_pos_b), dim=-1
            )
            # set the buffer data and timestamp
            self._object_link_vel_w.data = velocity
            self._object_link_vel_w.timestamp = self._sim_timestamp

        return self._object_link_vel_w.data

    @property
    def object_com_pose_w(self):
        """Object center of mass pose ``[pos, quat]`` in simulation world frame.
        Shape is (num_instances, num_objects, 7).

        The position and orientation are of the rigid body's center of mass frame.
        """
        """在仿真世界框架中，物体质量中心 ``[pos， quat]``。
        形状是 (num_instances，num_objects， 7)。

        它的位置和方向是硬体质量框架中心。
        """
        if self._object_com_pose_w.timestamp < self._sim_timestamp:
            # adjust pose to center of mass
            pos, quat = math_utils.combine_frame_transforms(
                self.object_link_pos_w, self.object_link_quat_w, self.object_com_pos_b, self.object_com_quat_b
            )
            # set the buffer data and timestamp
            self._object_com_pose_w.data = torch.cat((pos, quat), dim=-1)
            self._object_com_pose_w.timestamp = self._sim_timestamp

        return self._object_com_pose_w.data

    @property
    def object_com_vel_w(self):
        """Object center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_objects, 6).

        The linear and angular velocities are of the rigid body's center of mass frame.
        """
        """在仿真世界框架中的质量速度``[lin_vel， ang_vel]``的对象中心。
        形状是 (num_instances，num_objects， 6)。

        线性和角的速度是固体质量框架的中心。
        """
        if self._object_com_vel_w.timestamp < self._sim_timestamp:
            self._object_com_vel_w.data = self._reshape_view_to_data(self._root_physx_view.get_velocities())
            self._object_com_vel_w.timestamp = self._sim_timestamp

        return self._object_com_vel_w.data

    @property
    def object_state_w(self):
        """Object state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_objects, 13).

        The position and orientation are of the rigid body's actor frame. Meanwhile, the linear and angular
        velocities are of the rigid body's center of mass frame.
        """
        """在仿真世界框架中对象状态``[pos， quat， lin_vel， ang_vel]``。
        形状是 (num_instances，num_objects， 13)。

        位置和方向是硬体的演员框架。
        与此同时，线性和角的速度是固体质量框架的中心。
        """
        if self._object_state_w.timestamp < self._sim_timestamp:
            self._object_state_w.data = torch.cat((self.object_link_pose_w, self.object_com_vel_w), dim=-1)
            self._object_state_w.timestamp = self._sim_timestamp

        return self._object_state_w.data

    @property
    def object_link_state_w(self):
        """Object center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_objects, 13).

        The position, quaternion, and linear/angular velocity are of the rigid body root frame relative to the
        world.
        """
        """在仿真世界框架中的质量状态``[pos， quat， lin_vel， ang_vel]``的对象中心。
        形状是 (num_instances，num_objects， 13)。

        位置，四元数和线性/角的速度是与世界相对的硬体根框架。
        """
        if self._object_link_state_w.timestamp < self._sim_timestamp:
            self._object_link_state_w.data = torch.cat((self.object_link_pose_w, self.object_link_vel_w), dim=-1)
            self._object_link_state_w.timestamp = self._sim_timestamp

        return self._object_link_state_w.data

    @property
    def object_com_state_w(self):
        """Object state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_objects, 13).

        The position, quaternion, and linear/angular velocity are of the rigid body's center of mass frame
        relative to the world. Center of mass frame has the orientation along the principle axes of inertia.
        """
        """在仿真世界框架中对象状态``[pos， quat， lin_vel， ang_vel]``。
        形状是 (num_instances，num_objects， 13)。

        位置，四元数和线性/角的速度是固体质量框架的中心相对于世界。
        质量框架的中心沿着惯性的主要轴的方向。
        """
        if self._object_com_state_w.timestamp < self._sim_timestamp:
            self._object_com_state_w.data = torch.cat((self.object_com_pose_w, self.object_com_vel_w), dim=-1)
            self._object_com_state_w.timestamp = self._sim_timestamp

        return self._object_com_state_w.data

    @property
    def object_com_acc_w(self):
        """Acceleration of all objects. Shape is (num_instances, num_objects, 6).

        This quantity is the acceleration of the rigid bodies' center of mass frame.
        """
        """所有物体的加速。
        形状是 (num_instances，num_objects， 6)。

        这个数量是硬体质量框架中心的加速。
        """
        if self._object_com_acc_w.timestamp < self._sim_timestamp:
            self._object_com_acc_w.data = self._reshape_view_to_data(self._root_physx_view.get_accelerations())
            self._object_com_acc_w.timestamp = self._sim_timestamp
        return self._object_com_acc_w.data

    @property
    def object_com_pose_b(self):
        """Object center of mass pose ``[pos, quat]`` in their respective body's link frame.
        Shape is (num_instances, num_objects, 7).

        The position and orientation are of the rigid body's center of mass frame.
        The orientation is provided in (w, x, y, z) format.
        """
        """物体质量中心在各自身体的链接框架中呈现``[pos， quat]``。
        形状是 (num_instances，num_objects， 7)。

        它的位置和方向是硬体质量框架中心。
        导向提供 (w， x， y， z) 格式。
        """
        if self._object_com_pose_b.timestamp < self._sim_timestamp:
            # obtain the coms
            poses = self._root_physx_view.get_coms().to(self.device)
            poses[:, 3:7] = math_utils.convert_quat(poses[:, 3:7], to="wxyz")
            # read data from simulation
            self._object_com_pose_b.data = self._reshape_view_to_data(poses)
            self._object_com_pose_b.timestamp = self._sim_timestamp

        return self._object_com_pose_b.data

    ##
    # Derived properties.
    ##

    @property
    def projected_gravity_b(self):
        """Projection of the gravity direction on base frame. Shape is (num_instances, num_objects, 3)."""
        """在基架上投射重力方向。
        形状是 (num_instances，num_objects， 3)。
        """
        return math_utils.quat_apply_inverse(self.object_link_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self):
        """Yaw heading of the base frame (in radians). Shape is (num_instances, num_objects,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        """基架的 Yaw方向 (在半径中)。
        形状是 (num_instances，num_objects，)。

        说明：
            这个数量是通过假设基架的前向方向沿着x方向计算的，i.e.:数学:`(1， 0， 0)`。
        """
        forward_w = math_utils.quat_apply(self.object_link_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[..., 1], forward_w[..., 0])

    @property
    def object_link_lin_vel_b(self) -> torch.Tensor:
        """Object link linear velocity in base frame. Shape is (num_instances, num_objects, 3).

        This quantity is the linear velocity of the actor frame of the root rigid body frame with
        respect to the rigid body's actor frame.
        """
        """在基础框架中的对象连接线性速度。
        形状是 (num_instances，num_objects， 3)。

        这个数量是根固体框架的演员框架的线性速度，
        """
        return math_utils.quat_apply_inverse(self.object_link_quat_w, self.object_link_lin_vel_w)

    @property
    def object_link_ang_vel_b(self) -> torch.Tensor:
        """Object link angular velocity in base world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body frame with
        respect to the rigid body's actor frame.
        """
        """在基础世界框架中的对象链接角速度。
        形状是 (num_instances，num_objects， 3)。

        这个数量是根固体框架的演员框架与硬体的演员框架的角速度。
        """
        return math_utils.quat_apply_inverse(self.object_link_quat_w, self.object_link_ang_vel_w)

    @property
    def object_com_lin_vel_b(self) -> torch.Tensor:
        """Object center of mass linear velocity in base frame. Shape is (num_instances, num_objects, 3).

        This quantity is the linear velocity of the center of mass frame of the root rigid body frame with
        respect to the rigid body's actor frame.
        """
        """在基础框架中，质量线性速度的对象中心。
        形状是 (num_instances，num_objects， 3)。

        这个数量是根固体框架的质量框架中心的线性速度与硬体的演员框架相比。
        """
        return math_utils.quat_apply_inverse(self.object_link_quat_w, self.object_com_lin_vel_w)

    @property
    def object_com_ang_vel_b(self) -> torch.Tensor:
        """Object center of mass angular velocity in base world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the angular velocity of the center of mass frame of the root rigid body frame with
        respect to the rigid body's actor frame.
        """
        """在基础世界框架中的质量角速度对象中心。
        形状是 (num_instances，num_objects， 3)。

        这个数量是根固体框架的质量框架中心的角速度与硬体的演员框架相比。
        """
        return math_utils.quat_apply_inverse(self.object_link_quat_w, self.object_com_ang_vel_w)

    ##
    # Sliced properties.
    ##

    @property
    def object_link_pos_w(self) -> torch.Tensor:
        """Object link position in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the position of the actor frame of the rigid bodies.
        """
        """在仿真世界框架中的对象链接位置。
        形状是 (num_instances，num_objects， 3)。

        这种数量是硬体演员框架的位置。
        """
        return self.object_link_pose_w[..., :3]

    @property
    def object_link_quat_w(self) -> torch.Tensor:
        """Object link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, num_objects, 4).

        This quantity is the orientation of the actor frame of the rigid bodies.
        """
        """在仿真世界框架中对象链接导向 (w，x，y，z)。
        形状是 (num_instances，num_objects， 4)。

        这种数量是硬体的演员框架的方向。
        """
        return self.object_link_pose_w[..., 3:7]

    @property
    def object_link_lin_vel_w(self) -> torch.Tensor:
        """Object link linear velocity in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the linear velocity of the rigid bodies' actor frame.
        """
        """在仿真世界框架中的对象链接线性速度。
        形状是 (num_instances，num_objects， 3)。

        这种数量是硬体演员框架的线性速度。
        """
        return self.object_link_vel_w[..., :3]

    @property
    def object_link_ang_vel_w(self) -> torch.Tensor:
        """Object link angular velocity in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the angular velocity of the rigid bodies' actor frame.
        """
        """在仿真世界框架中的对象链接角速度。
        形状是 (num_instances，num_objects， 3)。

        这种数量是硬体演员框架的角速度。
        """
        return self.object_link_vel_w[..., 3:6]

    @property
    def object_com_pos_w(self) -> torch.Tensor:
        """Object center of mass position in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the position of the center of mass frame of the rigid bodies.
        """
        """在仿真世界框架中的物体质位置中心。
        形状是 (num_instances，num_objects， 3)。

        这种数量是硬体质量框架中心的位置。
        """
        return self.object_com_pose_w[..., :3]

    @property
    def object_com_quat_w(self) -> torch.Tensor:
        """Object center of mass orientation (w, x, y, z) in simulation world frame.
        Shape is (num_instances, num_objects, 4).

        This quantity is the orientation of the center of mass frame of the rigid bodies.
        """
        """在仿真世界框架中的质量导向对象中心 (w，x，y，z)。
        形状是 (num_instances，num_objects， 4)。

        这种数量是硬体质量框架中心的方向。
        """
        return self.object_com_pose_w[..., 3:7]

    @property
    def object_com_lin_vel_w(self) -> torch.Tensor:
        """Object center of mass linear velocity in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        """在仿真世界框架中的质量线性速度对象中心。
        形状是 (num_instances，num_objects， 3)。

        这个数量是刚体质量框架中心的线性速度。
        """
        return self.object_com_vel_w[..., :3]

    @property
    def object_com_ang_vel_w(self) -> torch.Tensor:
        """Object center of mass angular velocity in simulation world frame. Shape is (num_instances, num_objects, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        """在仿真世界框架中的质量角速度对象中心。
        形状是 (num_instances，num_objects， 3)。

        这个数量是硬体质量框架中心的角速度。
        """
        return self.object_com_vel_w[..., 3:6]

    @property
    def object_com_lin_acc_w(self) -> torch.Tensor:
        """Object center of mass linear acceleration in simulation world frame.
        Shape is (num_instances, num_objects, 3).

        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        """在仿真世界框架中的质量线性加速对象中心。
        形状是 (num_instances，num_objects， 3)。

        这个数量是刚体质量框架中心的线性加速。
        """
        return self.object_com_acc_w[..., :3]

    @property
    def object_com_ang_acc_w(self) -> torch.Tensor:
        """Object center of mass angular acceleration in simulation world frame.
        Shape is (num_instances, num_objects, 3).

        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        """在仿真世界框架中的质量角加速对象中心。
        形状是 (num_instances，num_objects， 3)。

        这个数量是硬体质量框架中心的角加速。
        """
        return self.object_com_acc_w[..., 3:6]

    @property
    def object_com_pos_b(self) -> torch.Tensor:
        """Center of mass of all of the bodies in their respective body's link frame.
        Shape is (num_instances, num_objects, 3).

        This quantity is the center of mass location relative to its body link frame.
        """
        """在各自身体的环节框架中，所有身体的质量中心。
        形状是 (num_instances，num_objects， 3)。

        这个数量是相对于其身体链接框架的质量位置中心。
        """
        return self.object_com_pose_b[..., :3]

    @property
    def object_com_quat_b(self) -> torch.Tensor:
        """Orientation (w,x,y,z) of the principle axis of inertia of all of the bodies in simulation world frame.
        Shape is (num_instances, num_objects, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body link frame.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中的所有物体的主动性轴的导向 (w，x，y，z)。
        形状是 (num_instances，num_objects， 4)。

        这种数量是对其身体链接框架的惯性轴的方向。
        导向提供 (w， x， y， z) 格式。
        """
        return self.object_com_pose_b[..., 3:7]

    ##
    # Properties for backwards compatibility.
    ##

    @property
    def object_pose_w(self) -> torch.Tensor:
        """Same as :attr:`object_link_pose_w`."""
        """像:attr:`object_link_pose_w`一样。"""
        return self.object_link_pose_w

    @property
    def object_pos_w(self) -> torch.Tensor:
        """Same as :attr:`object_link_pos_w`."""
        """像:attr:`object_link_pos_w`一样。"""
        return self.object_link_pos_w

    @property
    def object_quat_w(self) -> torch.Tensor:
        """Same as :attr:`object_link_quat_w`."""
        """像:attr:`object_link_quat_w`一样。"""
        return self.object_link_quat_w

    @property
    def object_vel_w(self) -> torch.Tensor:
        """Same as :attr:`object_com_vel_w`."""
        """像:attr:`object_com_vel_w`一样。"""
        return self.object_com_vel_w

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        """Same as :attr:`object_com_lin_vel_w`."""
        """像:attr:`object_com_lin_vel_w`一样。"""
        return self.object_com_lin_vel_w

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        """Same as :attr:`object_com_ang_vel_w`."""
        """像:attr:`object_com_ang_vel_w`一样。"""
        return self.object_com_ang_vel_w

    @property
    def object_lin_vel_b(self) -> torch.Tensor:
        """Same as :attr:`object_com_lin_vel_b`."""
        """像:attr:`object_com_lin_vel_b`一样。"""
        return self.object_com_lin_vel_b

    @property
    def object_ang_vel_b(self) -> torch.Tensor:
        """Same as :attr:`object_com_ang_vel_b`."""
        """像:attr:`object_com_ang_vel_b`一样。"""
        return self.object_com_ang_vel_b

    @property
    def object_acc_w(self) -> torch.Tensor:
        """Same as :attr:`object_com_acc_w`."""
        """像:attr:`object_com_acc_w`一样。"""
        return self.object_com_acc_w

    @property
    def object_lin_acc_w(self) -> torch.Tensor:
        """Same as :attr:`object_com_lin_acc_w`."""
        """像:attr:`object_com_lin_acc_w`一样。"""
        return self.object_com_lin_acc_w

    @property
    def object_ang_acc_w(self) -> torch.Tensor:
        """Same as :attr:`object_com_ang_acc_w`."""
        """像:attr:`object_com_ang_acc_w`一样。"""
        return self.object_com_ang_acc_w

    @property
    def com_pos_b(self) -> torch.Tensor:
        """Same as :attr:`object_com_pos_b`."""
        """像:attr:`object_com_pos_b`一样。"""
        return self.object_com_pos_b

    @property
    def com_quat_b(self) -> torch.Tensor:
        """Same as :attr:`object_com_quat_b`."""
        """像:attr:`object_com_quat_b`一样。"""
        return self.object_com_quat_b

    ##
    # Helpers.
    ##

    def _reshape_view_to_data(self, data: torch.Tensor) -> torch.Tensor:
        """Reshapes and arranges the data from the physics view to (num_instances, num_objects, data_size).

        Args:
            data: The data from the physics view. Shape is (num_instances * num_objects, data_size).

        Returns:
            The reshaped data. Shape is (num_objects, num_instances, data_size).
        """
        """从物理视图到 (num_instances，num_objects，data_size) 调整和排列数据。

        参数：
            data: 从物理角度的数据。
                  形状是 (num_instances * num_objects， data_size)。

        返回：
            改造的数据。
            形状是 (num_objects，num_instances，data_size)。
        """
        return torch.einsum("ijk -> jik", data.reshape(self.num_objects, self.num_instances, -1))
