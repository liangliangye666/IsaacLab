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


class RigidObjectData:
    """Data container for a rigid object.

    This class contains the data for a rigid object in the simulation. The data includes the state of
    the root rigid body and the state of all the bodies in the object. The data is stored in the simulation
    world frame unless otherwise specified.

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
    """对硬体的数据容器。

    这个类包含在仿真中的硬体数据。
    数据包括根固体的状态和对象中的所有身体的状态。
    除非另有说明，则数据存储在仿真世界框架中。

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

    def __init__(self, root_physx_view: physx.RigidBodyView, device: str):
        """Initializes the rigid object data.

        Args:
            root_physx_view: The root rigid body view.
            device: The device used for processing.
        """
        """启动硬体数据。

        参数：
            root_physx_view: 根固体视图。
            device: 用于加工的装置。
        """
        # Set the parameters
        self.device = device
        # Set the root rigid body view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._root_physx_view: physx.RigidBodyView = weakref.proxy(root_physx_view)

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
        self.GRAVITY_VEC_W = gravity_dir.repeat(self._root_physx_view.count, 1)
        self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_physx_view.count, 1)

        # Initialize the lazy buffers.
        # -- link frame w.r.t. world frame
        self._root_link_pose_w = TimestampedBuffer()
        self._root_link_vel_w = TimestampedBuffer()
        # -- com frame w.r.t. link frame
        self._body_com_pose_b = TimestampedBuffer()
        # -- com frame w.r.t. world frame
        self._root_com_pose_w = TimestampedBuffer()
        self._root_com_vel_w = TimestampedBuffer()
        self._body_com_acc_w = TimestampedBuffer()
        # -- combined state (these are cached as they concatenate)
        self._root_state_w = TimestampedBuffer()
        self._root_link_state_w = TimestampedBuffer()
        self._root_com_state_w = TimestampedBuffer()

    def update(self, dt: float):
        """Updates the data for the rigid object.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        """更新硬体的数据。

        参数：
            dt: 更新的时间。
                这一定是积极的价值。
        """
        # update the simulation timestamp
        self._sim_timestamp += dt

    ##
    # Names.
    ##

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""
    """在仿真视图中分析的顺序中，"""

    ##
    # Defaults.
    ##

    default_root_state: torch.Tensor = None
    """Default root state ``[pos, quat, lin_vel, ang_vel]`` in local environment frame. Shape is (num_instances, 13).

    The position and quaternion are of the rigid body's actor frame. Meanwhile, the linear and angular velocities are
    of the center of mass frame.
    """
    """在本地环境框架中默认根状态 ``[pos， quat， lin_vel， ang_vel]``。
    形状是 (num_instances， 13)。

    位置和四元数是固体的演员框架。
    与此同时，线性和角度速度是质量框架的中心。
    """

    default_mass: torch.Tensor = None
    """Default mass read from the simulation. Shape is (num_instances, 1)."""
    """在仿真中读取默认质量。
    形状是 (num_instances， 1)。
    """

    default_inertia: torch.Tensor = None
    """Default inertia tensor read from the simulation. Shape is (num_instances, 9).

    The inertia tensor should be given with respect to the center of mass, expressed in the rigid body's actor frame.
    The values are stored in the order :math:`[I_{xx}, I_{yx}, I_{zx}, I_{xy}, I_{yy}, I_{zy}, I_{xz}, I_{yz}, I_{zz}]`.
    However, due to the symmetry of inertia tensors, row- and column-major orders are equivalent.

    This quantity is parsed from the USD schema at the time of initialization.
    """
    """在仿真中读取默认惯性子。
    形状是 (num_instances， 9)。

    should惰性子应与质量中心相比，表达在硬体的演员框架中。
    值存储的顺序:数学:`[I_{xx}， I_{yx}， I_{zx}， I_{xy}， I_{yy}， I_{zy}， I_{xz}， I_{yz}， I_{zz}]`。
    然而，由于惯性子的对称性，排列和柱子主要顺序是等等。

    在初始化时，这个数量从USD方案中解析。
    """

    ##
    # Root state properties.
    ##

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        """Root link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the actor frame of the root rigid body relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中，根链接呈现``[pos， quat]``。
        形状是 (num_instances， 7)。

        这种数量是根固体与世界相对的演员框架的姿势。
        导向提供 (w， x， y， z) 格式。
        """
        if self._root_link_pose_w.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_physx_view.get_transforms().clone()
            pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._root_link_pose_w.data = pose
            self._root_link_pose_w.timestamp = self._sim_timestamp

        return self._root_link_pose_w.data

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """Root link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        """在仿真世界框架中的根链速度``[lin_vel， ang_vel]``。
        形状是 (num_instances， 6)。

        这个数量包含根固体与世界相对的行为体框架的线性和角速度。
        """
        if self._root_link_vel_w.timestamp < self._sim_timestamp:
            # read the CoM velocity
            vel = self.root_com_vel_w.clone()
            # adjust linear velocity to link from center of mass
            vel[:, :3] += torch.linalg.cross(
                vel[:, 3:], math_utils.quat_apply(self.root_link_quat_w, -self.body_com_pos_b[:, 0]), dim=-1
            )
            # set the buffer data and timestamp
            self._root_link_vel_w.data = vel
            self._root_link_vel_w.timestamp = self._sim_timestamp

        return self._root_link_vel_w.data

    @property
    def root_com_pose_w(self) -> torch.Tensor:
        """Root center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 7).

        This quantity is the pose of the center of mass frame of the root rigid body relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中，质量中心 ``[pos， quat]``。
        形状是 (num_instances， 7)。

        这个数量是根固体质量框架的中心相对世界的姿势。
        导向提供 (w， x， y， z) 格式。
        """
        if self._root_com_pose_w.timestamp < self._sim_timestamp:
            # apply local transform to center of mass frame
            pos, quat = math_utils.combine_frame_transforms(
                self.root_link_pos_w, self.root_link_quat_w, self.body_com_pos_b[:, 0], self.body_com_quat_b[:, 0]
            )
            # set the buffer data and timestamp
            self._root_com_pose_w.data = torch.cat((pos, quat), dim=-1)
            self._root_com_pose_w.timestamp = self._sim_timestamp

        return self._root_com_pose_w.data

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """Root center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame
        relative to the world.
        """
        """大量速度的根中心``[lin_vel， ang_vel]``在仿真世界框架中。
        形状是 (num_instances， 6)。

        这个数量包含根固体质量框架中心的线性和角速度相对于世界。
        """
        if self._root_com_vel_w.timestamp < self._sim_timestamp:
            self._root_com_vel_w.data = self._root_physx_view.get_velocities()
            self._root_com_vel_w.timestamp = self._sim_timestamp

        return self._root_com_vel_w.data

    @property
    def root_state_w(self) -> torch.Tensor:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position and orientation are of the rigid body's actor frame. Meanwhile, the linear and angular
        velocities are of the rigid body's center of mass frame.
        """
        """在仿真世界框架中的根状态``[pos， quat， lin_vel， ang_vel]``。
        形状是 (num_instances， 13)。

        位置和方向是硬体的演员框架。
        与此同时，线性和角的速度是固体质量框架的中心。
        """
        if self._root_state_w.timestamp < self._sim_timestamp:
            self._root_state_w.data = torch.cat((self.root_link_pose_w, self.root_com_vel_w), dim=-1)
            self._root_state_w.timestamp = self._sim_timestamp

        return self._root_state_w.data

    @property
    def root_link_state_w(self) -> torch.Tensor:
        """Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the rigid body root frame relative to the
        world. The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中的根状态``[pos， quat， lin_vel， ang_vel]``。
        形状是 (num_instances， 13)。

        位置，四元数和线性/角的速度是与世界相对的硬体根框架。
        导向提供 (w， x， y， z) 格式。
        """
        if self._root_link_state_w.timestamp < self._sim_timestamp:
            self._root_link_state_w.data = torch.cat((self.root_link_pose_w, self.root_link_vel_w), dim=-1)
            self._root_link_state_w.timestamp = self._sim_timestamp

        return self._root_link_state_w.data

    @property
    def root_com_state_w(self) -> torch.Tensor:
        """Root center of mass state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, 13).

        The position, quaternion, and linear/angular velocity are of the rigid body's center of mass frame
        relative to the world. Center of mass frame is the orientation principle axes of inertia.
        """
        """在仿真世界框架中的质量状态``[pos， quat， lin_vel， ang_vel]``的根中心。
        形状是 (num_instances， 13)。

        位置，四元数和线性/角的速度是固体质量框架的中心相对于世界。
        质量框架的中心是惯性的导向原理轴。
        """
        if self._root_com_state_w.timestamp < self._sim_timestamp:
            self._root_com_state_w.data = torch.cat((self.root_com_pose_w, self.root_com_vel_w), dim=-1)
            self._root_com_state_w.timestamp = self._sim_timestamp

        return self._root_com_state_w.data

    ##
    # Body state properties.
    ##

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        """Body link pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 1, 7).

        This quantity is the pose of the actor frame of the rigid body relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中，身体链接呈现``[pos， quat]``。
        形状是 (num_instances， 1， 7)。

        这种数量是对世界而言，
        导向提供 (w， x， y， z) 格式。
        """
        return self.root_link_pose_w.view(-1, 1, 7)

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """Body link velocity ``[lin_vel, ang_vel]`` in simulation world frame. Shape is (num_instances, 1, 6).

        This quantity contains the linear and angular velocities of the actor frame of the root
        rigid body relative to the world.
        """
        """在仿真世界框架中，身体链接速度``[lin_vel， ang_vel]``。
        形状是 (num_instances， 1， 6)。

        这个数量包含根固体与世界相对的行为体框架的线性和角速度。
        """
        return self.root_link_vel_w.view(-1, 1, 6)

    @property
    def body_com_pose_w(self) -> torch.Tensor:
        """Body center of mass pose ``[pos, quat]`` in simulation world frame. Shape is (num_instances, 1, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中，体质中心 ``[pos， quat]``。
        形状是 (num_instances， 1， 7)。

        这种数量是对世界相对的硬体质量框架中心的姿势。
        导向提供 (w， x， y， z) 格式。
        """
        return self.root_com_pose_w.view(-1, 1, 7)

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """Body center of mass velocity ``[lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, 1, 6).

        This quantity contains the linear and angular velocities of the root rigid body's center of mass frame
        relative to the world.
        """
        """在仿真世界框架中体积速度``[lin_vel， ang_vel]``的中心。
        形状是 (num_instances， 1， 6)。

        这个数量包含根固体质量框架中心的线性和角速度相对于世界。
        """
        return self.root_com_vel_w.view(-1, 1, 6)

    @property
    def body_state_w(self) -> torch.Tensor:
        """State of all bodies `[pos, quat, lin_vel, ang_vel]` in simulation world frame.
        Shape is (num_instances, 1, 13).

        The position and orientation are of the rigid bodies' actor frame. Meanwhile, the linear and angular
        velocities are of the rigid bodies' center of mass frame.
        """
        """在仿真世界框架中所有物体的状态 `[pos， quat， lin_vel， ang_vel]`。
        形状是 (num_instances， 1， 13)。

        体的位置和方向是硬体的演员框架。
        与此同时，线性和角的速度是硬体质量框架的中心。
        """
        return self.root_state_w.view(-1, 1, 13)

    @property
    def body_link_state_w(self) -> torch.Tensor:
        """State of all bodies ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, 1, 13).

        The position, quaternion, and linear/angular velocity are of the body's link frame relative to the world.
        The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中所有物体的状态 ``[pos， quat， lin_vel， ang_vel]``。
        形状是 (num_instances， 1， 13)。

        位置，四元数和线性/角速度是身体与世界相对的链接框架。
        导向提供 (w， x， y， z) 格式。
        """
        return self.root_link_state_w.view(-1, 1, 13)

    @property
    def body_com_state_w(self) -> torch.Tensor:
        """State of all bodies ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame.
        Shape is (num_instances, num_bodies, 13).

        The position, quaternion, and linear/angular velocity are of the body's center of mass frame relative to the
        world. Center of mass frame is assumed to be the same orientation as the link rather than the orientation of the
        principle inertia. The orientation is provided in (w, x, y, z) format.
        """
        """在仿真世界框架中所有物体的状态 ``[pos， quat， lin_vel， ang_vel]``。
        形状是 (num_instances，num_bodies， 13)。

        位置，四元数和线性/角的速度是身体与世界相对的质量框架中心。
        质量框架的中心被认为是与链接相同的方向，而不是惯性原则的方向。
        导向提供 (w， x， y， z) 格式。
        """
        return self.root_com_state_w.view(-1, 1, 13)

    @property
    def body_com_acc_w(self) -> torch.Tensor:
        """Acceleration of all bodies ``[lin_acc, ang_acc]`` in the simulation world frame.
        Shape is (num_instances, 1, 6).

        This quantity is the acceleration of the rigid bodies' center of mass frame relative to the world.
        """
        """在仿真世界框架中的所有物体的加速``[lin_acc， ang_acc]``。
        形状是 (num_instances， 1， 6)。

        这种数量是硬体质量框架中心与世界相比的加速。
        """
        if self._body_com_acc_w.timestamp < self._sim_timestamp:
            self._body_com_acc_w.data = self._root_physx_view.get_accelerations().unsqueeze(1)
            self._body_com_acc_w.timestamp = self._sim_timestamp

        return self._body_com_acc_w.data

    @property
    def body_com_pose_b(self) -> torch.Tensor:
        """Center of mass pose ``[pos, quat]`` of all bodies in their respective body's link frames.
        Shape is (num_instances, 1, 7).

        This quantity is the pose of the center of mass frame of the rigid body relative to the body's link frame.
        The orientation is provided in (w, x, y, z) format.
        """
        """所有体体在各自体的链接框架中，质量中心 ``[pos， quat]``。
        形状是 (num_instances， 1， 7)。

        这个数量是硬体质量框架中心的姿势与身体的链接框架相比。
        导向提供 (w， x， y， z) 格式。
        """
        if self._body_com_pose_b.timestamp < self._sim_timestamp:
            # read data from simulation
            pose = self._root_physx_view.get_coms().to(self.device)
            pose[:, 3:7] = math_utils.convert_quat(pose[:, 3:7], to="wxyz")
            # set the buffer data and timestamp
            self._body_com_pose_b.data = pose.view(-1, 1, 7)
            self._body_com_pose_b.timestamp = self._sim_timestamp

        return self._body_com_pose_b.data

    ##
    # Derived Properties.
    ##

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """Projection of the gravity direction on base frame. Shape is (num_instances, 3)."""
        """在基架上投射重力方向。
        形状是 (num_instances， 3)。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

    @property
    def heading_w(self) -> torch.Tensor:
        """Yaw heading of the base frame (in radians). Shape is (num_instances,).

        Note:
            This quantity is computed by assuming that the forward-direction of the base
            frame is along x-direction, i.e. :math:`(1, 0, 0)`.
        """
        """基架的 Yaw方向 (在半径中)。
        形状是 (num_instances，)。

        说明：
            这个数量是通过假设基架的前向方向沿着x方向计算的，i.e.:数学:`(1， 0， 0)`。
        """
        forward_w = math_utils.quat_apply(self.root_link_quat_w, self.FORWARD_VEC_B)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """Root link linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        """根链的线性速度在基架中。
        形状是 (num_instances， 3)。

        这个数量是根固体框架的演员框架的线性速度，
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """Root link angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body frame with respect to the
        rigid body's actor frame.
        """
        """根链角速度在基础世界框架。
        形状是 (num_instances， 3)。

        这个数量是根固体框架的演员框架与硬体的演员框架的角速度。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """Root center of mass linear velocity in base frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        """在基架中，质量线性速度的根中心。
        形状是 (num_instances， 3)。

        这种数量是根固体质量框架中心的线性速度与硬体演员框架相比。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """Root center of mass angular velocity in base world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame with respect to the
        rigid body's actor frame.
        """
        """基层世界框架中的质量角速度的根中心。
        形状是 (num_instances， 3)。

        这个数量是根固体质量框架中心与硬体演员框架的角速度。
        """
        return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_ang_vel_w)

    ##
    # Sliced properties.
    ##

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """Root link position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        """在仿真世界框架中的根链位置。
        形状是 (num_instances， 3)。

        这种数量是根固体与世界相对的演员框架的位置。
        """
        return self.root_link_pose_w[:, :3]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """Root link orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body.
        """
        """在仿真世界框架中，根链的导向 (w，x，y，z)。
        形状是 (num_instances， 4)。

        这种数量是根固体的演员框架的方向。
        """
        return self.root_link_pose_w[:, 3:7]

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's actor frame relative to the world.
        """
        """在仿真世界框架中的根线性速度。
        形状是 (num_instances， 3)。

        这个数量是根固体的演员框架相对于世界的线性速度。
        """
        return self.root_link_vel_w[:, :3]

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """Root link angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the actor frame of the root rigid body relative to the world.
        """
        """在仿真世界框架中的根链角速度。
        形状是 (num_instances， 3)。

        这个数量是根固体与世界相对的演员框架的角速度。
        """
        return self.root_link_vel_w[:, 3:6]

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """Root center of mass position in simulation world frame. Shape is (num_instances, 3).

        This quantity is the position of the actor frame of the root rigid body relative to the world.
        """
        """在仿真世界框架中，
        形状是 (num_instances， 3)。

        这种数量是根固体与世界相对的演员框架的位置。
        """
        return self.root_com_pose_w[:, :3]

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """Root center of mass orientation (w, x, y, z) in simulation world frame. Shape is (num_instances, 4).

        This quantity is the orientation of the actor frame of the root rigid body relative to the world.
        """
        """在仿真世界框架中的质量导向的根中心 (w，x，y，z)。
        形状是 (num_instances， 4)。

        这种数量是根固体与世界相对的演员框架的方向。
        """
        return self.root_com_pose_w[:, 3:7]

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """Root center of mass linear velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the linear velocity of the root rigid body's center of mass frame relative to the world.
        """
        """在仿真世界框架中的质量线性速度的根中心。
        形状是 (num_instances， 3)。

        这个数量是根固体质量框架中心与世界相对的线性速度。
        """
        return self.root_com_vel_w[:, :3]

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """Root center of mass angular velocity in simulation world frame. Shape is (num_instances, 3).

        This quantity is the angular velocity of the root rigid body's center of mass frame relative to the world.
        """
        """在仿真世界框架中的质量角速度的根中心。
        形状是 (num_instances， 3)。

        这个数量是根固体质量框架中心与世界相对的角速度。
        """
        return self.root_com_vel_w[:, 3:6]

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the position of the rigid bodies' actor frame relative to the world.
        """
        """仿真世界框架中的所有物体的位置。
        形状是 (num_instances， 1， 3)。

        这种数量是硬体与世界相对的演员框架的位置。
        """
        return self.body_link_pose_w[..., :3]

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (num_instances, 1, 4).

        This quantity is the orientation of the rigid bodies' actor frame  relative to the world.
        """
        """在仿真世界框架中的所有物体的导向 (w，x，y，z)。
        形状是 (num_instances， 1， 4)。

        这种数量是硬体的演员框架与世界相对的方向。
        """
        return self.body_link_pose_w[..., 3:7]

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame relative to the world.
        """
        """在仿真世界框架中的所有物体的线性速度。
        形状是 (num_instances， 1， 3)。

        这个数量是硬体质量框架中心的线性速度相对于世界。
        """
        return self.body_link_vel_w[..., :3]

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame relative to the world.
        """
        """在仿真世界框架中的所有物体的角速度。
        形状是 (num_instances， 1， 3)。

        这个数量是硬体质量框架中心与世界相对的角速度。
        """
        return self.body_link_vel_w[..., 3:6]

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the position of the rigid bodies' actor frame.
        """
        """仿真世界框架中的所有物体的位置。
        形状是 (num_instances， 1， 3)。

        这种数量是硬体演员框架的位置。
        """
        return self.body_com_pose_w[..., :3]

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all bodies in simulation world frame.

        Shape is (num_instances, 1, 4). This quantity is the orientation of the rigid bodies' actor frame.
        """
        """在仿真世界框架中所有物体的惯性基本轴的导向 (w，x，y，z)。

        形状是 (num_instances， 1， 4)。
        这种数量是硬体演员框架的方向。
        """
        return self.body_com_pose_w[..., 3:7]

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the linear velocity of the rigid bodies' center of mass frame.
        """
        """在仿真世界框架中的所有物体的线性速度。
        形状是 (num_instances， 1， 3)。

        这个数量是刚体质量框架中心的线性速度。
        """
        return self.body_com_vel_w[..., :3]

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the angular velocity of the rigid bodies' center of mass frame.
        """
        """在仿真世界框架中的所有物体的角速度。
        形状是 (num_instances， 1， 3)。

        这个数量是硬体质量框架中心的角速度。
        """
        return self.body_com_vel_w[..., 3:6]

    @property
    def body_com_lin_acc_w(self) -> torch.Tensor:
        """Linear acceleration of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the linear acceleration of the rigid bodies' center of mass frame.
        """
        """在仿真世界框架中的所有物体的线性加速。
        形状是 (num_instances， 1， 3)。

        这个数量是刚体质量框架中心的线性加速。
        """
        return self.body_com_acc_w[..., :3]

    @property
    def body_com_ang_acc_w(self) -> torch.Tensor:
        """Angular acceleration of all bodies in simulation world frame. Shape is (num_instances, 1, 3).

        This quantity is the angular acceleration of the rigid bodies' center of mass frame.
        """
        """在仿真世界框架中的所有物体的角加速。
        形状是 (num_instances， 1， 3)。

        这个数量是硬体质量框架中心的角加速。
        """
        return self.body_com_acc_w[..., 3:6]

    @property
    def body_com_pos_b(self) -> torch.Tensor:
        """Center of mass position of all of the bodies in their respective link frames.
        Shape is (num_instances, 1, 3).

        This quantity is the center of mass location relative to its body'slink frame.
        """
        """所有物体在各自的链接框架中的质量位置中心。
        形状是 (num_instances， 1， 3)。

        这个数量是相对于其身体的斜体位置的中心。
        """
        return self.body_com_pose_b[..., :3]

    @property
    def body_com_quat_b(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of the principle axis of inertia of all of the bodies in their
        respective link frames. Shape is (num_instances, 1, 4).

        This quantity is the orientation of the principles axes of inertia relative to its body's link frame.
        """
        """所有物体在各自的链接框架中的惯性轴的方向 (w，x，y，z)。
        形状是 (num_instances， 1， 4)。

        这种数量是对其身体的链接框架的惯性轴的方向。
        """
        return self.body_com_pose_b[..., 3:7]

    ##
    # Properties for backwards compatibility.
    ##

    @property
    def root_pose_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_pose_w`."""
        """像:attr:`root_link_pose_w`一样。"""
        return self.root_link_pose_w

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_pos_w`."""
        """像:attr:`root_link_pos_w`一样。"""
        return self.root_link_pos_w

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Same as :attr:`root_link_quat_w`."""
        """像:attr:`root_link_quat_w`一样。"""
        return self.root_link_quat_w

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_vel_w`."""
        """像:attr:`root_com_vel_w`一样。"""
        return self.root_com_vel_w

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_lin_vel_w`."""
        """像:attr:`root_com_lin_vel_w`一样。"""
        return self.root_com_lin_vel_w

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        """Same as :attr:`root_com_ang_vel_w`."""
        """像:attr:`root_com_ang_vel_w`一样。"""
        return self.root_com_ang_vel_w

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        """Same as :attr:`root_com_lin_vel_b`."""
        """像:attr:`root_com_lin_vel_b`一样。"""
        return self.root_com_lin_vel_b

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        """Same as :attr:`root_com_ang_vel_b`."""
        """像:attr:`root_com_ang_vel_b`一样。"""
        return self.root_com_ang_vel_b

    @property
    def body_pose_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_pose_w`."""
        """像:attr:`body_link_pose_w`一样。"""
        return self.body_link_pose_w

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_pos_w`."""
        """像:attr:`body_link_pos_w`一样。"""
        return self.body_link_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Same as :attr:`body_link_quat_w`."""
        """像:attr:`body_link_quat_w`一样。"""
        return self.body_link_quat_w

    @property
    def body_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_vel_w`."""
        """像:attr:`body_com_vel_w`一样。"""
        return self.body_com_vel_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_lin_vel_w`."""
        """像:attr:`body_com_lin_vel_w`一样。"""
        return self.body_com_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_ang_vel_w`."""
        """像:attr:`body_com_ang_vel_w`一样。"""
        return self.body_com_ang_vel_w

    @property
    def body_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_acc_w`."""
        """像:attr:`body_com_acc_w`一样。"""
        return self.body_com_acc_w

    @property
    def body_lin_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_lin_acc_w`."""
        """像:attr:`body_com_lin_acc_w`一样。"""
        return self.body_com_lin_acc_w

    @property
    def body_ang_acc_w(self) -> torch.Tensor:
        """Same as :attr:`body_com_ang_acc_w`."""
        """像:attr:`body_com_ang_acc_w`一样。"""
        return self.body_com_ang_acc_w

    @property
    def com_pos_b(self) -> torch.Tensor:
        """Same as :attr:`body_com_pos_b`."""
        """像:attr:`body_com_pos_b`一样。"""
        return self.body_com_pos_b

    @property
    def com_quat_b(self) -> torch.Tensor:
        """Same as :attr:`body_com_quat_b`."""
        """像:attr:`body_com_quat_b`一样。"""
        return self.body_com_quat_b
