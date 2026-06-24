# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.utils.wrench_composer import WrenchComposer

from ..asset_base import AssetBase
from .rigid_object_data import RigidObjectData

if TYPE_CHECKING:
    from .rigid_object_cfg import RigidObjectCfg

# import logger
logger = logging.getLogger(__name__)


class RigidObject(AssetBase):
    """A rigid object asset class.

    Rigid objects are assets comprising of rigid bodies. They can be used to represent dynamic objects
    such as boxes, spheres, etc. A rigid body is described by its pose, velocity and mass distribution.

    For an asset to be considered a rigid object, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid body. On playing the
    simulation, the physics engine will automatically register the rigid body and create a corresponding
    rigid body handle. This handle can be accessed using the :attr:`root_physx_view` attribute.

    .. note::

        For users familiar with Isaac Sim, the PhysX view class API is not the exactly same as Isaac Sim view
        class API. Similar to Isaac Lab, Isaac Sim wraps around the PhysX view API. However, as of now (2023.1 release),
        we see a large difference in initializing the view classes in Isaac Sim. This is because the view classes
        in Isaac Sim perform additional USD-related operations which are slow and also not required.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """
    """一个固体物体资产类。

    硬物体是由硬体组成的资产。
    它们可以用来表示动态物体，如盒子，球体等。
    一个硬体以其姿势，速度和质量分布来描述。

    为了使资产被视为刚性对象，资产的根prim必须有`USD RigidBodyAPI`_应用于它。
    这种API用于定义硬体的仿真性能。
    在播放仿真时，物理引擎将自动登记硬体并创建相应的硬体句柄。
    使用:attr:`root_physx_view`属性访问此句柄。

    .. 说明::

        对于熟悉Isaac Sim的用户来说，PhysX视觉类API不完全与Isaac Sim视觉相同
        class API. Similar to Isaac Lab, Isaac Sim wraps around the PhysX view API. However, as of now (2023.1 release),
        我们在Isaac Sim中初始化视频类中看到很大的区别。
        这是因为Isaac Sim中的视图类执行额外的USD相关操作，这些操作是缓慢的，也不需要。

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: RigidObjectCfg
    """Configuration instance for the rigid object."""
    """对硬体的配置实例。"""

    def __init__(self, cfg: RigidObjectCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        """启动硬体。

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
    def data(self) -> RigidObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single rigid body.
        """
        """资产中的尸体数量。

        这总是1因为每个对象都是一个固体。
        """
        return 1

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        """在固体物体中排列的尸体名称。"""
        prim_paths = self.root_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def root_physx_view(self) -> physx.RigidBodyView:
        """Rigid body view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        """对资产的硬体视图 (PhysX)。

        说明：
            用这种观点谨慎。
            它需要以特定的方式处理子。
        """
        return self._root_physx_view

    @property
    def instantaneous_wrench_composer(self) -> WrenchComposer:
        """Instantaneous wrench composer.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer are only valid for the current simulation step. At the end of the simulation step, the wrenches set
        to this object are discarded. This is useful to apply forces that change all the time, things like drag forces
        for instance.
        """
        """立刻的 w钥匙作曲家。

        返回一个:class:`~isaaclab.utils.wrench_composer.WrenchComposer`实例。
        添加或设置到此钥匙组件的关键仅适用于当前仿真步骤。
        在仿真步骤结束时，将对此物体设置的 w钥匙丢弃。
        这对于不断变化的力量来说是有用的。
        for instance.
        """
        return self._instantaneous_wrench_composer

    @property
    def permanent_wrench_composer(self) -> WrenchComposer:
        """Permanent wrench composer.

        Returns a :class:`~isaaclab.utils.wrench_composer.WrenchComposer` instance. Wrenches added or set to this wrench
        composer are persistent and are applied to the simulation at every step. This is useful to apply forces that
        are constant over a period of time, things like the thrust of a motor for instance.
        """
        """一个永久的 w钥匙作曲家。

        返回一个:class:`~isaaclab.utils.wrench_composer.WrenchComposer`实例。
        加入或设置到这个匙组件的关键是持久的，并且在每一步都应用于仿真。
        这对于在时间段内恒定的力量来说是有用的，例如电机的推力。
        """
        return self._permanent_wrench_composer

    """
    Operations.
    """
    """操作。
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # resolve all indices
        if env_ids is None:
            env_ids = slice(None)
        # reset external wrench
        self._instantaneous_wrench_composer.reset(env_ids)
        self._permanent_wrench_composer.reset(env_ids)

    def write_data_to_sim(self):
        """Write external wrench to the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        """在仿真中写出外部钥匙。

        说明：
            我们写出仿真的外部关键，因为这个函数在仿真步骤之前被调用。
            这确保在每个仿真步骤上使用外部钥匙。
        """
        # write external wrench
        if self._instantaneous_wrench_composer.active or self._permanent_wrench_composer.active:
            if self._instantaneous_wrench_composer.active:
                # Compose instantaneous wrench with permanent wrench
                self._instantaneous_wrench_composer.add_forces_and_torques(
                    forces=self._permanent_wrench_composer.composed_force,
                    torques=self._permanent_wrench_composer.composed_torque,
                    body_ids=self._ALL_BODY_INDICES_WP,
                    env_ids=self._ALL_INDICES_WP,
                )
                # Apply both instantaneous and permanent wrench to the simulation
                self.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=self._instantaneous_wrench_composer.composed_force_as_torch.view(-1, 3),
                    torque_data=self._instantaneous_wrench_composer.composed_torque_as_torch.view(-1, 3),
                    position_data=None,
                    indices=self._ALL_INDICES,
                    is_global=False,
                )
            else:
                # Apply permanent wrench to the simulation
                self.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=self._permanent_wrench_composer.composed_force_as_torch.view(-1, 3),
                    torque_data=self._permanent_wrench_composer.composed_torque_as_torch.view(-1, 3),
                    position_data=None,
                    indices=self._ALL_INDICES,
                    is_global=False,
                )
        self._instantaneous_wrench_composer.reset()

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """
    """搜索器
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the rigid body based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        """根据名称键，找到身体在硬体中。

        请查看:meth:`isaaclab.utils.string_utils.resolve_matching_names`函数，了解更多关于名称匹配的信息。

        参数：
            name_keys: 一个正则表达式或一个与体名相匹配的正则表达式列表。
            preserve_order: 在输出中是否保留名称键的顺序。
                            默认为 False。

        返回：
            一个包含身体指标和名称的列表。
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Operations - Write to simulation.
    """
    """操作 - 写入仿真。
    """

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选定的环境索引上的根状态。

        根状态包括卡特西亚位置，在 (w，x，y，z) 中的四元数方向以及线性和角的速度。
        所有数量都在仿真框架中。

        参数：
            root_state: 在仿真框架中的根状态。
                        形状是 (len(env_ids)，13。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_com_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置了选择的环境索引上质量状态的根中心。

        根状态包括卡特西亚位置，在 (w，x，y，z) 中的四元数方向以及线性和角的速度。
        所有数量都在仿真框架中。

        参数：
            root_state: 在仿真框架中的根状态。
                        形状是 (len(env_ids)，13。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        self.write_root_com_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_link_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选定的环境索引上根链状态。

        根状态包括卡特西亚位置，在 (w，x，y，z) 中的四元数方向以及线性和角的速度。
        所有数量都在仿真框架中。

        参数：
            root_state: 在仿真框架中的根状态。
                        形状是 (len(env_ids)，13。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_link_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选定的环境索引上的根姿势。

        根姿势包括在 (w，x，y，z) 中的卡特西亚位置和四元数方向。

        参数：
            root_pose: 根链在仿真框架中呈现。
                       形状是 (len(env_ids)， 7)。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        self.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

    def write_root_link_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选定的环境索引上根链接姿势。

        根姿势包括在 (w，x，y，z) 中的卡特西亚位置和四元数方向。

        参数：
            root_pose: 根链在仿真框架中呈现。
                       形状是 (len(env_ids)， 7)。
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
        self._data.root_link_pose_w[env_ids] = root_pose.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[env_ids, :7] = self._data.root_link_pose_w[env_ids]
        if self._data._root_state_w.data is not None:
            self._data.root_state_w[env_ids, :7] = self._data.root_link_pose_w[env_ids]
        if self._data._root_com_state_w.data is not None:
            expected_com_pos, expected_com_quat = math_utils.combine_frame_transforms(
                self._data.root_link_pose_w[env_ids, :3],
                self._data.root_link_pose_w[env_ids, 3:7],
                self.data.body_com_pos_b[env_ids, 0, :],
                self.data.body_com_quat_b[env_ids, 0, :],
            )
            self._data.root_com_state_w[env_ids, :3] = expected_com_pos
            self._data.root_com_state_w[env_ids, 3:7] = expected_com_quat
        # convert root quaternion from wxyz to xyzw
        root_poses_xyzw = self._data.root_link_pose_w.clone()
        root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")
        # set into simulation
        self.root_physx_view.set_transforms(root_poses_xyzw, indices=physx_env_ids)

    def write_root_com_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选择的环境索引上，

        根姿势包括在 (w，x，y，z) 中的卡特西亚位置和四元数方向。
        导向是惯性的主要轴的导向。

        参数：
            root_pose: 在仿真框架中，质量的根中心姿势。
                       形状是 (len(env_ids)， 7)。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        # resolve all indices
        if env_ids is None:
            local_env_ids = slice(env_ids)
        else:
            local_env_ids = env_ids

        # set into internal buffers
        self._data.root_com_pose_w[local_env_ids] = root_pose.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_com_state_w.data is not None:
            self._data.root_com_state_w[local_env_ids, :7] = self._data.root_com_pose_w[local_env_ids]

        # get CoM pose in link frame
        com_pos_b = self.data.body_com_pos_b[local_env_ids, 0, :]
        com_quat_b = self.data.body_com_quat_b[local_env_ids, 0, :]
        # transform input CoM pose to link frame
        root_link_pos, root_link_quat = math_utils.combine_frame_transforms(
            root_pose[..., :3],
            root_pose[..., 3:7],
            math_utils.quat_apply(math_utils.quat_inv(com_quat_b), -com_pos_b),
            math_utils.quat_inv(com_quat_b),
        )
        root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)

        # write transformed pose in link frame to sim
        self.write_root_link_pose_to_sim(root_link_pose, env_ids=env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置质量速度的根中心，

        速度包括线性速度 (x，y，z) 和角速度 (x，y，z) 在这个顺序中。
        NOTE: 这设定了根的质量中心的速度，而不是根框架。

        参数：
            root_velocity: 在仿真世界框架中，
                           形状是 (len(env_ids)， 6。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        self.write_root_com_velocity_to_sim(root_velocity=root_velocity, env_ids=env_ids)

    def write_root_com_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置质量速度的根中心，

        速度包括线性速度 (x，y，z) 和角速度 (x，y，z) 在这个顺序中。
        NOTE: 这设定了根的质量中心的速度，而不是根框架。

        参数：
            root_velocity: 在仿真世界框架中，
                           形状是 (len(env_ids)， 6。
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
        self._data.root_com_vel_w[env_ids] = root_velocity.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_com_state_w.data is not None:
            self._data.root_com_state_w[env_ids, 7:] = self._data.root_com_vel_w[env_ids]
        if self._data._root_state_w.data is not None:
            self._data.root_state_w[env_ids, 7:] = self._data.root_com_vel_w[env_ids]
        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[env_ids, 7:] = self._data.root_com_vel_w[env_ids]
        # make the acceleration zero to prevent reporting old values
        self._data.body_com_acc_w[env_ids] = 0.0
        # set into simulation
        self.root_physx_view.set_velocities(self._data.root_com_vel_w, indices=physx_env_ids)

    def write_root_link_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        """在仿真中设置选定的环境索引上根链速度。

        速度包括线性速度 (x，y，z) 和角速度 (x，y，z) 在这个顺序中。
        NOTE: 这设定了根框架的速度而不是根质中心。

        参数：
            root_velocity: 在仿真世界框架中的根框架速度。
                           形状是 (len(env_ids)， 6。
            env_ids: 环境索引
                     如果 None，则使用所有索引。
        """
        # resolve all indices
        if env_ids is None:
            local_env_ids = slice(env_ids)
        else:
            local_env_ids = env_ids

        # set into internal buffers
        self._data.root_link_vel_w[local_env_ids] = root_velocity.clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[local_env_ids, 7:] = self._data.root_link_vel_w[local_env_ids]

        # get CoM pose in link frame
        quat = self.data.root_link_quat_w[local_env_ids]
        com_pos_b = self.data.body_com_pos_b[local_env_ids, 0, :]
        # transform input velocity to center of mass frame
        root_com_velocity = root_velocity.clone()
        root_com_velocity[:, :3] += torch.linalg.cross(
            root_com_velocity[:, 3:], math_utils.quat_apply(quat, com_pos_b), dim=-1
        )

        # write transformed velocity in CoM frame to sim
        self.write_root_com_velocity_to_sim(root_com_velocity, env_ids=env_ids)

    """
    Operations - Setters.
    """
    """运营 - 设置器。
    """

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        positions: torch.Tensor | None = None,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        is_global: bool = False,
    ):
        """Set external force and torque to apply on the asset's bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step. Optionally, set the position to apply the
        external wrench at (in the local link frame of the bodies).

        .. caution::
            If the function is called with empty forces and torques, then this function disables the application
            of external wrench to the simulation.

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=torch.zeros(0, 3), torques=torch.zeros(0, 3))

        .. note::
            This function does not apply the external wrench to the simulation. It only fills the buffers with
            the desired values. To apply the external wrench, call the :meth:`write_data_to_sim` function
            right before the simulation step.

        Args:
            forces: External forces in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            torques: External torques in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
            positions: External wrench positions in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
                Defaults to None.
            body_ids: Body indices to apply external wrench to. Defaults to None (all bodies).
            env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
            is_global: Whether to apply the external wrench in the global frame. Defaults to False. If set to False,
                the external wrench is applied in the link frame of the bodies.
        """
        """设置外部力和扭矩应在本地框架中的资产体上应用。

        在许多应用中，我们希望在一段时间内 (例如，在策略控制期间) 保持对硬体的外力稳定。
        这种功能使我们能够将外部力和扭矩存储在缓冲器中，然后在每一步都应用于仿真。
        选择地设置将外部钥匙应用到 (在机器的本地链接框中)。

        .. 谨慎::
            如果函数被用空力和扭矩调用，则该函数将外部钥匙被禁用在仿真中。

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=torch.zeros(0, 3), torques=torch.zeros(0, 3))

        .. 说明::
            这项函数不适用于仿真的外部关键。
            它只用所需的值填充缓冲器。
            在仿真步骤之前，请调用:meth:`write_data_to_sim`函数。

        参数：
            forces: 在身体的局部框架中，
                    形状是 (len(env_ids)，len(body_ids)，3)。
            torques: 身体的局部体内外部扭矩。
                     形状是 (len(env_ids)，len(body_ids)，3)。
            positions: 在尸体的局部框架中，外部钥匙的位置。
                       形状是 (len(env_ids)，len(body_ids)，3)。
                       默认为 None。
            body_ids: 机体指标应用外部钥匙。
                      在None (所有机体) 上默认设置。
            env_ids: 环境索引应使用外部匙。
                     在 None 中默认设置 (所有实例)。
            is_global: 在全球框架中是否应使用外部 w钥匙。
                       默认为 False。
                       如果设置为False，则将外部钥匙应用在车身的链框中。
        """
        logger.warning(
            "The function 'set_external_force_and_torque' will be deprecated in a future release. Please"
            " use 'permanent_wrench_composer.set_forces_and_torques' instead."
        )
        if forces is None and torques is None:
            logger.warning("No forces or torques provided. No permanent external wrench will be applied.")

        # resolve all indices
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES_WP
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = wp.array(env_ids, dtype=wp.int32, device=self.device)
        else:
            env_ids = wp.from_torch(env_ids.to(torch.int32), dtype=wp.int32)
        # -- body_ids
        body_ids = self._ALL_BODY_INDICES_WP

        # Write to wrench composer
        self._permanent_wrench_composer.set_forces_and_torques(
            forces=wp.from_torch(forces, dtype=wp.vec3f) if forces is not None else None,
            torques=wp.from_torch(torques, dtype=wp.vec3f) if torques is not None else None,
            positions=wp.from_torch(positions, dtype=wp.vec3f) if positions is not None else None,
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=is_global,
        )

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

        # find rigid root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
            traverse_instance_prims=False,
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a rigid body when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim has 'USD RigidBodyAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single rigid body when resolving '{self.cfg.prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one rigid body in the prim path tree."
            )

        articulation_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
            traverse_instance_prims=False,
        )
        if len(articulation_prims) != 0:
            if articulation_prims[0].GetAttribute("physxArticulation:articulationEnabled").Get():
                raise RuntimeError(
                    f"Found an articulation root when resolving '{self.cfg.prim_path}' for rigid objects. These are"
                    f" located at: '{articulation_prims}' under '{template_prim_path}'. Please disable the articulation"
                    " root in the USD or from code by setting the parameter"
                    " 'ArticulationRootPropertiesCfg.articulation_enabled' to False in the spawn configuration."
                )

        # resolve root prim back into regex expression
        root_prim_path = root_prims[0].GetPath().pathString
        root_prim_path_expr = self.cfg.prim_path + root_prim_path[len(template_prim_path) :]
        # -- object view
        self._root_physx_view = self._physics_sim_view.create_rigid_body_view(root_prim_path_expr.replace(".*", "*"))

        # check if the rigid body was created
        if self._root_physx_view._backend is None:
            raise RuntimeError(f"Failed to create rigid body at: {self.cfg.prim_path}. Please check PhysX logs.")

        # log information about the rigid body
        logger.info(f"Rigid body initialized at: {self.cfg.prim_path} with root '{root_prim_path_expr}'.")
        logger.info(f"Number of instances: {self.num_instances}")
        logger.info(f"Number of bodies: {self.num_bodies}")
        logger.info(f"Body names: {self.body_names}")

        # container for data access
        self._data = RigidObjectData(self.root_physx_view, self.device)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        # update the rigid body data
        self.update(0.0)

    def _create_buffers(self):
        """Create buffers for storing data."""
        """创建存储数据的缓冲器。"""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        self._ALL_INDICES_WP = wp.from_torch(self._ALL_INDICES.to(torch.int32), dtype=wp.int32)
        self._ALL_BODY_INDICES_WP = wp.from_torch(
            torch.arange(self.num_bodies, dtype=torch.int32, device=self.device), dtype=wp.int32
        )

        # external wrench composer
        self._instantaneous_wrench_composer = WrenchComposer(self)
        self._permanent_wrench_composer = WrenchComposer(self)

        # set information about rigid body into data
        self._data.body_names = self.body_names
        self._data.default_mass = self.root_physx_view.get_masses().clone()
        self._data.default_inertia = self.root_physx_view.get_inertias().clone()

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        """配置参数后处理。"""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)
            + tuple(self.cfg.init_state.lin_vel)
            + tuple(self.cfg.init_state.ang_vel)
        )
        default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)

    """
    Internal simulation callbacks.
    """
    """内部仿真回调。
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        """破坏场景元素。"""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._root_physx_view = None
