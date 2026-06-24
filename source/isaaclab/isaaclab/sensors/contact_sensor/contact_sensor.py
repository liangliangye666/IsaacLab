# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import convert_quat

from ..sensor_base import SensorBase
from .contact_sensor_data import ContactSensorData

if TYPE_CHECKING:
    from .contact_sensor_cfg import ContactSensorCfg


class ContactSensor(SensorBase):
    """A contact reporting sensor.

    The contact sensor reports the normal contact forces on a rigid body in the world frame.
    It relies on the `PhysX ContactReporter`_ API to be activated on the rigid bodies.

    To enable the contact reporter on a rigid body, please make sure to enable the
    :attr:`isaaclab.sim.spawner.RigidObjectSpawnerCfg.activate_contact_sensors` on your
    asset spawner configuration. This will enable the contact reporter on all the rigid bodies
    in the asset.

    The sensor can be configured to report the contact forces on a set of bodies with a given
    filter pattern using the :attr:`ContactSensorCfg.filter_prim_paths_expr`. This is useful
    when you want to report the contact forces between the sensor bodies and a specific set of
    bodies in the scene. The data can be accessed using the :attr:`ContactSensorData.force_matrix_w`.
    Please check the documentation on `RigidContact`_ for more details.

    The reporting of the filtered contact forces is only possible as one-to-many. This means that only one
    sensor body in an environment can be filtered against multiple bodies in that environment. If you need to
    filter multiple sensor bodies against multiple bodies, you need to create separate sensors for each sensor
    body.

    As an example, suppose you want to report the contact forces for all the feet of a robot against an object
    exclusively. In that case, setting the :attr:`ContactSensorCfg.prim_path` and
    :attr:`ContactSensorCfg.filter_prim_paths_expr` with ``{ENV_REGEX_NS}/Robot/.*_FOOT`` and ``{ENV_REGEX_NS}/Object``
    respectively will not work. Instead, you need to create a separate sensor for each foot and filter
    it against the object.

    .. _PhysX ContactReporter: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_contact_report_a_p_i.html
    .. _RigidContact: https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.sensors.RigidContactView
    """
    """一个传感器。

    接触传感器报告了世界框架中的硬体上的正常接触力。
    它依赖于`PhysX ContactReporter`_API在硬体上激活。

    为了在硬体上启用联系记者，请确保在您的资产产产子器配置上启用:attr:`isaaclab.sim.spawner.RigidObjectSpawnerCfg.activate_contact_sen
    sors`。
    这将使记者能够接触资产中的所有硬体。

    传感器可以配置以使用:attr:`ContactSensorCfg.filter_prim_paths_expr`来报告特定的过器模式的集体上的接触力。
    这对于要报告传感器体与场景特定的体体之间的接触力时是有用的。
    通过:attr:`ContactSensorData.force_matrix_w`可以访问数据。
    请查看有关`RigidContact`_的文档。

    过的接触力只能以一个对许多的形式报告。
    这意味着只能在环境中过一个传感器体对该环境中的多个体。
    如果您需要过多个传感器体对多个传感器体，

    举个例子，假设你想将机器人所有脚的接触力报告到一个物体。
    在这种情况下，将:attr:`ContactSensorCfg.prim_path`和:attr:`ContactSensorCfg.filter_prim_paths_expr`分别设置为``{EN
    V_REGEX_NS}/Robot/.*_FOOT``和``{ENV_REGEX_NS}/Object``将不会工作。
    而是，你需要为每条脚创建一个独立的传感器，

    .. _PhysX ContactReporter: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_contact_report_a_p_i.html
    .. _RigidContact: https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.sensors.RigidContactView
    """

    cfg: ContactSensorCfg
    """The configuration parameters."""
    """配置参数。"""

    def __init__(self, cfg: ContactSensorCfg):
        """Initializes the contact sensor object.

        Args:
            cfg: The configuration parameters.
        """
        """启动接触传感器对象。

        参数：
            cfg: 配置参数。
        """
        # initialize base class
        super().__init__(cfg)

        # Enable contact processing
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/physics/disableContactProcessing", False)

        # Create empty variables for storing output data
        self._data: ContactSensorData = ContactSensorData()
        # initialize self._body_physx_view for running in extension mode
        self._body_physx_view = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        """Returns: 包含有关实例的信息。"""
        return (
            f"Contact sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type         : {self.body_physx_view.__class__}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of bodies  : {self.num_bodies}\n"
            f"\tbody names        : {self.body_names}\n"
        )

    """
    Properties
    """
    """产品
    """

    @property
    def num_instances(self) -> int:
        return self.body_physx_view.count

    @property
    def data(self) -> ContactSensorData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_bodies(self) -> int:
        """Number of bodies with contact sensors attached."""
        """连接传感器的身体数量"""
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies with contact sensors attached."""
        """配列的接触传感器的尸体名称。"""
        prim_paths = self.body_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def body_physx_view(self) -> physx.RigidBodyView:
        """View for the rigid bodies captured (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        """捕获的硬体的视图 (PhysX)。

        说明：
            用这种观点谨慎。
            它需要以特定的方式处理子。
        """
        return self._body_physx_view

    @property
    def contact_physx_view(self) -> physx.RigidContactView:
        """Contact reporter view for the bodies (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        """联系记者查看尸体 (PhysX)。

        说明：
            用这种观点谨慎。
            它需要以特定的方式处理子。
        """
        return self._contact_physx_view

    """
    Operations
    """
    """运营
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset accumulative data buffers
        self._data.net_forces_w[env_ids] = 0.0
        self._data.net_forces_w_history[env_ids] = 0.0
        # reset force matrix
        if len(self.cfg.filter_prim_paths_expr) != 0:
            self._data.force_matrix_w[env_ids] = 0.0
            self._data.force_matrix_w_history[env_ids] = 0.0
        # reset the current air time
        if self.cfg.track_air_time:
            self._data.current_air_time[env_ids] = 0.0
            self._data.last_air_time[env_ids] = 0.0
            self._data.current_contact_time[env_ids] = 0.0
            self._data.last_contact_time[env_ids] = 0.0
        # reset contact positions
        if self.cfg.track_contact_points:
            self._data.contact_pos_w[env_ids, :] = torch.nan
        # reset friction forces
        if self.cfg.track_friction_forces:
            self._data.friction_forces_w[env_ids, :] = 0.0

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        """根据名字键，在关节中找到尸体。

        参数：
            name_keys: 一个正则表达式或一个与体名相匹配的正则表达式列表。
            preserve_order: 在输出中是否保留名称键的顺序。
                            默认为 False。

        返回：
            一个包含身体指标和名称的列表。
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        """Checks if bodies that have established contact within the last :attr:`dt` seconds.

        This function checks if the bodies have established contact within the last :attr:`dt` seconds
        by comparing the current contact time with the given time period. If the contact time is less
        than the given time period, then the bodies are considered to be in contact.

        Note:
            The function assumes that :attr:`dt` is a factor of the sensor update time-step. In other
            words :math:`dt / dt_sensor = n`, where :math:`n` is a natural number. This is always true
            if the sensor is updated by the physics or the environment stepping time-step and the sensor
            is read by the environment stepping time-step.

        Args:
            dt: The time period since the contact was established.
            abs_tol: The absolute tolerance for the comparison.

        Returns:
            A boolean tensor indicating the bodies that have established contact within the last
            :attr:`dt` seconds. Shape is (N, B), where N is the number of sensors and B is the
            number of bodies in each sensor.

        Raises:
            RuntimeError: If the sensor is not configured to track contact time.
        """
        """检查在过去的:attr:`dt`秒内建立接触的物体。

        该函数通过比较当前接触时间与所给定的时间段来检查物体在过去的:attr:`dt`秒内是否取得接触。
        如果接触时间低于所述时间段，则将被视为接触物体。

        说明：
            函数假设:attr:`dt`是传感器更新时间步骤的因素。
            在其他
            words :数学:`dt / dt_sensor = n`，其中:数学:`n`是自然数。
                   这总是真的。
            if the sensor is updated by the physics or the environment stepping time-step and the sensor
            环境会逐步阅读。

        参数：
            dt: 自接触建立以来的时间。
            abs_tol: 对于比较的绝对宽容。

        返回：
            在过去的:attr:`dt`秒内建立接触的体体的布尔式子。
            形状是 (N，B)，其中N是传感器的数量，B是每个传感器的体体数量。

        异常：
            RuntimeError: 如果传感器不配置以追踪接触时间。
        """
        # check if the sensor is configured to track contact time
        if not self.cfg.track_air_time:
            raise RuntimeError(
                "The contact sensor is not configured to track contact time."
                "Please enable the 'track_air_time' in the sensor configuration."
            )
        # check if the bodies are in contact
        currently_in_contact = self.data.current_contact_time > 0.0
        less_than_dt_in_contact = self.data.current_contact_time < (dt + abs_tol)
        return currently_in_contact * less_than_dt_in_contact

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        """Checks if bodies that have broken contact within the last :attr:`dt` seconds.

        This function checks if the bodies have broken contact within the last :attr:`dt` seconds
        by comparing the current air time with the given time period. If the air time is less
        than the given time period, then the bodies are considered to not be in contact.

        Note:
            It assumes that :attr:`dt` is a factor of the sensor update time-step. In other words,
            :math:`dt / dt_sensor = n`, where :math:`n` is a natural number. This is always true if
            the sensor is updated by the physics or the environment stepping time-step and the sensor
            is read by the environment stepping time-step.

        Args:
            dt: The time period since the contract is broken.
            abs_tol: The absolute tolerance for the comparison.

        Returns:
            A boolean tensor indicating the bodies that have broken contact within the last :attr:`dt` seconds.
            Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

        Raises:
            RuntimeError: If the sensor is not configured to track contact time.
        """
        """检查是否在最后一次接触中断了身体:attr:`dt`几秒钟。

        该函数通过比较当前空气时间与所给定的时间段来检查物体在过去的:attr:`dt`秒内是否断裂接触。
        如果空气时间低于所给定的时间段，则认为尸体没有接触。

        说明：
            它假设:attr:`dt`是传感器更新时间步骤的因素。
            换句话说，
            :math:`dt / dt_sensor = n`，其中:数学:`n`是一个自然数。
            传感器由物理或环境步骤时间更新，传感器由环境步骤时间读取。

        参数：
            dt: 自合同破产以来的时间。
            abs_tol: 对于比较的绝对宽容。

        返回：
            在最后的:attr:`dt`秒内断交的体体表示的布尔式子。
            形状是 (N，B)，其中N是传感器的数量，B是每个传感器的体体数量。

        异常：
            RuntimeError: 如果传感器不配置以追踪接触时间。
        """
        # check if the sensor is configured to track contact time
        if not self.cfg.track_air_time:
            raise RuntimeError(
                "The contact sensor is not configured to track contact time."
                "Please enable the 'track_air_time' in the sensor configuration."
            )
        # check if the sensor is configured to track contact time
        currently_detached = self.data.current_air_time > 0.0
        less_than_dt_detached = self.data.current_air_time < (dt + abs_tol)
        return currently_detached * less_than_dt_detached

    """
    Implementation.
    """
    """执行。
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # check that only rigid bodies are selected
        leaf_pattern = self.cfg.prim_path.rsplit("/", 1)[-1]
        template_prim_path = self._parent_prims[0].GetPath().pathString
        body_names = list()
        for prim in sim_utils.find_matching_prims(template_prim_path + "/" + leaf_pattern):
            # check if prim has contact reporter API
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                prim_path = prim.GetPath().pathString
                body_names.append(prim_path.rsplit("/", 1)[-1])
        # check that there is at least one body with contact reporter API
        if not body_names:
            raise RuntimeError(
                f"Sensor at path '{self.cfg.prim_path}' could not find any bodies with contact reporter API."
                "\nHINT: Make sure to enable 'activate_contact_sensors' in the corresponding asset spawn configuration."
            )

        # construct regex expression for the body names
        body_names_regex = r"(" + "|".join(body_names) + r")"
        body_names_regex = f"{self.cfg.prim_path.rsplit('/', 1)[0]}/{body_names_regex}"
        # convert regex expressions to glob expressions for PhysX
        body_names_glob = body_names_regex.replace(".*", "*")
        filter_prim_paths_glob = [expr.replace(".*", "*") for expr in self.cfg.filter_prim_paths_expr]

        # create a rigid prim view for the sensor
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_names_glob)
        self._contact_physx_view = self._physics_sim_view.create_rigid_contact_view(
            body_names_glob,
            filter_patterns=filter_prim_paths_glob,
            max_contact_data_count=self.cfg.max_contact_data_count_per_prim * len(body_names) * self._num_envs,
        )
        # resolve the true count of bodies
        self._num_bodies = self.body_physx_view.count // self._num_envs
        # check that contact reporter succeeded
        if self._num_bodies != len(body_names):
            raise RuntimeError(
                "Failed to initialize contact reporter for specified bodies."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tResolved prim paths: {body_names_regex}"
            )

        # prepare data buffers
        self._data.net_forces_w = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
        # optional buffers
        # -- history of net forces
        if self.cfg.history_length > 0:
            self._data.net_forces_w_history = torch.zeros(
                self._num_envs, self.cfg.history_length, self._num_bodies, 3, device=self._device
            )
        else:
            self._data.net_forces_w_history = self._data.net_forces_w.unsqueeze(1)
        # -- pose of sensor origins
        if self.cfg.track_pose:
            self._data.pos_w = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
            self._data.quat_w = torch.zeros(self._num_envs, self._num_bodies, 4, device=self._device)

        # check if filter paths are valid
        if self.cfg.track_contact_points or self.cfg.track_friction_forces:
            if len(self.cfg.filter_prim_paths_expr) == 0:
                raise ValueError(
                    "The 'filter_prim_paths_expr' is empty. Please specify a valid filter pattern to track"
                    f" {'contact points' if self.cfg.track_contact_points else 'friction forces'}."
                )
            if self.cfg.max_contact_data_count_per_prim < 1:
                raise ValueError(
                    f"The 'max_contact_data_count_per_prim' is {self.cfg.max_contact_data_count_per_prim}. "
                    "Please set it to a value greater than 0 to track"
                    f" {'contact points' if self.cfg.track_contact_points else 'friction forces'}."
                )

        # -- position of contact points
        if self.cfg.track_contact_points:
            self._data.contact_pos_w = torch.full(
                (self._num_envs, self._num_bodies, self.contact_physx_view.filter_count, 3),
                torch.nan,
                device=self._device,
            )
        # -- friction forces at contact points
        if self.cfg.track_friction_forces:
            self._data.friction_forces_w = torch.full(
                (self._num_envs, self._num_bodies, self.contact_physx_view.filter_count, 3),
                0.0,
                device=self._device,
            )
        # -- air/contact time between contacts
        if self.cfg.track_air_time:
            self._data.last_air_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
            self._data.current_air_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
            self._data.last_contact_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
            self._data.current_contact_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
        # force matrix: (num_envs, num_bodies, num_filter_shapes, 3)
        # force matrix history: (num_envs, history_length, num_bodies, num_filter_shapes, 3)
        if len(self.cfg.filter_prim_paths_expr) != 0:
            num_filters = self.contact_physx_view.filter_count
            self._data.force_matrix_w = torch.zeros(
                self._num_envs, self._num_bodies, num_filters, 3, device=self._device
            )
            if self.cfg.history_length > 0:
                self._data.force_matrix_w_history = torch.zeros(
                    self._num_envs, self.cfg.history_length, self._num_bodies, num_filters, 3, device=self._device
                )
            else:
                self._data.force_matrix_w_history = self._data.force_matrix_w.unsqueeze(1)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        """填充传感器数据的缓冲器。"""
        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        # obtain the contact forces
        # TODO: We are handling the indexing ourself because of the shape; (N, B) vs expected (N * B).
        #   This isn't the most efficient way to do this, but it's the easiest to implement.
        net_forces_w = self.contact_physx_view.get_net_contact_forces(dt=self._sim_physics_dt)
        self._data.net_forces_w[env_ids, :, :] = net_forces_w.view(-1, self._num_bodies, 3)[env_ids]
        # update contact force history
        if self.cfg.history_length > 0:
            self._data.net_forces_w_history[env_ids] = self._data.net_forces_w_history[env_ids].roll(1, dims=1)
            self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]

        # obtain the contact force matrix
        if len(self.cfg.filter_prim_paths_expr) != 0:
            # shape of the filtering matrix: (num_envs, num_bodies, num_filter_shapes, 3)
            num_filters = self.contact_physx_view.filter_count
            # acquire and shape the force matrix
            force_matrix_w = self.contact_physx_view.get_contact_force_matrix(dt=self._sim_physics_dt)
            force_matrix_w = force_matrix_w.view(-1, self._num_bodies, num_filters, 3)
            self._data.force_matrix_w[env_ids] = force_matrix_w[env_ids]
            if self.cfg.history_length > 0:
                self._data.force_matrix_w_history[env_ids] = self._data.force_matrix_w_history[env_ids].roll(1, dims=1)
                self._data.force_matrix_w_history[env_ids, 0] = self._data.force_matrix_w[env_ids]

        # obtain the pose of the sensor origin
        if self.cfg.track_pose:
            pose = self.body_physx_view.get_transforms().view(-1, self._num_bodies, 7)[env_ids]
            pose[..., 3:] = convert_quat(pose[..., 3:], to="wxyz")
            self._data.pos_w[env_ids], self._data.quat_w[env_ids] = pose.split([3, 4], dim=-1)

        # obtain contact points
        if self.cfg.track_contact_points:
            _, buffer_contact_points, _, _, buffer_count, buffer_start_indices = (
                self.contact_physx_view.get_contact_data(dt=self._sim_physics_dt)
            )
            self._data.contact_pos_w[env_ids] = self._unpack_contact_buffer_data(
                buffer_contact_points, buffer_count, buffer_start_indices
            )[env_ids]

        # obtain friction forces
        if self.cfg.track_friction_forces:
            friction_forces, _, buffer_count, buffer_start_indices = self.contact_physx_view.get_friction_data(
                dt=self._sim_physics_dt
            )
            self._data.friction_forces_w[env_ids] = self._unpack_contact_buffer_data(
                friction_forces, buffer_count, buffer_start_indices, avg=False, default=0.0
            )[env_ids]

        # obtain the air time
        if self.cfg.track_air_time:
            # -- time elapsed since last update
            # since this function is called every frame, we can use the difference to get the elapsed time
            elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
            # -- check contact state of bodies
            is_contact = torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1) > self.cfg.force_threshold
            is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact
            is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact
            # -- update the last contact time if body has just become in contact
            self._data.last_air_time[env_ids] = torch.where(
                is_first_contact,
                self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
                self._data.last_air_time[env_ids],
            )
            # -- increment time for bodies that are not in contact
            self._data.current_air_time[env_ids] = torch.where(
                ~is_contact, self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0
            )
            # -- update the last contact time if body has just detached
            self._data.last_contact_time[env_ids] = torch.where(
                is_first_detached,
                self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
                self._data.last_contact_time[env_ids],
            )
            # -- increment time for bodies that are in contact
            self._data.current_contact_time[env_ids] = torch.where(
                is_contact, self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0
            )

    def _unpack_contact_buffer_data(
        self,
        contact_data: torch.Tensor,
        buffer_count: torch.Tensor,
        buffer_start_indices: torch.Tensor,
        avg: bool = True,
        default: float = float("nan"),
    ) -> torch.Tensor:
        """
        Unpacks and aggregates contact data for each (env, body, filter) group.

        This function vectorizes the following nested loop:

        for i in range(self._num_bodies * self._num_envs):
            for j in range(self.contact_physx_view.filter_count):
                start_index_ij = buffer_start_indices[i, j]
                count_ij = buffer_count[i, j]
                self._contact_position_aggregate_buffer[i, j, :] = torch.mean(
                    contact_data[start_index_ij : (start_index_ij + count_ij), :], dim=0
                )

        For more details, see the `RigidContactView.get_contact_data() documentation <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.RigidContactView.get_contact_data>`_.

        Args:
            contact_data: Flat tensor of contact data, shape (N_envs * N_bodies, 3).
            buffer_count: Number of contact points per (env, body, filter), shape (N_envs * N_bodies, N_filters).
            buffer_start_indices: Start indices for each (env, body, filter), shape (N_envs * N_bodies, N_filters).
            avg: If True, average the contact data for each group; if False, sum the data. Defaults to True.
            default: Default value to use for groups with zero contacts. Defaults to NaN.

        Returns:
            Aggregated contact data, shape (N_envs, N_bodies, N_filters, 3).
        """
        """解包和汇集每个 (env，车身，过器) 组的联系数据。

        这个函数向量化了以下嵌套循环:

        for i in range(self._num_bodies * self._num_envs):
            for j in range(self.contact_physx_view.filter_count):
                start_index_ij = buffer_start_indices[i, j]
                count_ij = buffer_count[i, j]
                self._contact_position_aggregate_buffer[i， j， :] = torch.mean(contact_data[start_index_ij :
                (start_index_ij + count_ij)， :]，dim=0)

        更多详情请参见`RigidContactView.get_contact_data() documentation <https://docs.omniverse.nvidia.com/kit/doc
        s/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physic
        s.tensors.impl.api.RigidContactView.get_contact_data>`_。

        参数：
            contact_data: 接触数据的平坦数，形状 (N_envs * N_body， 3)。
            buffer_count: 每个接触点的数量 (env，体，过器)，形状 (N_envs * N_body， N_filters)。
            buffer_start_indices: 每个 (env，机体，过器)，形状 (N_envs * N_body， N_filters) 的启动索引。
            avg: 如果True，平均每个组的联系数据；如果False，总结数据。
                 默认为 True。
            default: 默认值用于零接触组。
                     在NaN上默认。

        返回：
            总结的联系数据，形状 (N_envs，N_body，N_filters， 3)。
        """
        counts, starts = buffer_count.view(-1), buffer_start_indices.view(-1)
        n_rows, total = counts.numel(), int(counts.sum())
        agg = torch.full((n_rows, 3), default, device=self._device, dtype=contact_data.dtype)
        if total > 0:
            row_ids = torch.repeat_interleave(torch.arange(n_rows, device=self._device), counts)

            block_starts = counts.cumsum(0) - counts
            deltas = torch.arange(row_ids.numel(), device=counts.device) - block_starts.repeat_interleave(counts)
            flat_idx = starts[row_ids] + deltas

            pts = contact_data.index_select(0, flat_idx)
            agg = agg.zero_().index_add_(0, row_ids, pts)
            agg = agg / counts.clamp_min(1).unsqueeze(-1) if avg else agg
            agg[counts == 0] = default

        return agg.view(self._num_envs * self.num_bodies, -1, 3).view(
            self._num_envs, self._num_bodies, self.contact_physx_view.filter_count, 3
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "contact_visualizer"):
                self.contact_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.contact_visualizer.set_visibility(True)
        else:
            if hasattr(self, "contact_visualizer"):
                self.contact_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # safely return if view becomes invalid
        # note: this invalidity happens because of isaac sim view callbacks
        if self.body_physx_view is None:
            return
        # marker indices
        # 0: contact, 1: no contact
        net_contact_force_w = torch.norm(self._data.net_forces_w, dim=-1)
        marker_indices = torch.where(net_contact_force_w > self.cfg.force_threshold, 0, 1)
        # check if prim is visualized
        if self.cfg.track_pose:
            frame_origins: torch.Tensor = self._data.pos_w
        else:
            pose = self.body_physx_view.get_transforms()
            frame_origins = pose.view(-1, self._num_bodies, 7)[:, :, :3]
        # visualize
        self.contact_visualizer.visualize(frame_origins.view(-1, 3), marker_indices=marker_indices.view(-1))

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
        self._body_physx_view = None
        self._contact_physx_view = None
