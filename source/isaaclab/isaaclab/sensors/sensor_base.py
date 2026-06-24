# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for sensors.

This class defines an interface for sensors similar to how the :class:`isaaclab.assets.AssetBase` class works.
Each sensor class should inherit from this class and implement the abstract methods.
"""

from __future__ import annotations
"""传感器的基础类。

这类定义了类似于:class:`isaaclab.assets.AssetBase`类的传感器界面。
每个传感器类应继承这一类，并实施抽象方法。
"""

import builtins
import inspect
import re
import weakref
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

import omni.kit.app
import omni.timeline
from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager

import isaaclab.sim as sim_utils
from isaaclab.sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from .sensor_base_cfg import SensorBaseCfg


class SensorBase(ABC):
    """The base class for implementing a sensor.

    The implementation is based on lazy evaluation. The sensor data is only updated when the user
    tries accessing the data through the :attr:`data` property or sets ``force_compute=True`` in
    the :meth:`update` method. This is done to avoid unnecessary computation when the sensor data
    is not used.

    The sensor is updated at the specified update period. If the update period is zero, then the
    sensor is updated at every simulation step.
    """
    """基于传感器的基础类。

    实施基于惰的评估。
    传感器数据只有当用户通过:attr:`data`属性访问数据或设置``force_compute=True``时才会更新
    the :麻:`update`方法。
         这样才能避免传感器数据在
    不使用。

    传感器在指定更新期间更新。
    如果更新时间为零，则传感器每次仿真步骤都会更新。
    """

    def __init__(self, cfg: SensorBaseCfg):
        """Initialize the sensor class.

        Args:
            cfg: The configuration parameters for the sensor.
        """
        """启动传感器类。

        参数：
            cfg: 传感器的配置参数
        """
        # check that config is valid
        if cfg.history_length < 0:
            raise ValueError(f"History length must be greater than 0! Received: {cfg.history_length}")
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg.copy()
        # flag for whether the sensor is initialized
        self._is_initialized = False
        # flag for whether the sensor is in visualization mode
        self._is_visualizing = False
        # get stage handle
        self.stage = get_current_stage()

        # register various callback functions
        self._register_callbacks()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        """取消回电话。"""
        # clear physics events handles
        self._clear_callbacks()

    """
    Properties
    """
    """产品
    """

    @property
    def is_initialized(self) -> bool:
        """Whether the sensor is initialized.

        Returns True if the sensor is initialized, False otherwise.
        """
        """传感器是否启动。

        如果传感器启动，则返回True，否则返回False。
        """
        return self._is_initialized

    @property
    def num_instances(self) -> int:
        """Number of instances of the sensor.

        This is equal to the number of sensors per environment multiplied by the number of environments.
        """
        """传感器的实例数。

        这等于每个环境的传感器数乘以环境数。
        """
        return self._num_envs

    @property
    def device(self) -> str:
        """Memory device for computation."""
        """计算的内存设备。"""
        return self._device

    @property
    @abstractmethod
    def data(self) -> Any:
        """Data from the sensor.

        This property is only updated when the user tries to access the data. This is done to avoid
        unnecessary computation when the sensor data is not used.

        For updating the sensor when this property is accessed, you can use the following
        code snippet in your sensor implementation:

        .. code-block:: python

            # update sensors if needed
            self._update_outdated_buffers()
            # return the data (where `_data` is the data for the sensor)
            return self._data
        """
        """传感器的数据。

        当用户试图访问数据时才会更新此属性。
        在没有使用传感器数据时，这可以避免不必要的计算。

        在访问此属性时，可在传感器实现中使用以下代码片段更新传感器:

        .. code-block:: python

            # update sensors if needed
            self._update_outdated_buffers()
            # return the data (where `_data` is the data for the sensor)
            return self._data
        """
        raise NotImplementedError

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the sensor has a debug visualization implemented."""
        """传感器是否实现了调试可视化。"""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    """
    Operations
    """
    """运营
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the sensor data.

        Args:
            debug_vis: Whether to visualize the sensor data.

        Returns:
            Whether the debug visualization was successfully set. False if the sensor
            does not support debug visualization.
        """
        """设定是否可可视化传感器数据。

        参数：
            debug_vis: 是否可视化传感器数据。

        返回：
            设置错误可视化是否成功。
            False如果传感器不支持调试可视化。
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False
        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
        # toggle debug visualization flag
        self._is_visualizing = debug_vis
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the sensor internals.

        Args:
            env_ids: The sensor ids to reset. Defaults to None.
        """
        """调整传感器内部。

        参数：
            env_ids: 传感器的识别要重置。
                     默认为 None。
        """
        # Resolve sensor ids
        if env_ids is None:
            env_ids = slice(None)
        # Reset the timestamp for the sensors
        self._timestamp[env_ids] = 0.0
        self._timestamp_last_update[env_ids] = 0.0
        # Set all reset sensors to outdated so that they are updated when data is called the next time.
        self._is_outdated[env_ids] = True

    def update(self, dt: float, force_recompute: bool = False):
        # Update the timestamp for the sensors
        self._timestamp += dt
        self._is_outdated |= self._timestamp - self._timestamp_last_update + 1e-6 >= self.cfg.update_period
        # Update the buffers
        # TODO (from @mayank): Why is there a history length here when it doesn't mean anything in the sensor base?!?
        #   It is only for the contact sensor but there we should redefine the update function IMO.
        if force_recompute or self._is_visualizing or (self.cfg.history_length > 0):
            self._update_outdated_buffers()

    """
    Implementation specific.
    """
    """具体实施情况
    """

    @abstractmethod
    def _initialize_impl(self):
        """Initializes the sensor-related handles and internal buffers."""
        """启动与传感器相关的句柄和内部缓冲器。"""
        # Obtain Simulation Context
        sim = sim_utils.SimulationContext.instance()
        if sim is None:
            raise RuntimeError("Simulation Context is not initialized!")
        # Obtain device and backend
        self._device = sim.device
        self._backend = sim.backend
        self._sim_physics_dt = sim.get_physics_dt()
        # Count number of environments
        env_prim_path_expr = self.cfg.prim_path.rsplit("/", 1)[0]
        self._parent_prims = sim_utils.find_matching_prims(env_prim_path_expr)
        self._num_envs = len(self._parent_prims)
        # Boolean tensor indicating whether the sensor data has to be refreshed
        self._is_outdated = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        # Current timestamp (in seconds)
        self._timestamp = torch.zeros(self._num_envs, device=self._device)
        # Timestamp from last update
        self._timestamp_last_update = torch.zeros_like(self._timestamp)

        # Initialize debug visualization handle
        if self._debug_vis_handle is None:
            # set initial state of debug visualization
            self.set_debug_vis(self.cfg.debug_vis)

    @abstractmethod
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the sensor data for provided environment ids.

        This function does not perform any time-based checks and directly fills the data into the
        data container.

        Args:
            env_ids: The indices of the sensors that are ready to capture.
        """
        """填写提供环境身份的传感器数据。

        这种函数不会进行基于时间的检查，而是直接填充数据在数据容器中。

        参数：
            env_ids: 传感器准备捕获的指标。
        """
        raise NotImplementedError

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        """设置调试可视化到可视化对象。

        如果它们不存在，并且输入 ``debug_vis`` 是 True，
        如果可视化对象存在，函数应该将它们的可视性设置在舞台上。
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        This function calls the visualization objects and sets the data to visualize into them.
        """
        """检查错误可视化。

        这个函数将可视化对象调用，并设置数据可视化到它们中。
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    """
    Internal simulation callbacks.
    """
    """内部仿真回调。
    """

    def _register_callbacks(self):
        """Registers the timeline and prim deletion callbacks."""
        """记录时间表和prim删除回调。"""

        # register simulator callbacks (with weakref safety to avoid crashes on deletion)
        def safe_callback(callback_name, event, obj_ref):
            """Safely invoke a callback on a weakly-referenced object, ignoring ReferenceError if deleted."""
            """安全地调用一个弱引用的对象，如果删除ReferenceError，则忽略。"""
            try:
                obj = obj_ref
                getattr(obj, callback_name)(event)
            except ReferenceError:
                # Object has been deleted; ignore.
                pass

        # note: use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        # add callbacks for stage play/stop
        obj_ref = weakref.proxy(self)
        timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()

        # the order is set to 10 which is arbitrary but should be lower priority than the default order of 0
        # register timeline PLAY event callback (lower priority with order=10)
        self._initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY),
            lambda event, obj_ref=obj_ref: safe_callback("_initialize_callback", event, obj_ref),
            order=10,
        )
        # register timeline STOP event callback (lower priority with order=10)
        self._invalidate_initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda event, obj_ref=obj_ref: safe_callback("_invalidate_initialize_callback", event, obj_ref),
            order=10,
        )
        # register prim deletion callback
        self._prim_deletion_callback_id = SimulationManager.register_callback(
            lambda event, obj_ref=obj_ref: safe_callback("_on_prim_deletion", event, obj_ref),
            event=IsaacEvents.PRIM_DELETION,
        )

    def _initialize_callback(self, event):
        """Initializes the scene elements.

        Note:
            PhysX handles are only enabled once the simulator starts playing. Hence, this function needs to be
            called whenever the simulator "plays" from a "stop" state.
        """
        """启动场景元素。

        说明：
            在仿真器开始播放后才启用PhysX句柄。
            因此，每当仿真器从"停止"状态中"播放"时，需要调用此函数。
        """
        if not self._is_initialized:
            try:
                self._initialize_impl()
            except Exception as e:
                if builtins.ISAACLAB_CALLBACK_EXCEPTION is None:
                    builtins.ISAACLAB_CALLBACK_EXCEPTION = e
            self._is_initialized = True

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        """破坏场景元素。"""
        self._is_initialized = False
        if self._debug_vis_handle is not None:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    def _on_prim_deletion(self, prim_path: str) -> None:
        """Invalidates and deletes the callbacks when the prim is deleted.

        Args:
            prim_path: The path to the prim that is being deleted.

        Note:
            This function is called when the prim is deleted.
        """
        """在删除prim时，将反调无效和删除。

        参数：
            prim_path: 删除的prim的路径。

        说明：
            当删除prim时，这个函数会被调用。
        """
        if prim_path == "/":
            self._clear_callbacks()
            return
        result = re.match(
            pattern="^" + "/".join(self.cfg.prim_path.split("/")[: prim_path.count("/") + 1]) + "$", string=prim_path
        )
        if result:
            self._clear_callbacks()

    def _clear_callbacks(self) -> None:
        """Clears the callbacks."""
        """清除回调。"""
        if self._prim_deletion_callback_id:
            SimulationManager.deregister_callback(self._prim_deletion_callback_id)
            self._prim_deletion_callback_id = None
        if self._initialize_handle:
            self._initialize_handle.unsubscribe()
            self._initialize_handle = None
        if self._invalidate_initialize_handle:
            self._invalidate_initialize_handle.unsubscribe()
            self._invalidate_initialize_handle = None
        # clear debug visualization
        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    """
    Helper functions.
    """
    """辅助函数。
    """

    def _update_outdated_buffers(self):
        """Fills the sensor data for the outdated sensors."""
        """填充过时传感器的传感器数据。"""
        outdated_env_ids = self._is_outdated.nonzero().squeeze(-1)
        if len(outdated_env_ids) > 0:
            # obtain new data
            self._update_buffers_impl(outdated_env_ids)
            # update the timestamp from last update
            self._timestamp_last_update[outdated_env_ids] = self._timestamp[outdated_env_ids]
            # set outdated flag to false for the updated sensors
            self._is_outdated[outdated_env_ids] = False
