# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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
    from .asset_base_cfg import AssetBaseCfg


class AssetBase(ABC):
    """The base interface class for assets.

    An asset corresponds to any physics-enabled object that can be spawned in the simulation. These include
    rigid objects, articulated objects, deformable objects etc. The core functionality of an asset is to
    provide a set of buffers that can be used to interact with the simulator. The buffers are updated
    by the asset class and can be written into the simulator using the their respective ``write`` methods.
    This allows a convenient way to perform post-processing operations on the buffers before writing them
    into the simulator and obtaining the corresponding simulation results.

    The class handles both the spawning of the asset into the USD stage as well as initialization of necessary
    physics handles to interact with the asset. Upon construction of the asset instance, the prim corresponding
    to the asset is spawned into the USD stage if the spawn configuration is not None. The spawn configuration
    is defined in the :attr:`AssetBaseCfg.spawn` attribute. In case the configured :attr:`AssetBaseCfg.prim_path`
    is an expression, then the prim is spawned at all the matching paths. Otherwise, a single prim is spawned
    at the configured path. For more information on the spawn configuration, see the
    :mod:`isaaclab.sim.spawners` module.

    Unlike Isaac Sim interface, where one usually needs to call the
    :meth:`isaacsim.core.prims.XFormPrim.initialize` method to initialize the PhysX handles, the asset
    class automatically initializes and invalidates the PhysX handles when the stage is played/stopped. This
    is done by registering callbacks for the stage play/stop events.

    Additionally, the class registers a callback for debug visualization of the asset if a debug visualization
    is implemented in the asset class. This can be enabled by setting the :attr:`AssetBaseCfg.debug_vis` attribute
    to True. The debug visualization is implemented through the :meth:`_set_debug_vis_impl` and
    :meth:`_debug_vis_callback` methods.
    """
    """资产的基础接口类。

    一个资产与任何可以在仿真中产生的物理支持的对象相匹配。
    这些包括硬物体，关节物体，变形物体等。
    一个资产的核心功能是提供一组可用于与仿真器交互的缓冲器。
    按资产类别更新缓冲，可使用其各自的``write``方法写入仿真器。
    这使得在写入仿真器并获得相应的仿真结果之前，在缓冲器上进行后处理操作的方便方式。

    该类处理资产进入USD阶段的产卵，以及与资产交互的必要物理句柄的初始化。
    在构建资产实例时，对资产相应的prim将产生到USD阶段，如果生成配置不是None。
    在:attr:`AssetBaseCfg.spawn`属性中定义了产卵配置。
    如果配置的:attr:`AssetBaseCfg.prim_path`是表达式，则在所有匹配的路径上产生prim。
    否则，在配置的路径上产生单个prim。
    查看:mod:`isaaclab.sim.spawners`模块。

    与Isaac Sim界面不同，通常需要调用:meth:`isaacsim.core.prims.XFormPrim.initialize`方法来初始化PhysX句柄，
    class automatically initializes and invalidates the PhysX handles when the stage is played/stopped. This
    通过记录舞台剧/停止事件的回调。

    此外，如果在资产类别中实现了调试可视化，该类还会记录对资产的调试可视化回调。
    这可以通过设置:attr:`AssetBaseCfg.debug_vis`属性为True来实现。
    通过:meth:`_set_debug_vis_impl`和:meth:`_debug_vis_callback`方法实现了调试可视化。
    """

    def __init__(self, cfg: AssetBaseCfg):
        """Initialize the asset base.

        Args:
            cfg: The configuration class for the asset.

        Raises:
            RuntimeError: If no prims found at input prim path or prim path expression.
        """
        """启动资产基础。

        参数：
            cfg: 资产的配置类。

        异常：
            RuntimeError: 如果输入prim路径或prim路径表达没有prims。
        """
        # check that the config is valid    检查所有 MISSING 字段是否已填
        cfg.validate()
        # store inputs      深拷贝配置，防止外部修改影响内部状态
        self.cfg = cfg.copy()
        # flag for whether the asset is initialized     标记"尚未初始化"（PhysX 句柄等 PLAY 时才创建）
        self._is_initialized = False
        # get stage handle      获取 USD Stage 句柄
        self.stage = get_current_stage()

        # check if base asset path is valid
        # note: currently the spawner does not work if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Robot_[1,2]" since the spawner will not
        #   know which prim to spawn. This is a limitation of the spawner and not the asset.
        asset_path = self.cfg.prim_path.split("/")[-1]
        asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", asset_path) is None
        # spawn the asset
        if self.cfg.spawn is not None and not asset_path_is_regex:
            self.cfg.spawn.func(
                self.cfg.prim_path,
                self.cfg.spawn,
                translation=self.cfg.init_state.pos,
                orientation=self.cfg.init_state.rot,
            )
            '''
            正则叶子检测 + 执行生成
                只检查路径最后一段（叶子节点）是不是正则。
                    如果 prim_path 是 /World/envs/env_.*/Robot，叶子是 Robot（纯字母），可以 spawn。
                    如果路径是 /World/Robot_[1,2]，叶子是 Robot_[1,2]（含正则），就不能 spawn——因为 spawn 函数不知道具体该在哪个 prim 下创建资产。

                注意 init_state 的两个字段在这里被消费：
                    pos 传给 translation，rot 传给 orientation。这就是配置中初始位姿真正生效的地方。
            '''

        # check that spawn was successful
        matching_prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        if len(matching_prims) == 0:
            raise RuntimeError(f"Could not find prim with path {self.cfg.prim_path}.")
        '''
        验证生成结果
            即使 spawn 跳过了（None 或 正则叶子），也会检查该路径下是否已存在 prim。这在 spawn=None 时特别有用——确保你预先放置的资产确实存在。
        '''

        # register various callback functions
        self._register_callbacks()
        '''
        注册生命周期回调
            注册了三个 Isaac Sim 事件回调（见 asset_base.py:366-403）：
                事件	            回调方法	                            作用
                PLAY	            _initialize_callback	            仿真开始播放时，创建 PhysX 句柄、初始化 buffer、设置 _is_initialized = True
                STOP	            _invalidate_initialize_callback	    仿真停止时，清理 PhysX 句柄、设置 _is_initialized = False
                PRIM_DELETION	    _on_prim_deletion	                prim 被删除时，自动清理所有回调订阅
        '''

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        '''
        调试可视化初始化
            如果配置中 debug_vis=True，会注册一个渲染帧回调——每帧刷新时绘制碰撞体线框、关节轴等调试元素。
        '''

    def __del__(self):
        """Unsubscribe from the callbacks."""
        """取消回电话。"""
        # clear events handles
        self._clear_callbacks()

    """
    Properties
    """
    """产品
    """

    @property
    def is_initialized(self) -> bool:
        """Whether the asset is initialized.

        Returns True if the asset is initialized, False otherwise.
        """
        """资产是否启动。

        返回True如果资产初始化，否则False。
        """
        return self._is_initialized
    '''
    作用
        一个只读的布尔标志，告诉外界这个资产是否已经完成了 PhysX 初始化（即 _initialize_impl() 是否已执行）。
    设计意图
        还记得 __init__ 中的两步初始化吗？
        __init__ 创建了 USD prim，但 PhysX 句柄要等到 PLAY 回调才创建。
        is_initialized 就是这两阶段的分界线——在初始化完成前，任何依赖物理句柄的操作都应该被阻止。
    '''

    @property
    @abstractmethod
    def num_instances(self) -> int:
        """Number of instances of the asset.

        This is equal to the number of asset instances per environment multiplied by the number of environments.
        """
        """资产的实例数

        这等于每个环境的资产实例数乘以环境数。
        """
        return NotImplementedError
    '''
    作用
        返回资产的总实例数 = 每个环境的资产数 × 环境数。例如 4096 个并行环境，每个环境 1 个机器人，num_instances 就是 4096。
    设计意图
        Isaac Lab 中资产分为两类：
            单例资产（如 RigidObject）：每个环境一个实例
            多实例资产（如 Articulation）：可能一个环境有多个机器人
    '''

    @property
    def device(self) -> str:
        """Memory device for computation."""
        """计算的内存设备。"""
        return self._device
    '''
    作用
        返回资产计算所在的设备，通常是 "cuda:0" 或 "cpu"。所有 tensor 操作（位置读取、力矩写入）都在这个设备上执行。
    '''

    @property
    @abstractmethod
    def data(self) -> Any:
        """Data related to the asset."""
        """与资产相关的数据。"""
        return NotImplementedError
    '''
    作用
        返回资产的所有物理数据的结构化视图。不同子类返回不同类型：
            Articulation.data 返回 ArticulationData（含 joint_pos、joint_vel、root_pos_w 等）
            RigidObject.data 返回 RigidObjectData（含 root_pos_w、root_lin_vel_w 等）
    设计意图
        data 是 Isaac Lab 中最核心的抽象之一。
        它统一了所有资产数据的访问方式——不管你是四足机器人还是桌面方块，外部代码都通过 asset.data.xxx 来读取状态。这就是多态在数据层的体现。
    '''

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the asset has a debug visualization implemented."""
        """资产是否实现了调试可视化。"""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code
    '''
    作用
        在运行时检测子类是否真正实现了 _set_debug_vis_impl 方法，而不是简单继承了基类的 raise NotImplementedError 桩。
    这是怎么工作的？
        has_debug_vis_implementation 用 inspect.getsource() 获取该方法的源代码文本，然后检查其中是否包含字符串 "NotImplementedError"：
            如果包含 → 子类没有重写，返回 False
            如果不包含 → 子类实现了真正逻辑，返回 True
    '''

    """
    Operations.
    """
    """操作。
    """

    '''
    输入参数
        visible：
            True 让资产可见，False 隐藏资产。直接在 USD Stage 上操作节点的可见性属性。
        env_ids：
            要操作的环境索引列表。可以是 Python list、range 对象或 torch.Tensor。默认为 None，表示操作全部实例。
    '''
    def set_visibility(self, visible: bool, env_ids: Sequence[int] | None = None):
        """Set the visibility of the prims corresponding to the asset.

        This operation affects the visibility of the prims corresponding to the asset in the USD stage.
        It is useful for toggling the visibility of the asset in the simulator. For instance, one can
        hide the asset when it is not being used to reduce the rendering overhead.

        Note:
            This operation uses the PXR API to set the visibility of the prims. Thus, the operation
            may have an overhead if the number of prims is large.

        Args:
            visible: Whether to make the prims visible or not.
            env_ids: The indices of the object to set visibility. Defaults to None (all instances).
        """
        """设置对资产相应的prims的可见性。

        这一操作影响prims相应的USD阶段资产的可视性。
        它可用于仿真器中的资产可见性转换。
        例如，如果不用于降低 over渲染总费用，则可以隐藏资产。

        说明：
            这种操作使用PXR API来设置prims的可见性。
            因此，如果prims的数量很大，操作可能会有上层费用。

        参数：
            visible: 是否让prims可见。
            env_ids: 设置可见性对象的指标。
                     在 None 中默认设置 (所有实例)。
        """
        # resolve the environment ids
        if env_ids is None:
            env_ids = range(len(self._prims))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()

        # obtain the prims corresponding to the asset
        # note: we only want to find the prims once since this is a costly operation
        if not hasattr(self, "_prims"):
            self._prims = sim_utils.find_matching_prims(self.cfg.prim_path)

        # iterate over the environment ids
        for env_id in env_ids:
            sim_utils.set_prim_visibility(self._prims[env_id], visible)

    '''
    输入参数 debug_vis：
        True 打开调试可视化（显示碰撞体线框、关节轴等），False 关闭调试可视化。
    返回值 bool：
        True 表示设置成功；False 表示该资产根本没有实现调试可视化（子类未覆写 _set_debug_vis_impl）。
    副作用：
        如果开启，会在 Omniverse 的渲染事件流中注册一个每帧回调，持续更新调试绘制内容。
    '''
    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the asset data.

        Args:
            debug_vis: Whether to visualize the asset data.

        Returns:
            Whether the debug visualization was successfully set. False if the asset
            does not support debug visualization.
        """
        """设定是否可查看资产数据。

        参数：
            debug_vis: 是否可视化资产数据。

        返回：
            设置错误可视化是否成功。
            False如果资产不支持调试可视化。
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False
        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
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

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets all internal buffers of selected environments.

        Args:
            env_ids: The indices of the object to reset. Defaults to None (all instances).
        """
        """重置选定的环境的所有内部缓冲器。

        参数：
            env_ids: 将重置的对象的索引。
                     在 None 中默认设置 (所有实例)。
        """
        raise NotImplementedError

    @abstractmethod
    def write_data_to_sim(self):
        """Writes data to the simulator."""
        """在仿真器上写数据。"""
        raise NotImplementedError

    @abstractmethod
    def update(self, dt: float):
        """Update the internal buffers.

        The time step ``dt`` is used to compute numerical derivatives of quantities such as joint
        accelerations which are not provided by the simulator.

        Args:
            dt: The amount of time passed from last ``update`` call.
        """
        """更新内部缓冲器。

        时间步骤 ``dt``用于计算数值衍生物，例如仿真器未提供的联合加速。

        参数：
            dt: 从最后一次``update``电话以来的时间。
        """
        raise NotImplementedError

    """
    Implementation specific.
    """
    """具体实施情况
    """

    @abstractmethod
    def _initialize_impl(self):
        """Initializes the PhysX handles and internal buffers."""
        """启动PhysX句柄和内部缓冲器。"""
        raise NotImplementedError
    '''
    延迟初始化的"最终执行者"
    作用
        创建 PhysX 物理句柄并初始化所有内部缓冲区。
        这是真正的初始化——__init__ 只是做了轻量级的 USD 生成和回调注册，PhysX 句柄要等到仿真"播放"时才创建。
    '''

    '''
    可视化几何体的"开关"
    作用
        由 set_debug_vis() 调用，负责创建或隐藏调试可视化几何体（如碰撞体网格、关节坐标轴）。
            debug_vis=True：如果可视化对象还不存在，创建它们，然后显示
            debug_vis=False：隐藏可视化对象（不销毁，下次开启更快）
    '''
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

    '''
    每帧更新的"动画师"
    作用
        由 Omniverse 的 Post-Update 事件流每帧触发，负责更新调试可视化几何体的位置和姿态，以反映当前物理状态。
    '''
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

    '''
    在 Omniverse Kit 的事件系统中注册三个回调订阅，
        返回的句柄保存在 self._initialize_handle、self._invalidate_initialize_handle、self._prim_deletion_callback_id 中。
    调用时机：
        在 __init__ 末尾调用一次（asset_base.py:127），即每个资产实例创建时执行一次。
    '''
    def _register_callbacks(self):
        """Registers the timeline and prim deletion callbacks."""
        """记录时间表和prim删除回调。"""
        '''
        _register_callbacks()
                │
                ▼
        ┌─ 1. 定义 safe_callback 内嵌函数 ────────────────────┐
        │    包装回调调用，捕获 ReferenceError 防止崩溃           │
        └──────────────────────────────────────────────────────┘
                │
                ▼
        ┌─ 2. 创建 self 的弱引用代理 ───────────────────────────┐
        │    obj_ref = weakref.proxy(self)                      │
        │    ↑ 所有回调通过它访问 self，防止循环引用              │
        └──────────────────────────────────────────────────────┘
                │
                ▼
        ┌─ 3. 获取时间轴事件流 ──────────────────────────────────┐
        │    timeline_event_stream =                             │
        │        get_timeline_interface()                        │
        │            .get_timeline_event_stream()                │
        └──────────────────────────────────────────────────────┘
                │
                ├── 注册 PLAY 事件（order=10）─────────────────────┐
                │    → 回调: safe_callback(                        │
                │              "_initialize_callback",             │
                │              event, obj_ref)                     │
                │    → 句柄: self._initialize_handle               │
                └────────────────────────────────────────────────┘
                │
                ├── 注册 STOP 事件（order=10）─────────────────────┐
                │    → 回调: safe_callback(                        │
                │              "_invalidate_initialize_callback",  │
                │              event, obj_ref)                     │
                │    → 句柄: self._invalidate_initialize_handle    │
                └────────────────────────────────────────────────┘
                │
                └── 注册 PRIM_DELETION 事件 ──────────────────────┐
                    → 回调: safe_callback(                        │
                            "_on_prim_deletion",                │
                            event, obj_ref)                     │
                    → 句柄: self._prim_deletion_callback_id       │
                    └────────────────────────────────────────────┘
        '''

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
            # obtain simulation related information
            self._backend = SimulationManager.get_backend()
            self._device = SimulationManager.get_physics_sim_device()
            # initialize the asset
            try:
                self._initialize_impl()
            except Exception as e:
                if builtins.ISAACLAB_CALLBACK_EXCEPTION is None:
                    builtins.ISAACLAB_CALLBACK_EXCEPTION = e
            # set flag
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
