# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""
"""环境的基本配置。

本模块定义了环境的一般配置。
它包括配置环境实例，观众设置和仿真参数的参数。
"""

from typing import Any, Literal

from isaaclab.utils import configclass

from .spawners.materials import RigidBodyMaterialCfg


@configclass
class PhysxCfg:
    """Configuration for PhysX solver-related parameters.

    These parameters are used to configure the PhysX solver. For more information, see the `PhysX 5 SDK
    documentation`_.

    PhysX 5 supports GPU-accelerated physics simulation. This is enabled by default, but can be disabled
    by setting the :attr:`~SimulationCfg.device` to ``cpu`` in :class:`SimulationCfg`. Unlike CPU PhysX, the GPU
    simulation feature is unable to dynamically grow all the buffers. Therefore, it is necessary to provide
    a reasonable estimate of the buffer sizes for GPU features. If insufficient buffer sizes are provided, the
    simulation will fail with errors and lead to adverse behaviors. The buffer sizes can be adjusted through the
    ``gpu_*`` parameters.

    .. _PhysX 5 SDK documentation: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxSceneDesc.html

    """
    """对PhysX溶解器相关参数的配置。

    这些参数用于配置PhysX解决器。
    更多信息请参见`PhysX 5 SDK documentation`_。

    PhysX5支持GPU加速物理仿真。
    默认启用，但可以通过设置:attr:`~SimulationCfg.device`到``cpu``在:class:`SimulationCfg`中禁用。
    与CPUPhysX不同，GPU仿真功能无法动态增长所有缓冲器。
    因此，对于GPU特征，必须提供合理的缓冲尺寸估计。
    如果提供不够的缓冲尺寸，则仿真会出现错误，导致不良行为。
    缓冲器尺寸可以通过``gpu_*``参数调整。

    .. _PhysX 5 SDK documentation: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxSceneDesc.html
    """

    solver_type: Literal[0, 1] = 1
    """The type of solver to use.Default is 1 (TGS).

    Available solvers:

    * :obj:`0`: PGS (Projective Gauss-Seidel)
    * :obj:`1`: TGS (Temporal Gauss-Seidel)
    """
    """use.Default的溶剂类型为1 (TGS)。

    可用的解决器:

    * :obj:`0`:PGS (可预测高斯-西德尔)
    * :obj:`1`: TGS(临时加斯-西德尔)
    """

    solve_articulation_contact_last: bool = False
    """Changes the ordering inside the articulation solver. Default is False.

    PhysX employs a strict ordering for handling constraints in an articulation. The outcome of
    each constraint resolution modifies the joint and associated link speeds. However, the default
    ordering may not be ideal for gripping scenarios because the solver favours the constraint
    types that are resolved last. This is particularly true of stiff constraint systems that are hard
    to resolve without resorting to vanishingly small simulation timesteps.

    With dynamic contact resolution being such an important part of gripping, it may make
    more sense to solve dynamic contact towards the end of the solver rather than at the
    beginning. This parameter modifies the default ordering to enable this change.

    For more information, please check `here <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html#articulation-solver-order>`__.

    .. versionadded:: v2.3
        This parameter is only available with Isaac Sim 5.1.

    """
    """在关节溶解器内部的排序变化。
    默认是False。

    PhysX采用严格的排序来处理关节中的约束。
    每个限制分辨率的结果改变了关联和相关的链接速度。
    然而，默认排序可能不适合抓住场景，因为解决器更喜欢最后解决的限制类型。
    这尤其适用于难以解决的严格的制约系统，

    由于动态接触分辨率是抓取的重要组成部分，因此解决动态接触的方法可能比在开始更有意义。
    这一参数修改了默认的排序，以实现此变化。

    更多信息请查看`here <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articul
    ation_stability_guide.html#articulation-solver-order>`__。

    ..
    此参数仅可用于Isaac Sim 5.1
    """

    min_position_iteration_count: int = 1
    """Minimum number of solver position iterations (rigid bodies, cloth, particles etc.). Default is 1.

    .. note::

        Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
        the number of iterations specified by the actor with the highest iteration and clamps it to
        the range ``[min_position_iteration_count, max_position_iteration_count]``.
    """
    """溶剂位置回复的最小数量 (硬体，布料，颗粒等)。
    默认是1。

    .. 说明::

        在全宇宙中，每个物理演员都指定了自己的解决器代数。
        解析器将具有最高回复率的演员指定的回复数量取并将其扣到``[min_position_iteration_count， max_position_iteration_count]``范围。
    """

    max_position_iteration_count: int = 255
    """Maximum number of solver position iterations (rigid bodies, cloth, particles etc.). Default is 255.

    .. note::

        Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
        the number of iterations specified by the actor with the highest iteration and clamps it to
        the range ``[min_position_iteration_count, max_position_iteration_count]``.
    """
    """解决器位置反复的最大数量 (硬体，布料，颗粒等)。
    默认是255。

    .. 说明::

        在全宇宙中，每个物理演员都指定了自己的解决器代数。
        解析器将具有最高回复率的演员指定的回复数量取并将其扣到``[min_position_iteration_count， max_position_iteration_count]``范围。
    """

    min_velocity_iteration_count: int = 0
    """Minimum number of solver velocity iterations (rigid bodies, cloth, particles etc.). Default is 0.

    .. note::

        Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
        the number of iterations specified by the actor with the highest iteration and clamps it to
        the range ``[min_velocity_iteration_count, max_velocity_iteration_count]``.
    """
    """溶剂速度反复的最小数量 (硬体，布料，粒子等)。
    默认是0。

    .. 说明::

        在全宇宙中，每个物理演员都指定了自己的解决器代数。
        解析器将具有最高回复率的演员指定的回复数量取并将其扣到``[min_velocity_iteration_count， max_velocity_iteration_count]``范围。
    """

    max_velocity_iteration_count: int = 255
    """Maximum number of solver velocity iterations (rigid bodies, cloth, particles etc.). Default is 255.

    .. note::

        Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
        the number of iterations specified by the actor with the highest iteration and clamps it to
        the range ``[min_velocity_iteration_count, max_velocity_iteration_count]``.
    """
    """溶剂速度回复的最大数量 (硬体，布料，颗粒等)。
    默认是255。

    .. 说明::

        在全宇宙中，每个物理演员都指定了自己的解决器代数。
        解析器将具有最高回复率的演员指定的回复数量取并将其扣到``[min_velocity_iteration_count， max_velocity_iteration_count]``范围。
    """

    enable_ccd: bool = False
    """Enable a second broad-phase pass that makes it possible to prevent objects from tunneling through each other.
    Default is False."""
    """允许第二个宽相通过，使得物体可以防止彼此道。
    默认是False。
    """

    enable_stabilization: bool = False
    """Enable/disable additional stabilization pass in solver. Default is False.

    .. note::

        We recommend setting this flag to true only when the simulation step size is large
        (i.e., less than 30 Hz or more than 0.0333 seconds).

    .. warning::

        Enabling this flag may lead to incorrect contact forces report from the contact sensor.
    """
    """在溶剂中启用/禁用额外的稳定通过。
    默认是False。

    .. 说明::

        我们建议只有仿真步骤大小 (i.e.，低于30 Hz或超过0.0333秒) 时将此标志设置为 true。

    .. 警告::

        启用此标志可能会导致接触传感器错误的接触力报告。
    """

    enable_external_forces_every_iteration: bool = False
    """Enable/disable external forces every position iteration in the TGS solver. Default is False.

    When using the TGS solver (:attr:`solver_type` is 1), this flag allows enabling external forces every solver
    position iteration. This can help improve the accuracy of velocity updates. Consider enabling this flag if
    the velocities generated by the simulation are noisy. Increasing the number of velocity iterations, together
    with this flag, can help improve the accuracy of velocity updates.

    .. note::

        This flag is ignored when using the PGS solver (:attr:`solver_type` is 0).
    """
    """启用/禁用TGS解决器中的每个位置代的外部力量。
    默认是False。

    在使用TGS溶解器时 (:attr:`solver_type`为 1)，该旗允许在每个溶解器位置代中启用外部力量。
    这可以帮助提高速度更新的准确性。
    如果仿真所产生的速度有噪音，
    增加速度代数，共计
    with this flag, can help improve the accuracy of velocity updates.

    .. 说明::

        在使用PGS解决器时，这个标志被忽略 (:attr:`solver_type`是0)。
    """

    enable_enhanced_determinism: bool = False
    """Enable/disable improved determinism at the expense of performance. Defaults to False.

    For more information on PhysX determinism, please check `here`_.

    .. _here: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html#enhanced-determinism
    """
    """在性能代价上，可以/不能提高确定性。
    默认为 False。

    有关PhysX确定性更多信息，请查看`here`_。

    .. _here: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html#enhanced-determinism
    """

    bounce_threshold_velocity: float = 0.5
    """Relative velocity threshold for contacts to bounce (in m/s). Default is 0.5 m/s."""
    """接触者跳转的相对速度门 (m/s)。
    默认速度为0.5m/s。
    """

    friction_offset_threshold: float = 0.04
    """Threshold for contact point to experience friction force (in m). Default is 0.04 m."""
    """接触点经历摩擦力的门 (m)。
    默认值为0.04m。
    """

    friction_correlation_distance: float = 0.025
    """Distance threshold for merging contacts into a single friction anchor point (in m). Default is 0.025 m."""
    """将接触物合并成单个摩擦点的距离门 (m)。
    默认值为0.025m。
    """

    gpu_max_rigid_contact_count: int = 2**23
    """Size of rigid contact stream buffer allocated in pinned host memory. Default is 2 ** 23."""
    """在固定主机内存中分配的硬式接触流缓冲的尺寸。
    默认是2**23。
    """

    gpu_max_rigid_patch_count: int = 5 * 2**15
    """Size of the rigid contact patch stream buffer allocated in pinned host memory. Default is 5 * 2 ** 15."""
    """在固定的主机内存中分配的硬接口补丁流缓冲器的大小。
    默认是5*2**15。
    """

    gpu_found_lost_pairs_capacity: int = 2**21
    """Capacity of found and lost buffers allocated in GPU global memory. Default is 2 ** 21.

    This is used for the found/lost pair reports in the BP.
    """
    """在GPU全球内存中分配的发现和丢失缓冲容量。
    默认是2**21。

    在BP中用于发现/丢失对报告。
    """

    gpu_found_lost_aggregate_pairs_capacity: int = 2**25
    """Capacity of found and lost buffers in aggregate system allocated in GPU global memory.
    Default is 2 ** 25.

    This is used for the found/lost pair reports in AABB manager.
    """
    """在 GPU 全球内存中分配的集成系统中找到和丢失缓冲器的容量。
    默认是2**25。

    在 AABB 管理器中，用于找到/丢失对报告。
    """

    gpu_total_aggregate_pairs_capacity: int = 2**21
    """Capacity of total number of aggregate pairs allocated in GPU global memory. Default is 2 ** 21."""
    """在 GPU 全球内存中分配的总数组对的容量。
    默认是2**21。
    """

    gpu_collision_stack_size: int = 2**26
    """Size of the collision stack buffer allocated in pinned host memory. Default is 2 ** 26."""
    """在固定主机内存中分配的碰撞堆积缓冲器的大小。
    默认是2*26。
    """

    gpu_heap_capacity: int = 2**26
    """Initial capacity of the GPU and pinned host memory heaps. Additional memory will be allocated
    if more memory is required. Default is 2 ** 26."""
    """GPU 的初始容量和固定的主机内存堆。
    将分配额外的内存
    if more memory is required. Default is 2 ** 26.
    """

    gpu_temp_buffer_capacity: int = 2**24
    """Capacity of temp buffer allocated in pinned host memory. Default is 2 ** 24."""
    """在固定的主机内存中分配的临时缓冲容量。
    默认是2**24。
    """

    gpu_max_num_partitions: int = 8
    """Limitation for the partitions in the GPU dynamics pipeline. Default is 8.

    This variable must be power of 2. A value greater than 32 is currently not supported. Range: (1, 32)
    """
    """在GPU动力管道中的分区的限制。
    默认是8。

    这种变量必须是2。
    目前不支持超过32的值。
    范围: (1， 32)
    """

    gpu_max_soft_body_contacts: int = 2**20
    """Size of soft body contacts stream buffer allocated in pinned host memory. Default is 2 ** 20."""
    """在固定的主机内存中分配的软体接触流缓冲器的大小。
    默认是2**20。
    """

    gpu_max_particle_contacts: int = 2**20
    """Size of particle contacts stream buffer allocated in pinned host memory. Default is 2 ** 20."""
    """在固定的主机内存中分配的粒子接触流缓冲器的大小。
    默认是2**20。
    """


@configclass
class RenderCfg:
    """Configuration for Omniverse RTX Renderer.

    These parameters are used to configure the Omniverse RTX Renderer.

    The defaults for IsaacLab are set in the experience files:

    * ``apps/isaaclab.python.rendering.kit``: Setting used when running the simulation with the GUI enabled.
    * ``apps/isaaclab.python.headless.rendering.kit``: Setting used when running the simulation in headless mode.

    Setting any value here will override the defaults of the experience files.

    For more information, see the `Omniverse RTX Renderer documentation`_.

    .. _Omniverse RTX Renderer documentation: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer.html
    """
    """对于全宇宙RTX发射器的配置。

    这些参数用于配置Omniverse RTX Renderer。

    对于 IsaacLab 的默认设置在体验文件中:

    * ``apps/isaaclab.python.rendering.kit``:在启用GUI的情况下运行仿真时使用设置。
    * ``apps/isaaclab.python.headless.rendering.kit``:在无头模式下运行仿真时使用设置。

    在此设置任何值将取消体验文件的默认设置。

    更多信息请参见`Omniverse RTX Renderer documentation`_。

    .. _Omniverse RTX Renderer documentation: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer.html
    """

    enable_translucency: bool | None = None
    """Enables translucency for specular transmissive surfaces such as glass at the cost of some performance.
    Default is False.

    This is set by the variable: ``/rtx/translucency/enabled``.
    """
    """通过一些性能，可使玻璃等镜像传输表面具有透明度。
    默认是False。

    这是由变量:``/rtx/translucency/enabled``设置的。
    """

    enable_reflections: bool | None = None
    """Enables reflections at the cost of some performance. Default is False.

    This is set by the variable: ``/rtx/reflections/enabled``.
    """
    """能使一些表现产生反思。
    默认是False。

    这是由变量:``/rtx/reflections/enabled``设置的。
    """

    enable_global_illumination: bool | None = None
    """Enables Diffused Global Illumination at the cost of some performance. Default is False.

    This is set by the variable: ``/rtx/indirectDiffuse/enabled``.
    """
    """在某些性能成本下，可实现全球光。
    默认是False。

    这是由变量:``/rtx/indirectDiffuse/enabled``设置的。
    """

    antialiasing_mode: Literal["Off", "FXAA", "DLSS", "TAA", "DLAA"] | None = None
    """Selects the anti-aliasing mode to use. Defaults to DLSS.

    - **DLSS**: Boosts performance by using AI to output higher resolution frames from a lower resolution input.
      DLSS samples multiple lower resolution images and uses motion data and feedback from prior frames to
      reconstruct native quality images.
    - **DLAA**: Provides higher image quality with an AI-based anti-aliasing technique. DLAA uses the same
      Super Resolution technology developed for DLSS, reconstructing a native resolution image to maximize
      image quality.

    This is set by the variable: ``/rtx/post/dlss/execMode``.
    """
    """选择使用的防模式。
    在DLSS上默认。

    - **DLSS**:通过使用 AI 来输出更高分辨率的框架，提高性能。 DLSS 采样多个更低分辨率的图像，并使用之前的框架的运动数据和反来重建原生质量图像。
    - **DLAA**:通过基于AI的反ali化技术提供更高的图像质量.DLAA使用了DLSS开发的相同的超级分辨率技术，重建原生分辨率图像以最大化图像质量。

    这是由变量:``/rtx/post/dlss/execMode``设置的。
    """

    enable_dlssg: bool | None = None
    """"Enables the use of DLSS-G. Default is False.

    DLSS Frame Generation boosts performance by using AI to generate more frames. DLSS analyzes sequential frames
    and motion data to create additional high quality frames.

    .. note::

        This feature requires an Ada Lovelace architecture GPU. Enabling this feature also enables additional
        thread-related activities, which can hurt performance.

    This is set by the variable: ``/rtx-transient/dlssg/enabled``.
    """
    """"可使用DLSS-G。
    默认是False。

    DLSS通过使用AI为了产生更多的框架。
    DLSS分析序列框架和运动数据，以创建额外的高质量框架。

    .. 说明::

        该函数需要Ada Lovelace架构GPU。
        启用此功能也可以增加与线程相关的事件，这可能会损害性能。

    这是由变量:``/rtx-transient/dlssg/enabled``设置的。
    """

    enable_dl_denoiser: bool | None = None
    """Enables the use of a DL denoiser.

    The DL denoiser can help improve the quality of renders, but comes at a cost of performance.

    This is set by the variable: ``/rtx-transient/dldenoiser/enabled``.
    """
    """允许使用DL指标。

    在DL渲染器可以帮助提高渲染的质量，

    这是由变量:``/rtx-transient/dldenoiser/enabled``设置的。
    """

    dlss_mode: Literal[0, 1, 2, 3] | None = None
    """For DLSS anti-aliasing, selects the performance/quality tradeoff mode. Default is 0.

    Valid values are:

    * 0 (Performance)
    * 1 (Balanced)
    * 2 (Quality)
    * 3 (Auto)

    This is set by the variable: ``/rtx/post/dlss/execMode``.
    """
    """对于DLSS防，选择性能/质量权衡模式。
    默认是0。

    有效值为:

    * 0 (性能)
    * 1 (平衡)
    * 2 (质量)
    * 3 (汽车)

    这是由变量:``/rtx/post/dlss/execMode``设置的。
    """

    enable_direct_lighting: bool | None = None
    """Enable direct light contributions from lights. Default is False.

    This is set by the variable: ``/rtx/directLighting/enabled``.
    """
    """启用直接从灯光中提供光。
    默认是False。

    这是由变量:``/rtx/directLighting/enabled``设置的。
    """

    samples_per_pixel: int | None = None
    """Defines the Direct Lighting samples per pixel. Default is 1.

    A higher value increases the direct lighting quality at the cost of performance.

    This is set by the variable: ``/rtx/directLighting/sampledLighting/samplesPerPixel``.
    """
    """定义每像素的直接照明样本。
    默认是1。

    较高的价值增加了直接照明的质量，以牺牲性能。

    这是由变量:``/rtx/directLighting/sampledLighting/samplesPerPixel``设置的。
    """

    enable_shadows: bool | None = None
    """Enables shadows at the cost of performance. Defaults to True.

    When disabled, lights will not cast shadows.

    This is set by the variable: ``/rtx/shadows/enabled``.
    """
    """能使影子以性能为代价。
    默认为 True。

    当被禁用时，灯光不会影。

    这是由变量:``/rtx/shadows/enabled``设置的。
    """

    enable_ambient_occlusion: bool | None = None
    """Enables ambient occlusion at the cost of some performance. Default is False.

    This is set by the variable: ``/rtx/ambientOcclusion/enabled``.
    """
    """能使环境被遮蔽，
    默认是False。

    这是由变量:``/rtx/ambientOcclusion/enabled``设置的。
    """

    dome_light_upper_lower_strategy: Literal[0, 3, 4] | None = None
    """Selects how to sample the Dome Light. Default is 0.
    For more information, refer to the `documentation`_.

    .. _documentation: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html#dome-light

    Valid values are:

    * 0: **Image-Based Lighting (IBL)** - Most accurate even for high-frequency Dome Light textures.
      Can introduce sampling artifacts in real-time mode.
    * 3: **Limited Image-Based Lighting** - Only sampled for reflection and refraction. Fastest, but least
      accurate. Good for cases where the Dome Light contributes less than other light sources.
    * 4: **Approximated Image-Based Lighting** - Fast and artifacts-free sampling in real-time mode but only
      works well with a low-frequency texture (e.g., a sky with no sun disc where the sun is instead a separate
      Distant Light). Requires enabling Direct Lighting denoiser.

    This is set by the variable: ``/rtx/domeLight/upperLowerStrategy``.
    """
    """选择如何样品圆顶光。
    默认是0。
    更多信息请参阅`documentation`_。

    .. _documentation: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html#dome-light

    有效值为:

    * 0: **基于图像的照明 (IBL) ** - 即使是高频圆顶光纹理的最准确。 可以实时模式中引入样品采集文物。
    * 3: **限度基于图像的照明** - 仅用于反射和折射。 最快，但最不准确。 适用于 Dome Light 比其他光源贡献较少的情况。
    * 4: **近似基于图像的照明** - 实时模式的快速，无文物采样，但仅适用于低频纹理 (e.g.，没有太阳盘的天空，而太阳是独立的远光)。 需要启用直接照明表示器。

    这是由变量:``/rtx/domeLight/upperLowerStrategy``设置的。
    """

    carb_settings: dict[str, Any] | None = None
    """A general dictionary for users to supply all carb rendering settings with native names.

    The keys of the dictionary can be formatted like a carb setting, .kit file setting, or python variable.
    For instance, a key value pair can be:

    - ``/rtx/translucency/enabled: False`` (carb)
    - ``rtx.translucency.enabled: False`` (.kit)
    - ``rtx_translucency_enabled: False`` (python)
    """
    """一个一般的字典供用户提供所有碳水化合物渲染设置的本地名称。

    字典的键可以格式化为碳水化合物设置， .kit文件设置或python变量。
    例如，一个关键值对可能是:

    - ``/rtx/translucency/enabled: False``(碳水化合物)
    - ``rtx.translucency.enabled: False``(套件)
    - ``rtx_translucency_enabled: False``(鱼)
    """

    rendering_mode: Literal["performance", "balanced", "quality"] | None = None
    """The rendering mode.

    This behaves the same as the passing the CLI arg ``--rendering_mode`` to an executable script.
    """
    """渲染模式。

    这与将CLI arg ``--rendering_mode``传递到可执行脚本一样。
    """


@configclass
class SimulationCfg:
    """Configuration for simulation physics."""
    """仿真物理的配置。"""

    physics_prim_path: str = "/physicsScene"
    """The prim path where the USD PhysicsScene is created. Default is "/physicsScene"."""
    """在 USD PhysicsScene 创建的 prim 路径。
    默认是"/physicsScene"。
    """

    device: str = "cuda:0"
    """The device to run the simulation on. Default is ``"cuda:0"``.

    Valid options are:

    - ``"cpu"``: Use CPU.
    - ``"cuda"``: Use GPU, where the device ID is inferred from :class:`~isaaclab.app.AppLauncher`'s config.
    - ``"cuda:N"``: Use GPU, where N is the device ID. For example, "cuda:0".
    """
    """仿真的设备。
    默认是``"cuda:0"``。

    有效的选项是:

    - ``"cpu"``使用:CPU。
    - ``"cuda"``:使用GPU，该设备ID是从:class:`~isaaclab.app.AppLauncher`的配置中推断的。
    - ``"cuda:N"``:使用GPU，其中N是设备ID。 例如"，cuda:0"。
    """

    dt: float = 1.0 / 60.0
    """The physics simulation time-step (in seconds). Default is 0.0167 seconds."""
    """物理仿真时间步骤 (在秒钟)。
    默认为0.0167秒。
    """

    render_interval: int = 1
    """The number of physics simulation steps per rendering step. Default is 1."""
    """物理仿真步骤的数量，每一步的渲染。
    默认是1。
    """

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    """The gravity vector (in m/s^2). Default is (0.0, 0.0, -9.81).

    If set to (0.0, 0.0, 0.0), gravity is disabled.
    """
    """重力向量 (m/s^2)。
    默认是 (0.0，0.0， -9.81)。

    如果设置为 (0.0，0.0，0.0)，重力将被禁用。
    """

    enable_scene_query_support: bool = False
    """Enable/disable scene query support for collision shapes. Default is False.

    This flag allows performing collision queries (raycasts, sweeps, and overlaps) on actors and
    attached shapes in the scene. This is useful for implementing custom collision detection logic
    outside of the physics engine.

    If set to False, the physics engine does not create the scene query manager and the scene query
    functionality will not be available. However, this provides some performance speed-up.

    Note:
        This flag is overridden to True inside the :class:`SimulationContext` class when running the simulation
        with the GUI enabled. This is to allow certain GUI features to work properly.
    """
    """启用/禁用对碰撞形状的场景查询支持。
    默认是False。

    这种旗允许在场景中执行碰撞查询 (辐射，扫描和重叠)
    这对于在物理引擎之外实现定制碰撞检测逻辑是有用的。

    如果设置为False，物理引擎不会创建场景查询管理器，并且场景查询功能将无法使用。
    然而，这可以提高性能。

    说明：
        在运行仿真时，这个标志在 :class:`SimulationContext` 类内被过置为 True
        with the GUI enabled. This is to allow certain GUI features to work properly.
    """

    use_fabric: bool = True
    """Enable/disable reading of physics buffers directly. Default is True.

    When running the simulation, updates in the states in the scene is normally synchronized with USD.
    This leads to an overhead in reading the data and does not scale well with massive parallelization.
    This flag allows disabling the synchronization and reading the data directly from the physics buffers.

    It is recommended to set this flag to :obj:`True` when running the simulation with a large number
    of primitives in the scene.

    Note:
        When enabled, the GUI will not update the physics parameters in real-time. To enable real-time
        updates, please set this flag to :obj:`False`.

        When using GPU simulation, it is required to enable Fabric to visualize updates in the renderer.
        Transform updates are propagated to the renderer through Fabric. If Fabric is disabled with GPU simulation,
        the renderer will not be able to render any updates in the simulation, although simulation will still be
        running under the hood.
    """
    """直接启用/禁用物理缓冲的阅读。
    默认是True。

    在执行仿真时，场景状态的更新通常与USD同步。
    这导致数据阅读成本过高，并且与大规模并行并行并行并行并行并行并行并行并行并行并行并行。
    这种旗可以禁用同步和直接从物理缓冲器读取数据。

    建议在运行场景中大量原始的仿真时设置这个标志为:obj:`True`。

    说明：
        当启用时，GUI不会实时更新物理参数。
        为了启用实时更新，请设置这个标志为:obj:`False`。

        在使用GPU仿真时，需要使Fabric可视化渲染器中的更新。
        转换更新通过Fabric传递到渲染器。
        如果在GPU仿真中禁用Fabric，则渲染器将无法在仿真中提供任何更新，尽管仿真仍将在罩杯下运行。
    """

    physx: PhysxCfg = PhysxCfg()
    """PhysX solver settings. Default is PhysxCfg()."""
    """PhysX解决器设置
    默认是PhysxCfg()。
    """

    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg()
    """Default physics material settings for rigid bodies. Default is RigidBodyMaterialCfg().

    The physics engine defaults to this physics material for all the rigid body prims that do not have any
    physics material specified on them.

    The material is created at the path: ``{physics_prim_path}/defaultMaterial``.
    """
    """对于硬体的物理材料设置。
    默认是RigidBodyMaterialCfg()。

    物理发动机默认对所有硬体prims而不是任何物理材料的物理材料。

    材料是在``{physics_prim_path}/defaultMaterial``路径上创建的。
    """

    render: RenderCfg = RenderCfg()
    """Render settings. Default is RenderCfg()."""
    """提供设置。
    默认是RenderCfg()。
    """

    create_stage_in_memory: bool = False
    """If stage is first created in memory. Default is False.

    Creating the stage in memory can reduce start-up time.
    """
    """如果舞台首先在记忆中创建。
    默认是False。

    记忆中的舞台可以减少启动时间。
    """

    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
    """The logging level. Default is "WARNING"."""
    """伐木水平。
    默认是"WARNING"。
    """

    save_logs_to_file: bool = True
    """Save logs to a file. Default is True."""
    """保存日志到一个文件。
    默认是True。
    """

    log_dir: str | None = None
    """The directory to save the logs to. Default is None.

    If :attr:`save_logs_to_file` is True, the logs will be saved to the directory specified by :attr:`log_dir`.
    If None, the logs will be saved to the temp directory.
    """
    """保存记录的目录。
    默认是None。

    If :attr:`save_logs_to_file`是True，记录将被保存到:attr:`log_dir`指定的目录中。
    如果None，记录将被保存到临时目录中。
    """
