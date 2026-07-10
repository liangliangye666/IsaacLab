# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import logging
import warnings
from collections.abc import Sequence
from typing import Any

import torch

import omni.physx
from isaacsim.core.simulation_manager import SimulationManager

from isaaclab.managers import ActionManager, EventManager, ObservationManager, RecorderManager
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import attach_stage_to_usd_context, use_stage
from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.timer import Timer
from isaaclab.utils.version import get_isaac_sim_version

from .common import VecEnvObs
from .manager_based_env_cfg import ManagerBasedEnvCfg
from .ui import ViewportCameraController
from .utils.io_descriptors import export_articulations_data, export_scene_data

# import logger
logger = logging.getLogger(__name__)


class ManagerBasedEnv:
    """The base environment encapsulates the simulation scene and the environment managers for
    the manager-based workflow.

    While a simulation scene or world comprises of different components such as the robots, objects,
    and sensors (cameras, lidars, etc.), the environment is a higher level abstraction
    that provides an interface for interacting with the simulation. The environment is comprised of
    the following components:

    * **Scene**: The scene manager that creates and manages the virtual world in which the robot operates.
      This includes defining the robot, static and dynamic objects, sensors, etc.
    * **Observation Manager**: The observation manager that generates observations from the current simulation
      state and the data gathered from the sensors. These observations may include privileged information
      that is not available to the robot in the real world. Additionally, user-defined terms can be added
      to process the observations and generate custom observations. For example, using a network to embed
      high-dimensional observations into a lower-dimensional space.
    * **Action Manager**: The action manager that processes the raw actions sent to the environment and
      converts them to low-level commands that are sent to the simulation. It can be configured to accept
      raw actions at different levels of abstraction. For example, in case of a robotic arm, the raw actions
      can be joint torques, joint positions, or end-effector poses. Similarly for a mobile base, it can be
      the joint torques, or the desired velocity of the floating base.
    * **Event Manager**: The event manager orchestrates operations triggered based on simulation events.
      This includes resetting the scene to a default state, applying random pushes to the robot at different intervals
      of time, or randomizing properties such as mass and friction coefficients. This is useful for training
      and evaluating the robot in a variety of scenarios.
    * **Recorder Manager**: The recorder manager that handles recording data produced during different steps
      in the simulation. This includes recording in the beginning and end of a reset and a step. The recorded data
      is distinguished per episode, per environment and can be exported through a dataset file handler to a file.

    The environment provides a unified interface for interacting with the simulation. However, it does not
    include task-specific quantities such as the reward function, or the termination conditions. These
    quantities are often specific to defining Markov Decision Processes (MDPs) while the base environment
    is agnostic to the MDP definition.

    The environment steps forward in time at a fixed time-step. The physics simulation is decimated at a
    lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
    independently using the :attr:`ManagerBasedEnvCfg.decimation` (number of simulation steps per environment step)
    and the :attr:`ManagerBasedEnvCfg.sim.dt` (physics time-step) parameters. Based on these parameters, the
    environment time-step is computed as the product of the two. The two time-steps can be obtained by
    querying the :attr:`physics_dt` and the :attr:`step_dt` properties respectively.
    """
    """基于管理器工作流的基础环境，封装仿真场景以及各类环境管理器。

    仿真场景（或仿真世界）由机器人、物体和传感器（摄像头、激光雷达等）组成；
    环境则是更高层的抽象，为外部程序提供统一的仿真交互接口。环境包含以下组件：

    * **场景**：创建并管理机器人运行的虚拟世界，包括机器人、静态/动态物体和传感器等。
    * **观测管理器**：根据当前仿真状态和传感器数据生成观测。观测可以包含真实机器人无法直接获取的
      特权信息，也可以通过用户自定义项进一步处理，例如利用神经网络将高维观测编码到低维空间。
    * **动作管理器**：处理环境接收的原始动作，并将其转换为发送给仿真器的低层命令。动作可以采用不同
      抽象层级，例如机械臂的关节力矩、关节位置或末端执行器位姿，以及移动底盘的关节力矩或基座目标速度。
    * **事件管理器**：编排由仿真事件触发的操作，例如将场景重置到默认状态、按时间间隔向机器人施加
      随机扰动，或者随机化质量、摩擦系数等属性，以支持多样化场景下的训练和评估。
    * **记录器管理器**：记录仿真 reset 和 step 前后的数据。记录数据按环境和回合组织，并可通过
      dataset file handler 导出到文件。

    环境本身不包含奖励函数、终止条件等任务特定量。这些量属于 Markov Decision Process（MDP）的定义，
    而本基础环境与具体 MDP 无关。

    环境按固定的环境时间步推进，而底层物理仿真使用更小的物理时间步执行多次，以保证仿真稳定性。
    两个时间尺度分别由 :attr:`ManagerBasedEnvCfg.decimation`（每个环境步包含的物理仿真步数）和
    :attr:`ManagerBasedEnvCfg.sim.dt`（物理时间步）配置。环境时间步等于二者的乘积，可分别通过
    :attr:`physics_dt` 和 :attr:`step_dt` 属性查询物理时间步与环境时间步。
    """

    '''
    仿真环境的完整初始化流程：把一个 ManagerBasedEnvCfg 变成可运行的 IsaacLab 环境。
    '''
    def __init__(self, cfg: ManagerBasedEnvCfg):
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment.

        Raises:
            RuntimeError: If a simulation context already exists. The environment must always create one
                since it configures the simulation context and controls the simulation.
        """
        """初始化环境。

        参数：
            cfg: 环境的配置对象。

        异常：
            RuntimeError: 如果仿真上下文已经存在。
                          环境必须始终自行创建仿真上下文，因为环境需要配置并控制该仿真上下文。
        """
        '''
        ManagerBasedEnv.__init__(cfg)
            │
            ├── ① cfg.validate() + 存储配置
            ├── ② 设置随机种子
            ├── ③ 创建/复用 SimulationContext（物理引擎）
            ├── ④ 创建 InteractiveScene（场景 + 实体）
            ├── ⑤ 设置视口相机
            ├── ⑥ 创建 EventManager + 执行 prestartup 事件
            ├── ⑦ 启动仿真 + scene.update()
            ├── ⑧ 调用 load_managers()（多态！跳转到子类版本）
            ├── ⑨ 创建 UI 窗口 + 可视化器
            ├── ⑩ 导出 IO 描述符（如果配置要求）
            └── ⑪ 废弃警告（rerender_on_reset）
        '''
        # check that the config is valid
        cfg.validate()      # 确保所有 MISSING 字段都已填写
        '''
        @configclass 把它变成 IsaacLab 风格的 dataclass 配置类。字段里出现 MISSING 表示子类或具体任务必须填，比如：
            decimation: int = MISSING
            scene: InteractiveSceneCfg = MISSING
            observations: object = MISSING
            actions: object = MISSING
        如果任务配置没填这些，cfg.validate() 会报错。
        '''

        # store inputs to class
        self.cfg = cfg
        # initialize internal variables
        self._is_closed = False

        # set the seed for the environment
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)    # 设置 Python/numpy/torch 全局种子
        else:
            logger.warning("Seed not set for the environment. The environment creation may not be deterministic.")

        '''
        两条路径：终端运行时自己创建物理引擎上下文；在 Isaac Sim 扩展中运行时复用已有的。
        '''
        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            # the type-annotation is required to avoid a type-checking error
            # since it gets confused with Isaac Sim's SimulationContext class
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            # simulation context should only be created before the environment
            # when in extension mode
            if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
                raise RuntimeError("Simulation context already exists. Cannot create a new one.")
            self.sim: SimulationContext = SimulationContext.instance()

        # make sure torch is running on the correct device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)  # 保证 PyTorch 当前 device 和仿真配置一致。

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) is smaller than the decimation "
                f"({self.cfg.decimation}). Multiple render calls will happen for each environment step. "
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)
            '''
            如果 render_interval < decimation，说明一个环境 step 内可能 render 多次，所以给 warning。
            '''

        # counter for simulation steps  记录底层物理步数
        self._sim_step_counter = 0

        # allocate dictionary to store metrics  用来放日志、调试信息等
        self.extras = {}

        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            # set the stage context for scene creation steps which use the stage
            with use_stage(self.sim.get_initial_stage()):
                self.scene = InteractiveScene(self.cfg.scene)
                attach_stage_to_usd_context()
        print("[INFO]: Scene manager: ", self.scene)

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI functionalities even for
        # non-rendering modes.
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:   # 只在有渲染时创建（无头模式不需要）。
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        else:
            self.viewport_camera_controller = None

        # create event manager
        # note: this is needed here (rather than after simulation play) to allow USD-related randomization events
        #   that must happen before the simulation starts. Example: randomizing mesh scale
        self.event_manager = EventManager(self.cfg.events, self)

        # apply USD-related randomization events
        if "prestartup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="prestartup")
            '''
            在仿真启动前执行 prestartup 事件——这是 USD 级别随机化的时机（如随机改变 mesh 缩放、替换材质）。
            这些操作必须在 PhysX 加载场景之前完成。
            '''

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:  # 终端模式分支（不是 extension 模式）:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                # since the reset can trigger callbacks which use the stage,
                # we need to set the stage context here
                with use_stage(self.sim.get_initial_stage()):
                    self.sim.reset()    # 播放仿真 → 触发 PLAY 事件
                    '''
                    sim.reset() 会激活 PhysX/Tensor API 句柄。
                    很多 manager 初始化时需要读取机器人关节、刚体、传感器 tensor，所以必须等仿真 reset 后再创建。
                    '''
                # update scene to pre populate data buffers for assets and sensors.
                # this is needed for the observation manager to get valid tensors for initialization.
                # this shouldn't cause an issue since later on, users do a reset over all the environments
                # so the lazy buffers would be reset.
                self.scene.update(dt=self.physics_dt)   # 从 PhysX 读回初始状态
            # add timeline event to load managers
            self.load_managers()    # 多态调用 → ManagerBasedRLEnv.load_managers()

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            # setup live visualizers
            self.setup_manager_visualizers()    # 创建各 Manager 的实时数据可视化
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")  # 创建主窗口
            '''
            在 Manager 全部初始化之后（此时知道有哪些 term），搭建实时可视化面板。
            '''
        else:
            # if no window, then we don't need to store the window
            self._window = None

        # initialize observation buffers
        self.obs_buf = {}   # 观测缓冲区占位

        # export IO descriptors if requested
        if self.cfg.export_io_descriptors:
            self.export_IO_descriptors()    # 如果配置要求导出 IO 规格

        # show deprecation message for rerender_on_reset
        if self.cfg.rerender_on_reset:  # 废弃兼容
            msg = (
                "\033[93m\033[1m[DEPRECATION WARNING] ManagerBasedEnvCfg.rerender_on_reset is deprecated. Use"
                " ManagerBasedEnvCfg.num_rerenders_on_reset instead.\033[0m"
            )
            warnings.warn(
                msg,
                FutureWarning,
                stacklevel=2,
            )
            if self.cfg.num_rerenders_on_reset == 0:
                self.cfg.num_rerenders_on_reset = 1 # 旧字段 → 新字段
                '''
                rerender_on_reset 已废弃。
                现在推荐用 num_rerenders_on_reset。
                这里是向后兼容：旧配置写 rerender_on_reset=True 时，自动等价为 reset 后多 render 1 次。
                '''
    '''
    两种启动模式的路径差异
        终端模式 (python train.py):
            ③ 创建 SimulationContext
            ④ 创建 InteractiveScene
            ⑥ EventManager + prestartup
            ⑦ sim.reset() + scene.update() + load_managers()
            ⑨ UI + 可视化

        扩展模式 (Isaac Sim GUI):
            ③ 复用已有 SimulationContext
            ④ 创建 InteractiveScene
            ⑥ EventManager + prestartup
            ...
            (sim.reset() 和 load_managers() 分开执行，不在 __init__ 中)
    '''

    def __del__(self):
        """Cleanup for the environment."""
        """清理环境。"""
        self.close()

    """
    Properties.
    """
    """属性。
    """

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        """正在运行的环境实例数量。"""
        return self.scene.num_envs

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which the simulation is happening.
        """
        """物理时间步（单位：s）。

        这是仿真采用的最小时间步。
        """
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        """环境步进时间步（单位：s）。

        这是环境向前推进一个步长时使用的时间步。
        """
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def device(self):
        """The device on which the environment is running."""
        """环境运行所在的设备。"""
        return self.sim.device

    '''
    聚合所有输入输出规格
    '''
    @property
    def get_IO_descriptors(self):
        """Get the IO descriptors for the environment.

        Returns:
            A dictionary with keys as the group names and values as the IO descriptors.
        """
        """获取环境的 IO 描述符。

        返回：
            一个字典，键为组名、值为 IO 描述符。
        """
        return {
            "observations": self.observation_manager.get_IO_descriptors,
            "actions": self.action_manager.get_IO_descriptors,
            "articulations": export_articulations_data(self),
            "scene": export_scene_data(self),
        }
    '''
    键	            内容
    observations	每个观测 term 的维度、类型、缩放
    actions	        每个动作 term 的维度、范围
    articulations	机器人的关节名、类型、限位
    scene	        场景结构（地面、物体的 prim_path）
    '''

    '''
    将环境规格导出为 YAML 文件
    这是之前讲过的所有 IO 描述符（观测、动作、关节体、场景）的落地出口——把聚合好的规格字典写入磁盘，供模型部署工具读取。
    '''
    def export_IO_descriptors(self, output_dir: str | None = None):
        """Export the IO descriptors for the environment.

        Args:
            output_dir: The directory to export the IO descriptors to.
        """
        """导出环境的 IO 描述符。

        参数：
            output_dir: 导出IO描述符的目录
        """
        '''
        export_IO_descriptors(output_dir)
            │
            ├── ① 调 self.get_IO_descriptors → 聚合所有 IO 规格字典
            ├── ② 确定输出路径（优先 output_dir → log_dir/io_descriptors）
            └── ③ 用 yaml.safe_dump 写入 IO_descriptors.yaml
        '''
        import os

        import yaml

        IO_descriptors = self.get_IO_descriptors

        if output_dir is None:
            if self.cfg.log_dir is not None:
                output_dir = os.path.join(self.cfg.log_dir, "io_descriptors")
            else:
                raise ValueError(
                    "Output directory is not set. Please set the log directory using the `log_dir`"
                    " configuration or provide an explicit output_dir parameter."
                )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "IO_descriptors.yaml"), "w") as f:
            print(f"[INFO]: Exporting IO descriptors to {os.path.join(output_dir, 'IO_descriptors.yaml')}")
            yaml.safe_dump(IO_descriptors, f)

    """
    Operations - Setup.
    """
    """操作 - 初始化。
    """

    def load_managers(self):
        """Load the managers for the environment.

        This function is responsible for creating the various managers (action, observation,
        events, etc.) for the environment. Since the managers require access to physics handles,
        they can only be created after the simulator is reset (i.e. played for the first time).

        .. note::
            In case of standalone application (when running simulator from Python), the function is called
            automatically when the class is initialized.

            However, in case of extension mode, the user must call this function manually after the simulator
            is reset. This is because the simulator is only reset when the user calls
            :meth:`SimulationContext.reset_async` and it isn't possible to call async functions in the constructor.

        """
        """加载环境管理器。

        该函数负责为环境创建各种管理器，包括动作管理器、观测管理器和事件管理器等。
        由于这些管理器需要访问物理句柄，因此只能在仿真器完成首次 reset（即第一次开始运行）后创建。

        .. 说明::
            在独立运行应用中（即从 Python 启动仿真器时），该函数会在类初始化期间自动调用。

            但在 extension 模式下，用户必须在仿真器 reset 后手动调用该函数。
            这是因为只有用户调用 :meth:`SimulationContext.reset_async` 时仿真器才会 reset，而构造函数中无法调用 async 函数。
        """
        # prepare the managers
        # -- event manager (we print it here to make the logging consistent)
        print("[INFO] Event Manager: ", self.event_manager) #（打印信息，不创建——已在 __init__ 创建）
        # -- recorder manager
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)

        # perform events at the start of the simulation
        # in-case a child implementation creates other managers, the randomization should happen
        # when all the other managers are created
        if self.__class__ == ManagerBasedEnv and "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
            '''
            self.__class__ == ManagerBasedEnv 的精确判断
                这不是 isinstance，而是精确的 ==。为什么不用 isinstance？
                    判断方式	                            当 self 是 ManagerBasedRLEnv 时
                    isinstance(self, ManagerBasedEnv)	    True（子类也算）→ 会执行 startup
                    self.__class__ == ManagerBasedEnv	    False（只认基类本身）→ 不执行
                设计意图：ManagerBasedRLEnv.load_managers() 在最后一步（line 134）自己负责执行 startup：
                    # ManagerBasedRLEnv.load_managers():
                    super().load_managers()                   # → 基类版本（不执行 startup）
                    self.termination_manager = ...
                    self.reward_manager = ...
                    self.curriculum_manager = ...
                    self._configure_gym_env_spaces()
                    self.event_manager.apply(mode="startup")   # ← 这里执行！在 RL Manager 创建完毕后
                RL 子类需要等 所有 RL Manager 创建完毕后再执行 startup（因为 startup 事件可能依赖 commands/rewards/termination）。
                如果基类版本用 isinstance，RL 子类还在半路时就被触发了 startup——时机不对。

                用 == 限定了：只有直接实例化基类 ManagerBasedEnv 时（非 RL 场景，如遥操作）才在这里执行 startup。
                任何子类（如 ManagerBasedRLEnv）都自己负责 startup 时机——因为 RL 的 startup 需要等 Command/Reward/Termination Manager 全部就绪。
            '''

    '''
    实时数据面板的创建
        ManagerLiveVisualizer 就是训练时 GUI 中显示动作值、观测值实时曲线的那个面板。
    '''
    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""
        """为各管理器项创建实时可视化器。"""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
        }

    """
    Operations - MDP.
    """
    """操作 - MDP。
    """

    '''
    Gymnasium 标准的公开重置接口
        这是 gym.Env 要求实现的 reset() 方法。
        ManagerBasedRLEnv 不覆盖 reset()，只覆盖内部的 _reset_idx()——这是模板方法模式的又一次应用。
    '''
    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VecEnvObs, dict]:
        """Resets the specified environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset the specified environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        """重置指定环境并返回观测。

        该函数调用 :meth:`_reset_idx` 重置指定环境，但不会重复执行初始化阶段完成的操作，
        例如程序化地形生成。

        参数：
            seed: 随机化使用的随机种子。默认为 None，表示不重新设置随机种子。
            env_ids: 需要重置的环境 ID。默认为 None，表示重置全部环境。
            options: 指定环境重置方式的附加信息。默认为 None。

                说明：
                    该参数用于兼容 Gymnasium 的环境接口定义。

        返回：
            包含观测和附加信息的元组。
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # trigger recorder terms for pre-reset calls
        self.recorder_manager.record_pre_reset(env_ids)

        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        self._reset_idx(env_ids)
        '''
        基类 ManagerBasedEnv._reset_idx() 只做场景重置 + EventManager reset。
        RL 子类覆盖 _reset_idx()，在里面做完整的 RL 重置流程：
        '''

        '''
        重置后的物理同步
        '''
        # update articulation kinematics
        self.scene.write_data_to_sim()    # 把新状态推送到 PhysX
        self.sim.forward()                # PhysX 执行一个物理步
        '''
        重置时修改了关节角度、根位姿，这些写到了 Python 侧缓冲区。
        write_data_to_sim() 推到 PhysX，sim.forward() 让 PhysX 算一步（如检测碰撞），确保重置后的状态在物理上合法。
        '''

        '''
        传感器渲染
        '''
        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()
        '''
        重置后环境状态变了，但相机/激光雷达的渲染画面还是旧的。
        额外渲染几帧让传感器数据反映新状态。
        num_rerenders_on_reset 默认为 0（不渲染，省性能）。
        '''

        # trigger recorder terms for post-reset calls
        self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        '''
        纹理等待
        '''
        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()
                '''
                纹理加载是异步的。如果配置要求等待，就循环渲染直到加载完毕——确保传感器拍到的画面不是灰模。
                '''

        # return observations
        return self.obs_buf, self.extras

    '''
    Checkpoint 恢复式重置
        和 reset() 不同——reset() 通过随机化事件生成初始状态，reset_to() 直接把环境恢复到指定的精确状态（checkpoint / demo 回放）。
    '''
    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None,
        seed: int | None = None,
        is_relative: bool = False,
    ):
        """Resets specified environments to provided states.

        This function resets the environments to the provided states. The state is a dictionary
        containing the state of the scene entities. Please refer to :meth:`InteractiveScene.get_state`
        for the format.

        The function is different from the :meth:`reset` function as it resets the environments to specific states,
        instead of using the randomization events for resetting the environments.

        Args:
            state: The state to reset the specified environments to. Please refer to
                :meth:`InteractiveScene.get_state` for the format.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.
        """
        """将指定环境重置到给定状态。

        ``state`` 是包含场景实体状态的字典，其格式请参阅 :meth:`InteractiveScene.get_state`。
        与 :meth:`reset` 不同，该函数直接恢复指定状态，而不是通过随机化事件生成重置状态。

        参数：
            state: 指定环境要恢复到的状态。格式请参阅 :meth:`InteractiveScene.get_state`。
            env_ids: 需要重置的环境 ID。默认为 None，表示重置全部环境。
            seed: 随机化使用的随机种子。默认为 None，表示不重新设置随机种子。
            is_relative: 若为 True，则将给定状态视为相对于各环境原点的状态。默认为 False。
        """
        # reset all envs in the scene if env_ids is None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # trigger recorder terms for pre-reset calls
        self.recorder_manager.record_pre_reset(env_ids)

        # set the seed
        if seed is not None:
            self.seed(seed)

        self._reset_idx(env_ids)

        # set the state
        self.scene.reset_to(state, env_ids, is_relative=is_relative)

        # update articulation kinematics
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        # trigger recorder terms for post-reset calls
        self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations
        return self.obs_buf, self.extras

    '''
    环境一帧的完整物理循环
        这是基类版本的 step()，只做物理仿真和观测计算，不涉及奖励/终止/命令（那由 ManagerBasedRLEnv.step() 叠加）。
        这里定义了物理循环的核心框架。
    '''
    def step(self, action: torch.Tensor) -> tuple[VecEnvObs, dict]:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is
        decimated at a lower time-step. This is to ensure that the simulation is stable. These two
        time-steps can be configured independently using the :attr:`ManagerBasedEnvCfg.decimation` (number of
        simulation steps per environment step) and the :attr:`ManagerBasedEnvCfg.sim.dt` (physics time-step).
        Based on these parameters, the environment time-step is computed as the product of the two.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations and extras.
        """
        """执行一个环境时间步的动力学更新。

        环境以固定的环境时间步推进，底层物理仿真则以更小的时间步执行 ``decimation`` 次。
        :attr:`ManagerBasedEnvCfg.decimation` 指定每个环境步包含的物理仿真步数，
        :attr:`ManagerBasedEnvCfg.sim.dt` 指定物理时间步；环境时间步为二者的乘积。

        参数：
            action: 施加到环境的动作，形状为 ``(num_envs, action_dim)``。

        返回：
            包含观测和附加信息的元组。
        """
        '''
        执行流程
            step(action)
                │
                ├── ① process_action(action)      动作预处理
                ├── ② record_pre_step             录制预回调
                │
                ├── ③ 物理循环 (× decimation 次)
                │     ├── apply_action()           写目标值
                │     ├── write_data_to_sim()      推到 PhysX
                │     ├── sim.step()               PhysX 一步
                │     ├── (if 渲染间隔) render()
                │     └── scene.update()           读回新状态
                │
                ├── ④ interval 事件
                ├── ⑤ compute observations
                └── ⑥ return (obs_buf, extras)
        '''

        # process actions
        self.action_manager.process_action(action.to(self.device))  # 处理动作

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        '''
        如果开了 GUI，或者场景里有 RTX 相机/传感器，就可能需要在物理循环中 render。这里提前算一次，避免每个物理步重复检查。
        '''

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()  # ① 把力矩/位置目标写到 articulation buffer
            # set actions into simulator
            self.scene.write_data_to_sim()      # ② 推到 PhysX C++ API
            # simulate
            self.sim.step(render=False)         # ③ PhysX 积分一个物理步长
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()               # ④ 渲染（如果 GUI 开或 RTX 传感器用）
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)   # ⑤ 从 PhysX 读回关节角、速度等

        # post-step: step interval event
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
            '''
            物理循环结束后，处理按时间间隔触发的事件，比如随机推一下机器人、周期性扰动等。
            这里传的是 step_dt，因为 interval 事件按环境步节奏更新倒计时。
            '''

        # -- compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)
        '''
        计算所有 observation group。update_history=True 表示如果某些 observation term 配了 history_length，这里会把当前观测追加进历史缓冲区。
        '''

        self.recorder_manager.record_post_step()

        # return observations and extras
        return self.obs_buf, self.extras

    @staticmethod   # 调用时不依赖实例——ManagerBasedEnv.seed(42) 和 env.seed(42) 效果一样。这里的 seed 方法是工具函数，和 self 无关。
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        """为环境提供种子。

        参数：
            seed: 种子是随机发电机。
                  设置为 -1。

        返回：
            种子用于随机发电机。
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return configure_seed(seed) # 设置 torch、numpy、Python random、warp 四个随机数生成器的全局种子，确保训练可复现。

    def close(self):
        """Cleanup for the environment."""
        """清理环境。"""
        if not self._is_closed:
            # destructor is order-sensitive
            del self.viewport_camera_controller
            del self.action_manager
            del self.observation_manager
            del self.event_manager
            del self.recorder_manager
            del self.scene
            '''
            顺序很重要——Manager 内部持有 scene 的引用（如 ActionTerm._asset = scene["robot"]）。
            如果先删 scene，再删 Manager 时会访问已释放的内存。
            后创建的先删，先创建的后删——栈式清理。
            '''

            # clear callbacks and instance
            if get_isaac_sim_version().major >= 5:
                if self.cfg.sim.create_stage_in_memory:
                    # detach physx stage
                    omni.physx.get_physx_simulation_interface().detach_stage()
                    self.sim.stop()
                    self.sim.clear()
                    '''
                    当场景完全在内存中运行时，需要手动从 PhysX 卸下 stage，然后停止仿真并清空内存——否则显存泄漏。
                    '''

            self.sim.clear_all_callbacks()     # 清理 ManagerBase 中注册的 PLAY 事件回调等
            self.sim.clear_instance()          # 销毁 SimulationContext 单例

            # destroy the window
            if self._window is not None:
                self._window = None
            # update closing status
            self._is_closed = True

    """
    Helper functions.
    """
    """辅助函数。
    """

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        """根据指定索引重置环境。

        参数：
            env_ids: 必须重置的环境ID列表
        """
        '''
        三步重置
            _reset_idx(env_ids)
                │
                ├── ① scene.reset(env_ids)          物理状态恢复默认
                ├── ② event_manager.apply("reset")  触发随机化事件（如随机关节角度）
                └── ③ 4 个 Manager 逐个 reset + 收集日志
        '''

        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)

        # apply events such as randomization for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation  # 物理步转环境步
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)  # ① 观测：清空历史缓冲区
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)       # ② 动作：清零 _action / _prev_action
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)        # ③ 事件：重置 interval 倒计时
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)     # ④ 录制：导出 episode 数据
        self.extras["log"].update(info)
        '''
        注释明确写了 "order-sensitive"——观测必须在动作之前重置，因为观测可能引用 action_manager.prev_action。
        先清空历史缓冲区，再清除动作值，最后重置事件倒计时。
        '''
