# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import inspect
import logging
import math
import warnings
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import MISSING
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch

import omni.kit.app
import omni.physx
from isaacsim.core.simulation_manager import SimulationManager

from isaaclab.managers import EventManager
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import attach_stage_to_usd_context, use_stage
from isaaclab.utils.noise import NoiseModel
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.timer import Timer
from isaaclab.utils.version import get_isaac_sim_version

from .common import VecEnvObs, VecEnvStepReturn
from .direct_rl_env_cfg import DirectRLEnvCfg
from .ui import ViewportCameraController
from .utils.spaces import sample_space, spec_to_gym_space

# import logger
logger = logging.getLogger(__name__)


class DirectRLEnv(gym.Env):
    """The superclass for the direct workflow to design environments.

    This class implements the core functionality for reinforcement learning (RL)
    environments. It is designed to be used with any RL library. The class is designed
    to be used with vectorized environments, i.e., the environment is expected to be run
    in parallel with multiple sub-environments.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """
    """对于设计环境的直接工作流，

    这类课程实现了强化学习 (RL) 环境的核心功能。
    它是用于任何RL库。
    该类是用于向量化环境，i.e.，环境预计将与多个子环境并行运行。

    虽然环境本身被实现为向量化环境，但我们并没有继承:class:`gym.vector.VectorEnv`。
    这主要是因为该类添加了各种方法 (等待和异步更新)，这些方法不需要。
    此外，每个RL库通常对向量化环境有自己的定义。
    因此，为了减少复杂性，我们直接使用:class:`gym.Env`在这里，

    说明：
        对于向量化环境，建议在第一次调用到:meth:`step`之前只****调用:meth:`reset`方法，环境创建后i.e.。
        在此之后，:meth:`step`函数处理终止子环境的重置。
        这是因为仿真器不支持在向量化环境中重置单个子环境。
    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    """环境是否是一个向量化环境。"""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
    }
    """Metadata for the environment."""
    """对环境的元数据。"""

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.

        Raises:
            RuntimeError: If a simulation context already exists. The environment must always create one
                since it configures the simulation context and controls the simulation.
        """
        """初始化环境。

        参数：
            cfg: 环境的配置对象。
            render_mode: 环境的渲染模式。
                         默认为 None，类似于``"human"``。

        异常：
            RuntimeError: 如果仿真上下文已经存在。
                          环境必须始终自行创建仿真上下文，因为环境需要配置并控制该仿真上下文。
        """
        # check that the config is valid
        cfg.validate()
        # store inputs to class
        self.cfg = cfg
        # store the render mode
        self.render_mode = render_mode
        # initialize internal variables
        self._is_closed = False

        # set the seed for the environment
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            logger.warning("Seed not set for the environment. The environment creation may not be deterministic.")

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")

        # make sure torch is running on the correct device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

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
                f"({self.cfg.decimation}). Multiple render calls will happen for each environment step."
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)

        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            # set the stage context for scene creation steps which use the stage
            with use_stage(self.sim.get_initial_stage()):
                self.scene = InteractiveScene(self.cfg.scene)
                self._setup_scene()
                attach_stage_to_usd_context()
        print("[INFO]: Scene manager: ", self.scene)

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI functionalities even for
        # non-rendering modes.
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        else:
            self.viewport_camera_controller = None

        # create event manager
        # note: this is needed here (rather than after simulation play) to allow USD-related randomization events
        #   that must happen before the simulation starts. Example: randomizing mesh scale
        if self.cfg.events:
            self.event_manager = EventManager(self.cfg.events, self)

            # apply USD-related randomization events
            if "prestartup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="prestartup")

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                # since the reset can trigger callbacks which use the stage,
                # we need to set the stage context here
                with use_stage(self.sim.get_initial_stage()):
                    self.sim.reset()
                # update scene to pre populate data buffers for assets and sensors.
                # this is needed for the observation manager to get valid tensors for initialization.
                # this shouldn't cause an issue since later on, users do a reset over all the environments
                # so the lazy buffers would be reset.
                self.scene.update(dt=self.physics_dt)

        # check if debug visualization is has been implemented by the environment
        source_code = inspect.getsource(self._set_debug_vis_impl)
        self.has_debug_vis_implementation = "NotImplementedError" not in source_code
        self._debug_vis_handle = None

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            # if no window, then we don't need to store the window
            self._window = None

        # allocate dictionary to store metrics
        self.extras = {}

        # initialize data and constants
        # -- counter for simulation steps
        self._sim_step_counter = 0
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- init buffers
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # setup noise cfg for adding action and observation noise
        if self.cfg.action_noise_model:
            self._action_noise_model: NoiseModel = self.cfg.action_noise_model.class_type(
                self.cfg.action_noise_model, num_envs=self.num_envs, device=self.device
            )
        if self.cfg.observation_noise_model:
            self._observation_noise_model: NoiseModel = self.cfg.observation_noise_model.class_type(
                self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
            )

        # perform events at the start of the simulation
        if self.cfg.events:
            # we print it here to make the logging consistent
            print("[INFO] Event Manager: ", self.event_manager)

            if "startup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="startup")

        # set the framerate of the gym video recorder wrapper so that the playback speed of the produced
        # video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        # show deprecation message for rerender_on_reset
        if self.cfg.rerender_on_reset:
            msg = (
                "\033[93m\033[1m[DEPRECATION WARNING] DirectRLEnvCfg.rerender_on_reset is deprecated. Use"
                " DirectRLEnvCfg.num_rerenders_on_reset instead.\033[0m"
            )
            warnings.warn(
                msg,
                FutureWarning,
                stacklevel=2,
            )
            if self.cfg.num_rerenders_on_reset == 0:
                self.cfg.num_rerenders_on_reset = 1

        # print the environment information
        print("[INFO]: Completed setting up the environment...")

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

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        """在几秒钟内最大的回合长度。"""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self):
        """The maximum episode length in steps adjusted from s."""
        """从s调整的步骤中最大回合长度。"""
        return math.ceil(self.max_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))

    """
    Operations.
    """
    """操作。
    """

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """Resets all the environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset all the environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        """设置所有环境并返回观测。

        这个函数叫做:meth:`_reset_idx`函数，重置所有环境。
        然而，在启动过程中发生的某些操作，如程序地形生成，不会重复。

        参数：
            seed: 种子用于随机化。
                  默认为None，在这种情况下，种子没有设置。
            options: 具体说明环境如何重置的额外信息。
                     默认为 None。

                说明：
                    这种参数用于与体育馆环境定义的兼容性。

        返回：
            一个包含观测和额外的图布。
        """
        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()

        # return observations
        return self._get_observations(), self.extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        """执行一个时间步骤的环境动态。

        环境在一个固定的时间步骤上向前迈进，而物理仿真在一个较低的时间步骤上被除。
        这样可以保证仿真的稳定性。
        这些两个时间步骤可以使用:attr:`DirectRLEnvCfg.decimation` (每个环境步骤的仿真步骤数量) 和:attr:`DirectRLEnvCfg.sim.physics_dt`
        (物理时间步骤) 独立配置。
        根据这些参数，环境时间步骤被计算为两者的乘积。

        这个函数执行以下步骤:

        1. 在进入物理之前，先处理操作。
        2. 运行操作到仿真器上，并通过物理进行 step化。
        3. 计算奖励和完成的信号。
        4. 设置已结束或达到最高回合长度的环境。
        5. 如果它们已启用，应应用间隔事件。
        6. 计算观测。

        参数：
            action: 应对环境的动作。
                    形状是 (num_envs，action_dim)。

        返回：
            包含观测，奖励，重置 (终止和缩短) 和额外的元组。
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    @staticmethod
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
        return configure_seed(seed)

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return a numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        """没有通过物理进行渲染。

        根据"公约"，如果模式是:

        - **人**:将其放到当前的显示屏上，没有任何东西返回.通常用于人类消费。
        - **rgb_array**:以形状 (x， y， 3) 的 numpy.ndarray 返回，表示RGB值为x-by-y像素图像，适合转换为视频。

        参数：
            recompute: 如果仿真器已经将场景呈现出来。
                       默认为 False。

        返回：
            如果模式是"rgb_array"，则将呈现成一个 numpy 阵列。
            否则，返回None。

        异常：
            RuntimeError: 如果模式设置为"rgb_data"，并没有支持仿真渲染模式。
                          在这种情况下，仿真渲染模式必须设置为``RenderMode.PARTIAL_RENDERING``或``RenderMode.FULL_RENDERING``。
            NotImplementedError: 如果指定未支持的渲染模式。
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        """Cleanup for the environment."""
        """清理环境。"""
        if not self._is_closed:
            # close entities related to the environment
            # note: this is order-sensitive to avoid any dangling references
            if self.cfg.events:
                del self.event_manager
            del self.scene
            if self.viewport_camera_controller is not None:
                del self.viewport_camera_controller

            # clear callbacks and instance
            if get_isaac_sim_version().major >= 5:
                if self.cfg.sim.create_stage_in_memory:
                    # detach physx stage
                    omni.physx.get_physx_simulation_interface().detach_stage()
                    self.sim.stop()
                    self.sim.clear()

            self.sim.clear_all_callbacks()
            self.sim.clear_instance()

            # destroy the window
            if self._window is not None:
                self._window = None
            # update closing status
            self._is_closed = True

    """
    Operations - Debug Visualization.
    """
    """操作 - 调试可视化。
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Toggles the environment debug visualization.

        Args:
            debug_vis: Whether to visualize the environment debug visualization.

        Returns:
            Whether the debug visualization was successfully set. False if the environment
            does not support debug visualization.
        """
        """关闭环境调试可视化。

        参数：
            debug_vis: 是否可视化环境调试可视化。

        返回：
            设置错误可视化是否成功。
            False如果环境不支持调试可视化。
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

    """
    Helper functions.
    """
    """辅助函数。
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        """为Gym环境设置事件和观测空间。"""
        # show deprecation message and overwrite configuration
        if self.cfg.num_actions is not None:
            logger.warning("DirectRLEnvCfg.num_actions is deprecated. Use DirectRLEnvCfg.action_space instead.")
            if isinstance(self.cfg.action_space, type(MISSING)):
                self.cfg.action_space = self.cfg.num_actions
        if self.cfg.num_observations is not None:
            logger.warning(
                "DirectRLEnvCfg.num_observations is deprecated. Use DirectRLEnvCfg.observation_space instead."
            )
            if isinstance(self.cfg.observation_space, type(MISSING)):
                self.cfg.observation_space = self.cfg.num_observations
        if self.cfg.num_states is not None:
            logger.warning("DirectRLEnvCfg.num_states is deprecated. Use DirectRLEnvCfg.state_space instead.")
            if isinstance(self.cfg.state_space, type(MISSING)):
                self.cfg.state_space = self.cfg.num_states

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = spec_to_gym_space(self.cfg.observation_space)
        self.single_action_space = spec_to_gym_space(self.cfg.action_space)

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # optional state space for asymmetric actor-critic architectures
        self.state_space = None
        if self.cfg.state_space:
            self.single_observation_space["critic"] = spec_to_gym_space(self.cfg.state_space)
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)

        # instantiate actions (needed for tasks for which the observations computation is dependent on the actions)
        self.actions = sample_space(self.single_action_space, self.sim.device, batch_size=self.num_envs, fill_value=0)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        """根据指定索引重置环境。

        参数：
            env_ids: 必须重置的环境ID列表
        """
        self.scene.reset(env_ids)

        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # reset noise models
        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

    """
    Implementation-specific functions.
    """
    """具体执行功能
    """

    def _setup_scene(self):
        """Setup the scene for the environment.

        This function is responsible for creating the scene objects and setting up the scene for the environment.
        The scene creation can happen through :class:`isaaclab.scene.InteractiveSceneCfg` or through
        directly creating the scene objects and registering them with the scene manager.

        We leave the implementation of this function to the derived classes. If the environment does not require
        any explicit scene setup, the function can be left empty.
        """
        """为环境设置场景。

        这种功能负责创建场景对象和为环境设置场景。
        场景创建可以通过:class:`isaaclab.scene.InteractiveSceneCfg`或通过直接创建场景对象并将它们注册到场景管理器。

        我们把这个函数的实现留给了衍生类。
        如果环境不需要任何明确的场景设置，则可以将函数留空。
        """
        pass

    @abstractmethod
    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-process actions before stepping through the physics.

        This function is responsible for pre-processing the actions before stepping through the physics.
        It is called before the physics stepping (which is decimated).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, action_dim).
        """
        """在进入物理之前，我们做了前处理的操作。

        这项功能负责在通过物理之前预先处理操作。
        它被称为物理步骤前 (被除)。

        参数：
            actions: 应对环境的动作。
                     形状是 (num_envs，action_dim)。
        """
        raise NotImplementedError(f"Please implement the '_pre_physics_step' method for {self.__class__.__name__}.")

    @abstractmethod
    def _apply_action(self):
        """Apply actions to the simulator.

        This function is responsible for applying the actions to the simulator. It is called at each
        physics time-step.
        """
        """在仿真器上执行操作。

        这项功能负责将操作应用于仿真器。
        在每一个物理时间步骤中，
        """
        raise NotImplementedError(f"Please implement the '_apply_action' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_observations(self) -> VecEnvObs:
        """Compute and return the observations for the environment.

        Returns:
            The observations for the environment.
        """
        """计算和返回对环境的观测。

        返回：
            对环境的观测。
        """
        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")

    def _get_states(self) -> VecEnvObs | None:
        """Compute and return the states for the environment.

        The state-space is used for asymmetric actor-critic architectures. It is configured
        using the :attr:`DirectRLEnvCfg.state_space` parameter.

        Returns:
            The states for the environment. If the environment does not have a state-space, the function
            returns a None.
        """
        """计算和返回环境状态。

        国家空间用于不对称的演员-批评架构。
        它是使用:attr:`DirectRLEnvCfg.state_space`参数配置的。

        返回：
            为了环境。
            如果环境没有状态空间，函数返回None。
        """
        return None  # noqa: R501

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor:
        """Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).
        """
        """计算和回报环境的回报。

        返回：
            环境的回报。
            形状是 (num_envs，)。
        """
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Shape of individual tensors is (num_envs,).
        """
        """计算和返回已完成的环境标志。

        返回：
            包含终止和休息的已完成标志。
            单个子的形状是 (num_envs，)。
        """
        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")

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
