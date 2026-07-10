# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch

from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.ui.widgets import ManagerLiveVisualizer

from .common import VecEnvStepReturn
from .manager_based_env import ManagerBasedEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


class ManagerBasedRLEnv(ManagerBasedEnv, gym.Env):
    """The superclass for the manager-based workflow reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

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
    """基于管理器工作流的强化学习环境基类。

    该类继承 :class:`ManagerBasedEnv`，实现强化学习环境所需的核心功能，并可与不同的 RL 库配合使用。
    环境采用向量化实现，即同时并行运行多个子环境，子环境数量由 ``num_envs`` 指定。

    环境返回的每组观测都包含所有子环境的批量数据，:meth:`step` 同样接收所有子环境的批量动作。

    尽管该环境是向量化环境，但没有继承 :class:`gym.vector.VectorEnv`。后者包含本项目不需要的等待和
    异步更新接口，而且不同 RL 库通常具有各自的向量环境定义。因此这里直接继承 :class:`gym.Env`，
    由各 RL 库提供的 wrapper 完成适配。

    说明：
        对于向量化环境，建议只在环境创建后、第一次调用 :meth:`step` 之前调用一次 :meth:`reset`。
        此后，:meth:`step` 会自动重置已终止的子环境。
    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    """环境是否是一个向量化环境。"""
    '''
    告诉 Gymnasium 这是一个向量化环境（内部并行管理 N 个环境），而非单环境。
        ClassVar 是 typing 模块的标记——表示这是类变量（所有实例共享），不是实例变量。
        mypy 和 IDE 用它区分。
    '''

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
    }
    """Metadata for the environment."""
    """对环境的元数据。"""
    '''
    定义环境的元数据,可以理解为：这个环境对外声明的一些基本信息
    Gymnasium 要求的元数据字典，声明支持的渲染模式：
        模式	        用途
        None	        不渲染
        "human"	        GUI 窗口显示,表示给人看的渲染模式
        "rgb_array"	    返回 RGB 像素数组（录制视频用）
    '''

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the environment."""
    """对环境的配置。"""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        """初始化环境。

        参数：
            cfg: 环境的配置。
            render_mode: 环境的渲染模式。默认为 None，其行为与 ``"human"`` 类似。
        """
        # -- counter for curriculum
        self.common_step_counter = 0    # 课程学习计数器
        '''
        和 _sim_step_counter（物理步计数器）不同，这是环境步计数器——每环境步 +1。
        CurriculumManager 可能用它判断"训练到第几步了，该提高难度了吗"。
        '''

        # initialize the episode length buffer BEFORE loading the managers to use it in mdp functions.
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)

        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)
        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the
        #    produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt  # 视频录制帧率

        print("[INFO]: Completed setting up the environment...")
        '''
        和 ManagerBasedEnv.__init__ 的继承关系
            ManagerBasedEnv.__init__()
                → 11 个阶段（校验、SimulationContext、InteractiveScene、EventManager、仿真启动...）
                → 在第 ⑧ 阶段: self.load_managers()
                    → 多态到 ManagerBasedRLEnv.load_managers()
        '''

    """
    Properties.
    """
    """属性。
    """

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        """以秒为单位的最大回合长度。"""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        """以环境步数表示的最大回合长度。"""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """
    """操作 - 初始化。
    """

    def load_managers(self):
        '''
        ManagerBasedRLEnv.load_managers()
            │
            ├── 1. CommandManager
            │
            ├── super().load_managers()
            │       ├── 2. EventManager 已经存在，只打印
            │       ├── 3. RecorderManager
            │       ├── 4. ActionManager
            │       └── 5. ObservationManager
            │
            ├── 6. TerminationManager
            │
            ├── 7. RewardManager
            │
            ├── 8. CurriculumManager
            │
            ├── 9. 配置 Gym spaces
            │
            └── 10. 执行 startup events
        为什么 CommandManager 最先创建
            Observation 可能包含：
            ObsTerm(
                func=mdp.generated_commands,
                params={"command_name": "base_velocity"},
            )
            因此 ObservationManager 初始化时必须已经有 command_manager。
        为什么 ActionManager 在 ObservationManager 前
            Observation 可能包含：
            ObsTerm(func=mdp.last_action)
            所以必须先有 action_manager。
        为什么 TerminationManager 在 RewardManager 前
            奖励可能使用：
            mdp.is_alive
            mdp.is_terminated
            它们内部读取：
            env.termination_manager.terminated
            所以必须先有 termination_manager。
        为什么 startup event 最后执行
            startup event 可能依赖：
            robot。
            sensors。
            commands。
            action manager。
            reward manager。
            其他 term。
            因此等所有 Manager 创建完后再执行。
        '''
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    # 实时数据面板
    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""
        """为各管理器项创建实时可视化器。"""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }

    """
    Operations - MDP
    """
    """操作 - MDP。
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        """执行一个环境时间步，并重置已经结束的环境。

        与 :class:`ManagerBasedEnv.step` 不同，该函数依次执行：

        1. 处理动作。
        2. 推进物理仿真。
        3. 在启用 GUI 或 RTX 传感器时执行渲染。
        4. 更新环境计数器并计算奖励与终止信号。
        5. 重置已经终止或超时的环境。
        6. 计算观测。
        7. 返回观测、奖励、终止信号、截断信号和附加信息。

        参数：
            action: 施加到环境的动作，形状为 ``(num_envs, action_dim)``。

        返回：
            包含观测、奖励、终止信号、截断信号和附加信息的元组。
        """
        '''
        策略动作 action
            ▼
        step(action)
            │
            ├── ① process_action                   动作预处理
            ├── ② 物理循环 (× decimation 次)         高频物理仿真
            ├── ③ 更新计数器                         episode_length_buf += 1
            ├── ④ 终止判断 + 奖励计算                 termination → reward
            ├── ⑤ 自动重置已终止环境                   _reset_idx(reset_env_ids)
            ├── ⑥ 更新命令 + interval 事件
            ├── ⑦ 计算观测                           observation after reset
            └── return (obs, reward, terminated, truncated, extras)

        '''

        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
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
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)
            '''
            Gymnasium 向量化环境中，step() 返回的 observation 会自动反映重置后的新状态，不需要你手动调 reset()。
            这节省了一轮 Gymnasium 的 auto_reset wrapper。
            '''

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)
        '''
        为什么 observation 在 reset 后计算
            step 中先自动 reset，再计算 observation：
            物理推进
                ↓
            计算旧 episode 的 terminated/reward
                ↓
            reset 结束环境
                ↓
            计算 observations
            因此对于刚终止的环境：
            reward / done
                对应 reset 前最后一个状态

            返回的 observation
                对应 reset 后新 episode 初始状态
            这是向量化 RL 环境中的常见约定。
            可以表示为：
            transition:
            s_t --a_t--> terminal state
                        │
                        ├── reward_t
                        ├── done_t = True
                        └── 自动 reset
                                ↓
                            返回 s_0_new
            RSL-RL 通过 done 知道这个 observation 已属于新 episode。
        '''

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
        #       [N, obs_dim]        [N]                 [N] bool            [N] bool            dict


    '''
    Gymnasium 标准的渲染输出
        两种渲染模式
            render(mode="human")     → 返回 None（GUI 实时显示，不需要额外输出）
            render(mode="rgb_array") → 返回 np.ndarray [H, W, 3]（像素数据，用于录视频/截图）
    '''
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
        """在不推进物理仿真的情况下执行渲染。

        按照 Gymnasium 约定，不同模式的行为如下：

        - **human**：渲染到当前显示设备，不返回图像，主要用于人工查看。
        - **rgb_array**：返回形状为 ``(x, y, 3)`` 的 ``numpy.ndarray``，表示 RGB 图像，可用于生成视频。

        参数：
            recompute: 即使仿真器已经渲染过当前场景，是否仍强制重新渲染。默认为 False。

        返回：
            当模式为 ``"rgb_array"`` 时返回 NumPy 图像数组，否则返回 None。

        异常：
            RuntimeError: 当前仿真渲染模式不支持请求的 RGB 图像输出。
                          此时仿真渲染模式必须为 ``RenderMode.PARTIAL_RENDERING`` 或
                          ``RenderMode.FULL_RENDERING``。
            NotImplementedError: 指定了不支持的渲染模式。
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

    '''
    RL 子类的析构入口
    和创建时一样，销毁也是按依赖的反序进行——RL 专属 Manager 先销毁，基类后销毁。
    '''
    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager
            # call the parent class to close the environment
            super().close()

    """
    Helper functions.
    """
    """辅助函数。
    """

    '''
    构建 Gymnasium 标准空间定义
        IsaacLab 内部用 7 个 Manager 灵活管理观测和动作——观测可以有多个 group（policy/critic），每个 group 可以拼接或字典格式，term 可以带 clip/scale
        但 RL 训练库（RSL-RL、SB3、SKRL）只懂 Gymnasium 的标准 gym.spaces。

        这个方法就是翻译层——把 IsaacLab 的灵活配置翻译成 Gymnasium 能理解的空间定义。
            IsaacLab 内部:                          Gymnasium 标准:
            ObservationManager                         observation_space
                policy: joint_pos(7) + joint_vel(7)   →  Box(shape=(15,))
                critic: ...                           →  Box(shape=(16,))

            ActionManager                         →  action_space = Box(shape=(8,))
    '''
    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        """为 Gym 环境配置动作空间和观测空间。"""
        # observation space (unbounded since we don't impose any limits)    一、观测空间：两种模式
        self.single_observation_space = gym.spaces.Dict()   # → {"policy": Box(...), "critic": Box(...) 或 Dict{...}}
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            # check if group is concatenated or not
            # if not concatenated, then we need to add each term separately as a dictionary
            if has_concatenated_obs:    # 模式 A：拼接模式（concatenate_terms=True）
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
                '''
                一个 group 的所有 term 拼成一个大张量，空间就是一个 Box。
                '''
            else:   # 模式 B：字典模式（concatenate_terms=False）
                group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
                term_dict = {}
                for term_name, term_dim, term_cfg in zip(group_term_names, group_dim, group_term_cfgs):
                    low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
                    high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
                    term_dict[term_name] = gym.spaces.Box(low=low, high=high, shape=term_dim)
                self.single_observation_space[group_name] = gym.spaces.Dict(term_dict)
                '''
                三层 zip 并行遍历——group_term_names（名字列表）、group_dim（维度列表）、group_term_cfgs（配置列表）在 _prepare_terms 中按相同顺序填充，索引严格对齐。
                clip 的传递：如果用户配置了 clip=(-5.0, 5.0)，Gym Box 的 low/high 就反映真实裁剪范围。没配则 -inf ~ inf。这让 RL 库可以对观测做合法范围验证。
                '''
        # action space (unbounded since we don't impose any limits)     二、动作空间：简单 Box
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))
        '''
        动作无界（-inf ~ inf），因为实际裁剪由 ActionTerm 的 clip 在内部处理，不需要反映到 Gym 空间定义。
        '''

        # batch the spaces for vectorized environments      三、向量化：单环境 → 批量环境
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        '''
        batch_space 把单环境空间 Box(shape=(15,)) 转为向量化空间 Box(shape=(4096, 15))。
        batch_space 是 Gymnasium 提供的工具：
            # 输入: Box(shape=(15,))              ← 单环境，15 维观测
            # 输出: Box(shape=(4096, 15))         ← 4096 并行环境

            # 输入: Dict{"policy": Box(shape=(15,)), "critic": Box(shape=(16,))}
            # 输出: Dict{"policy": Box(shape=(4096, 15)), "critic": Box(shape=(4096, 16))}
        single_xxx 和 xxx 的命名区分：
            single_observation_space 是一个环境的空间定义（Gymnasium 注册时需要的），observation_space 是批量环境的（RL wrapper 用的）。
            前者描述"一个样本长什么样"，后者描述"一批样本长什么样"。
        '''
        '''
        完整的数据流
            _configure_gym_env_spaces()
                │
                ├── observation_manager.active_terms
                │     {"policy": ["joint_pos", "joint_vel"], "critic": [...]}
                │
                ├── observation_manager.group_obs_dim
                │     {"policy": (15,), "critic": (16,)}
                │
                ├── observation_manager.group_obs_concatenate
                │     {"policy": True, "critic": True}
                │
                ├── observation_manager._group_obs_term_cfgs
                │     {"policy": [ObsTermCfg(clip=None), ObsTermCfg(clip=None)]}
                │
                ├── action_manager.action_term_dim
                │     [7, 1]  →  sum →  8
                │
                └── → single_xxx + batch_xxx → Gym 标准空间
        '''

    '''
    RL 环境的重置总调度
        当环境摔倒或超时，step() 自动调它完成"重新开一局"的全部操作。
    '''
    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        """根据指定索引重置环境。

        参数：
            env_ids: 必须重置的环境 ID 列表。
        """
        '''
        _reset_idx(env_ids)
            │
            ├── ① curriculum_manager.compute(env_ids)      课程难度更新
            ├── ② scene.reset(env_ids)                     物理状态恢复默认
            ├── ③ event_manager.apply("reset", ...)        随机化事件触发
            │
            └── ④ 9 个 Manager 逐个 reset + 收集日志
                ├── observation_manager.reset(env_ids)    清空观测历史缓冲区
                ├── action_manager.reset(env_ids)         清零动作缓冲区
                ├── reward_manager.reset(env_ids)         输出回合奖励统计
                ├── curriculum_manager.reset(env_ids)     重置课程状态
                ├── command_manager.reset(env_ids)        重新采样命令
                ├── event_manager.reset(env_ids)          重置 interval 倒计时
                ├── termination_manager.reset(env_ids)    输出终止原因统计
                ├── recorder_manager.reset(env_ids)       导出 episode 数据
                │
                └── episode_length_buf[env_ids] = 0       回合步数归零
        '''
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0
        '''
        为什么 CurriculumManager 最先出现两次？
            compute 在 scene.reset 之前——因为课程学习可能在环境重置时决定"下一个回合应该多难"。
            例如，如果某个环境表现太好，compute 可以决定"下回合把摩擦系数范围扩大 20%"。
            这个决策必须在 event_manager.apply("reset")（真正随机化物理参数）之前做完，否则随机化用的是旧难度。

            reset 在后面——负责清理课程内部状态（如记录该环境的难度变化历史），和其余 Manager 的 reset 时序对齐。

        self.extras["log"].update(info) — 日志聚合模式
            每个 Manager 的 reset() 返回一个字典（如 {"Episode_Reward/alive": 1.2, "Episode_Reward/pole_pos": -0.5}）。
            全部 update 到同一个 extras["log"] 字典中，最后一步由训练框架写入 TensorBoard。
            这是一个累加聚合模式——每个 Manager 只管自己的日志，不需要知道其他 Manager 输出了什么。
        '''
