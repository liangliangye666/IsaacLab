# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an environment instance to RL-Games vectorized environment.

The following example shows how to wrap an environment for RL-Games and register the environment construction
for RL-Games :class:`Runner` class:

.. code-block:: python

    from rl_games.common import env_configurations, vecenv

    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    # configuration parameters
    rl_device = "cuda:0"
    clip_obs = 10.0
    clip_actions = 1.0

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

"""

# needed to import for allowing type-hinting:gym.spaces.Box | None
from __future__ import annotations
"""将环境实例配置为RL-Games的向量化环境。

下面的例子显示了如何包装RL游戏的环境，并记录环境建设
for RL-Games :class:`Runner` class:

.. code-block:: python

    from rl_games.common import env_configurations, vecenv

    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    # configuration parameters
    rl_device = "cuda:0"
    clip_obs = 10.0
    clip_actions = 1.0

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
"""

from collections.abc import Callable

import gym.spaces  # needed for rl-games incompatibility: https://github.com/Denys88/rl_games/issues/261
import gymnasium
import torch
from rl_games.common import env_configurations
from rl_games.common.vecenv import IVecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv, VecEnvObs

"""
Vectorized environment wrapper.
"""
"""面向环境包装。
"""


class RlGamesVecEnvWrapper(IVecEnv):
    """Wraps around Isaac Lab environment for RL-Games.

    This class wraps around the Isaac Lab environment. Since RL-Games works directly on
    GPU buffers, the wrapper handles moving of buffers from the simulation environment
    to the same device as the learning agent. Additionally, it performs clipping of
    observations and actions.

    For algorithms like asymmetric actor-critic, RL-Games expects a dictionary for
    observations. This dictionary contains "obs" and "states" which typically correspond
    to the actor and critic observations respectively.

    To use asymmetric actor-critic, map privileged observation groups under ``"states"`` (e.g. ``["critic"]``).

    The wrapper supports **either** concatenated tensors (default) **or** Dict inputs:
    when wrapper is concate mode, rl-games sees {"obs": Tensor, (optional)"states": Tensor}
    when wrapper is not concate mode, rl-games sees {"obs": dict[str, Tensor], (optional)"states": dict[str, Tensor]}

    - Concatenated mode (``concate_obs_group=True``): ``observation_space``/``state_space`` are ``gym.spaces.Box``.
    - Dict mode (``concate_obs_group=False``): ``observation_space``/``state_space`` are ``gym.spaces.Dict`` keyed by
      the requested groups. When no ``"states"`` groups are provided, the states Dict is omitted at runtime.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.


    Reference:
        https://github.com/Denys88/rl_games/blob/master/rl_games/common/ivecenv.py
        https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
    """
    """围绕艾萨克实验室环境进行RL游戏。

    这类课程围绕着艾萨克实验室环境。
    由于RL-Games直接在GPU缓冲器上工作，包装处理缓冲器从仿真环境移动到与学习代理相同的设备。
    此外，它还执行了观测和动作的裁剪。

    对于不对称的演员-批评者等算法，RL-Games期望有一个字典来观测。
    这个字典包含"obs"和"states"，通常与演员和评论家的观测相符。

    为了使用不对称的演员-批评者，在``"states"`` (e.g。 ``["critic"]``) 下映射特权观测组。

    包装支持**或**连锁紧器 (默认) **或** 字幕输入:当包装是连接模式时，rl-游戏看到{"obs": Tensor， (optional)"states":
    Tensor}当包装不是连接模式时，rl-游戏看到{"obs": dict[str， Tensor]， (optional)"states": dict[str， Tensor]}

    - 连接模式 (``concate_obs_group=True``):``observation_space``/``state_space``是``gym.spaces.Box``。
    - 语句模式 (``concate_obs_group=False``):``observation_space``/``state_space``是要求组键 ``gym.spaces.Dict``。
      当没有提供``"states"``组时，在运行时，语句状态被遗漏。

    .. 谨慎::

        这类必须是包装链中的最后一个包装。
        这是因为包裹不跟随
        the :类:`gym.Wrapper`接口。
             任何随后的包装都需要修改，以使用此
        包装。


    Reference:
        https://github.com/丹尼斯88/rl_games/ master/老师rl_games/常见/ivecenv.py
        https://github.com/NVIDIA- 全球IsaacGymEnvs
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv | DirectRLEnv,
        rl_device: str,
        clip_obs: float,
        clip_actions: float,
        obs_groups: dict[str, list[str]] | None = None,
        concate_obs_group: bool = True,
    ):
        """Initializes the wrapper instance.

        Args:
            env: The environment to wrap around.
            rl_device: The device on which agent computations are performed.
            clip_obs: The clipping value for observations.
            clip_actions: The clipping value for actions.
            obs_groups: The remapping from isaaclab observation to rl-games, default to None for backward compatible.
            concate_obs_group: The boolean value indicates if input to rl-games network is dict or tensor. Default to
                True for backward compatible.

        Raises:
            ValueError: The environment is not inherited from :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
            ValueError: If specified, the privileged observations (critic) are not of type :obj:`gym.spaces.Box`.
        """
        """启动包装实例。

        参数：
            env: 周围的环境。
            rl_device: 执行代理计算的设备。
            clip_obs: 对观测的裁剪值。
            clip_actions: 裁剪值为动作。
            obs_groups: 从 isaaclab 观测到 rl-游戏，默认到None适用于后退兼容性。
            concate_obs_group: 布尔值表示，如果输入rl-games网络是 dict或 tensor。
                               在 True 默认情况下，

        异常：
            ValueError: 环境不是从:class:`ManagerBasedRLEnv`或:class:`DirectRLEnv`中继承的。
            ValueError: 如果指定，特权观测 (批评) 不属于:obj:`gym.spaces.Box`类型。
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        # store provided arguments
        self._rl_device = rl_device
        self._clip_obs = clip_obs
        self._clip_actions = clip_actions
        self._sim_device = env.unwrapped.device

        # resolve the observation group
        self._concate_obs_groups = concate_obs_group
        self._obs_groups = obs_groups
        if obs_groups is None:
            self._obs_groups = {"obs": ["policy"], "states": []}
            if not self.unwrapped.single_observation_space.get("policy"):
                raise KeyError("Policy observation group is expected if no explicit groups is defined")
            if self.unwrapped.single_observation_space.get("critic"):
                self._obs_groups["states"] = ["critic"]

        if (
            self._concate_obs_groups
            and isinstance(self.state_space, gym.spaces.Box)
            and isinstance(self.observation_space, gym.spaces.Box)
        ):
            self.rlg_num_states = self.state_space.shape[0]
        elif (
            not self._concate_obs_groups
            and isinstance(self.state_space, gym.spaces.Dict)
            and isinstance(self.observation_space, gym.spaces.Dict)
        ):
            space = [space.shape[0] for space in self.state_space.values()]
            self.rlg_num_states = sum(space)
        else:
            raise TypeError(
                "only valid combination for state space is gym.space.Box when concate_obs_groups is True,             "
                "   and gym.space.Dict when concate_obs_groups is False. You have concate_obs_groups:                "
                f" {self._concate_obs_groups}, and state_space: {self.state_space.__class__}"
            )

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        """返回包装名称和:attr:`env`表示字符串。"""
        return (
            f"<{type(self).__name__}{self.env}>"
            f"\n\tObservations clipping: {self._clip_obs}"
            f"\n\tActions clipping     : {self._clip_actions}"
            f"\n\tAgent device         : {self._rl_device}"
            f"\n\tAsymmetric-learning  : {self.rlg_num_states != 0}"
        )

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        """返回包装的字符串表示。"""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """
    """属性 - Gym.Wrapper
    """

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        """返回了:attr:`Env`:attr:`render_mode`。"""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.spaces.Box | gym.spaces.Dict:
        """Returns the :attr:`Env` :attr:`observation_space` (``Box`` if concatenated, otherwise ``Dict``)."""
        """返回:attr:`Env` :attr:`observation_space` (如果连接``Box``，否则``Dict``)。"""
        # note: rl-games only wants single observation space
        space = self.unwrapped.single_observation_space
        clip = self._clip_obs
        if not self._concate_obs_groups:
            policy_space = {grp: gym.spaces.Box(-clip, clip, space.get(grp).shape) for grp in self._obs_groups["obs"]}
            return gym.spaces.Dict(policy_space)
        else:
            shapes = [space.get(group).shape for group in self._obs_groups["obs"]]
            cat_shape, self._obs_concat_fn = make_concat_plan(shapes)
            return gym.spaces.Box(-clip, clip, cat_shape)

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        """返回了:attr:`Env`:attr:`action_space`。"""
        # note: rl-games only wants single action space
        action_space = self.unwrapped.single_action_space
        if not isinstance(action_space, gymnasium.spaces.Box):
            raise NotImplementedError(
                f"The RL-Games wrapper does not currently support action space: '{type(action_space)}'."
                f" If you need to support this, please modify the wrapper: {self.__class__.__name__},"
                " and if you are nice, please send a merge-request."
            )
        # return casted space in gym.spaces.Box (OpenAI Gym)
        # note: maybe should check if we are a sub-set of the actual space. don't do it right now since
        #   in ManagerBasedRLEnv we are setting action space as (-inf, inf).
        return gym.spaces.Box(-self._clip_actions, self._clip_actions, action_space.shape)

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        """返回包装的类名字。"""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        """返回包装的基础环境。

        这将是赤裸裸的:class:`gymnasium.Env`环境，
        """
        return self.env.unwrapped

    """
    Properties
    """
    """产品
    """

    @property
    def num_envs(self) -> int:
        """Returns the number of sub-environment instances."""
        """返回子环境实例数。"""
        return self.unwrapped.num_envs

    @property
    def device(self) -> str:
        """Returns the base environment simulation device."""
        """返回基环境仿真设备。"""
        return self.unwrapped.device

    @property
    def state_space(self) -> gym.spaces.Box | gym.spaces.Dict | None:
        """Returns the privileged observation space for the critic (``Box`` if concatenated, otherwise ``Dict``)."""
        """返回对评论者的特权观测空间 (如果连锁 ``Box``，否则 ``Dict``)。"""
        # # note: rl-games only wants single observation space
        space = self.unwrapped.single_observation_space
        clip = self._clip_obs
        if not self._concate_obs_groups:
            state_space = {grp: gym.spaces.Box(-clip, clip, space.get(grp).shape) for grp in self._obs_groups["states"]}
            return gym.spaces.Dict(state_space)
        else:
            shapes = [space.get(group).shape for group in self._obs_groups["states"]]
            cat_shape, self._states_concat_fn = make_concat_plan(shapes)
            return gym.spaces.Box(-self._clip_obs, self._clip_obs, cat_shape)

    def get_number_of_agents(self) -> int:
        """Returns number of actors in the environment."""
        """返回环境中参与者的数量。"""
        return getattr(self, "num_agents", 1)

    def get_env_info(self) -> dict:
        """Returns the Gym spaces for the environment."""
        """恢复体育馆的环境空间。"""
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }

    """
    Operations - MDP
    """
    """运营 - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self):  # noqa: D102
        obs_dict, _ = self.env.reset()
        # process observations and states
        return self._process_obs(obs_dict)

    def step(self, actions):  # noqa: D102
        # move actions to sim-device
        actions = actions.detach().clone().to(device=self._sim_device)
        # clip the actions
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        # perform environment step
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        # note: only useful when `value_bootstrap` is True in the agent configuration
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated.to(device=self._rl_device)
        # process observations and states
        obs_and_states = self._process_obs(obs_dict)
        # move buffers to rl-device
        # note: we perform clone to prevent issues when rl-device and sim-device are the same.
        rew = rew.to(device=self._rl_device)
        dones = (terminated | truncated).to(device=self._rl_device)
        extras = {
            k: v.to(device=self._rl_device, non_blocking=True) if hasattr(v, "to") else v for k, v in extras.items()
        }
        # remap extras from "log" to "episode"
        if "log" in extras:
            extras["episode"] = extras.pop("log")

        return obs_and_states, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """
    """助理功能
    """

    def _process_obs(self, obs_dict: VecEnvObs) -> dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]]:
        """Processing of the observations and states from the environment.

        Note:
            States typically refers to privileged observations for the critic function. It is typically used in
            asymmetric actor-critic algorithms.

        Args:
            obs_dict: The current observations from environment.

         Returns:
            A dictionary for RL-Games with keys:
            - ``"obs"``: either a concatenated tensor (``concate_obs_group=True``) or a Dict of group tensors.
            - ``"states"`` (optional): same structure as above when state groups are configured; omitted otherwise.
        """
        """处理环境中的观测和状态。

        说明：
            国家通常指为批判功能的特权观测。
            它通常用于不对称的演员批判算法。

        参数：
            obs_dict: 目前的环境观测。

         返回：
            对于RL-Games的字典，有键:
            - ``"obs"``:是连接式子 (``concate_obs_group=True``) 或是组 group子的 Dict。
            - ``"states"`` (可选):当配置状态组时，相同的结构如上所述；否则省略。
        """
        # move observations to RL device if different from sim device
        if self._rl_device != self._sim_device:
            obs_dict = {key: obs.to(device=self._rl_device) for key, obs in obs_dict.items()}

        # clip the observations
        for key, obs in obs_dict.items():
            obs_dict[key] = torch.clamp(obs, -self._clip_obs, self._clip_obs)

        # process input obs dict
        rl_games_obs = {"obs": {group: obs_dict[group] for group in self._obs_groups["obs"]}}
        if len(self._obs_groups["states"]) > 0:
            rl_games_obs["states"] = {group: obs_dict[group] for group in self._obs_groups["states"]}

        if self._concate_obs_groups:
            rl_games_obs["obs"] = self._obs_concat_fn(list(rl_games_obs["obs"].values()))
            if "states" in rl_games_obs:
                rl_games_obs["states"] = self._states_concat_fn(list(rl_games_obs["states"].values()))

        return rl_games_obs


def make_concat_plan(shapes: list[tuple[int, ...]]) -> tuple[tuple[int, ...], Callable]:
    """
    Given per-sample shapes (no batch dim), return:
      - the concatenated per-sample shape
      - a function that concatenates a list of batch tensors accordingly.

    Rules:
      0) Empty -> (0,), No-op
      1) All 1D -> concat features (dim=1).
      2) Same rank > 1:
         2a) If all s[:-1] equal -> concat along last dim (channels-last, dim=-1).
         2b) If all s[1:] equal  -> concat along first dim (channels-first, dim=1).
    """
    """根据每样品的形状 (不含批量薄)，返回:
      - 每个样品的连接形状
      - 一个函数，相应连接批量子列表。

    Rules: 0) 空 -> (0，)， No-op 1) 所有1D -> concat功能 (dim=1)。
           2) 同等级 > 1: 2a) 如果所有 s[:-1]等于 -> 沿着最后的暗 (道-最后，暗=-1)。
           2b) 如果所有的s[1:]等于 -> concat沿着第一个暗 (道-第一，暗=1)。
    """
    if len(shapes) == 0:
        return (0,), lambda x: x
    # case 1: all vectors
    if all(len(s) == 1 for s in shapes):
        return (sum(s[0] for s in shapes),), lambda x: torch.cat(x, dim=1)
    # case 2: same rank > 1
    rank = len(shapes[0])
    if all(len(s) == rank for s in shapes) and rank > 1:
        # 2a: concat along last axis (…C)
        if all(s[:-1] == shapes[0][:-1] for s in shapes):
            out_shape = shapes[0][:-1] + (sum(s[-1] for s in shapes),)
            return out_shape, lambda x: torch.cat(x, dim=-1)
        # 2b: concat along first axis (C…)
        if all(s[1:] == shapes[0][1:] for s in shapes):
            out_shape = (sum(s[0] for s in shapes),) + shapes[0][1:]
            return out_shape, lambda x: torch.cat(x, dim=1)
        else:
            raise ValueError(f"Could not find a valid concatenation plan for rank {[(len(s),) for s in shapes]}")
    else:
        raise ValueError("Could not find a valid concatenation plan, please make sure all value share the same size")


"""
Environment Handler.
"""
"""环境管理器。
"""


class RlGamesGpuEnv(IVecEnv):
    """Thin wrapper to create instance of the environment to fit RL-Games runner."""
    """薄包装，使环境的实例适应RL- 游戏跑步。"""

    # TODO: Adding this for now but do we really need this?

    def __init__(self, config_name: str, num_actors: int, **kwargs):
        """Initialize the environment.

        Args:
            config_name: The name of the environment configuration.
            num_actors: The number of actors in the environment. This is not used in this wrapper.
        """
        """初始化环境。

        参数：
            config_name: 环境配置名称
            num_actors: 环境中的参与者数量
                        这种包装不用。
        """
        self.env: RlGamesVecEnvWrapper = env_configurations.configurations[config_name]["env_creator"](**kwargs)

    def step(self, action):  # noqa: D102
        return self.env.step(action)

    def reset(self):  # noqa: D102
        return self.env.reset()

    def get_number_of_agents(self) -> int:
        """Get number of agents in the environment.

        Returns:
            The number of agents in the environment.
        """
        """查看环境中的代理人数。

        返回：
            环境中的代理人数。
        """
        return self.env.get_number_of_agents()

    def get_env_info(self) -> dict:
        """Get the Gym spaces for the environment.

        Returns:
            The Gym spaces for the environment.
        """
        """为环境提供Gym空间。

        返回：
            Gym为环境提供空间。
        """
        return self.env.get_env_info()
