# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an environment instance to Stable-Baselines3 vectorized environment.

The following example shows how to wrap an environment for Stable-Baselines3:

.. code-block:: python

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    env = Sb3VecEnvWrapper(env)

"""

# needed to import for allowing type-hinting: torch.Tensor | dict[str, torch.Tensor]
from __future__ import annotations
"""包装器将环境实例配置为稳定基线3向量化环境。

下面的例子显示了如何包装稳定基线的环境3:

.. code-block:: python

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    env = Sb3VecEnvWrapper(env)
"""

import warnings
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn  # noqa: F401
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

# remove SB3 warnings because PPO with bigger net actually benefits from GPU
warnings.filterwarnings("ignore", message="You are trying to run PPO on the GPU")

"""
Configuration Parser.
"""
"""配置解析器。
"""


def process_sb3_cfg(cfg: dict, num_envs: int) -> dict:
    """Convert simple YAML types to Stable-Baselines classes/components.

    Args:
        cfg: A configuration dictionary.
        num_envs: the number of parallel environments (used to compute `batch_size` for a desired number of minibatches)

    Returns:
        A dictionary containing the converted configuration.

    Reference:
        https://github.com/DLR-RM/rl-baselines3-zoo/blob/0e5eb145faefa33e7d79c7f8c179788574b20da5/utils/exp_manager.py#L358
    """
    """将简单的YAML类型转换为稳定基线类/组件。

    参数：
        cfg: 一个配置字典。
        num_envs: 平行环境数量 (用于计算`batch_size`为所需数量的小型批量)

    返回：
        包含转换配置的字典。

    Reference:
        https://github.com/DLR-RM/rl-baselines3-zoo/blob/0e5eb145faefa33e7d79c7f8c179788574b20da5/工具/exp
              _manager.py#L358
    """

    def update_dict(hyperparams: dict[str, Any], depth: int) -> dict[str, Any]:
        for key, value in hyperparams.items():
            if isinstance(value, dict):
                update_dict(value, depth + 1)
            if isinstance(value, str):
                if value.startswith("nn."):
                    hyperparams[key] = getattr(nn, value[3:])
            if depth == 0:
                if key in ["learning_rate", "clip_range", "clip_range_vf"]:
                    if isinstance(value, str):
                        _, initial_value = value.split("_")
                        initial_value = float(initial_value)
                        hyperparams[key] = lambda progress_remaining: progress_remaining * initial_value
                    elif isinstance(value, (float, int)):
                        # negative value: ignore (ex: for clipping)
                        if value < 0:
                            continue
                        hyperparams[key] = constant_fn(float(value))
                    else:
                        raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")

        # Convert to a desired batch_size (n_steps=2048 by default for SB3 PPO)
        if "n_minibatches" in hyperparams:
            hyperparams["batch_size"] = (hyperparams.get("n_steps", 2048) * num_envs) // hyperparams["n_minibatches"]
            del hyperparams["n_minibatches"]

        return hyperparams

    # parse agent configuration and convert to classes
    return update_dict(cfg, depth=0)


"""
Vectorized environment wrapper.
"""
"""面向环境包装。
"""


class Sb3VecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for Stable Baselines3.

    Isaac Sim internally implements a vectorized environment. However, since it is
    still considered a single environment instance, Stable Baselines tries to wrap
    around it using the :class:`DummyVecEnv`. This is only done if the environment
    is not inheriting from their :class:`VecEnv`. Thus, this class thinly wraps
    over the environment from :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.

    Note:
        While Stable-Baselines3 supports Gym 0.26+ API, their vectorized environment
        uses their own API (i.e. it is closer to Gym 0.21). Thus, we implement
        the API for the vectorized environment.

    We also add monitoring functionality that computes the un-discounted episode
    return and length. This information is added to the info dicts under key `episode`.

    In contrast to the Isaac Lab environment, stable-baselines expect the following:

    1. numpy datatype for MDP signals
    2. a list of info dicts for each sub-environment (instead of a dict)
    3. when environment has terminated, the observations from the environment should correspond
       to the one after reset. The "real" final observation is passed using the info dicts
       under the key ``terminal_observation``.

    .. warning::

        By the nature of physics stepping in Isaac Sim, it is not possible to forward the
        simulation buffers without performing a physics step. Thus, reset is performed
        inside the :meth:`step()` function after the actual physics step is taken.
        Thus, the returned observations for terminated environments is the one after the reset.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:

    1. https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    2. https://stable-baselines3.readthedocs.io/en/master/common/monitor.html

    """
    """围绕伊萨克实验室环境进行稳定基线3。

    伊萨克·西姆内部实现了向量化环境。
    然而，由于它仍然被认为是一个单一的环境实例，稳定基线试图使用:class:`DummyVecEnv`包裹它。
    如果环境不继承他们的:class:`VecEnv`，
    因此，这种类型从:class:`ManagerBasedRLEnv`或:class:`DirectRLEnv`的环境中很薄。

    说明：
        虽然Stable-Baselines3支持Gym 0.26+API，但它们的向量化环境使用了自己的API (i.e.更接近Gym 0.21)。
        因此，我们将API用于向量化环境。

    我们还添加了监控功能，
    return and length. This information is added to the info dicts under key `episode`.

    与艾萨克实验室环境不同，稳定基线预计:

    1. MDP信号的数据类型
    2. 每个子环境的信息指令清单 (而不是指令)
    3. 在环境结束时，环境中的观测应与重置后的观测相匹配.使用``terminal_observation``键下的信息指示通过"真实"的最终观测。

    .. 警告::

        由于物理在伊萨克·西姆中踏入，
        因此，在实际物理步骤完成后，在:meth:`step()`函数内进行重置。
        因此，终止环境的返回观测是重置后的。

    .. 谨慎::

        这类必须是包装链中的最后一个包装。
        这是因为包裹不跟随
        the :类:`gym.Wrapper`接口。
             任何随后的包装都需要修改，以使用此
        包装。

    Reference:

    1. https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    2. https://stable-baselines3.readthedocs.io/en/master/common/monitor.html
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, fast_variant: bool = True):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.
            fast_variant: Use fast variant for processing info
                (Only episodic reward, lengths and truncation info are included)
        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        """启动包装。

        参数：
            env: 周围的环境。
            fast_variant: 使用快速变量来处理信息 (仅包含回合奖励，长度和缩短信息)
        异常：
            ValueError: 当环境不是:class:`ManagerBasedRLEnv`或:class:`DirectRLEnv`的实例时。
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        self.fast_variant = fast_variant
        # collect common information
        self.num_envs = self.unwrapped.num_envs
        self.sim_device = self.unwrapped.device
        self.render_mode = self.unwrapped.render_mode
        self.observation_processors = {}
        self._process_spaces()
        # add buffer for logging episodic information
        self._ep_rew_buf = np.zeros(self.num_envs)
        self._ep_len_buf = np.zeros(self.num_envs)

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        """返回包装名称和:attr:`env`表示字符串。"""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        """返回包装的字符串表示。"""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """
    """属性 - Gym.Wrapper
    """

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

    def get_episode_rewards(self) -> list[float]:
        """Returns the rewards of all the episodes."""
        """返回了所有集中的奖励。"""
        return self._ep_rew_buf.tolist()

    def get_episode_lengths(self) -> list[int]:
        """Returns the number of time-steps of all the episodes."""
        """返回所有集中的时间步骤。"""
        return self._ep_len_buf.tolist()

    """
    Operations - MDP
    """
    """运营 - MDP
    """

    def seed(self, seed: int | None = None) -> list[int | None]:  # noqa: D102
        return [self.unwrapped.seed(seed)] * self.unwrapped.num_envs

    def reset(self) -> VecEnvObs:  # noqa: D102
        obs_dict, _ = self.env.reset()
        # reset episodic information buffers
        self._ep_rew_buf = np.zeros(self.num_envs)
        self._ep_len_buf = np.zeros(self.num_envs)
        # convert data types to numpy depending on backend
        return self._process_obs(obs_dict)

    def step_async(self, actions):  # noqa: D102
        # convert input to numpy array
        if not isinstance(actions, torch.Tensor):
            actions = np.asarray(actions)
            actions = torch.from_numpy(actions).to(device=self.sim_device, dtype=torch.float32)
        else:
            actions = actions.to(device=self.sim_device, dtype=torch.float32)
        # convert to tensor
        self._async_actions = actions

    def step_wait(self) -> VecEnvStepReturn:  # noqa: D102
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(self._async_actions)
        # compute reset ids
        dones = terminated | truncated

        # convert data types to numpy depending on backend
        # note: ManagerBasedRLEnv uses torch backend (by default).
        obs = self._process_obs(obs_dict)
        rewards = rew.detach().cpu().numpy()
        terminated = terminated.detach().cpu().numpy()
        truncated = truncated.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()

        reset_ids = dones.nonzero()[0]

        # update episode un-discounted return and length
        self._ep_rew_buf += rewards
        self._ep_len_buf += 1
        # convert extra information to list of dicts
        infos = self._process_extras(obs, terminated, truncated, extras, reset_ids)

        # reset info for terminated environments
        self._ep_rew_buf[reset_ids] = 0.0
        self._ep_len_buf[reset_ids] = 0

        return obs, rewards, dones, infos

    def close(self):  # noqa: D102
        self.env.close()

    def get_attr(self, attr_name, indices=None):  # noqa: D102
        # resolve indices
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices)
        # obtain attribute value
        attr_val = getattr(self.env, attr_name)
        # return the value
        if not isinstance(attr_val, torch.Tensor):
            return [attr_val] * num_indices
        else:
            return attr_val[indices].detach().cpu().numpy()

    def set_attr(self, attr_name, value, indices=None):  # noqa: D102
        raise NotImplementedError("Setting attributes is not supported.")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):  # noqa: D102
        if method_name == "render":
            # gymnasium does not support changing render mode at runtime
            return self.env.render()
        else:
            # this isn't properly implemented but it is not necessary.
            # mostly done for completeness.
            env_method = getattr(self.env, method_name)
            return env_method(*method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):  # noqa: D102
        # fake implementation to be able to use `evaluate_policy()` helper
        return [False]

    def get_images(self):  # noqa: D102
        raise NotImplementedError("Getting images is not supported.")

    """
    Helper functions.
    """
    """辅助函数。
    """

    def _process_spaces(self):
        # process observation space
        observation_space = self.unwrapped.single_observation_space["policy"]
        if isinstance(observation_space, gym.spaces.Dict):
            for obs_key, obs_space in observation_space.spaces.items():
                processors: list[callable[[torch.Tensor], Any]] = []
                # assume normalized, if not, it won't pass is_image_space, which check [0-255].
                # for scale like image space that has right shape but not scaled, we will scale it later
                if is_image_space(obs_space, check_channels=True, normalized_image=True):
                    actually_normalized = np.all(obs_space.low == -1.0) and np.all(obs_space.high == 1.0)
                    if not actually_normalized:
                        if np.any(obs_space.low != 0) or np.any(obs_space.high != 255):
                            raise ValueError(
                                "Your image observation is not normalized in environment, and will not be"
                                "normalized by sb3 if its min is not 0 and max is not 255."
                            )
                        # sb3 will handle normalization and transpose, but sb3 expects uint8 images
                        if obs_space.dtype != np.uint8:
                            processors.append(lambda obs: obs.to(torch.uint8))
                        observation_space.spaces[obs_key] = gym.spaces.Box(0, 255, obs_space.shape, np.uint8)
                    else:
                        # sb3 will NOT handle the normalization, while sb3 will transpose, its transpose applies to all
                        # image terms and maybe non-ideal, more, if we can do it in torch on gpu, it will be faster then
                        # sb3 transpose it in numpy with cpu.
                        if not is_image_space_channels_first(obs_space):

                            def tranp(img: torch.Tensor) -> torch.Tensor:
                                return img.permute(2, 0, 1) if len(img.shape) == 3 else img.permute(0, 3, 1, 2)

                            processors.append(tranp)
                            h, w, c = obs_space.shape
                            observation_space.spaces[obs_key] = gym.spaces.Box(-1.0, 1.0, (c, h, w), obs_space.dtype)

                    def chained_processor(obs: torch.Tensor, procs=processors) -> Any:
                        for proc in procs:
                            obs = proc(obs)
                        return obs

                    # add processor to the dictionary
                    if len(processors) > 0:
                        self.observation_processors[obs_key] = chained_processor

        # obtain gym spaces
        # note: stable-baselines3 does not like when we have unbounded action space so
        #   we set it to some high value here. Maybe this is not general but something to think about.
        action_space = self.unwrapped.single_action_space
        if isinstance(action_space, gym.spaces.Box) and not action_space.is_bounded("both"):
            action_space = gym.spaces.Box(low=-100, high=100, shape=action_space.shape)

        # initialize vec-env
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)

    def _process_obs(self, obs_dict: torch.Tensor | dict[str, torch.Tensor]) -> np.ndarray | dict[str, np.ndarray]:
        """Convert observations into NumPy data type."""
        """将观测转换为NumPy数据类型。"""
        # Sb3 doesn't support asymmetric observation spaces, so we only use "policy"
        obs = obs_dict["policy"]
        # note: ManagerBasedRLEnv uses torch backend (by default).
        if isinstance(obs, dict):
            for key, value in obs.items():
                if key in self.observation_processors:
                    obs[key] = self.observation_processors[key](value)
                obs[key] = obs[key].detach().cpu().numpy()
        elif isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")
        return obs

    def _process_extras(
        self, obs: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, extras: dict, reset_ids: np.ndarray
    ) -> list[dict[str, Any]]:
        """Convert miscellaneous information into dictionary for each sub-environment."""
        """转换各种信息为每个子环境的字典。"""
        # faster version: only process env that terminated and add bootstrapping info
        if self.fast_variant:
            infos = [{} for _ in range(self.num_envs)]

            for idx in reset_ids:
                # fill-in episode monitoring info
                infos[idx]["episode"] = {
                    "r": self._ep_rew_buf[idx],
                    "l": self._ep_len_buf[idx],
                }

                # fill-in bootstrap information
                infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]

                # add information about terminal observation separately
                if isinstance(obs, dict):
                    terminal_obs = {key: value[idx] for key, value in obs.items()}
                else:
                    terminal_obs = obs[idx]
                infos[idx]["terminal_observation"] = terminal_obs

            return infos

        # create empty list of dictionaries to fill
        infos: list[dict[str, Any]] = [dict.fromkeys(extras.keys()) for _ in range(self.num_envs)]
        # fill-in information for each sub-environment
        # note: This loop becomes slow when number of environments is large.
        for idx in range(self.num_envs):
            # fill-in episode monitoring info
            if idx in reset_ids:
                infos[idx]["episode"] = dict()
                infos[idx]["episode"]["r"] = float(self._ep_rew_buf[idx])
                infos[idx]["episode"]["l"] = float(self._ep_len_buf[idx])
            else:
                infos[idx]["episode"] = None
            # fill-in bootstrap information
            infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]
            # fill-in information from extras
            for key, value in extras.items():
                # 1. remap extra episodes information safely
                # 2. for others just store their values
                if key == "log":
                    # only log this data for episodes that are terminated
                    if infos[idx]["episode"] is not None:
                        for sub_key, sub_value in value.items():
                            infos[idx]["episode"][sub_key] = sub_value
                else:
                    infos[idx][key] = value[idx]
            # add information about terminal observation separately
            if idx in reset_ids:
                # extract terminal observations
                if isinstance(obs, dict):
                    terminal_obs = dict.fromkeys(obs.keys())
                    for key, value in obs.items():
                        terminal_obs[key] = value[idx]
                else:
                    terminal_obs = obs[idx]
                # add info to dict
                infos[idx]["terminal_observation"] = terminal_obs
            else:
                infos[idx]["terminal_observation"] = None
        # return list of dictionaries
        return infos
