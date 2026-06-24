# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""
    """对于PPO演员批评网络的配置。"""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""
    """策略类名字。
    默认是ActorCritic。
    """

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""
    """策略的最初噪音标准偏差。"""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""
    """策略的噪音标准偏差类型。
    默认是 skalar。
    """

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Default is False."""
    """对策略是否使用依赖国家标准偏差。
    默认是False。
    """

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""
    """对于演员网络来说，"""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""
    """对于批评网络来说，"""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""
    """演员网络的隐藏维度。"""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""
    """批评网络的隐藏维度。"""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""
    """演员和评论家网络的激活功能。"""


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""
    """对PPO演员-批评网络进行配置，具有复发层。"""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticRecurrent."""
    """策略类名字。
    默认是ActorCriticRecurrent。
    """

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""
    """如何使用RNN?
    无论是"Istm"还是"Gru"。
    """

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""
    """它们的尺寸RNN它们的层次。"""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""
    """总数RNN它们的层次。"""


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""
    """为PPO算法的配置。"""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""
    """算法类名称。
    默认是PPO。
    """

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""
    """每次更新的学习期数。"""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""
    """每次更新的小型批量。"""

    learning_rate: float = MISSING
    """The learning rate for the policy."""
    """策略的学习率。"""

    schedule: str = MISSING
    """The learning rate schedule."""
    """学习率的时间表。"""

    gamma: float = MISSING
    """The discount factor."""
    """折扣因素。"""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""
    """一般优势估计 (GAE) 的Lambda参数。"""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""
    """输入力损失的系数。"""

    desired_kl: float = MISSING
    """The desired KL divergence."""
    """想要的KL分离。"""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""
    """最高梯度标准。"""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""
    """值损失的系数"""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""
    """如果使用减值损失。"""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""
    """策略的裁剪参数。"""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """
    """无论是将每一批的优势正常化。
    默认是False。

    如果是True， 优势仅对小型批量正常化。
    否则，整个收集的轨道上将优势正常化。
    """

    rnd_cfg: RslRlRndCfg | None = None
    """The RND configuration. Default is None, in which case RND is not used."""
    """这就是RND配置。
    默认是None，在这种情况下不使用RND。
    """

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""
    """它们的对称性配置。
    默认是None，在这种情况下不使用对称。
    """


#########################
# Runner configurations #
#########################


@configclass
class RslRlBaseRunnerCfg:
    """Base configuration of the runner."""
    """跑步机的基本配置。"""

    seed: int = 42
    """The seed for the experiment. Default is 42."""
    """试验的种子。
    默认是42。
    """

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""
    """机器的Rl代理。
    默认是 cuda:0。
    """

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""
    """每个环境每次更新的步骤数。"""

    max_iterations: int = MISSING
    """The maximum number of iterations."""
    """最多的代数。"""

    empirical_normalization: bool | None = None
    """This parameter is deprecated and will be removed in the future.

    Use `actor_obs_normalization` and `critic_obs_normalization` instead.
    """
    """这一参数已过时，将来将被删除。

    而不是`actor_obs_normalization`和`critic_obs_normalization`。
    """

    obs_groups: dict[str, list[str]] = MISSING
    """A mapping from observation groups to observation sets.

    The keys of the dictionary are predefined observation sets used by the underlying algorithm
    and values are lists of observation groups provided by the environment.

    For instance, if the environment provides a dictionary of observations with groups "policy", "images",
    and "privileged", these can be mapped to algorithmic observation sets as follows:

    .. code-block:: python

        obs_groups = {
            "policy": ["policy", "images"],
            "critic": ["policy", "privileged"],
        }

    This way, the policy will receive the "policy" and "images" observations, and the critic will
    receive the "policy" and "privileged" observations.

    For more details, please check ``vec_env.py`` in the rsl_rl library.
    """
    """从观测组到观测组的映射。

    字典的关键是由底层算法使用的预定义观测集合，值是环境提供的观测组列表。

    例如，如果环境提供了"策略"，"图像"和"特权"组的观测字典，这些可以将其映射到如下算法观测集中:

    .. code-block:: python

        obs_groups = {
            "policy": ["policy", "images"],
            "critic": ["policy", "privileged"],
        }

    这样，策略将收到"策略"和"图像"的意见，批评者将收到"策略"和"特权"的意见。

    查看``vec_env.py``在rsl_rl库。
    """

    clip_actions: float | None = None
    """The clipping value for actions. If None, then no clipping is done. Defaults to None.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """
    """裁剪值为动作。
    如果是None，那么没有裁剪。
    默认为 None。

    .. 说明::
        这种裁剪是在:class:`RslRlVecEnvWrapper`包装内进行的。
    """

    save_interval: int = MISSING
    """The number of iterations between saves."""
    """保存之间的反复数量"""

    experiment_name: str = MISSING
    """The experiment name."""
    """实验的名字。"""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """
    """运行名称。
    默认是空的字符串。

    运行目录的名称通常是执行时的时间盖章。
    如果运行名字不空，则将其添加到运行目录的名字，i.e.记录目录的名字将成为``{time-stamp}_{run_name}``。
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""
    """树木砍伐机。
    默认是电压板。
    """

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""
    """尼普敦项目名称。
    默认是isaaclab。
    """

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""
    """这就是Wandb项目名称。
    默认是isaaclab。
    """

    resume: bool = False
    """Whether to resume a previous training. Default is False.

    This flag will be ignored for distillation.
    """
    """是否恢复以前的训练。
    默认是False。

    这支旗将被忽视为蒸。
    """

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """
    """运行目录要加载。
    默认是 ".*" (所有)。

    如果是regex表达式，将加载最新 (字母顺序) 的匹配运行。
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
    """检查站文件要加载。
    默认是``"model_.*.pt"`` (全部)。

    如果是regex表达式，将加载最新的 (字母顺序) 匹配文件。
    """


@configclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""
    """对策略上算法的运行器配置。"""

    class_name: str = "OnPolicyRunner"
    """The runner class name. Default is OnPolicyRunner."""
    """跑者类名字。
    默认是OnPolicyRunner。
    """

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""
    """策略配置。"""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""
    """算法配置。"""
