# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for environment definitions.

Environments define the interface between the agent and the simulation.
In the simplest case, the environment provides the agent with the current
observations and executes the actions provided by the agent. However, the
environment can also provide additional information such as the current
reward, done flag, and information about the current episode.

There are two types of environment designing workflows:

* **Manager-based**: The environment is decomposed into individual components (or managers)
  for different aspects (such as computing observations, applying actions, and applying
  randomization. The users mainly configure the managers and the environment coordinates the
  managers and calls their functions.
* **Direct**: The user implements all the necessary functionality directly into a single class
  directly without the need for additional managers.

Based on these workflows, there are the following environment classes for single and multi-agent RL:

**Single-Agent RL:**

* :class:`ManagerBasedEnv`: The manager-based workflow base environment which only provides the
  agent with the current observations and executes the actions provided by the agent.
* :class:`ManagerBasedRLEnv`: The manager-based workflow RL task environment which besides the
  functionality of the base environment also provides additional Markov Decision Process (MDP)
  related information such as the current reward, done flag, and information.
* :class:`DirectRLEnv`: The direct workflow RL task environment which provides implementations for
  implementing scene setup, computing dones, performing resets, and computing reward and observation.

**Multi-Agent RL (MARL):**

* :class:`DirectMARLEnv`: The direct workflow MARL task environment which provides implementations for
  implementing scene setup, computing dones, performing resets, and computing reward and observation.

For more information about the workflow design patterns, see the `Task Design Workflows`_ section.

.. _`Task Design Workflows`: https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/workflows.html
"""
"""环境定义的子包。

环境定义了代理和仿真之间的界面。
在最简单的情况下，环境向代理提供当前的观测，并执行代理提供的操作。
然而，环境也可以提供额外的信息，如当前的奖励，完成的旗和关于当前的事件的信息。

设计工作流的环境有两种类型:

* **基于管理器**:环境被分解成单个组件 (或管理器)
  for different aspects (such as computing observations, applying actions, and applying
  随机化。
  用户主要配置管理器，环境协调管理器，调用他们的功能。
* **直接**:用户直接将所有必要的功能直接实现在单一类中，而不需要额外的管理器。

基于这些工作流，单机和多代理RL的环境类型如下:

**单身代理 RL:**

* :class:`ManagerBasedEnv`:基于管理器的工作流基础环境，只为代理提供当前的观测，并执行代理提供的操作。
* :class:`ManagerBasedRLEnv`:基于管理器的工作流 RL任务环境，除了基础环境的功能外，还提供了有关马尔可夫决策过程 (MDP) 的额外信息，如当前的奖励，完成的标志和信息。
* :class:`DirectRLEnv`:直接工作流 RL任务环境，为实场景景设置，计算数据，执行重置以及计算奖励和观测提供了实现。

**多剂RL (MARL):**

* :class:`DirectMARLEnv`:直接工作流 MARL任务环境，为实场景景设置，计算数据，执行重置以及计算奖励和观测提供了实现。

更多关于工作流设计模式的信息，请参见`Task Design Workflows`_部分。

.. _`Task Design Workflows`: https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/workflows.html
"""

from . import mdp, ui
from .common import VecEnvObs, VecEnvStepReturn, ViewerCfg
from .direct_marl_env import DirectMARLEnv
from .direct_marl_env_cfg import DirectMARLEnvCfg
from .direct_rl_env import DirectRLEnv
from .direct_rl_env_cfg import DirectRLEnvCfg
from .manager_based_env import ManagerBasedEnv
from .manager_based_env_cfg import ManagerBasedEnvCfg
from .manager_based_rl_env import ManagerBasedRLEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from .manager_based_rl_mimic_env import ManagerBasedRLMimicEnv
from .mimic_env_cfg import *
from .utils.marl import multi_agent_to_single_agent, multi_agent_with_one_agent
