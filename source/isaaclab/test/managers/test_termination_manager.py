# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""
"""首先发射艾萨克仿真器。"""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""
"""休息，一切都跟着。"""

import pytest
import torch

from isaaclab.managers import TerminationManager, TerminationTermCfg
from isaaclab.sim import SimulationContext


class DummyEnv:
    """Minimal mutable env stub for the termination manager tests."""
    """终止管理器测试的最小可变env。"""

    def __init__(self, num_envs: int, device: str, sim: SimulationContext):
        self.num_envs = num_envs
        self.device = device
        self.sim = sim
        self.counter = 0  # mutable step counter used by test terms


def fail_every_5_steps(env) -> torch.Tensor:
    """Returns True for all envs when counter is a positive multiple of 5."""
    """返回所有envs的True，当计数是5的正倍数。"""
    cond = env.counter > 0 and (env.counter % 5 == 0)
    return torch.full((env.num_envs,), cond, dtype=torch.bool, device=env.device)


def fail_every_10_steps(env) -> torch.Tensor:
    """Returns True for all envs when counter is a positive multiple of 10."""
    """返回所有envs的True，当计数是10的正倍数。"""
    cond = env.counter > 0 and (env.counter % 10 == 0)
    return torch.full((env.num_envs,), cond, dtype=torch.bool, device=env.device)


def fail_every_3_steps(env) -> torch.Tensor:
    """Returns True for all envs when counter is a positive multiple of 3."""
    """返回所有envs的True，当计数是正倍数为3。"""
    cond = env.counter > 0 and (env.counter % 3 == 0)
    return torch.full((env.num_envs,), cond, dtype=torch.bool, device=env.device)


@pytest.fixture
def env():
    sim = SimulationContext()
    return DummyEnv(num_envs=20, device="cpu", sim=sim)


def test_initial_state_and_shapes(env):
    cfg = {
        "term_5": TerminationTermCfg(func=fail_every_5_steps),
        "term_10": TerminationTermCfg(func=fail_every_10_steps),
    }
    tm = TerminationManager(cfg, env)

    # Active term names
    assert tm.active_terms == ["term_5", "term_10"]

    # Internal buffers have expected shapes and start as all False
    assert tm._term_dones.shape == (env.num_envs, 2)
    assert tm._last_episode_dones.shape == (env.num_envs, 2)
    assert tm.dones.shape == (env.num_envs,)
    assert tm.time_outs.shape == (env.num_envs,)
    assert tm.terminated.shape == (env.num_envs,)
    assert torch.all(~tm._term_dones) and torch.all(~tm._last_episode_dones)


def test_term_transitions_and_persistence(env):
    """Concise transitions: single fire, persist, switch, both, persist.

    Uses 3-step and 5-step terms and verifies current-step values and last-episode persistence.
    """
    """简短的过渡:单次火，持续，切换，两者，持续。

    使用3步和5步项，验证当前步骤值和最后一集的持久性。
    """
    cfg = {
        "term_3": TerminationTermCfg(func=fail_every_3_steps, time_out=False),
        "term_5": TerminationTermCfg(func=fail_every_5_steps, time_out=False),
    }
    tm = TerminationManager(cfg, env)

    # step 3: only term_3 -> last_episode [True, False]
    env.counter = 3
    out = tm.compute()
    assert torch.all(tm.get_term("term_3")) and torch.all(~tm.get_term("term_5"))
    assert torch.all(out)
    assert torch.all(tm._last_episode_dones[:, 0]) and torch.all(~tm._last_episode_dones[:, 1])

    # step 4: none -> last_episode persists [True, False]
    env.counter = 4
    out = tm.compute()
    assert torch.all(~out)
    assert torch.all(~tm.get_term("term_3")) and torch.all(~tm.get_term("term_5"))
    assert torch.all(tm._last_episode_dones[:, 0]) and torch.all(~tm._last_episode_dones[:, 1])

    # step 5: only term_5 -> last_episode [False, True]
    env.counter = 5
    out = tm.compute()
    assert torch.all(~tm.get_term("term_3")) and torch.all(tm.get_term("term_5"))
    assert torch.all(out)
    assert torch.all(~tm._last_episode_dones[:, 0]) and torch.all(tm._last_episode_dones[:, 1])

    # step 15: both -> last_episode [True, True]
    env.counter = 15
    out = tm.compute()
    assert torch.all(tm.get_term("term_3")) and torch.all(tm.get_term("term_5"))
    assert torch.all(out)
    assert torch.all(tm._last_episode_dones[:, 0]) and torch.all(tm._last_episode_dones[:, 1])

    # step 16: none -> persist [True, True]
    env.counter = 16
    out = tm.compute()
    assert torch.all(~out)
    assert torch.all(~tm.get_term("term_3")) and torch.all(~tm.get_term("term_5"))
    assert torch.all(tm._last_episode_dones[:, 0]) and torch.all(tm._last_episode_dones[:, 1])


def test_time_out_vs_terminated_split(env):
    cfg = {
        "term_5": TerminationTermCfg(func=fail_every_5_steps, time_out=False),  # terminated
        "term_10": TerminationTermCfg(func=fail_every_10_steps, time_out=True),  # timeout
    }
    tm = TerminationManager(cfg, env)

    # Step 5: terminated fires, not timeout
    env.counter = 5
    out = tm.compute()
    assert torch.all(out)
    assert torch.all(tm.terminated) and torch.all(~tm.time_outs)

    # Step 10: both fire; timeout and terminated both True
    env.counter = 10
    out = tm.compute()
    assert torch.all(out)
    assert torch.all(tm.terminated) and torch.all(tm.time_outs)
