# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""
"""首先发射艾萨克仿真器。"""

from isaaclab.app import AppLauncher

# launch omniverse app
# need to set "enable_cameras" true to be able to do rendering tests
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""
"""休息，一切都跟着。"""

import pytest
import torch

import omni.usd

from isaaclab.envs import (
    DirectRLEnv,
    DirectRLEnvCfg,
    ManagerBasedEnv,
    ManagerBasedEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass


@configclass
class EmptyManagerCfg:
    """Empty specifications for the environment."""
    """环境的空格规格。"""

    pass


def create_manager_based_env(render_interval: int):
    """Create a manager based environment."""
    """建立一个基于管理器的环境。"""

    @configclass
    class EnvCfg(ManagerBasedEnvCfg):
        """Configuration for the test environment."""
        """对测试环境的配置。"""

        decimation: int = 4
        episode_length_s: float = 100.0
        sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=render_interval)
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)
        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyManagerCfg = EmptyManagerCfg()

    return ManagerBasedEnv(cfg=EnvCfg())


def create_manager_based_rl_env(render_interval: int):
    """Create a manager based RL environment."""
    """创建一个基于管理器的RL环境。"""

    @configclass
    class EnvCfg(ManagerBasedRLEnvCfg):
        """Configuration for the test environment."""
        """对测试环境的配置。"""

        decimation: int = 4
        episode_length_s: float = 100.0
        sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=render_interval)
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)
        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyManagerCfg = EmptyManagerCfg()
        rewards: EmptyManagerCfg = EmptyManagerCfg()
        terminations: EmptyManagerCfg = EmptyManagerCfg()

    return ManagerBasedRLEnv(cfg=EnvCfg())


def create_direct_rl_env(render_interval: int):
    """Create a direct RL environment."""
    """创建一个直接的RL环境。"""

    @configclass
    class EnvCfg(DirectRLEnvCfg):
        """Configuration for the test environment."""
        """对测试环境的配置。"""

        decimation: int = 4
        action_space: int = 0
        observation_space: int = 0
        episode_length_s: float = 100.0
        sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=render_interval)
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0)

    class Env(DirectRLEnv):
        """Test environment."""
        """测试环境。"""

        def _pre_physics_step(self, actions):
            pass

        def _apply_action(self):
            pass

        def _get_observations(self):
            return {}

        def _get_rewards(self):
            return {}

        def _get_dones(self):
            return torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)

    return Env(cfg=EnvCfg())


@pytest.fixture
def physics_callback():
    """Create a physics callback for tracking physics steps."""
    """创建一个物理回调来跟踪物理步骤。"""
    physics_time = 0.0
    num_physics_steps = 0

    def callback(dt):
        nonlocal physics_time, num_physics_steps
        physics_time += dt
        num_physics_steps += 1

    return callback, lambda: (physics_time, num_physics_steps)


@pytest.fixture
def render_callback():
    """Create a render callback for tracking render steps."""
    """创建一个转换回调来跟踪转换步骤。"""
    render_time = 0.0
    num_render_steps = 0

    def callback(event):
        nonlocal render_time, num_render_steps
        render_time += event.payload["dt"]
        num_render_steps += 1

    return callback, lambda: (render_time, num_render_steps)


@pytest.mark.parametrize("env_type", ["manager_based_env", "manager_based_rl_env", "direct_rl_env"])
@pytest.mark.parametrize("render_interval", [1, 2, 4, 8, 10])
def test_env_rendering_logic(env_type, render_interval, physics_callback, render_callback):
    """Test the rendering logic of the different environment workflows."""
    """测试不同环境工作流的 logic渲染逻辑。"""
    physics_cb, get_physics_stats = physics_callback
    render_cb, get_render_stats = render_callback

    # create a new stage
    omni.usd.get_context().new_stage()
    try:
        # create environment
        if env_type == "manager_based_env":
            env = create_manager_based_env(render_interval)
        elif env_type == "manager_based_rl_env":
            env = create_manager_based_rl_env(render_interval)
        else:
            env = create_direct_rl_env(render_interval)
    except Exception as e:
        if "env" in locals() and hasattr(env, "_is_closed"):
            env.close()
        else:
            if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                e.obj.close()
        pytest.fail(f"Failed to set-up the environment {env_type}. Error: {e}")

    # enable the flag to render the environment
    # note: this is only done for the unit testing to "fake" camera rendering.
    #   normally this is set to True when cameras are created.
    env.sim.set_setting("/isaaclab/render/rtx_sensors", True)

    # disable the app from shutting down when the environment is closed
    # FIXME: Why is this needed in this test but not in the other tests?
    #   Without it, the test will exit after the environment is closed
    env.sim._app_control_on_stop_handle = None  # type: ignore

    # check that we are in partial rendering mode for the environment
    # this is enabled due to app launcher setting "enable_cameras=True"
    assert env.sim.render_mode == SimulationContext.RenderMode.PARTIAL_RENDERING

    # add physics and render callbacks
    env.sim.add_physics_callback("physics_step", physics_cb)
    env.sim.add_render_callback("render_step", render_cb)

    # create a zero action tensor for stepping the environment
    actions = torch.zeros((env.num_envs, 0), device=env.device)

    # run the environment and check the rendering logic
    for i in range(50):
        # apply zero actions
        env.step(action=actions)

        # check that we have completed the correct number of physics steps
        _, num_physics_steps = get_physics_stats()
        assert num_physics_steps == (i + 1) * env.cfg.decimation, "Physics steps mismatch"
        # check that we have simulated physics for the correct amount of time
        physics_time, _ = get_physics_stats()
        assert abs(physics_time - num_physics_steps * env.cfg.sim.dt) < 1e-6, "Physics time mismatch"

        # check that we have completed the correct number of rendering steps
        _, num_render_steps = get_render_stats()
        assert num_render_steps == (i + 1) * env.cfg.decimation // env.cfg.sim.render_interval, "Render steps mismatch"
        # check that we have rendered for the correct amount of time
        render_time, _ = get_render_stats()
        assert abs(render_time - num_render_steps * env.cfg.sim.dt * env.cfg.sim.render_interval) < 1e-6, (
            "Render time mismatch"
        )

    # close the environment
    env.close()
