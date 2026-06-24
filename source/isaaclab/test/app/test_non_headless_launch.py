# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script checks if the app can be launched with non-headless app and start the simulation.
"""
"""这种脚本检查应用程序是否可以使用非无头应用程序启动并启动仿真。
"""

"""Launch Isaac Sim Simulator first."""
"""首先发射艾萨克仿真器。"""


import pytest

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(experience="isaaclab.python.kit", headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""
"""休息，一切都跟着。"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""
    """用机器人的传感器设计场景。"""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())


def run_simulator(
    sim: sim_utils.SimulationContext,
):
    """Run the simulator."""
    """运行仿真器。"""

    count = 0

    # Simulate physics
    while simulation_app.is_running() and count < 100:
        # perform step
        sim.step()
        count += 1


@pytest.mark.isaacsim_ci
def test_non_headless_launch():
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    print(scene)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim)
