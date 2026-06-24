# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates reference count of the robot view in Isaac Sim.

When we make a class instance, the reference count of the class instance should always be 1.
However, in this script, the reference count of the robot view is 2 after the class is created.
This causes a memory leak in the Isaac Sim simulator and the robot view is not garbage collected.

The issue is observed with torch 2.2 and Isaac Sim 4.0. It works fine with torch 2.0.1 and Isaac Sim 2023.1.
It can be resolved by uncommenting the line that creates a dummy tensor in the main function.

To reproduce the issue, run this script and check the reference count of the robot view.

For more details, please check: https://github.com/isaac-sim/IsaacLab/issues/639
"""
"""这本脚本显示了Isaac Sim中的机器人视图的参考数量。

当我们做一个类实例时，该类实例的参考数应该是1。
然而，在这个脚本中，在类创建后，机器人视图的参考数量为2。
这会导致Isaac Sim仿真器的内存泄漏，

这种问题在火2.2和Isaac Sim 4.0中被观测到。
它与火2.0.1和艾萨克西姆20231工作得很好。
它可以通过不评论在主函数中创建仿真子的行来解决。

为了复制问题，运行此脚本并检查机器人视图的参考数量。

更多详情请查看:https://github.com/isaac-sim/IsaacLab/issues/639
"""

"""Launch Isaac Sim Simulator first."""
"""首先发射艾萨克仿真器。"""


import contextlib

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim  # noqa: F401

from isaacsim import SimulationApp

# launch omniverse app
simulation_app = SimulationApp({"headless": True})

"""Rest everything follows."""
"""休息，一切都跟着。"""

import ctypes
import gc
import logging

import torch  # noqa: F401

import isaacsim.core.utils.nucleus as nucleus_utils
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import Articulation

# import logger
logger = logging.getLogger(__name__)


# check nucleus connection
if nucleus_utils.get_assets_root_path() is None:
    msg = (
        "Unable to perform Nucleus login on Omniverse. Assets root path is not set.\n"
        "\tPlease check: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html#omniverse-nucleus"
    )
    logger.error(msg)
    raise RuntimeError(msg)


ISAAC_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac"
"""Path to the `Isaac` directory on the NVIDIA Nucleus Server."""
"""在NVIDIA核服务器上的`Isaac`目录。"""

ISAACLAB_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
"""Path to the `Isaac/IsaacLab` directory on the NVIDIA Nucleus Server."""
"""在NVIDIA核服务器上的`Isaac/IsaacLab`目录。"""


"""
Classes
"""
"""类
"""


class AnymalArticulation:
    """Anymal articulation class."""
    """任何一个关节课程。"""

    def __init__(self):
        """Initialize the Anymal articulation class."""
        """启动Anymal的关节课程。"""
        # resolve asset
        usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
        # add asset
        print("Loading robot from: ", usd_path)
        prim_utils.create_prim("/World/Robot", usd_path=usd_path, translation=(0.0, 0.0, 0.6))

        # Resolve robot prim paths
        root_prim_path = "/World/Robot/base"
        # Setup robot
        self.view = Articulation(root_prim_path, name="ANYMAL")

    def __del__(self):
        """Delete the Anymal articulation class."""
        """删除 Anymal 关节类。"""
        print("Deleting the Anymal view.")
        self.view = None

    def initialize(self):
        """Initialize the Anymal view."""
        """启动Anymal视图。"""
        self.view.initialize()


"""
Main
"""
"""主要
"""


def main():
    """Spawns the ANYmal robot and clones it using Isaac Sim Cloner API."""
    """发育的ANYmal机器人使用Isaac Sim Cloner进行克隆API。"""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.005, rendering_dt=0.005, backend="torch", device="cuda:0")

    # Enable hydra scene-graph instancing
    # this is needed to visualize the scene when flatcache is enabled
    sim._settings.set_bool("/persistent/omnihydra/useSceneGraphInstancing", True)

    # Create a dummy tensor for testing
    # Uncommenting the following line will yield a reference count of 1 for the robot (as desired)
    # dummy_tensor = torch.zeros(1, device="cuda:0")

    # Robot
    robot = AnymalArticulation()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot))
    print("---" * 10)

    # Play the simulator
    sim.reset()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot))
    print("---" * 10)

    robot.initialize()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot))
    print("---" * 10)

    # Stop the simulator
    sim.stop()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot))
    print("---" * 10)

    # Clean up
    sim.clear()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot))
    print("---" * 10)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
