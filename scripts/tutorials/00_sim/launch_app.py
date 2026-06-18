# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run IsaacSim via the AppLauncher

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/launch_app.py

"""

"""Launch Isaac Sim Simulator first."""

'''
AppLauncher 是 Isaac Lab 对 Isaac Sim 的 SimulationApp 做的一层包装。
    SimulationApp = Isaac Sim 原生启动器
    AppLauncher = Isaac Lab 封装后的启动器
为什么要封装一层？
    因为直接使用 SimulationApp 时，你需要自己处理很多启动参数，比如：
        是否 headless
        是否 livestream
        是否启用摄像头
        窗口宽度
        窗口高度
        加载哪个 experience 文件
        日志是否 verbose
    而 AppLauncher 帮你把这些东西统一处理，并且支持：
        命令行参数 CLI
        环境变量 envars
        用户自定义参数
        Isaac Sim 启动参数
    简单来说就是，SimulationApp需要设置启动参数，没有默认启动参数，而AppLauncher去启动SimulationApp时自动的设置好了一些启动参数
'''
'''
CLI 是：Command Line Interface,命令行接口
'''


import argparse

from isaaclab.app import AppLauncher

'''
创建一个命令行参数解析器:让 Python 脚本可以读取终端里的参数。
description 是脚本说明，运行 --help 时会显示

定义一些用户参数，以下增加的这些自定义参数后续会变成 args_cli.size/width/height属性
'''
# create argparser
parser = argparse.ArgumentParser(description="Tutorial on running IsaacSim via the AppLauncher.")
parser.add_argument("--size", type=float, default=1.0, help="Side-length of cuboid")
# SimulationApp arguments https://docs.omniverse.nvidia.com/py/isaacsim/source/isaacsim.simulation_app/docs/index.html?highlight=simulationapp#isaacsim.simulation_app.SimulationApp
parser.add_argument(
    "--width", type=int, default=1280, help="Width of the viewport and generated images. Defaults to 1280"
)
parser.add_argument(
    "--height", type=int, default=720, help="Height of the viewport and generated images. Defaults to 720"
)

# append AppLauncher cli args       添加 AppLauncher 自带参数：把 AppLauncher 支持的启动参数添加到 parser 里
# 例如 --headless、--livestream、--enable_cameras、--verbose、--experience 等
AppLauncher.add_app_launcher_args(parser)
# parse the arguments   解析终端传入的所有参数,解析结果保存在 args_cli 中
args_cli = parser.parse_args()
'''
比如你运行：
    python launch_app.py --size 2.0 --width 1920 --height 1080 --headless
那么解析后：
    args_cli.size = 2.0
    args_cli.width = 1920
    args_cli.height = 1080
    args_cli.headless = True
'''
# launch omniverse app  AppLauncher 根据 args_cli 里的参数启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
# 从 AppLauncher 中取出真正的 SimulationApp 对象
# 后续用 simulation_app.is_running() 判断程序是否仍在运行
# 最后用 simulation_app.close() 关闭程序
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils


def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # spawn a cuboid
    cfg_cuboid = sim_utils.CuboidCfg(
        size=[args_cli.size] * 3,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    )
    # Spawn cuboid, altering translation on the z-axis to scale to its size
    cfg_cuboid.func("/World/Object", cfg_cuboid, translation=(0.0, 0.0, args_cli.size / 2))


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Design scene by adding assets to it
    design_scene()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
