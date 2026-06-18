# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""
'''
在 Isaac Lab / Isaac Sim 里，必须先启动仿真应用，再导入很多和仿真相关的模块。
'''


import argparse

from isaaclab.app import AppLauncher
'''
AppLauncher = Isaac Sim 的启动器
    它负责处理：
        是否打开图形界面
        是否 headless 无界面运行
        使用哪个设备
        是否启用摄像头
        是否加载必要扩展
        如何初始化 Omniverse App
'''

# create argparser      创建命令行参数解析器
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args       把 Isaac Lab / Isaac Sim 启动时常用的参数添加到 parser 里
AppLauncher.add_app_launcher_args(parser)
# parse the arguments   解析命令行参数
args_cli = parser.parse_args()
# launch omniverse app  根据刚才解析出来的命令行参数，创建 Isaac Sim 启动器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app   # app_launcher.app 就是真正启动好的 Isaac Sim / Omniverse 应用对象

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext
'''
导入 Isaac Lab 的仿真配置和仿真上下文
    SimulationCfg 是仿真配置类。
        它用来设置仿真的基本参数，比如：
            仿真步长 dt
            物理设备 device
            重力
            物理引擎参数
            渲染间隔
            是否使用 GPU pipeline
    SimulationContext 是 Isaac Lab 里管理仿真世界的核心对象。
        你可以把它理解为：
            仿真上下文 / 仿真管理器
        它负责：
            创建仿真世界
            控制仿真播放
            控制仿真暂停
            执行 step
            reset 仿真
            设置相机
            管理物理时间
'''

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)    # 创建仿真配置对象 sim_cfg，dt是物理仿真时间步长
    sim = SimulationContext(sim_cfg)    # 创建一个仿真上下文 sim（Isaac Lab 里的仿真世界管理器）
    # Set main camera   把相机放在世界坐标 [2.5, 2.5, 2.5] 的位置，让它看向世界坐标原点 [0.0, 0.0, 0.0]。
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    '''
    在 Isaac Sim 里，通常坐标系是：
        x 轴：前后方向
        y 轴：左右方向
        z 轴：上下方向
    '''
    

    # Play the simulator    初始化仿真世界
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():  # 判断 Isaac Sim 是否还在运行
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app 关闭 Isaac Sim 应用，释放资源
    simulation_app.close()
