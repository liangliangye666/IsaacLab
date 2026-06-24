# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

'''
Articulation 指的是由多个刚体通过关节连接起来的系统。

ImplicitActuatorCfg:一种常用执行器配置
ArticulationCfg：定义机器人本身
InteractiveSceneCfg：定义场景里放哪些东西
InteractiveScene：真正创建场景实例
ISAAC_NUCLEUS_DIR:表示 Isaac Sim 官方资产库的位置
'''

# 机器人配置说明书
JETBOT_CONFIG = ArticulationCfg(
    # 从一个 USD 文件加载机器人模型
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    '''
    创建一组执行器，名字叫：wheel_acts
    regular expression:正则表达式
    joint_names_expr=[".*"]:把这个机器人 USD 里的所有关节都分配给 wheel_acts 这组执行器。
    在隐式执行器中必须指定刚度和阻尼，但 None 值将使用 USD 资产中定义的默认值。
    '''
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Yahboom/Dofbot/dofbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(   # 刚体属性配置
            disable_gravity=False,  # 受到重力影响
            max_depenetration_velocity=5.0, # 物体从穿透状态中被推出时，允许的最大修正速度
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg( # 关节系统根属性配置:控制整个 articulation 的物理求解行为
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # 启用机器人自身各个 link 之间的碰撞检测
        # 位置约束求解迭代次数
        # 速度约束求解迭代次数
    ),
    '''
    机器人初始状态
        主要包括：
        机器人初始位置
        机器人初始姿态
        机器人初始关节角
        机器人初始速度
    '''
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={ # 初始关节角
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
        },
        pos=(0.25, -0.25, 0.0), # 机器人初始生成位置:这个位置是相对于环境原点，而不是世界原点
    ),
    actuators={
        "front_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=100.0,     # 仿真中的最大力矩或最大作用力限制
            velocity_limit_sim=100.0,   # 仿真中的关节速度限制
            stiffness=10000.0,          # 刚度
            damping=100.0,              # 阻尼
        ),
        "joint3_act": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint4_act": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)

'''
InteractiveSceneCfg:Isaac Lab 里的“场景配置模板”。
    它不是直接创建场景，而是先描述：
        这个场景里应该有哪些东西？
        这些东西放在哪里？
        这些东西从什么配置生成？
'''
class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""
    """设计场景。"""

    # Ground-plane  在 /World/defaultGroundPlane 位置生成一个默认地面。
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights    穹顶光配置
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    # 原来的 JETBOT_CONFIG 不变；新复制出来的一份配置使用新的 prim_path。
    '''
    .replace
        原来的 JETBOT_CONFIG 不变；新复制出来的一份配置使用新的 prim_path。

    ENV_REGEX_NS 可以理解为：每个并行环境的路径占位符。

    "{ENV_REGEX_NS}/Jetbot"
        最终会匹配每个环境下的 Jetbot，例如：
            /World/envs/env_0/Jetbot
            /World/envs/env_1/Jetbot
            /World/envs/env_2/Jetbot
            ...
    所以它不是只添加一个 Jetbot，而是：在每个环境里都添加一个 Jetbot。
    '''
    Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")


'''
SimulationContext:由仿真配置 SimulationCfg 创建出来的仿真上下文/仿真管理器
InteractiveScene：由场景配置 InteractiveSceneCfg 创建出来的交互式场景对象
'''
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()   # 获取物理仿真的时间步长
    sim_time = 0.0  # 当前仿真时间
    count = 0   # 仿真循环计数器

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:    # 每 500 步重置一次机器人
            # reset counters
            count = 0
            '''
            root state 是机器人根节点状态，一般包括：
                位置
                姿态
                线速度
                角速度
            常见格式是 13 维：
                [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            其中：
                前 3 个：位置
                接下来 4 个：四元数姿态
                后 6 个：线速度和角速度
            '''
            # reset the scene entities to their initial positions offset by the environment origins
            root_jetbot_state = scene["Jetbot"].data.default_root_state.clone()
            '''
            default_root_state 中的位置通常是相对于环境原点的。
            但是写入仿真时，需要世界坐标位置。
            机器人相对环境原点的位置 + 每个环境自己的世界原点 = 机器人最终世界坐标位置
            '''
            root_jetbot_state[:, :3] += scene.env_origins
            root_dofbot_state = scene["Dofbot"].data.default_root_state.clone()
            root_dofbot_state[:, :3] += scene.env_origins

            '''
            把 Jetbot 的根位姿写入仿真器:把 Jetbot 移动到初始位置，并设置初始朝向。
            把 Jetbot 根速度写入仿真器:重置 Jetbot 的速度
            '''
            # copy the default root state to the sim for the jetbot's orientation and velocity
            scene["Jetbot"].write_root_pose_to_sim(root_jetbot_state[:, :7])
            scene["Jetbot"].write_root_velocity_to_sim(root_jetbot_state[:, 7:])
            scene["Dofbot"].write_root_pose_to_sim(root_dofbot_state[:, :7])
            scene["Dofbot"].write_root_velocity_to_sim(root_dofbot_state[:, 7:])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["Jetbot"].data.default_joint_pos.clone(), # 默认关节位置
                scene["Jetbot"].data.default_joint_vel.clone(), # 默认关节速度
            )
            scene["Jetbot"].write_joint_state_to_sim(joint_pos, joint_vel)  # 机械臂关节重置到默认角度和默认速度
            joint_pos, joint_vel = (
                scene["Dofbot"].data.default_joint_pos.clone(),
                scene["Dofbot"].data.default_joint_vel.clone(),
            )
            scene["Dofbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()   # 清空或刷新 InteractiveScene 内部缓存。
            print("[INFO]: Resetting Jetbot and Dofbot state...")

        # drive around  前 75 步：直行,后 25 步：转弯
        if count % 100 < 75:
            # Drive straight by setting equal wheel velocities
            action = torch.Tensor([[10.0, 10.0]])
        else:
            # Turn by applying different velocities
            action = torch.Tensor([[5.0, -5.0]])

        scene["Jetbot"].set_joint_velocity_target(action)   # 设置轮速目标

        # wave
        # 给所有环境中的 Dofbot 的前 4 个关节设置同一个目标角度。
        wave_action = scene["Dofbot"].data.default_joint_pos
        wave_action[:, 0:4] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)
        scene["Dofbot"].set_joint_position_target(wave_action)  # 设置关节位置目标

        # 把控制目标写入仿真：把 scene 中缓存的控制目标、状态修改等数据，写入 Isaac Sim / PhysX 仿真器。
        scene.write_data_to_sim()
        sim.step()  # 推进物理仿真
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)  # 从仿真器读取最新状态到 scene 中的缓存


def main():
    """Main function."""
    # Initialize the simulation context
    # 1. 创建仿真配置
    # device=args_cli.device 表示使用命令行传入的设备，例如 cuda:0 或 cpu
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    # 2. 根据仿真配置创建仿真上下文
    # sim 负责 reset、step、获取 dt、设置相机等仿真控制
    sim = sim_utils.SimulationContext(sim_cfg)
    # 3. 设置主相机视角
    # 第一个列表是相机位置，第二个列表是相机看向的目标点
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    # 4. 创建场景配置
    # args_cli.num_envs 表示并行环境数量
    # env_spacing=2.0 表示每个环境之间相隔 2 米
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)   # 创建场景配置
    # 5. 根据场景配置真正生成 InteractiveScene
    # scene 里包含地面、灯光、Jetbot、Dofbot 等对象
    scene = InteractiveScene(scene_cfg) # 创建真正的交互式场景
    # Play the simulator
    # 6. 重置仿真器，让场景和物理引擎初始化完成
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    # 8. 进入仿真循环
    # 在 run_simulator 里会控制 Jetbot 和 Dofbot，并不断 sim.step()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
