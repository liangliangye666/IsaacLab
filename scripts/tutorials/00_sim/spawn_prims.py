# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    '''
    在 Isaac Sim 里，所有东西都放在一个叫 Stage 的场景树里。你可以把它理解成：
        Stage = 整个仿真世界
        Prim = Stage 里的一个节点/对象
    比如：
        /World/defaultGroundPlane
        /World/lightDistant
        /World/Objects/Cone1
        /World/Objects/Table
        这些路径就是场景树里的对象路径。
    '''
    '''
    Isaac Lab 的标准生成方式：
        cfg_xxx.func("/World/xxx", cfg_xxx, translation=...)
            配置对象 cfg_xxx 里面包含一个 func
            这个 func 知道如何根据 cfg_xxx 在指定路径创建对象
    
    # Create a configuration class instance
    cfg = MyPrimCfg()
    prim_path = "/path/to/prim"

    # Spawn the prim into the scene using the corresponding spawner function
    spawn_my_prim(prim_path, cfg, translation=[0, 0, 0], orientation=[1, 0, 0, 0], scale=[1, 1, 1])
    # OR
    # Use the spawner function directly from the configuration class
    cfg.func(prim_path, cfg, translation=[0, 0, 0], orientation=[1, 0, 0, 0], scale=[1, 1, 1])
    '''
    
    # Ground-plane  # 1. 创建默认地面配置
    cfg_ground = sim_utils.GroundPlaneCfg()
    # 按照配置，在 /World/defaultGroundPlane 路径下生成地面
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    # 2. 创建远处平行光配置
    # intensity 表示光照强度
    # color 表示光的颜色，RGB=(0.75, 0.75, 0.75)，即灰白色
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    # 在 /World/lightDistant 路径生成灯光，并放在 (1, 0, 10)
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    # 3. 创建一个 Xform 父节点，用来统一管理后面生成的物体
    sim_utils.create_prim("/World/Objects", "Xform")
    '''
    什么是 Xform？
        Xform 可以理解成一个空的变换节点，类似文件夹。          简单来说就是分组
        它本身不一定显示几何形状，但可以作为其他对象的父节点。
        比如后面所有物体都放在：
            /World/Objects/...
        下面：
            /World
            ├── defaultGroundPlane
            ├── lightDistant
            └── Objects
                ├── Cone1
                ├── Cone2
                ├── ConeRigid
                ├── CuboidDeformable
                └── Table
        这样做的好处是：
            场景结构更清晰
            所有物体集中放在 /World/Objects 下
            以后可以整体移动、隐藏或管理 Objects 节点
    '''

    # spawn a red cone
    # 4. 创建红色圆锥配置
    # 半径 0.15m，高度 0.5m，颜色为红色
    cfg_cone = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),     # 设置显示材质
    )
    # 使用同一个红色圆锥配置，生成两个圆锥
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

    # spawn a green cone with colliders and rigid body
    # 5. 创建绿色刚体圆锥配置
    # 和普通圆锥相比，多了刚体属性、质量属性、碰撞属性
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),     # 表示这个圆锥具有刚体属性
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),   # 设置质量属性
        collision_props=sim_utils.CollisionPropertiesCfg(), # 表示给这个圆锥添加碰撞属性
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    # 生成绿色刚体圆锥
    # 初始位置在空中 z=2.0，运行后会受重力影响下落
    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
    )
    '''
    translation 表示位置，用xyz表示，单位：m
    orientation 表示姿态，用四元数(w, x, y, z)表示
    '''

    # spawn a blue cuboid with deformable body
    # 6. 创建蓝色可变形长方体配置
    # size 表示尺寸，deformable_props 表示它是可变形体
    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.5, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),   # 表示这个物体是可变形体
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),     # 可变形体的物理材料配置,没有传具体参数，所以使用默认可变形材料属性
    )
    # 生成蓝色可变形长方体
    cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))
    '''
    可变形体和刚体的区别是：
        刚体：形状不变，例如铁块、木块
        可变形体：形状可以变，例如橡胶、布料、软材料
    '''

    # spawn a usd file of a table into the scene
    # 7. 从 Isaac Nucleus 资源库中加载一张桌子的 USD 文件
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    # 把桌子生成到场景中
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))
    '''
    执行完这个函数后，场景树大概是：
        /World
        ├── defaultGroundPlane
        ├── lightDistant
        └── Objects
            ├── Cone1
            ├── Cone2
            ├── ConeRigid
            ├── CuboidDeformable
            └── Table
    其中：
        defaultGroundPlane：地面
        lightDistant：远处平行光
        Objects：物体父节点
        Cone1：红色普通圆锥
        Cone2：红色普通圆锥
        ConeRigid：绿色刚体圆锥
        CuboidDeformable：蓝色可变形长方体
        Table：从 USD 文件加载的桌子
    '''


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    # Design scene
    design_scene()
    '''
    所有的场景设计必须在仿真开始之前进行。一旦仿真开始，我们建议保持场景冻结，并仅更改基本物体的属性。
    这对于GPU仿真特别重要，因为在仿真过程中添加新的基本物体可能会改变GPU上的物理仿真缓冲区，并导致意外行为。
    '''

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
