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

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


@pytest.fixture
def sim():
    """Create a simulation context."""
    """创建一个仿真环境。"""
    sim_utils.create_new_stage()
    dt = 0.1
    sim = SimulationContext(SimulationCfg(dt=dt))
    sim_utils.update_stage()
    yield sim
    sim._disable_app_control_on_stop_handle = True  # prevent timeout
    sim.stop()
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


"""
Basic spawning.
"""
"""基本的繁殖。
"""


def test_spawn_cone(sim):
    """Test spawning of UsdGeom.Cone prim."""
    """测试UsdGeom.Coneprim的产卵。"""
    cfg = sim_utils.ConeCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Cone"
    assert prim.GetAttribute("radius").Get() == cfg.radius
    assert prim.GetAttribute("height").Get() == cfg.height
    assert prim.GetAttribute("axis").Get() == cfg.axis


def test_spawn_capsule(sim):
    """Test spawning of UsdGeom.Capsule prim."""
    """测试UsdGeom.Capsuleprim的产卵。"""
    cfg = sim_utils.CapsuleCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Capsule", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Capsule").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Capsule/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Capsule"
    assert prim.GetAttribute("radius").Get() == cfg.radius
    assert prim.GetAttribute("height").Get() == cfg.height
    assert prim.GetAttribute("axis").Get() == cfg.axis


def test_spawn_cylinder(sim):
    """Test spawning of UsdGeom.Cylinder prim."""
    """测试UsdGeom.Cylinderprim的产卵。"""
    cfg = sim_utils.CylinderCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Cylinder", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cylinder").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cylinder/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Cylinder"
    assert prim.GetAttribute("radius").Get() == cfg.radius
    assert prim.GetAttribute("height").Get() == cfg.height
    assert prim.GetAttribute("axis").Get() == cfg.axis


def test_spawn_cuboid(sim):
    """Test spawning of UsdGeom.Cube prim."""
    """测试UsdGeom.Cubeprim的产卵。"""
    cfg = sim_utils.CuboidCfg(size=(1.0, 2.0, 3.0))
    prim = cfg.func("/World/Cube", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cube").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cube/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Cube"
    assert prim.GetAttribute("size").Get() == min(cfg.size)


def test_spawn_sphere(sim):
    """Test spawning of UsdGeom.Sphere prim."""
    """测试UsdGeom.Sphereprim的产卵。"""
    cfg = sim_utils.SphereCfg(radius=1.0)
    prim = cfg.func("/World/Sphere", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Sphere").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Sphere/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Sphere"
    assert prim.GetAttribute("radius").Get() == cfg.radius


"""
Physics properties.
"""
"""物理特性。
"""


def test_spawn_cone_with_rigid_props(sim):
    """Test spawning of UsdGeom.Cone prim with rigid body API.

    Note:
        Playing the simulation in this case will give a warning that no mass is specified!
        Need to also setup mass and colliders.
    """
    """测试UsdGeom.Coneprim的胎，使用硬体API。

    说明：
        在这种情况下，玩仿真将给出警告，没有指定质量!
        需要安装质量和碰撞机。
    """
    cfg = sim_utils.ConeCfg(
        radius=1.0,
        height=2.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, solver_position_iteration_count=8, sleep_threshold=0.1
        ),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() == cfg.rigid_props.rigid_body_enabled
    assert (
        prim.GetAttribute("physxRigidBody:solverPositionIterationCount").Get()
        == cfg.rigid_props.solver_position_iteration_count
    )
    assert prim.GetAttribute("physxRigidBody:sleepThreshold").Get() == pytest.approx(cfg.rigid_props.sleep_threshold)


def test_spawn_cone_with_rigid_and_mass_props(sim):
    """Test spawning of UsdGeom.Cone prim with rigid body and mass API."""
    """测试UsdGeom.Cone prim的硬体和质量API产卵。"""
    cfg = sim_utils.ConeCfg(
        radius=1.0,
        height=2.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, solver_position_iteration_count=8, sleep_threshold=0.1
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("physics:mass").Get() == cfg.mass_props.mass

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


def test_spawn_cone_with_rigid_and_density_props(sim):
    """Test spawning of UsdGeom.Cone prim with rigid body and mass API.

    Note:
        In this case, we specify the density instead of the mass. In that case, physics need to know
        the collision shape to compute the mass. Thus, we have to set the collider properties. In
        order to not have a collision shape, we disable the collision.
    """
    """测试UsdGeom.Cone prim的硬体和质量API产卵。

    说明：
        在这种情况下，我们指定密度而不是质量。
        在这种情况下，物理学需要知道碰撞形状来计算质量。
        因此，我们必须设置碰撞机的特性。
        为了避免碰撞形状，我们将碰撞禁用。
    """
    cfg = sim_utils.ConeCfg(
        radius=1.0,
        height=2.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, solver_position_iteration_count=8, sleep_threshold=0.1
        ),
        mass_props=sim_utils.MassPropertiesCfg(density=10.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("physics:density").Get() == cfg.mass_props.density

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


def test_spawn_cone_with_all_props(sim):
    """Test spawning of UsdGeom.Cone prim with all properties."""
    """测试UsdGeom.Coneprim的所有特性。"""
    cfg = sim_utils.ConeCfg(
        radius=1.0,
        height=2.0,
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
        physics_material=sim_utils.materials.RigidBodyMaterialCfg(),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone/geometry/material").IsValid()
    # Check properties
    # -- rigid body properties
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() is True
    # -- collision properties
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
    assert prim.GetAttribute("physics:collisionEnabled").Get() is True

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


"""
Cloning.
"""
"""克隆。
"""


def test_spawn_cone_clones_invalid_paths(sim):
    """Test spawning of cone clones on invalid cloning paths."""
    """在不有效的克隆路径上测试 cl子克隆。"""
    num_clones = 10
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))
    # Spawn cone on invalid cloning path -- should raise an error
    cfg = sim_utils.ConeCfg(radius=1.0, height=2.0, copy_from_source=True)
    with pytest.raises(RuntimeError):
        cfg.func("/World/env/env_.*/Cone", cfg)


def test_spawn_cone_clones(sim):
    """Test spawning of cone clones."""
    """测试 cl子克隆的繁殖。"""
    num_clones = 10
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))
    # Spawn cone on valid cloning path
    cfg = sim_utils.ConeCfg(radius=1.0, height=2.0, copy_from_source=True)
    prim = cfg.func("/World/env_.*/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert str(prim.GetPath()) == "/World/env_0/Cone"
    # find matching prims
    prims = sim_utils.find_matching_prim_paths("/World/env_.*/Cone")
    assert len(prims) == num_clones


def test_spawn_cone_clone_with_all_props_global_material(sim):
    """Test spawning of cone clones with global material reference."""
    """测试使用全球材料参考的子克隆。"""
    num_clones = 10
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))
    # Spawn cone on valid cloning path
    cfg = sim_utils.ConeCfg(
        radius=1.0,
        height=2.0,
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
        physics_material=sim_utils.materials.RigidBodyMaterialCfg(),
        visual_material_path="/Looks/visualMaterial",
        physics_material_path="/Looks/physicsMaterial",
    )
    prim = cfg.func("/World/env_.*/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert str(prim.GetPath()) == "/World/env_0/Cone"
    # find matching prims
    prims = sim_utils.find_matching_prim_paths("/World/env_.*/Cone")
    assert len(prims) == num_clones
    # find matching material prims
    prims = sim_utils.find_matching_prim_paths("/Looks/visualMaterial.*")
    assert len(prims) == 1
