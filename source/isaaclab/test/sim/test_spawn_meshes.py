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
    """Create a simulation context for testing."""
    """为测试创建仿真环境。"""
    # Create a new stage
    sim_utils.create_new_stage()
    # Simulation time-step
    dt = 0.1
    # Load kit helper
    sim = SimulationContext(SimulationCfg(dt=dt))
    # Wait for spawning
    sim_utils.update_stage()
    yield sim
    # Cleanup
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
    """Test spawning of UsdGeomMesh as a cone prim."""
    """测试UsdGeomMesh作为子prim的繁殖。"""
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


def test_spawn_capsule(sim):
    """Test spawning of UsdGeomMesh as a capsule prim."""
    """测试UsdGeomMesh作为prim囊。"""
    # Spawn capsule
    cfg = sim_utils.MeshCapsuleCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Capsule", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Capsule").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Capsule/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


def test_spawn_cylinder(sim):
    """Test spawning of UsdGeomMesh as a cylinder prim."""
    """测试UsdGeomMesh作为prim的产卵。"""
    # Spawn cylinder
    cfg = sim_utils.MeshCylinderCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Cylinder", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cylinder").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cylinder/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


def test_spawn_cuboid(sim):
    """Test spawning of UsdGeomMesh as a cuboid prim."""
    """测试UsdGeomMesh作为立体prim的产卵。"""
    # Spawn cuboid
    cfg = sim_utils.MeshCuboidCfg(size=(1.0, 2.0, 3.0))
    prim = cfg.func("/World/Cube", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cube").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cube/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


def test_spawn_sphere(sim):
    """Test spawning of UsdGeomMesh as a sphere prim."""
    """测试UsdGeomMesh作为球 prim的产卵。"""
    # Spawn sphere
    cfg = sim_utils.MeshSphereCfg(radius=1.0)
    prim = cfg.func("/World/Sphere", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Sphere").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Sphere/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


"""
Physics properties.
"""
"""物理特性。
"""


def test_spawn_cone_with_deformable_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with deformable body API."""
    """测试 UsdGeomMesh prim的子，用于具有变形体 API的子。"""
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=True),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()

    # Check properties
    # Unlike rigid body, deformable body properties are on the mesh prim
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
    assert prim.GetAttribute("physxDeformable:deformableEnabled").Get() == cfg.deformable_props.deformable_enabled


def test_spawn_cone_with_deformable_and_mass_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with deformable body and mass API."""
    """测试UsdGeomMesh prim的子，具有可变体和质量API的 spa子。"""
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
    assert prim.GetAttribute("physics:mass").Get() == cfg.mass_props.mass

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


def test_spawn_cone_with_deformable_and_density_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with deformable body and mass API.

    Note:
        In this case, we specify the density instead of the mass. In that case, physics need to know
        the collision shape to compute the mass. Thus, we have to set the collider properties. In
        order to not have a collision shape, we disable the collision.
    """
    """测试UsdGeomMesh prim的子，具有可变体和质量API的 spa子。

    说明：
        在这种情况下，我们指定密度而不是质量。
        在这种情况下，物理学需要知道碰撞形状来计算质量。
        因此，我们必须设置碰撞机的特性。
        为了避免碰撞形状，我们将碰撞禁用。
    """
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(density=10.0),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
    assert prim.GetAttribute("physics:density").Get() == cfg.mass_props.density
    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


def test_spawn_cone_with_all_deformable_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with all deformable properties."""
    """测试UsdGeomMesh prim的子，具有所有可变性质。"""
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
        physics_material=sim_utils.materials.DeformableBodyMaterialCfg(),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone/geometry/material").IsValid()
    # Check properties
    # -- deformable body
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
    assert prim.GetAttribute("physxDeformable:deformableEnabled").Get() is True

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


def test_spawn_cone_with_all_rigid_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with all rigid properties."""
    """测试UsdGeomMeshprim的子，用于具有所有硬性特性。"""
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, solver_position_iteration_count=8, sleep_threshold=0.1
        ),
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
    # -- rigid body
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() == cfg.rigid_props.rigid_body_enabled
    assert (
        prim.GetAttribute("physxRigidBody:solverPositionIterationCount").Get()
        == cfg.rigid_props.solver_position_iteration_count
    )
    assert prim.GetAttribute("physxRigidBody:sleepThreshold").Get() == pytest.approx(cfg.rigid_props.sleep_threshold)
    # -- mass
    assert prim.GetAttribute("physics:mass").Get() == cfg.mass_props.mass
    # -- collision shape
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
    assert prim.GetAttribute("physics:collisionEnabled").Get() is True

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()
