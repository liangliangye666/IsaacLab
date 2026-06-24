# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demonstration of randomized bin-packing with Isaac Lab.

This script tiles multiple environments, spawns a configurable set of grocery
objects, and continuously randomizes their poses, velocities, mass properties,
and active/cached state to mimic a bin filling workflow. It showcases how to
use ``RigidObjectCollection`` utilities for bulk pose resets, cache management,
and out-of-bounds recovery inside an interactive simulation loop.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/bin_packing.py --num_envs 32

"""

from __future__ import annotations
"""随机垃圾包装与艾萨克实验室的示范。

这种脚本将多个环境构建，产生一个可配置的杂货对象集，并不断随机化它们的姿势，速度，质量属性和活性/缓存状态，以模仿垃圾填充工作流。
它展示了如何使用``RigidObjectCollection``公用程序进行批量重置姿势，缓存管理和在交互式仿真循环内进行非界限恢复。

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/bin_packing.py --num_envs 32
"""

"""Launch Isaac Sim Simulator first."""
"""首先发射艾萨克仿真器。"""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Demo usage of RigidObjectCollection through bin packing example")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
"""休息，一切都跟着。"""

import math

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import Timer, configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Scene Configuration
##

# Layout and spawn counts.
MAX_NUM_OBJECTS = 24  # Hard cap on objects managed per environment (active + cached).
MAX_OBJECTS_PER_BIN = 24  # Maximum active objects we plan to fit inside the bin.
MIN_OBJECTS_PER_BIN = 1  # Lower bound for randomized active object count.
NUM_OBJECTS_PER_LAYER = 4  # Number of groceries spawned on each layer of the active stack.

# Cached staging area and grid spacing.
CACHE_HEIGHT = 2.5  # Height (m) at which inactive groceries wait out of view.
ACTIVE_LAYER_SPACING = 0.1  # Vertical spacing (m) between layers inside the bin.
CACHE_SPACING = 0.25  # XY spacing (m) between cached groceries.

# Bin dimensions and bounds.
BIN_DIMENSIONS = (0.2, 0.3, 0.15)  # Physical size (m) of the storage bin.
BIN_XY_BOUND = ((-0.2, -0.3), (0.2, 0.3))  # Valid XY region (min/max) for active groceries.

# Randomization ranges (radians for rotations, m/s and rad/s for velocities).
POSE_RANGE = {"roll": (-3.14, 3.14), "pitch": (-3.14, 3.14), "yaw": (-3.14, 3.14)}
VELOCITY_RANGE = {"roll": (-0.2, 1.0), "pitch": (-0.2, 1.0), "yaw": (-0.2, 1.0)}

# Object layout configuration

GROCERIES = {
    "OBJECT_A": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
    ),
    "OBJECT_B": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
    ),
    "OBJECT_C": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
    ),
    "OBJECT_D": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
    ),
}


@configclass
class MultiObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a multi-object scene."""
    """为多个物体场景的配置。"""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # rigid object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            scale=(2.0, 2.0, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
    )

    groceries: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        # Instantiate four grocery variants per layer and replicate across all layers in each environment.
        rigid_objects={
            f"Object_{label}_Layer{layer}": RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Object_{label}_Layer{layer}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 0.2 + (layer) * 0.2)),
                spawn=GROCERIES.get(f"OBJECT_{label}"),
            )
            for layer in range(MAX_NUM_OBJECTS // NUM_OBJECTS_PER_LAYER)
            for label, (x, y) in zip(["A", "B", "C", "D"], [(-0.035, -0.1), (-0.035, 0.1), (0.035, 0.1), (0.035, -0.1)])
        }
    )


def reset_object_collections(
    scene: InteractiveScene, asset_name: str, view_states: torch.Tensor, view_ids: torch.Tensor, noise: bool = False
) -> None:
    """Apply states to a subset of a collection, with optional noise.

    Updates ``view_states`` in-place for ``view_ids`` and writes transforms/velocities
    to the PhysX view for the collection ``asset_name``. When ``noise`` is True, adds
    uniform perturbations to pose (XYZ + Euler) and velocities using ``POSE_RANGE`` and
    ``VELOCITY_RANGE``.

    Args:
        scene: Interactive scene containing the collection.
        asset_name: Key in the scene (e.g., ``"groceries"``) for the RigidObjectCollection.
        view_states: Flat tensor (N, 13) with [x, y, z, qx, qy, qz, qw, lin(3), ang(3)] in world frame.
        view_ids: 1D tensor of indices into ``view_states`` to update.
        noise: If True, apply pose and velocity noise before writing.

    Returns:
        None: This function updates ``view_states`` and the underlying PhysX view in-place.
    """
    """将状态应用于集合的子集，可选择噪音。

    更新``view_states``在``view_ids``的位置，并将转换/速度写入PhysX视图 ``asset_name``的集合。
    当``noise``是True时，将``POSE_RANGE``和``VELOCITY_RANGE``的速度和波音 (XYZ + Euler) 添加到均的 perturbations。

    参数：
        scene: 集中的交互场景。
        asset_name: 在场景的关键 (e.g.， ``"groceries"``) 为RigidObjectCollection。
        view_states: 在世界框架中，具有 [x， y， z， qx， qy， qz， qw， lin(3)， ang(3) 的平坦子。
        view_ids: 在 ``view_states`` 中进行更新。
        noise: 如果True，在写作前应对姿势和速度噪音。

    返回：
        None: 这个函数会更新``view_states``和底层PhysX视图。
    """
    rigid_object_collection: RigidObjectCollection = scene[asset_name]
    sel_view_states = view_states[view_ids]
    positions = sel_view_states[:, :3]
    orientations = sel_view_states[:, 3:7]
    # poses
    if noise:
        range_list = [POSE_RANGE.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=scene.device)
        samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(view_ids), 6), device=scene.device)
        positions += samples[..., 0:3]

        # Compose new orientations by applying the sampled euler noise in quaternion space.
        orientations_delta = math_utils.quat_from_euler_xyz(samples[..., 3], samples[..., 4], samples[..., 5])
        orientations = math_utils.convert_quat(orientations, to="wxyz")
        orientations = math_utils.quat_mul(orientations, orientations_delta)
        orientations = math_utils.convert_quat(orientations, to="xyzw")

    # velocities
    new_velocities = sel_view_states[:, 7:13]
    if noise:
        range_list = [VELOCITY_RANGE.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=scene.device)
        samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(view_ids), 6), device=scene.device)
        new_velocities += samples
    else:
        new_velocities[:] = 0.0

    view_states[view_ids, :7] = torch.concat((positions, orientations), dim=-1)
    view_states[view_ids, 7:] = new_velocities

    rigid_object_collection.root_physx_view.set_transforms(view_states[:, :7], indices=view_ids)
    rigid_object_collection.root_physx_view.set_velocities(view_states[:, 7:], indices=view_ids)


def build_grocery_defaults(
    num_envs: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create default active/cached spawn poses for all environments.

    - Active poses: stacked 3D grid over the bin with ``ACTIVE_LAYER_SPACING`` per layer.
    - Cached poses: 2D grid at ``CACHE_HEIGHT`` to park inactive objects out of view.

    Args:
        num_envs: Number of environments to tile the poses for.
        device: Torch device for allocation (e.g., ``"cuda:0"`` or ``"cpu"``).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Active and cached spawn poses, each shaped
        ``(num_envs, M, 7)`` with ``[x, y, z, qx, qy, qz, qw]`` where ``M`` equals
        ``MAX_NUM_OBJECTS``.
    """
    """为所有环境创建默认的活跃/缓存产生的姿势。

    - 活跃姿势:每层 ``ACTIVE_LAYER_SPACING``的3D格子堆叠在垃圾桶上。
    - 隐藏的姿势:在``CACHE_HEIGHT``的2D格子，将无事件物体置于视野之外。

    参数：
        num_envs: 设置的环境数量。
        device: 用于分配的火装置 (e.g.，``"cuda:0"``或``"cpu"``)。

    返回：
        tuple[torch.Tensor，torch.Tensor]: 活跃和缓存的生殖姿势，每个形状是``(num_envs， M， 7)``和``[x， y， z， qx， qy， qz，
        qw]``，其中``M``等于``MAX_NUM_OBJECTS``。
    """

    # The bin has a size of 0.2 x 0.3 x 0.15 m
    bin_x_dim, bin_y_dim, bin_z_dim = BIN_DIMENSIONS
    # First, we calculate the number of layers and objects per layer
    num_layers = math.ceil(MAX_OBJECTS_PER_BIN / NUM_OBJECTS_PER_LAYER)
    num_x_objects = math.ceil(math.sqrt(NUM_OBJECTS_PER_LAYER))
    num_y_objects = math.ceil(NUM_OBJECTS_PER_LAYER / num_x_objects)
    total_objects = num_x_objects * num_y_objects * num_layers
    # Then, we create a 3D grid that allows for IxJxN objects to be placed on top of the bin.
    x = torch.linspace(-bin_x_dim * (2 / 6), bin_x_dim * (2 / 6), num_x_objects, device=device)
    y = torch.linspace(-bin_y_dim * (2 / 6), bin_y_dim * (2 / 6), num_y_objects, device=device)
    z = torch.linspace(0, ACTIVE_LAYER_SPACING * (num_layers - 1), num_layers, device=device) + bin_z_dim * 2
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")  # Note Z first, this stacks the layers.
    # Using this grid plus a reference quaternion, create the poses for the groceries to be spawned above the bin.
    ref_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(total_objects, 1)
    positions = torch.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), dim=-1)
    poses = torch.cat((positions, ref_quat), dim=-1)
    # Duplicate across environments, cap at max_num_objects
    active_spawn_poses = poses.unsqueeze(0).repeat(num_envs, 1, 1)[:, :MAX_NUM_OBJECTS, :]

    # We'll also create a buffer for the cached groceries. They'll be spawned below the bin so they can't be seen.
    num_x_objects = math.ceil(math.sqrt(MAX_NUM_OBJECTS))
    num_y_objects = math.ceil(MAX_NUM_OBJECTS / num_x_objects)
    # We create a XY grid only and fix the Z height for the cache.
    x = CACHE_SPACING * torch.arange(num_x_objects, device=device)
    y = CACHE_SPACING * torch.arange(num_y_objects, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_z = CACHE_HEIGHT * torch.ones_like(grid_x)
    # We can then create the poses for the cached groceries.
    ref_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_x_objects * num_y_objects, 1)
    positions = torch.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), dim=-1)
    poses = torch.cat((positions, ref_quat), dim=-1)
    # Duplicate across environments, cap at max_num_objects
    cached_spawn_poses = poses.unsqueeze(0).repeat(num_envs, 1, 1)[:, :MAX_NUM_OBJECTS, :]

    return active_spawn_poses, cached_spawn_poses


##
# Simulation Loop
##


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """Runs the simulation loop that coordinates spawn randomization and stepping.

    Returns:
        None: The simulator side-effects are applied through ``scene`` and ``sim``.
    """
    """运行一个仿真循环，该循环协调生殖随机化和步骤。

    返回：
        None: 仿真器的副作用通过``scene``和``sim``进行应用。
    """
    # Extract scene entities
    # note: we only do this here for readability.
    groceries: RigidObjectCollection = scene["groceries"]
    num_objects = groceries.num_objects
    num_envs = scene.num_envs
    device = scene.device
    view_indices = torch.arange(num_envs * num_objects, device=device)
    default_state_w = groceries.data.default_object_state.clone()
    default_state_w[..., :3] = default_state_w[..., :3] + scene.env_origins.unsqueeze(1)
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Pre-compute canonical spawn poses for each object both inside the bin and in the cache.
    active_spawn_poses, cached_spawn_poses = build_grocery_defaults(num_envs, device)
    # Offset poses into each environment's world frame.
    active_spawn_poses[..., :3] += scene.env_origins.view(-1, 1, 3)
    cached_spawn_poses[..., :3] += scene.env_origins.view(-1, 1, 3)
    active_spawn_poses = groceries.reshape_data_to_view(active_spawn_poses)
    cached_spawn_poses = groceries.reshape_data_to_view(cached_spawn_poses)
    spawn_w = groceries.reshape_data_to_view(default_state_w).clone()

    groceries_mask_helper = torch.arange(num_objects * num_envs, device=device) % num_objects
    # Precompute a helper mask to toggle objects between active and cached sets.
    # Precompute XY bounds [[x_min,y_min],[x_max,y_max]]
    bounds_xy = torch.as_tensor(BIN_XY_BOUND, device=device, dtype=spawn_w.dtype)
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 250 == 0:
            # reset counter
            count = 0
            # Randomly choose how many groceries stay active in each environment.
            num_active_groceries = torch.randint(MIN_OBJECTS_PER_BIN, num_objects, (num_envs, 1), device=device)
            groceries_mask = (groceries_mask_helper.view(num_envs, -1) < num_active_groceries).view(-1, 1)
            spawn_w[:, :7] = cached_spawn_poses * (~groceries_mask) + active_spawn_poses * groceries_mask
            # Retrieve positions
            with Timer("[INFO] Time to reset scene: "):
                reset_object_collections(scene, "groceries", spawn_w, view_indices[~groceries_mask.view(-1)])
                reset_object_collections(scene, "groceries", spawn_w, view_indices[groceries_mask.view(-1)], noise=True)
                # Vary the mass and gravity settings so cached objects stay parked.
                random_masses = torch.rand(groceries.num_instances * num_objects, device=device) * 0.2 + 0.2
                groceries.root_physx_view.set_masses(random_masses.cpu(), view_indices.cpu())
                groceries.root_physx_view.set_disable_gravities((~groceries_mask).cpu(), indices=view_indices.cpu())
                scene.reset()

        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()

        # Bring out-of-bounds objects back to the bin in one pass.
        xy = groceries.reshape_data_to_view(groceries.data.object_pos_w - scene.env_origins.unsqueeze(1))[:, :2]
        out_bound = torch.nonzero(~((xy >= bounds_xy[0]) & (xy <= bounds_xy[1])).all(dim=1), as_tuple=False).flatten()
        if out_bound.numel():
            # Teleport stray objects back into the active stack to keep the bin tidy.
            reset_object_collections(scene, "groceries", spawn_w, out_bound)
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main() -> None:
    """Main function.

    Returns:
        None: The function drives the simulation for its side-effects.
    """
    """主要功能。

    返回：
        None: 函数为其副作用驱动仿真。
    """
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))

    # Design scene
    scene_cfg = MultiObjectSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0, replicate_physics=False)
    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
