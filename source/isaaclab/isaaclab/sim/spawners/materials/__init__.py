# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn USD-based and PhysX-based materials.

`Materials`_ are used to define the appearance and physical properties of objects in the simulation.
In Omniverse, they are defined using NVIDIA's `Material Definition Language (MDL)`_. MDL is based on
the physically-based rendering (PBR) model, which is a set of equations that describe how light
interacts with a surface. The PBR model is used to create realistic-looking materials.

While MDL is primarily used for defining the appearance of objects, it can be extended to define
the physical properties of objects. For example, the friction and restitution coefficients of a
rubber material. A `physics material`_ can be assigned to a physics object to
define its physical properties.  There are different kinds of physics materials, such as rigid body
material, deformable material, and fluid material.

In order to apply a material to an object, we "bind" the geometry of the object to the material.
For this, we use the `USD Material Binding API`_. The material binding API takes in the path to
the geometry and the path to the material, and binds them together.

For physics material, the material is bound to the physics object with the 'physics' purpose.
When parsing physics material properties on an object, the following priority is used:

1. Material binding with a 'physics' purpose (physics material)
2. Material binding with no purpose (visual material)
3. Material binding with a 'physics' purpose on the `Physics Scene`_ prim.
4. Default values of material properties inside PhysX.

Usage:
    .. code-block:: python

        import isaaclab.sim as sim_utils

        # create a visual material
        visual_material_cfg = sim_utils.GlassMdlCfg(glass_ior=1.0, thin_walled=True)
        visual_material_cfg.func("/World/Looks/glassMaterial", visual_material_cfg)

        # create a mesh prim
        cube_cfg = sim_utils.CubeCfg(size=[1.0, 1.0, 1.0])
        cube_cfg.func("/World/Primitives/Cube", cube_cfg)

        # bind the cube to the visual material
        sim_utils.bind_visual_material("/World/Primitives/Cube", "/World/Looks/glassMaterial")


.. _Material Definition Language (MDL): https://raytracing-docs.nvidia.com/mdl/introduction/index.html#mdl_introduction#
.. _Materials: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html
.. _physics material: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.SimulationCfg.physics_material
.. _USD Material Binding API: https://openusd.org/dev/api/class_usd_shade_material_binding_a_p_i.html
.. _Physics Scene: https://openusd.org/dev/api/usd_physics_page_front.html
"""
"""对于产物产物以USD和PhysX为基础的材料的子模块。

`Materials`_用于定义仿真中的对象的外观和物理性质。
在全宇宙中，它们是用NVIDIA的`Material Definition Language (MDL)`_定义的。
MDL是基于基于物理的渲染 (PBR) 模型，这是一组描述光与表面相互作用的方程。
采用PBR模型来制造现实看似的材料。

虽然MDL主要用于定义对象的外观，但可以扩展到对象的物理特性。
例如，材料的摩擦和恢复系数。
可以将`physics material`_分配给物理对象来定义其物理特性。
物理材料有不同的类型，例如硬体材料，可变形材料和流体材料。

为了将物质应用到物体上，我们将物体的几何结构"绑定"到物体上。
我们使用`USD Material Binding API`_。
材料结合API接收到几何的路径和材料的路径，并将它们结合在一起。

对于物理材料，材料与"物理"目的的物理对象有关。
在对象的物理材料属性分析时，使用以下优先级:

1. "物理"目的的材料 (物理材料)
2. 无目的的结合材料 (视觉材料)
3. 在`Physics Scene`_prim上具有"物理"目的的材料结合。
4. 在PhysX内部的材料性能的默认值。

Usage:
    .. code-block:: python

        import isaaclab.sim as sim_utils

        # create a visual material
        visual_material_cfg = sim_utils.GlassMdlCfg(glass_ior=1.0, thin_walled=True)
        visual_material_cfg.func("/World/Looks/glassMaterial", visual_material_cfg)

        # create a mesh prim
        cube_cfg = sim_utils.CubeCfg(size=[1.0, 1.0, 1.0])
        cube_cfg.func("/World/Primitives/Cube", cube_cfg)

        # bind the cube to the visual material
        sim_utils.bind_visual_material("/World/Primitives/Cube", "/World/Looks/glassMaterial")


.. _Material Definition Language (MDL): https://raytracing-docs.nvidia.com/mdl/introduction/index.html#mdl_introduction#
.. _Materials: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html
.. _physics material: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.SimulationCfg.physics_material
.. _USD Material Binding API: https://openusd.org/dev/api/class_usd_shade_material_binding_a_p_i.html
.. _Physics Scene: https://openusd.org/dev/api/usd_physics_page_front.html
"""

from .physics_materials import spawn_deformable_body_material, spawn_rigid_body_material
from .physics_materials_cfg import DeformableBodyMaterialCfg, PhysicsMaterialCfg, RigidBodyMaterialCfg
from .visual_materials import spawn_from_mdl_file, spawn_preview_surface
from .visual_materials_cfg import GlassMdlCfg, MdlFileCfg, PreviewSurfaceCfg, VisualMaterialCfg
