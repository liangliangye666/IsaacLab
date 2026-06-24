# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for different assets, such as rigid objects and articulations.

An asset is a physical object that can be spawned in the simulation. The class handles both
the spawning of the asset into the USD stage as well as initialization of necessary physics
handles to interact with the asset.

Upon construction of the asset instance, the prim corresponding to the asset is spawned into the
USD stage if the spawn configuration is not None. The spawn configuration is defined in the
:attr:`AssetBaseCfg.spawn` attribute. In case the configured :attr:`AssetBaseCfg.prim_path` is
an expression, then the prim is spawned at all the matching paths. Otherwise, a single prim is
spawned at the configured path. For more information on the spawn configuration, see the
:mod:`isaaclab.sim.spawners` module.

The asset class also registers callbacks for the stage play/stop events. These are used to
construct the physics handles for the asset as the physics engine is only available when the
stage is playing. Additionally, the class registers a callback for debug visualization of the
asset. This can be enabled by setting the :attr:`AssetBaseCfg.debug_vis` attribute to True.

The asset class follows the following naming convention for its methods:

* **set_xxx()**: These are used to only set the buffers into the :attr:`data` instance. However, they
  do not write the data into the simulator. The writing of data only happens when the
  :meth:`write_data_to_sim` method is called.
* **write_xxx_to_sim()**: These are used to set the buffers into the :attr:`data` instance and write
  the corresponding data into the simulator as well.
* **update(dt)**: These are used to update the buffers in the :attr:`data` instance. This should
  be called after a simulation step is performed.

The main reason to separate the ``set`` and ``write`` operations is to provide flexibility to the
user when they need to perform a post-processing operation of the buffers before applying them
into the simulator. A common example for this is dealing with explicit actuator models where the
specified joint targets are not directly applied to the simulator but are instead used to compute
the corresponding actuator torques.
"""
"""对于不同资产的子包装，如硬物体和关节。

一个资产是物理对象，可以在仿真中产生。
该类处理资产进入USD阶段的产卵，以及与资产交互的必要物理句柄的初始化。

在构建资产实例时，对资产相应的prim将产生到USD阶段，如果生成配置不是None。
在:attr:`AssetBaseCfg.spawn`属性中定义了产卵配置。
如果配置的:attr:`AssetBaseCfg.prim_path`是表达式，则在所有匹配的路径上产生prim。
否则，在配置的路径上产生单个prim。
查看:mod:`isaaclab.sim.spawners`模块。

资产类别还记录了舞台播放/停止事件的回调。
这些用于构建对资产的物理句柄，因为物理引擎只有在舞台上播放时才可用。
此外，该类还会记录一个调用回来，以便对资产进行调试视觉化。
这可以通过设置:attr:`AssetBaseCfg.debug_vis`属性为True来实现。

资产类别遵循以下命名规范:

* **set_xxx() **:这些用于只设置缓冲器在:attr:`data`实例中.然而，它们不会将数据写入仿真器中.只有在调用:meth:`write_data_to_sim`方法时才会写数据。
* **write_xxx_to_sim()**:它们用于将缓冲器设置在:attr:`data`实例中，并将相应的数据写入仿真器中。
* **update(dt) **:这些用于更新:attr:`data`实例中的缓冲器。

区分``set``和``write``操作的主要原因是为用户提供灵活性，以便在将缓冲器应用到仿真器中之前需要进行后处理操作。
这种情况常见的例子是明确的执行器模型，其中指定的联合目标不直接应用于仿真器，而是用于计算相应的执行器扭矩。
"""

from .articulation import Articulation, ArticulationCfg, ArticulationData
from .asset_base import AssetBase
from .asset_base_cfg import AssetBaseCfg
from .deformable_object import DeformableObject, DeformableObjectCfg, DeformableObjectData
from .rigid_object import RigidObject, RigidObjectCfg, RigidObjectData
from .rigid_object_collection import RigidObjectCollection, RigidObjectCollectionCfg, RigidObjectCollectionData
from .surface_gripper import SurfaceGripper, SurfaceGripperCfg
