# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for creating prims in Omniverse.

Spawners are used to create prims into Omniverse simulator. At their core, they are calling the
USD Python API or Omniverse Kit Commands to create prims. However, they also provide a convenient
interface for creating prims from their respective config classes.

There are two main ways of using the spawners:

1. Using the function from the module

   .. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    # spawn from USD file
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd")
    prim_path = "/World/myAsset"

    # spawn using the function from the module
    sim_utils.spawn_from_usd(prim_path, cfg)

2. Using the `func` reference in the config class

   .. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    # spawn from USD file
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd")
    prim_path = "/World/myAsset"

    # use the `func` reference in the config class
    cfg.func(prim_path, cfg)

For convenience, we recommend using the second approach, as it allows to easily change the config
class and the function call in a single line of code.

Depending on the type of prim, the spawning-functions can also deal with the creation of prims
over multiple prim path. These need to be provided as a regex prim path expressions, which are
resolved based on the parent prim paths using the :meth:`isaaclab.sim.utils.clone` function decorator.
For example:

* ``/World/Table_[1,2]/Robot`` will create the prims ``/World/Table_1/Robot`` and ``/World/Table_2/Robot``
  only if the parent prim ``/World/Table_1`` and ``/World/Table_2`` exist.
* ``/World/Robot_[1,2]`` will **NOT** create the prims ``/World/Robot_1`` and
  ``/World/Robot_2`` as the prim path expression can be resolved to multiple prims.

"""
"""包含在Omniverse中创建prims的工具的子模块。

它们可以将prims转化为全宇宙仿真器。
它们的核心是:USD字符串API或全宇宙套件命令创建prims。
然而，它们还提供了一个方便的接口来从各自的配置类中创建prims。

有两种主要方式使用子:

1. 使用模块的函数

   .. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    # spawn from USD file
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd")
    prim_path = "/World/myAsset"

    # spawn using the function from the module
    sim_utils.spawn_from_usd(prim_path, cfg)

2. 在配置类中使用`func`引用

   .. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    # spawn from USD file
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd")
    prim_path = "/World/myAsset"

    # use the `func` reference in the config class
    cfg.func(prim_path, cfg)

为方便，我们建议使用第二种方法，因为它可以轻松改变配置
class and the function call in a single line of code.

取决于prim的类型，生殖功能也可以处理在多个prim路径上创建prims。
这些必须作为一个regex prim路径表达式提供，这些表达式基于:meth:`isaaclab.sim.utils.clone`函数装饰器的母prim路径进行解决。
例如:

* 只有当母 prim ``/World/Table_1`` 和 ``/World/Table_2`` 存在时，``/World/Table_[1，2]/Robot`` 才会创建prims
  ``/World/Table_1/Robot`` 和 ``/World/Table_2/Robot``。
* ``/World/Robot_[1，2]``将**NOT**创建prims
  ``/World/Robot_1``和``/World/Robot_2``，因为prim路径表达式可以被解决为多个prims。
"""

from .from_files import *  # noqa: F401, F403
from .lights import *  # noqa: F401, F403
from .materials import *  # noqa: F401, F403
from .meshes import *  # noqa: F401, F403
from .sensors import *  # noqa: F401, F403
from .shapes import *  # noqa: F401, F403
from .spawner_cfg import *  # noqa: F401, F403
from .wrappers import *  # noqa: F401, F403
