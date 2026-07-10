# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
from collections.abc import Sequence
from typing import Any

import torch

import carb
from isaacsim.core.cloner import GridCloner
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObject,
    DeformableObjectCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
    SurfaceGripper,
    SurfaceGripperCfg,
)
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, SensorBase, SensorBaseCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage, get_current_stage_id
from isaaclab.sim.views import XformPrimView
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils.version import get_isaac_sim_version

# Note: This is a temporary import for the VisuoTactileSensorCfg class.
# It will be removed once the VisuoTactileSensor class is added to the core Isaac Lab framework.
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

from .interactive_scene_cfg import InteractiveSceneCfg

# import logger
logger = logging.getLogger(__name__)


class InteractiveScene:
    """A scene that contains entities added to the simulation.

    The interactive scene parses the :class:`InteractiveSceneCfg` class to create the scene.
    Based on the specified number of environments, it clones the entities and groups them into different
    categories (e.g., articulations, sensors, etc.).

    Cloning can be performed in two ways:

    * For tasks where all environments contain the same assets, a more performant cloning paradigm
      can be used to allow for faster environment creation. This is specified by the ``replicate_physics`` flag.

      .. code-block:: python

          scene = InteractiveScene(cfg=InteractiveSceneCfg(replicate_physics=True))

    * For tasks that require having separate assets in the environments, ``replicate_physics`` would have to
      be set to False, which will add some costs to the overall startup time.

      .. code-block:: python

          scene = InteractiveScene(cfg=InteractiveSceneCfg(replicate_physics=False))

    Each entity is registered to scene based on its name in the configuration class. For example, if the user
    specifies a robot in the configuration class as follows:

    .. code-block:: python

        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass

        from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


        @configclass
        class MySceneCfg(InteractiveSceneCfg):
            # ANYmal-C robot spawned in each environment
            robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    Then the robot can be accessed from the scene as follows:

    .. code-block:: python

        from isaaclab.scene import InteractiveScene

        # create 128 environments
        scene = InteractiveScene(cfg=MySceneCfg(num_envs=128))

        # access the robot from the scene
        robot = scene["robot"]
        # access the robot based on its type
        robot = scene.articulations["robot"]

    If the :class:`InteractiveSceneCfg` class does not include asset entities, the cloning process
    can still be triggered if assets were added to the stage outside of the :class:`InteractiveScene` class:

    .. code-block:: python

        scene = InteractiveScene(cfg=InteractiveSceneCfg(num_envs=128, replicate_physics=True))
        scene.clone_environments()

    .. note::
        It is important to note that the scene only performs common operations on the entities. For example,
        resetting the internal buffers, writing the buffers to the simulation and updating the buffers from the
        simulation. The scene does not perform any task specific to the entity. For example, it does not apply
        actions to the robot or compute observations from the robot. These tasks are handled by different
        modules called "managers" in the framework. Please refer to the :mod:`isaaclab.managers` sub-package
        for more details.
    """
    """一个包含实体添加到仿真的场景。

    交互场景分析:class:`InteractiveSceneCfg`类，创建场景。
    基于指定数量的环境，它克隆实体并将它们分为不同类别 (e.g.，关节，传感器等)。

    克隆可以通过两种方式进行:

    * 对于所有环境都含有相同的资产的任务，可以使用更高性能的克隆范式，以便更快地创建环境.``replicate_physics``标志。

      .. code-block:: python

          scene = InteractiveScene(cfg=InteractiveSceneCfg(replicate_physics=True))

    * 对于需要在环境中拥有单独的资产的任务，``replicate_physics``必须设置为False，这将增加一些成本。

      .. code-block:: python

          scene = InteractiveScene(cfg=InteractiveSceneCfg(replicate_physics=False))

    每个实体都根据其在配置类中的名称进行登记。
    例如，如果用户在配置类中指定以下机器人:

    .. code-block:: python

        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass

        from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


        @configclass
        class MySceneCfg(InteractiveSceneCfg):
            # ANYmal-C robot spawned in each environment
            robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    然后可以从场景访问机器人如下:

    .. code-block:: python

        from isaaclab.scene import InteractiveScene

        # create 128 environments
        scene = InteractiveScene(cfg=MySceneCfg(num_envs=128))

        # access the robot from the scene
        robot = scene["robot"]
        # access the robot based on its type
        robot = scene.articulations["robot"]

    如果:class:`InteractiveSceneCfg`类不包括资产实体，如果资产在:class:`InteractiveScene`类之外的阶段被添加，则仍然可以启动克隆过程:

    .. code-block:: python

        scene = InteractiveScene(cfg=InteractiveSceneCfg(num_envs=128, replicate_physics=True))
        scene.clone_environments()

    .. 说明::
        值得注意的是，场景只对实体进行共同操作。
        例如，重置内部缓冲器，将缓冲器写入仿真并更新仿真中的缓冲器。
        场景不执行对实体特定的任务。
        例如，它不适用于对机器人的操作或计算机器人的观测。
        这些任务由框架中的不同的模块称为"管理器"来处理。
        请参阅:mod:`isaaclab.managers`子包
        for more details.
    """
    '''
    仿真场景的实际建造者
        InteractiveSceneCfg 是图纸，InteractiveScene 是建筑队。
        它读取配置，在 Isaac Sim 中创建所有实体（机器人、地面、传感器），克隆出 N 个并行环境。

    一、核心职责一句话
        把 SceneCfg 翻译成 Isaac Sim 中的实际 3D 场景，提供 scene["robot"] 这样的字典式访问。

    二、两种克隆模式
            replicate_physics = True  → 高性能，所有环境共享 USD 资产
            replicate_physics = False → 灵活（每个环境可独立随机化），但创建慢
        和 InteractiveSceneCfg 中讲的一致。这里文档用两个 InteractiveScene(...) 调用清晰地展示了两者用法。

    三、访问场景实体的两种方式
            # 方式 1：按名字（通用）
            robot = scene["robot"]

            # 方式 2：按类型（有语法提示）
            robot = scene.articulations["robot"]
        配置中字段名就是实体的名字——robot = ANYMAL_C_CFG.replace(...) 对应 scene["robot"]。

    四、最重要的声明（docstring 第 108-114 行）
        场景只负责通用操作：创建实体、重置缓冲区、写入/读取 PhysX 数据。具体任务（控制机器人、算奖励）由 Manager 层处理。

        这就是之前那 8 个 Manager 和 InteractiveScene 的分工：
            InteractiveScene:   "哪些实体存在，怎么读写 PhysX"
            Manager 系统:       "怎么控制实体，怎么评估表现"
    五、InteractiveScene 和 InteractiveSceneCfg 的关系
        你写的:
        CartpoleSceneCfg(InteractiveSceneCfg)
            robot = ArticulationCfg(...)
            ground = AssetBaseCfg(...)
                        │
                        ▼
        框架构建:
        scene = InteractiveScene(cfg=CartpoleSceneCfg(num_envs=4096))
            → 创建 4096 套 Cartpole 环境
            → scene["robot"] = <Articulation 对象>
            → scene["ground"] = <RigidObject 对象>
                        │
                        ▼
        Manager 使用:
        action_manager 用 scene["robot"] 施加力矩
        observation_manager 用 scene["robot"].data.joint_pos 读取角度
    '''

    '''
    仿真场景的建造工程
        这是 IsaacLab 中场景创建的总施工流程。从配置到 4096 个复制好的并行环境，全在这里完成。
    '''
    def __init__(self, cfg: InteractiveSceneCfg):
        """Initializes the scene.

        Args:
            cfg: The configuration class for the scene.
        """
        """启动场景。

        参数：
            cfg: 场景的配置类。
        """
        '''
        整体施工流程
            __init__(cfg)
                │
                ├── 阶段 1: 配置校验 + 初始化实体容器
                │
                ├── 阶段 2: 准备环境克隆器 (GridCloner)
                │
                ├── 阶段 3: 首次克隆判断
                │     ├── replicate_physics=False → 先克隆坐标框架 → 记录 env_origins
                │     └── replicate_physics=True  → _default_env_origins = None (稍后克隆时计算)
                │
                └── 阶段 4: 添加实体 + 克隆
                    ├── _add_entities_from_cfg()   → 在源环境中创建所有实体
                    ├── if replicate_physics:
                    │     clone_environments()       → 克隆 N 份坐标框架
                    │     replicate_physics()        → replicate 物理状态
                    └── if 需要手动碰撞过滤 → filter_collisions()
        '''

        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        # initialize scene elements
        self._terrain = None                         # 地形（如果有）
        self._articulations = dict()                 # 机器人等关节体
        self._deformable_objects = dict()            # 软体物理（布料等）
        self._rigid_objects = dict()                 # 刚性物体（方块、球）
        self._rigid_object_collections = dict()      # 刚性物体集合
        self._sensors = dict()                       # 传感器（相机、激光雷达）
        self._surface_grippers = dict()              # 表面夹具
        self._extras = dict()                        # 灯光等非物理资产
        # get stage handle
        self.sim = SimulationContext.instance()     # 物理引擎句柄
        self.stage = get_current_stage()
        self.stage_id = get_current_stage_id()
        # physics scene path
        self._physics_scene_path = None
        # prepare cloner for environment replication
        self.cloner = GridCloner(spacing=self.cfg.env_spacing, stage=self.stage)    # GridCloner 是 Isaac Sim 的环境复制器——它知道所有环境应该排成网格，间距为 env_spacing
        self.cloner.define_base_env(self.env_ns)    # 定义"基环境"命名空间  env_ns 是 /World/envs
        self.env_prim_paths = self.cloner.generate_paths(f"{self.env_ns}/env", self.cfg.num_envs)   # 生成 4096 个路径
        '''
        self.env_prim_paths = ["/World/envs/env_0", "/World/envs/env_1", ..., "/World/envs/env_4095"]
        '''

        # create source prim        只创建第一个路径（env_prim_paths[0]）的 USD Prim——这是源环境，后续所有环境都从它复制。
        self.stage.DefinePrim(self.env_prim_paths[0], "Xform")      # 创建源环境坐标节点
        # allocate env indices
        self._ALL_INDICES = torch.arange(self.cfg.num_envs, dtype=torch.long, device=self.device)
        # when replicate_physics=False, we assume heterogeneous environments and clone the xforms first.
        # this triggers per-object level cloning in the spawner.
        if not self.cfg.replicate_physics:  # 路径 A：replicate_physics=False（灵活模式）
            # check version of Isaac Sim to determine whether clone_in_fabric is valid
            if get_isaac_sim_version().major < 5:
                # clone the env xform
                env_origins = self.cloner.clone(
                    source_prim_path=self.env_prim_paths[0],
                    prim_paths=self.env_prim_paths,
                    replicate_physics=False,    # ← 独立副本
                    copy_from_source=True,
                    enable_env_ids=(
                        self.cfg.filter_collisions if self.device != "cpu" else False
                    ),  # this won't do anything because we are not replicating physics
                )
            else:
                # clone the env xform
                env_origins = self.cloner.clone(
                    source_prim_path=self.env_prim_paths[0],
                    prim_paths=self.env_prim_paths,
                    replicate_physics=False,
                    copy_from_source=True,
                    enable_env_ids=(
                        self.cfg.filter_collisions if self.device != "cpu" else False
                    ),  # this won't do anything because we are not replicating physics
                    clone_in_fabric=self.cfg.clone_in_fabric,   # 新参数！
                )
            '''
            Isaac Sim 版本判断
                Isaac Sim 5.0 在 replicate_physics 中新增了 clone_in_fabric 参数。框架用版本判断兼容旧版 API——这是长期维护的大型项目中常见的向后兼容写法。
            '''
            '''
            为什么要先克隆一次？
                因为 replicate_physics=False 意味着每个环境有独立的 USD 资产副本。不先克隆坐标框架的话，后面添加实体时所有环境都共享同一个空间，会重叠在一起。
                先克隆坐标框架，每个环境就有独立的 "挂载点"，后续 _add_entities_from_cfg() 中实体会自动克隆到每个环境。
            '''

            # env_origins 被记录下来，用于重置时把环境移回原位。
            self._default_env_origins = torch.tensor(env_origins, device=self.device, dtype=torch.float32)
        else:   # replicate_physics=True（高性能模式）
            # otherwise, environment origins will be initialized during cloning at the end of environment creation
            self._default_env_origins = None
            '''
            不提前克隆。
                等实体全部添加到源环境后，一次性 clone_environments() + replicate_physics() 完成。
                这是默认路径，更快。
            '''

        self._global_prim_paths = list()
        if self._is_scene_setup_from_cfg():
            # add entities from config  # ① 从配置中创建所有实体（机器人、地面、传感器等）
            self._add_entities_from_cfg()
            # clone environments on a global scope if environment is homogeneous
            if self.cfg.replicate_physics:  # ② 如果是 replicate_physics 模式，克隆整个场景到 N 份
                self.clone_environments(copy_from_source=False)
            # replicate physics if we have more than one environment
            # this is done to make scene initialization faster at play time
            if self.cfg.replicate_physics and self.cfg.num_envs > 1:    # ③ 复制物理状态
                # check version of Isaac Sim to determine whether clone_in_fabric is valid
                if get_isaac_sim_version().major < 5:
                    self.cloner.replicate_physics(
                        source_prim_path=self.env_prim_paths[0],
                        prim_paths=self.env_prim_paths,
                        base_env_path=self.env_ns,
                        root_path=self.env_regex_ns.replace(".*", ""),
                        enable_env_ids=self.cfg.filter_collisions if self.device != "cpu" else False,
                    )
                else:
                    self.cloner.replicate_physics(
                        source_prim_path=self.env_prim_paths[0],
                        prim_paths=self.env_prim_paths,
                        base_env_path=self.env_ns,
                        root_path=self.env_regex_ns.replace(".*", ""),
                        enable_env_ids=self.cfg.filter_collisions if self.device != "cpu" else False,
                        clone_in_fabric=self.cfg.clone_in_fabric,
                    )
            '''
            顺序是关键：
                先在源环境中创建所有实体（只创建一份）
                用 GridCloner 把源环境复制 N 份到网格位置
                用 replicate_physics 让 PhysX 知道"这 N 份是同一个东西的副本，共享底层数据"
            '''

            # since env_ids is only applicable when replicating physics, we have to fallback to the previous method
            # to filter collisions if replicate_physics is not enabled
            # additionally, env_ids is only supported in GPU simulation
            if (not self.cfg.replicate_physics and self.cfg.filter_collisions) or self.device == "cpu":
                self.filter_collisions(self._global_prim_paths)
                '''
                replicate_physics 模式下，enable_env_ids 参数在物理复制时已经处理碰撞过滤。
                但如果不使用物理复制（replicate_physics=False）或者在 CPU 上运行，需要手动调用 filter_collisions() 来阻止相邻环境的物体碰撞。
                '''
            '''
            在 __init__ 中的两次克隆调用：
                # replicate_physics=False 模式下（阶段 3）：先克隆坐标框架
                self.cloner.clone(copy_from_source=True)    # 独立副本

                # replicate_physics=True 模式下（阶段 4）：克隆完整场景
                self.clone_environments(copy_from_source=False)  # 引用克隆
            '''

    '''
    从源环境复制出 N 个并行环境
        把源环境 env_0 复制到 env_1 ~ env_4095，并处理碰撞过滤和原点坐标记录。
    '''
    def clone_environments(self, copy_from_source: bool = False):
        """Creates clones of the environment ``/World/envs/env_0``.

        Args:
            copy_from_source: (bool): If set to False, clones inherit from /World/envs/env_0 and mirror its changes.
            If True, clones are independent copies of the source prim and won't reflect its changes (start-up time
            may increase). Defaults to False.
        """
        """创建了环境的克隆``/World/envs/env_0``。

        参数：
            copy_from_source: (bool): 如果设置为False，克隆将继承 /世界/envs/env_0，并反映其变化。
            如果True，克隆是源prim的独立副本，不会反映其变化 (启动时间可能会增加)。
            默认为 False。
        """
        '''
        执行流程
            clone_environments(copy_from_source=False)
                │
                ├── ① 检查 multi_assets + replicate_physics 冲突（警告）
                ├── ② 调用 GridCloner.clone() 复制环境
                │       Isaac Sim 版本 < 5 → 旧 API
                │       Isaac Sim 版本 ≥ 5 → 新 API（支持 clone_in_fabric）
                ├── ③ 非物理复制模式下的碰撞过滤提醒
                └── ④ 如果 _default_env_origins 还没填 → 记录
        '''

        # check if user spawned different assets in individual environments
        # this flag will be None if no multi asset is spawned
        carb_settings_iface = carb.settings.get_settings()
        has_multi_assets = carb_settings_iface.get("/isaaclab/spawn/multi_assets")
        if has_multi_assets and self.cfg.replicate_physics:
            logger.warning(
                "Varying assets might have been spawned under different environments."
                " However, the replicate physics flag is enabled in the 'InteractiveScene' configuration."
                " This may adversely affect PhysX parsing. We recommend disabling this property."
            )
        '''
        multi_assets 和 replicate_physics 冲突检查
            multi_assets 表示不同环境加载了不同的资产（如环境 0 是 Anymal-B，环境 1 是 Anymal-D）。
            但这和 replicate_physics=True（所有环境共享同一个底层物理结构）矛盾——PhysX 无法对异构环境做物理复制。
            只是一个警告，不阻止运行。
        '''

        # check version of Isaac Sim to determine whether clone_in_fabric is valid
        if get_isaac_sim_version().major < 5:
            # clone the environment
            env_origins = self.cloner.clone(
                source_prim_path=self.env_prim_paths[0],
                prim_paths=self.env_prim_paths,
                replicate_physics=self.cfg.replicate_physics,
                copy_from_source=copy_from_source,
                enable_env_ids=(
                    self.cfg.filter_collisions if self.device != "cpu" else False
                ),  # this automatically filters collisions between environments
            )
        else:
            # clone the environment
            env_origins = self.cloner.clone(
                source_prim_path=self.env_prim_paths[0],
                prim_paths=self.env_prim_paths,
                replicate_physics=self.cfg.replicate_physics,
                copy_from_source=copy_from_source,
                enable_env_ids=(
                    self.cfg.filter_collisions if self.device != "cpu" else False
                ),  # this automatically filters collisions between environments
                clone_in_fabric=self.cfg.clone_in_fabric,
            )

        # since env_ids is only applicable when replicating physics, we have to fallback to the previous method
        # to filter collisions if replicate_physics is not enabled
        # additionally, env_ids is only supported in GPU simulation
        if (not self.cfg.replicate_physics and self.cfg.filter_collisions) or self.device == "cpu":
            # if scene is specified through cfg, this is already taken care of
            if not self._is_scene_setup_from_cfg():
                logger.warning(
                    "Collision filtering can only be automatically enabled when replicate_physics=True and using GPU"
                    " simulation. Please call scene.filter_collisions(global_prim_paths) to filter collisions across"
                    " environments."
                )
                '''
                如果场景不是通过配置创建的（手动添加实体的场景），在非物理复制模式下的碰撞过滤无法自动触发，框架提醒需要手动调用 filter_collisions()。
                '''

        # in case of heterogeneous cloning, the env origins is specified at init
        if self._default_env_origins is None:
            self._default_env_origins = torch.tensor(env_origins, device=self.device, dtype=torch.float32)

    '''
    相邻环境的碰撞隔离
        当你把 4096 个机器人排成 64×64 的大网格时，环境 3 的机器人手臂不应该碰到环境 4 的桌子。
        这个方法在 PhysX 层面设置碰撞规则，阻止跨环境碰撞，同时保留与全局物体（如地面）的碰撞。
    '''
    def filter_collisions(self, global_prim_paths: list[str] | None = None):
        """Filter environments collisions.

        Disables collisions between the environments in ``/World/envs/env_.*`` and enables collisions with the prims
        in global prim paths (e.g. ground plane).

        Args:
            global_prim_paths: A list of global prim paths to enable collisions with.
                Defaults to None, in which case no global prim paths are considered.
        """
        """过环境碰撞。

        禁用``/World/envs/env_.*``中环境之间的碰撞，并使全球prim路径 (e.g.地面平面) 中与prims发生碰撞。

        参数：
            global_prim_paths: 一份全球prim路径列表，
                               默认对None的故障，在这种情况下，不考虑全球prim路径。
        """
        # validate paths in global prim paths
        if global_prim_paths is None:
            global_prim_paths = []
        else:
            # remove duplicates in paths
            global_prim_paths = list(set(global_prim_paths))    # 去重
        '''
        global_prim_paths — 例外名单
            全局 prim 是不属于任何单个环境的物体——比如大地（terrain），它是唯一且所有环境共享的。机器人需要和地面碰撞（否则就掉下去了），但不能和相邻环境的机器人碰撞。

            prim 类型	                            和环境关系	        碰撞规则
            "/World/ground"（地面）	                所有环境共享	    ✅ 允许碰撞
            "/World/envs/env_0/robot"（机器人）	    属于 env_0	        ❌ 和 env_1 的物体不碰撞
            "/World/envs/env_1/robot"（机器人）	    属于 env_1	        ❌ 和 env_0 的物体不碰撞
        '''

        # if "/World/collisions" already exists in the stage, we don't filter again
        if self.stage.GetPrimAtPath("/World/collisions"):
            return
        '''
        幂等检查 — 防止重复执行
            "/World/collisions" 是 PhysX 碰撞过滤 API 的标记节点。如果它已经存在，说明碰撞规则已经设置过了，直接返回。
            这个保护很重要——因为 __init__ 的 330 行和这里的调用顺序复杂，可能多次进入过滤逻辑。
            幂等检查避免了重复设置碰撞规则导致的性能问题。
        '''

        # set global prim paths list if not previously defined
        if len(self._global_prim_paths) < 1:
            self._global_prim_paths += global_prim_paths
        '''
        _global_prim_paths 的懒更新
            只在第一次调用时更新。后续调用（如手动重设碰撞规则）不会覆盖已记录的全局路径。
        '''

        # filter collisions within each environment instance
        self.cloner.filter_collisions(
            self.physics_scene_path,   # PhysX 场景根路径
            "/World/collisions",       # 碰撞规则存储路径
            self.env_prim_paths,       # 所有环境路径 ["/World/envs/env_0", "/World/envs/env_1", ...]
            global_paths=self._global_prim_paths,  # 全局例外（地面等）
        )
        '''
        GridCloner 知道 4096 个环境在网格中的排列方式，它在 PhysX 中设置规则：
            "同一个环境内的物体正常碰撞，不同环境的物体互相穿模，全局物体和所有人碰撞"。
            ┌─────────┬─────────┬─────────┐
            │ env_0   │ env_1   │ env_2   │
            │  ┌───┐  │  ┌───┐  │  ┌───┐  │
            │  │🤖 │  │  │🤖 │  │  │🤖 │  │
            │  └───┘  │  └───┘  │  └───┘  │
            │         │         │         │
            ├─────────┴─────────┴─────────┤
            │                               │
            │      🌍🌍🌍 共享地面 🌍🌍🌍     │  ← global_prim_paths
            │                               │
            └───────────────────────────────┘

            碰撞规则:
            env_0 的 🤖 ↔ env_1 的 🤖  → ❌ 禁止（不碰撞）
            env_0 的 🤖 ↔ env_0 的 🪑  → ✅ 允许（同环境）
            所有 🤖    ↔ 🌍 地面      → ✅ 允许（全局例外）
        '''

    def __str__(self) -> str:
        """Returns a string representation of the scene."""
        """返回场景的字符串表示。"""
        msg = f"<class {self.__class__.__name__}>\n"
        msg += f"\tNumber of environments: {self.cfg.num_envs}\n"
        msg += f"\tEnvironment spacing   : {self.cfg.env_spacing}\n"
        msg += f"\tSource prim name      : {self.env_prim_paths[0]}\n"
        msg += f"\tGlobal prim paths     : {self._global_prim_paths}\n"
        msg += f"\tReplicate physics     : {self.cfg.replicate_physics}"
        return msg
    '''
    输出示例
        <class InteractiveScene>
            Number of environments: 4096
            Environment spacing   : 4.0
            Source prim name      : /World/envs/env_0
            Global prim paths     : ['/World/ground']
            Replicate physics     : True
    '''

    """
    Properties.
    """
    """属性。
    """

    '''
    physics_scene_path 用懒加载模式查找 PhysX 场景的 USD 路径——第一次访问时 Traverse() 遍历场景树，找挂载了 PhysxSceneAPI 的 prim，缓存路径后返回；
    后续访问直接返回缓存。
    '''
    @property
    def physics_scene_path(self) -> str:
        """The path to the USD Physics Scene."""
        """进入USD物理场景的道路。"""
        if self._physics_scene_path is None:
            for prim in self.stage.Traverse():
                if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                    self._physics_scene_path = prim.GetPrimPath().pathString
                    logger.info(f"Physics scene prim path: {self._physics_scene_path}")
                    break
            if self._physics_scene_path is None:
                raise RuntimeError("No physics scene found! Please make sure one exists.")
        return self._physics_scene_path

    '''
    物理仿真最小步长，比如 sim.dt，不是 RL 环境步 step_dt。step_dt 通常是 physics_dt * decimation。
    '''
    @property
    def physics_dt(self) -> float:
        """The physics timestep of the scene."""
        """场景的物理时间。"""
        return sim_utils.SimulationContext.instance().get_physics_dt()  # pyright: ignore [reportOptionalMemberAccess]

    @property
    def device(self) -> str:
        """The device on which the scene is created."""
        """场景的装置。"""
        return sim_utils.SimulationContext.instance().device  # pyright: ignore [reportOptionalMemberAccess]

    @property
    def env_ns(self) -> str:
        """The namespace ``/World/envs`` in which all environments created.

        The environments are present w.r.t. this namespace under "env_{N}" prim,
        where N is a natural number.
        """
        """在所有环境中创建的名称空间``/World/envs``。

        环境存在 w.r.t。
        在"env_"下的这个名字空间{N}" prim，其中N是自然数。
        """
        return "/World/envs"

    @property
    def env_regex_ns(self) -> str:
        """The namespace ``/World/envs/env_.*`` in which all environments created."""
        """在所有环境中创建的名称空间``/World/envs/env_.*``。"""
        return f"{self.env_ns}/env_.*"
    '''
    例如机器人配置里常见：
        prim_path="{ENV_REGEX_NS}/Robot"
    '''

    @property
    def num_envs(self) -> int:
        """The number of environments handled by the scene."""
        """场景所处理的环境数量。"""
        return self.cfg.num_envs

    @property
    def env_origins(self) -> torch.Tensor:
        """The origins of the environments in the scene. Shape is (num_envs, 3)."""
        """场景环境的起源。
        形状是 (num_envs， 3)。
        """
        if self._terrain is not None:
            return self._terrain.env_origins
        else:
            return self._default_env_origins
    '''
    env_origins 是唯一有逻辑的属性——地形优先：
        如果有 terrain，环境原点由 terrain 决定。比如 rough terrain 每个 env 可能分布在不同地形块上，高度也可能不同，环境原点必须根据地形高度动态计算。
        如果没有 terrain，就用 cloner 创建环境时记录的默认网格原点 _default_env_origins。
    '''

    @property
    def terrain(self) -> TerrainImporter | None:
        """The terrain in the scene. If None, then the scene has no terrain.

        Note:
            We treat terrain separate from :attr:`extras` since terrains define environment origins and are
            handled differently from other miscellaneous entities.
        """
        """在场景的地形。
        如果是None，那么场景没有地形。

        说明：
            我们将地形与:attr:`extras`分开，因为地形定义了环境的起源，
        """
        return self._terrain
    '''
    地面/崎岖地形
    '''

    @property
    def articulations(self) -> dict[str, Articulation]:
        """A dictionary of articulations in the scene."""
        """在场景的关节字典。"""
        return self._articulations
    '''
    {"robot": AnymalC}
    '''

    @property
    def deformable_objects(self) -> dict[str, DeformableObject]:
        """A dictionary of deformable objects in the scene."""
        """场景中的可变物体的字典。"""
        return self._deformable_objects
    '''
    布料、软体
    '''

    @property
    def rigid_objects(self) -> dict[str, RigidObject]:
        """A dictionary of rigid objects in the scene."""
        """在场景中，一个ction固物体的字典。"""
        return self._rigid_objects
    '''
    {"cube": box}
    '''

    @property
    def rigid_object_collections(self) -> dict[str, RigidObjectCollection]:
        """A dictionary of rigid object collections in the scene."""
        """一个关于场景的硬物体的字典。"""
        return self._rigid_object_collections
    '''
    批量方块集合
    '''

    @property
    def sensors(self) -> dict[str, SensorBase]:
        """A dictionary of the sensors in the scene, such as cameras and contact reporters."""
        """场景传感器的字典，例如摄像头和联系记者。"""
        return self._sensors
    '''
    相机、激光雷达
    '''

    @property
    def surface_grippers(self) -> dict[str, SurfaceGripper]:
        """A dictionary of the surface grippers in the scene."""
        """一个处于场景的表面抓住器字典。"""
        return self._surface_grippers
    '''
    表面夹具
    '''

    @property
    def extras(self) -> dict[str, XformPrimView]:
        """A dictionary of miscellaneous simulation objects that neither inherit from assets nor sensors.

        The keys are the names of the miscellaneous objects, and the values are the
        :class:`~isaaclab.sim.views.XformPrimView` instances of the corresponding prims.

        As an example, lights or other props in the scene that do not have any attributes or properties that you
        want to alter at runtime can be added to this dictionary.

        Note:
            These are not reset or updated by the scene. They are mainly other prims that are not necessarily
            handled by the interactive scene, but are useful to be accessed by the user.

        """
        """一个由资产或传感器继承的各种仿真物体的字典。

        键是各种对象的名称，值是相应的prims的:class:`~isaaclab.sim.views.XformPrimView`实例。

        例如，在场景中没有任何你想要在运行时改变的属性或属性的灯光或其他道具可以添加到这个字典中。

        说明：
            这些都没有被场景重置或更新。
            它们主要是其他prims，它们不一定由交互场景处理，但用户可以访问它们。
        """
        return self._extras
    '''
    灯光等非物理资产
    '''

    @property
    def state(self) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """A dictionary of the state of the scene entities in the simulation world frame.

        Please refer to :meth:`get_state` for the format.
        """
        """仿真世界框架中的场景实体状态字典。

        请查看:meth:`get_state`的格式。
        """
        return self.get_state(is_relative=False)
    '''
    调用 get_state() 遍历所有实体收集当前物理状态（位置、速度等）。
        它返回当前场景实体在世界坐标系下的状态快照。get_state() 的格式大概是：
            {
                "articulation": {
                    "robot": {
                        "root_pose": ...,
                        "root_velocity": ...,
                        "joint_position": ...,
                        "joint_velocity": ...,
                    }
                },
                "rigid_object": {
                    "object": {
                        "root_pose": ...,
                        "root_velocity": ...,
                    }
                }
            }
        因为这里传的是 is_relative=False，所以位置是 world frame。如果传 True，get_state() 会减去 env_origins，变成每个环境局部坐标系下的状态。
    '''

    """
    Operations.
    """
    """操作。
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the scene entities.

        Args:
            env_ids: The indices of the environments to reset.
                Defaults to None (all instances).
        """
        """重置场景实体。

        参数：
            env_ids: 设置环境的索引。
                     在 None 中默认设置 (所有实例)。
        """
        # -- assets
        for articulation in self._articulations.values():
            articulation.reset(env_ids)
        for deformable_object in self._deformable_objects.values():
            deformable_object.reset(env_ids)
        for rigid_object in self._rigid_objects.values():
            rigid_object.reset(env_ids)
        for surface_gripper in self._surface_grippers.values():
            surface_gripper.reset(env_ids)
        for rigid_object_collection in self._rigid_object_collections.values():
            rigid_object_collection.reset(env_ids)
        # -- sensors
        for sensor in self._sensors.values():
            sensor.reset(env_ids)

    def write_data_to_sim(self):
        """Writes the data of the scene entities to the simulation."""
        """写出场景实体的数据。"""
        # -- assets
        for articulation in self._articulations.values():
            articulation.write_data_to_sim()
        for deformable_object in self._deformable_objects.values():
            deformable_object.write_data_to_sim()
        for rigid_object in self._rigid_objects.values():
            rigid_object.write_data_to_sim()
        for surface_gripper in self._surface_grippers.values():
            surface_gripper.write_data_to_sim()
        for rigid_object_collection in self._rigid_object_collections.values():
            rigid_object_collection.write_data_to_sim()

    def update(self, dt: float) -> None:
        """Update the scene entities.

        Args:
            dt: The amount of time passed from last :meth:`update` call.
        """
        """更新场景实体。

        参数：
            dt: 从最后一次:meth:`update`电话以来的时间。
        """
        # -- assets
        for articulation in self._articulations.values():
            articulation.update(dt)
        for deformable_object in self._deformable_objects.values():
            deformable_object.update(dt)
        for rigid_object in self._rigid_objects.values():
            rigid_object.update(dt)
        for rigid_object_collection in self._rigid_object_collections.values():
            rigid_object_collection.update(dt)
        for surface_gripper in self._surface_grippers.values():
            surface_gripper.update(dt)
        # -- sensors
        for sensor in self._sensors.values():
            sensor.update(dt, force_recompute=not self.cfg.lazy_sensor_update)

    """
    Operations: Scene State.
    """
    """Operations: 场景状态。
    """

    '''
    将场景恢复到指定状态快照
        和 reset()（恢复到默认初始状态）不同，reset_to() 将环境恢复到任意指定的状态。
        用于 checkpoint 恢复和 demo 回放。
    '''
    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None = None,
        is_relative: bool = False,
    ):
        """Resets the entities in the scene to the provided state.

        Args:
            state: The state to reset the scene entities to. Please refer to :meth:`get_state` for the format.
            env_ids: The indices of the environments to reset. Defaults to None, in which case
                all environment instances are reset.
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.
        """
        """将场景实体重置为所提供的状态。

        参数：
            state: 设置场景实体的状态。
                   请查看:meth:`get_state`的格式。
            env_ids: 设置环境的索引。
                     默认为 None，在这种情况下，所有环境实例都会重置。
            is_relative: 如果设置为True，状态将被视为与环境起源相对。
                         默认为 False。
        """
        '''
        执行流程
            reset_to(state, env_ids, is_relative)
                │
                ├── env_ids=None → self._ALL_INDICES (所有环境)
                │
                ├── 关节体 (Articulations): 恢复根位姿 + 关节状态 + 设置 PD 目标
                ├── 软体 (DeformableObjects): 恢复节点位置 + 速度
                ├── 刚体 (RigidObjects): 恢复根位姿 + 速度
                ├── 夹具 (SurfaceGrippers): 恢复夹具指令
                │
                └── write_data_to_sim() ← 推送到 PhysX
        '''
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # articulations
        for asset_name, articulation in self._articulations.items():
            asset_state = state["articulation"][asset_name]
            # root state
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self.env_origins[env_ids]   # 相对坐标 → 世界坐标
                '''
                状态可以以两种坐标系存储：
                    坐标系	    is_relative	    含义
                    相对坐标	True	        相对于每个环境自己的原点
                    世界坐标	False	        相对于整个仿真世界的原点
                训练中通常保存相对坐标（可泛化到不同环境排列），恢复时加上 env_origins 转为世界坐标写入 PhysX。
                '''
            root_velocity = asset_state["root_velocity"].clone()
            # ① 恢复根位姿（机器人底座在世界坐标系的位置/朝向）
            articulation.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            articulation.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
            # joint state
            joint_position = asset_state["joint_position"].clone()
            joint_velocity = asset_state["joint_velocity"].clone()
            # ② 恢复关节状态（每个关节的角度和角速度）
            articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
            # FIXME: This is not generic as it assumes PD control over the joints.
            #   This assumption does not hold for effort controlled joints.
            # ③ 设置 PD 控制器的目标（让驱动力把关节保持在恢复的位置）
            articulation.set_joint_position_target(joint_position, env_ids=env_ids)
            articulation.set_joint_velocity_target(joint_velocity, env_ids=env_ids)
            '''
            FIXME:
                这里假设所有关节都是 PD 位置控制（把目标位置设回恢复值，PD 控制器保持它）。
                但对于力矩控制（effort control）的关节，没有"目标位置"概念——不需要这一步。
                当前代码对所有关节统一设了 PD 目标，这是不完全通用的实现。
            '''
        # deformable objects
        for asset_name, deformable_object in self._deformable_objects.items():
            asset_state = state["deformable_object"][asset_name]
            nodal_position = asset_state["nodal_position"].clone()
            if is_relative:
                nodal_position[:, :3] += self.env_origins[env_ids]
            nodal_velocity = asset_state["nodal_velocity"].clone()
            # 软体: 恢复每个节点（vertex）的位置和速度
            deformable_object.write_nodal_pos_to_sim(nodal_position, env_ids=env_ids)
            deformable_object.write_nodal_velocity_to_sim(nodal_velocity, env_ids=env_ids)
        # rigid objects
        for asset_name, rigid_object in self._rigid_objects.items():
            asset_state = state["rigid_object"][asset_name]
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone()
            # 刚体: 只需恢复根位姿和速度（没有关节）
            rigid_object.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        # surface grippers
        for asset_name, surface_gripper in self._surface_grippers.items():
            asset_state = state["gripper"][asset_name]
            # 夹具不像其他实体有物理状态（位置/速度），它是逻辑状态（开/关），直接用设置指令恢复。
            surface_gripper.set_grippers_command(asset_state)

        # write data to simulation to make sure initial state is set
        # this propagates the joint targets to the simulation
        self.write_data_to_sim()
        '''
        前面的 write_root_pose_to_sim() 等方法只是把值写入了各实体的内部缓冲区（Python 侧），最后的 write_data_to_sim() 统一把所有缓冲推送到 PhysX C++ 引擎。
        不调这行的话，关节状态、根位姿都不会生效。
        '''
        '''
        和 reset() 的本质区别
                        reset()	                reset_to()
            恢复目标	配置中定义的默认初始状态	任意指定的状态（checkpoint）
            数据来源	各实体内部存储的默认值	    外部传入的 state 字典
            用途	    环境摔倒时重新开一局	    加载 checkpoint / 回放 demo
            传感器	    ✅ reset 缓冲区	        ❌ 不处理
        '''

    '''
    场景全状态快照
        reset_to() 的逆操作。读取当前仿真中所有实体的物理状态，打包成一个三层嵌套字典。
    '''
    def get_state(self, is_relative: bool = False) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """Returns the state of the scene entities.

        Based on the type of the entity, the state comprises of different components.

        * For an articulation, the state comprises of the root pose, root velocity, and joint position and velocity.
        * For a deformable object, the state comprises of the nodal position and velocity.
        * For a rigid object, the state comprises of the root pose and root velocity.

        The returned state is a dictionary with the following format:

        .. code-block:: python

            {
                "articulation": {
                    "entity_1_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                        "joint_position": torch.Tensor,
                        "joint_velocity": torch.Tensor,
                    },
                    "entity_2_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                        "joint_position": torch.Tensor,
                        "joint_velocity": torch.Tensor,
                    },
                },
                "deformable_object": {
                    "entity_3_name": {
                        "nodal_position": torch.Tensor,
                        "nodal_velocity": torch.Tensor,
                    }
                },
                "rigid_object": {
                    "entity_4_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                    }
                },
            }

        where ``entity_N_name`` is the name of the entity registered in the scene.

        Args:
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.

        Returns:
            A dictionary of the state of the scene entities.
        """
        """返回场景实体的状态。

        根据实体类型，该实体由不同的组成部分组成。

        * 对于关节，状态包括根姿势，根速度和关节位置和速度。
        * 对于可变形的物体，状态包括节点位置和速度。
        * 对于硬体，状态包括根姿势和根速度。

        返回的状态是以下格式的字典:

        .. code-block:: python

            {
                "articulation": {
                    "entity_1_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                        "joint_position": torch.Tensor,
                        "joint_velocity": torch.Tensor,
                    },
                    "entity_2_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                        "joint_position": torch.Tensor,
                        "joint_velocity": torch.Tensor,
                    },
                },
                "deformable_object": {
                    "entity_3_name": {
                        "nodal_position": torch.Tensor,
                        "nodal_velocity": torch.Tensor,
                    }
                },
                "rigid_object": {
                    "entity_4_name": {
                        "root_pose": torch.Tensor,
                        "root_velocity": torch.Tensor,
                    }
                },
            }

        在 ``entity_N_name`` 是场景注册的实体名称。

        参数：
            is_relative: 如果设置为True，状态将被视为与环境起源相对。
                         默认为 False。

        返回：
            一个关于场景实体的字典。
        """
        '''
        和 reset_to() 的对称关系
                get_state() → state 字典 → save to disk → load → reset_to(state)
                ↑ 快照                                                ↑ 恢复
            一对"保存-加载"操作，共享完全相同的字典格式。
        返回结构（三层嵌套）
            state = {
                "articulation": {                    ← 第 1 层: 实体类型
                    "robot": {                       ← 第 2 层: 实体名
                        "root_pose":     [N, 7],     ← 第 3 层: 具体数据
                        "root_velocity": [N, 6],
                        "joint_position": [N, 12],
                        "joint_velocity": [N, 12],
                    },
                },
                "rigid_object": {
                    "cube": {
                        "root_pose":     [N, 7],
                        "root_velocity": [N, 6],
                    },
                },
                "deformable_object": {...},
                "gripper": {...},
            }
        '''

        state = dict()
        # articulations
        state["articulation"] = dict()
        for asset_name, articulation in self._articulations.items():
            asset_state = dict()
            asset_state["root_pose"] = articulation.data.root_pose_w.clone()
            if is_relative:
                asset_state["root_pose"][:, :3] -= self.env_origins
            asset_state["root_velocity"] = articulation.data.root_vel_w.clone()
            asset_state["joint_position"] = articulation.data.joint_pos.clone()
            asset_state["joint_velocity"] = articulation.data.joint_vel.clone()
            state["articulation"][asset_name] = asset_state
        # deformable objects
        state["deformable_object"] = dict()
        for asset_name, deformable_object in self._deformable_objects.items():
            asset_state = dict()
            asset_state["nodal_position"] = deformable_object.data.nodal_pos_w.clone()
            if is_relative:
                asset_state["nodal_position"][:, :3] -= self.env_origins
            asset_state["nodal_velocity"] = deformable_object.data.nodal_vel_w.clone()
            state["deformable_object"][asset_name] = asset_state
        # rigid objects
        state["rigid_object"] = dict()
        for asset_name, rigid_object in self._rigid_objects.items():
            asset_state = dict()
            asset_state["root_pose"] = rigid_object.data.root_pose_w.clone()
            if is_relative:
                asset_state["root_pose"][:, :3] -= self.env_origins
            asset_state["root_velocity"] = rigid_object.data.root_vel_w.clone()
            state["rigid_object"][asset_name] = asset_state
        # surface grippers
        state["gripper"] = dict()
        for asset_name, gripper in self._surface_grippers.items():
            state["gripper"][asset_name] = gripper.state.clone()
        return state

    """
    Operations: Iteration.
    """
    """Operations: 代。
    """

    '''
    所有实体名的枚举，支持字典式访问
        这个方法让 InteractiveScene 可以像 dict 一样使用——scene["robot"]、"robot" in scene、for name in scene 都依赖它。
    '''
    def keys(self) -> list[str]:
        """Returns the keys of the scene entities.

        Returns:
            The keys of the scene entities.
        """
        """返回场景实体的钥匙。

        返回：
            场景实体的钥匙。
        """
        all_keys = ["terrain"]     # ← 地形排第一
        for asset_family in [
            self._articulations,           # 关节体
            self._deformable_objects,      # 软体
            self._rigid_objects,           # 刚体
            self._rigid_object_collections, # 物体集合
            self._sensors,                 # 传感器
            self._surface_grippers,        # 夹具
            self._extras,                  # 灯光等
        ]:
            all_keys += list(asset_family.keys())
        return all_keys
    '''
    支持的操作
        # ① 按下标访问（__getitem__ 内部用 keys() 检查存在性）
        robot = scene["robot"]

        # ② in 操作符
        if "robot" in scene: ...

        # ③ for 遍历
        for entity_name in scene: ...
    这些 Python 的字典式接口都依赖 __contains__ 和 __iter__，而它们通常委托给 keys()。
    '''

    '''
    scene["robot"] 的背后
        Python 的魔法方法——当你写 scene["robot"] 时，Python 自动调用 scene.__getitem__("robot")。
    '''
    def __getitem__(self, key: str) -> Any:
        """Returns the scene entity with the given key.

        Args:
            key: The key of the scene entity.

        Returns:
            The scene entity.
        """
        """返回场景实体，用给定的键。

        参数：
            key: 场景实体的钥匙。

        返回：
            场景实体。
        """
        '''
        查找顺序
            scene["robot"]
                │
                ├── ① "robot" == "terrain"？ → 返回 self._terrain
                │
                ├── ② 逐个查找 7 个容器:
                │     articulations → deformable_objects → rigid_objects
                │     → rigid_object_collections → sensors → surface_grippers → extras
                │
                └── ③ 都没找到 → raise KeyError（列出所有可用名字）
        '''
        # check if it is a terrain
        if key == "terrain":
            return self._terrain

        all_keys = ["terrain"]
        # check if it is in other dictionaries
        for asset_family in [
            self._articulations,
            self._deformable_objects,
            self._rigid_objects,
            self._rigid_object_collections,
            self._sensors,
            self._surface_grippers,
            self._extras,
        ]:
            out = asset_family.get(key)
            # if found, return
            if out is not None:
                return out
            all_keys += list(asset_family.keys())
        # if not found, raise error
        raise KeyError(f"Scene entity with key '{key}' not found. Available Entities: '{all_keys}'")
    '''
    为什么 terrain 单独处理？
        terrain 是 TerrainImporter | None，没有挂在 dict 容器里（其他实体都是 dict）。所以它不能通过 .get(key) 访问，只能硬编码判断。
    dict.get(key) 和 dict[key] 的区别：
        dict["nonexistent"]   # → KeyError！
        dict.get("nonexistent")  # → None（不报错）
        用 .get() 是因为不知道 key 在不在当前容器——尝试取，取到了就返回，取不到就换下一个容器。直到所有容器都试完才报错。
    '''

    """
    Internal methods.
    """
    """内部方法。
    """

    '''
    检查场景是否从配置创建
    '''
    def _is_scene_setup_from_cfg(self) -> bool:
        """Check if scene entities are setup from the config or not.

        Returns:
            True if scene entities are setup from the config, False otherwise.
        """
        """检查从配置中是否设置了场景实体。

        返回：
            如果从配置中设置了场景实体，则True，否则False。
        """
        return any(
            not (asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None)
            for asset_name, asset_cfg in self.cfg.__dict__.items()
        )
    '''
    问题的本质
        InteractiveSceneCfg 的子类可能有两种状态：
            # 状态 A: 用户添加了实体（正常情况）
            @configclass
            class MySceneCfg(InteractiveSceneCfg):
                robot = ArticulationCfg(...)   # ← 用户加的
                ground = AssetBaseCfg(...)     # ← 用户加的

            # 状态 B: 纯空壳（手动模式）
            scene = InteractiveScene(cfg=InteractiveSceneCfg(num_envs=128))
            # 没有 robot、ground 等字段，所有实体通过代码手动添加
        这个方法就是回答："配置里有没有用户定义的实体？"
    在 __init__ 中的影响
        # interactive_scene.py:298
        if self._is_scene_setup_from_cfg():
            self._add_entities_from_cfg()      # → 从配置创建实体
            if self.cfg.replicate_physics:
                self.clone_environments()      # → 克隆环境
        如果用户没在配置里加实体（手动模式），跳过这些自动步骤——用户自己负责创建和克隆。
    '''

    '''
    配置到实体的就工厂流水线
    '''
    def _add_entities_from_cfg(self):
        """Add scene entities from the config."""
        """在配置中添加场景实体。"""
        # store paths that are in global collision filter
        self._global_prim_paths = list()
        # parse the entire scene config and resolve regex
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            # skip keywords
            # note: easier than writing a list of keywords: [num_envs, env_spacing, lazy_sensor_update]
            # 基类字段过滤——跳过 num_envs、env_spacing 等 6 个基类字段
            if asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
                continue
            '''
            __dataclass_fields__ 是什么？
                @configclass（本质是 @dataclass）自动维护的类属性——记录了基类 InteractiveSceneCfg 声明了哪些字段：
                InteractiveSceneCfg.__dataclass_fields__
                # → {
                #     "num_envs": Field(...),
                #     "env_spacing": Field(...),
                #     "lazy_sensor_update": Field(...),
                #     "replicate_physics": Field(...),
                #     "filter_collisions": Field(...),
                #     "clone_in_fabric": Field(...),
                # }
                遍历 self.cfg.__dict__ 时，num_envs 在这个集合里 → 跳过。robot 不在 → 它是一个实体 → 返回 True。
            '''

            # resolve regex     命名空间替换
            if hasattr(asset_cfg, "prim_path"):
                asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                '''
                用户在配置中写 prim_path="{ENV_REGEX_NS}/Robot"，这里替换为 prim_path="/World/envs/env_.*/Robot"。

                # 替换前:
                    asset_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
                format() 的机制
                    "{ENV_REGEX_NS}/Robot".format(ENV_REGEX_NS="/World/envs/env_.*")
                    #       ↑ 这是一个占位符                            ↑ 用这个值替换
                    这是 Python 字符串的 str.format() 方法。{变量名} 在字符串中是占位符，format(变量名=值) 把它替换掉。
                '''

            # create asset
            if isinstance(asset_cfg, TerrainImporterCfg):
                # terrains are special entities since they define environment origins
                asset_cfg.num_envs = self.cfg.num_envs
                asset_cfg.env_spacing = self.cfg.env_spacing
                self._terrain = asset_cfg.class_type(asset_cfg)
                '''
                Terrain（地形）— 唯一不放在 dict 里的
                    地形特殊：直接存到 self._terrain（而非 self._xxx[asset_name]）。
                    地形定义了环境原点（env_origins），必须知道 num_envs 和 env_spacing 才能计算 N 个环境在 3D 空间中的位置。
                '''
            elif isinstance(asset_cfg, ArticulationCfg):    # 关节体（机器人）
                self._articulations[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, DeformableObjectCfg):    # 软体
                self._deformable_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, RigidObjectCfg): # 刚体（方块、球等）
                self._rigid_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, RigidObjectCollectionCfg):   # 刚体集合 — 批量物体
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    rigid_object_cfg.prim_path = rigid_object_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                self._rigid_object_collections[asset_name] = asset_cfg.class_type(asset_cfg)
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    if hasattr(rigid_object_cfg, "collision_group") and rigid_object_cfg.collision_group == -1:
                        asset_paths = sim_utils.find_matching_prim_paths(rigid_object_cfg.prim_path)
                        self._global_prim_paths += asset_paths
            elif isinstance(asset_cfg, SurfaceGripperCfg):  # 夹具
                # add surface grippers to scene
                self._surface_grippers[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, SensorBaseCfg):  # 传感器
                # Update target frame path(s)' regex name space for FrameTransformer
                # 传感器配置可能包含子引用指向其他 prim（如 FrameTransformer 的 target_frames），这些引用也需要命名空间替换：
                if isinstance(asset_cfg, FrameTransformerCfg):  # FrameTransformer: 更新所有 target_frames 的 prim_path
                    updated_target_frames = []
                    for target_frame in asset_cfg.target_frames:
                        target_frame.prim_path = target_frame.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                        updated_target_frames.append(target_frame)
                    asset_cfg.target_frames = updated_target_frames
                elif isinstance(asset_cfg, ContactSensorCfg):   # ContactSensor: 更新过滤路径
                    updated_filter_prim_paths_expr = []
                    for filter_prim_path in asset_cfg.filter_prim_paths_expr:
                        updated_filter_prim_paths_expr.append(filter_prim_path.format(ENV_REGEX_NS=self.env_regex_ns))
                    asset_cfg.filter_prim_paths_expr = updated_filter_prim_paths_expr
                elif isinstance(asset_cfg, VisuoTactileSensorCfg):  # VisuoTactileSensor: 更新 camera_cfg 和 contact_object 的 prim_path
                    if hasattr(asset_cfg, "camera_cfg") and asset_cfg.camera_cfg is not None:
                        asset_cfg.camera_cfg.prim_path = asset_cfg.camera_cfg.prim_path.format(
                            ENV_REGEX_NS=self.env_regex_ns
                        )
                    if (
                        hasattr(asset_cfg, "contact_object_prim_path_expr")
                        and asset_cfg.contact_object_prim_path_expr is not None
                    ):
                        asset_cfg.contact_object_prim_path_expr = asset_cfg.contact_object_prim_path_expr.format(
                            ENV_REGEX_NS=self.env_regex_ns
                        )

                self._sensors[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, AssetBaseCfg):   # 静态资产（AssetBaseCfg）— 直接生成，不存为实体对象
                # manually spawn asset
                if asset_cfg.spawn is not None:
                    asset_cfg.spawn.func(   # 立即创建 USD Prim
                        asset_cfg.prim_path,
                        asset_cfg.spawn,
                        translation=asset_cfg.init_state.pos,
                        orientation=asset_cfg.init_state.rot,
                    )
                # store xform prim view corresponding to this asset
                # all prims in the scene are Xform prims (i.e. have a transform component)
                self._extras[asset_name] = XformPrimView(asset_cfg.prim_path, device=self.device, stage=self.stage) # 存为 XformPrimView（轻量级引用，非物理实体）
                '''
                灯光等静态资产直接生成 USD Prim，不需要物理引擎管理。
                用 XformPrimView 而非 Articulation/RigidObject 存——因为无需读写 PhysX 数据，只需要一个轻量的场景引用。
                '''
            else:
                raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")
            # store global collision paths
            if hasattr(asset_cfg, "collision_group") and asset_cfg.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
                self._global_prim_paths += asset_paths
                '''
                collision_group == -1 在 Isaac Sim 中表示全局碰撞组——这些物体应该和所有环境的物体碰撞（如地面）。
                收集到 _global_prim_paths 中，后续传给 filter_collisions() 作为例外白名单。
                '''
