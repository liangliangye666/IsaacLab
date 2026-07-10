# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.sim import SpawnerCfg
from isaaclab.utils import configclass

from .asset_base import AssetBase


@configclass
class AssetBaseCfg:
    """The base configuration class for an asset's parameters.

    Please see the :class:`AssetBase` class for more information on the asset class.
    """
    """资产参数的基础配置类。

    请查看:class:`AssetBase`类，了解资产类别的更多信息。
    """

    @configclass
    class InitialStateCfg:
        """Initial state of the asset.

        This defines the default initial state of the asset when it is spawned into the simulation, as
        well as the default state when the simulation is reset.

        After parsing the initial state, the asset class stores this information in the :attr:`data`
        attribute of the asset class. This can then be accessed by the user to modify the state of the asset
        during the simulation, for example, at resets.
        """
        """资产的初始状态。

        根据该指标，在仿真中产生的资产的默认初始状态，以及在仿真重置时的默认状态。

        在分析初始状态后，资产类别将这些信息存储在资产类别的:attr:`data`属性中。
        然后，用户可以访问该数据，以在仿真过程中修改资产状态，例如在重置时。
        """
        '''
        资产初始状态的配置类，它定义了一个资产刚"出生"在仿真世界里时站在哪里、朝向哪里。
        '''

        # root position
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Position of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        """仿真世界框架中的根位置。
        在 (0.0，0.0，0.0) 之前的默认设置。
        """
        '''
        │  ├─ 作用: 资产根节点在仿真世界坐标系中的位置
        │  ├─ 类型: 三元组 (x, y, z)，单位是米
        │  └─ 默认: 原点 (0, 0, 0)
        '''

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) of the root in simulation world frame.
        Defaults to (1.0, 0.0, 0.0, 0.0).
        """
        """在仿真世界框架中根的四元数旋转 (w，x，y，z)。
        在 (1.0，0.0，0.0，0.0，0.0的默认情况下。
        """
        '''
        │  ├─ 作用: 资产根节点在仿真世界坐标系中的四元数旋转
        │  ├─ 类型: 四元组 (w, x, y, z)
        │  └─ 默认: 单位四元数 (1, 0, 0, 0)，表示无旋转
        '''

    class_type: type[AssetBase] = None
    """The associated asset class. Defaults to None, which means that the asset will be spawned
    but cannot be interacted with via the asset class.

    The class should inherit from :class:`isaaclab.assets.asset_base.AssetBase`.
    """
    """相关资产类别。
    对None的默认值，这意味着资产将产生，但无法通过资产类别进行交互。

    这类应该继承:class:`isaaclab.assets.asset_base.AssetBase`。
    """
    '''
    指定"谁来管理这个资产"(Python 类，继承自 AssetBase)
    '''
    '''
    作用
        这是 Isaac Lab 框架的依赖注入机制。
        class_type 告诉框架："创建资产后，用哪个 Python 类来管理和操作它"。
        它是一个类型引用（type[AssetBase]），而不是类的实例。

        为什么这么设计？
            Isaac Lab 采用配置与行为分离的模式：
                配置层（AssetBaseCfg）：描述资产"长什么样"、"放哪里"
                行为层（AssetBase 子类）：描述资产"能做什么"
            class_type 就是连接两者的桥梁。框架读取配置后，通过 class_type 知道该实例化哪个类来管理这个资产。
    '''

    prim_path: str = MISSING
    """Prim path (or expression) to the asset.

    .. note::
        The expression can contain the environment namespace regex ``{ENV_REGEX_NS}`` which
        will be replaced with the environment namespace.

        Example: ``{ENV_REGEX_NS}/Robot`` will be replaced with ``/World/envs/env_.*/Robot``.
    """
    """资产的原始路径 (或表达)。

    .. 说明::
        这个表达式可以包含环境命名空间regex ``{ENV_REGEX_NS}``，将被环境命名空间取代。

        Example: ``{ENV_REGEX_NS}/Robot``将被 ``/World/envs/env_.*/Robot`` 取代。
    """
    '''
    指定"资产在仿真世界中的地址" (USD Stage 的路径字符串)
    '''

    spawn: SpawnerCfg | None = None
    """Spawn configuration for the asset. Defaults to None.

    If None, then no prims are spawned by the asset class. Instead, it is assumed that the
    asset is already present in the scene.
    """
    """产品的产品配置。
    默认为 None。

    如果None，则没有prims由资产类别产生。
    相反，假设该资产已经存在场景。
    """
    '''
    指定"如何把资产放到仿真正里" (SpawnerCfg，控制生成方式)
    '''
    '''
    作用
        指定如何将资产"生成"到仿真场景中。
        SpawnerCfg 定义了以下关键子字段（见 spawner_cfg.py:18-105）：
            子字段	                类型	                        作用
            func	                Callable[..., Usd.Prim]	        生成资产的具体函数，必填（MISSING）
            visible	                bool	                        生成后是否可见，默认 True
            semantic_tags	        list[tuple[str,str]]	        语义标签，用于图像分割等任务
            copy_from_source	    bool	                        是否从源 prim 拷贝（默认 True），与继承模式对立
        生成函数 func 是整个流程的核心——它接受 prim 路径、配置和变换矩阵，在 USD Stage 中创建出实际的几何体。

        为什么默认是 None？
            两种使用场景：
                spawn 有值：框架帮你在场景中创建资产（适用于大多数训练任务）。
                spawn = None：你自己提前在场景中放好了资产，框架不再生成（适用于自定义场景或手动搭建的环境）。
    '''

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose."""
    """硬体的初始状态。
    认同状态的默认问题。
    """
    '''
    指定"资产出生时的姿势和位置" (InitialStateCfg，pos + rot)
    '''

    collision_group: Literal[0, -1] = 0
    """Collision group of the asset. Defaults to ``0``.

    * ``-1``: global collision group (collides with all assets in the scene).
    * ``0``: local collision group (collides with other assets in the same environment).
    """
    """资产的碰撞组。
    在``0``上默认。

    * ``-1``:全球碰撞组 (与场景的所有资产发生碰撞)。
    * ``0``:局部碰撞组 (与同一个环境中的其他资产碰撞)。
    """
    '''
    指定"资产和谁发生碰撞" (0=只和同环境碰撞, -1=和全局碰撞)
    '''
    '''
    作用
        控制资产的碰撞过滤范围。这是强化学习仿真中的关键设计：
            0（默认）：
                只和自己所在环境的其他资产碰撞。Env_0 的机器人不会和 Env_1 的桌子碰撞——这正是我们想要的，因为每个环境是独立训练的。
            -1：
                和全局所有资产碰撞。通常用于地面（GroundPlane）——不管哪个环境，机器人只要踩在地上就应该产生碰撞。
    '''

    debug_vis: bool = False
    """Whether to enable debug visualization for the asset. Defaults to ``False``."""
    """是否启用对资产的调试可视化。
    在``False``上默认。
    """
    '''
    指定"要不要打开调试可视化" (true/false)
    作用
        控制是否显示资产的调试可视化信息。
        开启后，框架可能会渲染出碰撞体线框、关节轴、坐标架等调试元素，帮助你检查资产在仿真中的实际状态。
    '''
'''
示例 2：地面平面 — 全局碰撞的典型用法
cabinet_env_cfg.pyL109-L114
应用
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
'''