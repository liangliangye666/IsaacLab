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

    def __init__(self, cfg: InteractiveSceneCfg):
        """Initializes the scene.

        Args:
            cfg: The configuration class for the scene.
        """
        """启动场景。

        参数：
            cfg: 场景的配置类。
        """
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        # initialize scene elements
        self._terrain = None
        self._articulations = dict()
        self._deformable_objects = dict()
        self._rigid_objects = dict()
        self._rigid_object_collections = dict()
        self._sensors = dict()
        self._surface_grippers = dict()
        self._extras = dict()
        # get stage handle
        self.sim = SimulationContext.instance()
        self.stage = get_current_stage()
        self.stage_id = get_current_stage_id()
        # physics scene path
        self._physics_scene_path = None
        # prepare cloner for environment replication
        self.cloner = GridCloner(spacing=self.cfg.env_spacing, stage=self.stage)
        self.cloner.define_base_env(self.env_ns)
        self.env_prim_paths = self.cloner.generate_paths(f"{self.env_ns}/env", self.cfg.num_envs)
        # create source prim
        self.stage.DefinePrim(self.env_prim_paths[0], "Xform")
        # allocate env indices
        self._ALL_INDICES = torch.arange(self.cfg.num_envs, dtype=torch.long, device=self.device)
        # when replicate_physics=False, we assume heterogeneous environments and clone the xforms first.
        # this triggers per-object level cloning in the spawner.
        if not self.cfg.replicate_physics:
            # check version of Isaac Sim to determine whether clone_in_fabric is valid
            if get_isaac_sim_version().major < 5:
                # clone the env xform
                env_origins = self.cloner.clone(
                    source_prim_path=self.env_prim_paths[0],
                    prim_paths=self.env_prim_paths,
                    replicate_physics=False,
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
                    clone_in_fabric=self.cfg.clone_in_fabric,
                )
            self._default_env_origins = torch.tensor(env_origins, device=self.device, dtype=torch.float32)
        else:
            # otherwise, environment origins will be initialized during cloning at the end of environment creation
            self._default_env_origins = None

        self._global_prim_paths = list()
        if self._is_scene_setup_from_cfg():
            # add entities from config
            self._add_entities_from_cfg()
            # clone environments on a global scope if environment is homogeneous
            if self.cfg.replicate_physics:
                self.clone_environments(copy_from_source=False)
            # replicate physics if we have more than one environment
            # this is done to make scene initialization faster at play time
            if self.cfg.replicate_physics and self.cfg.num_envs > 1:
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

            # since env_ids is only applicable when replicating physics, we have to fallback to the previous method
            # to filter collisions if replicate_physics is not enabled
            # additionally, env_ids is only supported in GPU simulation
            if (not self.cfg.replicate_physics and self.cfg.filter_collisions) or self.device == "cpu":
                self.filter_collisions(self._global_prim_paths)

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

        # in case of heterogeneous cloning, the env origins is specified at init
        if self._default_env_origins is None:
            self._default_env_origins = torch.tensor(env_origins, device=self.device, dtype=torch.float32)

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
            global_prim_paths = list(set(global_prim_paths))

        # if "/World/collisions" already exists in the stage, we don't filter again
        if self.stage.GetPrimAtPath("/World/collisions"):
            return

        # set global prim paths list if not previously defined
        if len(self._global_prim_paths) < 1:
            self._global_prim_paths += global_prim_paths

        # filter collisions within each environment instance
        self.cloner.filter_collisions(
            self.physics_scene_path,
            "/World/collisions",
            self.env_prim_paths,
            global_paths=self._global_prim_paths,
        )

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

    """
    Properties.
    """
    """属性。
    """

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

    @property
    def articulations(self) -> dict[str, Articulation]:
        """A dictionary of articulations in the scene."""
        """在场景的关节字典。"""
        return self._articulations

    @property
    def deformable_objects(self) -> dict[str, DeformableObject]:
        """A dictionary of deformable objects in the scene."""
        """场景中的可变物体的字典。"""
        return self._deformable_objects

    @property
    def rigid_objects(self) -> dict[str, RigidObject]:
        """A dictionary of rigid objects in the scene."""
        """在场景中，一个ction固物体的字典。"""
        return self._rigid_objects

    @property
    def rigid_object_collections(self) -> dict[str, RigidObjectCollection]:
        """A dictionary of rigid object collections in the scene."""
        """一个关于场景的硬物体的字典。"""
        return self._rigid_object_collections

    @property
    def sensors(self) -> dict[str, SensorBase]:
        """A dictionary of the sensors in the scene, such as cameras and contact reporters."""
        """场景传感器的字典，例如摄像头和联系记者。"""
        return self._sensors

    @property
    def surface_grippers(self) -> dict[str, SurfaceGripper]:
        """A dictionary of the surface grippers in the scene."""
        """一个处于场景的表面抓住器字典。"""
        return self._surface_grippers

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

    @property
    def state(self) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """A dictionary of the state of the scene entities in the simulation world frame.

        Please refer to :meth:`get_state` for the format.
        """
        """仿真世界框架中的场景实体状态字典。

        请查看:meth:`get_state`的格式。
        """
        return self.get_state(is_relative=False)

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
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # articulations
        for asset_name, articulation in self._articulations.items():
            asset_state = state["articulation"][asset_name]
            # root state
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone()
            articulation.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            articulation.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
            # joint state
            joint_position = asset_state["joint_position"].clone()
            joint_velocity = asset_state["joint_velocity"].clone()
            articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
            # FIXME: This is not generic as it assumes PD control over the joints.
            #   This assumption does not hold for effort controlled joints.
            articulation.set_joint_position_target(joint_position, env_ids=env_ids)
            articulation.set_joint_velocity_target(joint_velocity, env_ids=env_ids)
        # deformable objects
        for asset_name, deformable_object in self._deformable_objects.items():
            asset_state = state["deformable_object"][asset_name]
            nodal_position = asset_state["nodal_position"].clone()
            if is_relative:
                nodal_position[:, :3] += self.env_origins[env_ids]
            nodal_velocity = asset_state["nodal_velocity"].clone()
            deformable_object.write_nodal_pos_to_sim(nodal_position, env_ids=env_ids)
            deformable_object.write_nodal_velocity_to_sim(nodal_velocity, env_ids=env_ids)
        # rigid objects
        for asset_name, rigid_object in self._rigid_objects.items():
            asset_state = state["rigid_object"][asset_name]
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone()
            rigid_object.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        # surface grippers
        for asset_name, surface_gripper in self._surface_grippers.items():
            asset_state = state["gripper"][asset_name]
            surface_gripper.set_grippers_command(asset_state)

        # write data to simulation to make sure initial state is set
        # this propagates the joint targets to the simulation
        self.write_data_to_sim()

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

    def keys(self) -> list[str]:
        """Returns the keys of the scene entities.

        Returns:
            The keys of the scene entities.
        """
        """返回场景实体的钥匙。

        返回：
            场景实体的钥匙。
        """
        all_keys = ["terrain"]
        for asset_family in [
            self._articulations,
            self._deformable_objects,
            self._rigid_objects,
            self._rigid_object_collections,
            self._sensors,
            self._surface_grippers,
            self._extras,
        ]:
            all_keys += list(asset_family.keys())
        return all_keys

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

    """
    Internal methods.
    """
    """内部方法。
    """

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

    def _add_entities_from_cfg(self):
        """Add scene entities from the config."""
        """在配置中添加场景实体。"""
        # store paths that are in global collision filter
        self._global_prim_paths = list()
        # parse the entire scene config and resolve regex
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            # skip keywords
            # note: easier than writing a list of keywords: [num_envs, env_spacing, lazy_sensor_update]
            if asset_name in InteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
                continue
            # resolve regex
            if hasattr(asset_cfg, "prim_path"):
                asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
            # create asset
            if isinstance(asset_cfg, TerrainImporterCfg):
                # terrains are special entities since they define environment origins
                asset_cfg.num_envs = self.cfg.num_envs
                asset_cfg.env_spacing = self.cfg.env_spacing
                self._terrain = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, ArticulationCfg):
                self._articulations[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, DeformableObjectCfg):
                self._deformable_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, RigidObjectCfg):
                self._rigid_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, RigidObjectCollectionCfg):
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    rigid_object_cfg.prim_path = rigid_object_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                self._rigid_object_collections[asset_name] = asset_cfg.class_type(asset_cfg)
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    if hasattr(rigid_object_cfg, "collision_group") and rigid_object_cfg.collision_group == -1:
                        asset_paths = sim_utils.find_matching_prim_paths(rigid_object_cfg.prim_path)
                        self._global_prim_paths += asset_paths
            elif isinstance(asset_cfg, SurfaceGripperCfg):
                # add surface grippers to scene
                self._surface_grippers[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, SensorBaseCfg):
                # Update target frame path(s)' regex name space for FrameTransformer
                if isinstance(asset_cfg, FrameTransformerCfg):
                    updated_target_frames = []
                    for target_frame in asset_cfg.target_frames:
                        target_frame.prim_path = target_frame.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                        updated_target_frames.append(target_frame)
                    asset_cfg.target_frames = updated_target_frames
                elif isinstance(asset_cfg, ContactSensorCfg):
                    updated_filter_prim_paths_expr = []
                    for filter_prim_path in asset_cfg.filter_prim_paths_expr:
                        updated_filter_prim_paths_expr.append(filter_prim_path.format(ENV_REGEX_NS=self.env_regex_ns))
                    asset_cfg.filter_prim_paths_expr = updated_filter_prim_paths_expr
                elif isinstance(asset_cfg, VisuoTactileSensorCfg):
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
            elif isinstance(asset_cfg, AssetBaseCfg):
                # manually spawn asset
                if asset_cfg.spawn is not None:
                    asset_cfg.spawn.func(
                        asset_cfg.prim_path,
                        asset_cfg.spawn,
                        translation=asset_cfg.init_state.pos,
                        orientation=asset_cfg.init_state.rot,
                    )
                # store xform prim view corresponding to this asset
                # all prims in the scene are Xform prims (i.e. have a transform component)
                self._extras[asset_name] = XformPrimView(asset_cfg.prim_path, device=self.device, stage=self.stage)
            else:
                raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")
            # store global collision paths
            if hasattr(asset_cfg, "collision_group") and asset_cfg.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
                self._global_prim_paths += asset_paths
