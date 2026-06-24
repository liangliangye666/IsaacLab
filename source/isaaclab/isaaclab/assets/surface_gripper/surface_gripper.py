# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import torch

from isaacsim.core.utils.extensions import enable_extension

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBase
from isaaclab.utils.version import get_isaac_sim_version

if TYPE_CHECKING:
    from isaacsim.robot.surface_gripper import GripperView

    from .surface_gripper_cfg import SurfaceGripperCfg

# import logger
logger = logging.getLogger(__name__)


class SurfaceGripper(AssetBase):
    """A surface gripper actuator class.

    Surface grippers are actuators capable of grasping objects when in close proximity with them.

    Each surface gripper in the collection must be a `Isaac Sim SurfaceGripper` primitive.
    On playing the simulation, the physics engine will automatically register the surface grippers into a
    SurfaceGripperView object. This object can be accessed using the :attr:`gripper_view` attribute.

    To interact with the surface grippers, the user can use the :attr:`state` to get the current state of the grippers,
    :attr:`command` to get the current command sent to the grippers, and :func:`update_gripper_properties` to update the
    properties of the grippers at runtime. Finally, the :func:`set_grippers_command` function should be used to set the
    desired command for the grippers.

    Note:
        The :func:`set_grippers_command` function does not write to the simulation. The simulation automatically
         calls :func:`write_data_to_sim` function to write the command to the simulation. Similarly, the update
         function is called automatically for every simulation step, and does not need to be called by the user.

    Note:
        The SurfaceGripper is only supported on CPU for now. Please set the simulation backend to run on CPU.
        Use `--device cpu` to run the simulation on CPU.
    """
    """一个表面抓紧机动机类。

    表面抓住器是能够在接近物体时抓住物体的驱动器。

    收藏中的每一个表面必须是`Isaac Sim SurfaceGripper`原始。
    在播放仿真时，物理引擎将自动注册表面抓住器在SurfaceGripperView对象中。
    这个对象可以使用:attr:`gripper_view`属性访问。

    用户可以使用:attr:`state`来获取抓住器的当前状态，:attr:`command`来将当前命令发送给抓住器，:func:`update_gripper_properties`可以在运行时更新
    抓住器的性能。
    最后，应使用:func:`set_grippers_command`函数来设置对抓住器所需的命令。

    说明：
        The :函数:`set_grippers_command`函数不会写入仿真。
             自动仿真
         calls :函数:`write_data_to_sim`函数用于编写命令到仿真中。
                同样，更新
         每个仿真步骤都会自动调用函数，不需要被用户调用。

    说明：
        目前，SurfaceGripper仅支持CPU。
        请设置仿真后端在CPU上运行。
        使用`--device cpu`来运行CPU的仿真。
    """

    def __init__(self, cfg: SurfaceGripperCfg):
        """Initialize the surface gripper.

        Args:
            cfg: A configuration instance.
        """
        """启动表面抓住器。

        参数：
            cfg: 一个配置实例。
        """
        # copy the configuration
        self._cfg = cfg.copy()

        # checks for Isaac Sim v5.0 to ensure that the surface gripper is supported
        if get_isaac_sim_version().major < 5:
            raise NotImplementedError(
                "SurfaceGrippers are only supported by IsaacSim 5.0 and newer. Current version is"
                f" '{get_isaac_sim_version()}'. Please update to IsaacSim 5.0 or newer to use this feature."
            )

        # flag for whether the sensor is initialized
        self._is_initialized = False
        self._debug_vis_handle = None

        # register various callback functions
        self._register_callbacks()

    """
    Properties
    """
    """产品
    """

    @property
    def data(self):
        raise NotImplementedError("SurfaceGripper does have a data interface.")

    @property
    def num_instances(self) -> int:
        """Number of instances of the gripper.

        This is equal to the total number of grippers (the view can only contain one gripper per environment).
        """
        """抓住器的例数。

        这等于抓住器的总数 (视图只能包含每个环境的抓住器)。
        """
        return self._num_envs

    @property
    def state(self) -> torch.Tensor:
        """Returns the gripper state buffer.

        The gripper state is a list of integers:
        - -1 --> Open
        - 0 --> Closing
        - 1 --> Closed
        """
        """返回抓住器状态缓冲器。

        抓住器状态是整数列表:
        - -1 --> 开放
        - 0 --> 关闭
        - 1 --> 关闭
        """
        return self._gripper_state

    @property
    def command(self) -> torch.Tensor:
        """Returns the gripper command buffer.

        The gripper command is a list of floats:
        - [-1, -0.3] --> Open
        - [-0.3, 0.3] --> Do nothing
        - [0.3, 1] --> Close
        """
        """返回抓住器命令缓冲器。

        抓住器指令是浮动器的列表:
        - [-1， -0.3] --> 开放
        - [-0.3，0.3] --> 别做什么
        - [0.3， 1] --> 关闭
        """
        return self._gripper_command

    @property
    def gripper_view(self) -> GripperView:
        """Returns the gripper view object."""
        """返回抓住器视图对象。"""
        return self._gripper_view

    """
    Operations
    """
    """运营
    """

    def update_gripper_properties(
        self,
        max_grip_distance: torch.Tensor | None = None,
        coaxial_force_limit: torch.Tensor | None = None,
        shear_force_limit: torch.Tensor | None = None,
        retry_interval: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Update the gripper properties.

        Args:
            max_grip_distance: The maximum grip distance of the gripper. Should be a tensor of shape (num_envs,).
            coaxial_force_limit: The coaxial force limit of the gripper. Should be a tensor of shape (num_envs,).
            shear_force_limit: The shear force limit of the gripper. Should be a tensor of shape (num_envs,).
            retry_interval: The retry interval of the gripper. Should be a tensor of shape (num_envs,).
            indices: The indices of the grippers to update the properties for. Can be a tensor of any shape.
        """
        """更新抓住器性能。

        参数：
            max_grip_distance: 抓住器的最大抓住距离。
                               应该是形状张量 (num_envs，)。
            coaxial_force_limit: 抓住器的同轴力限制。
                                 应该是形状张量 (num_envs，)。
            shear_force_limit: 抓住器的切割力限制。
                               应该是形状张量 (num_envs，)。
            retry_interval: 抓住器的重试间隔。
                            应该是形状张量 (num_envs，)。
            indices: 为了更新性能，抓住器的指标。
                     它可以是任何形状的子。
        """

        if indices is None:
            indices = self._ALL_INDICES

        indices_as_list = indices.tolist()

        if max_grip_distance is not None:
            self._max_grip_distance[indices] = max_grip_distance
        if coaxial_force_limit is not None:
            self._coaxial_force_limit[indices] = coaxial_force_limit
        if shear_force_limit is not None:
            self._shear_force_limit[indices] = shear_force_limit
        if retry_interval is not None:
            self._retry_interval[indices] = retry_interval

        self._gripper_view.set_surface_gripper_properties(
            max_grip_distance=self._max_grip_distance.tolist(),
            coaxial_force_limit=self._coaxial_force_limit.tolist(),
            shear_force_limit=self._shear_force_limit.tolist(),
            retry_interval=self._retry_interval.tolist(),
            indices=indices_as_list,
        )

    def update(self, dt: float) -> None:
        """Update the gripper state using the SurfaceGripperView.

        This function is called every simulation step.
        The data fetched from the gripper view is a list of strings containing 3 possible states:
            - "Open" --> 0
            - "Closing" --> 1
            - "Closed" --> 2

        To make this more neural network friendly, we convert the list of strings to a list of floats:
            - "Open" --> -1.0
            - "Closing" --> 0.0
            - "Closed" --> 1.0

        Note:
            We need to do this conversion for every single step of the simulation because the gripper can lose contact
            with the object if some conditions are met: such as if a large force is applied to the gripped object.
        """
        """通过SurfaceGripperView更新抓住器状态。

        这种函数被称为每个仿真步骤。
        从抓住器视图中获取的数据是包含3种可能状态的字符串列表:
            - "开放" --> 0
            - "关闭" --> 1
            - "关闭" --> 2

        为了使这个神经网络更加友好， 我们将字符串列表转换为浮动列表:
            - "开放" --> -1.0
            - "关闭" --> 0.0
            - "关闭" --> 1.0

        说明：
            我们需要在仿真的每一步都进行这种转换，因为抓住器可能会失去接触
            with the object if some conditions are met: such as if a large force is applied to the gripped object.
        """
        state_list: list[int] = self._gripper_view.get_surface_gripper_status()
        self._gripper_state = torch.tensor(state_list, dtype=torch.float32, device=self._device) - 1.0

    def write_data_to_sim(self) -> None:
        """Write the gripper command to the SurfaceGripperView.

        The gripper command is a list of integers that needs to be converted to a list of strings:
            - [-1, -0.3] --> Open
            - ]-0.3, 0.3[ --> Do nothing
            - [0.3, 1] --> Closed

        The Do nothing command is not applied, and is only used to indicate whether the gripper state has changed.
        """
        """给SurfaceGripperView写下抓住器命令。

        抓住器命令是整数列表，需要转换为字符串列表:
            - [-1， -0.3] --> 开放
            - [-0.3，0.3[ --> 什么都不要做
            - 关闭

        没有执行命令，只用于表示抓住器状态是否改变。
        """
        # Remove the SurfaceGripper indices that have a commanded value of 2
        indices = (
            torch.argwhere(torch.logical_or(self._gripper_command < -0.3, self._gripper_command > 0.3))
            .to(torch.int32)
            .tolist()
        )
        # Write to the SurfaceGripperView if there are any indices to write to
        if len(indices) > 0:
            self._gripper_view.apply_gripper_action(self._gripper_command.tolist(), indices)

    def set_grippers_command(self, states: torch.Tensor, indices: torch.Tensor | None = None) -> None:
        """Set the internal gripper command buffer. This function does not write to the simulation.

        Possible values for the gripper command are:
            - [-1, -0.3] --> Open
            - ]-0.3, 0.3[ --> Do nothing
            - [0.3, 1] --> Close

        Args:
            states: A tensor of integers representing the gripper command. Shape must match that of indices.
            indices: A tensor of integers representing the indices of the grippers to set the command for. Defaults
                     to None, in which case all grippers are set.
        """
        """设置内置抓住器命令缓冲器。
        这种函数不会写入仿真。

        抓住器指令的可能值是:
            - [-1， -0.3] --> 开放
            - [-0.3，0.3[ --> 什么都不要做
            - [0.3， 1] --> 关闭

        参数：
            states: 一个代表抓住器命令的整数数。
                    它们的形状必须与指标的形状相匹配。
            indices: 一个代表控制器索引的整数数。
                     默认设置None，在这种情况下，所有抓住器都设置。
        """
        if indices is None:
            indices = self._ALL_INDICES

        self._gripper_command[indices] = states

    def reset(self, indices: torch.Tensor | None = None) -> None:
        """Reset the gripper command buffer.

        Args:
            indices: A tensor of integers representing the indices of the grippers to reset the command for. Defaults
                     to None, in which case all grippers are reset.
        """
        """恢复抓住器命令缓冲器。

        参数：
            indices: 一个代表控制器的索引的整数数。
                     默认设置为None，在这种情况下，所有抓住器都会重置。
        """
        # Would normally set the buffer to 0, for now we won't do that
        if indices is None:
            indices = self._ALL_INDICES

        # Reset the selected grippers to an open status
        self._gripper_command[indices] = -1.0
        self.write_data_to_sim()
        # Sets the gripper last command to be 0.0 (do nothing)
        self._gripper_command[indices] = 0
        # Force set the state to open. It will read open in the next update call.
        self._gripper_state[indices] = -1.0

    """
    Initialization.
    """
    """启动。
    """

    def _initialize_impl(self) -> None:
        """Initializes the gripper-related handles and internal buffers.

        Raises:
            ValueError: If the simulation backend is not CPU.
            RuntimeError: If the Simulation Context is not initialized or if gripper prims are not found.

        Note:
            The SurfaceGripper is only supported on CPU for now. Please set the simulation backend to run on CPU.
            Use `--device cpu` to run the simulation on CPU.
        """
        """启动与抓住器相关的句柄和内部缓冲器。

        异常：
            ValueError: 如果仿真后端不是CPU。
            RuntimeError: 如果仿真文本未启动或没有找到prims抓住器。

        说明：
            目前，SurfaceGripper仅支持CPU。
            请设置仿真后端在CPU上运行。
            使用`--device cpu`来运行CPU的仿真。
        """

        enable_extension("isaacsim.robot.surface_gripper")
        from isaacsim.robot.surface_gripper import GripperView

        # Check that we are using the CPU backend.
        if self._device != "cpu":
            raise Exception(
                "SurfaceGripper is only supported on CPU for now. Please set the simulation backend to run on CPU. Use"
                " `--device cpu` to run the simulation on CPU."
            )

        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self._cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self._cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        # find surface gripper prims
        gripper_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.GetTypeName() == "IsaacSurfaceGripper",
            traverse_instance_prims=False,
        )
        if len(gripper_prims) == 0:
            raise RuntimeError(
                f"Failed to find a surface gripper when resolving '{self._cfg.prim_path}'."
                " Please ensure that the prim has type 'IsaacSurfaceGripper'."
            )
        if len(gripper_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single surface gripper when resolving '{self._cfg.prim_path}'."
                f" Found multiple '{gripper_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one surface gripper in the prim path tree."
            )

        # resolve gripper prim back into regex expression
        gripper_prim_path = gripper_prims[0].GetPath().pathString
        gripper_prim_path_expr = self._cfg.prim_path + gripper_prim_path[len(template_prim_path) :]

        # Count number of environments
        self._prim_expr = gripper_prim_path_expr
        env_prim_path_expr = self._prim_expr.rsplit("/", 1)[0]
        self._parent_prims = sim_utils.find_matching_prims(env_prim_path_expr)
        self._num_envs = len(self._parent_prims)

        # Create buffers
        self._create_buffers()

        # Process the configuration
        self._process_cfg()

        # Initialize gripper view and set properties. Note we do not set the properties through the gripper view
        # to avoid having to convert them to list of floats here. Instead, we do it in the update_gripper_properties
        # function which does this conversion internally.
        self._gripper_view = GripperView(
            self._prim_expr,
        )
        self.update_gripper_properties(
            max_grip_distance=self._max_grip_distance.clone(),
            coaxial_force_limit=self._coaxial_force_limit.clone(),
            shear_force_limit=self._shear_force_limit.clone(),
            retry_interval=self._retry_interval.clone(),
        )

        # log information about the surface gripper
        logger.info(f"Surface gripper initialized at: {self._cfg.prim_path} with root '{gripper_prim_path_expr}'.")
        logger.info(f"Number of instances: {self._num_envs}")

        # Reset grippers
        self.reset()

    def _create_buffers(self) -> None:
        """Create the buffers for storing the gripper state, command, and properties."""
        """创建缓冲器来存储抓住器状态，命令和属性。"""
        self._gripper_state = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._gripper_command = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._ALL_INDICES = torch.arange(self._num_envs, device=self._device, dtype=torch.long)

        self._max_grip_distance = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._coaxial_force_limit = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._shear_force_limit = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)
        self._retry_interval = torch.zeros(self._num_envs, device=self._device, dtype=torch.float32)

    def _process_cfg(self) -> None:
        """Process the configuration for the gripper properties."""
        """处理对抓住器性能的配置。"""
        # Get one of the grippers as defined in the default stage
        gripper_prim = self._parent_prims[0]
        try:
            max_grip_distance = gripper_prim.GetAttribute("isaac:maxGripDistance").Get()
        except Exception as e:
            warnings.warn(
                f"Failed to retrieve max_grip_distance from stage, defaulting to user provided cfg. Exception: {e}"
            )
            max_grip_distance = None

        try:
            coaxial_force_limit = gripper_prim.GetAttribute("isaac:coaxialForceLimit").Get()
        except Exception as e:
            warnings.warn(
                f"Failed to retrieve coaxial_force_limit from stage, defaulting to user provided cfg. Exception: {e}"
            )
            coaxial_force_limit = None

        try:
            shear_force_limit = gripper_prim.GetAttribute("isaac:shearForceLimit").Get()
        except Exception as e:
            warnings.warn(
                f"Failed to retrieve shear_force_limit from stage, defaulting to user provided cfg. Exception: {e}"
            )
            shear_force_limit = None

        try:
            retry_interval = gripper_prim.GetAttribute("isaac:retryInterval").Get()
        except Exception as e:
            warnings.warn(
                f"Failed to retrieve retry_interval from stage defaulting to user provided cfg. Exception: {e}"
            )
            retry_interval = None

        self._max_grip_distance = self.parse_gripper_parameter(self._cfg.max_grip_distance, max_grip_distance)
        self._coaxial_force_limit = self.parse_gripper_parameter(self._cfg.coaxial_force_limit, coaxial_force_limit)
        self._shear_force_limit = self.parse_gripper_parameter(self._cfg.shear_force_limit, shear_force_limit)
        self._retry_interval = self.parse_gripper_parameter(self._cfg.retry_interval, retry_interval)

    """
    Helper functions.
    """
    """辅助函数。
    """

    def parse_gripper_parameter(
        self, cfg_value: float | int | tuple | None, default_value: float | int | tuple | None, ndim: int = 0
    ) -> torch.Tensor:
        """Parse the gripper parameter.

        Args:
            cfg_value: The value to parse. Can be a float, int, tuple, or None.
            default_value: The default value to use if cfg_value is None. Can be a float, int, tuple, or None.
            ndim: The number of dimensions of the parameter. Defaults to 0.
        """
        """检查抓住器参数。

        参数：
            cfg_value: 分析的价值。
                       可以是浮动， int，tuple，或None。
            default_value: 如果cfg_value是None，则使用的默认值。
                           可以是浮动， int，tuple，或None。
            ndim: 参数的维度数。
                  默认为0。
        """
        # Adjust the buffer size based on the number of dimensions
        if ndim == 0:
            param = torch.zeros(self._num_envs, device=self._device)
        elif ndim == 3:
            param = torch.zeros(self._num_envs, 3, device=self._device)
        elif ndim == 4:
            param = torch.zeros(self._num_envs, 4, device=self._device)
        else:
            raise ValueError(f"Invalid number of dimensions: {ndim}")

        # Parse the parameter
        if cfg_value is not None:
            if isinstance(cfg_value, (float, int)):
                param[:] = float(cfg_value)
            elif isinstance(cfg_value, tuple):
                if len(cfg_value) == ndim:
                    param[:] = torch.tensor(cfg_value, dtype=torch.float, device=self._device)
                else:
                    raise ValueError(f"Invalid number of values for parameter. Got: {len(cfg_value)}\nExpected: {ndim}")
            else:
                raise TypeError(f"Invalid type for parameter value: {type(cfg_value)}. " + "Expected float or int.")
        elif default_value is not None:
            if isinstance(default_value, (float, int)):
                param[:] = float(default_value)
            elif isinstance(default_value, tuple):
                assert len(default_value) == ndim, f"Expected {ndim} values, got {len(default_value)}"
                param[:] = torch.tensor(default_value, dtype=torch.float, device=self._device)
            else:
                raise TypeError(
                    f"Invalid type for default value: {type(default_value)}. " + "Expected float or Tensor."
                )
        else:
            raise ValueError("The parameter value is None and no default value is provided.")

        return param
