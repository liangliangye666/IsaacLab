# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import yaml

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.configclass import configclass


@configclass
class CuroboPlannerCfg:
    """Configuration for CuRobo motion planner.

    This dataclass provides a flexible configuration system for the CuRobo motion planner.
    The base configuration is robot-agnostic, with factory methods providing pre-configured
    settings for specific robots and tasks.

    Example Usage:
        >>> # Use a pre-configured robot
        >>> config = CuroboPlannerCfg.franka_config()
        >>>
        >>> # Or create from task name
        >>> config = CuroboPlannerCfg.from_task_name("Isaac-Stack-Cube-Franka-v0")
        >>>
        >>> # Initialize planner with config
        >>> planner = CuroboPlanner(env, robot, config)

    To add support for a new robot, see the factory methods section below for detailed instructions.
    """
    """设置CuRobo运动规划器。

    这个数据类为CuRobo运动规划器提供了灵活的配置系统。
    基础配置是机器人无知，工厂方法为特定机器人和任务提供预先配置的设置。

    例子使用:
        >>> # Use a pre-configured robot
        >>> config = CuroboPlannerCfg.franka_config()
        >>>
        >>> # Or create from task name
        >>> config = CuroboPlannerCfg.from_task_name("Isaac-Stack-Cube-Franka-v0")
        >>>
        >>> # Initialize planner with config
        >>> planner = CuroboPlanner(env, robot, config)

    若要增加对新机器人的支持，请参阅下面的工厂方法部分。
    """

    # Robot configuration
    robot_config_file: str | None = None
    """cuRobo robot configuration file (path defined by curobo api)."""
    """cuRobo机器人配置文件 (路径由curobo api定义)。"""

    robot_name: str = ""
    """Robot name for visualization and identification."""
    """机器人名字用于可视化和识别。"""

    ee_link_name: str | None = None
    """End-effector link name (auto-detected from robot config if None)."""
    """末端执行器链接名称 (如果 None，自动从机器人配置中检测)。"""

    # Gripper configuration
    gripper_joint_names: list[str] = []
    """Names of gripper joints."""
    """抓紧关节的名称。"""

    gripper_open_positions: dict[str, float] = {}
    """Open gripper positions for cuRobo to update spheres"""
    """开放 cuRobo 抓住位置更新球"""

    gripper_closed_positions: dict[str, float] = {}
    """Closed gripper positions for cuRobo to update spheres"""
    """关闭的抓住位置为cuRobo更新球"""

    # Hand link configuration (for contact planning)
    hand_link_names: list[str] = []
    """Names of hand/finger links to disable during contact planning."""
    """在接触规划过程中禁用手指链接的名称。"""

    # Attachment configuration
    attached_object_link_name: str = "attached_object"
    """Name of the link used for attaching objects."""
    """用于连接物体的链接名称。"""

    # World configuration
    world_config_file: str = "collision_table.yml"
    """CuRobo world configuration file (without path)."""
    """CuRobo世界配置文件 (没有路径)。"""

    # Static objects to not update in the world model
    static_objects: list[str] = []
    """Names of static objects to not update in the world model."""
    """在世界模型中不可更新的静态物体名称。"""

    # Optional prim path configuration
    robot_prim_path: str | None = None
    """Absolute USD prim path to the robot root for world extraction; None derives it from environment root."""
    """绝对USD prim通往机器人根的路径用于世界提取；None从环境根中提取。"""

    world_ignore_substrings: list[str] | None = None
    """List of substring patterns to ignore when extracting world obstacles
    (e.g., default ground plane, debug prims).
    """
    """在提取世界障碍时忽略的子字符串模式列表 (e.g.，默认地面平面，故障prims)。
    """

    # Motion planning parameters
    collision_checker_type: CollisionCheckerType = CollisionCheckerType.MESH
    """Type of collision checker to use."""
    """使用的碰撞检查器类型。"""

    num_trajopt_seeds: int = 12
    """Number of seeds for trajectory optimization."""
    """轨道优化种子数量"""

    num_graph_seeds: int = 12
    """Number of seeds for graph search."""
    """图表搜索的种子数量。"""

    interpolation_dt: float = 0.05
    """Time step for interpolating waypoints."""
    """时间步骤，用于插入路线。"""

    collision_cache_size: dict[str, int] = {"obb": 150, "mesh": 150}
    """Cache sizes for different collision types."""
    """对不同类型的碰撞缓存尺寸。"""

    trajopt_tsteps: int = 32
    """Number of trajectory optimization time steps."""
    """轨道优化时间步骤数量"""

    collision_activation_distance: float = 0.0
    """Distance at which collision constraints are activated."""
    """碰撞约束的距离"""

    approach_distance: float = 0.05
    """Distance to approach at the end of the plan."""
    """在计划结束时距离接近。"""

    retreat_distance: float = 0.05
    """Distance to retreat at the start of the plan."""
    """在计划开始时退缩的距离。"""

    grasp_gripper_open_val: float = 0.04
    """Gripper joint value when considered open for grasp detection."""
    """当被视为开放，可检测抓地时，抓地关节值。"""

    # Planning configuration
    enable_graph: bool = True
    """Whether to enable graph-based planning."""
    """是否实现基于图的规划。"""

    enable_graph_attempt: int = 5
    """Number of graph planning attempts."""
    """图形规划尝试数量"""

    max_planning_attempts: int = 15
    """Maximum number of planning attempts."""
    """计划尝试的最大数量"""

    enable_finetune_trajopt: bool = True
    """Whether to enable trajectory optimization fine-tuning."""
    """是否实现轨迹优化细节调整。"""

    time_dilation_factor: float = 1.0
    """Time dilation factor for planning."""
    """时间扩展因素"""

    surface_sphere_radius: float = 0.005
    """Radius of surface spheres for collision checking."""
    """对于碰撞检查的表面球半径。"""

    # Debug and visualization
    n_repeat: int | None = None
    """Number of times to repeat final waypoint for stabilization. If None, no repetition."""
    """稳定的最后路线重复次数。
    如果是None，没有重复。
    """

    motion_step_size: float | None = None
    """Step size (in radians) for retiming motion plans. If None, no retiming."""
    """步骤尺寸 (在半径中)，用于重复运动计划。
    如果是None，就不会再拍。
    """

    visualize_spheres: bool = False
    """Visualize robot collision spheres. Note: only works for env 0."""
    """设想机器人碰撞球。
    注:仅适用于env 0。
    """

    visualize_plan: bool = False
    """Visualize motion plan in Rerun. Note: only works for env 0."""
    """设想Rerun的运动计划。
    注:仅适用于env 0。
    """

    debug_planner: bool = False
    """Enable detailed motion planning debug information."""
    """启用详细的运动计划调试信息。"""

    sphere_update_freq: int = 5
    """Frequency to update sphere visualization, specified in number of frames."""
    """频率更新球体可视化，在数个框架中指定。"""

    motion_noise_scale: float = 0.0
    """Scale of Gaussian noise to add to the planned waypoints. Defaults to 0.0 (no noise)."""
    """为了增加计划的路线点。
    默认到0.0 (无噪音)。
    """

    # Collision sphere configuration
    collision_spheres_file: str | None = None
    """Collision spheres configuration file (auto-detected if None)."""
    """碰撞球配置文件 (如果 None，自动检测)。"""

    extra_collision_spheres: dict[str, int] = {"attached_object": 100}
    """Extra collision spheres for attached objects."""
    """附着物体的额外碰撞球。"""

    position_threshold: float = 0.005
    """Position threshold for motion planning."""
    """运动规划的位置门。"""

    rotation_threshold: float = 0.05
    """Rotation threshold for motion planning."""
    """运动规划的旋转门。"""

    cuda_device: int | None = 0
    """Preferred CUDA device index; None uses torch.cuda.current_device() (respects CUDA_VISIBLE_DEVICES)."""
    """首选CUDA设备索引；None使用torch.cuda.current_device() (指 CUDA_VISIBLE_DEVICES)。"""

    def get_world_config(self) -> WorldConfig:
        """Load and prepare the world configuration.

        This method can be overridden in subclasses or customized per task
        to provide different world configuration setups.

        Returns:
            WorldConfig: The configured world for collision checking
        """
        """装载和准备世界配置。

        这种方法可以在子类中覆盖或按任务定制以提供不同的世界配置设置。

        返回：
            WorldConfig: 对碰撞检查的配置世界
        """
        # Default implementation: just load the world config file
        world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), self.world_config_file)))
        return world_cfg

    def _get_world_config_with_table_adjustment(self) -> WorldConfig:
        """Load world config with standard table adjustments.

        This is a helper method that implements the common pattern of adjusting
        table height and combining mesh/cuboid worlds. Used by specific task configs.

        Returns:
            WorldConfig: World configuration with adjusted table
        """
        """装载世界配置标准表调节。

        这是一种辅助方法，它实现了调节表高度和结合网格/立体世界的常见模式。
        用于特定任务配置。

        返回：
            WorldConfig: 世界配置与调整表
        """
        # Load the base world config
        world_cfg_table = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), self.world_config_file)))

        # Adjust table height if cuboid exists and has a pose
        if world_cfg_table.cuboid and len(world_cfg_table.cuboid) > 0 and world_cfg_table.cuboid[0].pose:
            world_cfg_table.cuboid[0].pose[2] -= 0.02

        # Get mesh world for additional collision objects
        world_cfg_mesh = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), self.world_config_file))
        ).get_mesh_world()

        # Adjust mesh configuration if it exists
        if world_cfg_mesh.mesh and len(world_cfg_mesh.mesh) > 0:
            mesh_obj = world_cfg_mesh.mesh[0]
            if mesh_obj.name:
                mesh_obj.name += "_mesh"
            if mesh_obj.pose:
                mesh_obj.pose[2] = -10.5  # Move mesh below scene

        # Combine cuboid and mesh worlds
        world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg_mesh.mesh)
        return world_cfg

    @classmethod
    def _create_temp_robot_yaml(cls, base_yaml: str, urdf_path: str) -> str:
        """Create a temporary robot configuration YAML with custom URDF path.

        Args:
            base_yaml: Base robot configuration file name
            urdf_path: Absolute path to the URDF file

        Returns:
            Path to the temporary YAML file

        Raises:
            FileNotFoundError: If the URDF file doesn't exist
        """
        """创建一个临时机器人配置YAML与定制URDF路径。

        参数：
            base_yaml: 基础机器人配置文件名称
            urdf_path: 进入URDF文件的绝对路径

        返回：
            暂时YAML文件的路径

        异常：
            FileNotFoundError: 如果URDF文件不存在
        """
        # Validate URDF path
        if not os.path.isabs(urdf_path) or not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF must be a local file: {urdf_path}")

        # Load base configuration
        robot_cfg_path = get_robot_configs_path()
        base_path = join_path(robot_cfg_path, base_yaml)
        data = load_yaml(base_path)
        print(f"urdf_path: {urdf_path}")
        # Update URDF path
        data["robot_cfg"]["kinematics"]["urdf_path"] = urdf_path

        # Write to temporary file
        tmp_dir = tempfile.mkdtemp(prefix="curobo_robot_cfg_")
        out_path = os.path.join(tmp_dir, base_yaml)
        with open(out_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        return out_path

    # =====================================================================================
    # FACTORY METHODS FOR ROBOT CONFIGURATIONS
    # =====================================================================================
    """
    Creating Custom Robot Configurations
    =====================================

    To create a configuration for your own robot, follow these steps:

    1. Create a Factory Method
    ---------------------------
    Define a classmethod that returns a configured instance:

    .. code-block:: python

        @classmethod
        def my_robot_config(cls) -> "CuroboPlannerCfg":
            # Option 1: Download from Nucleus (like Franka example)
            urdf_path = f"{ISAACLAB_NUCLEUS_DIR}/path/to/my_robot.urdf"
            local_urdf = retrieve_file_path(urdf_path, force_download=True)

            # Option 2: Use local file directly
            # local_urdf = "/absolute/path/to/my_robot.urdf"

            # Create temporary YAML with custom URDF path
            robot_cfg_file = cls._create_temp_robot_yaml("my_robot.yml", local_urdf)

            return cls(
                # Required: Specify robot configuration file
                robot_config_file=robot_cfg_file,  # Use the generated YAML with custom URDF
                robot_name="my_robot",

                # Gripper configuration (if robot has grippers)
                gripper_joint_names=["gripper_left", "gripper_right"],
                gripper_open_positions={"gripper_left": 0.05, "gripper_right": 0.05},
                gripper_closed_positions={"gripper_left": 0.01, "gripper_right": 0.01},

                # Hand/finger links to disable during contact planning
                hand_link_names=["finger_link_1", "finger_link_2", "palm_link"],

                # Optional: Absolute USD prim path to the robot root for world extraction;
                # None derives it from environment root.
                robot_prim_path=None,

                # Optional: List of substring patterns to ignore when extracting world obstacles
                # (e.g., default ground plane, debug prims).
                # None derives it from the environment root and adds some default patterns.
                # This is useful for environments with a lot of prims.
                world_ignore_substrings=None,

                # Optional: Custom collision spheres configuration
                # Path relative to curobo (can override with custom spheres file)
                collision_spheres_file="spheres/my_robot_spheres.yml",

                # Grasp detection threshold
                grasp_gripper_open_val=0.05,

                # Motion planning parameters (tune for your robot)
                approach_distance=0.05,  # Distance to approach before grasping
                retreat_distance=0.05,   # Distance to retreat after grasping
                time_dilation_factor=0.5,  # Speed factor (0.5 = half speed)

                # Visualization options
                visualize_spheres=False,
                visualize_plan=False,
                debug_planner=False,
            )

    2. Task-Specific Configurations
    --------------------------------
    For task-specific variants, create methods that modify the base config:

    .. code-block:: python

        @classmethod
        def my_robot_pick_place_config(cls) -> "CuroboPlannerCfg":
            config = cls.my_robot_config()  # Start from base config

            # Override for pick-and-place tasks
            config.approach_distance = 0.08
            config.retreat_distance = 0.10
            config.enable_finetune_trajopt = True
            config.collision_activation_distance = 0.02

            # Custom world configuration if needed
            config.get_world_config = lambda: config._get_world_config_with_table_adjustment()

            return config

    3. Register in from_task_name()
    --------------------------------
    Add your robot detection logic to the from_task_name method:

    .. code-block:: python

        @classmethod
        def from_task_name(cls, task_name: str) -> "CuroboPlannerCfg":
            task_lower = task_name.lower()

            # Add your robot detection
            if "my-robot" in task_lower:
                if "pick-place" in task_lower:
                    return cls.my_robot_pick_place_config()
                else:
                    return cls.my_robot_config()

            # ... existing robot checks ...

    Important Notes
    ---------------
    - The _create_temp_robot_yaml() helper creates a temporary YAML with your custom URDF
    - If using Nucleus assets, retrieve_file_path() downloads them to a local temp directory
    - The base robot YAML (e.g., "my_robot.yml") should exist in cuRobo's robot configs

    Best Practices
    --------------
    1. Start with conservative parameters (slow speed, large distances)
    2. Test with visualization enabled (visualize_plan=True) for debugging
    3. Tune collision_activation_distance based on controller precision to follow collision-free motion
    4. Adjust sphere counts in extra_collision_spheres for attached objects
    5. Use debug_planner=True when developing new configurations
    """
    """创建自定义机器人配置

    为了为自己的机器人创建配置，

    1. 创建一个工厂方法
    定义返回配置实例的类方法:

    .. code-block:: python

        @classmethod
        def my_robot_config(cls) -> "CuroboPlannerCfg":
            # Option 1: Download from Nucleus (like Franka example)
            urdf_path = f"{ISAACLAB_NUCLEUS_DIR}/path/to/my_robot.urdf"
            local_urdf = retrieve_file_path(urdf_path, force_download=True)

            # Option 2: Use local file directly
            # local_urdf = "/absolute/path/to/my_robot.urdf"

            # Create temporary YAML with custom URDF path
            robot_cfg_file = cls._create_temp_robot_yaml("my_robot.yml", local_urdf)

            return cls(
                # Required: Specify robot configuration file
                robot_config_file=robot_cfg_file,  # Use the generated YAML with custom URDF
                robot_name="my_robot",

                # Gripper configuration (if robot has grippers)
                gripper_joint_names=["gripper_left", "gripper_right"],
                gripper_open_positions={"gripper_left": 0.05, "gripper_right": 0.05},
                gripper_closed_positions={"gripper_left": 0.01, "gripper_right": 0.01},

                # Hand/finger links to disable during contact planning
                hand_link_names=["finger_link_1", "finger_link_2", "palm_link"],

                # Optional: Absolute USD prim path to the robot root for world extraction;
                # None derives it from environment root.
                robot_prim_path=None,

                # Optional: List of substring patterns to ignore when extracting world obstacles
                # (e.g., default ground plane, debug prims).
                # None derives it from the environment root and adds some default patterns.
                # This is useful for environments with a lot of prims.
                world_ignore_substrings=None,

                # Optional: Custom collision spheres configuration
                # Path relative to curobo (can override with custom spheres file)
                collision_spheres_file="spheres/my_robot_spheres.yml",

                # Grasp detection threshold
                grasp_gripper_open_val=0.05,

                # Motion planning parameters (tune for your robot)
                approach_distance=0.05,  # Distance to approach before grasping
                retreat_distance=0.05,   # Distance to retreat after grasping
                time_dilation_factor=0.5,  # Speed factor (0.5 = half speed)

                # Visualization options
                visualize_spheres=False,
                visualize_plan=False,
                debug_planner=False,
            )

    2. 具体任务配置
    -------------------------------- 对于具体任务的变体，创建修改基础配置的方法:

    .. code-block:: python

        @classmethod
        def my_robot_pick_place_config(cls) -> "CuroboPlannerCfg":
            config = cls.my_robot_config()  # Start from base config

            # Override for pick-and-place tasks
            config.approach_distance = 0.08
            config.retreat_distance = 0.10
            config.enable_finetune_trajopt = True
            config.collision_activation_distance = 0.02

            # Custom world configuration if needed
            config.get_world_config = lambda: config._get_world_config_with_table_adjustment()

            return config

    3. 在from_task_name注册)
    添加你的机器人检测逻辑到from_task_name方法:

    .. code-block:: python

        @classmethod
        def from_task_name(cls, task_name: str) -> "CuroboPlannerCfg":
            task_lower = task_name.lower()

            # Add your robot detection
            if "my-robot" in task_lower:
                if "pick-place" in task_lower:
                    return cls.my_robot_pick_place_config()
                else:
                    return cls.my_robot_config()

            # ... existing robot checks ...

    重要说明
    - 随着您的自定义URDF创建临时YAML
    - 如果使用Nucleus资产，retrieve_file_path() 将它们下载到本地临时目录
    - 在cuRobo的机器人配置中应该存在的基机器人YAML (e.g.，"my_robot.yml")

    最好的做法
    1. 开始使用保守的参数 (速度缓慢，距离很大)
    2. 测试设置可视化 (visualize_plan=True)
    3. 根据控制器精度调整collision_activation_distance以跟踪无碰撞的运动
    4. 在 extra_collision_spheres 中调整附加物体的球数
    5. 在开发新配置时使用debug_planner=True
    """

    @classmethod
    def franka_config(cls) -> "CuroboPlannerCfg":
        """Create configuration for Franka Panda robot.

        This method uses a custom URDF from Nucleus for the Franka robot.

        Returns:
            CuroboPlannerCfg: Configuration for Franka robot
        """
        """创建了弗兰卡潘达机器人的配置。

        这种方法使用了来自核电的自定义URDF用于弗兰卡机器人。

        返回：
            CuroboPlannerCfg: 弗兰卡机器人配置
        """
        urdf_path = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/SkillGenAssets/FrankaPanda/franka_panda.urdf"
        local_urdf = retrieve_file_path(urdf_path, force_download=True)

        robot_cfg_file = cls._create_temp_robot_yaml("franka.yml", local_urdf)

        return cls(
            robot_config_file=robot_cfg_file,
            robot_name="franka",
            gripper_joint_names=["panda_finger_joint1", "panda_finger_joint2"],
            gripper_open_positions={"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
            gripper_closed_positions={"panda_finger_joint1": 0.023, "panda_finger_joint2": 0.023},
            hand_link_names=["panda_leftfinger", "panda_rightfinger", "panda_hand"],
            collision_spheres_file="spheres/franka_mesh.yml",
            grasp_gripper_open_val=0.04,
            approach_distance=0.0,
            retreat_distance=0.0,
            max_planning_attempts=1,
            time_dilation_factor=0.6,
            enable_finetune_trajopt=True,
            n_repeat=None,
            motion_step_size=None,
            visualize_spheres=False,
            visualize_plan=False,
            debug_planner=False,
            sphere_update_freq=5,
            motion_noise_scale=0.02,
            # World extraction tuning for Franka envs
            world_ignore_substrings=["/World/defaultGroundPlane", "/curobo"],
        )

    @classmethod
    def franka_stack_cube_bin_config(cls) -> "CuroboPlannerCfg":
        """Create configuration for Franka stacking cube in a bin."""
        """在垃圾桶中创建了弗兰卡堆叠立方体的配置。"""
        config = cls.franka_config()
        config.static_objects = ["bin", "table"]
        config.gripper_closed_positions = {"panda_finger_joint1": 0.024, "panda_finger_joint2": 0.024}
        config.approach_distance = 0.05
        config.retreat_distance = 0.07
        config.surface_sphere_radius = 0.01
        config.debug_planner = False
        config.collision_activation_distance = 0.02
        config.visualize_plan = False
        config.enable_finetune_trajopt = True
        config.motion_noise_scale = 0.02
        config.get_world_config = lambda: config._get_world_config_with_table_adjustment()
        return config

    @classmethod
    def franka_stack_cube_config(cls) -> "CuroboPlannerCfg":
        """Create configuration for Franka stacking a normal cube."""
        """创建一个配置，让弗兰卡堆一个正常的立方体。"""
        config = cls.franka_config()
        config.static_objects = ["table"]
        config.visualize_plan = False
        config.debug_planner = False
        config.motion_noise_scale = 0.02
        config.collision_activation_distance = 0.01
        config.approach_distance = 0.05
        config.retreat_distance = 0.05
        config.surface_sphere_radius = 0.01
        config.get_world_config = lambda: config._get_world_config_with_table_adjustment()
        return config

    @classmethod
    def from_task_name(cls, task_name: str) -> "CuroboPlannerCfg":
        """Create configuration from task name.

        Args:
            task_name: Task name (e.g., "Isaac-Stack-Cube-Bin-Franka-v0")

        Returns:
            CuroboPlannerCfg: Configuration for the specified task
        """
        """从任务名称创建配置。

        参数：
            task_name: 任务名称 (e.g.， "Isaac-Stack-Cube-Bin-Franka-v0")

        返回：
            CuroboPlannerCfg: 指定任务的配置
        """
        task_lower = task_name.lower()

        if "stack-cube-bin" in task_lower:
            return cls.franka_stack_cube_bin_config()
        elif "stack-cube" in task_lower:
            return cls.franka_stack_cube_config()
        else:
            # Default to Franka configuration
            print(f"Warning: Unknown robot in task '{task_name}', using Franka configuration")
            return cls.franka_config()
