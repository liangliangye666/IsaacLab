# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv


class MotionPlannerBase(ABC):
    """Abstract base class for motion planners.

    This class defines the public interface that all motion planners must implement.
    It focuses on the essential functionality that users interact with, while leaving
    implementation details to specific planner backends.

    The core workflow is:
    1. Initialize planner with environment and robot
    2. Call update_world_and_plan_motion() to plan to a target
    3. Execute plan using has_next_waypoint() and get_next_waypoint_ee_pose()

    Example:
        >>> from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        >>> from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
        >>> config = CuroboPlannerCfg.franka_config()
        >>> planner = CuroboPlanner(env, robot, config)
        >>> success = planner.update_world_and_plan_motion(target_pose)
        >>> if success:
        >>>     while planner.has_next_waypoint():
        >>>         action = planner.get_next_waypoint_ee_pose()
        >>>         obs, info = env.step(action)
    """
    """对于运动规划者来说，抽象基础类。

    这类定义了所有运动规划者必须实现的公共界面。
    它专注于用户交互的基本功能，同时将实施细节留给特定的规划者后台。

    核心工作流是:
    1. 使用环境和机器人启动规划器
    2. 呼叫update_world_and_plan_motion() 计划到目标
    3. 使用has_next_waypoint() 和get_next_waypoint_ee_pose() 执行计划

    示例：
        >>> from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        >>> from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
        >>> config = CuroboPlannerCfg.franka_config()
        >>> planner = CuroboPlanner(env, robot, config)
        >>> success = planner.update_world_and_plan_motion(target_pose)
        >>> if success:
        >>>     while planner.has_next_waypoint():
        >>>         action = planner.get_next_waypoint_ee_pose()
        >>>         obs, info = env.step(action)
    """

    def __init__(
        self, env: ManagerBasedEnv, robot: Articulation, env_id: int = 0, debug: bool = False, **kwargs
    ) -> None:
        """Initialize the motion planner.

        Args:
            env: The environment instance
            robot: Robot articulation to plan motions for
            env_id: Environment ID (0 to num_envs-1)
            debug: Whether to print detailed debugging information
            **kwargs: Additional planner-specific arguments
        """
        """启动运动规划器。

        参数：
            env: 环境实例
            robot: 机器人关节来规划运动
            env_id: 环境ID (0至num_envs-1)
            debug: 要否打印详细的调试信息
            **kwargs: 其他规划者特定参数
        """
        self.env = env
        self.robot = robot
        self.env_id = env_id
        self.debug = debug

    @abstractmethod
    def update_world_and_plan_motion(self, target_pose: torch.Tensor, **kwargs: Any) -> bool:
        """Update collision world and plan motion to target pose.

        This is the main entry point for motion planning. It should:
        1. Update the planner's internal world representation
        2. Plan a collision-free path to the target pose
        3. Store the plan internally for execution

        Args:
            target_pose: Target pose to plan motion to (4x4 transformation matrix)
            **kwargs: Planner-specific arguments (e.g., retiming, contact planning)

        Returns:
            bool: True if planning succeeded, False otherwise
        """
        """更新碰撞世界，并计划运动，以定位目标。

        这就是运动规划的主要入口点。
        它应该:
        1. 更新规划者内部世界表示
        2. 规划到目标位置的无碰撞路径
        3. 内部存储计划进行执行

        参数：
            target_pose: 目标姿势为计划运动 (4x4转换矩阵)
            **kwargs: 规划器特定的参数 (e.g.，重复时间，联系规划)

        返回：
            bool: 如果计划成功，True，否则False
        """
        raise NotImplementedError

    @abstractmethod
    def has_next_waypoint(self) -> bool:
        """Check if there are more waypoints in current plan.

        Returns:
            bool: True if there are more waypoints, False otherwise
        """
        """查看目前计划是否有更多的路线。

        返回：
            bool: 如果有更多的路线点，则True，否则False
        """
        raise NotImplementedError

    @abstractmethod
    def get_next_waypoint_ee_pose(self) -> Any:
        """Get next waypoint's end-effector pose from current plan.

        This method should only be called after checking has_next_waypoint().

        Returns:
            Any: End-effector pose for the next waypoint in the plan.
        """
        """从目前的计划中得到下一个路线点的终端效应。

        只有检查has_next_waypoint后才应调用这种方法。

        返回：
            Any: 在计划中下一个路线点的最终效应。
        """
        raise NotImplementedError

    def get_planned_poses(self) -> list[Any]:
        """Get all planned poses from current plan.

        Returns:
            list[Any]: List of planned poses.

        Note:
            Default implementation iterates through waypoints.
            Child classes can override for a more efficient implementation.
        """
        """从目前的计划中得到所有计划的姿势。

        返回：
            列表[任何]:计划的姿势列表。

        说明：
            默认实现通过路线点进行反复执行。
            儿童课程可以取消，以实现更高效的实施。
        """
        planned_poses = []
        # Create a copy of the planner state to not affect the original plan execution
        # This is a placeholder and may need to be implemented by child classes
        # if they manage complex internal state.
        # For now, we assume the planner can be reset and we can iterate through the plan.
        # A more robust solution might involve a dedicated method to get the full plan.
        self.reset_plan()
        while self.has_next_waypoint():
            pose = self.get_next_waypoint_ee_pose()
            planned_poses.append(pose)
        return planned_poses

    @abstractmethod
    def reset_plan(self) -> None:
        """Reset the current plan and execution state.

        This should clear any stored plan and reset the execution index or iterator.
        """
        """重置当前的计划和执行状态。

        这应该清除任何存储的计划并重置执行索引或代器。
        """
        raise NotImplementedError

    def get_planner_info(self) -> dict[str, Any]:
        """Get information about the planner.

        Returns:
            dict: Information about the planner (name, version, capabilities, etc.)
        """
        """获取有关规划者的信息。

        返回：
            dict: 规划者信息 (名称，版本，功能等)
        """
        return {
            "name": self.__class__.__name__,
            "env_id": self.env_id,
            "debug": self.debug,
        }
