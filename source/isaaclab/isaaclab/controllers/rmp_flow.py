# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import torch

from isaacsim.core.prims import SingleArticulation

# enable motion generation extensions
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.robot_motion.lula")
enable_extension("isaacsim.robot_motion.motion_generation")

from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy
from isaacsim.robot_motion.motion_generation.lula.motion_policies import RmpFlow, RmpFlowSmoothed

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import retrieve_file_path


@configclass
class RmpFlowControllerCfg:
    """Configuration for RMP-Flow controller (provided through LULA library)."""
    """RMP-Flow控制器的配置 (通过LULA库提供)。"""

    name: str = "rmp_flow"
    """Name of the controller. Supported: "rmp_flow", "rmp_flow_smoothed". Defaults to "rmp_flow"."""
    """控制器的名字。
    支持: "rmp_flow"， "rmp_flow_smoothed"。
    在"rmp_flow"上默认。
    """
    config_file: str = MISSING
    """Path to the configuration file for the controller."""
    """控制器的配置文件的路径。"""
    urdf_file: str = MISSING
    """Path to the URDF model of the robot."""
    """进入机器人的URDF模型。"""
    collision_file: str = MISSING
    """Path to collision model description of the robot."""
    """机器人的撞击模式描述。"""
    frame_name: str = MISSING
    """Name of the robot frame for task space (must be present in the URDF)."""
    """任务空间机器人框架名称 (必须在URDF中存在)。"""
    evaluations_per_frame: float = MISSING
    """Number of substeps during Euler integration inside LULA world model."""
    """在LULA世界模型中的尤勒集成过程中的子步骤数。"""
    ignore_robot_state_updates: bool = False
    """If true, then state of the world model inside controller is rolled out. Defaults to False."""
    """如果是真的，那么控制器内部的世界模型的状态将被推出。
    默认为 False。
    """


class RmpFlowController:
    """Wraps around RMPFlow from IsaacSim for batched environments."""
    """包裹RMPFlow从IsaacSim到批量环境。"""

    def __init__(self, cfg: RmpFlowControllerCfg, device: str):
        """Initialize the controller.

        Args:
            cfg: The configuration for the controller.
            device: The device to use for computation.
        """
        """启动控制器。

        参数：
            cfg: 控制器的配置。
            device: 用于计算的设备。
        """
        # store input
        self.cfg = cfg
        self._device = device
        # display info
        print(f"[INFO]: Loading RMPFlow controller URDF from: {self.cfg.urdf_file}")

    """
    Properties.
    """
    """属性。
    """

    @property
    def num_actions(self) -> int:
        """Dimension of the action space of controller."""
        """控制器操作空间的尺寸"""
        return 7

    """
    Operations.
    """
    """操作。
    """

    def initialize(self, prim_paths_expr: str):
        """Initialize the controller.

        Args:
            prim_paths_expr: The expression to find the articulation prim paths.
        """
        """启动控制器。

        参数：
            prim_paths_expr: 找到关节 prim 路径的表达式。
        """
        # obtain the simulation time
        physics_dt = sim_utils.SimulationContext.instance().get_physics_dt()
        # find all prims
        self._prim_paths = sim_utils.find_matching_prim_paths(prim_paths_expr)
        self.num_robots = len(self._prim_paths)
        # resolve controller
        if self.cfg.name == "rmp_flow":
            controller_cls = RmpFlow
        elif self.cfg.name == "rmp_flow_smoothed":
            controller_cls = RmpFlowSmoothed
        else:
            raise ValueError(f"Unsupported controller in Lula library: {self.cfg.name}")
        # create all franka robots references and their controllers
        self.articulation_policies = list()
        for prim_path in self._prim_paths:
            # add robot reference
            robot = SingleArticulation(prim_path)
            robot.initialize()
            # download files if they are not local

            local_urdf_file = retrieve_file_path(self.cfg.urdf_file, force_download=True)
            local_collision_file = retrieve_file_path(self.cfg.collision_file, force_download=True)
            local_config_file = retrieve_file_path(self.cfg.config_file, force_download=True)

            # add controller
            rmpflow = controller_cls(
                robot_description_path=local_collision_file,
                urdf_path=local_urdf_file,
                rmpflow_config_path=local_config_file,
                end_effector_frame_name=self.cfg.frame_name,
                maximum_substep_size=physics_dt / self.cfg.evaluations_per_frame,
                ignore_robot_state_updates=self.cfg.ignore_robot_state_updates,
            )
            # wrap rmpflow to connect to the Franka robot articulation
            articulation_policy = ArticulationMotionPolicy(robot, rmpflow, physics_dt)
            self.articulation_policies.append(articulation_policy)
        # get number of active joints
        self.active_dof_names = self.articulation_policies[0].get_motion_policy().get_active_joints()
        self.num_dof = len(self.active_dof_names)
        # create buffers
        # -- for storing command
        self._command = torch.zeros(self.num_robots, self.num_actions, device=self._device)
        # -- for policy output
        self.dof_pos_target = torch.zeros((self.num_robots, self.num_dof), device=self._device)
        self.dof_vel_target = torch.zeros((self.num_robots, self.num_dof), device=self._device)

    def reset_idx(self, robot_ids: torch.Tensor = None):
        """Reset the internals."""
        """重置内部。"""
        # if no robot ids are provided, then reset all robots
        if robot_ids is None:
            robot_ids = torch.arange(self.num_robots, device=self._device)
        # reset policies for specified robots
        for index in robot_ids:
            self.articulation_policies[index].motion_policy.reset()

    def set_command(self, command: torch.Tensor):
        """Set target end-effector pose command."""
        """设置目标末端执行器姿势命令。"""
        # store command
        self._command[:] = command

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs inference with the controller.

        Returns:
            The target joint positions and velocity commands.
        """
        """通过控制器进行推断。

        返回：
            目标关节位置和速度指令。
        """
        # convert command to numpy
        command = self._command.cpu().numpy()
        # compute control actions
        for i, policy in enumerate(self.articulation_policies):
            # enable type-hinting
            policy: ArticulationMotionPolicy
            # set rmpflow target to be the current position of the target cube.
            policy.get_motion_policy().set_end_effector_target(
                target_position=command[i, 0:3], target_orientation=command[i, 3:7]
            )
            # apply action on the robot
            action = policy.get_next_articulation_action()
            # copy actions into buffer
            self.dof_pos_target[i, :] = torch.from_numpy(action.joint_positions[:]).to(self.dof_pos_target)
            self.dof_vel_target[i, :] = torch.from_numpy(action.joint_velocities[:]).to(self.dof_vel_target)

        return self.dof_pos_target, self.dof_vel_target
