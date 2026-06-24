# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
A collection of classes used to represent waypoints and trajectories.
"""
"""用于表示路线点和轨迹的类组。
"""

import asyncio
import inspect
from copy import deepcopy

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import TerminationTermCfg


class Waypoint:
    """
    Represents a single desired 6-DoF waypoint, along with corresponding gripper actuation for this point.
    """
    """代表一个所需的6-DoF路线点，以及对此点的相应的抓紧动力。
    """

    def __init__(self, pose, gripper_action, noise=None):
        """
        Args:
            pose (torch.Tensor): 4x4 pose target for robot controller
            gripper_action (torch.Tensor): gripper action for robot controller
            noise (float or None): action noise amplitude to apply during execution at this timestep
                (for arm actions, not gripper actions)
        """
        """参数：
            pose (torch.Tensor): 机器人控制器的4×4姿势目标
            gripper_action (torch.Tensor): 机器人控制器的抓住器操作
            noise (float or None): 在此时间步骤执行过程中应应用的动作噪声幅度 (用于臂动作，而不是抓住器动作)
        """
        self.pose = pose
        self.gripper_action = gripper_action
        self.noise = noise

    def __str__(self):
        """String representation of the waypoint."""
        """字符串表示路线点。"""
        return f"Waypoint:\n  Pose:\n{self.pose}\n"


class WaypointSequence:
    """
    Represents a sequence of Waypoint objects.
    """
    """代表了一系列的Waypoint物体。
    """

    def __init__(self, sequence=None):
        """
        Args:
            sequence (list or None): if provided, should be a list of Waypoint objects
        """
        """参数：
            sequence (list or None): 如果提供，应列出Waypoint对象列表
        """
        if sequence is None:
            self.sequence = []
        else:
            for waypoint in sequence:
                assert isinstance(waypoint, Waypoint)
            self.sequence = deepcopy(sequence)

    @classmethod
    def from_poses(cls, poses, gripper_actions, action_noise):
        """
        Instantiate a WaypointSequence object given a sequence of poses,
        gripper actions, and action noise.

        Args:
            poses (torch.Tensor): sequence of pose matrices of shape (T, 4, 4)
            gripper_actions (torch.Tensor): sequence of gripper actions
                that should be applied at each timestep of shape (T, D).
            action_noise (float or torch.Tensor): sequence of action noise
                magnitudes that should be applied at each timestep. If a
                single float is provided, the noise magnitude will be
                constant over the trajectory.
        """
        """设置一个WaypointSequence对象，给出一系列姿势，抓住器操作和动作噪音。

        参数：
            poses (torch.Tensor): 形状的姿势矩阵序列 (T， 4， 4)
            gripper_actions (torch.Tensor): 在每个形状时间步骤 (T，D) 中应应用的抓住器操作序列。
            action_noise (float or torch.Tensor): 每个时间步骤都应应用的动作噪音大小序列。
                                                  如果提供单一的浮动，噪声大小将在轨道上保持一致。
        """
        assert isinstance(action_noise, (float, torch.Tensor))

        # handle scalar to tensor conversion
        num_timesteps = poses.shape[0]
        if isinstance(action_noise, float):
            action_noise = action_noise * torch.ones((num_timesteps, 1), dtype=torch.float32)
        action_noise = action_noise.reshape(-1, 1)

        # make WaypointSequence instance
        sequence = [
            Waypoint(
                pose=poses[t],
                gripper_action=gripper_actions[t],
                noise=action_noise[t, 0],
            )
            for t in range(num_timesteps)
        ]
        return cls(sequence=sequence)

    def get_poses(self):
        poses = []
        for waypoint in self.sequence:
            poses.append(waypoint.pose[:2, 3])
        return poses

    def __len__(self):
        # length of sequence
        return len(self.sequence)

    def __getitem__(self, ind):
        """
        Returns waypoint at index.

        Returns:
            waypoint (Waypoint instance)
        """
        """返回索引的路线点。

        返回：
            路线点 (路线点实例)
        """
        return self.sequence[ind]

    def __add__(self, other):
        """
        Defines addition (concatenation) of sequences
        """
        """定义序列的加算 (结)
        """
        return WaypointSequence(sequence=(self.sequence + other.sequence))

    def __str__(self):
        """Prints all waypoints in the sequence."""
        """在序列中打印所有路线点。"""
        output = []
        for idx, waypoint in enumerate(self.sequence):
            output.append(f"Waypoint {idx}: {waypoint}")
        return "\n".join(output)

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        """顺序返回最后一个路线点。

        返回：
            路线点 (路线点实例)
        """
        return deepcopy(self.sequence[-1])

    def split(self, ind):
        """
        Splits this sequence into 2 pieces, the part up to time index @ind, and the
        rest. Returns 2 WaypointSequence objects.
        """
        """这条序列分为2个部分，部分是时间索引@ind，其余部分。
        返回2个WaypointSequence对象。
        """
        seq_1 = self.sequence[:ind]
        seq_2 = self.sequence[ind:]
        return WaypointSequence(sequence=seq_1), WaypointSequence(sequence=seq_2)


class WaypointTrajectory:
    """
    A sequence of WaypointSequence objects that corresponds to a full 6-DoF trajectory.
    """
    """顺序 WaypointSequence 对象，对应一个完整的 6-DoF轨迹。
    """

    def __init__(self):
        self.waypoint_sequences = []

    def __len__(self):
        # sum up length of all waypoint sequences
        return sum(len(s) for s in self.waypoint_sequences)

    def __getitem__(self, ind):
        """
        Returns waypoint at time index.

        Returns:
            waypoint (Waypoint instance)
        """
        """返回时间索引的路径点。

        返回：
            路线点 (路线点实例)
        """
        assert len(self.waypoint_sequences) > 0
        assert (ind >= 0) and (ind < len(self))

        # find correct waypoint sequence we should index
        end_ind = 0
        for seq_ind in range(len(self.waypoint_sequences)):
            start_ind = end_ind
            end_ind += len(self.waypoint_sequences[seq_ind])
            if (ind >= start_ind) and (ind < end_ind):
                break

        # index within waypoint sequence
        return self.waypoint_sequences[seq_ind][ind - start_ind]

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        """顺序返回最后一个路线点。

        返回：
            路线点 (路线点实例)
        """
        return self.waypoint_sequences[-1].last_waypoint

    def get_poses(self):
        poses = []
        for waypoint_sequence in self.waypoint_sequences:
            for waypoint in waypoint_sequence:
                poses.append(waypoint.pose[:2, 3])
        return poses

    def add_waypoint_sequence(self, sequence):
        """
        Directly append sequence to list (no interpolation).

        Args:
            sequence (WaypointSequence instance): sequence to add
        """
        """直接将序列添加到列表中 (无回合)。

        参数：
            sequence (WaypointSequence instance): 连接序列
        """
        assert isinstance(sequence, WaypointSequence)
        self.waypoint_sequences.append(sequence)

    def add_waypoint_sequence_for_target_pose(
        self,
        pose,
        gripper_action,
        num_steps,
        skip_interpolation=False,
        action_noise=0.0,
    ):
        """
        Adds a new waypoint sequence corresponding to a desired target pose. A new WaypointSequence
        will be constructed consisting of @num_steps intermediate Waypoint objects. These can either
        be constructed with linear interpolation from the last waypoint (default) or be a
        constant set of target poses (set @skip_interpolation to True).

        Args:
            pose (torch.Tensor): 4x4 target pose

            gripper_action (torch.Tensor): value for gripper action

            num_steps (int): number of action steps when trying to reach this waypoint. Will
                add intermediate linearly interpolated points between the last pose on this trajectory
                and the target pose, so that the total number of steps is @num_steps.

            skip_interpolation (bool): if True, keep the target pose fixed and repeat it @num_steps
                times instead of using linearly interpolated targets.

            action_noise (float): scale of random gaussian noise to add during action execution (e.g.
                when @execute is called)
        """
        """添加一个新的路点序列，与所需的目标姿势相符。
        新的WaypointSequence将由@num_steps中间的Waypoint对象构建。
        这些可以从最后的路线点 (默认) 构建以线性插图，或者是目标姿势的恒定集合 (设置 @skip_interpolation到 True)。

        参数：
            pose (torch.Tensor): 4x4目标姿势

            gripper_action (torch.Tensor): 抓住器作用的值

            num_steps (int): 在试图达到这个路线点时采取的动作步骤数量。
                             将在这个轨道上最后的姿势和目标姿势之间添加中间线性插入点，使步骤的总数为 @num_steps。

            skip_interpolation (bool): 如果True，保持目标姿势固定，并重复 @num_steps次，而不是使用线性插入目标。

            action_noise (float): 在执行操作时添加随机高斯声的规模 (当调用@execute时e.g.)
        """
        if len(self.waypoint_sequences) == 0:
            assert skip_interpolation, "cannot interpolate since this is the first waypoint sequence"

        if skip_interpolation:
            # repeat the target @num_steps times
            assert num_steps is not None
            poses = pose.unsqueeze(0).repeat((num_steps, 1, 1))
            gripper_actions = gripper_action.unsqueeze(0).repeat((num_steps, 1))
        else:
            # linearly interpolate between the last pose and the new waypoint
            last_waypoint = self.last_waypoint
            poses, num_steps_2 = PoseUtils.interpolate_poses(
                pose_1=last_waypoint.pose,
                pose_2=pose,
                num_steps=num_steps,
            )
            assert num_steps == num_steps_2
            gripper_actions = gripper_action.unsqueeze(0).repeat((num_steps + 2, 1))
            # make sure to skip the first element of the new path, which already exists on the current trajectory path
            poses = poses[1:]
            gripper_actions = gripper_actions[1:]

        # add waypoint sequence for this set of poses
        sequence = WaypointSequence.from_poses(
            poses=poses,
            gripper_actions=gripper_actions,
            action_noise=action_noise,
        )
        self.add_waypoint_sequence(sequence)

    def pop_first(self):
        """
        Removes first waypoint in first waypoint sequence and returns it. If the first waypoint
        sequence is now empty, it is also removed.

        Returns:
            waypoint (Waypoint instance)
        """
        """取消第一个路线点在第一个路线点序列中，然后返回它。
        如果第一个路线线序列现在是空的，它也会被删除。

        返回：
            路线点 (路线点实例)
        """
        first, rest = self.waypoint_sequences[0].split(1)
        if len(rest) == 0:
            # remove empty waypoint sequence
            self.waypoint_sequences = self.waypoint_sequences[1:]
        else:
            # update first waypoint sequence
            self.waypoint_sequences[0] = rest
        return first

    def merge(
        self,
        other,
        num_steps_interp=None,
        num_steps_fixed=None,
        action_noise=0.0,
    ):
        """
        Merge this trajectory with another (@other).

        Args:
            other (WaypointTrajectory object): the other trajectory to merge into this one

            num_steps_interp (int or None): if not None, add a waypoint sequence that interpolates
                between the end of the current trajectory and the start of @other

            num_steps_fixed (int or None): if not None, add a waypoint sequence that has constant
                target poses corresponding to the first target pose in @other

            action_noise (float): noise to use during the interpolation segment
        """
        """合并这个轨迹与另一个 (@other)。

        参数：
            other (WaypointTrajectory object): 其他轨迹将融入这个轨迹

            num_steps_interp (int or None): 如果不是None，则添加一个路线点序列，该序列在当前轨迹的结束和 @other 的开始之间进行交互

            num_steps_fixed (int or None): 如果不是None，则添加一个具有与 @other中的第一个目标姿势相匹配的恒定目标姿势的路点序列

            action_noise (float): 在插射段中使用的噪音
        """
        need_interp = (num_steps_interp is not None) and (num_steps_interp > 0)
        need_fixed = (num_steps_fixed is not None) and (num_steps_fixed > 0)
        use_interpolation_segment = need_interp or need_fixed

        if use_interpolation_segment:
            # pop first element of other trajectory
            other_first = other.pop_first()

            # Get first target pose of other trajectory.
            # The interpolated segment will include this first element as its last point.
            target_for_interpolation = other_first[0]

            if need_interp:
                # interpolation segment
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_interp,
                    action_noise=action_noise,
                    skip_interpolation=False,
                )

            if need_fixed:
                # segment of constant target poses equal to @other's first target pose

                # account for the fact that we pop'd the first element of
                # @other in anticipation of an interpolation segment
                num_steps_fixed_to_use = num_steps_fixed if need_interp else (num_steps_fixed + 1)
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_fixed_to_use,
                    action_noise=action_noise,
                    skip_interpolation=True,
                )

            # make sure to preserve noise from first element of other trajectory
            self.waypoint_sequences[-1][-1].noise = target_for_interpolation.noise

        # concatenate the trajectories
        self.waypoint_sequences += other.waypoint_sequences

    def get_full_sequence(self):
        """
        Returns the full sequence of waypoints in the trajectory.

        Returns:
            sequence (WaypointSequence instance)
        """
        """返回轨迹中的通路点的全部序列。

        返回：
            序列 (WaypointSequence实例)
        """
        return WaypointSequence(sequence=[waypoint for seq in self.waypoint_sequences for waypoint in seq.sequence])


class MultiWaypoint:
    """
    A collection of Waypoint objects for multiple end effectors in the environment.
    """
    """环境中的多个末端执行器的Waypoint对象集合。
    """

    def __init__(self, waypoints: dict[str, Waypoint]):
        """
        Args:
            waypoints (dict): a dictionary of waypionts of end effectors
        """
        """参数：
            waypoints (dict): 末端执行器的路线指标字典
        """
        self.waypoints = waypoints

    async def execute(
        self,
        env: ManagerBasedRLMimicEnv,
        success_term: TerminationTermCfg,
        env_id: int = 0,
        env_action_queue: asyncio.Queue | None = None,
    ):
        """
        Executes the multi-waypoint eef actions in the environment.

        Args:
            env: The environment to execute the multi-waypoint actions in.
            success_term: The termination term to check for task success.
            env_id: The environment ID to execute the multi-waypoint actions in.
            env_action_queue: The asyncio queue to put the action into.

        Returns:
            A dictionary containing the state, observation, action, and success of the multi-waypoint actions.
        """
        """在环境中执行多个方向的有效动作。

        参数：
            env: 执行多方向动作的环境。
            success_term: 终止时间检查任务是否成功。
            env_id: 环境 ID执行多方向操作。
            env_action_queue: 让我们在线观看。

        返回：
            一个包含多方向动作的状态，观测，动作和成功的字典。
        """
        # current state
        state = env.scene.get_state(is_relative=True)

        # construct action from target poses and gripper actions
        target_eef_pose_dict = {eef_name: waypoint.pose for eef_name, waypoint in self.waypoints.items()}
        gripper_action_dict = {eef_name: waypoint.gripper_action for eef_name, waypoint in self.waypoints.items()}
        if "action_noise_dict" in inspect.signature(env.target_eef_pose_to_action).parameters:
            action_noise_dict = {eef_name: waypoint.noise for eef_name, waypoint in self.waypoints.items()}
            play_action = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_eef_pose_dict,
                gripper_action_dict=gripper_action_dict,
                action_noise_dict=action_noise_dict,
                env_id=env_id,
            )
        else:
            # calling user-defined env.target_eef_pose_to_action() with noise parameter is deprecated
            # (replaced by action_noise_dict)
            play_action = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_eef_pose_dict,
                gripper_action_dict=gripper_action_dict,
                noise=max([waypoint.noise for waypoint in self.waypoints.values()]),
                env_id=env_id,
            )

        if play_action.dim() == 1:
            play_action = play_action.unsqueeze(0)  # Reshape with additional env dimension

        # step environment
        if env_action_queue is None:
            obs, _, _, _, _ = env.step(play_action)
        else:
            await env_action_queue.put((env_id, play_action[0]))
            await env_action_queue.join()
            obs = env.obs_buf

        success = bool(success_term.func(env, **success_term.params)[env_id])

        result = dict(
            states=[state],
            observations=[obs],
            actions=[play_action],
            success=success,
        )
        return result
