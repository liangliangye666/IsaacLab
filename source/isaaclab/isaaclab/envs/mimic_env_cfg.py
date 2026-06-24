# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Base MimicEnvCfg object for Isaac Lab Mimic data generation.
"""
"""基于IacakLab仿真数据生成的MimicEnvCfg对象。
"""

import enum

from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab.utils import configclass


@configclass
class DataGenConfig:
    """Configuration settings for data generation processes within the Isaac Lab Mimic environment."""
    """在 Isaac Lab Mimic 环境中生成数据过程的配置设置。"""

    name: str = "demo"
    """The name of the data generation process. Defaults to "demo"."""
    """数据生成过程名称
    默认调整"调试"。
    """

    generation_guarantee: bool = True
    """Whether to retry generation until generation_num_trials successful demos have been generated.

    If True, generation will be retried until generation_num_trials successful demos are created.
    If False, generation will stop after generation_num_trails, regardless of success.
    """
    """在generation_num_trials成功演示生成之前，

    如果True，将重新尝试到generation_num_trials成功的演示。
    如果False，generation_num_trails后的生成将停止，不管成功。
    """

    generation_keep_failed: bool = False
    """Whether to keep failed generation trials.

    Keeping failed demonstrations is useful for visualizing and debugging low success rates.
    """
    """我们是否要保持失败的世代试验。

    保持失败的示范是可视化和调试低成功率的有用。
    """

    max_num_failures: int = 50
    """Maximum number of failures allowed before stopping generation."""
    """在停产前允许的最大故障数量。"""

    seed: int = 1
    """Seed for randomization to ensure reproducibility."""
    """种子用于随机化，以确保可复制性。"""

    """The following configuration values can be changed on the command line, and only serve as defaults."""
    """下列配置值可以在命令行上更改，并且只作为默认。"""

    source_dataset_path: str = None
    """Path to the source dataset for mimic generation."""
    """仿真生成的源数据集的路径。"""

    generation_path: str = None
    """Path where the generated data will be saved."""
    """创建数据将存储的路径。"""

    generation_num_trials: int = 10
    """Number of trials to be generated."""
    """需要进行的试验数量。"""

    task_name: str = None
    """Name of the task being configured."""
    """配置任务的名称"""

    """The following configurations are advanced and do not usually need to be changed."""
    """下列配置是先进的，通常不需要更改。"""

    generation_select_src_per_subtask: bool = False
    """Whether to select source data per subtask.

    Note:
        This requires subtasks to be properly temporally constrained, and may require
        additional subtasks to allow for time synchronization.
    """
    """选择每个子任务的源数据。

    说明：
        这需要适当的临时限制子任务，并且可能需要额外的子任务以允许时间同步。
    """

    generation_select_src_per_arm: bool = False
    """Whether to select source data per arm."""
    """选择每一个手臂的源数据。"""

    generation_transform_first_robot_pose: bool = False
    """Whether to transform the first robot pose during generation."""
    """在生产过程中，是否要改变第一位机器人姿势。"""

    generation_interpolate_from_last_target_pose: bool = True
    """Whether to interpolate from last target pose."""
    """是否从最后一个目标姿势中进行回合。"""

    use_skillgen: bool = False
    """Whether to use skillgen to generate motion trajectories."""
    """是否使用技能生成运动轨迹。"""

    use_navigation_controller: bool = False
    """Whether to use a navigation controller to generate loco-manipulation trajectories."""
    """是否使用导航控制器来生成场景操纵轨迹。"""


@configclass
class SubTaskConfig:
    """
    Configuration settings for specifying subtasks used in Mimic environments.
    """
    """在仿真环境中使用的子任务的配置设置。
    """

    """Mandatory options that should be defined for every subtask."""
    """每个子任务都应该定义的强制性选项。"""

    object_ref: str = None
    """Reference to the object involved in this subtask.

    Set to None if no object is involved (this is rarely the case).
    """
    """引用本次任务中的对象。

    设置为None如果没有对象 (这种情况很少发生)。
    """

    subtask_term_signal: str = None
    """Subtask termination signal name."""
    """任务终止信号名称。"""

    """Advanced options for tuning the generation results."""
    """提高生成结果的优势。"""

    selection_strategy: str = "random"
    """Strategy for selecting a subtask segment.

    Can be one of:
        * 'random'
        * 'nearest_neighbor_object'
        * 'nearest_neighbor_robot_distance'

    Note:
        For 'nearest_neighbor_object' and 'nearest_neighbor_robot_distance', the subtask needs
        to have 'object_ref' set to a value other than 'None'. These strategies typically yield
        higher success rates than the default 'random' strategy when object_ref is set.
    """
    """选择子任务细节的战略。

    可能是:
        * "随机"
        * "nearest_neighbor_object"
        * "nearest_neighbor_robot_distance"

    说明：
        对于"nearest_neighbor_object"和"nearest_neighbor_robot_distance"，子任务必须设置"object_ref"为"None"以外的值。
        在设置object_ref时，这些策略通常比默认"随机"策略更高的成功率。
    """

    selection_strategy_kwargs: dict = {}
    """Additional arguments to the selected strategy. See details on each strategy in
    source/isaaclab_mimic/isaaclab_mimic/datagen/selection_strategy.py
    Arguments will be passed through to the `select_source_demo` method."""
    """选择战略的额外参数。
    查看来源/isaaclab_mimic/isaaclab_mimic/datagen/selection_strategy.py参数中的每个策略的详细信息将转移到`select_source_demo
    `方法。
    """

    first_subtask_start_offset_range: tuple = (0, 0)
    """Range for start offset of the first subtask."""
    """在第一个子任务的开始抵消范围。"""

    subtask_start_offset_range: tuple = (0, 0)
    """Range for start offset of the subtask (only used if use_skillgen is True)

    Note: This value overrides the first_subtask_start_offset_range when skillgen is enabled
    """
    """部分任务的启动偏移范围 (仅用于use_skillgen是True)

    Note: 当 skillgen 启用时，这个值超过first_subtask_start_offset_range
    """

    subtask_term_offset_range: tuple = (0, 0)
    """Range for offsetting subtask termination."""
    """补偿子任务终止范围。"""

    action_noise: float = 0.03
    """Amplitude of action noise applied."""
    """应用于动作噪音的幅度。"""

    num_interpolation_steps: int = 5
    """Number of steps for interpolation between waypoints."""
    """路线点之间的插射步骤数"""

    num_fixed_steps: int = 0
    """Number of fixed steps for the subtask."""
    """部分任务的固定步骤数量。"""

    apply_noise_during_interpolation: bool = False
    """Whether to apply noise during interpolation."""
    """在插射过程中是否应使用噪音。"""

    description: str = ""
    """Description of the subtask"""
    """副任务的描述"""

    next_subtask_description: str = ""
    """Instructions for the next subtask"""
    """下一个子任务的指示"""


class SubTaskConstraintType(enum.IntEnum):
    """Enum for subtask constraint types."""
    """对于子任务限制类型的Enum。"""

    SEQUENTIAL = 0
    COORDINATION = 1

    _SEQUENTIAL_FORMER = 2
    _SEQUENTIAL_LATTER = 3


class SubTaskConstraintCoordinationScheme(enum.IntEnum):
    """Enum for coordination schemes."""
    """协调方案的信息。"""

    REPLAY = 0
    TRANSFORM = 1
    TRANSLATE = 2


@configclass
class SubTaskConstraintConfig:
    """
    Configuration settings for specifying subtask constraints used in multi-eef Mimic environments.
    """
    """在多eef仿真环境中使用的子任务限制的配置设置。
    """

    eef_subtask_constraint_tuple: list[tuple[str, int]] = (("", 0), ("", 0))
    """List of associated subtasks tuples in order.

    The first element of the tuple refers to the eef name.
    The second element of the tuple refers to the subtask index of the eef.
    """
    """相关的子任务列表

    元组的第一个元素指的是eef名称。
    元组的第二个元素指的是eef的子任务索引。
    """

    constraint_type: SubTaskConstraintType = None
    """Type of constraint to apply between subtasks."""
    """在子任务之间应应用的限制类型。"""

    sequential_min_time_diff: int = -1
    """Minimum time difference between two sequential subtasks finishing.

    The second subtask will execute until sequential_min_time_diff steps left in its subtask trajectory
    and wait until the first (preconditioned) subtask is finished to continue executing the rest.
    If set to -1, the second subtask will start only after the first subtask is finished.
    """
    """两个连续的子任务完成的最小时间差异。

    第二个子任务将执行直到sequential_min_time_diff步骤离开其子任务轨迹，然后等到第一个 (预定) 子任务完成，然后继续执行其余的任务。
    如果设置为 -1，第二次子任务将在第一次子任务完成后才开始。
    """

    coordination_scheme: SubTaskConstraintCoordinationScheme = SubTaskConstraintCoordinationScheme.REPLAY
    """Scheme to use for coordinating subtasks."""
    """用于协调子任务的方案。"""

    coordination_scheme_pos_noise_scale: float = 0.0
    """Scale of position noise to apply during coordination."""
    """在协调过程中应使用的位置噪音尺度。"""

    coordination_scheme_rot_noise_scale: float = 0.0
    """Scale of rotation noise to apply during coordination."""
    """在协调过程中应使用的旋转噪音尺度。"""

    coordination_synchronize_start: bool = False
    """Whether subtasks should start at the same time."""
    """部分任务是否应该同时开始。"""

    def generate_runtime_subtask_constraints(self):
        """
        Populate expanded task constraints dictionary based on the task constraint config.
        The task constraint config contains the configurations set by the user. While the
        task_constraints_dict contains flags used to implement the constraint logic in this class.

        The task_constraint_configs may include the following types:
        - "sequential"
        - "coordination"

        For a "sequential" constraint:
            - Data from task_constraint_configs is added to task_constraints_dict as "sequential former"
              task constraint.
            - The opposite constraint, of type "sequential latter", is also added to task_constraints_dict.
            - Additionally, a ("fulfilled", Bool) key-value pair is added to task_constraints_dict.
            - This is used to check if the precondition (i.e., the sequential former task) has been met.
            - Until the "fulfilled" flag in "sequential latter" is set by "sequential former",
                the "sequential latter" subtask will remain paused.

        For a "coordination" constraint:
            - Data from task_constraint_configs is added to task_constraints_dict.
            - The opposite constraint, of type "coordination", is also added to task_constraints_dict.
            - The number of synchronous steps is set to the minimum of subtask_len and concurrent_subtask_len.
            - This ensures both concurrent tasks end at the same time step.
            - A "selected_src_demo_ind" and "transform" field are used to ensure the transforms used by
              both subtasks are the same.
        """
        """基于任务限制配置的扩展任务限制字典。
        任务限制配置包含用户设置的配置。
        而task_constraints_dict包含用于实现限制逻辑的旗。

        task_constraint_configs可能包括以下类型:
        - "顺序"
        - "协调"

        对于"顺序"的约束:
            - 从task_constraint_configs的数据被添加到task_constraints_dict作为"序列前"任务限制。
            - 对于 task_constraints_dict，也添加了相反的约束，类型是"后续后者"。
            - 此外，在task_constraints_dict中添加一个"实现"，Bool"键值对。
            - 这用于检查是否满足了先决条件 (i.e.，序列前任务)。
            - 在"后续后者"中的"完成"标志被"后续前者"设置之前，"后续后者"子任务将暂停。

        对于"协调"的限制:
            - task_constraint_configs的数据被添加到task_constraints_dict。
            - 对于 task_constraints_dict，也添加了"协调"类型的相反的限制。
            - 同步步骤的数量设置为最小的subtask_len和concurrent_subtask_len。
            - 这确保两项同时任务同时完成。
            - 用"selected_src_demo_ind"和"转换"字段来确保两个子任务所使用的转换是相同的。
        """
        task_constraints_dict = dict()
        if self.constraint_type == SubTaskConstraintType.SEQUENTIAL:
            constrained_task_spec_key, constrained_subtask_ind = self.eef_subtask_constraint_tuple[1]
            assert isinstance(constrained_subtask_ind, int)
            pre_condition_task_spec_key, pre_condition_subtask_ind = self.eef_subtask_constraint_tuple[0]
            assert isinstance(pre_condition_subtask_ind, int)
            assert (
                constrained_task_spec_key,
                constrained_subtask_ind,
            ) not in task_constraints_dict, "only one constraint per subtask allowed"
            task_constraints_dict[(constrained_task_spec_key, constrained_subtask_ind)] = dict(
                type=SubTaskConstraintType._SEQUENTIAL_LATTER,
                pre_condition_task_spec_key=pre_condition_task_spec_key,
                pre_condition_subtask_ind=pre_condition_subtask_ind,
                min_time_diff=self.sequential_min_time_diff,
                fulfilled=False,
            )
            task_constraints_dict[(pre_condition_task_spec_key, pre_condition_subtask_ind)] = dict(
                type=SubTaskConstraintType._SEQUENTIAL_FORMER,
                constrained_task_spec_key=constrained_task_spec_key,
                constrained_subtask_ind=constrained_subtask_ind,
            )
        elif self.constraint_type == SubTaskConstraintType.COORDINATION:
            constrained_task_spec_key, constrained_subtask_ind = self.eef_subtask_constraint_tuple[0]
            assert isinstance(constrained_subtask_ind, int)
            concurrent_task_spec_key, concurrent_subtask_ind = self.eef_subtask_constraint_tuple[1]
            assert isinstance(concurrent_subtask_ind, int)
            if self.coordination_scheme is None:
                raise ValueError("Coordination scheme must be specified.")
            assert (
                constrained_task_spec_key,
                constrained_subtask_ind,
            ) not in task_constraints_dict, "only one constraint per subtask allowed"
            task_constraints_dict[(constrained_task_spec_key, constrained_subtask_ind)] = dict(
                concurrent_task_spec_key=concurrent_task_spec_key,
                concurrent_subtask_ind=concurrent_subtask_ind,
                type=SubTaskConstraintType.COORDINATION,
                fulfilled=False,
                finished=False,
                selected_src_demo_ind=None,
                coordination_scheme=self.coordination_scheme,
                coordination_scheme_pos_noise_scale=self.coordination_scheme_pos_noise_scale,
                coordination_scheme_rot_noise_scale=self.coordination_scheme_rot_noise_scale,
                coordination_synchronize_start=self.coordination_synchronize_start,
                synchronous_steps=None,  # to be calculated at runtime
            )
            task_constraints_dict[(concurrent_task_spec_key, concurrent_subtask_ind)] = dict(
                concurrent_task_spec_key=constrained_task_spec_key,
                concurrent_subtask_ind=constrained_subtask_ind,
                type=SubTaskConstraintType.COORDINATION,
                fulfilled=False,
                finished=False,
                selected_src_demo_ind=None,
                coordination_scheme=self.coordination_scheme,
                coordination_scheme_pos_noise_scale=self.coordination_scheme_pos_noise_scale,
                coordination_scheme_rot_noise_scale=self.coordination_scheme_rot_noise_scale,
                coordination_synchronize_start=self.coordination_synchronize_start,
                synchronous_steps=None,  # to be calculated at runtime
            )
        else:
            raise ValueError("Constraint type not supported.")

        return task_constraints_dict


@configclass
class MimicEnvCfg:
    """
    Configuration class for the Mimic environment integration.

    This class consolidates various configuration aspects for the
    Isaac Lab Mimic data generation pipeline.
    """
    """仿真环境集成的配置类。

    该类整合了艾萨克实验室仿真数据生成管道的各种配置方面。
    """

    # Overall configuration for the data generation
    datagen_config: DataGenConfig = DataGenConfig()

    # Dictionary of list of subtask configurations for each end-effector.
    # Keys are end-effector names.
    subtask_configs: dict[str, list[SubTaskConfig]] = {}

    # List of configurations for subtask constraints
    task_constraint_configs: list[SubTaskConstraintConfig] = []

    # Optional recorder configuration
    mimic_recorder_config: RecorderManagerBaseCfg | None = None
