# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass


@configclass
class PbtCfg:
    """
    Population-Based Training (PBT) configuration.

    leaders are policies with score > max(mean + threshold_std*std, mean + threshold_abs).
    underperformers are policies with score < min(mean - threshold_std*std, mean - threshold_abs).
    On replacement, selected hyperparameters are mutated multiplicatively in [change_min, change_max].
    """
    """人口培训 (PBT) 配置。

    领导者是有分数>最大的策略threshold_std*std，平均+threshold_abs)。
    低绩效的策略是以分数 < min(平均 - threshold_std*std，平均 - threshold_abs)。
    在更换时，在 [change_min， change_max] 中选择的超参数被突变多倍。
    """

    enabled: bool = False
    """Enable/disable PBT logic."""
    """启用/禁用PBT逻辑。"""

    policy_idx: int = 0
    """Index of this learner in the population (unique in [0, num_policies-1])."""
    """在人口中学习者的索引 (在 [0， num_policies-1] 中唯一)。"""

    num_policies: int = 8
    """Total number of learners participating in PBT."""
    """参与PBT的学生总数。"""

    directory: str = ""
    """Root directory for PBT artifacts (checkpoints, metadata)."""
    """PBT文物根目录 (检查点，元数据)。"""

    workspace: str = "pbt_workspace"
    """Subfolder under the training dir to isolate this PBT run."""
    """在训练 dir下面的子文件将PBT运行隔离。"""

    objective: str = "Episode_Reward/success"
    """The key in info returned by env.step that pbt measures to determine leaders and underperformers,
    If reward is stationary, using the term that corresponds to task success is usually enough, when reward
    are non-stationary, consider uses better objectives.
    """
    """返回信息的关键env.step如果奖励是静止的，使用与任务成功相匹配的项通常足够，当奖励是非静止的，考虑使用更好的目标。
    """

    interval_steps: int = 100_000
    """Environment steps between PBT iterations (save, compare, replace/mutate)."""
    """在 PBT 代之间环境步骤 (保存，比较，更换/变化)。"""

    threshold_std: float = 0.10
    """Std-based margin k in max(mean ± k·std, mean ± threshold_abs) for leader/underperformer cuts."""
    """基于Std的边缘 k在max ((平均 ± k·std，平均 ± threshold_abs) 中，用于领导/低表现者减产。"""

    threshold_abs: float = 0.05
    """Absolute margin A in max(mean ± threshold_std·std, mean ± A) for leader/underperformer cuts."""
    """对于领先/低绩效的减产，绝对边缘A在最大的平均 ± threshold_std·std，平均 ± A。"""

    mutation_rate: float = 0.25
    """Per-parameter probability of mutation when a policy is replaced."""
    """在策略更换时，因参数发生突变的概率。"""

    change_range: tuple[float, float] = (1.1, 2.0)
    """Lower and upper bound of multiplicative change factor (sampled in [change_min, change_max])."""
    """乘变因子的下边和上边 (在 [change_min， change_max] 中采样)。"""

    mutation: dict[str, str] = {}
    """Mutation strings indicating which parameter will be mutated when pbt restart
    example:
        {
            "agent.params.config.learning_rate": "mutate_float"
            "agent.params.config.grad_norm": "mutate_float"
            "agent.params.config.entropy_coef": "mutate_float"
        }
    """
    """突变字符串表示pbt重启时将发生哪个参数突变
    example: { "agent.params.config.learning_rate": "mutate_float" "agent.params.config.grad_norm":
             "mutate_float" "agent.params.config.entropy_coef": "mutate_float" }
    """
