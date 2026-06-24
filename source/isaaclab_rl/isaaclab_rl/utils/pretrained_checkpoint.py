# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for handling various pre-trained checkpoint tasks"""
"""处理各种预训练的检查站任务的子模块"""

import glob
import json
import os

import carb.settings

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: F401

PRETRAINED_CHECKPOINTS_ASSET_ROOT_DIR = carb.settings.get_settings().get(
    "/persistent/isaaclab/asset_root/pretrained_checkpoints"
)
"""Path to the root directory on the Nucleus Server."""
"""核服务器的根目录。"""

PRETRAINED_CHECKPOINT_PATH = str(PRETRAINED_CHECKPOINTS_ASSET_ROOT_DIR) + "/Isaac/IsaacLab/PretrainedCheckpoints"
"""URL for where we store all the pre-trained checkpoints"""
"""URL对于我们存储的所有预训练的检查站"""

WORKFLOWS = ["rl_games", "rsl_rl", "sb3", "skrl"]
"""The supported workflows for pre-trained checkpoints"""
"""预训练检查站的支持工作流"""

WORKFLOW_TRAINER = {w: f"scripts/reinforcement_learning/{w}/train.py" for w in WORKFLOWS}
"""A dict mapping workflow to their training program path"""
"""一个将工作流映射到他们的培训计划路径"""

WORKFLOW_PLAYER = {w: f"scripts/reinforcement_learning/{w}/play.py" for w in WORKFLOWS}
"""A dict mapping workflow to their play program path"""
"""命令将工作流映射到他们的播放程序路径"""

WORKFLOW_PRETRAINED_CHECKPOINT_FILENAMES = {
    "rl_games": "checkpoint.pth",
    "rsl_rl": "checkpoint.pt",
    "sb3": "checkpoint.zip",
    "skrl": "checkpoint.pt",
}
"""The filename for checkpoints used by the different workflows"""
"""各工作流所使用的检查站的文件名"""

WORKFLOW_EXPERIMENT_NAME_VARIABLE = {
    "rl_games": "agent.params.config.name",
    "rsl_rl": "agent.experiment_name",
    "sb3": None,
    "skrl": "agent.agent.experiment.directory",
}
"""Maps workflow to the agent variable name that determines the logging directory logs/{workflow}/{variable}"""
"""地图工作流向确定记录目录日志/{workflow}/{variable}的代理变量名称"""


def has_pretrained_checkpoints_asset_root_dir() -> bool:
    """Returns True if and only if /persistent/isaaclab/asset_root/pretrained_checkpoints exists"""
    """返回True如果只有存在 /persistent/isaaclab/asset_root/pretrained_checkpoints"""
    return PRETRAINED_CHECKPOINTS_ASSET_ROOT_DIR is not None


def get_log_root_path(workflow: str, task_name: str) -> str:
    """Returns the absolute path where the logs are written for a specific workflow and task_name"""
    """返回记录为特定工作流的绝对路径和task_name"""
    return os.path.abspath(os.path.join("logs", workflow, task_name))


def get_latest_job_run_path(workflow: str, task_name: str) -> str | None:
    """The local logs path of the most recent run of this workflow and task name"""
    """本工作流和任务名称最近运行的本地日志路径"""
    log_root_path = get_log_root_path(workflow, task_name)
    return _get_latest_file_or_directory(log_root_path)


def get_pretrained_checkpoint_path(workflow: str, task_name: str) -> str:
    """The local logs path where we get the pre-trained checkpoints from"""
    """我们从那里得到预训练的检查站"""

    path = get_latest_job_run_path(workflow, task_name)
    if not path:
        return None

    if workflow == "rl_games":
        return os.path.join(path, "nn", f"{task_name}.pth")
    elif workflow == "rsl_rl":
        return _get_latest_file_or_directory(path, "*.pt")  # type: ignore
    elif workflow == "sb3":
        return os.path.join(path, "model.zip")
    elif workflow == "skrl":
        return os.path.join(path, "checkpoints", "best_agent.pt")
    else:
        raise Exception(f"Unsupported workflow ({workflow})")


def get_pretrained_checkpoint_publish_path(workflow: str, task_name: str) -> str:
    """The path where pre-trained checkpoints are published to"""
    """预训练检查站公布的路径"""
    return os.path.join(
        PRETRAINED_CHECKPOINT_PATH, workflow, task_name, WORKFLOW_PRETRAINED_CHECKPOINT_FILENAMES[workflow]
    )


def get_published_pretrained_checkpoint_path(workflow: str, task_name: str) -> str:
    """The path where pre-trained checkpoints are fetched from"""
    """预训练过的检查站的路径"""
    return os.path.join(
        ISAACLAB_NUCLEUS_DIR,
        "PretrainedCheckpoints",
        workflow,
        task_name,
        WORKFLOW_PRETRAINED_CHECKPOINT_FILENAMES[workflow],
    )


def get_published_pretrained_checkpoint(workflow: str, task_name: str) -> str | None:
    """Gets the path for the pre-trained checkpoint.

    If the checkpoint is not cached locally then the file is downloaded.
    The cached path is then returned.

    Args:
        workflow: The workflow.
        task_name: The task name.

    Returns:
        The path.
    """
    """让我们进入预训练的检查站。

    如果检查点没有本地缓存，则文件会下载。
    然后返回缓存的路径。

    参数：
        workflow: 工作流。
        task_name: 任务名称。

    返回：
        路径。
    """
    ov_path = get_published_pretrained_checkpoint_path(workflow, task_name)
    download_dir = os.path.join(".pretrained_checkpoints", workflow, task_name)
    resume_path = os.path.join(download_dir, WORKFLOW_PRETRAINED_CHECKPOINT_FILENAMES[workflow])

    if not os.path.exists(resume_path):
        print(f"Fetching pre-trained checkpoint : {ov_path}")
        try:
            resume_path = retrieve_file_path(ov_path, download_dir)
        except Exception:
            print("A pre-trained checkpoint is currently unavailable for this task.")
            return None
    else:
        print("Using pre-fetched pre-trained checkpoint")
    return resume_path


def has_pretrained_checkpoint_job_run(workflow: str, task_name: str) -> bool:
    """Returns true if an experiment exists in the logs for the workflow and task"""
    """如果工作流和任务日志中存在实验，返回 true"""
    return os.path.exists(get_log_root_path(workflow, task_name))


def has_pretrained_checkpoint_job_finished(workflow: str, task_name: str) -> bool:
    """Returns true if an experiment has results which may or may not be final depending on workflow"""
    """如果实验结果可能是最终的或可能不是根据工作流"""
    local_path = get_pretrained_checkpoint_path(workflow, task_name)
    return local_path is not None and os.path.exists(local_path)


def get_pretrained_checkpoint_review_path(workflow: str, task_name: str) -> str | None:
    """The path of the review JSON file for a workflow and task"""
    """工作流和任务的审查JSON文件的路径"""
    run_path = get_latest_job_run_path(workflow, task_name)
    if not run_path:
        return None
    return os.path.join(run_path, "pretrained_checkpoint_review.json")


def get_pretrained_checkpoint_review(workflow: str, task_name: str) -> dict | None:
    """Returns the review JSON file as a dict if it exists"""
    """返回审查JSON文件，如果它存在"""
    review_path = get_pretrained_checkpoint_review_path(workflow, task_name)
    if not review_path:
        return None

    if os.path.exists(review_path):
        with open(review_path) as f:
            return json.load(f)

    return None


def _get_latest_file_or_directory(path: str, pattern: str = "*"):
    """Returns the path to the most recently modified file or directory at a path matching an optional pattern"""
    """返回最近修改的文件或目录的路径，与可选模式匹配的路径"""
    g = glob.glob(f"{path}/{pattern}")
    if len(g):
        return max(g, key=os.path.getmtime)
    return None
