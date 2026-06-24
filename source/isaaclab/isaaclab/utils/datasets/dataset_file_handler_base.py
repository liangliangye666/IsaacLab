# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

from .episode_data import EpisodeData


class DatasetFileHandlerBase(ABC):
    """Abstract class for handling dataset files."""
    """处理数据集文件的抽象类。"""

    def __init__(self):
        """Initializes the dataset file handler."""
        """启动数据集文件处理器。"""
        pass

    @abstractmethod
    def open(self, file_path: str, mode: str = "r"):
        """Open a file."""
        """打开一个文件。"""
        return NotImplementedError

    @abstractmethod
    def create(self, file_path: str, env_name: str = None):
        """Create a new file."""
        """创建一个新的文件。"""
        return NotImplementedError

    @abstractmethod
    def get_env_name(self) -> str | None:
        """Get the environment name."""
        """找一个环境名称。"""
        return NotImplementedError

    @abstractmethod
    def write_episode(self, episode: EpisodeData):
        """Write episode data to the file."""
        """在文件中写出事件数据。"""
        return NotImplementedError

    @abstractmethod
    def flush(self):
        """Flush the file."""
        """掉文件。"""
        return NotImplementedError

    @abstractmethod
    def close(self):
        """Close the file."""
        """关闭文件。"""
        return NotImplementedError

    @abstractmethod
    def load_episode(self, episode_name: str) -> EpisodeData | None:
        """Load episode data from the file."""
        """在文件中加载事件数据。"""
        return NotImplementedError

    @abstractmethod
    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        """在档案中查看事件数。"""
        return NotImplementedError
