# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generator that does nothing."""

from __future__ import annotations
"""没有任何操作的命令生成器。"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from .commands_cfg import NullCommandCfg


class NullCommand(CommandTerm):
    """Command generator that does nothing.

    This command generator does not generate any commands. It is used for environments that do not
    require any commands.
    """
    """命令发电器什么都不做。

    这种命令生成器不会生成任何命令。
    它用于不需要任何命令的环境。
    """

    cfg: NullCommandCfg
    """Configuration for the command generator."""
    """命令生成器的配置。"""

    def __str__(self) -> str:
        msg = "NullCommand:\n"
        msg += "\tCommand dimension: N/A\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """
    """产品
    """

    @property
    def command(self):
        """Null command.

        Raises:
            RuntimeError: No command is generated. Always raises this error.
        """
        """没有命令。

        异常：
            RuntimeError: 没有命令。
                          总是提起这个错误。
        """
        raise RuntimeError("NullCommandTerm does not generate any commands.")

    """
    Operations.
    """
    """操作。
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        return {}

    def compute(self, dt: float):
        pass

    """
    Implementation specific functions.
    """
    """具体执行功能。
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass
