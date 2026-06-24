# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for versioning."""

from __future__ import annotations
"""版本编辑的实用功能。"""

import functools

from packaging.version import Version


@functools.lru_cache(maxsize=1)
def get_isaac_sim_version() -> Version:
    """Get the Isaac Sim version as a Version object, cached for performance.

    This function wraps :func:`isaacsim.core.version.get_version()` and caches the result
    to avoid repeated file I/O operations. The underlying Isaac Sim function reads from
    a file each time it's called, which can be slow when called frequently.

    Returns:
        A :class:`packaging.version.Version` object representing the Isaac Sim version.
        This object supports rich comparison operators (<, <=, >, >=, ==, !=).

    Example:
        >>> from isaaclab.utils import get_isaac_sim_version
        >>> from packaging.version import Version
        >>>
        >>> isaac_version = get_isaac_sim_version()
        >>> print(isaac_version)
        5.0.0
        >>>
        >>> # Natural version comparisons
        >>> if isaac_version >= Version("5.0.0"):
        ...     print("Using Isaac Sim 5.0 or later")
        >>>
        >>> # Access components
        >>> print(isaac_version.major, isaac_version.minor, isaac_version.micro)
        5 0 0
    """
    """作为一个版本对象，以预存性能。

    这个函数将:func:`isaacsim.core.version.get_version()`包裹起来，并缓存结果以避免重复文件 I/O 操作。
    基础的Isaac Sim函数每次调用时都会从文件中读取，

    返回：
        A :类:`packaging.version.Version`对象代表了Isaac Sim版本。
        这个对象支持丰富的比较运算符 (<， <=， >， >=， ==， !=)。

    示例：
        >>> from isaaclab.utils import get_isaac_sim_version
        >>> from packaging.version import Version
        >>>
        >>> isaac_version = get_isaac_sim_version()
        >>> print(isaac_version)
        5.0.0
        >>>
        >>> # Natural version comparisons
        >>> if isaac_version >= Version("5.0.0"):
        ...     print("Using Isaac Sim 5.0 or later")
        >>>
        >>> # Access components
        >>> print(isaac_version.major, isaac_version.minor, isaac_version.micro)
        5 0 0
    """
    from isaacsim.core.version import get_version

    version_tuple = get_version()
    # version_tuple[2] = major (year), [3] = minor (release), [4] = micro (patch)
    return Version(f"{version_tuple[2]}.{version_tuple[3]}.{version_tuple[4]}")


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings and return the comparison result.

    The version strings are expected to be in the format "x.y.z" where x, y,
    and z are integers. The version strings are compared lexicographically.

    .. note::
        This function is provided for backward compatibility. For new code,
        prefer using :class:`packaging.version.Version` objects directly with
        comparison operators (``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``).

    Args:
        v1: The first version string.
        v2: The second version string.

    Returns:
        An integer indicating the comparison result:

        - :attr:`1` if v1 is greater
        - :attr:`-1` if v2 is greater
        - :attr:`0` if v1 and v2 are equal

    Example:
        >>> from isaaclab.utils import compare_versions
        >>> compare_versions("5.0.0", "4.5.0")
        1
        >>> compare_versions("4.5.0", "5.0.0")
        -1
        >>> compare_versions("5.0.0", "5.0.0")
        0
        >>>
        >>> # Better: use Version objects directly
        >>> from packaging.version import Version
        >>> Version("5.0.0") > Version("4.5.0")
        True
    """
    """将两个版本字符串进行比较，然后返回比较结果。

    预计版本字符串将以"x.y.z"格式进行，其中x，y和z是整数。
    版本字符串是用词汇来比较的。

    .. 说明::
        这种功能为后退兼容性提供。
        对于新代码，最好直接使用:class:`packaging.version.Version`对象与比较运算符
        (``<``，``<=``，``>``，``>=``，``==``，``!=``)。

    参数：
        v1: 第一个版本串。
        v2: 第二个版本串。

    返回：
        一个表示比较结果的整数:

        - 如果v1大于:attr:`1`
        - 如果v2大于:attr:`-1`
        - :attr:`0`如果v1和v2均等

    示例：
        >>> from isaaclab.utils import compare_versions
        >>> compare_versions("5.0.0", "4.5.0")
        1
        >>> compare_versions("4.5.0", "5.0.0")
        -1
        >>> compare_versions("5.0.0", "5.0.0")
        0
        >>>
        >>> # Better: use Version objects directly
        >>> from packaging.version import Version
        >>> Version("5.0.0") > Version("4.5.0")
        True
    """
    ver1 = Version(v1)
    ver2 = Version(v2)

    if ver1 > ver2:
        return 1
    elif ver1 < ver2:
        return -1
    else:
        return 0
