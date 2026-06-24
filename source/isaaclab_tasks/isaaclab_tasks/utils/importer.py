# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utility for importing all modules in a package recursively."""

from __future__ import annotations
"""一个子模块，可用于递归地进口一个包装中的所有模块。"""

import importlib
import pkgutil
import sys


def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None):
    """Import all sub-packages in a package recursively.

    It is easier to use this function to import all sub-packages in a package recursively
    than to manually import each sub-package.

    It replaces the need of the following code snippet on the top of each package's ``__init__.py`` file:

    .. code-block:: python

        import .locomotion.velocity
        import .manipulation.reach
        import .manipulation.lift

    Args:
        package_name: The package name.
        blacklist_pkgs: The list of blacklisted packages to skip. Defaults to None,
            which means no packages are blacklisted.
    """
    """一个包装中的所有子包装进行递归进口。

    使用此功能来递归地进口一包中的所有子包装，比手动地进口每个子包装更容易。

    它取代了每个包的``__init__.py``文件顶部的以下代码摘录:

    .. code-block:: python

        import .locomotion.velocity
        import .manipulation.reach
        import .manipulation.lift

    参数：
        package_name: 包装名称。
        blacklist_pkgs: 黑名单上列的包裹列表。
                        默认为 None，这意味着没有包被列入黑名单。
    """
    # Default blacklist
    if blacklist_pkgs is None:
        blacklist_pkgs = []
    # Import the package itself
    package = importlib.import_module(package_name)
    # Import all Python files
    for _ in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
        pass


"""
Internal helpers.
"""
"""内部助理。
"""


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror: callable | None = None,
    blacklist_pkgs: list[str] | None = None,
):
    """Yields ModuleInfo for all modules recursively on path, or, if path is None, all accessible modules.

    Note:
        This function is a modified version of the original ``pkgutil.walk_packages`` function. It adds
        the ``blacklist_pkgs`` argument to skip blacklisted packages. Please refer to the original
        ``pkgutil.walk_packages`` function for more details.

    """
    """在路径上的所有模块的收益率为ModuleInfo，或者，如果路径是None，所有可访问的模块。

    说明：
        这个函数是原始``pkgutil.walk_packages``函数的修改版本。
        它添加了``blacklist_pkgs``参数来跳过黑名单的包裹。
        请参阅原始``pkgutil.walk_packages``函数以了解更多细节。
    """
    # Default blacklist
    if blacklist_pkgs is None:
        blacklist_pkgs = []

    def seen(p: str, m: dict[str, bool] = {}) -> bool:
        """Check if a package has been seen before."""
        """检查是否曾见过包裹。"""
        if p in m:
            return True
        m[p] = True
        return False

    for info in pkgutil.iter_modules(path, prefix):
        # check blacklisted
        if any([black_pkg_name in info.name for black_pkg_name in blacklist_pkgs]):
            continue

        # yield the module info
        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path: list = getattr(sys.modules[info.name], "__path__", [])

                # don't traverse path items we've seen before
                path = [p for p in path if not seen(p)]

                yield from _walk_packages(path, info.name + ".", onerror, blacklist_pkgs)
