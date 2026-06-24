# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for querying the USD stage."""

from __future__ import annotations
"""查询USD阶段的公用事项。"""

import logging
import re
from collections.abc import Callable

import omni
import omni.kit.app
from pxr import Sdf, Usd, UsdPhysics

from .stage import get_current_stage

# import logger
logger = logging.getLogger(__name__)


def get_next_free_prim_path(path: str, stage: Usd.Stage | None = None) -> str:
    """Gets a new prim path that doesn't exist in the stage given a base path.

    If the given path doesn't exist in the stage already, it returns the given path. Otherwise,
    it appends a suffix with an incrementing number to the given path.

    Args:
        path: The base prim path to check.
        stage: The stage to check. Defaults to the current stage.

    Returns:
        A new path that is guaranteed to not exist on the current stage

    Example:
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # given the stage: /World/Cube, /World/Cube_01.
        >>> # Get the next available path for /World/Cube
        >>> sim_utils.get_next_free_prim_path("/World/Cube")
        /World/Cube_02
    """
    """得到一个新的prim路径，没有在阶段的基础路径。

    如果已在阶段没有给定的路径，则返回给定的路径。
    否则，它将一个增量数的后音添加到给定的路径上。

    参数：
        path: 要检查prim基地路径。
        stage: 我们要检查。
               默认的状态。

    返回：
        在目前的阶段，保证不存在的新道路

    示例：
        >>> import isaaclab.sim as sim_utils
        >>>
        >>> # given the stage: /World/Cube, /World/Cube_01.
        >>> # Get the next available path for /World/Cube
        >>> sim_utils.get_next_free_prim_path("/World/Cube")
        /世界/立方_02
    """
    # get current stage
    stage = get_current_stage() if stage is None else stage
    # get next free path
    return omni.usd.get_stage_next_free_path(stage, path, True)


def get_first_matching_ancestor_prim(
    prim_path: str | Sdf.Path,
    predicate: Callable[[Usd.Prim], bool],
    stage: Usd.Stage | None = None,
) -> Usd.Prim | None:
    """Gets the first ancestor prim that passes the predicate function.

    This function walks up the prim hierarchy starting from the target prim and returns the first ancestor prim
    that passes the predicate function. This includes the prim itself if it passes the predicate.

    Args:
        prim_path: The path of the prim in the stage.
        predicate: The function to test the prims against. It takes a prim as input and returns a boolean.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        The first ancestor prim that passes the predicate. If no ancestor prim passes the predicate, it returns None.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """
    """得到了通过预言函数的第一个祖先prim。

    这个函数从目标prim开始 prim等级上升，并返回通过预言函数的第一个祖先prim。
    这包括prim本身，如果它通过预言。

    参数：
        prim_path: 在舞台上prim的路径。
        predicate: 测试prims的功能。
                   它将prim作为输入，然后返回布尔式。
        stage: 在prim存在的阶段。
               在 None 上默认设置，此时使用当前阶段。

    返回：
        第一个通过预言的祖先prim。
        如果没有祖先prim通过预言，它返回None。

    异常：
        ValueError: 如果prim路径不是全球 (i.e:不以"/"开始)。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # make paths str type if they aren't already
    prim_path = str(prim_path)
    # check if prim path is global
    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")
    # get prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")

    # walk up to find the first matching ancestor prim
    ancestor_prim = prim
    while ancestor_prim and ancestor_prim.IsValid():
        # check if prim passes predicate
        if predicate(ancestor_prim):
            return ancestor_prim
        # get parent prim
        ancestor_prim = ancestor_prim.GetParent()

    # If no ancestor prim passes the predicate, return None
    return None


def get_first_matching_child_prim(
    prim_path: str | Sdf.Path,
    predicate: Callable[[Usd.Prim], bool],
    stage: Usd.Stage | None = None,
    traverse_instance_prims: bool = True,
) -> Usd.Prim | None:
    """Recursively get the first USD Prim at the path string that passes the predicate function.

    This function performs a depth-first traversal of the prim hierarchy starting from
    :attr:`prim_path`, returning the first prim that satisfies the provided :attr:`predicate`.
    It optionally supports traversal through instance prims, which are normally skipped in standard USD
    traversals.

    USD instance prims are lightweight copies of prototype scene structures and are not included
    in default traversals unless explicitly handled. This function allows traversing into instances
    when :attr:`traverse_instance_prims` is set to :attr:`True`.

    .. versionchanged:: 2.3.0

        Added :attr:`traverse_instance_prims` to control whether to traverse instance prims.
        By default, instance prims are now traversed.

    Args:
        prim_path: The path of the prim in the stage.
        predicate: The function to test the prims against. It takes a prim as input and returns a boolean.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.
        traverse_instance_prims: Whether to traverse instance prims. Defaults to True.

    Returns:
        The first prim on the path that passes the predicate. If no prim passes the predicate, it returns None.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """
    """在路径字符串中，一次性地得到第一个USD Prime，

    这个函数从:attr:`prim_path`开始执行prim等级的第一次深度穿越，返回满足提供的:attr:`predicate`的第一个prim。
    它可通过实例prims进行横跨，通常在标准USD横跨中被跳过。

    USD实例prims是原型场景结构的轻量拷贝，除非明确处理，否则不会被包含在默认的穿越中。
    这种函数允许通过实例
    when :attr:`traverse_instance_prims`设置为:attr:`True`。

    ..
    改版: 2.3.0

        Added :attr:`traverse_instance_prims`控制是否通过实例prims。
        默认情况下，实例prims现在被穿过。

    参数：
        prim_path: 在舞台上prim的路径。
        predicate: 测试prims的功能。
                   它将prim作为输入，然后返回布尔式。
        stage: 在prim存在的阶段。
               在 None 上默认设置，此时使用当前阶段。
        traverse_instance_prims: 是否通过实例prims。
                                 默认为 True。

    返回：
        通过预言的路径上的第一个prim。
        如果没有prim通过预言，它返回None。

    异常：
        ValueError: 如果prim路径不是全球 (i.e:不以"/"开始)。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # make paths str type if they aren't already
    prim_path = str(prim_path)
    # check if prim path is global
    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")
    # get prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")
    # iterate over all prims under prim-path
    all_prims = [prim]
    while len(all_prims) > 0:
        # get current prim
        child_prim = all_prims.pop(0)
        # check if prim passes predicate
        if predicate(child_prim):
            return child_prim
        # add children to list
        if traverse_instance_prims:
            all_prims += child_prim.GetFilteredChildren(Usd.TraverseInstanceProxies())
        else:
            all_prims += child_prim.GetChildren()
    return None


def get_all_matching_child_prims(
    prim_path: str | Sdf.Path,
    predicate: Callable[[Usd.Prim], bool] = lambda _: True,
    depth: int | None = None,
    stage: Usd.Stage | None = None,
    traverse_instance_prims: bool = True,
) -> list[Usd.Prim]:
    """Performs a search starting from the root and returns all the prims matching the predicate.

    This function performs a depth-first traversal of the prim hierarchy starting from
    :attr:`prim_path`, returning all prims that satisfy the provided :attr:`predicate`. It optionally
    supports traversal through instance prims, which are normally skipped in standard USD traversals.

    USD instance prims are lightweight copies of prototype scene structures and are not included
    in default traversals unless explicitly handled. This function allows traversing into instances
    when :attr:`traverse_instance_prims` is set to :attr:`True`.

    .. versionchanged:: 2.3.0

        Added :attr:`traverse_instance_prims` to control whether to traverse instance prims.
        By default, instance prims are now traversed.

    Args:
        prim_path: The root prim path to start the search from.
        predicate: The predicate that checks if the prim matches the desired criteria. It takes a prim as input
            and returns a boolean. Defaults to a function that always returns True.
        depth: The maximum depth for traversal, should be bigger than zero if specified.
            Defaults to None (i.e: traversal happens till the end of the tree).
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.
        traverse_instance_prims: Whether to traverse instance prims. Defaults to True.

    Returns:
        A list containing all the prims matching the predicate.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """
    """执行从根开始的搜索，并返回所有与预言相匹配的prims。

    这个函数从:attr:`prim_path`开始执行prim等级的第一次深度穿越，返回满足提供的:attr:`predicate`的所有prims。
    它可通过实例prims进行横跨，通常在标准USD横跨中被跳过。

    USD实例prims是原型场景结构的轻量拷贝，除非明确处理，否则不会被包含在默认的穿越中。
    这种函数允许通过实例
    when :attr:`traverse_instance_prims`设置为:attr:`True`。

    ..
    改版: 2.3.0

        Added :attr:`traverse_instance_prims`控制是否通过实例prims。
        默认情况下，实例prims现在被穿过。

    参数：
        prim_path: 根prim路径开始搜索。
        predicate: 这种预言检查prim是否符合所需的标准。
                   它将prim作为输入，然后返回布尔式。
                   总是返回True的函数。
        depth: 如果指定，可穿越的最大深度应大于零。
               在None中默认出现故障 (i.e:穿越到树尾发生)。
        stage: 在prim存在的阶段。
               在 None 上默认设置，此时使用当前阶段。
        traverse_instance_prims: 是否通过实例prims。
                                 默认为 True。

    返回：
        一个包含所有与预言符相匹配的prims的列表。

    异常：
        ValueError: 如果prim路径不是全球 (i.e:不以"/"开始)。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # make paths str type if they aren't already
    prim_path = str(prim_path)
    # check if prim path is global
    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")
    # get prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")
    # check if depth is valid
    if depth is not None and depth <= 0:
        raise ValueError(f"Depth must be bigger than zero, got {depth}.")

    # iterate over all prims under prim-path
    # list of tuples (prim, current_depth)
    all_prims_queue = [(prim, 0)]
    output_prims = []
    while len(all_prims_queue) > 0:
        # get current prim
        child_prim, current_depth = all_prims_queue.pop(0)
        # check if prim passes predicate
        if predicate(child_prim):
            output_prims.append(child_prim)
        # add children to list
        if depth is None or current_depth < depth:
            # resolve prims under the current prim
            if traverse_instance_prims:
                children = child_prim.GetFilteredChildren(Usd.TraverseInstanceProxies())
            else:
                children = child_prim.GetChildren()
            # add children to list
            all_prims_queue += [(child, current_depth + 1) for child in children]

    return output_prims


def find_first_matching_prim(prim_path_regex: str, stage: Usd.Stage | None = None) -> Usd.Prim | None:
    """Find the first matching prim in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        The first prim that matches input expression. If no prim matches, returns None.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """
    """根据输入regex表达式，在阶段找到第一个匹配的prim。

    参数：
        prim_path_regex: 为prim路径的regex表达式。
        stage: 在prim存在的阶段。
               在 None 上默认设置，此时使用当前阶段。

    返回：
        第一个与输入表达式匹配的prim。
        如果没有prim匹配，则返回None。

    异常：
        ValueError: 如果prim路径不是全球 (i.e:不以"/"开始)。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # check prim path is global
    if not prim_path_regex.startswith("/"):
        raise ValueError(f"Prim path '{prim_path_regex}' is not global. It must start with '/'.")
    prim_path_regex = _normalize_legacy_wildcard_pattern(prim_path_regex)
    # need to wrap the token patterns in '^' and '$' to prevent matching anywhere in the string
    pattern = f"^{prim_path_regex}$"
    compiled_pattern = re.compile(pattern)
    # obtain matching prim (depth-first search)
    for prim in stage.Traverse():
        # check if prim passes predicate
        if compiled_pattern.match(prim.GetPath().pathString) is not None:
            return prim
    return None


def _normalize_legacy_wildcard_pattern(prim_path_regex: str) -> str:
    """Convert legacy '*' wildcard usage to '.*' and warn users."""
    """转换传统的'*'野生卡使用为'*'并警告用户。"""
    fixed_regex = re.sub(r"(?<![\\\.])\*", ".*", prim_path_regex)
    if fixed_regex != prim_path_regex:
        logger.warning(
            "Using '*' as a wildcard in prim path regex is deprecated; automatically converting '%s' to '%s'. "
            "Please update your pattern to use '.*' explicitly.",
            prim_path_regex,
            fixed_regex,
        )
    return fixed_regex


def find_matching_prims(prim_path_regex: str, stage: Usd.Stage | None = None) -> list[Usd.Prim]:
    """Find all the matching prims in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        A list of prims that match input expression.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """
    """根据输入regex表达式，在阶段找到所有匹配的prims。

    参数：
        prim_path_regex: 为prim路径的regex表达式。
        stage: 在prim存在的阶段。
               在 None 上默认设置，此时使用当前阶段。

    返回：
        一个与输入表达式相匹配的prims列表。

    异常：
        ValueError: 如果prim路径不是全球 (i.e:不以"/"开始)。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # normalize legacy wildcard pattern
    prim_path_regex = _normalize_legacy_wildcard_pattern(prim_path_regex)

    # check prim path is global
    if not prim_path_regex.startswith("/"):
        raise ValueError(f"Prim path '{prim_path_regex}' is not global. It must start with '/'.")
    # need to wrap the token patterns in '^' and '$' to prevent matching anywhere in the string
    tokens = prim_path_regex.split("/")[1:]
    tokens = [f"^{token}$" for token in tokens]
    # iterate over all prims in stage (breath-first search)
    all_prims = [stage.GetPseudoRoot()]
    output_prims = []
    for index, token in enumerate(tokens):
        token_compiled = re.compile(token)
        for prim in all_prims:
            for child in prim.GetAllChildren():
                if token_compiled.match(child.GetName()) is not None:
                    output_prims.append(child)
        if index < len(tokens) - 1:
            all_prims = output_prims
            output_prims = []
    return output_prims


def find_matching_prim_paths(prim_path_regex: str, stage: Usd.Stage | None = None) -> list[str]:
    """Find all the matching prim paths in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        A list of prim paths that match input expression.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
    """
    """根据输入regex表达，在阶段找到所有匹配的prim路径。

    参数：
        prim_path_regex: 为prim路径的regex表达式。
        stage: 在prim存在的阶段。
               在 None 上默认设置，此时使用当前阶段。

    返回：
        一列与输入表达相匹配的prim路径。

    异常：
        ValueError: 如果prim路径不是全球 (i.e:不以"/"开始)。
    """
    # obtain matching prims
    output_prims = find_matching_prims(prim_path_regex, stage)
    # convert prims to prim paths
    output_prim_paths = []
    for prim in output_prims:
        output_prim_paths.append(prim.GetPath().pathString)
    return output_prim_paths


def find_global_fixed_joint_prim(
    prim_path: str | Sdf.Path, check_enabled_only: bool = False, stage: Usd.Stage | None = None
) -> UsdPhysics.Joint | None:
    """Find the fixed joint prim under the specified prim path that connects the target to the simulation world.

    A joint is a connection between two bodies. A fixed joint is a joint that does not allow relative motion
    between the two bodies. When a fixed joint has only one target body, it is considered to attach the body
    to the simulation world.

    This function finds the fixed joint prim that has only one target under the specified prim path. If no such
    fixed joint prim exists, it returns None.

    Args:
        prim_path: The prim path to search for the fixed joint prim.
        check_enabled_only: Whether to consider only enabled fixed joints. Defaults to False.
            If False, then all joints (enabled or disabled) are considered.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        The fixed joint prim that has only one target. If no such fixed joint prim exists, it returns None.

    Raises:
        ValueError: If the prim path is not global (i.e: does not start with '/').
        ValueError: If the prim path does not exist on the stage.
    """
    """在指定的prim路径下找到固定关联prim，将目标连接到仿真世界。

    关节是两个身体之间的连接。
    固定关节是不允许两体之间的相对运动的关节。
    当固定关节只有一个目标身体时，它被认为将身体连接到仿真世界。

    这个函数在指定prim路径下找到固定关节prim的目标。
    如果没有这样的固定关节prim，则返回None。

    参数：
        prim_path: 寻找固定关节prim的prim路径。
        check_enabled_only: 考虑是否只允许固定关节。
                            默认为 False。
                            如果 False，则将考虑所有关节 (启用或禁用)。
        stage: 在prim存在的阶段。
               在 None 上默认设置，此时使用当前阶段。

    返回：
        固定的关节prim只有一个目标。
        如果没有这样的固定关节prim，则返回None。

    异常：
        ValueError: 如果prim路径不是全球 (i.e:不以"/"开始)。
        ValueError: 如果prim路线不在舞台上。
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # check prim path is global
    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")

    # check if prim exists
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")

    fixed_joint_prim = None
    # we check all joints under the root prim and classify the asset as fixed base if there exists
    # a fixed joint that has only one target (i.e. the root link).
    for prim in Usd.PrimRange(prim):
        # note: ideally checking if it is FixedJoint would have been enough, but some assets use "Joint" as the
        # schema name which makes it difficult to distinguish between the two.
        joint_prim = UsdPhysics.Joint(prim)
        if joint_prim:
            # if check_enabled_only is True, we only consider enabled joints
            if check_enabled_only and not joint_prim.GetJointEnabledAttr().Get():
                continue
            # check body 0 and body 1 exist
            body_0_exist = joint_prim.GetBody0Rel().GetTargets() != []
            body_1_exist = joint_prim.GetBody1Rel().GetTargets() != []
            # if either body 0 or body 1 does not exist, we have a fixed joint that connects to the world
            if not (body_0_exist and body_1_exist):
                fixed_joint_prim = joint_prim
                break

    return fixed_joint_prim
