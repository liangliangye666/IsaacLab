# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for transforming strings and regular expressions."""
"""包含用于转换字符串和常态表达式的工具的子模块。"""

import ast
import importlib
import inspect
import re
from collections.abc import Callable, Sequence
from typing import Any

"""
String formatting.
"""
"""字符串格式化。
"""


def to_camel_case(snake_str: str, to: str = "cC") -> str:
    """Converts a string from snake case to camel case.

    Args:
        snake_str: A string in snake case (i.e. with '_')
        to: Convention to convert string to. Defaults to "cC".

    Raises:
        ValueError: Invalid input argument `to`, i.e. not "cC" or "CC".

    Returns:
        A string in camel-case format.
    """
    """转换一个字符串从蛇案子到驼案子。

    参数：
        snake_str: 蛇箱中的字符串 (i.e.含'_')
        to: 转换字符串为"公约"。
            在"cC"上默认。

    异常：
        ValueError: 无效输入参数 `to`， i.e。 不是"cC"或"CC"。

    返回：
        一个 cam驼形状的字符串。
    """
    # check input is correct
    if to not in ["cC", "CC"]:
        msg = "to_camel_case(): Choose a valid `to` argument (CC or cC)"
        raise ValueError(msg)
    # convert string to lower case and split
    components = snake_str.lower().split("_")
    if to == "cC":
        # We capitalize the first letter of each component except the first one
        # with the 'title' method and join them together.
        return components[0] + "".join(x.title() for x in components[1:])
    else:
        # Capitalize first letter in all the components
        return "".join(x.title() for x in components)


def to_snake_case(camel_str: str) -> str:
    """Converts a string from camel case to snake case.

    Args:
        camel_str: A string in camel case.

    Returns:
        A string in snake case (i.e. with '_')
    """
    """从驼子转换为蛇子。

    参数：
        camel_str: 一个 string驼子中的绳子。

    返回：
        蛇箱中的字符串 (i.e.含'_')
    """
    camel_str = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", camel_str).lower()


def string_to_slice(s: str):
    """Convert a string representation of a slice to a slice object.

    Args:
        s: The string representation of the slice.

    Returns:
        The slice object.
    """
    """转换一个切片的字符串成切片对象。

    参数：
        s: 切片的字符串表示。

    返回：
        切片物体。
    """
    # extract the content inside the slice()
    match = re.match(r"slice\((.*),(.*),(.*)\)", s)
    if not match:
        raise ValueError(f"Invalid slice string format: {s}")

    # extract start, stop, and step values
    start_str, stop_str, step_str = match.groups()

    # convert 'None' to None and other strings to integers
    start = None if start_str == "None" else int(start_str)
    stop = None if stop_str == "None" else int(stop_str)
    step = None if step_str == "None" else int(step_str)

    # create and return the slice object
    return slice(start, stop, step)


"""
String <-> Callable operations.
"""
"""字符串 <-> 可调用操作。
"""


def is_lambda_expression(name: str) -> bool:
    """Checks if the input string is a lambda expression.

    Args:
        name: The input string.

    Returns:
        Whether the input string is a lambda expression.
    """
    """检查输入字符串是否是 lambda 表达式。

    参数：
        name: 输入链。

    返回：
        输入字符串是否是 lambda 表达式。
    """
    try:
        ast.parse(name)
        return isinstance(ast.parse(name).body[0], ast.Expr) and isinstance(ast.parse(name).body[0].value, ast.Lambda)
    except SyntaxError:
        return False


def callable_to_string(value: Callable) -> str:
    """Converts a callable object to a string.

    Args:
        value: A callable object.

    Raises:
        ValueError: When the input argument is not a callable object.

    Returns:
        A string representation of the callable object.
    """
    """将可调用的对象转换为字符串。

    参数：
        value: 一个可调用的物体。

    异常：
        ValueError: 当输入参数不是可调用的对象时。

    返回：
        一个可调用对象的字符串表示。
    """
    # check if callable
    if not callable(value):
        raise ValueError(f"The input argument is not callable: {value}.")
    # check if lambda function
    if value.__name__ == "<lambda>":
        # we resolve the lambda expression by checking the source code and extracting the line with lambda expression
        # we also remove any comments from the line
        lambda_line = inspect.getsourcelines(value)[0][0].strip().split("lambda")[1].strip().split(",")[0]
        lambda_line = re.sub(r"#.*$", "", lambda_line).rstrip()
        return f"lambda {lambda_line}"
    else:
        # get the module and function name
        module_name = value.__module__
        function_name = value.__name__
        # return the string
        return f"{module_name}:{function_name}"


def string_to_callable(name: str) -> Callable:
    """Resolves the module and function names to return the function.

    Args:
        name: The function name. The format should be 'module:attribute_name' or a
            lambda expression of format: 'lambda x: x'.

    Raises:
        ValueError: When the resolved attribute is not a function.
        ValueError: When the module cannot be found.

    Returns:
        Callable: The function loaded from the module.
    """
    """解决模块和函数名称以返回函数。

    参数：
        name: 函数名称。
              格式应是"模块:attribute_name"或格式的lambda表达式:"lambda x:x"。

    异常：
        ValueError: 当解决属性不是函数时。
        ValueError: 当模块无法找到时。

    返回：
        Callable: 从模块上载的函数。
    """
    try:
        if is_lambda_expression(name):
            callable_object = eval(name)
        else:
            mod_name, attr_name = name.split(":")
            mod = importlib.import_module(mod_name)
            callable_object = getattr(mod, attr_name)
        # check if attribute is callable
        if callable(callable_object):
            return callable_object
        else:
            raise AttributeError(f"The imported object is not callable: '{name}'")
    except (ValueError, ModuleNotFoundError) as e:
        msg = (
            f"Could not resolve the input string '{name}' into callable object."
            " The format of input should be 'module:attribute_name'.\n"
            f"Received the error:\n {e}."
        )
        raise ValueError(msg)


"""
Regex operations.
"""
"""雷杰克斯动作。
"""


def resolve_matching_names(
    keys: str | Sequence[str], list_of_strings: Sequence[str], preserve_order: bool = False
) -> tuple[list[int], list[str]]:
    """Match a list of query regular expressions against a list of strings and return the matched indices and names.

    When a list of query regular expressions is provided, the function checks each target string against each
    query regular expression and returns the indices of the matched strings and the matched strings.

    If the :attr:`preserve_order` is True, the ordering of the matched indices and names is the same as the order
    of the provided list of strings. This means that the ordering is dictated by the order of the target strings
    and not the order of the query regular expressions.

    If the :attr:`preserve_order` is False, the ordering of the matched indices and names is the same as the order
    of the provided list of query regular expressions.

    For example, consider the list of strings is ['a', 'b', 'c', 'd', 'e'] and the regular expressions are ['a|c', 'b'].
    If :attr:`preserve_order` is False, then the function will return the indices of the matched strings and the
    strings as: ([0, 1, 2], ['a', 'b', 'c']). When :attr:`preserve_order` is True, it will return them as:
    ([0, 2, 1], ['a', 'c', 'b']).

    Note:
        The function does not sort the indices. It returns the indices in the order they are found.

    Args:
        keys: A regular expression or a list of regular expressions to match the strings in the list.
        list_of_strings: A list of strings to match.
        preserve_order: Whether to preserve the order of the query keys in the returned values. Defaults to False.

    Returns:
        A tuple of lists containing the matched indices and names.

    Raises:
        ValueError: When multiple matches are found for a string in the list.
        ValueError: When not all regular expressions are matched.
    """
    """匹配查询常态表达式列表与字符串列表，然后返回匹配的索引和名称。

    当提供查询正则表达式列表时，该函数将每个目标字符串与每个查询正则表达式进行检查，并返回匹配字符串和匹配字符串的索引。

    如果:attr:`preserve_order`是True，则匹配的索引和名称的顺序与提供列表的顺序相同。
    这意味着排序是由目标字符串的顺序决定的，而不是查询的顺序。

    如果:attr:`preserve_order`是False，则匹配的索引和名称的顺序与提供的查询正则表达式列表的顺序相同。

    例如，考虑字符串列表是 ['a'， 'b'， 'c'， 'd'， 'e'] 和正则表达式是 ['a拼c'， 'b'。
    If :attr:`preserve_order`是False，然后函数将返回匹配的字符串的索引，
    字符串如:[0， 1， 2]， ['a'， 'b'， 'c'])。
    当:attr:`preserve_order`是True时，它将返回它们为: ([0， 2， 1]， ['a'， 'c'， 'b'])。

    说明：
        这项函数不会分类索引。
        它以找到的顺序返回索引。

    参数：
        keys: 一个正则表达式或一个正则表达式列表，以匹配列表中的字符串。
        list_of_strings: 一个配合的字符串列表。
        preserve_order: 在返回值中是否保留查询键的顺序。
                        默认为 False。

    返回：
        包含相匹配的索引和名称的列表。

    异常：
        ValueError: 在列表中找到多个符串匹配时。
        ValueError: 当所有正则表达式都不匹配时。
    """
    # resolve name keys
    if isinstance(keys, str):
        keys = [keys]
    # find matching patterns
    index_list = []
    names_list = []
    key_idx_list = []
    # book-keeping to check that we always have a one-to-one mapping
    # i.e. each target string should match only one regular expression
    target_strings_match_found = [None for _ in range(len(list_of_strings))]
    keys_match_found = [[] for _ in range(len(keys))]
    # loop over all target strings
    for target_index, potential_match_string in enumerate(list_of_strings):
        for key_index, re_key in enumerate(keys):
            if re.fullmatch(re_key, potential_match_string):
                # check if match already found
                if target_strings_match_found[target_index]:
                    raise ValueError(
                        f"Multiple matches for '{potential_match_string}':"
                        f" '{target_strings_match_found[target_index]}' and '{re_key}'!"
                    )
                # add to list
                target_strings_match_found[target_index] = re_key
                index_list.append(target_index)
                names_list.append(potential_match_string)
                key_idx_list.append(key_index)
                # add for regex key
                keys_match_found[key_index].append(potential_match_string)
    # reorder keys if they should be returned in order of the query keys
    if preserve_order:
        reordered_index_list = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(keys)):
            for key_idx_position, key_idx_entry in enumerate(key_idx_list):
                if key_idx_entry == key_index:
                    reordered_index_list[key_idx_position] = global_index
                    global_index += 1
        # reorder index and names list
        index_list_reorder = [None] * len(index_list)
        names_list_reorder = [None] * len(index_list)
        for idx, reorder_idx in enumerate(reordered_index_list):
            index_list_reorder[reorder_idx] = index_list[idx]
            names_list_reorder[reorder_idx] = names_list[idx]
        # update
        index_list = index_list_reorder
        names_list = names_list_reorder
    # check that all regular expressions are matched
    if not all(keys_match_found):
        # make this print nicely aligned for debugging
        msg = "\n"
        for key, value in zip(keys, keys_match_found):
            msg += f"\t{key}: {value}\n"
        msg += f"Available strings: {list_of_strings}\n"
        # raise error
        raise ValueError(
            f"Not all regular expressions are matched! Please check that the regular expressions are correct: {msg}"
        )
    # return
    return index_list, names_list


def resolve_matching_names_values(
    data: dict[str, Any],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
    strict: bool = True,
) -> tuple[list[int], list[str], list[Any]]:
    """Match a list of regular expressions in a dictionary against a list of strings and return
    the matched indices, names, and values.

    If the :attr:`preserve_order` is True, the ordering of the matched indices and names is the same as the order
    of the provided list of strings. This means that the ordering is dictated by the order of the target strings
    and not the order of the query regular expressions.

    If the :attr:`preserve_order` is False, the ordering of the matched indices and names is the same as the order
    of the provided list of query regular expressions.

    For example, consider the dictionary is {"a|d|e": 1, "b|c": 2}, the list of strings is ['a', 'b', 'c', 'd', 'e'].
    If :attr:`preserve_order` is False, then the function will return the indices of the matched strings, the
    matched strings, and the values as: ([0, 1, 2, 3, 4], ['a', 'b', 'c', 'd', 'e'], [1, 2, 2, 1, 1]). When
    :attr:`preserve_order` is True, it will return them as:
    ([0, 3, 4, 1, 2], ['a', 'd', 'e', 'b', 'c'], [1, 1, 1, 2, 2]).

    Args:
        data: A dictionary of regular expressions and values to match the strings in the list.
        list_of_strings: A list of strings to match.
        preserve_order: Whether to preserve the order of the query keys in the returned values. Defaults to False.
        strict: Whether to require that all keys in the dictionary get matched. Defaults to True.

    Returns:
        A tuple of lists containing the matched indices, names, and values.

    Raises:
        TypeError: When the input argument :attr:`data` is not a dictionary.
        ValueError: When multiple matches are found for a string in the dictionary.
        ValueError: When not all regular expressions in the data keys are matched (if strict is True).
    """
    """按字典中的常用表达式和字符串列表进行匹配，并返回匹配的指标，名称和值。

    如果:attr:`preserve_order`是True，则匹配的索引和名称的顺序与提供列表的顺序相同。
    这意味着排序是由目标字符串的顺序决定的，而不是查询的顺序。

    如果:attr:`preserve_order`是False，则匹配的索引和名称的顺序与提供的查询正则表达式列表的顺序相同。

    例如，考虑字典是{"a|d|e": 1， "b|c": 2}，字符串列表是 ['a'， 'b'， 'c'， 'd'， 'e'。
    If :attr:`preserve_order`是False，然后函数将返回匹配的字符串的索引，
    匹配的字符串，并以 ([0， 1， 2， 3， 4]， ['a'， 'b'， 'c'， 'd'， 'e']， [1， 2， 2， 1， 1]为等值。
    当:attr:`preserve_order`是True时，它将返回它们为:[0， 3， 4， 1， 2]， ['a'， 'd'， 'e'， 'b'， 'c']， [1， 1， 2， 2])。

    参数：
        data: 一个正则表达式和值字典，以匹配列表中的字符串。
        list_of_strings: 一个配合的字符串列表。
        preserve_order: 在返回值中是否保留查询键的顺序。
                        默认为 False。
        strict: 要求字典中的所有键都匹配。
                默认为 True。

    返回：
        一组包含相匹配的索引，名称和值的列表。

    异常：
        TypeError: 当输入参数:attr:`data`不是字典时。
        ValueError: 在字典中找到一个字符串的多个匹配时。
        ValueError: 如果数据键中的所有正则表达式都不匹配 (如果严格是True)。
    """
    # check valid input
    if not isinstance(data, dict):
        raise TypeError(f"Input argument `data` should be a dictionary. Received: {data}")
    # find matching patterns
    index_list = []
    names_list = []
    values_list = []
    key_idx_list = []
    # book-keeping to check that we always have a one-to-one mapping
    # i.e. each target string should match only one regular expression
    target_strings_match_found = [None for _ in range(len(list_of_strings))]
    keys_match_found = [[] for _ in range(len(data))]
    # loop over all target strings
    for target_index, potential_match_string in enumerate(list_of_strings):
        for key_index, (re_key, value) in enumerate(data.items()):
            if re.fullmatch(re_key, potential_match_string):
                # check if match already found
                if target_strings_match_found[target_index]:
                    raise ValueError(
                        f"Multiple matches for '{potential_match_string}':"
                        f" '{target_strings_match_found[target_index]}' and '{re_key}'!"
                    )
                # add to list
                target_strings_match_found[target_index] = re_key
                index_list.append(target_index)
                names_list.append(potential_match_string)
                values_list.append(value)
                key_idx_list.append(key_index)
                # add for regex key
                keys_match_found[key_index].append(potential_match_string)
    # reorder keys if they should be returned in order of the query keys
    if preserve_order:
        reordered_index_list = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(data)):
            for key_idx_position, key_idx_entry in enumerate(key_idx_list):
                if key_idx_entry == key_index:
                    reordered_index_list[key_idx_position] = global_index
                    global_index += 1
        # reorder index and names list
        index_list_reorder = [None] * len(index_list)
        names_list_reorder = [None] * len(index_list)
        values_list_reorder = [None] * len(index_list)
        for idx, reorder_idx in enumerate(reordered_index_list):
            index_list_reorder[reorder_idx] = index_list[idx]
            names_list_reorder[reorder_idx] = names_list[idx]
            values_list_reorder[reorder_idx] = values_list[idx]
        # update
        index_list = index_list_reorder
        names_list = names_list_reorder
        values_list = values_list_reorder
    # check that all regular expressions are matched
    if strict and not all(keys_match_found):
        # make this print nicely aligned for debugging
        msg = "\n"
        for key, value in zip(data.keys(), keys_match_found):
            msg += f"\t{key}: {value}\n"
        msg += f"Available strings: {list_of_strings}\n"
        # raise error
        raise ValueError(
            f"Not all regular expressions are matched! Please check that the regular expressions are correct: {msg}"
        )
    # return
    return index_list, names_list, values_list


def find_unique_string_name(initial_name: str, is_unique_fn: Callable[[str], bool]) -> str:
    """Find a unique string name based on the predicate function provided.
    The string is appended with "_N", where N is a natural number till the resultant string
    is unique.
    Args:
        initial_name (str): The initial string name.
        is_unique_fn (Callable[[str], bool]): The predicate function to validate against.
    Returns:
        str: A unique string based on input function.
    """
    """根据所提供的预示函数找到一个独特的字符串名称。
    字符串附加"_N"，其中N是自然数，直到结果字符串是唯一的。
    参数：
        initial_name (str): 字符串的名称。
        is_unique_fn (Callable[[str], bool]): 根据该函数进行验证。
    返回：
        str: 基于输入函数的独特字符串。
    """
    if is_unique_fn(initial_name):
        return initial_name
    iterator = 1
    result = initial_name + "_" + str(iterator)
    while not is_unique_fn(result):
        result = initial_name + "_" + str(iterator)
        iterator += 1
    return result


def find_root_prim_path_from_regex(prim_path_regex: str) -> tuple[str, int]:
    """Find the first prim above the regex pattern prim and its position.
    Args:
        prim_path_regex (str): full prim path including the regex pattern prim.
    Returns:
        Tuple[str, int]: First position is the prim path to the parent of the regex prim.
                    Second position represents the level of the regex prim in the USD stage tree representation.
    """
    """找出第一 prim 在regex模式 prim及其位置上。
    参数：
        prim_path_regex (str): 包含regex模式 prim的全prim路径。
    返回：
        Tuple[str， int]:第一个位置是prim路径到regex prim的父母。
        第二个位置代表USD阶段树表现中的regex prim水平。
    """
    prim_paths_list = str(prim_path_regex).split("/")
    root_idx = None
    for prim_path_idx in range(len(prim_paths_list)):
        chars = set("[]*|^")
        if any((c in chars) for c in prim_paths_list[prim_path_idx]):
            root_idx = prim_path_idx
            break
    root_prim_path = None
    tree_level = None
    if root_idx is not None:
        root_prim_path = "/".join(prim_paths_list[:root_idx])
        tree_level = root_idx
    return root_prim_path, tree_level
