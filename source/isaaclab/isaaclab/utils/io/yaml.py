# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for file I/O with yaml."""
"""用于 Yaml 的文件 I/O。"""

import os

import yaml

from isaaclab.utils import class_to_dict


def load_yaml(filename: str) -> dict:
    """Loads an input PKL file safely.

    Args:
        filename: The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        The data read from the input file.
    """
    """安全地加载输入PKL文件。

    参数：
        filename: 走向皮文件的路径。

    异常：
        FileNotFoundError: 当指定文件不存在时。

    返回：
        在输入文件中读取的数据。
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename) as f:
        data = yaml.full_load(f)
    return data


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Saves data into a YAML file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save either a dictionary or class object.
        sort_keys: Whether to sort the keys in the output file. Defaults to False.
    """
    """保存数据到一个安全的YAML文件。

    说明：
        该函数在文件的路径上创建任何缺失的目录。

    参数：
        filename: 保存文件的路径。
        data: 保存字典或类对象的数据。
        sort_keys: 输出文件中的键是否进行排序。
                   默认为 False。
    """
    # check ending
    if not filename.endswith("yaml"):
        filename += ".yaml"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # convert data into dictionary
    if not isinstance(data, dict):
        data = class_to_dict(data)
    # save data
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)
