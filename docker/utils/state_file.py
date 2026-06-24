# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import configparser
from configparser import ConfigParser
from pathlib import Path
from typing import Any


class StateFile:
    """A class to manage state variables parsed from a configuration file.

    This class provides a simple interface to set, get, and delete variables from a configuration
    object. It also provides the ability to save the configuration object to a file.

    It thinly wraps around the ConfigParser class from the configparser module.
    """
    """从配置文件中解析的状态变量管理类。

    这个类提供了一个简单的界面来设置，从配置对象中获取和删除变量。
    它还可以将配置对象保存到文件中。

    它从配置器模块中 around绕着ConfigParser类。
    """

    def __init__(self, path: Path, namespace: str | None = None):
        """Initialize the class instance and load the configuration file.

        Args:
            path: The path to the configuration file.
            namespace: The default namespace to use when setting and getting variables.
                Namespace corresponds to a section in the configuration file. Defaults to None,
                meaning  all member functions will have to specify the section explicitly,
                or :attr:`StateFile.namespace` must be set manually.
        """
        """启动类实例并加载配置文件。

        参数：
            path: 设置文件的路径。
            namespace: 在设置和获取变量时使用的默认命名空间。
                       名称空间是配置文件中的部分。
                       默认为 None，这意味着所有成员函数都必须明确指定该部分，
                or :attr:`StateFile.namespace`必须手动设置。
        """
        self.path = path
        self.namespace = namespace

        # load the configuration file
        self.load()

    def __del__(self):
        """
        Save the loaded configuration to the initial file path upon deconstruction. This helps
        ensure that the configuration file is always up to date.
        """
        """在解构时保存加载配置到最初的文件路径。
        这有助于确保配置文件始终更新。
        """
        # save the configuration file
        self.save()

    """
    Operations.
    """
    """操作。
    """

    def set_variable(self, key: str, value: Any, section: str | None = None):
        """Set a variable into the configuration object.

        Note:
            Since we use the ConfigParser class, the section names are case-sensitive but the keys are not.

        Args:
            key: The key of the variable to be set.
            value: The value of the variable to be set.
            section: The section of the configuration object to set the variable in.
                Defaults to None, in which case the default section is used.

        Raises:
            configparser.Error: If no section is specified and the default section is None.
        """
        """在配置对象中设置变量。

        说明：
            由于我们使用ConfigParser类， 部分名称对案例敏感，

        参数：
            key: 要设置的变量的关键。
            value: 设置变量的值
            section: 设置变量的配置对象的部分。
                     默认为 None，在这种情况下使用默认部分。

        异常：
            configparser.Error: 如果没有指定部分，默认部分是None。
        """
        # resolve the section
        if section is None:
            if self.namespace is None:
                raise configparser.Error("No section specified. Please specify a section or set StateFile.namespace.")
            section = self.namespace

        # create section if it does not exist
        if section not in self.loaded_cfg.sections():
            self.loaded_cfg.add_section(section)
        # set the variable
        self.loaded_cfg.set(section, key, value)

    def get_variable(self, key: str, section: str | None = None) -> Any:
        """Get a variable from the configuration object.

        Note:
            Since we use the ConfigParser class, the section names are case-sensitive but the keys are not.

        Args:
            key: The key of the variable to be loaded.
            section: The section of the configuration object to read the variable from.
                Defaults to None, in which case the default section is used.

        Returns:
            The value of the variable. It is None if the key does not exist.

        Raises:
            configparser.Error: If no section is specified and the default section is None.
        """
        """从配置对象中得到一个变量。

        说明：
            由于我们使用ConfigParser类， 部分名称对案例敏感，

        参数：
            key: 要加载的变量的关键。
            section: 在配置对象中读取变量的部分。
                     默认为 None，在这种情况下使用默认部分。

        返回：
            变量的值。
            如果没有钥匙，则是None。

        异常：
            configparser.Error: 如果没有指定部分，默认部分是None。
        """
        # resolve the section
        if section is None:
            if self.namespace is None:
                raise configparser.Error("No section specified. Please specify a section or set StateFile.namespace.")
            section = self.namespace

        return self.loaded_cfg.get(section, key, fallback=None)

    def delete_variable(self, key: str, section: str | None = None):
        """Delete a variable from the configuration object.

        Note:
            Since we use the ConfigParser class, the section names are case-sensitive but the keys are not.

        Args:
            key: The key of the variable to be deleted.
            section: The section of the configuration object to remove the variable from.
                Defaults to None, in which case the default section is used.

        Raises:
            configparser.Error: If no section is specified and the default section is None.
            configparser.NoSectionError: If the section does not exist in the configuration object.
            configparser.NoOptionError: If the key does not exist in the section.
        """
        """从配置对象中删除变量。

        说明：
            由于我们使用ConfigParser类， 部分名称对案例敏感，

        参数：
            key: 要删除的变量的关键。
            section: 设置对象的部分将变量从中移除。
                     默认为 None，在这种情况下使用默认部分。

        异常：
            configparser.Error: 如果没有指定部分，默认部分是None。
            configparser.NoSectionError: 如果设置对象中没有部分。
            configparser.NoOptionError: 如果该部分没有钥匙。
        """
        # resolve the section
        if section is None:
            if self.namespace is None:
                raise configparser.Error("No section specified. Please specify a section or set StateFile.namespace.")
            section = self.namespace

        # check if the section exists
        if section not in self.loaded_cfg.sections():
            raise configparser.NoSectionError(f"Section '{section}' does not exist in the file: {self.path}")

        # check if the key exists
        if self.loaded_cfg.has_option(section, key):
            self.loaded_cfg.remove_option(section, key)
        else:
            raise configparser.NoOptionError(option=key, section=section)

    """
    Operations - File I/O.
    """
    """运营 - 文件 I/O
    """

    def load(self):
        """Load the configuration file into memory.

        This function reads the contents of the configuration file into memory.
        If the file does not exist, it creates an empty file.
        """
        """在内存中加载配置文件。

        这个函数将配置文件的内容读入内存中。
        如果文件不存在，它会创建一个空文件。
        """
        self.loaded_cfg = ConfigParser()
        self.loaded_cfg.read(self.path)

    def save(self):
        """Save the configuration file to disk."""
        """保存配置文件到磁盘上。"""
        with open(self.path, "w+") as f:
            self.loaded_cfg.write(f)
