# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for managing X11 forwarding in the docker container."""

from __future__ import annotations
"""运输器容器中的X11传输管理的实用功能。"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .state_file import StateFile


# This method of x11 enabling forwarding was inspired by osrf/rocker
# https://github.com/osrf/rocker
def configure_x11(statefile: StateFile) -> dict[str, str]:
    """Configure X11 forwarding by creating and managing a temporary .xauth file.

    If xauth is not installed, the function prints an error message and exits. The message
    instructs the user to install xauth with 'apt install xauth'.

    If the .xauth file does not exist, the function creates it and configures it with the necessary
    xauth cookie.

    Args:
        statefile: An instance of the configuration file class.

    Returns:
        A dictionary with two key-value pairs:

        - "__ISAACLAB_TMP_XAUTH": The path to the temporary .xauth file.
        - "__ISAACLAB_TMP_DIR": The path to the directory where the temporary .xauth file is stored.

    """
    """通过创建和管理临时.xauth文件来配置X11转发。

    如果 xauth 没有安装，函数会打印一个错误信息，然后退出。
    该消息指示用户安装xauth使用"apt install xauth"。

    如果 .xauth文件不存在，函数会创建它并配置它与必要的xauthcookie。

    参数：
        statefile: 配置类的一个实例。

    返回：
        一个有两个关键值对的字典:

        - "__ISAACLAB_TMP_XAUTH":暂时的 .xauth文件的路径。
        - "__ISAACLAB_TMP_DIR":临时.xauth文件存储的目录的路径。
    """
    # check if xauth is installed
    if not shutil.which("xauth"):
        print("[INFO] xauth is not installed.")
        print("[INFO] Please install it with 'apt install xauth'")
        exit(1)

    # set the namespace to X11 for the statefile
    statefile.namespace = "X11"
    # load the value of the temporary xauth file
    tmp_xauth_value = statefile.get_variable("__ISAACLAB_TMP_XAUTH")

    if tmp_xauth_value is None or not Path(tmp_xauth_value).exists():
        # create a temporary directory to store the .xauth file
        tmp_dir = subprocess.run(["mktemp", "-d"], capture_output=True, text=True, check=True).stdout.strip()
        # create the .xauth file
        tmp_xauth_value = create_x11_tmpfile(tmpdir=Path(tmp_dir))
        # set the statefile variable
        statefile.set_variable("__ISAACLAB_TMP_XAUTH", str(tmp_xauth_value))
    else:
        tmp_dir = Path(tmp_xauth_value).parent

    return {"__ISAACLAB_TMP_XAUTH": str(tmp_xauth_value), "__ISAACLAB_TMP_DIR": str(tmp_dir)}


def x11_check(statefile: StateFile) -> tuple[list[str], dict[str, str]] | None:
    """Check and configure X11 forwarding based on user input and existing state.

    This function checks if X11 forwarding is enabled in the configuration file. If it is not configured,
    the function prompts the user to enable or disable X11 forwarding. If X11 forwarding is enabled, the function
    configures X11 forwarding by creating a temporary .xauth file.

    Args:
        statefile: An instance of the configuration file class.

    Returns:
        If X11 forwarding is enabled, the function returns a tuple containing the following:

        - A list containing the x11.yaml file configuration option for docker-compose.
        - A dictionary containing the environment variables for the container.

        If X11 forwarding is disabled, the function returns None.
    """
    """根据用户输入和现有的状态检查和配置X11转发。

    该函数检查配置文件中是否启用X11转发。
    如果没有配置，该函数会要求用户启用或禁用X11转发。
    如果启用X11转发，该函数将通过创建临时.xauth文件来配置X11转发。

    参数：
        statefile: 配置类的一个实例。

    返回：
        如果已启用X11转发，函数将返回包含以下内容的tuple:

        - 包含docker-compose的x11.yaml文件配置选项的列表。
        - 包含容器环境变量的字典。

        如果禁用X11转发，函数将返回None。
    """
    # set the namespace to X11 for the statefile
    statefile.namespace = "X11"
    # check if X11 forwarding is enabled
    is_x11_forwarding_enabled = statefile.get_variable("X11_FORWARDING_ENABLED")

    if is_x11_forwarding_enabled is None:
        print("[INFO] X11 forwarding from the Isaac Lab container is disabled by default.")
        print(
            "[INFO] It will fail if there is no display, or this script is being run via ssh without proper"
            " configuration."
        )
        x11_answer = input("Would you like to enable it? (y/N) ")

        # parse the user's input
        if x11_answer.lower() == "y":
            is_x11_forwarding_enabled = "1"
            print("[INFO] X11 forwarding is enabled from the container.")
        else:
            is_x11_forwarding_enabled = "0"
            print("[INFO] X11 forwarding is disabled from the container.")

        # remember the user's choice and set the statefile variable
        statefile.set_variable("X11_FORWARDING_ENABLED", is_x11_forwarding_enabled)
    else:
        # print the current configuration
        print(f"[INFO] X11 Forwarding is configured as '{is_x11_forwarding_enabled}' in '.container.cfg'.")

        # print help message to enable/disable X11 forwarding
        if is_x11_forwarding_enabled == "1":
            print("\tTo disable X11 forwarding, set 'X11_FORWARDING_ENABLED=0' in '.container.cfg'.")
        else:
            print("\tTo enable X11 forwarding, set 'X11_FORWARDING_ENABLED=1' in '.container.cfg'.")

    if is_x11_forwarding_enabled == "1":
        x11_envars = configure_x11(statefile)
        # If X11 forwarding is enabled, return the proper args to
        # compose the x11.yaml file. Else, return an empty string.
        return ["--file", "x11.yaml"], x11_envars

    return None


def x11_cleanup(statefile: StateFile):
    """Clean up the temporary .xauth file used for X11 forwarding.

    If the .xauth file exists, this function deletes it and remove the corresponding state variable.

    Args:
        statefile: An instance of the configuration file class.
    """
    """清理用于X11转发的临时.xauth文件。

    如果 .xauth 文件存在，该函数会删除它，并删除相应的状态变量。

    参数：
        statefile: 配置类的一个实例。
    """
    # set the namespace to X11 for the statefile
    statefile.namespace = "X11"

    # load the value of the temporary xauth file
    tmp_xauth_value = statefile.get_variable("__ISAACLAB_TMP_XAUTH")

    # if the file exists, delete it and remove the state variable
    if tmp_xauth_value is not None and Path(tmp_xauth_value).exists():
        print(f"[INFO] Removing temporary Isaac Lab '.xauth' file: {tmp_xauth_value}.")
        Path(tmp_xauth_value).unlink()
        statefile.delete_variable("__ISAACLAB_TMP_XAUTH")


def create_x11_tmpfile(tmpfile: Path | None = None, tmpdir: Path | None = None) -> Path:
    """Creates an .xauth file with an MIT-MAGIC-COOKIE derived from the current ``DISPLAY`` environment variable.

    Args:
        tmpfile: A Path to a file which will be filled with the correct .xauth info.
        tmpdir: A Path to the directory where a random tmp file will be made.
            This is used as an ``--tmpdir arg`` to ``mktemp`` bash command.

    Returns:
        The Path to the .xauth file.
    """
    """创建一个.xauth文件，以MIT-MAGIC-COOKIE从当前``DISPLAY``环境变量中衍生出来。

    参数：
        tmpfile: 一个向文件的路径，将填写正确的 .xauth信息。
        tmpdir: 一个向目录的路径，将创建一个随机的tmp文件。
                这是一个``--tmpdir arg``到``mktemp`` bash命令。

    返回：
        进入 .xauth文件的路径。
    """
    if tmpfile is None:
        if tmpdir is None:
            add_tmpdir = ""
        else:
            add_tmpdir = f"--tmpdir={tmpdir}"
        # Create .tmp file with .xauth suffix
        tmp_xauth = Path(
            subprocess.run(
                ["mktemp", "--suffix=.xauth", f"{add_tmpdir}"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    else:
        tmpfile.touch()
        tmp_xauth = tmpfile

    # Derive current MIT-MAGIC-COOKIE and make it universally addressable
    xauth_cookie = subprocess.run(
        ["xauth", "nlist", os.environ["DISPLAY"]], capture_output=True, text=True, check=True
    ).stdout.replace("ffff", "")

    # Merge the new cookie into the create .tmp file
    subprocess.run(["xauth", "-f", tmp_xauth, "nmerge", "-"], input=xauth_cookie, text=True, check=True)

    return tmp_xauth


def x11_refresh(statefile: StateFile):
    """Refresh the temporary .xauth file used for X11 forwarding.

    If x11 is enabled, this function generates a new .xauth file with the current MIT-MAGIC-COOKIE-1.
    The new file uses the same filename so that the bind-mount and ``XAUTHORITY`` var from build-time
    still work.

    As the envar ``DISPLAY` informs the contents of the MIT-MAGIC-COOKIE-1, that value within the container
    will also need to be updated to the current value on the host. Currently, this done automatically in
    :meth:`ContainerInterface.enter` method.

    The function exits if X11 forwarding is enabled but the temporary .xauth file does not exist. In this case,
    the user must rebuild the container.

    Args:
        statefile: An instance of the configuration file class.
    """
    """更新用于X11转发的临时.xauth文件。

    如果启用x11，该函数将生成一个新的 .xauth文件，包含当前的MIT-MAGIC-COOKIE-1。
    新的文件使用相同的文件名，使bind-mount和``XAUTHORITY`` var从构建时间仍然工作。

    由于 envar ``DISPLAY` 通知MIT-MAGIC-COOKIE-1的内容，因此该容器内的值也需要更新到主机上的当前值。
    目前，这是在:meth:`ContainerInterface.enter`方法中自动完成的。

    如果启用X11转发，但暂时的 .xauth文件不存在，该函数会退出。
    在这种情况下，用户必须重建容器。

    参数：
        statefile: 配置类的一个实例。
    """
    # set the namespace to X11 for the statefile
    statefile.namespace = "X11"

    # check if X11 forwarding is enabled
    is_x11_forwarding_enabled = statefile.get_variable("X11_FORWARDING_ENABLED")
    # load the value of the temporary xauth file
    tmp_xauth_value = statefile.get_variable("__ISAACLAB_TMP_XAUTH")

    # print the current configuration
    if is_x11_forwarding_enabled is not None:
        status = "enabled" if is_x11_forwarding_enabled == "1" else "disabled"
        print(f"[INFO] X11 Forwarding is {status} from the settings in '.container.cfg'")

    # if the file exists, delete it and create a new one
    if tmp_xauth_value is not None and Path(tmp_xauth_value).exists():
        # remove the file and create a new one
        Path(tmp_xauth_value).unlink()
        create_x11_tmpfile(tmpfile=Path(tmp_xauth_value))
        # update the statefile with the new path
        statefile.set_variable("__ISAACLAB_TMP_XAUTH", str(tmp_xauth_value))
    elif tmp_xauth_value is None:
        if is_x11_forwarding_enabled is not None and is_x11_forwarding_enabled == "1":
            print(
                "[ERROR] X11 forwarding is enabled but the temporary .xauth file does not exist."
                " Please rebuild the container by running: './docker/container.py start'"
            )
            sys.exit(1)
        else:
            print("[INFO] X11 forwarding is disabled. No action taken.")
