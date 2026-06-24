# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script is a utility to install dependencies mentioned in an extension.toml file of an extension.

The script takes in two arguments:

1. type: The type of dependencies to install. It can be one of the following: ['all', 'apt', 'rosdep'].
2. extensions_dir: The path to the directory beneath which we search for extensions.

The script will search for all extensions in the extensions_dir and then look for an extension.toml file in each
extension's config directory. If the extension.toml file exists, the script will look for the following keys in the
[isaac_lab_settings] section:

* **apt_deps**: A list of apt packages to install.
* **ros_ws**: The path to the ROS workspace in the extension. If the path is not absolute, the script assumes that
  the path is relative to the extension root and resolves it accordingly.

If the type is 'all', the script will install both apt and rosdep packages. If the type is 'apt', the script will only
install apt packages. If the type is 'rosdep', the script will only install rosdep packages.

For more information, please check the `documentation`_.

.. _documentation: https://isaac-sim.github.io/IsaacLab/source/setup/developer.html#extension-dependency-management
"""
"""这个脚本是安装扩展的extension.toml文件中提到的依赖性。

剧本中有两个论点:

1. 类型:要安装的依赖性类型。 它可以是以下一个: ['all'， 'apt'， 'rosdep']。
2. extensions_dir:我们搜索扩展的目录的路径。

脚本将在extensions_dir中搜索所有扩展，然后在每个扩展的配置目录中搜索extension.toml文件。
如果extension.toml文件存在，脚本将在 [isaac_lab_settings] 部分寻找以下键:

* **apt_deps**:适合安装的包表。
* **ros_ws**:扩展中的ROS工作空间的路径.如果路径不是绝对的，脚本假设路径与扩展根相对，并相应地解决。

如果类型是"全部"，脚本将安装 apt 和 rosdep 包。
如果类型是'apt'，脚本只会安装apt包。
如果类型是"rosdep"，脚本只会安装rosdep包。

更多信息请查看`documentation`_。

.. _documentation: https://isaac-sim.github.io/IsaacLab/source/setup/developer.html#extension-dependency-management
"""

import argparse
import os
import shutil
from subprocess import PIPE, STDOUT, Popen

import toml

# add argparse arguments
parser = argparse.ArgumentParser(description="A utility to install dependencies based on extension.toml files.")
parser.add_argument("type", type=str, choices=["all", "apt", "rosdep"], help="The type of packages to install.")
parser.add_argument("extensions_dir", type=str, help="The path to the directory containing extensions.")
parser.add_argument("--ros_distro", type=str, default="humble", help="The ROS distribution to use for rosdep.")


def install_apt_packages(paths: list[str]):
    """Installs apt packages listed in the extension.toml file for Isaac Lab extensions.

    For each path in the input list of paths, the function looks in ``{path}/config/extension.toml`` for
    the ``[isaac_lab_settings][apt_deps]`` key. It then attempts to install the packages listed in the
    value of the key. The function exits on failure to stop the build process from continuing despite missing
    dependencies.

    Args:
        paths: A list of paths to the extension's root.

    Raises:
        SystemError: If 'apt' is not a known command. This is a system error.
    """
    """安装在extension.toml文件中列出的适用包，用于Isaac Lab扩展。

    在输入路径列表中的每个路径，函数在``{path}/config/extension.toml``中查看``[isaac_lab_settings][apt_deps]``键。
    然后它试图安装在关键值中列出的包。
    函数出于未能阻止构建过程继续，尽管缺乏依赖性。

    参数：
        paths: 扩展的根路径列表。

    异常：
        SystemError: 如果"apt"不是一个已知的命令。
                     这是系统错误。
    """
    for path in paths:
        if shutil.which("apt"):
            # Check if the extension.toml file exists
            if not os.path.exists(f"{path}/config/extension.toml"):
                print(
                    "[WARN] During the installation of 'apt' dependencies, unable to find a"
                    f" valid file at: {path}/config/extension.toml."
                )
                continue
            # Load the extension.toml file and check for apt_deps
            with open(f"{path}/config/extension.toml") as fd:
                ext_toml = toml.load(fd)
                if "isaac_lab_settings" in ext_toml and "apt_deps" in ext_toml["isaac_lab_settings"]:
                    deps = ext_toml["isaac_lab_settings"]["apt_deps"]
                    print(f"[INFO] Installing the following apt packages: {deps}")
                    run_and_print(["apt-get", "update"])
                    run_and_print(["apt-get", "install", "-y"] + deps)
                else:
                    print(f"[INFO] No apt packages specified for the extension at: {path}")
        else:
            raise SystemError("Unable to find 'apt' command. Please ensure that 'apt' is installed on your system.")


def install_rosdep_packages(paths: list[str], ros_distro: str = "humble"):
    """Installs ROS dependencies listed in the extension.toml file for Isaac Lab extensions.

    For each path in the input list of paths, the function looks in ``{path}/config/extension.toml`` for
    the ``[isaac_lab_settings][ros_ws]`` key. It then attempts to install the ROS dependencies under the workspace
    listed in the value of the key. The function exits on failure to stop the build process from continuing despite
    missing dependencies.

    If the path to the ROS workspace is not absolute, the function assumes that the path is relative to the extension
    root and resolves it accordingly. The function also checks if the ROS workspace exists before proceeding with
    the installation of ROS dependencies. If the ROS workspace does not exist, the function raises an error.

    Args:
        path: A list of paths to the extension roots.
        ros_distro: The ROS distribution to use for rosdep. Default is 'humble'.

    Raises:
        FileNotFoundError: If a valid ROS workspace is not found while installing ROS dependencies.
        SystemError: If 'rosdep' is not a known command. This is raised if 'rosdep' is not installed on the system.
    """
    """在extension.toml文件中列出的ROS依赖性安装为Isaac Lab扩展。

    在输入路径列表中的每个路径，函数在``{path}/config/extension.toml``中查看``[isaac_lab_settings][ros_ws]``键。
    然后它试图在关键值中列出的工作空间下安装ROS依赖性。
    函数出于未能阻止构建过程继续，尽管缺乏依赖性。

    如果 ROS 工作空间的路径不是绝对的，则函数假设路径与扩展根相对，并相应解决。
    在继续安装ROS依赖之前，该函数还检查ROS工作空间是否存在。
    如果ROS工作空间不存在，函数会产生错误。

    参数：
        path: 扩展根的路径列表。
        ros_distro: 用于rosdep的ROS分布。
                    默认是"谦虚"。

    异常：
        FileNotFoundError: 如果在安装ROS依赖时没有找到有效的ROS工作空间。
        SystemError: 如果"rosdep"不是已知的命令。
                     如果在系统上没有安装"rosdep"，则会出现这种情况。
    """
    for path in paths:
        if shutil.which("rosdep"):
            # Check if the extension.toml file exists
            if not os.path.exists(f"{path}/config/extension.toml"):
                print(
                    "[WARN] During the installation of 'rosdep' dependencies, unable to find a"
                    f" valid file at: {path}/config/extension.toml."
                )
                continue
            # Load the extension.toml file and check for ros_ws
            with open(f"{path}/config/extension.toml") as fd:
                ext_toml = toml.load(fd)
                if "isaac_lab_settings" in ext_toml and "ros_ws" in ext_toml["isaac_lab_settings"]:
                    # resolve the path to the ROS workspace
                    ws_path = ext_toml["isaac_lab_settings"]["ros_ws"]
                    if not os.path.isabs(ws_path):
                        ws_path = os.path.join(path, ws_path)
                    # check if the workspace exists
                    if not os.path.exists(f"{ws_path}/src"):
                        raise FileNotFoundError(
                            "During the installation of 'rosdep' dependencies, unable to find a"
                            f" valid ROS workspace at: {ws_path}."
                        )
                    # install rosdep if not already installed
                    if not os.path.exists("/etc/ros/rosdep/sources.list.d/20-default.list"):
                        run_and_print(["rosdep", "init"])
                        run_and_print(["rosdep", "update", f"--rosdistro={ros_distro}"])
                    # install rosdep packages
                    run_and_print(
                        [
                            "rosdep",
                            "install",
                            "--from-paths",
                            f"{ws_path}/src",
                            "--ignore-src",
                            "-y",
                            f"--rosdistro={ros_distro}",
                        ]
                    )
                else:
                    print(f"[INFO] No rosdep packages specified for the extension at: {path}")
        else:
            raise SystemError(
                "Unable to find 'rosdep' command. Please ensure that 'rosdep' is installed on your system."
                "You can install it by running:\n\t sudo apt-get install python3-rosdep"
            )


def run_and_print(args: list[str]):
    """Runs a subprocess and prints the output to stdout.

    This function wraps Popen and prints the output to stdout in real-time.

    Args:
        args: A list of arguments to pass to Popen.
    """
    """运行一个子进程，然后将输出打印到stdout。

    这个函数将Popen包裹起来，并将输出打印到实时。

    参数：
        args: 一份要向Popen传递的论点列表。
    """
    print(f'Running "{args}"')
    with Popen(args, stdout=PIPE, stderr=STDOUT, env=os.environ) as p:
        while p.poll() is None:
            text = p.stdout.read1().decode("utf-8")
            print(text, end="", flush=True)
        return_code = p.poll()
        if return_code != 0:
            raise RuntimeError(f'Subprocess with args: "{args}" failed. The returned error code was: {return_code}')


def main():
    # Parse the command line arguments
    args = parser.parse_args()
    # Get immediate children of args.extensions_dir
    extension_paths = [os.path.join(args.extensions_dir, x) for x in next(os.walk(args.extensions_dir))[1]]

    # Install dependencies based on the type
    if args.type == "all":
        install_apt_packages(extension_paths)
        install_rosdep_packages(extension_paths, args.ros_distro)
    elif args.type == "apt":
        install_apt_packages(extension_paths)
    elif args.type == "rosdep":
        install_rosdep_packages(extension_paths, args.ros_distro)
    else:
        raise ValueError(f"'Invalid dependency type: '{args.type}'. Available options: ['all', 'apt', 'rosdep'].")


if __name__ == "__main__":
    main()
