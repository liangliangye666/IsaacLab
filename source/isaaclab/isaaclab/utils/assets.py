# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module that defines the host-server where assets and resources are stored.

By default, we use the Isaac Sim Nucleus Server for hosting assets and resources. This makes
distribution of the assets easier and makes the repository smaller in size code-wise.

For more information, please check information on `Omniverse Nucleus`_.

.. _Omniverse Nucleus: https://docs.omniverse.nvidia.com/nucleus/latest/overview/overview.html
"""
"""定义存储资产和资源的主机服务器的子模块。

默认情况下，我们使用了Isaac Sim核服务器来托管资产和资源。
这使得资产的分配变得更容易，并且使存储库的规模更小。

更多信息请查看`Omniverse Nucleus`_的信息。

.. _Omniverse Nucleus: https://docs.omniverse.nvidia.com/nucleus/latest/overview/overview.html
"""

import asyncio
import io
import logging
import os
import tempfile
import time
from typing import Literal

import carb
import omni.client

# import logger
logger = logging.getLogger(__name__)

NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
"""Path to the root directory on the Nucleus Server."""
"""核服务器的根目录。"""

NVIDIA_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
"""Path to the root directory on the NVIDIA Nucleus Server."""
"""在NVIDIA核服务器上的根目录。"""

ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
"""Path to the ``Isaac`` directory on the NVIDIA Nucleus Server."""
"""在NVIDIA核服务器上的``Isaac``目录。"""

ISAACLAB_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
"""Path to the ``Isaac/IsaacLab`` directory on the NVIDIA Nucleus Server."""
"""在NVIDIA核服务器上的``Isaac/IsaacLab``目录。"""


def check_file_path(path: str) -> Literal[0, 1, 2]:
    """Checks if a file exists on the Nucleus Server or locally.

    Args:
        path: The path to the file.

    Returns:
        The status of the file. Possible values are listed below.

        * :obj:`0` if the file does not exist
        * :obj:`1` if the file exists locally
        * :obj:`2` if the file exists on the Nucleus Server
    """
    """检查一个文件是否存在于核服务器或本地。

    参数：
        path: 文件的路径。

    返回：
        文件的状态。
        下面列出了可能的值。

        * :obj:`0`如果文件不存在
        * :obj:`1` 如果文件存在本地
        * :obj:`2` 如果文件存在于Nucleus服务器
    """
    if os.path.isfile(path):
        return 1
    # we need to convert backslash to forward slash on Windows for omni.client API
    elif omni.client.stat(path.replace(os.sep, "/"))[0] == omni.client.Result.OK:
        return 2
    else:
        return 0


def retrieve_file_path(path: str, download_dir: str | None = None, force_download: bool = True) -> str:
    """Retrieves the path to a file on the Nucleus Server or locally.

    If the file exists locally, then the absolute path to the file is returned.
    If the file exists on the Nucleus Server, then the file is downloaded to the local machine
    and the absolute path to the file is returned.

    Args:
        path: The path to the file.
        download_dir: The directory where the file should be downloaded. Defaults to None, in which
            case the file is downloaded to the system's temporary directory.
        force_download: Whether to force download the file from the Nucleus Server. This will overwrite
            the local file if it exists. Defaults to True.

    Returns:
        The path to the file on the local machine.

    Raises:
        FileNotFoundError: When the file not found locally or on Nucleus Server.
        RuntimeError: When the file cannot be copied from the Nucleus Server to the local machine. This
            can happen when the file already exists locally and :attr:`force_download` is set to False.
    """
    """检索核服务器或本地文件的路径。

    如果文件存在本地，则返回文件的绝对路径。
    如果文件存在于Nucleus服务器上，则该文件将下载到本地机器，返回文件的绝对路径。

    参数：
        path: 文件的路径。
        download_dir: 文件应该下载的目录。
                      在 None 上默认，在这种情况下，文件被下载到系统的临时目录中。
        force_download: 是否强迫从核服务器下载文件。
                        如果存在，则将重写本地文件。
                        默认为 True。

    返回：
        在本地机器上的文件的路径。

    异常：
        FileNotFoundError: 当文件在本地或在Nucleus服务器上没有找到时。
        RuntimeError: 当文件不能从核服务器复制到本机时。
                      当文件已经存在本地，并且设置:attr:`force_download`为False时，
    """
    # check file status
    file_status = check_file_path(path)
    if file_status == 1:
        return os.path.abspath(path)
    elif file_status == 2:
        # resolve download directory
        if download_dir is None:
            download_dir = tempfile.gettempdir()
        else:
            download_dir = os.path.abspath(download_dir)
        # create download directory if it does not exist
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        # download file in temp directory using os
        file_name = os.path.basename(omni.client.break_url(path.replace(os.sep, "/")).path)
        target_path = os.path.join(download_dir, file_name)
        # check if file already exists locally
        if not os.path.isfile(target_path) or force_download:
            # copy file to local machine
            result = omni.client.copy(path.replace(os.sep, "/"), target_path, omni.client.CopyBehavior.OVERWRITE)
            if result != omni.client.Result.OK and force_download:
                raise RuntimeError(f"Unable to copy file: '{path}'. Is the Nucleus Server running?")
        return os.path.abspath(target_path)
    else:
        raise FileNotFoundError(f"Unable to find the file: {path}")


def read_file(path: str) -> io.BytesIO:
    """Reads a file from the Nucleus Server or locally.

    Args:
        path: The path to the file.

    Raises:
        FileNotFoundError: When the file not found locally or on Nucleus Server.

    Returns:
        The content of the file.
    """
    """读取来自核服务器或本地文件。

    参数：
        path: 文件的路径。

    异常：
        FileNotFoundError: 当文件在本地或在Nucleus服务器上没有找到时。

    返回：
        文件的内容。
    """
    # check file status
    file_status = check_file_path(path)
    if file_status == 1:
        with open(path, "rb") as f:
            return io.BytesIO(f.read())
    elif file_status == 2:
        file_content = omni.client.read_file(path.replace(os.sep, "/"))[2]
        return io.BytesIO(memoryview(file_content).tobytes())
    else:
        raise FileNotFoundError(f"Unable to find the file: {path}")


"""
Nucleus Connection.
"""
"""核连接。
"""


def check_usd_path_with_timeout(usd_path: str, timeout: float = 300, log_interval: float = 30) -> bool:
    """Checks whether the given USD file path is available on the NVIDIA Nucleus server.

    This function synchronously runs an asynchronous USD path availability check,
    logging progress periodically until it completes. The file is available on the server
    if the HTTP status code is 200. Otherwise, the file is not available on the server.

    This is useful for checking server responsiveness before attempting to load a remote
    asset. It will block execution until the check completes or times out.

    Args:
        usd_path: The remote USD file path to check.
        timeout: Maximum time (in seconds) to wait for the server check.
        log_interval: Interval (in seconds) at which progress is logged.

    Returns:
        Whether the given USD path is available on the server.
    """
    """检查该USD文件路径是否可用在NVIDIA核服务器上。

    这种函数同步运行异步USD路径可用性检查，定期记录进步直到完成。
    文件在服务器上可用
    if the HTTP status code is 200. Otherwise, the file is not available on the server.

    在试图加载远程资产之前，这有助于检查服务器响应能力。
    它将阻止执行，直到检查完成或结束。

    参数：
        usd_path: 需要检查远程USD文件路径。
        timeout: 最长时间 (几秒钟) 等待服务器检查。
        log_interval: 记录进步的间隔 (秒钟)。

    返回：
        服务器是否可用给定的USD路径。
    """
    start_time = time.time()
    loop = asyncio.get_event_loop()

    coroutine = _is_usd_path_available(usd_path, timeout)
    task = asyncio.ensure_future(coroutine)

    next_log_time = start_time + log_interval

    first_log = True
    while not task.done():
        now = time.time()
        if now >= next_log_time:
            elapsed = int(now - start_time)
            if first_log:
                logger.warning(f"Checking server availability for USD path: {usd_path} (timeout: {timeout}s)")
                first_log = False
            logger.warning(f"Waiting for server response... ({elapsed}s elapsed)")
            next_log_time += log_interval
        loop.run_until_complete(asyncio.sleep(0.1))  # Yield to allow async work

    return task.result()


"""
Helper functions.
"""
"""辅助函数。
"""


async def _is_usd_path_available(usd_path: str, timeout: float) -> bool:
    """Checks whether the given USD path is available on the Omniverse Nucleus server.

    This function is a asynchronous routine to check the availability of the given USD path on
    the Omniverse Nucleus server. It will return True if the USD path is available on the server,
    False otherwise.

    Args:
        usd_path: The remote or local USD file path to check.
        timeout: Timeout in seconds for the async stat call.

    Returns:
        Whether the given USD path is available on the server.
    """
    """检查Omniverse Nucleus服务器上是否可用给定的USD路径。

    这个函数是一个异步的程序来检查Omniverse Nucleus服务器上给定的USD路径的可用性。
    如果服务器上有USD路径，则将返回True，否则False。

    参数：
        usd_path: 要检查远程或本地USD文件路径。
        timeout: 在几秒钟内取消异步数据通话。

    返回：
        服务器是否可用给定的USD路径。
    """
    try:
        result, _ = await asyncio.wait_for(omni.client.stat_async(usd_path), timeout=timeout)
        return result == omni.client.Result.OK
    except asyncio.TimeoutError:
        logger.warning(f"Timed out after {timeout}s while checking for USD: {usd_path}")
        return False
    except Exception as ex:
        logger.warning(f"Exception during USD file check: {type(ex).__name__}: {ex}")
        return False
