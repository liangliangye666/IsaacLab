# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""
"""首先发射艾萨克仿真器。"""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""
"""休息，一切都跟着。"""

import os

import pytest

from isaacsim.core.utils.extensions import enable_extension, get_extension_path_from_name

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Setup and teardown for each test."""
    """每次测试的设置和拆除。"""
    # Setup: Create a new stage
    sim_utils.create_new_stage()

    # Setup: Create simulation context
    dt = 0.01
    sim = SimulationContext(SimulationCfg(dt=dt))

    # Setup: Create MJCF config
    enable_extension("isaacsim.asset.importer.mjcf")
    extension_path = get_extension_path_from_name("isaacsim.asset.importer.mjcf")
    config = MjcfConverterCfg(
        asset_path=f"{extension_path}/data/mjcf/nv_ant.xml",
        import_sites=True,
        fix_base=False,
        make_instanceable=True,
    )

    # Yield the resources for the test
    yield sim, config

    # Teardown: Cleanup simulation
    sim.stop()
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


@pytest.mark.isaacsim_ci
def test_no_change(test_setup_teardown):
    """Call conversion twice. This should not generate a new USD file."""
    """两次打电话转换。
    这不应该产生新的USD文件。
    """
    sim, mjcf_config = test_setup_teardown

    mjcf_converter = MjcfConverter(mjcf_config)
    time_usd_file_created = os.stat(mjcf_converter.usd_path).st_mtime_ns

    # no change to config only define the usd directory
    new_config = mjcf_config
    new_config.usd_dir = mjcf_converter.usd_dir
    # convert to usd but this time in the same directory as previous step
    new_mjcf_converter = MjcfConverter(new_config)
    new_time_usd_file_created = os.stat(new_mjcf_converter.usd_path).st_mtime_ns

    assert time_usd_file_created == new_time_usd_file_created


@pytest.mark.isaacsim_ci
def test_config_change(test_setup_teardown):
    """Call conversion twice but change the config in the second call. This should generate a new USD file."""
    """在第二次调用时，调用转换，但在第二次调用时，调用配置。
    这应该生成一个新的USD文件。
    """
    sim, mjcf_config = test_setup_teardown

    mjcf_converter = MjcfConverter(mjcf_config)
    time_usd_file_created = os.stat(mjcf_converter.usd_path).st_mtime_ns

    # change the config
    new_config = mjcf_config
    new_config.fix_base = not mjcf_config.fix_base
    # define the usd directory
    new_config.usd_dir = mjcf_converter.usd_dir
    # convert to usd but this time in the same directory as previous step
    new_mjcf_converter = MjcfConverter(new_config)
    new_time_usd_file_created = os.stat(new_mjcf_converter.usd_path).st_mtime_ns

    assert time_usd_file_created != new_time_usd_file_created


@pytest.mark.isaacsim_ci
def test_create_prim_from_usd(test_setup_teardown):
    """Call conversion and create a prim from it."""
    """打电话转换并从中创建prim。"""
    sim, mjcf_config = test_setup_teardown

    urdf_converter = MjcfConverter(mjcf_config)

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)

    assert sim.stage.GetPrimAtPath(prim_path).IsValid()
