# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Predefined configurations for GelSight tactile sensors."""
"""为GelSight触觉传感器的预定义配置。"""

from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_cfg import GelSightRenderCfg

##
# Predefined Configurations
##

GELSIGHT_R15_CFG = GelSightRenderCfg(
    sensor_data_dir_name="gelsight_r15_data",
    background_path="bg.jpg",
    calib_path="polycalib.npz",
    real_background="real_bg.npy",
    image_height=320,
    image_width=240,
    num_bins=120,
    mm_per_pixel=0.0877,
)
"""Configuration for GelSight R1.5 sensor rendering parameters.

The GelSight R1.5 is a high-resolution tactile sensor with a 320x240 pixel tactile image.
It uses a pixel-to-millimeter ratio of 0.0877 mm/pixel.

Reference: https://www.gelsight.com/gelsightinc-products/
"""
"""对GelSight R1.5传感器渲染参数的配置。

GelSight R1.5是一个高分辨率触觉传感器，具有320x240像素触觉图像。
它使用0.0877mm/pixel的像素比分。

Reference: https://www.gelsight.com/gelsightinc-products/
"""

GELSIGHT_MINI_CFG = GelSightRenderCfg(
    sensor_data_dir_name="gs_mini_data",
    background_path="bg.jpg",
    calib_path="polycalib.npz",
    real_background="real_bg.npy",
    image_height=240,
    image_width=320,
    num_bins=120,
    mm_per_pixel=0.065,
)
"""Configuration for GelSight Mini sensor rendering parameters.

The GelSight Mini is a compact tactile sensor with a 240x320 pixel tactile image.
It uses a pixel-to-millimeter ratio of 0.065 mm/pixel, providing higher spatial resolution
than the R1.5 model.

Reference: https://www.gelsight.com/gelsightinc-products/
"""
"""为GelSight Mini传感器渲染参数的配置。

GelSight Mini是一款紧的触觉传感器，具有240x320像素触觉图像。
它使用0.065mm/pixel的像素/毫米比率，提供比R1.5模型更高的空间分辨率。

Reference: https://www.gelsight.com/gelsightinc-products/
"""
