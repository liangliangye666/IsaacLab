# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Sensors.
##

RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)
"""Configuration for the ray-caster marker."""
"""射线标记器的配置。"""


CONTACT_SENSOR_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "contact": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "no_contact": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            visible=False,
        ),
    },
)
"""Configuration for the contact sensor marker."""
"""接触传感器标记的配置。"""

DEFORMABLE_TARGET_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.8)),
        ),
    },
)
"""Configuration for the deformable object's kinematic target marker."""
"""对可变物体的动态目标标记配置。"""

VISUO_TACTILE_SENSOR_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "tacsl_pts": sim_utils.SphereCfg(
            radius=0.0002,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    },
)
"""Configuration for the visuo-tactile sensor marker."""
"""视觉触觉传感器标记的配置。"""

##
# Frames.
##

FRAME_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.5, 0.5, 0.5),
        ),
        "connecting_line": sim_utils.CylinderCfg(
            radius=0.002,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), roughness=1.0),
        ),
    }
)
"""Configuration for the frame marker."""
"""框架标记的配置。"""


RED_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    }
)
"""Configuration for the red arrow marker (along x-direction)."""
"""红色箭头标记的配置 (沿 x 方向)。"""


BLUE_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        )
    }
)
"""Configuration for the blue arrow marker (along x-direction)."""
"""蓝色箭头标记的配置 (沿 x方向)。"""

GREEN_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        )
    }
)
"""Configuration for the green arrow marker (along x-direction)."""
"""绿色箭头标记的配置 (沿 x 方向)。"""


##
# Goals.
##

CUBOID_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cuboid": sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)
"""Configuration for the cuboid marker."""
"""立方体标记的配置。"""

SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)
"""Configuration for the sphere marker."""
"""球标记的配置。"""

POSITION_GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "target_far": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "target_near": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "target_invisible": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            visible=False,
        ),
    }
)
"""Configuration for the end-effector tracking marker."""
"""终端效应标记的配置"""
