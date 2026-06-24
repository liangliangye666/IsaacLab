# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing various sensor classes implementations.

This subpackage contains the sensor classes that are compatible with Isaac Sim. We include both
USD-based and custom sensors:

* **USD-prim sensors**: Available in Omniverse and require creating a USD prim for them.
  For instance, RTX ray tracing camera and lidar sensors.
* **USD-schema sensors**: Available in Omniverse and require creating a USD schema on an existing prim.
  For instance, contact sensors and frame transformers.
* **Custom sensors**: Implemented in Python and do not require creating any USD prim or schema.
  For instance, warp-based ray-casters.

Due to the above categorization, the prim paths passed to the sensor's configuration class
are interpreted differently based on the sensor type. The following table summarizes the
interpretation of the prim paths for different sensor types:

+---------------------+---------------------------+---------------------------------------------------------------+
| Sensor Type         | Example Prim Path         | Pre-check                                                     |
+=====================+===========================+===============================================================+
| Camera              | /World/robot/base/camera  | Leaf is available, and it will spawn a USD camera             |
+---------------------+---------------------------+---------------------------------------------------------------+
| Contact Sensor      | /World/robot/feet_*       | Leaf is available and checks if the schema exists             |
+---------------------+---------------------------+---------------------------------------------------------------+
| Ray Caster          | /World/robot/base         | Leaf exists and is a physics body (Articulation / Rigid Body) |
+---------------------+---------------------------+---------------------------------------------------------------+
| Frame Transformer   | /World/robot/base         | Leaf exists and is a physics body (Articulation / Rigid Body) |
+---------------------+---------------------------+---------------------------------------------------------------+
| Imu                 | /World/robot/base         | Leaf exists and is a physics body (Rigid Body)                |
+---------------------+---------------------------+---------------------------------------------------------------+

"""
"""含有各种传感器类型的实现子包。

本子包包含与Isaac Sim兼容的传感器类。
我们包括基于USD和定制传感器:

* **USD-prim传感器**:可在Omniverse中使用，需要为它们创建USD prim.例如，RTX射线追踪摄像头和Lidar传感器。
* **USD图案传感器**:可在Omniverse中使用，需要在现有的prim上创建USD图案.例如，接触传感器和框架转换器。
* **定制传感器**:在Python中实现，不需要创建任何USD prim或方案.例如，基于变形的射线播放器。

由于上述分类，通过传感器配置类的prim路径根据传感器类型被不同的解释。
下表概述了对不同传感器类型的prim路径的解释:

+---------------------+-----------------------------------+-----------------------------------------
-----------------------------------------+ ♬ 传感器类型 ♬ 举例 始路径 ♬ 预测 ♬ +================================
====================================================================================================
====================================================================================================
====================================================================+ ♬
摄像机器人USD卡马+----------------------+----------------------------------------------------------------+
网页是可用的，并检查该方案是否存在的。 网页是存在的。 网页是存在的。
"""

from .camera import *  # noqa: F401, F403
from .contact_sensor import *  # noqa: F401, F403
from .frame_transformer import *  # noqa: F401
from .imu import *  # noqa: F401, F403
from .ray_caster import *  # noqa: F401, F403
from .sensor_base import SensorBase  # noqa: F401
from .sensor_base_cfg import SensorBaseCfg  # noqa: F401
