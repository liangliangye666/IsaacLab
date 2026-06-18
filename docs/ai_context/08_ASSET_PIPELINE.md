# 资产管线

IsaacLab 的资产通常用配置类描述，然后由 scene 或 env 在仿真中生成。

## 资产核心类型

| 类型 | 配置类 | 说明 |
| --- | --- | --- |
| `Articulation` | `ArticulationCfg` | 多关节机器人、机械臂、手、腿式机器人。 |
| `RigidObject` | `RigidObjectCfg` | 单刚体物体。 |
| `RigidObjectCollection` | `RigidObjectCollectionCfg` | 一组刚体对象。 |
| `DeformableObject` | `DeformableObjectCfg` | 可变形物体。 |
| `AssetBase` | `AssetBaseCfg` | light、静态 prim 等通用资产基类。 |
| `SurfaceGripper` | `SurfaceGripperCfg` | 表面吸附 gripper。 |

核心源码：

- `source/isaaclab/isaaclab/assets/`
- `source/isaaclab/isaaclab/sim/spawners/`
- `source/isaaclab/isaaclab/sim/converters/`

## 生成配置

常见 spawn cfg：

- `UsdFileCfg`：加载 USD。
- `UrdfFileCfg`：转换并加载 URDF。
- `MjcfFileCfg`：转换并加载 MJCF。
- `GroundPlaneCfg`、`CuboidCfg`、`SphereCfg` 等 primitive。
- `RigidBodyPropertiesCfg`、`ArticulationRootPropertiesCfg`、`CollisionPropertiesCfg` 等物理 schema。

资产配置通常包含：

```text
spawn -> 如何生成 prim
init_state -> 初始 root pose、joint pos/vel
actuators -> actuator 分组、limit、stiffness、damping
soft_joint_pos_limit_factor -> 软关节限位缩放
```

## `isaaclab_assets`

官方资产配置在 `source/isaaclab_assets/isaaclab_assets/`：

- `robots/`：机器人配置，如 ANYmal、Unitree、Franka、UR、Agibot 等。
- `sensors/`：传感器配置。
- `data/`：本地资产数据目录。目录树索引中默认折叠该目录以避免噪声。

示例：`source/isaaclab_assets/isaaclab_assets/robots/agibot.py` 定义 `AGIBOT_A2D_CFG`，它是 `ArticulationCfg`，使用 Nucleus 路径下的 USD，并配置初始关节、左右臂、头、升降、夹爪等 actuator。

## 资产进入环境的路径

Manager-based 任务中，资产一般写在 `InteractiveSceneCfg` 子类里：

```python
robot = SOME_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

Direct 任务中，资产通常在 `_setup_scene()` 里创建并加入 scene。

## 传感器

传感器配置在 `source/isaaclab/isaaclab/sensors/`：

- camera / tiled camera
- contact sensor
- IMU
- ray caster / ray caster camera
- frame transformer

传感器是否实时更新受 `InteractiveSceneCfg.lazy_sensor_update` 和渲染设置影响。带 RTX sensor 的任务通常需要注意 `render_interval`、`rerender_on_reset` 或 `num_rerenders_on_reset`。

## 资产转换脚本

常用入口在 `scripts/tools/`：

- `convert_urdf.py`
- `convert_mjcf.py`
- `convert_mesh.py`
- `convert_instanceable.py`
- `check_instanceable.py`

这些脚本通常需要 Isaac Sim 环境，通过 `./isaaclab.sh -p ...` 运行。

