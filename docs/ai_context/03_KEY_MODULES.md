# 关键模块

## 框架内核 `isaaclab`

| 模块 | 重点文件 | 说明 |
| --- | --- | --- |
| app | `source/isaaclab/isaaclab/app/` | 启动 Isaac Sim / Kit app，训练脚本通常先创建 `AppLauncher`。 |
| sim | `source/isaaclab/isaaclab/sim/` | `SimulationContext`、物理配置、spawner、USD/URDF/MJCF converter。 |
| scene | `source/isaaclab/isaaclab/scene/` | `InteractiveScene` 和 `InteractiveSceneCfg`，负责多环境克隆、资产和传感器实例化。 |
| envs | `source/isaaclab/isaaclab/envs/` | `ManagerBasedEnv`、`ManagerBasedRLEnv`、`DirectRLEnv`、`DirectMARLEnv`。 |
| managers | `source/isaaclab/isaaclab/managers/` | Action、Observation、Reward、Termination、Command、Event、Curriculum、Recorder manager。 |
| assets | `source/isaaclab/isaaclab/assets/` | `Articulation`、`RigidObject`、`DeformableObject`、`AssetBase` 及其配置。 |
| actuators | `source/isaaclab/isaaclab/actuators/` | 隐式/显式执行器模型和 actuator cfg。 |
| sensors | `source/isaaclab/isaaclab/sensors/` | Camera、ContactSensor、IMU、RayCaster、FrameTransformer。 |
| controllers | `source/isaaclab/isaaclab/controllers/` | Differential IK、OSC、RMPFlow、Pink IK、joint impedance。 |
| terrains | `source/isaaclab/isaaclab/terrains/` | terrain importer、generator、height field、trimesh terrain。 |
| utils | `source/isaaclab/isaaclab/utils/` | `configclass`、IO、dict、noise、modifier、dataset、assets 路径等工具。 |

## 任务层 `isaaclab_tasks`

`isaaclab_tasks.__init__` 会调用 `import_packages(__name__, blacklist)` 导入任务子包，从而触发各级 `gym.register(...)`。任务分两类：

- `manager_based/`：用配置类组合 scene、actions、observations、commands、rewards、terminations、events。
- `direct/`：直接继承 `DirectRLEnv` 或 `DirectMARLEnv`，在类方法里实现 step hooks、reward、done、observation。

任务索引见 `04_TASK_REGISTRY_INDEX.md`。

## RL 层 `isaaclab_rl`

`isaaclab_rl` 不定义任务本身，而是把 IsaacLab env 适配到训练库：

- `rsl_rl/vecenv_wrapper.py`：`RslRlVecEnvWrapper`，负责 reset、step、done 合并、action clip、TensorDict 输出。
- `rsl_rl/rl_cfg.py`：RSL-RL runner、policy、algorithm 配置类。
- `rl_games/`、`skrl.py`、`sb3.py`：其他训练库 wrapper。

## 资产层 `isaaclab_assets`

资产包暴露机器人和传感器配置。典型机器人配置是一个 `ArticulationCfg`，包含：

- `spawn`：USD/URDF/MJCF 或 primitive 的生成配置。
- `init_state`：初始关节、root pose、速度。
- `actuators`：关节组、effort/velocity limit、stiffness、damping。

例如 Agibot 在 `source/isaaclab_assets/isaaclab_assets/robots/agibot.py` 中定义 `AGIBOT_A2D_CFG`。

## 扩展层

- `isaaclab_mimic`：复用 manager-based env，加入 imitation/mimic 所需的 subtask、pose/action 转换、数据集生成。
- `isaaclab_contrib`：实验性或贡献模块，例如 multirotor asset、`ThrustAction`、触觉传感器。

## `isaaclab_contrib` 详解

这是一个独立的社区贡献包，目前主要包含两个场景的扩展：

### 多旋翼无人机控制

```
isaaclab_contrib/
├── actuators/thruster.py       → Thruster (低层电机动力学，含非对称时间常数)
├── assets/multirotor/          → Multirotor (多旋翼资产，基于推力而非关节)
├── mdp/actions/thrust_actions.py → ThrustAction (RL 动作项，仿射变换)
└── utils/types.py              → MultiRotorActions (推力数据容器)
```

数据流：`策略输出[-1,1] → ThrustAction 缩放/偏移 → Multirotor 分配矩阵 → Thruster 电机延迟 → PhysX`

### 触觉传感器仿真 (TacSL)

```
isaaclab_contrib/sensors/tacsl_sensor/
├── visuotactile_sensor.py      → VisuoTactileSensor (相机触觉 + 力场触觉)
├── visuotactile_sensor_cfg.py  → GelSightRenderCfg, VisuoTactileSensorCfg
└── visuotactile_render.py      → GelsightRender (基于 Taxim 方法的触觉图像渲染)
```

### 与核心包的区别

不同于 `isaaclab`、`isaaclab_assets`、`isaaclab_tasks` 这些核心包，`isaaclab_contrib` 的 API 可能不稳定，主要用于实验和社区贡献孵化。

## `scripts/` 关键目录

### 教程脚本 (`scripts/tutorials/`)

按主题组织，从基础到高级：

| 子目录 | 主题 |
| --- | --- |
| `00_sim/` | 仿真基础：`create_empty.py`（空场景）、`spawn_prims.py`（生成物体）、`launch_app.py`（启动流程） |
| `01_assets/` | 资产加载：`run_rigid_object.py`（刚体）、`run_articulation.py`（关节体）、`add_new_robot.py`（添加新机器人） |
| `02_scene/` | 场景搭建：`create_scene.py`（多实体场景） |
| `03_envs/` | 环境创建：RL 环境从配置到运行的完整流程 |
| `04_sensors/` | 传感器：相机、接触传感器、IMU、RTX 传感器 |
| `05_controllers/` | 控制器：IK、OSC、阻抗控制 |
| `06_terrains/` | 地形：平坦/崎岖地形生成 |

### 状态机环境 (`scripts/environments/state_machine/`)

非 RL 的手写决策逻辑，使用 Warp GPU 并行库：

- `lift_cube_sm.py`：五状态抓取（REST → APPROACH_ABOVE → APPROACH → GRASP → LIFT）
- 核心模式：`wp.kernel` GPU 并行状态机 + `PickAndLiftSm` Python 类封装
- 状态机输出末端目标位姿 [N,7] + 夹爪状态，由环境内部的 IK + PD 驱动机械臂运动

