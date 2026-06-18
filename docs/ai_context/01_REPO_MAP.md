# 仓库地图

IsaacLab 是一个多 extension 的 Python 仓库，核心代码集中在 `source/`，脚本集中在 `scripts/`，文档和生成索引在 `docs/`。

## 顶层目录

| 路径 | 作用 |
| --- | --- |
| `source/` | IsaacLab 主要 Python extension，包括框架内核、任务、资产、RL wrapper、mimic 和 contrib。 |
| `scripts/` | 训练、play、demo、环境工具、资产转换、数据录制、benchmark 等命令行入口。 |
| `docs/` | 官方文档源码、license、以及本次新增的 `docs/ai_context/`。 |
| `apps/` | Isaac Sim/Kit app 配置。 |
| `tools/` | 测试工具、依赖安装、项目模板生成器。 |
| `io_descriptors/` | 导出的环境输入输出描述 YAML。 |
| `docker/` | 容器环境配置。 |
| `logs/`、`outputs/` | 运行输出目录，索引脚本默认排除。 |

## `source/` 下的 extension

| 路径 | import 包 | 主要职责 |
| --- | --- | --- |
| `source/isaaclab/isaaclab/` | `isaaclab` | 框架内核：app、sim、scene、envs、managers、assets、sensors、controllers、terrains、utils。 |
| `source/isaaclab_tasks/isaaclab_tasks/` | `isaaclab_tasks` | 官方任务集合：manager-based、direct、任务注册、任务配置和 MDP terms。 |
| `source/isaaclab_assets/isaaclab_assets/` | `isaaclab_assets` | 机器人和传感器资产配置，如 `ANYMAL_*_CFG`、`AGIBOT_A2D_CFG`。 |
| `source/isaaclab_rl/isaaclab_rl/` | `isaaclab_rl` | RSL-RL、RL-Games、SKRL、SB3 的 IsaacLab wrapper 和配置类。 |
| `source/isaaclab_mimic/isaaclab_mimic/` | `isaaclab_mimic` | imitation learning、mimic env、数据生成和 motion planning 相关扩展。 |
| `source/isaaclab_contrib/isaaclab_contrib/` | `isaaclab_contrib` | 贡献扩展：multirotor、thruster action、触觉传感器等非核心模块。 |

## `scripts/` 下主要子目录

| 路径 | 作用 |
| --- | --- |
| `scripts/reinforcement_learning/` | 各 RL 库训练入口：`rsl_rl/train.py`、`rsl_rl/play.py`、`rl_games/train.py`、`skrl/train.py`、`sb3/train.py`。 |
| `scripts/environments/` | 非 RL 环境入口，包含 `state_machine/`（Warp GPU 并行状态机）和 `teleoperation/`（遥操作数据采集）。 |
| `scripts/tutorials/` | 按主题组织的教程脚本：`00_sim/`（仿真基础）、`01_assets/`（资产加载）、`02_scene/`（场景搭建）、`03_envs/`（环境创建）、`04_sensors/`（传感器）、`05_controllers/`（控制器）。 |
| `scripts/tools/` | 工具脚本：`convert_urdf.py`、`convert_mjcf.py`、`convert_mesh.py`、`generate_reference_motion.py`。 |
| `scripts/benchmarks/` | 性能基准测试脚本。 |
| `scripts/demos/` | 演示脚本。 |
| `scripts/sim2sim_transfer/` | sim-to-sim 迁移脚本。 |
| `scripts/offline_rl/` | 离线 RL 数据生成和训练脚本。 |

## 最常用入口

- 环境基类：`source/isaaclab/isaaclab/envs/`
- Manager term 配置：`source/isaaclab/isaaclab/managers/manager_term_cfg.py`
- 通用 MDP terms：`source/isaaclab/isaaclab/envs/mdp/`
- 任务注册：`source/isaaclab_tasks/isaaclab_tasks/**/__init__.py`
- 任务配置：`source/isaaclab_tasks/isaaclab_tasks/manager_based/**` 和 `source/isaaclab_tasks/isaaclab_tasks/direct/**`
- RSL-RL 训练脚本：`scripts/reinforcement_learning/rsl_rl/train.py`
- RSL-RL wrapper：`source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`
- 资产定义：`source/isaaclab_assets/isaaclab_assets/robots/`

完整目录树见 `02_DIRECTORY_TREE.txt`。

