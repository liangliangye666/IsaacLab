# AI 阅读入口

这组文档用于让 AI 助手快速理解 IsaacLab 仓库结构、任务注册方式、RL 训练链路和常见源码入口。它们是导航索引，不替代源码；涉及精确行为时仍以 `source/` 下实现为准。

## 推荐阅读顺序

1. `01_REPO_MAP.md`：先确认仓库顶层目录和各 extension 包职责。
2. `04_TASK_REGISTRY_INDEX.md`：查任务名、Gym 注册、环境配置入口、RL agent 配置入口。
3. `11_SYMBOL_INDEX_RAW.txt`：查类、配置类、MDP term、Direct env hook 的源码位置。
4. 按问题类型进入专题文档：
   - `05_RL_WORKFLOW.md`：任务如何从注册到训练脚本运行。
   - `06_MANAGER_BASED_ENV.md`：ManagerBasedRLEnv 的配置和 step 流程。
   - `07_DIRECT_ENV.md`：DirectRLEnv / DirectMARLEnv 的 hook 流程。
   - `08_ASSET_PIPELINE.md`：机器人、USD/URDF/MJCF、资产配置、传感器。
   - `09_RSL_RL_WORKFLOW.md`：RSL-RL 训练、play、wrapper、日志和 checkpoint。
   - `10_COMMON_SEARCH_COMMANDS.md`：常用 `rg` 搜索命令。

## 快速定位规则

- 已知任务名，例如 `Isaac-Velocity-Flat-Anymal-D-v0`：先在 `04_TASK_REGISTRY_INDEX.md` 搜任务名，再打开表中 `env_cfg_entry_point` 和 agent 配置文件。
- 已知类名，例如 `ManagerBasedRLEnv`、`DirectRLEnv`、`RslRlVecEnvWrapper`：先在 `11_SYMBOL_INDEX_RAW.txt` 搜类名。
- 已知功能词，例如 reward、command、action、termination：先在 `11_SYMBOL_INDEX_RAW.txt` 搜函数名或 term 名，再看对应 `mdp/` 文件。
- 已知包或目录：先看 `02_DIRECTORY_TREE.txt` 或 `01_REPO_MAP.md`，再用 `rg --files` 精确过滤。

## 生成物说明

- `02_DIRECTORY_TREE.txt`：目录树。当前环境没有 `tree` 命令，因此由 `scripts/devtools/generate_ai_context_indices.py` 生成等价文本。
- `04_TASK_REGISTRY_INDEX.md`：静态解析 `gym.register(...)`，不 import IsaacLab，也不启动 Isaac Sim。
- `11_SYMBOL_INDEX_RAW.txt`：静态解析关键类、配置类、MDP 函数、Direct 环境 hook。

## 相关顶层入口文件

- `AGENTS.md`：AI 助手使用约定和快速入口。位于仓库根目录，可以被 IDE 的 agent 工具自动识别。
- `CODEBUDDY.md`：CodeBuddy 专用 AI 助手指南，包含英文快速参考和搜索模式。位于仓库根目录，可被 CodeBuddy 自动加载。

## 非 RL 的环境入口

除 RL 训练外，`scripts/environments/` 下还有状态机和遥操作等非 RL 环境：

- `scripts/environments/state_machine/`：GPU 并行状态机（使用 Warp 库），如 `lift_cube_sm.py` 实现了五状态抓取流程。
- `scripts/environments/teleoperation/`：遥操作入口，用于数据采集和 demonstration 记录。

重新生成索引：

```bash
python3 scripts/devtools/generate_ai_context_indices.py
```

