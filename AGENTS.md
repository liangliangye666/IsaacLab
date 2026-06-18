# IsaacLab AI Context

本仓库包含一套给 AI 助手快速定位代码用的索引文档，入口在 `docs/ai_context/00_START_HERE.md`。

## 使用约定

- 回答 IsaacLab 相关问题前，优先查看 `docs/ai_context/00_START_HERE.md`。
- 查找任务名、Gym 注册、`env_cfg_entry_point`、`rsl_rl_cfg_entry_point` 时，优先查看 `docs/ai_context/04_TASK_REGISTRY_INDEX.md`。
- 查找关键类、配置类、MDP term、Direct env hook 时，优先查看 `docs/ai_context/11_SYMBOL_INDEX_RAW.txt`。
- 查找目录和包边界时，优先查看 `docs/ai_context/02_DIRECTORY_TREE.txt` 和 `docs/ai_context/01_REPO_MAP.md`。
- 这些文档是索引和导航材料，不替代源码；涉及具体行为时仍以 `source/` 下源码为准。
- 回答问题前优先阅读 docs/ai_context/00_START_HERE.md；
- 不要随意修改 source 下的原始代码；
- 分析任务时先找 gym.register；
- 分析机器人资产时先找 ArticulationCfg、UsdFileCfg、ActuatorCfg；
- 默认用中文解释。

## 重新生成索引

运行：

```bash
python3 scripts/devtools/generate_ai_context_indices.py
```

该脚本只读取仓库内容，并写入 `docs/ai_context/02_DIRECTORY_TREE.txt`、`docs/ai_context/04_TASK_REGISTRY_INDEX.md`、`docs/ai_context/11_SYMBOL_INDEX_RAW.txt`。
