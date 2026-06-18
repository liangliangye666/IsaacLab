# CODEBUDDY.md - Isaac Lab AI Assistant Guide

This file provides guidance for AI coding assistants working in the Isaac Lab repository.

## Repository Overview

Isaac Lab is NVIDIA's open-source framework for robot reinforcement learning, built on top of Isaac Sim. It provides a unified interface for training robot policies across different embodiments (arms, hands, quadrupeds, humanoids, drones) using various RL libraries (RSL-RL, RL-Games, SKRL, SB3).

## Critical Rules

1. **Never modify `source/` code unless explicitly asked.** The `source/` directory contains the framework kernel, tasks, assets, and RL wrappers. Only edit when the user explicitly requests it.
2. **Always read AI context docs first.** Before answering any Isaac Lab question, read `docs/ai_context/00_START_HERE.md` to understand the navigation structure.
3. **Use the task registry index for task lookups.** When a user asks about a specific task (e.g., "Isaac-Velocity-Flat-Anymal-D-v0"), search `docs/ai_context/04_TASK_REGISTRY_INDEX.md` to find its config entry, agent configs, and registration location.
4. **Use the symbol index for class/config lookups.** When looking for key classes, search `docs/ai_context/11_SYMBOL_INDEX_RAW.txt`.
5. **Default to Chinese explanations.** The user prefers Chinese for code explanations (see user rules). Use Chinese for all explanatory content.

## Quick Reference

### AI Context Docs (Chinese)

| File | Purpose |
|------|---------|
| `docs/ai_context/00_START_HERE.md` | Entry point and navigation guide |
| `docs/ai_context/01_REPO_MAP.md` | Top-level directory and extension map |
| `docs/ai_context/02_DIRECTORY_TREE.txt` | Full directory tree (auto-generated) |
| `docs/ai_context/03_KEY_MODULES.md` | Key module descriptions |
| `docs/ai_context/04_TASK_REGISTRY_INDEX.md` | All 196 `gym.register()` entries (auto-generated) |
| `docs/ai_context/05_RL_WORKFLOW.md` | From task registration to training |
| `docs/ai_context/06_MANAGER_BASED_ENV.md` | ManagerBasedRLEnv architecture and step flow |
| `docs/ai_context/07_DIRECT_ENV.md` | DirectRLEnv/DirectMARLEnv hook architecture |
| `docs/ai_context/08_ASSET_PIPELINE.md` | Robot assets, sensors, URDF/MJCF converters |
| `docs/ai_context/09_RSL_RL_WORKFLOW.md` | RSL-RL training, play, wrapper, logging |
| `docs/ai_context/10_COMMON_SEARCH_COMMANDS.md` | Common `rg`/`grep` search commands |
| `docs/ai_context/11_SYMBOL_INDEX_RAW.txt` | Key classes, configs, MDP terms (auto-generated) |

### Regenerate Indices

```bash
python3 scripts/devtools/generate_ai_context_indices.py
```

### Source Extensions

| Package | Path | Purpose |
|---------|------|---------|
| `isaaclab` | `source/isaaclab/isaaclab/` | Core framework |
| `isaaclab_tasks` | `source/isaaclab_tasks/isaaclab_tasks/` | Official tasks |
| `isaaclab_assets` | `source/isaaclab_assets/isaaclab_assets/` | Robot/sensor configs |
| `isaaclab_rl` | `source/isaaclab_rl/isaaclab_rl/` | RL library wrappers |
| `isaaclab_mimic` | `source/isaaclab_mimic/isaaclab_mimic/` | Imitation learning |
| `isaaclab_contrib` | `source/isaaclab_contrib/isaaclab_contrib/` | Community contributions (multirotor, tactile) |

### Two Environment Architectures

1. **Manager-Based** (`ManagerBasedRLEnv`): Config-driven. Define actions/observations/rewards/terminations/commands/events in config classes. The framework composes them automatically. Good for reuse and multi-task variants.

2. **Direct** (`DirectRLEnv` / `DirectMARLEnv`): Code-driven. Override `_pre_physics_step()`, `_get_observations()`, `_get_rewards()`, `_get_dones()`, `_reset_idx()` directly. Good for custom logic.

### Common Workflow

```
gym.register() → parse_env_cfg() → gym.make() → env wrapper → runner.learn()
```

### Action Pipeline (for task-space control like IK)

```
State machine / Policy outputs: desired end-effector pose [N, 7]
    → DifferentialInverseKinematicsAction.process_actions()
    → DifferentialIKController.set_command()
    → DifferentialInverseKinematicsAction.apply_actions()
        → IK solver: Δq = Jᵀ(JJᵀ + λ²I)⁻¹ Δx
        → set_joint_position_target()
    → write_data_to_sim() → PhysX PD control
    → sim.step()
```

### Typical Search Patterns

```bash
# Find a task's registration and configs
grep -n "Isaac-Velocity-Flat-Anymal-D-v0" docs/ai_context/04_TASK_REGISTRY_INDEX.md

# Find a class definition
grep -rn "class ManagerBasedRLEnv" source/

# Find all action configs for a task
grep -rn "JointPositionActionCfg\|DifferentialInverseKinematicsActionCfg" source/isaaclab_tasks/

# Find reward functions
grep -rn "def.*reward\|RewardTermCfg" source/isaaclab_tasks/ source/isaaclab/isaaclab/envs/mdp/
```
