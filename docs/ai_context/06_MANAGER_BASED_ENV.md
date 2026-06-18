# Manager-Based 环境

Manager-based workflow 的核心类是 `ManagerBasedRLEnv`，配置基类是 `ManagerBasedRLEnvCfg`。

## 核心源码

| 主题 | 文件 |
| --- | --- |
| 基础环境 | `source/isaaclab/isaaclab/envs/manager_based_env.py` |
| RL 环境 step | `source/isaaclab/isaaclab/envs/manager_based_rl_env.py` |
| 基础配置 | `source/isaaclab/isaaclab/envs/manager_based_env_cfg.py` |
| RL 配置 | `source/isaaclab/isaaclab/envs/manager_based_rl_env_cfg.py` |
| manager term 配置类 | `source/isaaclab/isaaclab/managers/manager_term_cfg.py` |
| 通用 MDP terms | `source/isaaclab/isaaclab/envs/mdp/` |

## 配置结构

一个典型 `ManagerBasedRLEnvCfg` 会组合这些字段：

- `scene`：`InteractiveSceneCfg`，定义 terrain、robot、objects、sensors、lights。
- `actions`：动作 term，例如 joint position、joint velocity、IK、OSC、thrust。
- `observations`：观测 group 和 observation term。
- `commands`：命令生成器，例如速度命令、pose 命令。
- `rewards`：奖励 term，每项通常是 `RewardTermCfg(func=..., weight=..., params=...)`。
- `terminations`：终止条件，例如 timeout、姿态过大、非法接触。
- `events`：reset/startup/interval 事件和随机化。
- `curriculum`：课程学习 term。
- `sim`、`decimation`、`episode_length_s`、`viewer`：仿真和环境步长设置。

Manager term 配置类在 `manager_term_cfg.py` 中定义：

- `ActionTermCfg`
- `ObservationTermCfg`
- `ObservationGroupCfg`
- `RewardTermCfg`
- `TerminationTermCfg`
- `CommandTermCfg`
- `EventTermCfg`
- `CurriculumTermCfg`

## step 顺序

`ManagerBasedRLEnv.step(action)` 的关键顺序：

1. `action_manager.process_action(action)`
2. decimation 循环内：
   - `action_manager.apply_action()`
   - `scene.write_data_to_sim()`
   - `sim.step(render=False)`
   - 必要时 render
   - `scene.update(dt=physics_dt)`
3. 更新 episode counters。
4. `termination_manager.compute()`
5. `reward_manager.compute(dt=step_dt)`
6. reset 已终止或 timeout 的 env。
7. `command_manager.compute(dt=step_dt)`
8. interval events。
9. `observation_manager.compute(update_history=True)`
10. 返回 obs、reward、terminated、truncated、extras。

## 查 manager-based 任务的方法

1. 在 `04_TASK_REGISTRY_INDEX.md` 搜 task id。
2. 打开 `env_cfg_entry_point` 指向的 env cfg。
3. 从 env cfg 的嵌套类开始看：
   - `SceneCfg`
   - `ActionsCfg`
   - `ObservationsCfg`
   - `CommandsCfg`
   - `RewardsCfg`
   - `TerminationsCfg`
   - `EventCfg`
4. 如果某个 term 是 `mdp.xxx`，去同目录或通用 `isaaclab/envs/mdp/` 搜函数定义。
5. 如果涉及机器人/传感器，追到 `scene` 中对应实体配置。

## 常见定位入口

- locomotion velocity：`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/`
- manipulation reach/lift/cabinet/stack：`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/`
- classic cartpole/ant/humanoid：`source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/`
- navigation：`source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/`

## 常见 Action 类型详解

Manager-based 环境通过 `ActionsCfg` 中的 `ActionTermCfg` 子类配置动作类型：

### 关节层动作

| 配置类 | 动作维度 | 说明 |
| --- | --- | --- |
| `JointPositionActionCfg` | `num_joints` | 直接控制关节目标位置（最常用） |
| `RelativeJointPositionActionCfg` | `num_joints` | 相对于当前关节位置的增量控制 |
| `JointVelocityActionCfg` | `num_joints` | 直接控制关节目标速度 |
| `JointEffortActionCfg` | `num_joints` | 直接控制关节力矩 |
| `JointPositionToLimitsActionCfg` | `num_joints` | 策略输出 [-1,1] 映射到关节限位范围 |
| `EMAJointPositionToLimitsActionCfg` | `num_joints` | 带指数移动平均平滑的关节位置映射 |

### 任务空间动作

| 配置类 | 动作维度 | 说明 |
| --- | --- | --- |
| `DifferentialInverseKinematicsActionCfg` | 7 或 8 | IK 求解：策略输出末端位姿 → 自动转为关节目标 |
| `OperationalSpaceControllerActionCfg` | 6 | OSC：策略输出末端力 → 自动转为关节力矩 |
| `NonHolonomicActionCfg` | 2 | 非全向运动：线性速度 + 角速度 |

### 夹爪/特殊动作

| 配置类 | 动作维度 | 说明 |
| --- | --- | --- |
| `BinaryJointPositionActionCfg` | 1 | 二值夹爪控制（开/关） |
| `SurfaceGripperBinaryActionCfg` | 1 | 吸盘式抓手控制 |

### 动作从配置到执行的流程

```
ActionsCfg.arm_action = DifferentialInverseKinematicsActionCfg(...)
ActionsCfg.gripper_action = BinaryJointPositionActionCfg(...)
    │
    ▼  创建时:
ActionManager 根据配置创建对应的 ActionTerm 实例
    ├── arm_action: DifferentialInverseKinematicsAction
    └── gripper_action: BinaryJointPositionAction
    │
    ▼  env.step(action) 时:
1. ActionManager.process_action(action)
   → 拆分 [N,8] 为 [N,7] arm + [N,1] gripper
   → 各 ActionTerm 处理自己的子动作
   
2. ActionManager.apply_action()
   → arm_action.apply_actions() → IK 求解 → set_joint_position_target()
   → gripper_action.apply_actions() → 设置夹爪目标

