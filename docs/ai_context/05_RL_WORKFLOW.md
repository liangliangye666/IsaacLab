# RL 工作流

IsaacLab 的训练工作流可以理解为：

`任务注册 -> 解析配置 -> gym.make 创建环境 -> wrapper 适配训练库 -> runner 训练或 play`

## 1. 任务注册

任务注册发生在各任务目录的 `__init__.py`：

```python
gym.register(
    id="Isaac-...",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "...:SomeEnvCfg",
        "rsl_rl_cfg_entry_point": "...:SomePPORunnerCfg",
    },
)
```

静态索引见 `04_TASK_REGISTRY_INDEX.md`。该文件能回答：

- 某个 task id 对应哪个 env cfg。
- 是否有 RSL-RL / RL-Games / SKRL / SB3 配置。
- 注册代码位于哪个 `__init__.py`。

## 2. 配置解析

核心函数在 `source/isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py`：

- `load_cfg_from_registry(task_name, entry_point_key)`：从 Gym registry 读取 entry point，支持 YAML 和 Python class。
- `parse_env_cfg(task_name, device, num_envs, use_fabric)`：加载 `env_cfg_entry_point`，并覆盖设备、环境数量、Fabric 选项。

Hydra 训练入口还会通过 `source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py` 合并命令行和配置。

## 3. 创建环境

训练脚本通常做：

```python
env_cfg = parse_env_cfg(...)
env = gym.make(task_name, cfg=env_cfg, render_mode=...)
```

`entry_point` 决定实际环境基类：

- manager-based task 通常指向 `isaaclab.envs:ManagerBasedRLEnv`。
- direct task 通常指向具体 env class，例如 `isaaclab_tasks.direct.cartpole.cartpole_env:CartpoleEnv`。
- multi-agent direct task 使用 `DirectMARLEnv`，训练前可能转成 single-agent。

## 4. 环境 step

Manager-based 环境把动作、观测、奖励、终止、事件拆成 manager terms，由 manager 执行。Direct 环境把逻辑写进 env class 的 hook 方法。

详细见：

- `06_MANAGER_BASED_ENV.md`
- `07_DIRECT_ENV.md`

## 5. 训练库适配

不同训练库需要不同 vec env wrapper。RSL-RL 链路是：

1. `scripts/reinforcement_learning/rsl_rl/train.py`
2. `gym.make(...)`
3. 可选 `multi_agent_to_single_agent(env)`
4. 可选 `gym.wrappers.RecordVideo`
5. `RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)`
6. `OnPolicyRunner(...).learn(...)`

RSL-RL 细节见 `09_RSL_RL_WORKFLOW.md`。

## 6. 日志与输出

RSL-RL 默认写入：

```text
logs/rsl_rl/<experiment_name>/<timestamp>[_run_name]/
```

常见文件：

- `params/env.yaml`
- `params/agent.yaml`
- model checkpoint
- TensorBoard / wandb / neptune 日志
- 可选训练视频

## 7. 非 RL 状态机工作流

除了 RL 训练，IsaacLab 还支持手写状态机作为决策策略。这是一条独立的控制流程：

```
状态机定义 (Warp kernel) → PickAndLiftSm 类封装 → env.step(actions) → IK + PD 执行
```

关键文件：`scripts/environments/state_machine/lift_cube_sm.py`

与 RL 流程的对比：

| 点 | RL 工作流 | 状态机工作流 |
| --- | --- | --- |
| 决策方式 | 神经网络策略推理 | 手写 if-else 状态转移 |
| 输出 | 策略网络输出 (通常 [-1,1]) | 状态机直接输出目标位姿 |
| 训练 | 需要 PPO/SAC 等训练 | 不需要训练，规则驱动 |
| 适用场景 | 复杂、难以手写的策略 | 流程明确的序列任务 |

## 8. 从策略输出到机器人运动的完整链路

对于 IK 模式的任务空间控制，动作如何转化为关节运动：

```
策略/状态机输出: 末端目标位姿 [N,7]
    │
    ▼  env.step(actions) 内部:
ActionManager.process_action()
    ├── arm_action → DifferentialInverseKinematicsAction.process_actions()
    │       └── DifferentialIKController.set_command()  设置目标位姿
    └── gripper_action → BinaryJointPositionAction.process_actions()
    │
    ▼  decimation 循环内:
ActionManager.apply_action()
    └── DifferentialInverseKinematicsAction.apply_actions()
            ├── 计算当前末端位姿 + 雅可比矩阵
            ├── IK 求解: Δq = Jᵀ(JJᵀ + λ²I)⁻¹ Δx  (DLS 方法)
            └── articulation.set_joint_position_target(joint_pos_des)
    │
    ▼
scene.write_data_to_sim()
    └── physx_view.set_dof_position_targets()  → PhysX 内部 PD 控制
    │
    ▼
sim.step()  → PhysX 物理仿真一步
```

这个链路意味着：状态机或神经网络只需要输出末端去哪里，底层 IK+PD 自动完成关节控制。这是一个分层控制的设计模式。
