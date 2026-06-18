# RSL-RL 工作流

RSL-RL 是 IsaacLab 里最常用的训练入口之一，相关源码分两部分：训练脚本在 `scripts/`，wrapper 和配置类在 `source/isaaclab_rl/`。

## 核心文件

| 主题 | 文件 |
| --- | --- |
| 训练入口 | `scripts/reinforcement_learning/rsl_rl/train.py` |
| play 入口 | `scripts/reinforcement_learning/rsl_rl/play.py` |
| CLI 参数 | `scripts/reinforcement_learning/rsl_rl/cli_args.py` |
| vec env wrapper | `source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py` |
| runner/policy/algorithm cfg | `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py` |
| task agent cfg | `source/isaaclab_tasks/isaaclab_tasks/**/agents/rsl_rl_ppo_cfg.py` |

## 训练脚本顺序

`train.py` 的主流程：

1. argparse 读取 `--task`、`--num_envs`、`--device`、`--max_iterations`、`--agent` 等参数。
2. `AppLauncher(args_cli)` 启动 Isaac Sim app。
3. 检查 `rsl-rl-lib` 版本。
4. `@hydra_task_config(args_cli.task, args_cli.agent)` 加载 env cfg 和 agent cfg。
5. CLI 参数覆盖配置。
6. 设置 log 目录：`logs/rsl_rl/<experiment_name>/<timestamp>[_run_name]`。
7. `gym.make(args_cli.task, cfg=env_cfg, render_mode=...)` 创建环境。
8. 如果是 `DirectMARLEnv`，通过 `multi_agent_to_single_agent(env)` 转成单智能体。
9. 可选 `RecordVideo`。
10. `RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)`。
11. 根据 `agent_cfg.class_name` 创建 `OnPolicyRunner` 或 `DistillationRunner`。
12. 可选 resume checkpoint。
13. dump `env.yaml` 和 `agent.yaml`。
14. `runner.learn(...)`。

## `RslRlVecEnvWrapper` 做什么

位置：`source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`

关键职责：

- 要求底层 env 是 `ManagerBasedRLEnv` 或 `DirectRLEnv`。
- 记录 `num_envs`、`device`、`max_episode_length`、`num_actions`。
- 初始化时调用 `env.reset()`，因为 RSL-RL runner 不一定先 reset。
- `step(actions)` 中可选 `torch.clamp(actions, -clip_actions, clip_actions)`。
- 把 IsaacLab 返回的 `terminated | truncated` 合并成 RSL-RL 的 `dones`。
- 对 infinite horizon 任务，把 `truncated` 放到 `extras["time_outs"]`。
- 返回 `TensorDict(obs_dict, batch_size=[num_envs])`。

## agent 配置类

基础配置在 `rl_cfg.py`：

- `RslRlPpoActorCriticCfg`
- `RslRlPpoActorCriticRecurrentCfg`
- `RslRlPpoAlgorithmCfg`
- `RslRlBaseRunnerCfg`
- `RslRlOnPolicyRunnerCfg`

任务自己的配置一般在 `agents/rsl_rl_ppo_cfg.py`，由 `rsl_rl_cfg_entry_point` 指向。先在 `04_TASK_REGISTRY_INDEX.md` 找任务，再打开对应 agent cfg。

## 常用命令形态

训练：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --headless
```

播放：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Cartpole-v0
```

常见覆盖项：

- `--num_envs`
- `--device`
- `--max_iterations`
- `--seed`
- `--video`
- `--resume`
- `--load_run`
- `--load_checkpoint`

## 日志目录结构

RSL-RL 训练的完整输出结构：

```
logs/rsl_rl/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>[_<run_name>]/
├── params/
│   ├── env.yaml           # 环境配置 dump
│   └── agent.yaml         # Agent/runner 配置 dump
├── model_<iteration>.pt   # PyTorch checkpoint
├── events.out.tfevents.*  # TensorBoard 事件 (若使用)
├── wandb/                 # Weights & Biases 日志 (若使用)
├── neptune/               # Neptune.ai 日志 (若使用)
└── videos/                # 可选录制的训练视频
```

## 训练恢复

```bash
# 从 checkpoint 恢复训练
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Cartpole-v0 \
    --resume \
    --load_run <run_dir_name>

# 指定最大迭代数（在已有 checkpoint 基础上继续）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Cartpole-v0 \
    --resume \
    --load_run <run_dir_name> \
    --load_checkpoint model_1000.pt \
    --max_iterations 2000
```

## 与状态机工作流的对比

| 维度 | RSL-RL 训练 | 状态机 |
| --- | --- | --- |
| 需要训练 | 是（PPO 等算法） | 否（手写规则） |
| 策略类型 | 可学习 | 固定规则 |
| 推理速度 | 取决于网络大小 | 极快（GPU 并行 if-else） |
| 泛化能力 | 可泛化到新场景 | 仅覆盖预定状态 |
| 适用场景 | 运动控制、操作等复杂任务 | 分步明确的任务（如抓取-举起） |
| 文件位置 | `scripts/reinforcement_learning/rsl_rl/` | `scripts/environments/state_machine/` |
