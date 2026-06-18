# Direct 环境

Direct workflow 把任务逻辑直接写在 env class 中，核心基类是 `DirectRLEnv` 和 `DirectMARLEnv`。

## 核心源码

| 主题 | 文件 |
| --- | --- |
| 单智能体 Direct RL | `source/isaaclab/isaaclab/envs/direct_rl_env.py` |
| 多智能体 Direct RL | `source/isaaclab/isaaclab/envs/direct_marl_env.py` |
| Direct 配置 | `source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py` |
| Direct MARL 配置 | `source/isaaclab/isaaclab/envs/direct_marl_env_cfg.py` |
| Direct 任务 | `source/isaaclab_tasks/isaaclab_tasks/direct/` |

## 必看的 hook

Direct env 子类通常实现：

- `_setup_scene()`：创建 robot、object、terrain、sensor，并注册到 `InteractiveScene`。
- `_pre_physics_step(actions)`：接收 policy action，做 scale、clip、缓存或控制目标计算。
- `_apply_action()`：把处理后的动作写到 asset 或 actuator。
- `_get_observations()`：返回观测字典，至少包含 `policy`。
- `_get_rewards()`：返回 shape 为 `(num_envs,)` 的 reward tensor。
- `_get_dones()`：返回 `(terminated, time_outs)`。
- `_reset_idx(env_ids)`：重置指定环境。

这些 hook 已在 `11_SYMBOL_INDEX_RAW.txt` 中索引，可以直接搜 `_get_rewards` 或具体 env class。

## step 顺序

`DirectRLEnv.step(action)` 的关键顺序：

1. action 移到 env device。
2. 可选 action noise。
3. `_pre_physics_step(action)`
4. decimation 循环内：
   - `_apply_action()`
   - `scene.write_data_to_sim()`
   - `sim.step(render=False)`
   - 必要时 render
   - `scene.update(dt=physics_dt)`
5. 更新 episode counters。
6. `_get_dones()`
7. `_get_rewards()`
8. reset done env。
9. interval events。
10. `_get_observations()`
11. 可选 observation noise。
12. 返回 obs、reward、terminated、truncated、extras。

## Direct 与 manager-based 的区别

| 点 | Manager-based | Direct |
| --- | --- | --- |
| 动作处理 | `ActionManager` 和 action terms | env class 自己处理 |
| 观测 | `ObservationManager` | `_get_observations()` |
| 奖励 | `RewardManager` + reward terms | `_get_rewards()` |
| 终止 | `TerminationManager` + termination terms | `_get_dones()` |
| 适合场景 | 复用 term、配置化、多任务变体 | 快速写专用逻辑、复杂自定义 step |

## 常见 Direct 任务入口

- Cartpole：`source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/`
- Ant/Humanoid locomotion：`source/isaaclab_tasks/isaaclab_tasks/direct/ant/`、`direct/humanoid/`
- Franka cabinet：`source/isaaclab_tasks/isaaclab_tasks/direct/franka_cabinet/`
- Quadcopter：`source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/`
- Factory/Forge：`source/isaaclab_tasks/isaaclab_tasks/direct/factory/`、`direct/forge/`
- Shadow hand / Allegro hand：`source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/`、`direct/allegro_hand/`

