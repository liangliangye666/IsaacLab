# 常用搜索命令

优先使用 `rg`。当前环境中 `rg` 可用，`tree` 不可用；目录树由 `scripts/devtools/generate_ai_context_indices.py` 生成。

## 索引再生成

```bash
python3 scripts/devtools/generate_ai_context_indices.py
```

生成：

- `docs/ai_context/02_DIRECTORY_TREE.txt`
- `docs/ai_context/04_TASK_REGISTRY_INDEX.md`
- `docs/ai_context/11_SYMBOL_INDEX_RAW.txt`

## 查任务注册

```bash
rg -n "Isaac-Velocity-Flat-Anymal-D-v0" docs/ai_context/04_TASK_REGISTRY_INDEX.md
rg -n "gym\\.register|env_cfg_entry_point|rsl_rl_cfg_entry_point" source/isaaclab_tasks source/isaaclab_mimic
```

## 查某个类或配置

```bash
rg -n "class ManagerBasedRLEnv|class DirectRLEnv|class RslRlVecEnvWrapper" source
rg -n "AnymalDFlatEnvCfg|CartpoleEnvCfg|FrankaReachEnvCfg" docs/ai_context/11_SYMBOL_INDEX_RAW.txt
```

## 查 manager-based term

```bash
rg -n "RewardTerm|TerminationTerm|ObservationTerm|ActionTerm|CommandTerm" source/isaaclab_tasks/isaaclab_tasks/manager_based
rg -n "def .*reward|def .*termination|def .*command|def .*obs" source/isaaclab_tasks/isaaclab_tasks/manager_based source/isaaclab/isaaclab/envs/mdp
```

## 查 Direct env hook

```bash
rg -n "def _pre_physics_step|def _apply_action|def _get_observations|def _get_rewards|def _get_dones|def _reset_idx" source/isaaclab_tasks/isaaclab_tasks/direct
```

## 查 RSL-RL 链路

```bash
rg -n "RslRlVecEnvWrapper|OnPolicyRunner|DistillationRunner|clip_actions|obs_groups" scripts/reinforcement_learning/rsl_rl source/isaaclab_rl
rg -n "class .*PPORunnerCfg" source/isaaclab_tasks/isaaclab_tasks
```

## 查资产配置

```bash
rg -n "ArticulationCfg\\(|RigidObjectCfg\\(|UsdFileCfg\\(|UrdfFileCfg\\(|MjcfFileCfg\\(" source/isaaclab_assets source/isaaclab_tasks
rg -n "AGIBOT|ANYMAL|FRANKA|UNITREE|G1|H1" source/isaaclab_assets/isaaclab_assets/robots
```

## 查动作范围和控制路径

```bash
rg -n "clip_actions|clip=|scale=|offset=|JointPositionActionCfg|JointVelocityActionCfg|JointEffortActionCfg" source scripts
rg -n "stiffness|damping|effort_limit|velocity_limit|soft_joint_pos_limit_factor" source/isaaclab_assets source/isaaclab_tasks
```

## 查传感器

```bash
rg -n "CameraCfg|TiledCameraCfg|ContactSensorCfg|RayCasterCfg|ImuCfg|FrameTransformerCfg" source
rg -n "enable_cameras|render_interval|has_rtx_sensors|num_rerenders_on_reset" source scripts
```

## 查训练脚本入口

```bash
rg --files scripts/reinforcement_learning
rg -n "gym.make|parse_env_cfg|hydra_task_config|RecordVideo" scripts/reinforcement_learning scripts/environments
```

## 查目录结构

```bash
rg -n "manager_based/locomotion|direct/cartpole|isaaclab_rl|isaaclab_assets" docs/ai_context/02_DIRECTORY_TREE.txt
rg --files source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity
```

## 查 isaaclab_contrib 扩展

```bash
# 多旋翼/无人机相关
rg -n "Multirotor|ThrustAction|Thruster|MultiRotorActions" source/isaaclab_contrib
# 触觉传感器
rg -n "VisuoTactileSensor|GelSightRender|VisuoTactileSensorCfg" source/isaaclab_contrib
# 推力分配矩阵
rg -n "allocation_matrix|rotor_directions|thrust_command" source/isaaclab_contrib
```

## 查状态机环境

```bash
# 状态机相关
rg -rn "wp\.kernel|PickSmState|infer_state_machine|wp\.launch" scripts/environments/state_machine/
# Warp GPU 并行库使用
rg -rn "wp\.init\|wp\.func\|wp\.from_torch\|wp\.tid" scripts/environments/state_machine/
```

## 查四元数格式转换

```bash
# Isaac Sim 四元数 (w,x,y,z) vs Warp 四元数 (x,y,z,w) 的转换
rg -rn "\[0, 1, 2, 4, 5, 6, 3\]|\[0, 1, 2, 6, 3, 4, 5\]" scripts source
```

## 查仿真基础设置

```bash
# AppLauncher 启动流程
rg -n "AppLauncher\|add_app_launcher_args\|headless" scripts/tutorials/00_sim/
# 物理参数设置
rg -n "PhysxCfg\|dt\|decimation\|gravity\|solver_type" source/isaaclab/isaaclab/sim/
```

