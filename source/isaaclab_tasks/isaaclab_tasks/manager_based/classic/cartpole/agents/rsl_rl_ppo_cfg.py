# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

import isaaclab_tasks.manager_based.classic.cartpole.mdp.symmetry as symmetry


'''
17.1 Runner 参数
    num_steps_per_env：每次 rollout 每个环境收集多少步。
    max_iterations：最大 PPO 更新次数。
    save_interval：checkpoint 间隔。
    experiment_name：日志目录名。
    seed：随机种子。
    clip_actions：进入环境前的动作裁剪。
    resume/load_run/load_checkpoint：恢复训练。
17.2 Policy 参数
    actor/critic 网络层数。
    hidden dims。
    activation。
    初始动作噪声。
    observation normalization。
    recurrent policy 配置。
    旧网络参数应迁移到这里，而不是 env cfg。
17.3 Algorithm 参数
    learning_rate
    gamma
    lam
    clip_param
    entropy_coef
    value_loss_coef
    num_learning_epochs
    num_mini_batches
    desired_kl
    max_grad_norm
'''
@configclass
class CartpolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CartpolePPORunnerWithSymmetryCfg(CartpolePPORunnerCfg):
    """Configuration for the PPO agent with symmetry augmentation."""
    """设置PPO代理与对称增强。"""

    # all the other settings are inherited from the parent class
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=symmetry.compute_symmetric_states
        ),
    )
