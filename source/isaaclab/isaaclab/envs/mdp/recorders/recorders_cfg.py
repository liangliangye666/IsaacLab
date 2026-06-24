# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from . import recorders

##
# State recorders.
##


@configclass
class InitialStateRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state recorder term."""
    """首个状态记录器的配置。"""

    class_type: type[RecorderTerm] = recorders.InitialStateRecorder


@configclass
class PostStepStatesRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""
    """步骤状态记录器项的配置"""

    class_type: type[RecorderTerm] = recorders.PostStepStatesRecorder


@configclass
class PreStepActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step action recorder term."""
    """步骤动作记录器项的配置"""

    class_type: type[RecorderTerm] = recorders.PreStepActionsRecorder


@configclass
class PreStepFlatPolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""
    """步骤策略观测记录器的配置"""

    class_type: type[RecorderTerm] = recorders.PreStepFlatPolicyObservationsRecorder


@configclass
class PostStepProcessedActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the post step processed actions recorder term."""
    """后步处理动作记录器项的配置"""

    class_type: type[RecorderTerm] = recorders.PostStepProcessedActionsRecorder


##
# Recorder manager configurations.
##


@configclass
class ActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""
    """记录操作和状态的记录配置。"""

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()
    record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()
