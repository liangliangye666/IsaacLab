# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import (
    binary_joint_actions,
    joint_actions,
    joint_actions_to_limits,
    non_holonomic_actions,
    surface_gripper_actions,
    task_space_actions,
)

##
# Joint actions.
##


@configclass
class JointActionCfg(ActionTermCfg):
    """Configuration for the base joint action term.

    See :class:`JointAction` for more details.
    """
    """基本联合动作项的配置

    See :分类:`JointAction` 详细信息。
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    """列出将该动作映射到的联合名称或regex表达式。"""
    '''
    决定动作控制哪些关节
        支持两种写法：
            写法	        含义	    举例
            精确名称	    逐个写出	["L_hip_yaw", "L_hip_roll", "L_hip_pitch"]
            正则表达式	    通配匹配	["L_.*"] 匹配所有左腿关节
        正则匹配是第二个用法——你不需要逐个列举机器人所有关节名。对轮式双足来说，左腿 6 个关节写 ["L_.*"] 一条搞定。

        内部解析（joint_actions.py:87）：
            self._joint_ids, self._joint_names = self._asset.find_joints(
                self.cfg.joint_names, preserve_order=self.cfg.preserve_order
            )
        _asset.find_joints() 用 name 或正则去关节体模型里匹配，返回对应的索引列表和解析后的名字。
    '''

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    """动作的尺度因子 (regex表达式的浮动或定位)。
    默认到1.0。
    """
    '''
    动作缩放系数
        两种类型对应两种控制粒度：
            # 方式 A：统一缩放（所有关节一样）
            scale = 0.5   → 策略输出的每个维都 × 0.5

            # 方式 B：按关节正则分组差异化缩放
            scale = {
                ".*hip.*": 1.0,       # 髋关节：正常灵敏度
                ".*knee.*": 0.5,      # 膝关节：减半（更稳定）
                ".*ankle.*": 0.3,     # 踝关节：更小（微调用）
            }
    '''
    '''
        processed_action = offset + scale × raw_action
    '''

    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
    """动作的抵消因素 (regex表达式的浮动或定值)。
    默认为0.0。
    """
    '''
    动作偏移量
        两种类型对应两种控制粒度：
            # 统一偏移
            offset = 0.1   → 每个关节都偏移 0.1 弧度

            # 按关节正则分组差异化偏移
            offset = {
                ".*hip_pitch.*": 0.2,   # 髋关节前倾 0.2 rad
                ".*knee.*": -0.1,       # 膝关节微弯
            }
        offset 的默认值经常被子类覆盖：
            # JointPositionActionCfg (actions_cfg.py:72):
            use_default_offset: bool = True
            # → 内部把 offset 设为机器人 USD 文件中定义的默认关节角度
            # → 策略输出 0 → 关节回到"自然站立"姿态

            # RelativeJointPositionActionCfg (actions_cfg.py:103):
            use_zero_offset: bool = True
            # → 内部把 offset 强制设为 0
            # → 策略输出是相对于当前位置的增量
        通俗理解：offset 是关节的"零点"。对于绝对位置控制，零点 = 机器人的默认站姿；对于增量控制，零点 = 0（从当前角度出发偏移）。
    '''

    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""
    """在动作输出中是否保留联合名称的顺序。
    默认为 False。
    """
    '''
    保持关节顺序
        值	            行为
        False（默认）	 按 _asset 内部字典序排列（可能和你配置的顺序不同）
        True	        严格保持 joint_names 列表的排列顺序
    为什么默认 False？ 
        因为 _asset.find_joints() 后如果关节 ID 是连续的（如 [3,4,5,6,7,8]），会直接替换为 slice(None)（joint_actions.py:98-99），省去索引开销。
        而 preserve_order=True 强制保留索引列表，不能优化。

    什么时候需要 True？
        当你配置的关节顺序对策略有语义含义时（如"前 3 维是左腿，后 3 维是右腿"），必须保持。否则策略训练出的权重和实际的关节排列不一致。
    '''

'''
绝对关节位置控制的配置
'''
@configclass
class JointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """
    """联合立场动作项的配置

    See :分类:`JointPositionAction` 详细信息。
    """

    class_type: type[ActionTerm] = joint_actions.JointPositionAction
    '''
    class_type — 配置到实现的连接桥
        class_type: type[ActionTerm] = joint_actions.JointPositionAction
        这个字段的值是一个类引用（不是字符串，不是实例，而是类本身）。ActionManager 用它来创建实际的 ActionTerm 对象：
            # ActionManager._prepare_terms 中:
            term = term_cfg.class_type(term_cfg, env)
            #     = JointPositionAction(cfg, env)
    '''

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """
    """在关节资产中配置的默认关联位置是否应作为抵消。
    默认为 True。

    如果 True，该标志将:attr:`offset`的值重写到默认的关节位置
    from the articulation asset.
    """
    '''
    这个值决定了 JointPositionAction.__init__ 中 offset 的最终来源（joint_actions.py:237）：
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
            #                                   ↑
            #                    从机器人 USD 模型读取的"默认站姿"角度
    '''
    '''
    完整的数据流
        用户配置:
            JointPositionActionCfg(
                asset_name="robot",
                joint_names=[".*leg.*"],     ← JointActionCfg 的字段
                scale=0.5,                   ← JointActionCfg 的字段
                offset=0.0,                  ← JointActionCfg 的字段
                use_default_offset=True,     ← JointPositionActionCfg 的字段（覆盖 offset）
            )
                │
                ▼
        JointAction.__init__():
            ① _asset.find_joints(joint_names)  → 解析出关节 ID 列表
            ② 解析 scale/offset（处理 dict 类型的正则分组）
            ③ 如果 use_default_offset=True → offset = 关节默认角度
                │
                ▼
        JointPositionAction.apply_actions():
            processed = offset + scale × raw_action
            写入 _asset.set_joint_position_target(processed)
    '''


'''
增量关节位置控制的配置
'''
@configclass
class RelativeJointPositionActionCfg(JointActionCfg):
    """Configuration for the relative joint position action term.

    See :class:`RelativeJointPositionAction` for more details.
    """
    """对相对关节位置动作项的配置。

    See :分类:`RelativeJointPositionAction` 详细信息。
    """

    class_type: type[ActionTerm] = joint_actions.RelativeJointPositionAction

    use_zero_offset: bool = True
    """Whether to ignore the offset defined in articulation asset. Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to zero.
    """
    """在关节资产中定义的抵消是否被忽视。
    默认为 True。

    如果是True，这个标志将:attr:`offset`的值重写为零。
    """
    '''
    在 RelativeJointPositionAction.__init__ 中（joint_actions.py:287）：
        if cfg.use_zero_offset:
            self._offset = 0.0
    为什么 offset 必须为 0？ 
        回顾 process_actions 的公式和 apply_actions 的差异：
            # 父类 JointAction.process_actions:
            processed = raw × scale + offset

            # RelativeJointPositionAction.apply_actions:
            current_actions = processed + self._asset.data.joint_pos
            self._asset.set_joint_position_target(current_actions)
        把公式展开：
            实际目标 = (raw × scale + offset) + 当前关节角度
                    = raw × scale + offset + 当前关节角度
        如果 offset = default_joint_pos（像绝对位置那样）：
            实际目标 = raw × scale + default_joint_pos + 当前关节角度
            策略输出 0 → 目标 = default_joint_pos + 当前关节角度
                            = 从当前角度出发，再偏移一个完整站姿角度！
        这完全不合理——策略输出 0 理应"保持不动"，结果却跳了一个完整站姿。所以增量控制下 offset 必须强制为 0：
            实际目标 = raw × scale + 0.0 + 当前关节角度 = raw × scale + 当前关节角度
            策略输出 0 → 目标 = 当前关节角度 ✅ 保持不动
            策略输出 Δ → 目标 = 当前关节角度 + Δ  ✅ 偏移 Δ
    '''
    '''
    和 JointPositionActionCfg 的完整对比
                            JointPositionActionCfg（绝对）	        RelativeJointPositionActionCfg（增量）
        offset 来源	        default_joint_pos（自然站姿）	            强制 0.0
        默认标志	        use_default_offset=True	                 use_zero_offset=True
        process_actions	    raw × scale + default_joint_pos	        raw × scale + 0.0
        apply_actions	    set_joint_position_target(processed)	set_joint_position_target(current_joint_pos + processed)
        策略输出 0	        回到自然站姿	                            保持当前姿态
        策略输出 Δ	        到达 default + Δ 的绝对角度	                从当前位置偏移 Δ
    '''


'''
关节速度控制的配置
'''
@configclass
class JointVelocityActionCfg(JointActionCfg):
    """Configuration for the joint velocity action term.

    See :class:`JointVelocityAction` for more details.
    """
    """对于关节速度动作项的配置。

    See :分类:`JointVelocityAction` 详细信息。
    """

    class_type: type[ActionTerm] = joint_actions.JointVelocityAction

    use_default_offset: bool = True
    """Whether to use default joint velocities configured in the articulation asset as offset.
    Defaults to True.

    This overrides the settings from :attr:`offset` if set to True.
    """
    """在关节资产中配置的默认关节速度是否应作为抵消。
    默认为 True。

    如果设置为True，则将:attr:`offset`的设置取消。
    """
    '''
    在 JointVelocityAction.__init__ 中（joint_actions.py:312）：
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()
    和位置控制的类比与差异：
                                    JointPositionAction	                        JointVelocityAction
        use_default_offset 的语义	offset = 默认关节角度（如膝关节 -0.3 rad）	    offset = 默认关节角速度（通常全是 0.0）
        数据来源	                default_joint_pos	                        default_joint_vel
        默认 offset 的实际值	    机器人的自然站姿角度	                        全零向量（任何机器人从静止开始）
        策略输出 0	                关节去默认角度	                                关节不动（零速度）
    速度控制的物理公式
        回顾 process_actions（joint_actions.py:214）和 apply_actions（joint_actions.py:316）：
            # 父类 JointAction.process_actions:
            processed = raw × scale + offset

            # JointVelocityAction.apply_actions:
            self._asset.set_joint_velocity_target(processed)
        展开：
            关节目标速度 = raw × scale + default_joint_vel
                        = raw × scale + 0.0       （因为 default_joint_vel ≈ 0）

            策略输出 0    → 所有关节速度为 0 → 机器人保持当前姿态不动
            策略输出 +1.0 → 关节以 scale 弧度/秒的速度正向转动（如果 scale=1.0）
            策略输出 -0.5 → 关节以 0.5×scale 弧度/秒的速度反向转动
    '''


'''
关节力矩控制的配置
'''
@configclass
class JointEffortActionCfg(JointActionCfg):
    """Configuration for the joint effort action term.

    See :class:`JointEffortAction` for more details.
    """
    """共同努力动作项的配置。

    See :分类:`JointEffortAction` 详细信息。
    """

    class_type: type[ActionTerm] = joint_actions.JointEffortAction
    '''
    力矩控制的物理语义
        力矩控制（effort control / torque control）是最底层的控制模式——策略直接输出力矩命令（单位：N·m），没有 PD 控制器的中间层：
            位置控制: 策略 → PD(target) → 力矩 → 关节  （PD 自动纠偏）
            速度控制: 策略 → PD(target) → 力矩 → 关节  （同上）
            力矩控制: 策略 → 力矩 → 关节              （直接！无 PD 保护）
        优点：最本质的物理控制，适合力控、阻抗控制等高级场景。
        缺点：训练难度极高——没有 PD 的安全网，策略输出不当时机器人直接狂暴。
    '''


##
# Joint actions rescaled to limits.
##


'''
[-1,1] 映射到关节限位
归一化关节位置控制（限位映射）
    和前四个关节动作不同，它直接继承 ActionTermCfg 而非 JointActionCfg——因为它有自己独立的处理流水线，不需要 JointAction 基类的 offset 语义
'''
@configclass
class JointPositionToLimitsActionCfg(ActionTermCfg):
    """Configuration for the bounded joint position action term.

    See :class:`JointPositionToLimitsAction` for more details.
    """
    """限制关节位置动作项的配置

    See :分类:`JointPositionToLimitsAction` 详细信息。
    """

    class_type: type[ActionTerm] = joint_actions_to_limits.JointPositionToLimitsAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    """列出将该动作映射到的联合名称或regex表达式。"""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    """动作的尺度因子 (regex表达式的浮动或定位)。
    默认到1.0。
    """

    rescale_to_limits: bool = True
    """Whether to rescale the action to the joint limits. Defaults to True.

    If True, the input actions are rescaled to the joint limits, i.e., the action value in
    the range [-1, 1] corresponds to the joint lower and upper limits respectively.

    Note:
        This operation is performed after applying the scale factor.
    """
    """是否将动作重新扩展到联合限制。
    默认为 True。

    如果True，输入操作将重新扩展到联合限度，i.e.，范围内的动作值[-1， 1]相应于分别的联合下限和上限。

    说明：
        这种操作是在使用尺度因子后进行的。
    """
    '''
    在 process_actions 中（joint_actions_to_limits.py:198）：
            if self.cfg.rescale_to_limits:
                actions = self._processed_actions.clamp(-1.0, 1.0)    # ① 先夹紧到 [-1, 1]
                actions = math_utils.unscale_transform(                 # ② 再映射到限位
                    actions,
                    self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],  # lower
                    self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],  # upper
                )
        unscale_transform 做了什么（线性映射）：
            策略输出 -1.0  →  关节下限（joint_lower_limit）
            策略输出  0.0  →  关节中心（(lower + upper) / 2）
            策略输出  1.0  →  关节上限（joint_upper_limit）
        实际例子：假设膝关节限位 [-2.0, 1.5] rad：
            策略输出 -1.0 → 膝关节目标 = -2.0 rad（完全弯曲）
            策略输出  0.0 → 膝关节目标 = -0.25 rad（中心位置）
            策略输出  0.5 → 膝关节目标 =  0.625 rad（略伸展）
            策略输出  1.0 → 膝关节目标 =  1.5 rad（完全伸直）
            策略输出  2.0 → 先 clamp 到 1.0 → 1.5 rad（超出也安全）

    和 JointPositionAction 的流水线对比
        JointPositionAction:
            raw × scale → + offset(default_joint_pos) → clip → set_joint_position_target
                                                            ↑
                                                    策略需要知道自己该输出多少弧度

        JointPositionToLimitsAction:
            raw × scale → clip（可选）→ clamp [-1,1] → unscale to limits → set_joint_position_target
                                                        ↑
                                                策略只需输出 [-1,1] 归一化值！
    '''

    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""
    """在动作输出中是否保留联合名称的顺序。
    默认为 False。
    """


@configclass
class EMAJointPositionToLimitsActionCfg(JointPositionToLimitsActionCfg):
    """Configuration for the exponential moving average (EMA) joint position action term.

    See :class:`EMAJointPositionToLimitsAction` for more details.
    """
    """对指数移动平均值 (EMA) 关节位置动作项的配置。

    See :分类:`EMAJointPositionToLimitsAction` 详细信息。
    """

    class_type: type[ActionTerm] = joint_actions_to_limits.EMAJointPositionToLimitsAction

    alpha: float | dict[str, float] = 1.0
    """The weight for the moving average (float or dict of regex expressions). Defaults to 1.0.

    If set to 1.0, the processed action is applied directly without any moving average window.
    """
    """移动平均值的重量 (regex表达式的浮动或定值)。
    默认到1.0。

    如果设置为1.0，处理的操作将直接进行，没有任何移动平均窗口。
    """


##
# Gripper actions.
##


@configclass
class BinaryJointActionCfg(ActionTermCfg):
    """Configuration for the base binary joint action term.

    See :class:`BinaryJointAction` for more details.
    """
    """基本二元联合动作项的配置

    See :分类:`BinaryJointAction` 详细信息。
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    """列出将该动作映射到的联合名称或regex表达式。"""
    open_command_expr: dict[str, float] = MISSING
    """The joint command to move to *open* configuration."""
    """移动到#开#配置的联合命令。"""
    close_command_expr: dict[str, float] = MISSING
    """The joint command to move to *close* configuration."""
    """移动到"接近"配置。"""


@configclass
class BinaryJointPositionActionCfg(BinaryJointActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """
    """对二进制关节位置动作项的配置

    See :分类:`BinaryJointPositionAction` 详细信息。
    """

    class_type: type[ActionTerm] = binary_joint_actions.BinaryJointPositionAction


@configclass
class BinaryJointVelocityActionCfg(BinaryJointActionCfg):
    """Configuration for the binary joint velocity action term.

    See :class:`BinaryJointVelocityAction` for more details.
    """
    """对二进制关节速度操作项的配置。

    See :分类:`BinaryJointVelocityAction` 详细信息。
    """

    class_type: type[ActionTerm] = binary_joint_actions.BinaryJointVelocityAction


@configclass
class AbsBinaryJointPositionActionCfg(ActionTermCfg):
    """Configuration for the absolute binary joint position action term.

    This action term is used for robust grasping by converting continuous gripper joint position actions
    into binary open/close commands. Unlike directly applying continuous gripper joint position actions, this class
    applies a threshold-based decision mechanism to determine whether to
    open or close the gripper.

    The action works by:
    1. Taking a continuous input action value
    2. Comparing it against a configurable threshold
    3. Mapping the result to either open or close commands based on the threshold comparison
    4. Applying the corresponding gripper open/close commands

    This approach provides more predictable and stable grasping behavior compared to directly applying
    continuous gripper joint position actions.

    See :class:`AbsBinaryJointPositionAction` for more details.
    """
    """对于绝对二元关节位置动作项的配置。

    这种动作项用于通过将连续抓住器关节位置动作转换为双式开/关闭命令来强大的抓住。
    与直接应用连续抓紧机关位置操作不同，该类应用基于门的决策机制来确定是否打开或关闭抓紧机。

    动作由:
    1. 采用连续输入动作值
    2. 与可配置的门进行比较
    3. 根据门比较，将结果映射成开放或关闭命令
    4. 使用相应的抓住器打开/关闭命令

    这种方法比直接应用连续抓住器关节位置操作更具可预测性和稳定的抓住行为。

    See :分类:`AbsBinaryJointPositionAction` 详细信息。
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    """列出将该动作映射到的联合名称或regex表达式。"""
    open_command_expr: dict[str, float] = MISSING
    """The joint command to move to *open* configuration."""
    """移动到#开#配置的联合命令。"""
    close_command_expr: dict[str, float] = MISSING
    """The joint command to move to *close* configuration."""
    """移动到"接近"配置。"""
    threshold: float = 0.5
    """The threshold for the binary action. Defaults to 0.5."""
    """双向动作的门。
    默认为0.5。
    """
    positive_threshold: bool = True
    """Whether to use positive (Open actions > Close actions) threshold. Defaults to True."""
    """是否使用正值 (开放动作>关闭动作) 门。
    默认为 True。
    """

    class_type: type[ActionTerm] = binary_joint_actions.AbsBinaryJointPositionAction


##
# Non-holonomic actions.
##

'''
轮式底盘的配置（2D 速度控制）
    这是轮式双足底盘的核心配置。和之前所有关节动作不同——它控制的是三个虚拟关节（不是真实轮子），策略只需输出 2 维向量 [v_x, ω_z]。
'''
@configclass
class NonHolonomicActionCfg(ActionTermCfg):
    """Configuration for the non-holonomic action term with dummy joints at the base.

    See :class:`NonHolonomicAction` for more details.
    """
    """设置非全态动作项，底部设置模具关节。

    See :分类:`NonHolonomicAction` 详细信息。
    """

    class_type: type[ActionTerm] = non_holonomic_actions.NonHolonomicAction

    body_name: str = MISSING        # 底盘刚体名（如 "base"）
    """Name of the body which has the dummy mechanism connected to."""
    """机器的名称。"""
    x_joint_name: str = MISSING     # x 方向虚拟棱柱关节
    """The dummy joint name in the x direction."""
    """在 x 方向的仿真联合名称。"""
    y_joint_name: str = MISSING     # y 方向虚拟棱柱关节
    """The dummy joint name in the y direction."""
    """在"y"方向的模糊名称。"""
    yaw_joint_name: str = MISSING   # z 旋转虚拟旋转关节
    """The dummy joint name in the yaw direction."""
    """在的方向上，的联合名字。"""
    scale: tuple[float, float] = (1.0, 1.0)
    """Scale factor for the action. Defaults to (1.0, 1.0)."""
    """动作的规模因素。
    默认的 (1.0， 1.0)。
    """
    offset: tuple[float, float] = (0.0, 0.0)
    """Offset factor for the action. Defaults to (0.0, 0.0)."""
    """这种动作的抵消因素。
    默认值为 (0.0，0.0)。
    """


##
# Task-space Actions.
##


'''
末端位姿命令，经 IK 转关节位置
'''
@configclass
class DifferentialInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for inverse differential kinematics action term.

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """
    """对逆差差动力学操作项的配置。

    See :分类:`DifferentialInverseKinematicsAction` 详细信息。
    """

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """
        """从父母的框架到孩子的框架。

        在许多机器人上，最终效应器框架是虚构的框架，
        在这种情况下，更容易定义这个变化w.r.t。
        它们的父母的身体是固定的。
        例如，对于Franka Emika臂，末端执行器的定义是对"panda_hand"框架的偏移。
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        """翻译w.r.t
        它们的母体。
        在 (0.0，0.0，0.0) 之前的默认设置。
        """
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""
        """四元数旋转 ``(w， x， y， z)`` w.r.t。
        它们的母体。
        在 (1.0，0.0，0.0，0.0，0.0的默认情况下。
        """

    class_type: type[ActionTerm] = task_space_actions.DifferentialInverseKinematicsAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    """列出将该动作映射到的联合名称或regex表达式。"""
    body_name: str = MISSING
    """Name of the body or frame for which IK is performed."""
    """执行IK的机体或框架名称。"""
    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""
    """目标框架 w.r.t的抵消。
    在身体框架。
    默认对None的缺陷，在这种情况下，不使用任何抵消。
    """
    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    """动作的规模因素。
    默认到1.0。
    """
    controller: DifferentialIKControllerCfg = MISSING
    """The configuration for the differential IK controller."""
    """对差异性IK控制器的配置。"""


'''
任务空间控制
'''
@configclass
class OperationalSpaceControllerActionCfg(ActionTermCfg):
    """Configuration for operational space controller action term.

    See :class:`OperationalSpaceControllerAction` for more details.
    """
    """操作空间控制器操作项的配置

    See :分类:`OperationalSpaceControllerAction` 详细信息。
    """

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """
        """从父母的框架到孩子的框架。

        在许多机器人上，最终效应器框架是虚构的框架，
        在这种情况下，更容易定义这个变化w.r.t。
        它们的父母的身体是固定的。
        例如，对于Franka Emika臂，末端执行器的定义是对"panda_hand"框架的偏移。
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        """翻译w.r.t
        它们的母体。
        在 (0.0，0.0，0.0) 之前的默认设置。
        """
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""
        """四元数旋转 ``(w， x， y， z)`` w.r.t。
        它们的母体。
        在 (1.0，0.0，0.0，0.0，0.0的默认情况下。
        """

    class_type: type[ActionTerm] = task_space_actions.OperationalSpaceControllerAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    """列出将该动作映射到的联合名称或regex表达式。"""

    body_name: str = MISSING
    """Name of the body or frame for which motion/force control is performed."""
    """运动/力控制所执行的车身或框架名称。"""

    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""
    """目标框架 w.r.t的抵消。
    在身体框架。
    默认对None的缺陷，在这种情况下，不使用任何抵消。
    """

    task_frame_rel_path: str = None
    """The path of a ``RigidObject``, relative to the sub-environment, representing task frame. Defaults to None."""
    """对子环境的 ``RigidObject`` 路径，代表任务框架。
    默认为 None。
    """

    controller_cfg: OperationalSpaceControllerCfg = MISSING
    """The configuration for the operational space controller."""
    """操作空间控制器的配置。"""

    position_scale: float = 1.0
    """Scale factor for the position targets. Defaults to 1.0."""
    """位置目标的规模因素。
    默认到1.0。
    """

    orientation_scale: float = 1.0
    """Scale factor for the orientation (quad for ``pose_abs`` or axis-angle for ``pose_rel``). Defaults to 1.0."""
    """方向的尺度因子 (``pose_abs``的四角或``pose_rel``的轴角)。
    默认到1.0。
    """

    wrench_scale: float = 1.0
    """Scale factor for the wrench targets. Defaults to 1.0."""
    """关目标的规模因素。
    默认到1.0。
    """

    stiffness_scale: float = 1.0
    """Scale factor for the stiffness commands. Defaults to 1.0."""
    """硬度指令的尺度因素。
    默认到1.0。
    """

    damping_ratio_scale: float = 1.0
    """Scale factor for the damping ratio commands. Defaults to 1.0."""
    """放缓率指令的尺度因子。
    默认到1.0。
    """

    nullspace_joint_pos_target: str = "none"
    """The joint targets for the null-space control: ``"none"``, ``"zero"``, ``"default"``, ``"center"``.

    Note: Functional only when ``nullspace_control`` is set to ``"position"`` within the
        ``OperationalSpaceControllerCfg``.
    """
    """零空间控制的共同目标:``"none"``，``"zero"``，``"default"``，``"center"``。

    Note: 只有当``nullspace_control``在``OperationalSpaceControllerCfg``内设置为``"position"``时才会运行。
    """


##
# Surface Gripper actions.
##


@configclass
class SurfaceGripperBinaryActionCfg(ActionTermCfg):
    """Configuration for the binary surface gripper action term.

    See :class:`SurfaceGripperBinaryAction` for more details.
    """
    """对二进制表面抓住器操作项的配置。

    See :分类:`SurfaceGripperBinaryAction` 详细信息。
    """

    asset_name: str = MISSING
    """Name of the surface gripper asset in the scene."""
    """在场景的表面抓住器的名称。"""
    open_command: float = -1.0
    """The command value to open the gripper. Defaults to -1.0."""
    """打开抓住器的命令值。
    设置为 -1.0。
    """
    close_command: float = 1.0
    """The command value to close the gripper. Defaults to 1.0."""
    """关闭抓住器的命令值。
    默认到1.0。
    """

    class_type: type[ActionTerm] = surface_gripper_actions.SurfaceGripperBinaryAction
