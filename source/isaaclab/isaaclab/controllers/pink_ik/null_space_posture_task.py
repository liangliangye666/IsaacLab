# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pinocchio as pin
import scipy.linalg.blas as blas
import scipy.linalg.lapack as lapack
from pink.configuration import Configuration
from pink.tasks import Task


class NullSpacePostureTask(Task):
    r"""Pink-based task that adds a posture objective that is in the null space projection of other tasks.

    This task implements posture control in the null space of higher priority tasks
    (typically end-effector pose tasks) within the Pink inverse kinematics framework.

    **Mathematical Formulation:**

    For details on Pink Inverse Kinematics optimization formulation visit: https://github.com/stephane-caron/pink

    **Null Space Posture Task Implementation:**

    This task consists of two components:

    1. **Error Function**: The posture error is computed as:

    .. math::

        \mathbf{e}(\mathbf{q}) = \mathbf{M} \cdot (\mathbf{q}^* - \mathbf{q})

    where:
        - :math:`\mathbf{q}^*` is the target joint configuration
        - :math:`\mathbf{q}` is the current joint configuration
        - :math:`\mathbf{M}` is a joint selection mask matrix

    2. **Jacobian Matrix**: The task Jacobian is the null space projector:

    .. math::

        \mathbf{J}_{\text{posture}}(\mathbf{q}) = \mathbf{N}(\mathbf{q}) =
            \mathbf{I} -\mathbf{J}_{\text{primary}}^+ \mathbf{J}_{\text{primary}}

    where:
        - :math:`\mathbf{J}_{\text{primary}}` is the combined Jacobian of all higher priority tasks
        - :math:`\mathbf{J}_{\text{primary}}^+` is the pseudoinverse of the primary task Jacobian
        - :math:`\mathbf{N}(\mathbf{q})` is the null space projector matrix

    For example, if there are two frame tasks (e.g., controlling the pose of two end-effectors), the combined Jacobian
    :math:`\mathbf{J}_{\text{primary}}` is constructed by stacking the individual Jacobians for each frame vertically:

    .. math::

        \mathbf{J}_{\text{primary}} =
        \begin{bmatrix}
            \mathbf{J}_1(\mathbf{q}) \\
            \mathbf{J}_2(\mathbf{q})
        \end{bmatrix}

    where :math:`\mathbf{J}_1(\mathbf{q})` and :math:`\mathbf{J}_2(\mathbf{q})` are the Jacobians for the
    first and second frame tasks, respectively.

    The null space projector ensures that joint velocities in the null space produce zero velocity
    for the primary tasks: :math:`\mathbf{J}_{\text{primary}} \cdot \dot{\mathbf{q}}_{\text{null}} = \mathbf{0}`.

    **Task Integration:**

    When integrated into the Pink framework, this task contributes to the optimization as:

    .. math::

        \left\|
            \mathbf{N}(\mathbf{q}) \mathbf{v} + \mathbf{M} \cdot (\mathbf{q}^* - \mathbf{q})
        \right\|_{W_{\text{posture}}}^2

    This formulation allows the robot to maintain a desired posture while respecting the constraints
    imposed by higher priority tasks (e.g., end-effector positioning).

    """
    """基于粉红色的任务，添加在其他任务的零空间投影中的姿势目标。

    这项任务在粉红色反动动力学框架内实现了更高优先任务 (通常是最终效应器姿势任务) 的零空间姿势控制。

    **数学公式:**

    详细了解粉红色反动动力学优化配方访问:https://github.com/stephane-caron/pink

    **零空间姿势任务实施:**

    这项任务包括两个组成部分:

    1. **错误函数**:姿势错误计算为:

    .. math::

        \mathbf{e}(\mathbf{q}) = \mathbf{M} \cdot (\mathbf{q}^* - \mathbf{q})

    where:
        - :math:`\mathbf{q}^*`是目标联合配置
        - :math:`\mathbf{q}`是当前的联合配置
        - :数学:`\mathbf{M}`是一个联合选择面具矩阵

    2. **Jacobian Matrix**: Jacobian 的任务是零空间投影机:

    .. math::

        \mathbf{J}_{\text{posture}}(\mathbf{q}) = \mathbf{N}(\mathbf{q}) =
            \mathbf{I} -\mathbf{J}_{\text{primary}}^+ \mathbf{J}_{\text{primary}}

    where:
        - :数学:`\mathbf{J}_{\text{primary}}`是所有更高优先任务的联合Jacobian
        - :数学:`\mathbf{J}_{\text{primary}}^+`是主要任务Jacobian的伪逆
        - :数学:`\mathbf{N}(\mathbf{q})`是零空间投影矩阵

    例如，如果有两个框架任务 (e.g.，控制两个末端执行器位姿)，
    :math:`\mathbf{J}_{\text{primary}}`是通过垂直堆叠每个框架的单个Jacobians来构建的:

    .. math::

        \mathbf{J}_{\text{primary}} =
        \begin{bmatrix}
            \mathbf{J}_1(\mathbf{q}) \\
            \mathbf{J}_2(\mathbf{q})
        \end{bmatrix}

    where :数学:`\mathbf{J}_1(\mathbf{q})`和:数学:`\mathbf{J}_2(\mathbf{q})`是Jacobians为
    分别的第一和第二个框架任务。

    零空间投影器确保零空间中的关节速度产生零速度
    for the primary tasks: :math:`\mathbf{J}_{\text{primary}} \cdot \dot{\mathbf{q}}_{\text{null}} = \mathbf{0}`.

    **任务集成:**

    在 Pink 框架中集成时，该任务有助于优化:

    .. math::

        \left\|
            \mathbf{N}(\mathbf{q}) \mathbf{v} + \mathbf{M} \cdot (\mathbf{q}^* - \mathbf{q})
        \right\|_{W_{\text{posture}}}^2

    这种配方使机器人能够保持所需的姿势，同时尊重较高优先任务 (e.g.，终端效应物定位) 所造成的限制。
    """

    # Regularization factor for pseudoinverse computation to ensure numerical stability
    PSEUDOINVERSE_DAMPING_FACTOR: float = 1e-9

    def __init__(
        self,
        cost: float,
        lm_damping: float = 0.0,
        gain: float = 1.0,
        controlled_frames: list[str] | None = None,
        controlled_joints: list[str] | None = None,
    ) -> None:
        r"""Initialize the null space posture task.

        This task maintains a desired joint posture in the null space of higher-priority
        frame tasks. Joint selection allows excluding specific joints (e.g., wrist joints
        in humanoid manipulation) to prevent large rotational ranges from overwhelming
        errors in critical joints like shoulders and waist.

        Args:
            cost: Task weighting factor in the optimization objective.
                Units: :math:`[\text{cost}] / [\text{rad}]`.
            lm_damping: Levenberg-Marquardt regularization scale (unitless). Defaults to 0.0.
            gain: Task gain :math:`\alpha \in [0, 1]` for low-pass filtering.
                Defaults to 1.0 (no filtering).
            controlled_frames: Frame names whose Jacobians define the primary tasks for
                null space projection. If None or empty, no projection is applied.
            controlled_joints: Joint names to control in the posture task. If None or
                empty, all actuated joints are controlled.
        """
        """启动零空间姿势任务。

        这项任务在优先级框架任务的零空间中保持所需的联合姿势。
        关节选择允许排除特定关节 (e.g.，人形操纵中的手腕关节)，以防止肩膀和腰部等关键关节中的巨大旋转错误。

        参数：
            cost: 优化目标中的任务权重因素。
                Units: 数学:`[\text{cost}] / [\text{rad}]`。
            lm_damping: 利文伯格-马卡尔特规律化尺度 (无单位)。
                        默认为0.0。
            gain: 任务收益:数学`\alpha \in [0， 1]`为低通道过。
                  默认到1.0 (没有过)。
            controlled_frames: 框架名称，其雅可比人定义了零空间投影的主要任务。
                               如果None或空，则不使用投影。
            controlled_joints: 在姿势任务中控制的共同名称。
                               如果None或空，所有动力关节都控制。
        """
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        self.target_q: np.ndarray | None = None
        self.controlled_frames: list[str] = controlled_frames or []
        self.controlled_joints: list[str] = controlled_joints or []
        self._joint_mask: np.ndarray | None = None
        self._frame_names: list[str] | None = None

    def __repr__(self) -> str:
        """Human-readable representation of the task."""
        """人能阅读任务的表现。"""
        return (
            f"NullSpacePostureTask(cost={self.cost}, gain={self.gain}, lm_damping={self.lm_damping},"
            f" controlled_frames={self.controlled_frames}, controlled_joints={self.controlled_joints})"
        )

    def _build_joint_mapping(self, configuration: Configuration) -> None:
        """Build joint mask and cache frequently used values.

        Creates a binary mask that selects which joints should be controlled
        in the posture task.

        Args:
            configuration: Robot configuration containing the model and joint information.
        """
        """构建常用的联合面具和缓存值。

        建立一个二进制面具，选择在姿势任务中应该控制哪些关节。

        参数：
            configuration: 包含模型和联合信息的机器人配置。
        """
        # Create joint mask for full configuration size
        self._joint_mask = np.zeros(configuration.model.nq)

        # Create dictionary for joint names to indices (exclude root joint)
        joint_names = configuration.model.names.tolist()[1:]

        # Build joint mask efficiently
        for i, joint_name in enumerate(joint_names):
            if joint_name in self.controlled_joints:
                self._joint_mask[i] = 1.0

        # Cache frame names for performance
        self._frame_names = list(self.controlled_frames)

    def set_target(self, target_q: np.ndarray) -> None:
        """Set target posture configuration.

        Args:
            target_q: Target vector in the configuration space. If the model
                has a floating base, then this vector should include
                floating-base coordinates (although they have no effect on the
                posture task since only actuated joints are controlled).
        """
        """设置目标姿势配置。

        参数：
            target_q: 在配置空间中的目标向量。
                      如果模型有一个浮动基，那么这个向量应该包括浮动基座位 (尽管它们对姿势任务没有影响，因为只有动力关节控制)。
        """
        self.target_q = target_q.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set target posture from a robot configuration.

        Args:
            configuration: Robot configuration whose joint angles will be used
                as the target posture.
        """
        """从机器人配置设置目标姿势。

        参数：
            configuration: 机器人配置，其合角将作为目标姿势。
        """
        self.set_target(configuration.q)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute posture task error.

        The error computation follows:

        .. math::

            \mathbf{e}(\mathbf{q}) = \mathbf{M} \cdot (\mathbf{q}^* - \mathbf{q})

        where :math:`\mathbf{M}` is the joint selection mask and :math:`\mathbf{q}^* - \mathbf{q}`
        is computed using Pinocchio's difference function to handle joint angle wrapping.

        Args:
            configuration: Robot configuration :math:`\mathbf{q}`.

        Returns:
            Posture task error :math:`\mathbf{e}(\mathbf{q})` with the same dimension
            as the configuration vector, but with zeros for non-controlled joints.

        Raises:
            ValueError: If no posture target has been set.
        """
        """计算姿势任务错误。

        错误计算如下:

        .. math::

            \mathbf{e}(\mathbf{q}) = \mathbf{M} \cdot (\mathbf{q}^* - \mathbf{q})

        where :数学:`\mathbf{M}`是联合选择面具和:math:`\mathbf{q}^* - \mathbf{q}`
        通过皮诺基奥的区别函数来计算，

        参数：
            configuration: 机器人配置:数学:`\mathbf{q}`。

        返回：
            姿势任务错误:数学:`\mathbf{e}(\mathbf{q})` 与配置向量相同的维度，但对于不受控制的关节有零。

        异常：
            ValueError: 如果没有设定位置目标。
        """
        if self.target_q is None:
            raise ValueError("No posture target has been set. Call set_target() first.")

        # Initialize joint mapping if needed
        if self._joint_mask is None:
            self._build_joint_mapping(configuration)

        # Compute configuration difference using Pinocchio's difference function
        # This handles joint angle wrapping correctly
        err = pin.difference(
            configuration.model,
            self.target_q,
            configuration.q,
        )

        # Apply pre-computed joint mask to select only controlled joints
        return self._joint_mask * err

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the null space projector Jacobian.

        The null space projector is defined as:

        .. math::

            \mathbf{N}(\mathbf{q}) = \mathbf{I} - \mathbf{J}_{\text{primary}}^+ \mathbf{J}_{\text{primary}}

        where:
            - :math:`\mathbf{J}_{\text{primary}}` is the combined Jacobian of all controlled frames
            - :math:`\mathbf{J}_{\text{primary}}^+` is the pseudoinverse of the primary task Jacobian
            - :math:`\mathbf{I}` is the identity matrix

        The null space projector ensures that joint velocities in the null space produce
        zero velocity for the primary tasks:
        :math:`\mathbf{J}_{\text{primary}} \cdot \dot{\mathbf{q}}_{\text{null}} = \mathbf{0}`.

        If no controlled frames are specified, returns the identity matrix.

        Args:
            configuration: Robot configuration :math:`\mathbf{q}`.

        Returns:
            Null space projector matrix :math:`\mathbf{N}(\mathbf{q})` with dimensions
            :math:`n_q \times n_q` where :math:`n_q` is the number of configuration variables.
        """
        """计算零空间投影机Jacobian。

        零空间投影器定义为:

        .. math::

            \mathbf{N}(\mathbf{q}) = \mathbf{I} - \mathbf{J}_{\text{primary}}^+ \mathbf{J}_{\text{primary}}

        where:
            - :math:`\mathbf{J}_{\text{primary}}`是所有控制框架的联合Jacobian
            - :数学:`\mathbf{J}_{\text{primary}}^+`是主要任务Jacobian的伪逆
            - :数学:`\mathbf{I}`是身份矩阵

        零空间投影器确保零空间中的关节速度为主要任务产生零速度:
        :math:`\mathbf{J}_{\text{primary}} \cdot \dot{\mathbf{q}}_{\text{null}} = \mathbf{0}`。

        如果没有指定控制框架，则返回身份矩阵。

        参数：
            configuration: 机器人配置:数学:`\mathbf{q}`。

        返回：
            零空间投影仪矩阵:数学:`\mathbf{N}(\mathbf{q})` 尺寸
            :math:`n_q \times n_q`在哪里:`n_q`是配置变量的数量。
        """
        # Initialize joint mapping if needed
        if self._frame_names is None:
            self._build_joint_mapping(configuration)

        # If no frame tasks are defined, return identity matrix (no null space projection)
        if not self._frame_names:
            return np.eye(configuration.model.nq)

        # Get Jacobians for all frame tasks and combine them
        J_frame_tasks = [configuration.get_frame_jacobian(frame_name) for frame_name in self._frame_names]
        J_combined = np.concatenate(J_frame_tasks, axis=0)

        # Compute null space projector: N = I - J^+ * J
        # Use fast pseudoinverse computation with direct LAPACK/BLAS calls
        m, n = J_combined.shape

        # Wide matrix (typical for robotics): use left pseudoinverse
        # J^+ = J^T @ inv(J @ J^T + λ²I)
        # This is faster because we invert an m×m matrix instead of n×n

        # Compute J @ J^T using BLAS (faster than numpy)
        JJT = blas.dgemm(1.0, J_combined, J_combined.T)
        np.fill_diagonal(JJT, JJT.diagonal() + self.PSEUDOINVERSE_DAMPING_FACTOR**2)

        # Use LAPACK's Cholesky factorization (dpotrf = Positive definite TRiangular Factorization)
        L, info = lapack.dpotrf(JJT, lower=1, clean=False, overwrite_a=True)

        if info != 0:
            # Fallback if not positive definite: use numpy's pseudoinverse
            J_pinv = np.linalg.pinv(J_combined)
            return np.eye(n) - J_pinv @ J_combined

        # Solve (J @ J^T + λ²I) @ X = J using LAPACK's triangular solver (dpotrs)
        # This directly solves the system without computing the full inverse
        X, _ = lapack.dpotrs(L, J_combined, lower=1)

        # Compute null space projector: N = I - J^T @ X
        N_combined = np.eye(n) - J_combined.T @ X

        return N_combined
