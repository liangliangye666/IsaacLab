# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations
"""可以用来创建观测项的共同函数。

函数可以传递到:class:`isaaclab.managers.ObservationTermCfg`对象，以实现函数引入的观测。
"""

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)

"""
Root state.
"""
"""根源状态。
"""


@generic_io_descriptor(units="m", axes=["Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype])
def base_pos_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in the simulation world frame."""
    """在仿真世界框架中的根高度。"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


@generic_io_descriptor(
    units="m/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    """在资产的根框架中的根线性速度。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


@generic_io_descriptor(
    units="rad/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    """在资产的根框架中的根角速度。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


@generic_io_descriptor(
    units="m/s^2", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    """在资产的根框架上的重力投影。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


@generic_io_descriptor(
    units="m", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    """资产根位置在环境框架中"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


@generic_io_descriptor(
    units="unit", axes=["W", "X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_quat_w(
    env: ManagerBasedEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    """环境框架中的资产根导向 (w，x，y，z)。

    If :attr:`make_quat_unique`是True，然后返回的四元数通过确保
    四元数具有非负的真实组成部分。
    因为``q``和``-q``都代表着相同的方向。
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    # make the quaternion real-part positive if configured
    return math_utils.quat_unique(quat) if make_quat_unique else quat


@generic_io_descriptor(
    units="m/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    """在环境框架中的资产根线性速度。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


@generic_io_descriptor(
    units="rad/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the environment frame."""
    """在环境框架中的资产根角速度。"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w


"""
Body state
"""
"""身体状态
"""


@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def body_pose_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The flattened body poses of the asset w.r.t the env.scene.origin.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The poses of bodies in articulation [num_env, 7 * num_bodies]. Pose order is [x,y,z,qw,qx,qy,qz].
        Output is stacked horizontally per body.
    """
    """的体格设置的资产w.r.tXenv.scene.origin。

    Note: 只有在:attr:`asset_cfg.body_ids`中配置的尸体才能恢复姿势。

    参数：
        env: 环境。
        asset_cfg: 在SceneEntity这种观测与相关。

    返回：
        关节体的姿势 [num_env， 7 * num_bodies]。
        位置顺序是 [x，y，z，qw，qx，qy，qz]。
        每个机体的输出水平堆叠。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    pose = asset.data.body_pose_w[:, asset_cfg.body_ids, :7]
    if isinstance(asset_cfg.body_ids, (slice, int)):
        pose = pose.clone()  # if slice or int, make a copy to avoid modifying original data
    pose[..., :3] = pose[..., :3] - env.scene.env_origins.unsqueeze(1)
    return pose.reshape(env.num_envs, -1)


@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def body_projected_gravity_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The direction of gravity projected on to bodies of an Articulation.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The Articulation associated with this observation.

    Returns:
        The unit vector direction of gravity projected onto body_name's frame. Gravity projection vector order is
        [x,y,z]. Output is stacked horizontally per body.
    """
    """引力向关节体投射。

    Note: 只有在:attr:`asset_cfg.body_ids`中配置的尸体才能恢复姿势。

    参数：
        env: 环境。
        asset_cfg: 关节与这一观测有关。

    返回：
        引力的单位向量方向投射到body_name的框架上。
        引力投射向量顺序是 [x，y，z]。
        每个机体的输出水平堆叠。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    gravity_dir = asset.data.GRAVITY_VEC_W.unsqueeze(1)
    return math_utils.quat_apply_inverse(body_quat, gravity_dir).view(env.num_envs, -1)


"""
Joint state.
"""
"""联合州。
"""


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad"
)
def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    """资产的共同位置。

    Note: 只有在:attr:`asset_cfg.joint_ids`中配置的关节才能返回位置。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_pos_offsets],
    units="rad",
)
def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    """资产w.r.t的共同位置
    默认的关节位置。

    Note: 只有在:attr:`asset_cfg.joint_ids`中配置的关节才能返回位置。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


@generic_io_descriptor(observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape])
def joint_pos_limit_normalized(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    """资产的共同地位与资产的共同限制正常化。

    Note: 只有在:attr:`asset_cfg.joint_ids`中配置的关节才能恢复正常位置。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1],
    )


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad/s"
)
def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    """资产的关节速度。

    Note: 只有在:attr:`asset_cfg.joint_ids`中配置的关节才能恢复速度。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_vel_offsets],
    units="rad/s",
)
def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    """资产 w.r.t的关节速度。
    默认的关节速度。

    Note: 只有在:attr:`asset_cfg.joint_ids`中配置的关节才能恢复速度。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="N.m"
)
def joint_effort(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint applied effort of the robot.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their effort returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The joint effort (N or N-m) for joint_names in asset_cfg, shape is [num_env,num_joints].
    """
    """机器人的共同应用。

    NOTE: 只有在:attr:`asset_cfg.joint_ids`中配置的关节才能回复他们的努力。

    参数：
        env: 环境。
        asset_cfg: 在SceneEntity这种观测与相关。

    返回：
        在 asset_cfg，形状中的joint_names的联合努力 (N或N-m) 是 [num_env，num_joints]。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


"""
Sensors.
"""
"""传感器。
"""


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    """从给定的传感器w.r.t的高度扫描。
    传感器的框架。

    在返回的值中，所提供的抵消 (默认到0.5) 将减去。
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset


def body_incoming_wrench(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """
    """在仿真世界框架中的关节物体上的空间钥匙。

    这是一个6D关 (力和扭矩) 通过接入的联合力对车身链接进行应用。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # obtain the link incoming forces in world frame
    body_incoming_joint_wrench_b = asset.data.body_incoming_joint_wrench_b[:, asset_cfg.body_ids]
    return body_incoming_joint_wrench_b.view(env.num_envs, -1)


def imu_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor orientation in the simulation world frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        Orientation in the world frame in (w, x, y, z) quaternion form. Shape is (num_envs, 4).
    """
    """在仿真世界框架中的Imu传感器导向。

    参数：
        env: 环境。
        asset_cfg: SceneEntity与IMU传感器相关。
                   在 SceneEntityCfg (("imu") 中默认存在。

    返回：
        在世界框架中的导向 (w，x，y，z) 四角形。
        形状是 (num_envs， 4)。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the orientation quaternion
    return asset.data.quat_w


def imu_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor orientation w.r.t the env.scene.origin.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an Imu sensor.

    Returns:
        Gravity projected on imu_frame, shape of torch.tensor is (num_env,3).
    """
    """传感器导向w.r.t env.scene.origin。

    参数：
        env: 环境。
        asset_cfg: SceneEntity与Imu传感器相关。

    返回：
        在imu_frame上投射的重力，torch.tensor的形状是 (num_env，3)。
    """

    asset: Imu = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def imu_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor angular velocity w.r.t. environment origin expressed in the sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The angular velocity (rad/s) in the sensor frame. Shape is (num_envs, 3).
    """
    """图像传感器角速度w.r.t。
    在传感器框架中表达的环境来源。

    参数：
        env: 环境。
        asset_cfg: SceneEntity与IMU传感器相关。
                   在 SceneEntityCfg (("imu") 中默认存在。

    返回：
        传感器框架中的角速度 (rad/s)。
        形状是 (num_envs， 3)。
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the angular velocity
    return asset.data.ang_vel_b


def imu_lin_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor linear acceleration w.r.t. the environment origin expressed in sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The linear acceleration (m/s^2) in the sensor frame. Shape is (num_envs, 3).
    """
    """电感器线性加速w.r.t。
    在传感器框架中表达的环境来源。

    参数：
        env: 环境。
        asset_cfg: SceneEntity与IMU传感器相关。
                   在 SceneEntityCfg (("imu") 中默认存在。

    返回：
        传感器框架中的线性加速 (m/s^2)。
        形状是 (num_envs， 3)。
    """
    asset: Imu = env.scene[asset_cfg.name]
    return asset.data.lin_acc_b


def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    """摄像头传感器的特定数据类型的图像。

    如果标志:attr:`normalize`是True，则根据图像的
    data-types:

    - "rgb":将图像缩小到 (0， 1) 并以当前图像批量的平均值减去。
    - "深度"或"distance_to_camera"或"distance_to_plane":以零取代无限值。

    参数：
        env: 摄像机的环境。
        sensor_cfg: 需要的传感器。
                    默认的SceneEntityCfg("tiled_camera")。
        data_type: 从所需的相机中抽取的数据类型。
                   默认的"rgb"。
        convert_perspective_to_orthogonal: 是否直角化视角深度图像。
                                           只有数据类型是"distance_to_camera"时才使用。
                                           默认为 False。
        normalize: 是否将图像正常化。
                   这取决于选择的数据类型。
                   默认为 True。

    返回：
        在最后一步制作的图像
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth/normals image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0
        elif "normals" in data_type:
            images = (images + 1.0) * 0.5

    return images.clone()


class image_features(ManagerTermBase):
    """Extracted image features from a pre-trained frozen encoder.

    This term uses models from the model zoo in PyTorch and extracts features from the images.

    It calls the :func:`image` function to get the images and then processes them using the model zoo.

    A user can provide their own model zoo configuration to use different models for feature extraction.
    The model zoo configuration should be a dictionary that maps different model names to a dictionary
    that defines the model, preprocess and inference functions. The dictionary should have the following
    entries:

    - "model": A callable that returns the model when invoked without arguments.
    - "reset": A callable that resets the model. This is useful when the model has a state that needs to be reset.
    - "inference": A callable that, when given the model and the images, returns the extracted features.

    If the model zoo configuration is not provided, the default model zoo configurations are used. The default
    model zoo configurations include the models from Theia :cite:`shang2024theia` and ResNet :cite:`he2016deep`.
    These models are loaded from `Hugging-Face transformers <https://huggingface.co/docs/transformers/index>`_ and
    `PyTorch torchvision <https://pytorch.org/vision/stable/models.html>`_ respectively.

    Args:
        sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The sensor data type. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        model_zoo_cfg: A user-defined dictionary that maps different model names to their respective configurations.
            Defaults to None. If None, the default model zoo configurations are used.
        model_name: The name of the model to use for inference. Defaults to "resnet18".
        model_device: The device to store and infer the model on. This is useful when offloading the computation
            from the environment simulation device. Defaults to the environment device.
        inference_kwargs: Additional keyword arguments to pass to the inference function. Defaults to None,
            which means no additional arguments are passed.

    Returns:
        The extracted features tensor. Shape is (num_envs, feature_dim).

    Raises:
        ValueError: When the model name is not found in the provided model zoo configuration.
        ValueError: When the model name is not found in the default model zoo configuration.
    """
    """从预训练的冷编码器中提取了图像的特征。

    这个项使用PyTorch中的模型动物园模型，并从图像中提取特征。

    它调用:func:`image`函数来获取图像，

    用户可以提供自己的模型动物园配置，以使用不同的模型进行特征提取。
    模型动物园配置应是一个字典，将不同的模型名称映射到定义模型，预处理和推断函数的字典。
    字典应该包含以下内容:
    entries:

    - "模型":无论证地调用时返回模型的可调用器。
    - "重置":一个调用式，重置模型。 这在模型需要重置的状态时是有用的。
    - "推理":一个调用器，在给模型和图像时，返回取出的特征。

    如果没有提供模型动物园配置，则使用默认模型动物园配置。
    默认模型动物园配置包括来自Theia的模型:cite:`shang2024theia`和ResNet:cite:`he2016deep`。
    这些模型由`Hugging-Face transformers <https://huggingface.co/docs/transformers/index>`并且`PyTorch
    torchvision <https://pytorch.org/vision/stable/models.html>`它们是

    参数：
        sensor_cfg: 传感器配置到投票。
                    默认的SceneEntityCfg("tiled_camera")。
        data_type: 传感器数据类型。
                   默认的"rgb"。
        convert_perspective_to_orthogonal: 是否直角化视角深度图像。
                                           只有数据类型是"distance_to_camera"时才使用。
                                           默认为 False。
        model_zoo_cfg: 用户定义的字典，将不同的模型名称映射到各自的配置。
                       默认为 None。
                       如果 None，则使用默认模型动物园配置。
        model_name: 用于推断的模型名称。
                    在"resnet18"上默认设置。
        model_device: 存储和推断模型的设备。
                      这在卸载计算时有用。
            from the environment simulation device. Defaults to the environment device.
        inference_kwargs: 其他关键词参数将转移到推理函数。
                          在 None 中，默认情况下，这意味着没有其他参数被通过。

    返回：
        提取的特征是子。
        形状是 (num_envs，feature_dim)。

    异常：
        ValueError: 当模型名称不在提供的模型动物园配置中时。
        ValueError: 当模型名称不在默认模型动物园配置中时。
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # extract parameters from the configuration
        self.model_zoo_cfg: dict = cfg.params.get("model_zoo_cfg")  # type: ignore
        self.model_name: str = cfg.params.get("model_name", "resnet18")  # type: ignore
        self.model_device: str = cfg.params.get("model_device", env.device)  # type: ignore

        # List of Theia models - These are configured through `_prepare_theia_transformer_model` function
        default_theia_models = [
            "theia-tiny-patch16-224-cddsv",
            "theia-tiny-patch16-224-cdiv",
            "theia-small-patch16-224-cdiv",
            "theia-base-patch16-224-cdiv",
            "theia-small-patch16-224-cddsv",
            "theia-base-patch16-224-cddsv",
        ]
        # List of ResNet models - These are configured through `_prepare_resnet_model` function
        default_resnet_models = ["resnet18", "resnet34", "resnet50", "resnet101"]

        # Check if model name is specified in the model zoo configuration
        if self.model_zoo_cfg is not None and self.model_name not in self.model_zoo_cfg:
            raise ValueError(
                f"Model name '{self.model_name}' not found in the provided model zoo configuration."
                " Please add the model to the model zoo configuration or use a different model name."
                f" Available models in the provided list: {list(self.model_zoo_cfg.keys())}."
                "\nHint: If you want to use a default model, consider using one of the following models:"
                f" {default_theia_models + default_resnet_models}. In this case, you can remove the"
                " 'model_zoo_cfg' parameter from the observation term configuration."
            )
        if self.model_zoo_cfg is None:
            if self.model_name in default_theia_models:
                model_config = self._prepare_theia_transformer_model(self.model_name, self.model_device)
            elif self.model_name in default_resnet_models:
                model_config = self._prepare_resnet_model(self.model_name, self.model_device)
            else:
                raise ValueError(
                    f"Model name '{self.model_name}' not found in the default model zoo configuration."
                    f" Available models: {default_theia_models + default_resnet_models}."
                )
        else:
            model_config = self.model_zoo_cfg[self.model_name]

        # Retrieve the model, preprocess and inference functions
        self._model = model_config["model"]()
        self._reset_fn = model_config.get("reset")
        self._inference_fn = model_config["inference"]

    def reset(self, env_ids: torch.Tensor | None = None):
        # reset the model if a reset function is provided
        # this might be useful when the model has a state that needs to be reset
        # for example: video transformers
        if self._reset_fn is not None:
            self._reset_fn(self._model, env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "resnet18",
        model_device: str | None = None,
        inference_kwargs: dict | None = None,
    ) -> torch.Tensor:
        # obtain the images from the sensor
        image_data = image(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=False,  # we pre-process based on model
        )
        # store the device of the image
        image_device = image_data.device
        # forward the images through the model
        features = self._inference_fn(self._model, image_data, **(inference_kwargs or {}))

        # move the features back to the image device
        return features.detach().to(image_device)

    """
    Helper functions.
    """
    """辅助函数。
    """

    def _prepare_theia_transformer_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the Theia transformer model for inference.

        Args:
            model_name: The name of the Theia transformer model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        """准备Theia变压器模型进行推断。

        参数：
            model_name: 修亚变压器模型的名称。
            model_device: 存储和推断模型的设备。

        返回：
            包含模型和推断函数的字典。
        """
        from transformers import AutoModel

        def _load_model() -> torch.nn.Module:
            """Load the Theia transformer model."""
            """装载了Theia变压器模型。"""
            model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the Theia transformer model.

            Args:
                model: The Theia transformer model.
                images: The preprocessed image tensor. Shape is (num_envs, height, width, channel).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            """引入了Theia变压器模型。

            参数：
                model: 这种变压器模型。
                images: 预处理的图像子。
                        形状是 (num_envs，高度，宽度，道)。

            返回：
                提取的特征是子。
                形状是 (num_envs，feature_dim)。
            """
            # Move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # Normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # Taken from Transformers; inference converted to be GPU only
            features = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
            return features.last_hidden_state[:, 1:]

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}

    def _prepare_resnet_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the ResNet model for inference.

        Args:
            model_name: The name of the ResNet model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        """准备ResNet模型进行推断。

        参数：
            model_name: 准备ResNet模型的名称。
            model_device: 存储和推断模型的设备。

        返回：
            包含模型和推断函数的字典。
        """
        from torchvision import models

        def _load_model() -> torch.nn.Module:
            """Load the ResNet model."""
            """装载ResNet模型。"""
            # map the model name to the weights
            resnet_weights = {
                "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
                "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
                "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
                "resnet101": "ResNet101_Weights.IMAGENET1K_V1",
            }

            # load the model
            model = getattr(models, model_name)(weights=resnet_weights[model_name]).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the ResNet model.

            Args:
                model: The ResNet model.
                images: The preprocessed image tensor. Shape is (num_envs, channel, height, width).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            """引进ResNet模型。

            参数：
                model: 这就是ResNet模型。
                images: 预处理的图像子。
                        形状是 (num_envs，道，高度，宽度)。

            返回：
                提取的特征是子。
                形状是 (num_envs，feature_dim)。
            """
            # move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # forward the image through the model
            return model(image_proc)

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}


"""
Actions.
"""
"""动作。
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    """最后一次向环境输入动作。

    要求采取动作的动作项名称。
    如果 None，则返回整个动作张量。
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions


"""
Commands.
"""
"""命令。
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def generated_commands(env: ManagerBasedRLEnv, command_name: str | None = None) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    """在指令管理器中从指令项中生成的命令。"""
    return env.command_manager.get_command(command_name)


"""
Time.
"""
"""时间。
"""


def current_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The current time in the episode (in seconds)."""
    """回合中的当前时间 (秒钟)。"""
    return env.episode_length_buf.unsqueeze(1) * env.step_dt


def remaining_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The maximum time remaining in the episode (in seconds)."""
    """回合剩余的最大时间 (秒钟)。"""
    return env.max_episode_length_s - env.episode_length_buf.unsqueeze(1) * env.step_dt
