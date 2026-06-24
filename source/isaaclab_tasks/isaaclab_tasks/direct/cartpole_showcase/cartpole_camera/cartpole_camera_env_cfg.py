# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.cartpole.cartpole_camera_env import CartpoleRGBCameraEnvCfg as CartpoleCameraEnvCfg


def get_tiled_camera_cfg(data_type: str, width: int = 100, height: int = 100) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=[data_type],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=width,
        height=height,
    )


###
# Observation space as Box
###


@configclass
class BoxBoxEnvCfg(CartpoleCameraEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Box`` with shape (height, width, 3))

        ===  ===
        Idx  Observation
        ===  ===
        -    RGB image
        ===  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """
    """* 观测空间 (``~gymnasium.spaces.Box``与形状 (高度，宽度，3)

        现在，我们要做什么?
        -    RGB图像
        === ===

    * 动作空间 (``~gymnasium.spaces.Box``形状 (1，))

        === === Idx 动作 === === 0 卡车 DOF 努力规模: [-1， 1] === ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Box(
        low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)
    )  # or for simplicity: [height, width, 3]
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class BoxDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Box`` with shape (height, width, 3))

        ===  ===
        Idx  Observation
        ===  ===
        -    RGB image
        ===  ===

    * Action space (``~gymnasium.spaces.Discrete`` with 3 elements)

        ===  ===
        N    Action
        ===  ===
        0    Zero cart DOF effort
        1    Negative maximum cart DOF effort
        2    Positive maximum cart DOF effort
        ===  ===
    """
    """* 观测空间 (``~gymnasium.spaces.Box``与形状 (高度，宽度，3)

        现在，我们要做什么?
        -    RGB图像
        === ===

    * 动作空间 (``~gymnasium.spaces.Discrete``有3个元素)

        === === N 动作 === === 0 零行车DOF 努力 1 负最大行车DOF 努力 2 积极最大行车DOF 努力 === ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Box(
        low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)
    )  # or for simplicity: [height, width, 3]
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class BoxMultiDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Box`` with shape (height, width, 3))

        ===  ===
        Idx  Observation
        ===  ===
        -    RGB image
        ===  ===

    * Action space (``~gymnasium.spaces.MultiDiscrete`` with 2 discrete spaces)

        ===  ===
        N    Action (Discrete 0)
        ===  ===
        0    Zero cart DOF effort
        1    Half of maximum cart DOF effort
        2    Maximum cart DOF effort
        ===  ===

        ===  ===
        N    Action (Discrete 1)
        ===  ===
        0    Negative effort (one side)
        1    Positive effort (other side)
        ===  ===
    """
    """* 观测空间 (``~gymnasium.spaces.Box``与形状 (高度，宽度，3)

        现在，我们要做什么?
        -    RGB图像
        === ===

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete``有2个分离空间)

        === === N 动作 (分别0) === 0 零行车DOF 努力 1 半个最大行车DOF 努力 2 最大行车DOF 努力 === ===

        === === N 动作 (分别1) === === 0 负面努力 (一边) 1 积极努力 (另一边) === ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Box(
        low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)
    )  # or for simplicity: [height, width, 3]
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]


###
# Observation space as Dict
###


@configclass
class DictBoxEnvCfg(CartpoleCameraEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        joint-velocities  DOF velocities
        camera            RGB image
        ================  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """
    """* 观测空间 (``~gymnasium.spaces.Dict``有2个组成空间)

        关键观测 关键观测DOF速度摄像头RGB图片 ============

    * 动作空间 (``~gymnasium.spaces.Box``形状 (1，))

        === === Idx 动作 === === 0 卡车 DOF 努力规模: [-1， 1] === ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Dict(
        {
            "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            "camera": spaces.Box(
                low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)
            ),
        }
    )  # or for simplicity: {"joint-velocities": 2, "camera": [height, width, 3]}
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class DictDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        joint-velocities  DOF velocities
        camera            RGB image
        ================  ===

    * Action space (``~gymnasium.spaces.Discrete`` with 3 elements)

        ===  ===
        N    Action
        ===  ===
        0    Zero cart DOF effort
        1    Negative maximum cart DOF effort
        2    Positive maximum cart DOF effort
        ===  ===
    """
    """* 观测空间 (``~gymnasium.spaces.Dict``有2个组成空间)

        关键观测 关键观测DOF速度摄像头RGB图片 ============

    * 动作空间 (``~gymnasium.spaces.Discrete``有3个元素)

        === === N 动作 === === 0 零行车DOF 努力 1 负最大行车DOF 努力 2 积极最大行车DOF 努力 === ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Dict(
        {
            "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            "camera": spaces.Box(
                low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)
            ),
        }
    )  # or for simplicity: {"joint-velocities": 2, "camera": [height, width, 3]}
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class DictMultiDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        joint-velocities  DOF velocities
        camera            RGB image
        ================  ===

    * Action space (``~gymnasium.spaces.MultiDiscrete`` with 2 discrete spaces)

        ===  ===
        N    Action (Discrete 0)
        ===  ===
        0    Zero cart DOF effort
        1    Half of maximum cart DOF effort
        2    Maximum cart DOF effort
        ===  ===

        ===  ===
        N    Action (Discrete 1)
        ===  ===
        0    Negative effort (one side)
        1    Positive effort (other side)
        ===  ===
    """
    """* 观测空间 (``~gymnasium.spaces.Dict``有2个组成空间)

        关键观测 关键观测DOF速度摄像头RGB图片 ============

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete``有2个分离空间)

        === === N 动作 (分别0) === 0 零行车DOF 努力 1 半个最大行车DOF 努力 2 最大行车DOF 努力 === ===

        === === N 动作 (分别1) === === 0 负面努力 (一边) 1 积极努力 (另一边) === ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Dict(
        {
            "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            "camera": spaces.Box(
                low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)
            ),
        }
    )  # or for simplicity: {"joint-velocities": 2, "camera": [height, width, 3]}
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]


###
# Observation space as Tuple
###


@configclass
class TupleBoxEnvCfg(CartpoleCameraEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Tuple`` with 2 constituent spaces)

        ===  ===
        Idx  Observation
        ===  ===
        0    RGB image
        1    DOF velocities
        ===  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """
    """* 观测空间 (``~gymnasium.spaces.Tuple``有2个组成空间)

        现在，我们要做什么?RGB图 1DOF速度

    * 动作空间 (``~gymnasium.spaces.Box``形状 (1，))

        === === Idx 动作 === === 0 卡车 DOF 努力规模: [-1， 1] === ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Tuple(
        (
            spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
            spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        )
    )  # or for simplicity: ([height, width, 3], 2)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class TupleDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Tuple`` with 2 constituent spaces)

        ===  ===
        Idx  Observation
        ===  ===
        0    RGB image
        1    DOF velocities
        ===  ===

    * Action space (``~gymnasium.spaces.Discrete`` with 3 elements)

        ===  ===
        N    Action
        ===  ===
        0    Zero cart DOF effort
        1    Negative maximum cart DOF effort
        2    Positive maximum cart DOF effort
        ===  ===
    """
    """* 观测空间 (``~gymnasium.spaces.Tuple``有2个组成空间)

        现在，我们要做什么?RGB图 1DOF速度

    * 动作空间 (``~gymnasium.spaces.Discrete``有3个元素)

        === === N 动作 === === 0 零行车DOF 努力 1 负最大行车DOF 努力 2 积极最大行车DOF 努力 === ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Tuple(
        (
            spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
            spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        )
    )  # or for simplicity: ([height, width, 3], 2)
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class TupleMultiDiscreteEnvCfg(CartpoleCameraEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Tuple`` with 2 constituent spaces)

        ===  ===
        Idx  Observation
        ===  ===
        0    RGB image
        1    DOF velocities
        ===  ===

    * Action space (``~gymnasium.spaces.MultiDiscrete`` with 2 discrete spaces)

        ===  ===
        N    Action (Discrete 0)
        ===  ===
        0    Zero cart DOF effort
        1    Half of maximum cart DOF effort
        2    Maximum cart DOF effort
        ===  ===

        ===  ===
        N    Action (Discrete 1)
        ===  ===
        0    Negative effort (one side)
        1    Positive effort (other side)
        ===  ===
    """
    """* 观测空间 (``~gymnasium.spaces.Tuple``有2个组成空间)

        现在，我们要做什么?RGB图 1DOF速度

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete``有2个分离空间)

        === === N 动作 (分别0) === 0 零行车DOF 努力 1 半个最大行车DOF 努力 2 最大行车DOF 努力 === ===

        === === N 动作 (分别1) === === 0 负面努力 (一边) 1 积极努力 (另一边) === ===
    """

    # camera
    tiled_camera: TiledCameraCfg = get_tiled_camera_cfg("rgb")

    # spaces
    observation_space = spaces.Tuple(
        (
            spaces.Box(low=float("-inf"), high=float("inf"), shape=(tiled_camera.height, tiled_camera.width, 3)),
            spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        )
    )  # or for simplicity: ([height, width, 3], 2)
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]
