# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from gymnasium import spaces

from isaaclab.utils import configclass

from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnvCfg

###
# Observation space as Box
###


@configclass
class BoxBoxEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Box`` with shape (4,))

        ===  ===
        Idx  Observation
        ===  ===
        0    Pole DOF position
        1    Pole DOF velocity
        2    Cart DOF position
        3    Cart DOF velocity
        ===  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """
    """* 观测空间 (``~gymnasium.spaces.Box``形状 (4，))

        === === Idx观测 === === 0 极 DOF位置 1 极 DOF速度 2 车 DOF位置 3 车 DOF速度 === ===

    * 动作空间 (``~gymnasium.spaces.Box``形状 (1，))

        === === Idx 动作 === === 0 卡车 DOF 努力规模: [-1， 1] === ===
    """

    observation_space = spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))  # or for simplicity: 4 or [4]
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class BoxDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Box`` with shape (4,))

        ===  ===
        Idx  Observation
        ===  ===
        0    Pole DOF position
        1    Pole DOF velocity
        2    Cart DOF position
        3    Cart DOF velocity
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
    """* 观测空间 (``~gymnasium.spaces.Box``形状 (4，))

        === === Idx观测 === === 0 极 DOF位置 1 极 DOF速度 2 车 DOF位置 3 车 DOF速度 === ===

    * 动作空间 (``~gymnasium.spaces.Discrete``有3个元素)

        === === N 动作 === === 0 零行车DOF 努力 1 负最大行车DOF 努力 2 积极最大行车DOF 努力 === ===
    """

    observation_space = spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))  # or for simplicity: 4 or [4]
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class BoxMultiDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Box`` with shape (4,))

        ===  ===
        Idx  Observation
        ===  ===
        0    Pole DOF position
        1    Pole DOF velocity
        2    Cart DOF position
        3    Cart DOF velocity
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
    """* 观测空间 (``~gymnasium.spaces.Box``形状 (4，))

        === === Idx观测 === === 0 极 DOF位置 1 极 DOF速度 2 车 DOF位置 3 车 DOF速度 === ===

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete``有2个分离空间)

        === === N 动作 (分别0) === 0 零行车DOF 努力 1 半个最大行车DOF 努力 2 最大行车DOF 努力 === ===

        === === N 动作 (分别1) === === 0 负面努力 (一边) 1 积极努力 (另一边) === ===
    """

    observation_space = spaces.Box(low=float("-inf"), high=float("inf"), shape=(4,))  # or for simplicity: 4 or [4]
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]


###
# Observation space as Discrete
###


@configclass
class DiscreteBoxEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Discrete`` with 16 elements)

        ===  ===
        N    Observation (Value signs: pole position, cart position, pole velocity, cart velocity)
        ===  ===
        0    - - - -
        1    - - - +
        2    - - + -
        3    - - + +
        4    - + - -
        5    - + - +
        6    - + + -
        7    - + + +
        8    + - - -
        9    + - - +
        10   + - + -
        11   + - + +
        12   + + - -
        13   + + - +
        14   + + + -
        15   + + + +
        ===  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """
    """* 观测空间 (``~gymnasium.spaces.Discrete``含16个元素)

        === === N 观测 (值标志:极位置，车位，极速，车速) === === 0 - - - - - - - - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - -

    * 动作空间 (``~gymnasium.spaces.Box``形状 (1，))

        === === Idx 动作 === === 0 卡车 DOF 努力规模: [-1， 1] === ===
    """

    observation_space = spaces.Discrete(16)  # or for simplicity: {16}
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class DiscreteDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Discrete`` with 16 elements)

        ===  ===
        N    Observation (Value signs: pole position, cart position, pole velocity, cart velocity)
        ===  ===
        0    - - - -
        1    - - - +
        2    - - + -
        3    - - + +
        4    - + - -
        5    - + - +
        6    - + + -
        7    - + + +
        8    + - - -
        9    + - - +
        10   + - + -
        11   + - + +
        12   + + - -
        13   + + - +
        14   + + + -
        15   + + + +
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
    """* 观测空间 (``~gymnasium.spaces.Discrete``含16个元素)

        === === N 观测 (值标志:极位置，车位，极速，车速) === === 0 - - - - - - - - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - -

    * 动作空间 (``~gymnasium.spaces.Discrete``有3个元素)

        === === N 动作 === === 0 零行车DOF 努力 1 负最大行车DOF 努力 2 积极最大行车DOF 努力 === ===
    """

    observation_space = spaces.Discrete(16)  # or for simplicity: {16}
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class DiscreteMultiDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Discrete`` with 16 elements)

        ===  ===
        N    Observation (Value signs: pole position, cart position, pole velocity, cart velocity)
        ===  ===
        0    - - - -
        1    - - - +
        2    - - + -
        3    - - + +
        4    - + - -
        5    - + - +
        6    - + + -
        7    - + + +
        8    + - - -
        9    + - - +
        10   + - + -
        11   + - + +
        12   + + - -
        13   + + - +
        14   + + + -
        15   + + + +
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
    """* 观测空间 (``~gymnasium.spaces.Discrete``含16个元素)

        === === N 观测 (值标志:极位置，车位，极速，车速) === === 0 - - - - - - - - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        - - - - - - - - -

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete``有2个分离空间)

        === === N 动作 (分别0) === 0 零行车DOF 努力 1 半个最大行车DOF 努力 2 最大行车DOF 努力 === ===

        === === N 动作 (分别1) === === 0 负面努力 (一边) 1 积极努力 (另一边) === ===
    """

    observation_space = spaces.Discrete(16)  # or for simplicity: {16}
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]


###
# Observation space as MultiDiscrete
###


@configclass
class MultiDiscreteBoxEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.MultiDiscrete`` with 4 discrete spaces)

        ===  ===
        N    Observation (Discrete 0)
        ===  ===
        0    Negative pole position (-)
        1    Zero or positive pole position (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 1)
        ===  ===
        0    Negative cart position (-)
        1    Zero or positive cart position (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 2)
        ===  ===
        0    Negative pole velocity (-)
        1    Zero or positive pole velocity (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 3)
        ===  ===
        0    Negative cart velocity (-)
        1    Zero or positive cart velocity (+)
        ===  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """
    """* 观测空间 (``~gymnasium.spaces.MultiDiscrete``有4个离散空间)

        === === N 观测 (分别0) === === 0 负极位置 (-) 1 零或正极位置 (+) === ===

        === === N 观测 (分别1) === === 0 负货车位置 (-) 1 零或正货车位置 (+) === ===

        === === N观测 (分别2) === === 0 负极速 (-) 1 零或正极速 (+) === ===

        === === N观测 (分别3) === === 0 负车速 (-) 1 零或正车速 (+) === ===

    * 动作空间 (``~gymnasium.spaces.Box``形状 (1，))

        === === Idx 动作 === === 0 卡车 DOF 努力规模: [-1， 1] === ===
    """

    observation_space = spaces.MultiDiscrete([2, 2, 2, 2])  # or for simplicity: [{2}, {2}, {2}, {2}]
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class MultiDiscreteDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.MultiDiscrete`` with 4 discrete spaces)

        ===  ===
        N    Observation (Discrete 0)
        ===  ===
        0    Negative pole position (-)
        1    Zero or positive pole position (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 1)
        ===  ===
        0    Negative cart position (-)
        1    Zero or positive cart position (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 2)
        ===  ===
        0    Negative pole velocity (-)
        1    Zero or positive pole velocity (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 3)
        ===  ===
        0    Negative cart velocity (-)
        1    Zero or positive cart velocity (+)
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
    """* 观测空间 (``~gymnasium.spaces.MultiDiscrete``有4个离散空间)

        === === N 观测 (分别0) === === 0 负极位置 (-) 1 零或正极位置 (+) === ===

        === === N 观测 (分别1) === === 0 负货车位置 (-) 1 零或正货车位置 (+) === ===

        === === N观测 (分别2) === === 0 负极速 (-) 1 零或正极速 (+) === ===

        === === N观测 (分别3) === === 0 负车速 (-) 1 零或正车速 (+) === ===

    * 动作空间 (``~gymnasium.spaces.Discrete``有3个元素)

        === === N 动作 === === 0 零行车DOF 努力 1 负最大行车DOF 努力 2 积极最大行车DOF 努力 === ===
    """

    observation_space = spaces.MultiDiscrete([2, 2, 2, 2])  # or for simplicity: [{2}, {2}, {2}, {2}]
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class MultiDiscreteMultiDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.MultiDiscrete`` with 4 discrete spaces)

        ===  ===
        N    Observation (Discrete 0)
        ===  ===
        0    Negative pole position (-)
        1    Zero or positive pole position (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 1)
        ===  ===
        0    Negative cart position (-)
        1    Zero or positive cart position (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 2)
        ===  ===
        0    Negative pole velocity (-)
        1    Zero or positive pole velocity (+)
        ===  ===

        ===  ===
        N    Observation (Discrete 3)
        ===  ===
        0    Negative cart velocity (-)
        1    Zero or positive cart velocity (+)
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
    """* 观测空间 (``~gymnasium.spaces.MultiDiscrete``有4个离散空间)

        === === N 观测 (分别0) === === 0 负极位置 (-) 1 零或正极位置 (+) === ===

        === === N 观测 (分别1) === === 0 负货车位置 (-) 1 零或正货车位置 (+) === ===

        === === N观测 (分别2) === === 0 负极速 (-) 1 零或正极速 (+) === ===

        === === N观测 (分别3) === === 0 负车速 (-) 1 零或正车速 (+) === ===

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete``有2个分离空间)

        === === N 动作 (分别0) === 0 零行车DOF 努力 1 半个最大行车DOF 努力 2 最大行车DOF 努力 === ===

        === === N 动作 (分别1) === === 0 负面努力 (一边) 1 积极努力 (另一边) === ===
    """

    observation_space = spaces.MultiDiscrete([2, 2, 2, 2])  # or for simplicity: [{2}, {2}, {2}, {2}]
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]


###
# Observation space as Dict
###


@configclass
class DictBoxEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        joint-positions   DOF positions
        joint-velocities  DOF velocities
        ================  ===

    * Action space (``~gymnasium.spaces.Box`` with shape (1,))

        ===  ===
        Idx  Action
        ===  ===
        0    Cart DOF effort scale: [-1, 1]
        ===  ===
    """
    """* 观测空间 (``~gymnasium.spaces.Dict``有2个组成空间)

        关键观测 关键观测DOF位置关节速度DOF快速的速度

    * 动作空间 (``~gymnasium.spaces.Box``形状 (1，))

        === === Idx 动作 === === 0 卡车 DOF 努力规模: [-1， 1] === ===
    """

    observation_space = spaces.Dict(
        {
            "joint-positions": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        }
    )  # or for simplicity: {"joint-positions": 2, "joint-velocities": 2}
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class DictDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        joint-positions   DOF positions
        joint-velocities  DOF velocities
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

        关键观测 关键观测DOF位置关节速度DOF快速的速度

    * 动作空间 (``~gymnasium.spaces.Discrete``有3个元素)

        === === N 动作 === === 0 零行车DOF 努力 1 负最大行车DOF 努力 2 积极最大行车DOF 努力 === ===
    """

    observation_space = spaces.Dict(
        {
            "joint-positions": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        }
    )  # or for simplicity: {"joint-positions": 2, "joint-velocities": 2}
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class DictMultiDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Dict`` with 2 constituent spaces)

        ================  ===
        Key               Observation
        ================  ===
        joint-positions   DOF positions
        joint-velocities  DOF velocities
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

        关键观测 关键观测DOF位置关节速度DOF快速的速度

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete``有2个分离空间)

        === === N 动作 (分别0) === 0 零行车DOF 努力 1 半个最大行车DOF 努力 2 最大行车DOF 努力 === ===

        === === N 动作 (分别1) === === 0 负面努力 (一边) 1 积极努力 (另一边) === ===
    """

    observation_space = spaces.Dict(
        {
            "joint-positions": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            "joint-velocities": spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        }
    )  # or for simplicity: {"joint-positions": 2, "joint-velocities": 2}
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]


###
# Observation space as Tuple
###


@configclass
class TupleBoxEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Tuple`` with 2 constituent spaces)

        ===  ===
        Idx  Observation
        ===  ===
        0    DOF positions
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

        现在，我们要做什么?DOF位置1DOF速度

    * 动作空间 (``~gymnasium.spaces.Box``形状 (1，))

        === === Idx 动作 === === 0 卡车 DOF 努力规模: [-1， 1] === ===
    """

    observation_space = spaces.Tuple(
        (
            spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        )
    )  # or for simplicity: (2, 2)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))  # or for simplicity: 1 or [1]


@configclass
class TupleDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Tuple`` with 2 constituent spaces)

        ===  ===
        Idx  Observation
        ===  ===
        0    DOF positions
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

        现在，我们要做什么?DOF位置1DOF速度

    * 动作空间 (``~gymnasium.spaces.Discrete``有3个元素)

        === === N 动作 === === 0 零行车DOF 努力 1 负最大行车DOF 努力 2 积极最大行车DOF 努力 === ===
    """

    observation_space = spaces.Tuple(
        (
            spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        )
    )  # or for simplicity: (2, 2)
    action_space = spaces.Discrete(3)  # or for simplicity: {3}


@configclass
class TupleMultiDiscreteEnvCfg(CartpoleEnvCfg):
    """
    * Observation space (``~gymnasium.spaces.Tuple`` with 2 constituent spaces)

        ===  ===
        Idx  Observation
        ===  ===
        0    DOF positions
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

        现在，我们要做什么?DOF位置1DOF速度

    * 动作空间 (``~gymnasium.spaces.MultiDiscrete``有2个分离空间)

        === === N 动作 (分别0) === 0 零行车DOF 努力 1 半个最大行车DOF 努力 2 最大行车DOF 努力 === ===

        === === N 动作 (分别1) === === 0 负面努力 (一边) 1 积极努力 (另一边) === ===
    """

    observation_space = spaces.Tuple(
        (
            spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
            spaces.Box(low=float("-inf"), high=float("inf"), shape=(2,)),
        )
    )  # or for simplicity: (2, 2)
    action_space = spaces.MultiDiscrete([3, 2])  # or for simplicity: [{3}, {2}]
