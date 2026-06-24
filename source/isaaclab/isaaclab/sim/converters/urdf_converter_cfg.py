# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.sim.converters.asset_converter_base_cfg import AssetConverterBaseCfg
from isaaclab.utils import configclass


@configclass
class UrdfConverterCfg(AssetConverterBaseCfg):
    """The configuration class for UrdfConverter."""
    """UrdfConverter的配置类。"""

    @configclass
    class JointDriveCfg:
        """Configuration for the joint drive."""
        """联合驱动器的配置。"""

        @configclass
        class PDGainsCfg:
            """Configuration for the PD gains of the drive."""
            """驱动器的PD增长配置。"""

            stiffness: dict[str, float] | float = MISSING
            """The stiffness of the joint drive in Nm/rad or N/rad.

            If None, the stiffness is set to the value parsed from the URDF file.
            If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type` is set to ``"velocity"``, this value determines
            the drive strength in joint velocity space.
            """
            """联合驱动的硬度在Nm/rad或N/rad。

            如果 None，硬度设置为从 URDF 文件中解析的值。
            If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type`设置为``"velocity"``，这个值确定
            在关节速度空间中的驱动力。
            """

            damping: dict[str, float] | float | None = None
            """The damping of the joint drive in Nm/(rad/s) or N/(rad/s). Defaults to None.

            If None, the damping is set to the value parsed from the URDF file or 0.0 if no value is found in the URDF.
            If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type` is set to ``"velocity"``, this attribute is set to
            0.0 and :attr:`stiffness` serves as the drive's strength in joint velocity space.
            """
            """在 Nm/(rad/s) 或 N/(rad/s) 中，关联驱动的缩。
            默认为 None。

            如果None，缩设置为从URDF文件解析的值，或者如果没有值在URDF中找到0.0。
            If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type`设置为``"velocity"``，这个属性设置为
            0.0和:attr:`stiffness`在关节速度空间中的驱动力强度。
            """

        @configclass
        class NaturalFrequencyGainsCfg:
            r"""Configuration for the natural frequency gains of the drive.

            Computes the joint drive stiffness and damping based on the desired natural frequency using the formula:

            :math:`P = m \cdot f^2`, :math:`D = 2 \cdot r \cdot f \cdot m`

            where :math:`f` is the natural frequency, :math:`r` is the damping ratio, and :math:`m` is the total
            equivalent inertia at the joint. The damping ratio is such that:

            * :math:`r = 1.0` is a critically damped system,
            * :math:`r < 1.0` is underdamped,
            * :math:`r > 1.0` is overdamped.
            """
            """对驱动器自然频率增长的配置。

            根据所需的自然频率计算关节驱动硬度和缩，使用公式:

            :math:`P = m \cdot f^2`， 数学:`D = 2 \cdot r \cdot f \cdot m`

            where :数学:`f`是自然频率，`r`是缩比，和:数学:`m`是总数
            在关节的同等惯性。
            压缩比是这样的:

            * 数学:`r = 1.0`是一个极度化的系统，
            * 算数:`r < 1.0`是低温的，
            * `r > 1.0`已经过度蒸发。
            """

            natural_frequency: dict[str, float] | float = MISSING
            """The natural frequency of the joint drive.

            If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type` is set to ``"velocity"``, this value determines the
            drive's natural frequency in joint velocity space.
            """
            """联合驱动的自然频率。

            If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type`设置为``"velocity"``，这个值确定了
            在关节速度空间中的驱动器的自然频率。
            """

            damping_ratio: dict[str, float] | float = 0.005
            """The damping ratio of the joint drive. Defaults to 0.005.

            If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type` is set to ``"velocity"``, this value is ignored and
            only :attr:`natural_frequency` is used.
            """
            """关节驱动的缩比。
            默认为0.005。

            If :attr:`~UrdfConverterCfg.JointDriveCfg.target_type`设置为``"velocity"``，这个值被忽略，
            only :使用 attr:`natural_frequency`。
            """

        drive_type: dict[str, Literal["acceleration", "force"]] | Literal["acceleration", "force"] = "force"
        """The drive type used for the joint. Defaults to ``"force"``.

        * ``"acceleration"``: The joint drive normalizes the inertia before applying the joint effort so it's invariant
          to inertia and mass changes (equivalent to ideal damped oscillator).
        * ``"force"``: Applies effort through forces, so is subject to variations on the body inertia.
        """
        """电动驱动器的类型。
        在``"force"``上默认。

        * ``"acceleration"``: 联合驱动在应用联合力之前将惯性正常化，所以它对惯性和质量变化不变 (相当于理想的缩振动器)。
        * ``"force"``:通过力量施加努力，因此受体惯性变化的影响。
        """

        target_type: dict[str, Literal["none", "position", "velocity"]] | Literal["none", "position", "velocity"] = (
            "position"
        )
        """The drive target type used for the joint. Defaults to ``"position"``.

        If the target type is set to ``"none"``, the joint stiffness and damping are set to 0.0.
        """
        """驱动器目标类型用于关节。
        在``"position"``上默认。

        如果目标类型设置为``"none"``，关节硬度和缩量设置为0.0。
        """

        gains: PDGainsCfg | NaturalFrequencyGainsCfg = PDGainsCfg()
        """The drive gains configuration."""
        """驱动器获得配置。"""

    fix_base: bool = MISSING
    """Create a fix joint to the root/base link."""
    """创建根/基链接的固定关联。"""

    root_link_name: str | None = None
    """The name of the root link. Defaults to None.

    If None, the root link will be set by PhysX.
    """
    """根链的名称。
    默认为 None。

    如果 None，根链接将由 PhysX 设置。
    """

    link_density: float = 0.0
    """Default density in ``kg/m^3`` for links whose ``"inertial"`` properties are missing in the URDF.
    Defaults to 0.0.
    """
    """默认密度``kg/m^3``对于其链接``"inertial"``房产在URDF。
    默认为0.0。
    """

    merge_fixed_joints: bool = True
    """Consolidate links that are connected by fixed joints. Defaults to True."""
    """固定的关节连接的连接。
    默认为 True。
    """

    convert_mimic_joints_to_normal_joints: bool = False
    """Convert mimic joints to normal joints. Defaults to False."""
    """将模仿关节转换为正常关节。
    默认为 False。
    """

    joint_drive: JointDriveCfg | None = JointDriveCfg()
    """The joint drive settings. Defaults to :class:`JointDriveCfg`.

    The parameter can be set to ``None`` for URDFs without joints.
    """
    """联合驱动的设置。
    在:class:`JointDriveCfg`上默认。

    参数可以设置为``None``，对于没有关节的URDFs。
    """

    collision_from_visuals = False
    """Whether to create collision geometry from visual geometry. Defaults to False."""
    """是否从视觉几何学的创建碰撞几何。
    默认为 False。
    """

    collider_type: Literal["convex_hull", "convex_decomposition"] = "convex_hull"
    """The collision shape simplification. Defaults to "convex_hull".

    Supported values are:

    * ``"convex_hull"``: The collision shape is simplified to a convex hull.
    * ``"convex_decomposition"``: The collision shape is decomposed into smaller convex shapes for a closer fit.
    """
    """碰撞形状的简化。
    在"convex_hull"上默认。

    支持值为:

    * ``"convex_hull"``:碰撞形状简化为形体。
    * ``"convex_decomposition"``:碰撞形状被分解成较小的凸形形状，以便更接近。
    """

    self_collision: bool = False
    """Activate self-collisions between links of the articulation. Defaults to False."""
    """激活关节链之间的自我碰撞。
    默认为 False。
    """

    replace_cylinders_with_capsules: bool = False
    """Replace cylinder shapes with capsule shapes. Defaults to False."""
    """取代圆柱形状的囊形状。
    默认为 False。
    """
