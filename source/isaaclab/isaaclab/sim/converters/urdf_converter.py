# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from packaging.version import Version

import omni.kit.app
import omni.kit.commands

from isaaclab.utils.version import get_isaac_sim_version

from .asset_converter_base import AssetConverterBase
from .urdf_converter_cfg import UrdfConverterCfg

if TYPE_CHECKING:
    import isaacsim.asset.importer.urdf


class UrdfConverter(AssetConverterBase):
    """Converter for a URDF description file to a USD file.

    This class wraps around the `isaacsim.asset.importer.urdf`_ extension to provide a lazy implementation
    for URDF to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    .. caution::
        The current lazy conversion implementation does not automatically trigger USD generation if
        only the mesh files used by the URDF are modified. To force generation, either set
        :obj:`AssetConverterBaseCfg.force_usd_conversion` to True or delete the output directory.

    .. note::
        From Isaac Sim 4.5 onwards, the extension name changed from ``omni.importer.urdf`` to
        ``isaacsim.asset.importer.urdf``.

    .. note::
        In Isaac Sim 5.1, the URDF importer changed the default behavior of merging fixed joints.
        Links connected through ``fixed_joint`` elements are no longer merged when their URDF link
        entries specify mass and inertia, even if ``merge-joint`` is set to True. The new behavior
        treats links with mass/inertia as full bodies rather than zero-mass reference frames.

        To maintain backwards compatibility, **this converter pins to an older version of the
        URDF importer extension** (version 2.4.31) that still merges fixed joints by default.
        This allows existing URDFs to work as expected without modification.

    .. _isaacsim.asset.importer.urdf: https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html
    """
    """转换URDF描述文件为USD文件。

    这个类包围`isaacsim.asset.importer.urdf`_扩展以提供惰的实现
    for URDF to USD conversion. It stores the output USD file in an instanceable format since that is
    在所有学习相关应用中通常使用。

    .. 谨慎::
        如果只修改URDF所使用的网格文件，目前的惰转换实现不会自动触发USD生成。
        要强制生成，要么设置:obj:`AssetConverterBaseCfg.force_usd_conversion`为True，要么删除输出目录。

    .. 说明::
        从Isaac Sim4.5开始，扩展名称从``omni.importer.urdf``变为``isaacsim.asset.importer.urdf``。

    .. 说明::
        在Isaac Sim 5.1中，URDF进口商改变了固定结合的默认行为。
        通过``fixed_joint``元素连接的链接不再合并，当它们的URDF链接输入指定质量和惯性时，即使``merge-joint``设置为True。
        新的行为将与质量/惰性的链接视为整体而不是零质量的参考框架。

        为了保持向后兼容性，**这个转换器将其转换到旧版本的URDF进口扩展** (版本2.4.31) 中，该扩展器仍然默认合并固定关节。
        这使得现有的URDFs可以按照预期运行，而不需要修改。

    .. _isaacsim.asset.importer.urdf: https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html
    """

    cfg: UrdfConverterCfg
    """The configuration instance for URDF to USD conversion."""
    """为URDF转换到USD的配置实例。"""

    def __init__(self, cfg: UrdfConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for URDF to USD conversion.
        """
        """开始课程。

        参数：
            cfg: 为URDF转换到USD的配置实例。
        """
        # switch to older version of the URDF importer extension
        if get_isaac_sim_version() >= Version("5.1"):
            manager = omni.kit.app.get_app().get_extension_manager()
            if not manager.is_extension_enabled("isaacsim.asset.importer.urdf-2.4.31"):
                manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf-2.4.31", True)

        # acquire the URDF interface
        from isaacsim.asset.importer.urdf._urdf import acquire_urdf_interface

        self._urdf_interface = acquire_urdf_interface()
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """
    """具体实施方法。
    """

    def _convert_asset(self, cfg: UrdfConverterCfg):
        """Calls underlying Omniverse command to convert URDF to USD.

        Args:
            cfg: The URDF conversion configuration.
        """
        """调用底层的全宇宙命令将URDF转换为USD。

        参数：
            cfg: 转换配置的URDF。
        """

        import_config = self._get_urdf_import_config()
        # parse URDF file
        result, self._robot_model = omni.kit.commands.execute(
            "URDFParseFile", urdf_path=cfg.asset_path, import_config=import_config
        )

        if result:
            if cfg.joint_drive:
                # modify joint parameters
                self._update_joint_parameters()

            # set root link name
            if cfg.root_link_name:
                self._robot_model.root_link = cfg.root_link_name

            # convert the model to USD
            omni.kit.commands.execute(
                "URDFImportRobot",
                urdf_path=cfg.asset_path,
                urdf_robot=self._robot_model,
                import_config=import_config,
                dest_path=self.usd_path,
            )
        else:
            raise ValueError(f"Failed to parse URDF file: {cfg.asset_path}")

    """
    Helper methods.
    """
    """帮助方法。
    """

    def _get_urdf_import_config(self) -> isaacsim.asset.importer.urdf._urdf.ImportConfig:
        """Create and fill URDF ImportConfig with desired settings

        Returns:
            The constructed ``ImportConfig`` object containing the desired settings.
        """
        """创建和填写URDF ImportConfig与所需的设置

        返回：
            包含所需设置的构建``ImportConfig``对象。
        """
        # create a new import config
        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")

        # set the unit scaling factor, 1.0 means meters, 100.0 means cm
        import_config.set_distance_scale(1.0)
        # set imported robot as default prim
        import_config.set_make_default_prim(True)
        # add a physics scene to the stage on import if none exists
        import_config.set_create_physics_scene(False)

        # -- asset settings
        # default density used for links, use 0 to auto-compute
        import_config.set_density(self.cfg.link_density)
        # mesh simplification settings
        convex_decomp = self.cfg.collider_type == "convex_decomposition"
        import_config.set_convex_decomp(convex_decomp)
        # create collision geometry from visual geometry
        import_config.set_collision_from_visuals(self.cfg.collision_from_visuals)
        # consolidating links that are connected by fixed joints
        import_config.set_merge_fixed_joints(self.cfg.merge_fixed_joints)
        import_config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)
        # -- physics settings
        # create fix joint for base link
        import_config.set_fix_base(self.cfg.fix_base)
        # self collisions between links in the articulation
        import_config.set_self_collision(self.cfg.self_collision)
        # convert mimic joints to normal joints
        import_config.set_parse_mimic(self.cfg.convert_mimic_joints_to_normal_joints)
        # replace cylinder shapes with capsule shapes
        import_config.set_replace_cylinders_with_capsules(self.cfg.replace_cylinders_with_capsules)

        return import_config

    def _update_joint_parameters(self):
        """Update the joint parameters based on the configuration."""
        """根据配置更新联合参数。"""
        # set the drive type
        self._set_joints_drive_type()
        # set the drive target type
        self._set_joints_drive_target_type()
        # set the drive gains
        self._set_joint_drive_gains()

    def _set_joints_drive_type(self):
        """Set the joint drive type for all joints in the URDF model."""
        """设置URDF模型的所有关节的关节驱动类型。"""
        from isaacsim.asset.importer.urdf._urdf import UrdfJointDriveType

        drive_type_mapping = {
            "force": UrdfJointDriveType.JOINT_DRIVE_FORCE,
            "acceleration": UrdfJointDriveType.JOINT_DRIVE_ACCELERATION,
        }

        if isinstance(self.cfg.joint_drive.drive_type, str):
            for joint in self._robot_model.joints.values():
                joint.drive.set_drive_type(drive_type_mapping[self.cfg.joint_drive.drive_type])
        elif isinstance(self.cfg.joint_drive.drive_type, dict):
            for joint_name, drive_type in self.cfg.joint_drive.drive_type.items():
                # handle joint name being a regex
                matches = [s for s in self._robot_model.joints.keys() if re.search(joint_name, s)]
                if not matches:
                    raise ValueError(
                        f"The joint name {joint_name} in the drive type config was not found in the URDF file. The"
                        f" joint names in the URDF are {list(self._robot_model.joints.keys())}"
                    )
                for match in matches:
                    joint = self._robot_model.joints[match]
                    joint.drive.set_drive_type(drive_type_mapping[drive_type])

    def _set_joints_drive_target_type(self):
        """Set the joint drive target type for all joints in the URDF model."""
        """设置URDF模型的所有关节的关节驱动目标类型。"""
        from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType

        target_type_mapping = {
            "none": UrdfJointTargetType.JOINT_DRIVE_NONE,
            "position": UrdfJointTargetType.JOINT_DRIVE_POSITION,
            "velocity": UrdfJointTargetType.JOINT_DRIVE_VELOCITY,
        }

        if isinstance(self.cfg.joint_drive.target_type, str):
            for joint in self._robot_model.joints.values():
                joint.drive.set_target_type(target_type_mapping[self.cfg.joint_drive.target_type])
        elif isinstance(self.cfg.joint_drive.target_type, dict):
            for joint_name, target_type in self.cfg.joint_drive.target_type.items():
                # handle joint name being a regex
                matches = [s for s in self._robot_model.joints.keys() if re.search(joint_name, s)]
                if not matches:
                    raise ValueError(
                        f"The joint name {joint_name} in the target type config was not found in the URDF file. The"
                        f" joint names in the URDF are {list(self._robot_model.joints.keys())}"
                    )
                for match in matches:
                    joint = self._robot_model.joints[match]
                    joint.drive.set_target_type(target_type_mapping[target_type])

    def _set_joint_drive_gains(self):
        """Set the joint drive gains for all joints in the URDF model."""
        """设置URDF模型的所有关节的联合驱动增长率。"""

        # set the gains directly from stiffness and damping values
        if isinstance(self.cfg.joint_drive.gains, UrdfConverterCfg.JointDriveCfg.PDGainsCfg):
            # stiffness
            if isinstance(self.cfg.joint_drive.gains.stiffness, (float, int)):
                for joint in self._robot_model.joints.values():
                    self._set_joint_drive_stiffness(joint, self.cfg.joint_drive.gains.stiffness)
            elif isinstance(self.cfg.joint_drive.gains.stiffness, dict):
                for joint_name, stiffness in self.cfg.joint_drive.gains.stiffness.items():
                    # handle joint name being a regex
                    matches = [s for s in self._robot_model.joints.keys() if re.search(joint_name, s)]
                    if not matches:
                        raise ValueError(
                            f"The joint name {joint_name} in the drive stiffness config was not found in the URDF file."
                            f" The joint names in the URDF are {list(self._robot_model.joints.keys())}"
                        )
                    for match in matches:
                        joint = self._robot_model.joints[match]
                        self._set_joint_drive_stiffness(joint, stiffness)
            # damping
            if isinstance(self.cfg.joint_drive.gains.damping, (float, int)):
                for joint in self._robot_model.joints.values():
                    self._set_joint_drive_damping(joint, self.cfg.joint_drive.gains.damping)
            elif isinstance(self.cfg.joint_drive.gains.damping, dict):
                for joint_name, damping in self.cfg.joint_drive.gains.damping.items():
                    # handle joint name being a regex
                    matches = [s for s in self._robot_model.joints.keys() if re.search(joint_name, s)]
                    if not matches:
                        raise ValueError(
                            f"The joint name {joint_name} in the drive damping config was not found in the URDF file."
                            f" The joint names in the URDF are {list(self._robot_model.joints.keys())}"
                        )
                    for match in matches:
                        joint = self._robot_model.joints[match]
                        self._set_joint_drive_damping(joint, damping)

        # set the gains from natural frequency and damping ratio
        elif isinstance(self.cfg.joint_drive.gains, UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg):
            # damping ratio
            if isinstance(self.cfg.joint_drive.gains.damping_ratio, (float, int)):
                for joint in self._robot_model.joints.values():
                    joint.drive.damping_ratio = self.cfg.joint_drive.gains.damping_ratio
            elif isinstance(self.cfg.joint_drive.gains.damping_ratio, dict):
                for joint_name, damping_ratio in self.cfg.joint_drive.gains.damping_ratio.items():
                    # handle joint name being a regex
                    matches = [s for s in self._robot_model.joints.keys() if re.search(joint_name, s)]
                    if not matches:
                        raise ValueError(
                            f"The joint name {joint_name} in the damping ratio config was not found in the URDF file."
                            f" The joint names in the URDF are {list(self._robot_model.joints.keys())}"
                        )
                    for match in matches:
                        joint = self._robot_model.joints[match]
                        joint.drive.damping_ratio = damping_ratio

            # natural frequency (this has to be done after damping ratio is set)
            if isinstance(self.cfg.joint_drive.gains.natural_frequency, (float, int)):
                for joint in self._robot_model.joints.values():
                    joint.drive.natural_frequency = self.cfg.joint_drive.gains.natural_frequency
                    self._set_joint_drive_gains_from_natural_frequency(joint)
            elif isinstance(self.cfg.joint_drive.gains.natural_frequency, dict):
                for joint_name, natural_frequency in self.cfg.joint_drive.gains.natural_frequency.items():
                    # handle joint name being a regex
                    matches = [s for s in self._robot_model.joints.keys() if re.search(joint_name, s)]
                    if not matches:
                        raise ValueError(
                            f"The joint name {joint_name} in the natural frequency config was not found in the URDF"
                            f" file. The joint names in the URDF are {list(self._robot_model.joints.keys())}"
                        )
                    for match in matches:
                        joint = self._robot_model.joints[match]
                        joint.drive.natural_frequency = natural_frequency
                        self._set_joint_drive_gains_from_natural_frequency(joint)

    def _set_joint_drive_stiffness(self, joint, stiffness: float):
        """Set the joint drive stiffness.

        Args:
            joint: The joint from the URDF robot model.
            stiffness: The stiffness value.
        """
        """设置联合驱动硬度。

        参数：
            joint: 这是一个来自URDF机器人模型。
            stiffness: 硬度值。
        """
        from isaacsim.asset.importer.urdf._urdf import UrdfJointType

        if joint.type == UrdfJointType.JOINT_PRISMATIC:
            joint.drive.set_strength(stiffness)
        else:
            # we need to convert the stiffness from radians to degrees
            joint.drive.set_strength(math.pi / 180 * stiffness)

    def _set_joint_drive_damping(self, joint, damping: float):
        """Set the joint drive damping.

        Args:
            joint: The joint from the URDF robot model.
            damping: The damping value.
        """
        """设置关节驱动。

        参数：
            joint: 这是一个来自URDF机器人模型。
            damping: 压缩值。
        """
        from isaacsim.asset.importer.urdf._urdf import UrdfJointType

        if joint.type == UrdfJointType.JOINT_PRISMATIC:
            joint.drive.set_damping(damping)
        else:
            # we need to convert the damping from radians to degrees
            joint.drive.set_damping(math.pi / 180 * damping)

    def _set_joint_drive_gains_from_natural_frequency(self, joint):
        """Compute the joint drive gains from the natural frequency and damping ratio.

        Args:
            joint: The joint from the URDF robot model.
        """
        """根据自然频率和缩比计算联合驱动增长。

        参数：
            joint: 这是一个来自URDF机器人模型。
        """
        from isaacsim.asset.importer.urdf._urdf import UrdfJointDriveType, UrdfJointTargetType

        strength = self._urdf_interface.compute_natural_stiffness(
            self._robot_model,
            joint.name,
            joint.drive.natural_frequency,
        )
        self._set_joint_drive_stiffness(joint, strength)

        if joint.drive.target_type == UrdfJointTargetType.JOINT_DRIVE_POSITION:
            m_eq = 1.0
            if joint.drive.drive_type == UrdfJointDriveType.JOINT_DRIVE_FORCE:
                m_eq = joint.inertia
            damping = 2 * m_eq * joint.drive.natural_frequency * joint.drive.damping_ratio
            self._set_joint_drive_damping(joint, damping)
