# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

from isaaclab.utils.dict import dict_to_md5_hash
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.timer import Timer
from isaaclab.utils.warp import convert_to_warp_mesh

from .trimesh.utils import make_border
from .utils import color_meshes_by_height, find_flat_patches

if TYPE_CHECKING:
    from .sub_terrain_cfg import FlatPatchSamplingCfg, SubTerrainBaseCfg
    from .terrain_generator_cfg import TerrainGeneratorCfg

# import logger
logger = logging.getLogger(__name__)


class TerrainGenerator:
    r"""Terrain generator to handle different terrain generation functions.

    The terrains are represented as meshes. These are obtained either from height fields or by using the
    `trimesh <https://trimsh.org/trimesh.html>`__ library. The height field representation is more
    flexible, but it is less computationally and memory efficient than the trimesh representation.

    All terrain generation functions take in the argument :obj:`difficulty` which determines the complexity
    of the terrain. The difficulty is a number between 0 and 1, where 0 is the easiest and 1 is the hardest.
    In most cases, the difficulty is used for linear interpolation between different terrain parameters.
    For example, in a pyramid stairs terrain the step height is interpolated between the specified minimum
    and maximum step height.

    Each sub-terrain has a corresponding configuration class that can be used to specify the parameters
    of the terrain. The configuration classes are inherited from the :class:`SubTerrainBaseCfg` class
    which contains the common parameters for all terrains.

    If a curriculum is used, the terrains are generated based on their difficulty parameter.
    The difficulty is varied linearly over the number of rows (i.e. along x) with a small random value
    added to the difficulty to ensure that the columns with the same sub-terrain type are not exactly
    the same. The difficulty parameter for a sub-terrain at a given row is calculated as:

    .. math::

        \text{difficulty} =
            \frac{\text{row_id} + \eta}{\text{num_rows}} \times (\text{upper} - \text{lower}) + \text{lower}

    where :math:`\eta\sim\mathcal{U}(0, 1)` is a random perturbation to the difficulty, and
    :math:`(\text{lower}, \text{upper})` is the range of the difficulty parameter, specified using the
    :attr:`~TerrainGeneratorCfg.difficulty_range` parameter.

    If a curriculum is not used, the terrains are generated randomly. In this case, the difficulty parameter
    is randomly sampled from the specified range, given by the :attr:`~TerrainGeneratorCfg.difficulty_range`
    parameter:

    .. math::

        \text{difficulty} \sim \mathcal{U}(\text{lower}, \text{upper})

    If the :attr:`~TerrainGeneratorCfg.flat_patch_sampling` is specified for a sub-terrain, flat patches are
    sampled on the terrain. These can be used for spawning robots, targets, etc. The sampled patches are stored
    in the :obj:`flat_patches` dictionary. The key specifies the intention of the flat patches and the
    value is a tensor containing the flat patches for each sub-terrain.

    If the flag :attr:`~TerrainGeneratorCfg.use_cache` is set to True, the terrains are cached based on their
    sub-terrain configurations. This means that if the same sub-terrain configuration is used
    multiple times, the terrain is only generated once and then reused. This is useful when
    generating complex sub-terrains that take a long time to generate.

    .. attention::

        The terrain generation has its own seed parameter. This is set using the :attr:`TerrainGeneratorCfg.seed`
        parameter. If the seed is not set and the caching is disabled, the terrain generation may not be
        completely reproducible.

    """
    """用于处理不同的地形生成功能。

    面积的形象是网格。
    它们是从高度田中得到的，或者是通过使用`trimesh <https://trimsh.org/trimesh.html>`库
    高度字段表示更灵活，但计算和记忆效率低于trimesh表示。

    所有地形生成函数都采用:obj:`difficulty`参数，它决定了地形的复杂性。
    难度是0到1之间的数字，0是最容易的，1是最难的。
    在大多数情况下，难度用于不同地形参数之间的线性插图。
    例如，在金字塔楼梯地形上，阶梯高度是指定的最低和最高阶梯高度之间的回合。

    每个地下都有相应的配置类，可用于指定地形参数。
    配置类是继承 :class:`SubTerrainBaseCfg` 类的，其中包含所有地形的共同参数。

    如果使用课程，则根据其难度参数生成地形。
    难度在行数 (i.e.沿 x) 上有线性变化，小巧值增加了难度，以确保具有相同的地下类型的列不完全相同。
    在给定的行上，一个地下面的难度参数计算为:

    .. math::

        \text{difficulty} =
            \frac{\text{row_id} + \eta}{\text{num_rows}} \times (\text{upper} - \text{lower}) + \text{lower}

    where :数学:`\eta\sim\mathcal{U}(0， 1)`是难度的随机扰乱，
    :math:`(\text{lower}， \text{upper})`是使用以下标准的难度参数范围
    :attr:`~TerrainGeneratorCfg.difficulty_range`参数。

    如果没有使用课程，地形会随机生成。
    在这种情况下，难度参数从指定范围中随机取样，由:attr:`~TerrainGeneratorCfg.difficulty_range`给出
    parameter:

    .. math::

        \text{difficulty} \sim \mathcal{U}(\text{lower}, \text{upper})

    如果 :attr:`~TerrainGeneratorCfg.flat_patch_sampling` 指定为地下区域，则在地形上采样平坦的斑点。
    它们可以用于繁殖机器人，目标等。
    采样补丁存储在:obj:`flat_patches`字典中。
    键指定了平面贴片的目的，值为每个地下区域的平面贴片的子。

    如果标志:attr:`~TerrainGeneratorCfg.use_cache`设置为True，则根据其地下配置进行缓存。
    这意味着如果使用相同的地下配置多次，地形只会产生一次，然后再使用。
    这在产生长时间的复杂地下产生时很有用。

    .. 注意::

        土地产量有自己的种子参数。
        设置使用:attr:`TerrainGeneratorCfg.seed`参数。
        如果种子未设置，缓存被禁用，地形生成可能无法完全复制。
    """

    terrain_mesh: trimesh.Trimesh
    """A single trimesh.Trimesh object for all the generated sub-terrains."""
    """一个单个trimesh.Trimesh对所有生成的地表。"""
    terrain_meshes: list[trimesh.Trimesh]
    """List of trimesh.Trimesh objects for all the generated sub-terrains."""
    """对于所有生成的地下物体的trimesh.Trimesh物体列表。"""
    terrain_origins: np.ndarray
    """The origin of each sub-terrain. Shape is (num_rows, num_cols, 3)."""
    """每个地下区域的起源。
    形状是 (num_rows，num_cols， 3)。
    """
    flat_patches: dict[str, torch.Tensor]
    """A dictionary of sampled valid (flat) patches for each sub-terrain.

    The dictionary keys are the names of the flat patch sampling configurations. This maps to a
    tensor containing the flat patches for each sub-terrain. The shape of the tensor is
    (num_rows, num_cols, num_patches, 3).

    For instance, the key "root_spawn" maps to a tensor containing the flat patches for spawning an asset.
    Similarly, the key "target_spawn" maps to a tensor containing the flat patches for setting targets.
    """
    """每个地下区域的有效 (平面) 补丁样本字典。

    字典键是平面补丁采样配置的名称。
    这将每一个地下区域的平面斑块包含在一个度。
    子的形状是 (num_rows，num_cols，num_patches，3)。

    例如"，root_spawn"键将包含产物产生的平面补丁的子映射到一个子上。
    同样，关键"target_spawn"对包含目标设置平面补丁的子进行映射。
    """

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        """Initialize the terrain generator.

        Args:
            cfg: Configuration for the terrain generator.
            device: The device to use for the flat patches tensor.
        """
        """启动地形发电机。

        参数：
            cfg: 土地发电机的配置。
            device: 用于平面补丁度的设备。
        """
        # check inputs
        if len(cfg.sub_terrains) == 0:
            raise ValueError("No sub-terrains specified! Please add at least one sub-terrain.")
        # store inputs
        self.cfg = cfg
        self.device = device

        # set common values to all sub-terrains config
        from .height_field import HfTerrainBaseCfg  # prevent circular import

        for sub_cfg in self.cfg.sub_terrains.values():
            # size of all terrains
            sub_cfg.size = self.cfg.size
            # params for height field terrains
            if isinstance(sub_cfg, HfTerrainBaseCfg):
                sub_cfg.horizontal_scale = self.cfg.horizontal_scale
                sub_cfg.vertical_scale = self.cfg.vertical_scale
                sub_cfg.slope_threshold = self.cfg.slope_threshold

        # throw a warning if the cache is enabled but the seed is not set
        if self.cfg.use_cache and self.cfg.seed is None:
            logger.warning(
                "Cache is enabled but the seed is not set. The terrain generation will not be reproducible."
                " Please set the seed in the terrain generator configuration to make the generation reproducible."
            )

        # if the seed is not set, we assume there is a global seed set and use that.
        # this ensures that the terrain is reproducible if the seed is set at the beginning of the program.
        if self.cfg.seed is not None:
            seed = self.cfg.seed
        else:
            seed = np.random.get_state()[1][0]
        # set the seed for reproducibility
        # note: we create a new random number generator to avoid affecting the global state
        #  in the other places where random numbers are used.
        self.np_rng = np.random.default_rng(seed)

        # buffer for storing valid patches
        self.flat_patches = {}
        # create a list of all sub-terrains
        self.terrain_meshes = list()
        self.terrain_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))

        # parse configuration and add sub-terrains
        # create terrains based on curriculum or randomly
        if self.cfg.curriculum:
            with Timer("[INFO] Generating terrains based on curriculum took"):
                self._generate_curriculum_terrains()
        else:
            with Timer("[INFO] Generating terrains randomly took"):
                self._generate_random_terrains()
        # add a border around the terrains
        self._add_terrain_border()
        # combine all the sub-terrains into a single mesh
        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

        # color the terrain mesh
        if self.cfg.color_scheme == "height":
            self.terrain_mesh = color_meshes_by_height(self.terrain_mesh)
        elif self.cfg.color_scheme == "random":
            self.terrain_mesh.visual.vertex_colors = self.np_rng.choice(
                range(256), size=(len(self.terrain_mesh.vertices), 4)
            )
        elif self.cfg.color_scheme == "none":
            pass
        else:
            raise ValueError(f"Invalid color scheme: {self.cfg.color_scheme}.")

        # offset the entire terrain and origins so that it is centered
        # -- terrain mesh
        transform = np.eye(4)
        transform[:2, -1] = -self.cfg.size[0] * self.cfg.num_rows * 0.5, -self.cfg.size[1] * self.cfg.num_cols * 0.5
        self.terrain_mesh.apply_transform(transform)
        # -- terrain origins
        self.terrain_origins += transform[:3, -1]
        # -- valid patches
        terrain_origins_torch = torch.tensor(self.terrain_origins, dtype=torch.float, device=self.device).unsqueeze(2)
        for name, value in self.flat_patches.items():
            self.flat_patches[name] = value + terrain_origins_torch

    def __str__(self):
        """Return a string representation of the terrain generator."""
        """返回地形生成器的字符串表示。"""
        msg = "Terrain Generator:"
        msg += f"\n\tSeed: {self.cfg.seed}"
        msg += f"\n\tNumber of rows: {self.cfg.num_rows}"
        msg += f"\n\tNumber of columns: {self.cfg.num_cols}"
        msg += f"\n\tSub-terrain size: {self.cfg.size}"
        msg += f"\n\tSub-terrain types: {list(self.cfg.sub_terrains.keys())}"
        msg += f"\n\tCurriculum: {self.cfg.curriculum}"
        msg += f"\n\tDifficulty range: {self.cfg.difficulty_range}"
        msg += f"\n\tColor scheme: {self.cfg.color_scheme}"
        msg += f"\n\tUse cache: {self.cfg.use_cache}"
        if self.cfg.use_cache:
            msg += f"\n\tCache directory: {self.cfg.cache_dir}"

        return msg

    """
    Terrain generator functions.
    """
    """土地发电器功能。
    """

    def _generate_random_terrains(self):
        """Add terrains based on randomly sampled difficulty parameter."""
        """根据随机采样难度参数添加地形。"""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # randomly sample sub-terrains
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            # coordinate index of the sub-terrain
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            # randomly sample terrain index
            sub_index = self.np_rng.choice(len(proportions), p=proportions)
            # randomly sample difficulty parameter
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            # generate terrain
            mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
            # add to sub-terrains
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])

    def _generate_curriculum_terrains(self):
        """Add terrains based on the difficulty parameter."""
        """根据难度参数添加地形。"""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        # find the sub-terrain index for each column
        # we generate the terrains based on their proportion (not randomly sampled)
        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # curriculum-based sub-terrains
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                # vary the difficulty parameter linearly over the number of rows
                # note: based on the proportion, multiple columns can have the same sub-terrain type.
                #  Thus to increase the diversity along the rows, we add a small random value to the difficulty.
                #  This ensures that the terrains are not exactly the same. For example, if the
                #  the row index is 2 and the number of rows is 10, the nominal difficulty is 0.2.
                #  We add a small random value to the difficulty to make it between 0.2 and 0.3.
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                # generate terrain
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_indices[sub_col]])
                # add to sub-terrains
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_indices[sub_col]])

    """
    Internal helper functions.
    """
    """内部助理功能。
    """

    def _add_terrain_border(self):
        """Add a surrounding border over all the sub-terrains into the terrain meshes."""
        """在地形网格中，将所有地下面的周边边界加上。"""
        # border parameters
        border_size = (
            self.cfg.num_rows * self.cfg.size[0] + 2 * self.cfg.border_width,
            self.cfg.num_cols * self.cfg.size[1] + 2 * self.cfg.border_width,
        )
        inner_size = (self.cfg.num_rows * self.cfg.size[0], self.cfg.num_cols * self.cfg.size[1])
        border_center = (
            self.cfg.num_rows * self.cfg.size[0] / 2,
            self.cfg.num_cols * self.cfg.size[1] / 2,
            -self.cfg.border_height / 2,
        )
        # border mesh
        border_meshes = make_border(border_size, inner_size, height=abs(self.cfg.border_height), position=border_center)
        border = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1)
        border.update_faces(selector)
        # add the border to the list of meshes
        self.terrain_meshes.append(border)

    def _add_sub_terrain(
        self, mesh: trimesh.Trimesh, origin: np.ndarray, row: int, col: int, sub_terrain_cfg: SubTerrainBaseCfg
    ):
        """Add input sub-terrain to the list of sub-terrains.

        This function adds the input sub-terrain mesh to the list of sub-terrains and updates the origin
        of the sub-terrain in the list of origins. It also samples flat patches if specified.

        Args:
            mesh: The mesh of the sub-terrain.
            origin: The origin of the sub-terrain.
            row: The row index of the sub-terrain.
            col: The column index of the sub-terrain.
        """
        """增加输入地下地在地下地列表中。

        该函数将输入地下网加入地下网的列表，并更新地下网的起源。
        如果确定的，它还会采样平面补丁。

        参数：
            mesh: 在地下的网格。
            origin: 地下地区的起源。
            row: 地下区域的行列索引。
            col: 在地底的列索引。
        """
        # sample flat patches if specified
        if sub_terrain_cfg.flat_patch_sampling is not None:
            logger.info(f"Sampling flat patches for sub-terrain at (row, col):  ({row}, {col})")
            # convert the mesh to warp mesh
            wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
            # sample flat patches based on each patch configuration for that sub-terrain
            for name, patch_cfg in sub_terrain_cfg.flat_patch_sampling.items():
                patch_cfg: FlatPatchSamplingCfg
                # create the flat patches tensor (if not already created)
                if name not in self.flat_patches:
                    self.flat_patches[name] = torch.zeros(
                        (self.cfg.num_rows, self.cfg.num_cols, patch_cfg.num_patches, 3), device=self.device
                    )
                # add the flat patches to the tensor
                self.flat_patches[name][row, col] = find_flat_patches(
                    wp_mesh=wp_mesh,
                    origin=origin,
                    num_patches=patch_cfg.num_patches,
                    patch_radius=patch_cfg.patch_radius,
                    x_range=patch_cfg.x_range,
                    y_range=patch_cfg.y_range,
                    z_range=patch_cfg.z_range,
                    max_height_diff=patch_cfg.max_height_diff,
                )

        # transform the mesh to the correct position
        transform = np.eye(4)
        transform[0:2, -1] = (row + 0.5) * self.cfg.size[0], (col + 0.5) * self.cfg.size[1]
        mesh.apply_transform(transform)
        # add mesh to the list
        self.terrain_meshes.append(mesh)
        # add origin to the list
        self.terrain_origins[row, col] = origin + transform[:3, -1]

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg) -> tuple[trimesh.Trimesh, np.ndarray]:
        """Generate a sub-terrain mesh based on the input difficulty parameter.

        If caching is enabled, the sub-terrain is cached and loaded from the cache if it exists.
        The cache is stored in the cache directory specified in the configuration.

        .. Note:
            This function centers the 2D center of the mesh and its specified origin such that the
            2D center becomes :math:`(0, 0)` instead of :math:`(size[0] / 2, size[1] / 2).

        Args:
            difficulty: The difficulty parameter.
            cfg: The configuration of the sub-terrain.

        Returns:
            The sub-terrain mesh and origin.
        """
        """根据输入难度参数生成地下网格。

        如果启用缓存，地下区域将被缓存并从缓存中加载，如果存在。
        缓存存储在配置中指定的缓存目录中。

        ..
        注意:这个函数将网格的2D中心和其指定的起源集中，使2D中心成为:math:`(0， 0)`而不是:math:`(size[0] / 2，大小[1] / 2)。

        参数：
            difficulty: 难度参数。
            cfg: 地下地区的配置。

        返回：
            地下网格和来源。
        """
        # copy the configuration
        cfg = cfg.copy()
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # generate hash for the sub-terrain
        sub_terrain_hash = dict_to_md5_hash(cfg.to_dict())
        # generate the file name
        sub_terrain_cache_dir = os.path.join(self.cfg.cache_dir, sub_terrain_hash)
        sub_terrain_obj_filename = os.path.join(sub_terrain_cache_dir, "mesh.obj")
        sub_terrain_csv_filename = os.path.join(sub_terrain_cache_dir, "origin.csv")
        sub_terrain_meta_filename = os.path.join(sub_terrain_cache_dir, "cfg.yaml")

        # check if hash exists - if true, load the mesh and origin and return
        if self.cfg.use_cache and os.path.exists(sub_terrain_obj_filename):
            # load existing mesh
            mesh = trimesh.load_mesh(sub_terrain_obj_filename, process=False)
            origin = np.loadtxt(sub_terrain_csv_filename, delimiter=",")
            # return the generated mesh
            return mesh, origin

        # generate the terrain
        meshes, origin = cfg.function(difficulty, cfg)
        mesh = trimesh.util.concatenate(meshes)
        # offset mesh such that they are in their center
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        # change origin to be in the center of the sub-terrain
        origin += transform[0:3, -1]

        # if caching is enabled, save the mesh and origin
        if self.cfg.use_cache:
            # create the cache directory
            os.makedirs(sub_terrain_cache_dir, exist_ok=True)
            # save the data
            mesh.export(sub_terrain_obj_filename)
            np.savetxt(sub_terrain_csv_filename, origin, delimiter=",", header="x,y,z")
            dump_yaml(sub_terrain_meta_filename, cfg)
        # return the generated mesh
        return mesh, origin
