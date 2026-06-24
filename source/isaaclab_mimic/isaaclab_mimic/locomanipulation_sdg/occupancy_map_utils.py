# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import enum
import math
import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
import PIL.Image
import torch
import yaml
from PIL import ImageDraw

from pxr import Kind, Sdf, Usd, UsdGeom, UsdShade


@dataclass
class Point2d:
    x: float
    y: float


ROS_FREESPACE_THRESH_DEFAULT = 0.196
ROS_OCCUPIED_THRESH_DEFAULT = 0.65

OCCUPANCY_MAP_DEFAULT_Z_MIN = 0.1
OCCUPANCY_MAP_DEFAULT_Z_MAX = 0.62
OCCUPANCY_MAP_DEFAULT_CELL_SIZE = 0.05


class OccupancyMapDataValue(enum.IntEnum):
    UNKNOWN = 0
    FREESPACE = 1
    OCCUPIED = 2

    def ros_image_value(self, negate: bool = False) -> int:
        values = [0, 127, 255]

        if negate:
            values = values[::-1]

        if self == OccupancyMapDataValue.OCCUPIED:
            return values[0]
        elif self == OccupancyMapDataValue.UNKNOWN:
            return values[1]
        else:
            return values[2]


class OccupancyMapMergeMethod(enum.IntEnum):
    UNION = 0
    INTERSECTION = 1


class OccupancyMap:
    ROS_IMAGE_FILENAME = "map.png"
    ROS_YAML_FILENAME = "map.yaml"
    ROS_YAML_TEMPLATE = """
image: {image_filename}
resolution: {resolution}
origin: {origin}
negate: {negate}
occupied_thresh: {occupied_thresh}
free_thresh: {free_thresh}
"""

    def __init__(self, data: np.ndarray, resolution: int, origin: tuple[int, int, int]) -> None:
        self.data = data
        self.resolution = resolution  # meters per pixel
        self.origin = origin  # x, y, yaw.  where (x, y) is the bottom-left of image
        self._width_pixels = data.shape[1]
        self._height_pixels = data.shape[0]

    def freespace_mask(self) -> np.ndarray:
        """Get a binary mask representing the freespace of the occupancy map.

        Returns:
            np.ndarray: The binary mask representing freespace of the occupancy map.
        """
        """拿一个二进制面具来表示占用地图的自由空间。

        返回：
            np.ndarray: 占用地图的自由空间。
        """
        return self.data == OccupancyMapDataValue.FREESPACE

    def unknown_mask(self) -> np.ndarray:
        """Get a binary mask representing the unknown area of the occupancy map.

        Returns:
            np.ndarray: The binary mask representing unknown area of the occupancy map.
        """
        """拿一个二进制面具，代表居住地图的未知区域。

        返回：
            np.ndarray: 占用地图的未知区域的二进制面具。
        """
        return self.data == OccupancyMapDataValue.UNKNOWN

    def occupied_mask(self) -> np.ndarray:
        """Get a binary mask representing the occupied area of the occupancy map.

        Returns:
            np.ndarray: The binary mask representing occupied area of the occupancy map.
        """
        """获得一个占用地图的占用区域的二进制面具。

        返回：
            np.ndarray: 占用地图的占用区域的二进制面具。
        """
        return self.data == OccupancyMapDataValue.OCCUPIED

    def ros_image(self, negate: bool = False) -> PIL.Image.Image:
        """Get the ROS image for the occupancy map.

        Args:
            negate (bool, optional): See "negate" in ROS occupancy map documentation. Defaults to False.

        Returns:
            PIL.Image.Image: The ROS image for the occupancy map as a PIL image.
        """
        """拿出ROS图像，为居住地图。

        参数：
            negate (bool, optional): 在ROS占用地图文档中见"负值"。
                                     默认为 False。

        返回：
            PIL.Image.Image: 占用地图的ROS图像作为PIL图像。
        """
        occupied_mask = self.occupied_mask()
        ros_image = np.zeros(self.occupied_mask().shape, dtype=np.uint8)
        ros_image[occupied_mask] = OccupancyMapDataValue.OCCUPIED.ros_image_value(negate)
        ros_image[self.unknown_mask()] = OccupancyMapDataValue.UNKNOWN.ros_image_value(negate)
        ros_image[self.freespace_mask()] = OccupancyMapDataValue.FREESPACE.ros_image_value(negate)
        ros_image = PIL.Image.fromarray(ros_image)
        return ros_image

    def ros_yaml(self, negate: bool = False) -> str:
        """Get the ROS occupancy map YAML file content.

        Args:
            negate (bool, optional): See "negate" in ROS occupancy map documentation. Defaults to False.

        Returns:
            str: The ROS occupancy map YAML file contents.
        """
        """获取ROS占用地图YAML文件内容。

        参数：
            negate (bool, optional): 在ROS占用地图文档中见"负值"。
                                     默认为 False。

        返回：
            str: ROS占用地图YAML文件内容。
        """
        return self.ROS_YAML_TEMPLATE.format(
            image_filename=self.ROS_IMAGE_FILENAME,
            resolution=self.resolution,
            origin=list(self.origin),
            negate=1 if negate else 0,
            occupied_thresh=ROS_OCCUPIED_THRESH_DEFAULT,
            free_thresh=ROS_FREESPACE_THRESH_DEFAULT,
        )

    def save_ros(self, path: str):
        """Save the occupancy map to a folder in ROS format.

        This method saves both the ROS formatted PNG image, as well
        as the corresponding YAML file.

        Args:
            path (str): The output path to save the occupancy map.
        """
        """保存占用地图在ROS格式的文件中。

        这种方法可以保存ROS格式的PNG图像以及相应的YAML文件。

        参数：
            path (str): 输出路径保存占用地图。
        """
        if not os.path.exists(path):
            os.makedirs(path)
        assert os.path.isdir(path)  # safety check
        self.ros_image().save(os.path.join(path, self.ROS_IMAGE_FILENAME))
        with open(os.path.join(path, self.ROS_YAML_FILENAME), "w", encoding="utf-8") as f:
            f.write(self.ros_yaml())

    @staticmethod
    def from_ros_yaml(ros_yaml_path: str) -> "OccupancyMap":
        """Load an occupancy map from a ROS YAML file.

        This method loads an occupancy map from a ROS yaml file.
        This method looks up the occupancy map image from the
        value specified in the YAML file, and requires that
        the image exists at the specified path.

        Args:
            ros_yaml_path (str): The path to the ROS yaml file.

        Returns:
            _type_: OccupancyMap
        """
        """从ROSXYAML文件中加载一个占用地图。

        这种方法从ROS yaml文件中加载了占用地图。
        这种方法从YAML文件中指定的值中查找占用地图图像，并要求图像存在于指定路径。

        参数：
            ros_yaml_path (str): 进入ROS yamML文件的路径。

        返回：
            _type_: OccupancyMap
        """
        with open(ros_yaml_path, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        yaml_dir = os.path.dirname(ros_yaml_path)
        image_path = os.path.join(yaml_dir, yaml_data["image"])
        image = PIL.Image.open(image_path).convert("L")
        occupancy_map = OccupancyMap.from_ros_image(
            ros_image=image,
            resolution=yaml_data["resolution"],
            origin=yaml_data["origin"],
            negate=yaml_data["negate"],
            occupied_thresh=yaml_data["occupied_thresh"],
            free_thresh=yaml_data["free_thresh"],
        )
        return occupancy_map

    @staticmethod
    def from_ros_image(
        ros_image: PIL.Image.Image,
        resolution: float,
        origin: tuple[float, float, float],
        negate: bool = False,
        occupied_thresh: float = ROS_OCCUPIED_THRESH_DEFAULT,
        free_thresh: float = ROS_FREESPACE_THRESH_DEFAULT,
    ) -> "OccupancyMap":
        """Create an occupancy map from a ROS formatted image, and other data.

        This method is intended to be used as a utility by other methods,
        but not necessarily useful for end use cases.

        Args:
            ros_image (PIL.Image.Image): The ROS formatted PIL image.
            resolution (float): The resolution (meter/px) of the occupancy map.
            origin (tp.Tuple[float, float, float]): The origin of the occupancy map in world coordinates.
            negate (bool, optional): See "negate" in ROS occupancy map documentation. Defaults to False.
            occupied_thresh (float, optional): The threshold to consider a value occupied.
                Defaults to ROS_OCCUPIED_THRESH_DEFAULT.
            free_thresh (float, optional): The threshold to consider a value free. Defaults to
                ROS_FREESPACE_THRESH_DEFAULT.

        Returns:
            OccupancyMap: The occupancy map.
        """
        """从ROS格式图像创建一个占用地图，以及其他数据。

        这种方法旨在用作其他方法的实用性，但不一定适用于最终用途。

        参数：
            ros_image (PIL.Image.Image): 在ROS格式化PIL图像
            resolution (float): 占用地图的分辨率 (meter/px)。
            origin (tp.Tuple[float, float, float]): 在世界坐标中占用地图的起源。
            negate (bool, optional): 在ROS占用地图文档中见"负值"。
                                     默认为 False。
            occupied_thresh (float, optional): 考虑占用值的门。
                                               在ROS_OCCUPIED_THRESH_DEFAULT中设置错误。
            free_thresh (float, optional): 值值可以被视为自由。
                                           在ROS_FREESPACE_THRESH_DEFAULT中默认错误。

        返回：
            OccupancyMap: 居住地图。
        """
        ros_image = ros_image.convert("L")

        free_thresh = free_thresh * 255
        occupied_thresh = occupied_thresh * 255

        data = np.asarray(ros_image)

        if not negate:
            data = 255 - data

        freespace_mask = data < free_thresh
        occupied_mask = data > occupied_thresh

        return OccupancyMap.from_masks(
            freespace_mask=freespace_mask, occupied_mask=occupied_mask, resolution=resolution, origin=origin
        )

    @staticmethod
    def from_masks(
        freespace_mask: np.ndarray, occupied_mask: np.ndarray, resolution: float, origin: tuple[float, float, float]
    ) -> "OccupancyMap":
        """Creates an occupancy map from binary masks and other data

        This method is intended as a utility by other methods, but not necessarily
        useful for end use cases.

        Args:
            freespace_mask (np.ndarray): Binary mask for the freespace region.
            occupied_mask (np.ndarray): Binary mask for the occupied region.
            resolution (float): The resolution of the map (meters/px).
            origin (tp.Tuple[float, float, float]): The origin of the map in world coordinates.

        Returns:
            OccupancyMap: The occupancy map.
        """
        """从二元面具和其他数据创建一个占用地图

        这种方法是用其他方法的实用性，但不一定适用于最终用途。

        参数：
            freespace_mask (np.ndarray): 为自由空间区域进行二进制面具。
            occupied_mask (np.ndarray): 占据地区的二元面具。
            resolution (float): 地图的分辨率 (m/px)。
            origin (tp.Tuple[float, float, float]): 世界坐标地图的起源。

        返回：
            OccupancyMap: 居住地图。
        """

        data = np.zeros(freespace_mask.shape, dtype=np.uint8)
        data[...] = OccupancyMapDataValue.UNKNOWN
        data[freespace_mask] = OccupancyMapDataValue.FREESPACE
        data[occupied_mask] = OccupancyMapDataValue.OCCUPIED

        occupancy_map = OccupancyMap(data=data, resolution=resolution, origin=origin)

        return occupancy_map

    @staticmethod
    def from_occupancy_boundary(boundary: np.ndarray, resolution: float) -> "OccupancyMap":
        min_xy = boundary.min(axis=0)
        max_xy = boundary.max(axis=0)
        origin = (float(min_xy[0]), float(min_xy[1]), 0.0)
        width_meters = max_xy[0] - min_xy[0]
        height_meters = max_xy[1] - min_xy[1]
        width_pixels = math.ceil(width_meters / resolution)
        height_pixels = math.ceil(height_meters / resolution)

        points = boundary

        bot_left_world = (origin[0], origin[1])
        u = (points[:, 0] - bot_left_world[0]) / width_meters
        v = 1.0 - (points[:, 1] - bot_left_world[1]) / height_meters
        x_px = u * width_pixels
        y_px = v * height_pixels

        xy_px = np.concatenate([x_px[:, None], y_px[:, None]], axis=-1).flatten()

        image = np.zeros((height_pixels, width_pixels, 4), dtype=np.uint8)
        image = PIL.Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.polygon(xy_px.tolist(), fill="white", outline="red")
        image = np.asarray(image)

        occupied_mask = image[:, :, 0] > 0

        freespace_mask = ~occupied_mask

        return OccupancyMap.from_masks(freespace_mask, occupied_mask, resolution, origin)

    @staticmethod
    def make_empty(start: tuple[float, float], end: tuple[float, float], resolution: float) -> "OccupancyMap":
        origin = (start[0], start[1], 0.0)
        width_meters = end[0] - start[0]
        height_meters = end[1] - start[1]
        width_pixels = math.ceil(width_meters / resolution)
        height_pixels = math.ceil(height_meters / resolution)
        occupied_mask = np.zeros((height_pixels, width_pixels), dtype=np.uint8) > 0
        freespace_mask = np.ones((height_pixels, width_pixels), dtype=np.uint8) > 0
        return OccupancyMap.from_masks(freespace_mask, occupied_mask, resolution, origin)

    def width_pixels(self) -> int:
        """Get the width of the occupancy map in pixels.

        Returns:
            int: The width in pixels.
        """
        """在像素中获得占用地图的宽度。

        返回：
            int: 像素的宽度。
        """
        return self._width_pixels

    def height_pixels(self) -> int:
        """Get the height of the occupancy map in pixels.

        Returns:
            int: The height in pixels.
        """
        """在像素中获得占用地图的高度。

        返回：
            int: 像素的高度。
        """
        return self._height_pixels

    def width_meters(self) -> float:
        """Get the width of the occupancy map in meters.

        Returns:
            float: The width in meters.
        """
        """在米中得到占用地图的宽度。

        返回：
            float: 宽度在米。
        """
        return self.resolution * self.width_pixels()

    def height_meters(self) -> float:
        """Get the height of the occupancy map in meters.

        Returns:
            float: The height in meters.
        """
        """在米中查取占用地图的高度。

        返回：
            float: 在公尺的高度。
        """
        return self.resolution * self.height_pixels()

    def bottom_left_pixel_world_coords(self) -> tuple[float, float]:
        """Get the world coordinates of the bottom left pixel.

        Returns:
            tp.Tuple[float, float]: The (x, y) world coordinates of the
                bottom left pixel in the occupancy map.
        """
        """得到左下方像素的世界坐标。

        返回：
            tp.Tuple[漂浮，漂浮]:占用地图中左下方像素的 (x， y) 世界坐标。
        """
        return (self.origin[0], self.origin[1])

    def top_left_pixel_world_coords(self) -> tuple[float, float]:
        """Get the world coordinates of the top left pixel.

        Returns:
            tp.Tuple[float, float]: The (x, y) world coordinates of the
                top left pixel in the occupancy map.
        """
        """得到左上方像素的世界坐标。

        返回：
            tp.Tuple[漂浮，漂浮]:占用地图上方左边像素的 (x， y) 世界坐标。
        """
        return (self.origin[0], self.origin[1] + self.height_meters())

    def bottom_right_pixel_world_coords(self) -> tuple[float, float]:
        """Get the world coordinates of the bottom right pixel.

        Returns:
            tp.Tuple[float, float]: The (x, y) world coordinates of the
                bottom right pixel in the occupancy map.
        """
        """得到右下方像素的世界坐标。

        返回：
            tp.Tuple[漂浮，漂浮]:占用地图中右下方像素的 (x， y) 世界坐标。
        """
        return (self.origin[0] + self.width_meters(), self.origin[1])

    def top_right_pixel_world_coords(self) -> tuple[float, float]:
        """Get the world coordinates of the top right pixel.

        Returns:
            tp.Tuple[float, float]: The (x, y) world coordinates of the
                top right pixel in the occupancy map.
        """
        """得到右上方像素的世界坐标。

        返回：
            tp.Tuple[漂浮，漂浮]:占用地图上右边像素的 (x， y) 世界坐标。
        """
        return (self.origin[0] + self.width_meters(), self.origin[1] + self.height_meters())

    def buffered(self, buffer_distance_pixels: int) -> "OccupancyMap":
        """Get a buffered occupancy map by dilating the occupied regions.

        This method buffers (aka: pads / dilates) an occupancy map by dilating
        the occupied regions using a circular mask with the a radius
        specified by "buffer_distance_pixels".

        This is useful for modifying an occupancy map for path planning,
        collision checking, or robot spawning with the simple assumption
        that the robot has a circular collision profile.

        Args:
            buffer_distance_pixels (int): The buffer radius / distance in pixels.

        Returns:
            OccupancyMap: The buffered (aka: dilated / padded) occupancy map.
        """
        """通过扩大占据地区来获得缓冲地图。

        这种方法通过使用"buffer_distance_pixels"指定的半径圆形面具来扩大占用区域来缓冲 (也称为pads/dilates) 占用地图。

        这可用于修改路径规划，碰撞检查或机器人繁殖的占用地图，简单地假设机器人具有圆形碰撞形状。

        参数：
            buffer_distance_pixels (int): 在像素中缓冲半径/距离。

        返回：
            OccupancyMap: 缓冲式 (也称为扩展/填充) 居住地图。
        """

        buffer_distance_pixels = int(buffer_distance_pixels)

        radius = buffer_distance_pixels
        diameter = radius * 2
        kernel = np.zeros((diameter, diameter), np.uint8)
        cv2.circle(kernel, (radius, radius), radius, 255, -1)
        occupied = self.occupied_mask().astype(np.uint8) * 255
        occupied_dilated = cv2.dilate(occupied, kernel, iterations=1)
        occupied_mask = occupied_dilated == 255
        free_mask = self.freespace_mask()
        free_mask[occupied_mask] = False

        return OccupancyMap.from_masks(
            freespace_mask=free_mask, occupied_mask=occupied_mask, resolution=self.resolution, origin=self.origin
        )

    def buffered_meters(self, buffer_distance_meters: float) -> "OccupancyMap":
        """Get a buffered occupancy map by dilating the occupied regions.

        See OccupancyMap.buffer() for more details.

        Args:
            buffer_distance_meters (int): The buffer radius / distance in pixels.

        Returns:
            OccupancyMap: The buffered (aka: dilated / padded) occupancy map.
        """
        """通过扩大占据地区来获得缓冲地图。

        看看OccupancyMap.buffer更多详情。

        参数：
            buffer_distance_meters (int): 在像素中缓冲半径/距离。

        返回：
            OccupancyMap: 缓冲式 (也称为扩展/填充) 居住地图。
        """
        buffer_distance_pixels = int(buffer_distance_meters / self.resolution)
        return self.buffered(buffer_distance_pixels)

    def pixel_to_world(self, point: Point2d) -> Point2d:
        """Convert a pixel coordinate to world coordinates.

        Args:
            point (Point2d): The pixel coordinate.

        Returns:
            Point2d: The world coordinate.
        """
        """转换一个像素坐标为世界坐标。

        参数：
            point (Point2d): 像素坐标。

        返回：
            Point2d: 世界协调。
        """
        # currently doesn't handle rotations
        bot_left = self.bottom_left_pixel_world_coords()
        u = point.x / self.width_pixels()
        v = 1.0 - point.y / self.height_pixels()
        x_world = u * self.width_meters() + bot_left[0]
        y_world = v * self.height_meters() + bot_left[1]
        return Point2d(x=x_world, y=y_world)

    def pixel_to_world_numpy(self, points: np.ndarray) -> np.ndarray:
        """Convert an array of pixel coordinates to world coordinates.

        Args:
            points (np.ndarray): The Nx2 numpy array of pixel coordinates.

        Returns:
            np.ndarray: The Nx2 numpy array of world coordinates.
        """
        """转换一个像素坐标阵列为世界坐标。

        参数：
            points (np.ndarray): 像素坐标的Nx2 numpy阵列。

        返回：
            np.ndarray: 世界坐标的Nx2 numpy阵列。
        """
        bot_left = self.bottom_left_pixel_world_coords()
        u = points[:, 0] / self.width_pixels()
        v = 1.0 - points[:, 1] / self.height_pixels()
        x_world = u * self.width_meters() + bot_left[0]
        y_world = v * self.height_meters() + bot_left[1]
        return np.concatenate([x_world[:, None], y_world[:, None]], axis=-1)

    def world_to_pixel_numpy(self, points: np.ndarray) -> np.ndarray:
        """Convert an array of world coordinates to pixel coordinates.

        Args:
            points (np.ndarray): The Nx2 numpy array of world coordinates.

        Returns:
            np.ndarray: The Nx2 numpy array of pixel coordinates.
        """
        """将世界坐标数组转换为像素坐标。

        参数：
            points (np.ndarray): 世界坐标的Nx2 numpy阵列。

        返回：
            np.ndarray: 像素坐标的Nx2 numpy阵列。
        """
        bot_left_world = self.bottom_left_pixel_world_coords()
        u = (points[:, 0] - bot_left_world[0]) / self.width_meters()
        v = 1.0 - (points[:, 1] - bot_left_world[1]) / self.height_meters()
        x_px = u * self.width_pixels()
        y_px = v * self.height_pixels()
        return np.concatenate([x_px[:, None], y_px[:, None]], axis=-1)

    def check_world_point_in_bounds(self, point: Point2d) -> bool:
        """Check if a world coordinate is inside the bounds of the occupancy map.

        Args:
            point (Point2d): The world coordinate.

        Returns:
            bool: True if the coordinate is inside the bounds of
                the occupancy map.  False otherwise.
        """
        """检查世界坐标是否在占用地图的范围内。

        参数：
            point (Point2d): 世界协调。

        返回：
            bool: True如果坐标位于占用地图的边界内。
                  False否则。
        """

        pixel = self.world_to_pixel_numpy(np.array([[point.x, point.y]]))
        x_px = int(pixel[0, 0])
        y_px = int(pixel[0, 1])

        if (x_px < 0) or (x_px >= self.width_pixels()) or (y_px < 0) or (y_px >= self.height_pixels()):
            return False

        return True

    def check_world_point_in_freespace(self, point: Point2d) -> bool:
        """Check if a world coordinate is inside the freespace region of the occupancy map

        Args:
            point (Point2d): The world coordinate.

        Returns:
            bool: True if the world coordinate is inside the freespace region of the occupancy map.
                False otherwise.
        """
        """检查世界坐标是否位于占用地图的自由空间区域内

        参数：
            point (Point2d): 世界协调。

        返回：
            bool: True如果世界坐标位于占用地图的自由空间区域内。
                  False否则。
        """
        if not self.check_world_point_in_bounds(point):
            return False
        pixel = self.world_to_pixel_numpy(np.array([[point.x, point.y]]))
        x_px = int(pixel[0, 0])
        y_px = int(pixel[0, 1])
        freespace = self.freespace_mask()
        return bool(freespace[y_px, x_px])

    def transformed(self, transform: np.ndarray) -> "OccupancyMap":
        return transform_occupancy_map(self, transform)

    def merged(self, other: "OccupancyMap") -> "OccupancyMap":
        return merge_occupancy_maps([self, other])


def _omap_world_to_px(
    points: np.ndarray,
    origin: tuple[float, float, float],
    width_meters: float,
    height_meters: float,
    width_pixels: int,
    height_pixels: int,
) -> np.ndarray:
    bot_left_world = (origin[0], origin[1])
    u = (points[:, 0] - bot_left_world[0]) / width_meters
    v = 1.0 - (points[:, 1] - bot_left_world[1]) / height_meters
    x_px = u * width_pixels
    y_px = v * height_pixels
    return np.stack([x_px, y_px], axis=-1)


def merge_occupancy_maps(
    src_omaps: list[OccupancyMap], method: OccupancyMapMergeMethod = OccupancyMapMergeMethod.UNION
) -> OccupancyMap:
    """Merge occupancy maps by computing the union or intersection of the occupied regions."""
    """通过计算占领地区的联盟或交叉点，将占用地图合并。"""
    dst_resolution = min([o.resolution for o in src_omaps])

    min_x = min([o.bottom_left_pixel_world_coords()[0] for o in src_omaps])
    min_y = min([o.bottom_left_pixel_world_coords()[1] for o in src_omaps])
    max_x = max([o.top_right_pixel_world_coords()[0] for o in src_omaps])
    max_y = max([o.top_right_pixel_world_coords()[1] for o in src_omaps])

    dst_origin = (min_x, min_y, 0.0)

    dst_width_meters = max_x - min_x
    dst_height_meters = max_y - min_y
    dst_width_pixels = math.ceil(dst_width_meters / dst_resolution)
    dst_height_pixels = math.ceil(dst_height_meters / dst_resolution)

    dst_occupied_mask: np.ndarray
    if method == OccupancyMapMergeMethod.UNION:
        dst_occupied_mask = np.zeros((dst_height_pixels, dst_width_pixels), dtype=bool)
    elif method == OccupancyMapMergeMethod.INTERSECTION:
        dst_occupied_mask = np.ones((dst_height_pixels, dst_width_pixels), dtype=bool)
    else:
        raise ValueError(f"Unsupported merge method: {method}")

    for src_omap in src_omaps:
        omap_corners_in_world_coords = np.array(
            [src_omap.top_left_pixel_world_coords(), src_omap.bottom_right_pixel_world_coords()]
        )

        omap_corners_in_dst_image_coords = (
            _omap_world_to_px(
                omap_corners_in_world_coords,
                dst_origin,
                dst_width_meters,
                dst_height_meters,
                dst_width_pixels,
                dst_height_pixels,
            )
            .astype(np.int64)
            .flatten()
        )

        omap_dst_width = omap_corners_in_dst_image_coords[2] - omap_corners_in_dst_image_coords[0]
        omap_dst_height = omap_corners_in_dst_image_coords[3] - omap_corners_in_dst_image_coords[1]

        omap_occupied_image = PIL.Image.fromarray(255 * src_omap.occupied_mask().astype(np.uint8)).resize(
            (omap_dst_width, omap_dst_height)
        )

        omap_occupied_image_tmp = omap_occupied_image.copy()

        dst_occupied_image = PIL.Image.fromarray(np.zeros_like(dst_occupied_mask).astype(np.uint8))

        dst_occupied_image.paste(omap_occupied_image_tmp, box=omap_corners_in_dst_image_coords)

        if method == OccupancyMapMergeMethod.UNION:
            dst_occupied_mask = dst_occupied_mask | (np.asarray(dst_occupied_image) > 0)
        elif method == OccupancyMapMergeMethod.INTERSECTION:
            dst_occupied_mask = dst_occupied_mask & (np.asarray(dst_occupied_image) > 0)

    dst_occupancy_map = OccupancyMap.from_masks(
        freespace_mask=~dst_occupied_mask, occupied_mask=dst_occupied_mask, resolution=dst_resolution, origin=dst_origin
    )

    return dst_occupancy_map


def intersect_occupancy_maps(src_omaps: list[OccupancyMap]) -> OccupancyMap:
    """Compute a new occupancy map by intersecting the occupied regions of a list of occupancy maps."""
    """通过交叉占用地区的占用地图列表来计算一个新的占用地图。"""
    return merge_occupancy_maps(src_omaps=src_omaps, method=OccupancyMapMergeMethod.INTERSECTION)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform a set of points by a 2D transform."""
    """通过二维转换将一组点转换成。"""
    points = np.concatenate([points, np.ones_like(points[:, 0:1])], axis=-1).T
    points = transform @ points
    points = points.T
    points = points[:, :2]
    return points


def make_rotate_transform(angle: float) -> np.ndarray:
    """Create a 2D rotation transform."""
    """创建一个2D旋转转换。"""
    return np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])


def make_translate_transform(dx: float, dy: float) -> np.ndarray:
    """Create a 2D translation transform."""
    """创建一个2D翻译转换。"""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def transform_occupancy_map(omap: OccupancyMap, transform: np.ndarray) -> OccupancyMap:
    """Transform an occupancy map using a 2D transform."""
    """用二维转换来转换一个占用地图。"""

    src_box_world_coords = np.array(
        [
            [omap.origin[0], omap.origin[1]],
            [omap.origin[0] + omap.width_meters(), omap.origin[1]],
            [omap.origin[0] + omap.width_meters(), omap.origin[1] + omap.height_meters()],
            [omap.origin[0], omap.origin[1] + omap.height_meters()],
        ]
    )

    src_box_pixel_coords = omap.world_to_pixel_numpy(src_box_world_coords)

    dst_box_world_coords = transform_points(src_box_world_coords, transform)

    dst_min_xy = np.min(dst_box_world_coords, axis=0)
    dst_max_xy = np.max(dst_box_world_coords, axis=0)

    dst_origin = (float(dst_min_xy[0]), float(dst_min_xy[1]), 0)
    dst_width_meters = dst_max_xy[0] - dst_min_xy[0]
    dst_height_meters = dst_max_xy[1] - dst_min_xy[1]
    dst_resolution = omap.resolution
    dst_width_pixels = int(dst_width_meters / dst_resolution)
    dst_height_pixels = int(dst_height_meters / dst_resolution)

    dst_box_pixel_coords = _omap_world_to_px(
        dst_box_world_coords, dst_origin, dst_width_meters, dst_height_meters, dst_width_pixels, dst_height_pixels
    )

    persp_transform = cv2.getPerspectiveTransform(
        src_box_pixel_coords.astype(np.float32), dst_box_pixel_coords.astype(np.float32)
    )

    src_occupied_mask = omap.occupied_mask().astype(np.uint8) * 255

    dst_occupied_mask = cv2.warpPerspective(src_occupied_mask, persp_transform, (dst_width_pixels, dst_height_pixels))

    dst_occupied_mask = dst_occupied_mask > 0
    dst_freespace_mask = ~dst_occupied_mask

    dst_omap = OccupancyMap.from_masks(dst_freespace_mask, dst_occupied_mask, dst_resolution, dst_origin)

    return dst_omap


def occupancy_map_add_to_stage(
    occupancy_map: OccupancyMap,
    stage: Usd.Stage,
    path: str,
    z_offset: float = 0.0,
    draw_path: np.ndarray | torch.Tensor | None = None,
    draw_path_line_width_meter: float = 0.25,
) -> Usd.Prim:
    image_path = os.path.join(tempfile.mkdtemp(), "texture.png")
    image = occupancy_map.ros_image()

    if draw_path is not None:
        if isinstance(draw_path, torch.Tensor):
            draw_path = draw_path.detach().cpu().numpy()
        image = image.copy().convert("RGBA")
        draw = ImageDraw.Draw(image)
        line_coordinates = []
        path_pixels = occupancy_map.world_to_pixel_numpy(draw_path)
        for i in range(len(path_pixels)):
            line_coordinates.append(int(path_pixels[i, 0]))
            line_coordinates.append(int(path_pixels[i, 1]))
        width_pixels = draw_path_line_width_meter / occupancy_map.resolution
        draw.line(line_coordinates, fill="green", width=int(width_pixels / 2), joint="curve")

    # need to flip, ros uses inverted coordinates on y axis
    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    image.save(image_path)

    x0, y0 = occupancy_map.top_left_pixel_world_coords()
    x1, y1 = occupancy_map.bottom_right_pixel_world_coords()

    # Add model
    modelRoot = UsdGeom.Xform.Define(stage, path)
    Usd.ModelAPI(modelRoot).SetKind(Kind.Tokens.component)

    # Add mesh
    mesh = UsdGeom.Mesh.Define(stage, os.path.join(path, "mesh"))
    mesh.CreatePointsAttr([(x0, y0, z_offset), (x1, y0, z_offset), (x1, y1, z_offset), (x0, y1, z_offset)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateExtentAttr([(x0, y0, z_offset), (x1, y1, z_offset)])

    texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
    )

    texCoords.Set([(0, 0), (1, 0), (1, 1), (0, 1)])

    # Add material
    material_path = os.path.join(path, "material")
    material = UsdShade.Material.Define(stage, material_path)
    pbrShader = UsdShade.Shader.Define(stage, os.path.join(material_path, "shader"))
    pbrShader.CreateIdAttr("UsdPreviewSurface")
    pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

    # Add texture to material
    stReader = UsdShade.Shader.Define(stage, os.path.join(material_path, "st_reader"))
    stReader.CreateIdAttr("UsdPrimvarReader_float2")
    diffuseTextureSampler = UsdShade.Shader.Define(stage, os.path.join(material_path, "diffuse_texture"))
    diffuseTextureSampler.CreateIdAttr("UsdUVTexture")
    diffuseTextureSampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(image_path)
    diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        stReader.ConnectableAPI(), "result"
    )
    diffuseTextureSampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        diffuseTextureSampler.ConnectableAPI(), "rgb"
    )

    stInput = material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
    stInput.Set("st")
    stReader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(stInput)
    mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(mesh).Bind(material)

    return modelRoot
