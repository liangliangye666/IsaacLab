# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os

import torch
import torch.nn as nn
import torchvision

from isaaclab.sensors import save_images_to_file
from isaaclab.utils import configclass


class FeatureExtractorNetwork(nn.Module):
    """CNN architecture used to regress keypoint positions of the in-hand cube from image data."""
    """CNN架构用于从图像数据中退回手中的立方体的关键点位置。"""

    def __init__(self):
        super().__init__()
        num_channel = 7
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([16, 58, 58]),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([32, 28, 28]),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([64, 13, 13]),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm([128, 6, 6]),
            nn.AvgPool2d(6),
        )

        self.linear = nn.Sequential(
            nn.Linear(128, 27),
        )

        self.data_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x[:, 0:3, :, :] = self.data_transforms(x[:, 0:3, :, :])
        x[:, 4:7, :, :] = self.data_transforms(x[:, 4:7, :, :])
        cnn_x = self.cnn(x)
        out = self.linear(cnn_x.view(-1, 128))
        return out


@configclass
class FeatureExtractorCfg:
    """Configuration for the feature extractor model."""
    """功能提取器模型的配置。"""

    train: bool = True
    """If True, the feature extractor model is trained during the rollout process. Default is False."""
    """如果True，特征提取器模型在推出过程中训练。
    默认是False。
    """

    load_checkpoint: bool = False
    """If True, the feature extractor model is loaded from a checkpoint. Default is False."""
    """如果True，则从检查点上装载特征提取器模型。
    默认是False。
    """

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""
    """如果True，摄像头传感器的图像将被写成文件。
    默认是False。
    """


class FeatureExtractor:
    """Class for extracting features from image data.

    It uses a CNN to regress keypoint positions from normalized RGB, depth, and segmentation images.
    If the train flag is set to True, the CNN is trained during the rollout process.
    """
    """从图像数据中提取特征的类。

    它使用CNN从正常化的RGB，深度和细分图像中退回关键点位置。
    如果列车旗设置为True，则CNN在部署过程中训练。
    """

    def __init__(self, cfg: FeatureExtractorCfg, device: str, log_dir: str | None = None):
        """Initialize the feature extractor model.

        Args:
            cfg: Configuration for the feature extractor model.
            device: Device to run the model on.
            log_dir: Directory to save checkpoints. Default is None, which uses the local
                "logs" folder resolved relative to this file.
        """
        """启动特征提取模型。

        参数：
            cfg: 功能提取器模型的配置。
            device: 运行模型的设备。
            log_dir: 保存检查点的目录。
                     默认是None，它使用本地"日志"文件对此文件进行解决。
        """

        self.cfg = cfg
        self.device = device

        # Feature extractor model
        self.feature_extractor = FeatureExtractorNetwork()
        self.feature_extractor.to(self.device)

        self.step_count = 0
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.cfg.load_checkpoint:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint = os.path.join(self.log_dir, latest_file)
            print(f"[INFO]: Loading feature extractor checkpoint from {checkpoint}")
            self.feature_extractor.load_state_dict(torch.load(checkpoint, weights_only=True))

        if self.cfg.train:
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
            self.l2_loss = nn.MSELoss()
            self.feature_extractor.train()
        else:
            self.feature_extractor.eval()

    def _preprocess_images(
        self, rgb_img: torch.Tensor, depth_img: torch.Tensor, segmentation_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocesses the input images.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor): Depth image tensor. Shape: (N, H, W, 1).
            segmentation_img (torch.Tensor): Segmentation image tensor. Shape: (N, H, W, 3)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Preprocessed RGB, depth, and segmentation
        """
        """预处理输入图像。

        参数：
            rgb_img (torch.Tensor): RGB图像光器。
                                    形状: (N，H，W，3)
            depth_img (torch.Tensor): 深度图像子。
                                      形状: (N，H，W，1)
            segmentation_img (torch.Tensor): 细分图像子。
                                             形状: (N，H，W，3)

        返回：
            元组[torch.Tensor，torch.Tensor，torch.Tensor]:预加工 RGB，深度和细分
        """
        rgb_img = rgb_img / 255.0
        # process depth image
        depth_img[depth_img == float("inf")] = 0
        depth_img /= 5.0
        depth_img /= torch.max(depth_img)
        # process segmentation image
        segmentation_img = segmentation_img / 255.0
        mean_tensor = torch.mean(segmentation_img, dim=(1, 2), keepdim=True)
        segmentation_img -= mean_tensor
        return rgb_img, depth_img, segmentation_img

    def _save_images(self, rgb_img: torch.Tensor, depth_img: torch.Tensor, segmentation_img: torch.Tensor):
        """Writes image buffers to file.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor): Depth image tensor. Shape: (N, H, W, 1).
            segmentation_img (torch.Tensor): Segmentation image tensor. Shape: (N, H, W, 3).
        """
        """编写图像缓冲文件。

        参数：
            rgb_img (torch.Tensor): RGB图像光器。
                                    形状: (N，H，W，3)
            depth_img (torch.Tensor): 深度图像子。
                                      形状: (N，H，W，1)
            segmentation_img (torch.Tensor): 细分图像子。
                                             形状: (N，H，W，3)
        """
        save_images_to_file(rgb_img, "shadow_hand_rgb.png")
        save_images_to_file(depth_img, "shadow_hand_depth.png")
        save_images_to_file(segmentation_img, "shadow_hand_segmentation.png")

    def step(
        self, rgb_img: torch.Tensor, depth_img: torch.Tensor, segmentation_img: torch.Tensor, gt_pose: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the features using the images and trains the model if the train flag is set to True.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            depth_img (torch.Tensor): Depth image tensor. Shape: (N, H, W, 1).
            segmentation_img (torch.Tensor): Segmentation image tensor. Shape: (N, H, W, 3).
            gt_pose (torch.Tensor): Ground truth pose tensor (position and corners). Shape: (N, 27).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Pose loss and predicted pose.
        """
        """使用图像提取特征，如果火车旗设置为True，则将模型列车。

        参数：
            rgb_img (torch.Tensor): RGB图像光器。
                                    形状: (N，H，W，3)
            depth_img (torch.Tensor): 深度图像子。
                                      形状: (N，H，W，1)
            segmentation_img (torch.Tensor): 细分图像子。
                                             形状: (N，H，W，3)
            gt_pose (torch.Tensor): 地面真相呈现度 (位置和角落)。
                                    形状: (N，27)。

        返回：
            双重[torch.Tensor，torch.Tensor]: 姿势损失和预测姿势。
        """

        rgb_img, depth_img, segmentation_img = self._preprocess_images(rgb_img, depth_img, segmentation_img)

        if self.cfg.write_image_to_file:
            self._save_images(rgb_img, depth_img, segmentation_img)

        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    img_input = torch.cat((rgb_img, depth_img, segmentation_img), dim=-1)
                    self.optimizer.zero_grad()

                    predicted_pose = self.feature_extractor(img_input)
                    pose_loss = self.l2_loss(predicted_pose, gt_pose.clone()) * 100

                    pose_loss.backward()
                    self.optimizer.step()

                    if self.step_count % 50000 == 0:
                        torch.save(
                            self.feature_extractor.state_dict(),
                            os.path.join(self.log_dir, f"cnn_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                        )

                    self.step_count += 1

                    return pose_loss, predicted_pose
        else:
            img_input = torch.cat((rgb_img, depth_img, segmentation_img), dim=-1)
            predicted_pose = self.feature_extractor(img_input)
            return None, predicted_pose
