#!/usr/bin/env python3
"""
Depth Estimation Evaluation Script
使用训练好的 LoRA 模型在 DIODE/ETH3D/NYUv2/KITTI/ScanNet 数据集上进行深度估计评估

改进版：
- 使用最小二乘法对齐 (Disparity2Depth)
- 完整的指标集 (a1, a2, a3, abs_rel, rmse, sq_rel, rmse_log, silog, log_10)
- 支持 Garg/Eigen crop
- 改进可视化
"""

import argparse
import os
import json
import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import cv2
from matplotlib import cm

# The depth engine is imported only when inference is requested.



def cudagc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def read_depth_sintel(filename):
    """
    Read Sintel depth file (.dpt format)

    NOTE: We do NOT clip to 72m here. Sky/background pixels with very large depth
    values (e.g. > 2000m) are instead excluded by the valid mask in evaluation
    (depth < max_depth, typically 80m). Clipping to 72m caused all sky pixels to
    appear as valid 72m pixels, inflating AbsRel dramatically (e.g. 29→0.37).
    """
    TAG_FLOAT = 202021.25
    with open(filename, 'rb') as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert check == TAG_FLOAT, f'depth_read:: Wrong tag in flow file (should be: {TAG_FLOAT}, is: {check})'
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        size = width * height
        assert width > 0 and height > 0 and size > 1 and size < 100000000
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


# ============================================================
# 指标计算类和函数 (参考 DepthAnything-AC)
# ============================================================

class RunningAverage:
    """运行平均值计算器"""
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    """多指标运行平均值"""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return
        if self._dict is None:
            self._dict = {key: RunningAverage() for key in new_dict.keys()}
        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return {}
        return {key: value.get_value() for key, value in self._dict.items()}


class Disparity2Depth:
    """
    使用最小二乘法将预测视差转换为对齐的深度
    通过计算最优 scale 和 shift 参数来对齐预测和目标
    """
    def __init__(self, threshold=1.25, depth_cap=80):
        self.threshold = threshold
        self.depth_cap = depth_cap

    def compute_scale_and_shift(self, prediction, target, mask):
        """
        计算最优 scale 和 shift 参数
        解决最小二乘问题: min ||scale * pred + shift - target||^2
        """
        # 系统矩阵: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # 右侧: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # 解: x = A^-1 . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

    def __call__(self, prediction, target, mask):
        """
        将预测视差转换为对齐的深度

        Args:
            prediction: 预测视差 (B, H, W) 或 (H, W)
            target: 目标深度 (B, H, W) 或 (H, W)
            mask: 有效像素掩码 (B, H, W) 或 (H, W)

        Returns:
            对齐后的深度预测
        """
        # 确保是3D张量
        if prediction.dim() == 2:
            prediction = prediction.unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # 将目标深度转换为视差
        target_disparity = torch.zeros_like(target)
        target_disparity[mask == 1] = 1.0 / (target[mask == 1] + 1e-8)

        # 计算 scale 和 shift
        scale, shift = self.compute_scale_and_shift(prediction, target_disparity, mask)

        # 应用变换
        prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        # 深度限制
        disparity_cap = 1.0 / self.depth_cap
        prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

        # 转换回深度
        prediction_depth = 1.0 / prediction_aligned

        return prediction_depth.squeeze()


def compute_errors(gt, pred):
    """
    计算完整的深度估计指标

    Args:
        gt: Ground truth 深度 (numpy array)
        pred: 预测深度 (numpy array)

    Returns:
        dict: 包含所有指标的字典
    """
    eps = 1e-8

    thresh = np.maximum((gt / (pred + eps)), (pred / (gt + eps)))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / (gt + eps))
    sq_rel = np.mean(((gt - pred) ** 2) / (gt + eps))

    rmse = np.sqrt(np.mean((gt - pred) ** 2))

    # 避免 log 对负数/零的操作
    gt_log = np.log(np.maximum(gt, eps))
    pred_log = np.log(np.maximum(pred, eps))

    rmse_log = np.sqrt(np.mean((gt_log - pred_log) ** 2))

    err = pred_log - gt_log
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = np.mean(np.abs(np.log10(np.maximum(gt, eps)) - np.log10(np.maximum(pred, eps))))

    return dict(
        a1=a1, a2=a2, a3=a3,
        abs_rel=abs_rel, sq_rel=sq_rel,
        rmse=rmse, rmse_log=rmse_log,
        silog=silog, log_10=log_10
    )


def apply_crop(depth, mask, dataset='kitti', crop_type='garg'):
    """
    应用数据集特定的裁剪

    Args:
        depth: 深度图 (H, W)
        mask: 有效掩码 (H, W)
        dataset: 数据集名称
        crop_type: 裁剪类型 ('garg', 'eigen', 'none')

    Returns:
        裁剪后的 mask
    """
    if crop_type == 'none':
        return mask

    h, w = depth.shape
    eval_mask = np.zeros_like(mask)

    if crop_type == 'garg':
        # Garg crop (适用于 KITTI)
        eval_mask[int(0.40810811 * h):int(0.99189189 * h),
                  int(0.03594771 * w):int(0.96405229 * w)] = 1
    elif crop_type == 'eigen':
        if dataset in ['nyuv2', 'scannet'] and h == 480 and w == 640:
            # NYUv2/ScanNet 固定坐标 (480x640)
            eval_mask[45:471, 41:601] = 1
        else:
            # 其他数据集使用比例计算 (KITTI, RobotCar, etc.)
            eval_mask[int(0.3324324 * h):int(0.91351351 * h),
                      int(0.0359477 * w):int(0.96405229 * w)] = 1

    return mask * eval_mask


def compute_metrics(gt, pred, mask, dataset='kitti',
                   min_depth_eval=0.1, max_depth_eval=80,
                   crop_type='none', use_least_squares=True):
    """
    计算深度估计指标（完整版）

    Args:
        gt: Ground truth 深度 (torch.Tensor)
        pred: 预测深度 (torch.Tensor)
        mask: 有效掩码 (torch.Tensor)
        dataset: 数据集名称
        min_depth_eval: 最小评估深度
        max_depth_eval: 最大评估深度
        crop_type: 裁剪类型
        use_least_squares: 是否使用最小二乘法对齐

    Returns:
        dict: 包含所有指标的字典
    """
    # 转换为 numpy
    pred = pred.squeeze().cpu().numpy()
    gt_depth = gt.squeeze().cpu().numpy()
    valid_mask = mask.squeeze().cpu().numpy()

    # 限制深度范围
    pred = np.clip(pred, min_depth_eval, max_depth_eval)
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    # 应用裁剪
    if crop_type != 'none':
        valid_mask = apply_crop(gt_depth, valid_mask, dataset, crop_type)

    # 深度范围内的有效掩码
    valid_mask = valid_mask.astype(bool) & (gt_depth > min_depth_eval) & (gt_depth < max_depth_eval)

    if valid_mask.sum() < 100:
        return None

    return compute_errors(gt_depth[valid_mask], pred[valid_mask])


def fix_sky_depth(depth_pred, rgb_image, sky_threshold=0.7, top_ratio=0.4):
    """
    修正天空区域的深度值（后处理）

    天空区域通常位于图像上部，且在 RGB 图像中是高亮度区域。
    检测这些区域，强制将其深度值设为最大值（最远）。

    Args:
        depth_pred: 模型预测的深度图 (H, W), 范围 [0, 1]
        rgb_image: 原始 RGB 图像 (H, W, 3), 范围 [0, 255]
        sky_threshold: 亮度阈值，超过此值认为是天空
        top_ratio: 只检查图像上部的比例 (默认 40%)

    Returns:
        修正后的深度图
    """
    depth_fixed = depth_pred.copy()
    h = rgb_image.shape[0]
    top_h = int(h * top_ratio)

    # 只检查图像上部区域（天空通常在这里）
    top_region = rgb_image[:top_h, :]

    # 计算每个像素的亮度（RGB 平均值，归一化到 [0, 1]）
    if len(top_region.shape) == 3:
        brightness = top_region.mean(axis=2) / 255.0
    else:
        brightness = top_region / 255.0

    # 亮度超过阈值的区域被认为是天空
    sky_mask = brightness > sky_threshold

    # 同时检查深度预测是否异常低（应该远但预测为近）
    top_depth = depth_fixed[:top_h, :]

    # 计算天空区域应该修正的像素（亮度高但深度预测值低于中位数）
    median_depth = np.median(depth_fixed)
    incorrect_sky = sky_mask & (top_depth < median_depth)

    # 将不正确的天空区域深度设为 1.0（最远）
    depth_fixed[:top_h][incorrect_sky] = 1.0

    # 返回修正后的深度图和修正的像素数
    num_fixed = incorrect_sky.sum()

    return depth_fixed, num_fixed


def colorize_depth(depth, cmap='Spectral_r', vmin=None, vmax=None):
    """
    将深度图转换为彩色可视化

    Args:
        depth: 深度图 (H, W)
        cmap: colormap 名称
        vmin, vmax: 深度范围

    Returns:
        彩色深度图 (H, W, 3) uint8
    """
    if vmin is None:
        vmin = depth.min()
    if vmax is None:
        vmax = depth.max()

    depth_norm = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)

    colormap = cm.get_cmap(cmap)
    depth_color = (colormap(depth_norm) * 255).astype(np.uint8)[:, :, :3]

    return depth_color


def collect_diode_samples(base_dir, indoor_only=True):
    """Collect DIODE samples

    Args:
        base_dir: Path to DIODE val directory
        indoor_only: If True, only collect indoor samples (default: True)
    """
    samples = []

    # Only indoor if indoor_only=True
    envs = ["indoors"] if indoor_only else ["indoors", "outdoor"]

    for env in envs:
        env_dir = os.path.join(base_dir, env)
        if not os.path.exists(env_dir):
            continue

        for scene in os.listdir(env_dir):
            scene_dir = os.path.join(env_dir, scene)
            if not os.path.isdir(scene_dir):
                continue

            for scan in os.listdir(scene_dir):
                scan_dir = os.path.join(scene_dir, scan)
                if not os.path.isdir(scan_dir):
                    continue

                for f in os.listdir(scan_dir):
                    if f.endswith('.png') and '_depth' not in f and '_mask' not in f:
                        rgb_path = os.path.join(scan_dir, f)
                        depth_path = rgb_path.replace('.png', '_depth.npy')
                        mask_path = rgb_path.replace('.png', '_depth_mask.npy')

                        if os.path.exists(depth_path):
                            samples.append({
                                'rgb': rgb_path,
                                'depth': depth_path,
                                'mask': mask_path if os.path.exists(mask_path) else None,
                                'name': f"diode_{env}_{f.replace('.png', '')}",
                                'env': env,
                            })

    print(f"  DIODE: {len(samples)} samples (indoor_only={indoor_only})")
    return samples


def collect_kitti_samples(base_dir):
    """Collect all KITTI samples

    KITTI structure (data is in base_dir/kitti/ subdirectory):
    - RGB: base_dir/kitti/2011_09_26/2011_09_26_drive_XXXX_sync/image_02/data/XXXXXXXXXX.png
    - GT:  base_dir/kitti/2011_09_26_drive_XXXX_sync/proj_depth/groundtruth/image_02/XXXXXXXXXX.png

    Note: GT is directly under kitti_root, not under the date folder
    """
    samples = []

    # Check if there's a 'kitti' subdirectory
    kitti_root = base_dir
    if os.path.exists(os.path.join(base_dir, 'kitti')):
        kitti_root = os.path.join(base_dir, 'kitti')

    import glob
    rgb_files = glob.glob(os.path.join(kitti_root, "**/image_02/data/*.png"), recursive=True)

    for rgb_path in rgb_files:
        # Extract drive name (e.g., 2011_09_26_drive_0002_sync)
        parts = rgb_path.split('/')
        drive_name = None
        for p in parts:
            if 'drive' in p and 'sync' in p:
                drive_name = p
                break

        if drive_name:
            filename = parts[-1]
            # GT is directly under kitti_root/drive_name/proj_depth/groundtruth/image_02/
            depth_path = os.path.join(kitti_root, drive_name, 'proj_depth', 'groundtruth', 'image_02', filename)

            # Create unique name using drive and frame
            unique_name = f"{drive_name}_{filename.replace('.png', '')}"

            samples.append({
                'rgb': rgb_path,
                'depth': depth_path if os.path.exists(depth_path) else None,
                'name': unique_name,
            })

    # Filter samples with valid GT
    samples_with_gt = [s for s in samples if s['depth'] is not None]
    print(f"  KITTI: {len(samples)} RGB images, {len(samples_with_gt)} with GT")

    return samples


def collect_scannet_samples(base_dir):
    """Collect all ScanNet samples

    Handles nested structure: base_dir/scannet/scene.../color/
    """
    samples = []

    # Check for nested scannet directory
    scan_dir = base_dir
    if os.path.exists(os.path.join(base_dir, 'scannet')):
        nested = os.path.join(base_dir, 'scannet')
        # Check if nested/scene has color dir
        for item in os.listdir(nested):
            if os.path.isdir(os.path.join(nested, item, 'color')):
                scan_dir = nested
                break

    for scene in os.listdir(scan_dir):
        scene_dir = os.path.join(scan_dir, scene)
        if not os.path.isdir(scene_dir):
            continue

        color_dir = os.path.join(scene_dir, "color")
        depth_dir = os.path.join(scene_dir, "depth")

        if not os.path.exists(color_dir) or not os.path.exists(depth_dir):
            continue

        for f in os.listdir(color_dir):
            if f.endswith('.jpg') or f.endswith('.png'):
                rgb_path = os.path.join(color_dir, f)
                depth_name = f.rsplit('.', 1)[0] + '.png'
                depth_path = os.path.join(depth_dir, depth_name)

                samples.append({
                    'rgb': rgb_path,
                    'depth': depth_path if os.path.exists(depth_path) else None,
                    'name': f'{scene}_{f.rsplit(".", 1)[0]}',
                    'scene': scene,
                })

    print(f"  ScanNet: {len(samples)} samples found")
    return samples


def collect_nyuv2_samples(base_dir):
    """Collect all NYUv2 samples"""
    samples = []

    # 支持两种目录结构: base_dir/test/scene 或 base_dir/scene
    test_dir = os.path.join(base_dir, "test")
    if os.path.exists(test_dir):
        scan_dir = test_dir
    else:
        scan_dir = base_dir

    for scene in os.listdir(scan_dir):
        scene_dir = os.path.join(scan_dir, scene)
        if not os.path.isdir(scene_dir):
            continue

        for f in os.listdir(scene_dir):
            if f.startswith('rgb_') and f.endswith('.png'):
                rgb_path = os.path.join(scene_dir, f)
                depth_name = f.replace('rgb_', 'depth_')
                depth_path = os.path.join(scene_dir, depth_name)

                samples.append({
                    'rgb': rgb_path,
                    'depth': depth_path if os.path.exists(depth_path) else None,
                    'name': f"{scene}_{f.replace('rgb_', '').replace('.png', '')}",
                    'scene': scene,
                })

    return samples


def collect_eth3d_samples(base_dir):
    """Collect all ETH3D samples

    ETH3D structure:
    - RGB: base_dir/rgb/{scene}/images/dslr_images/{name}.JPG
    - GT:  base_dir/depth/{scene}_dslr_depth/{scene}/ground_truth_depth/dslr_images/{name}.JPG

    Note: GT files are float32 binary (6048x4032), stored as .JPG extension
    """
    samples = []

    rgb_base = os.path.join(base_dir, "rgb")
    depth_base = os.path.join(base_dir, "depth")

    if not os.path.exists(rgb_base):
        print(f"  ETH3D: rgb directory not found at {rgb_base}")
        return samples

    for scene in os.listdir(rgb_base):
        scene_rgb_dir = os.path.join(rgb_base, scene, "images", "dslr_images")
        if not os.path.isdir(scene_rgb_dir):
            continue

        # GT depth path: depth/{scene}_dslr_depth/{scene}/ground_truth_depth/dslr_images/
        scene_depth_dir = os.path.join(
            depth_base,
            f"{scene}_dslr_depth",
            scene,
            "ground_truth_depth",
            "dslr_images"
        )

        for f in os.listdir(scene_rgb_dir):
            if f.endswith('.JPG') or f.endswith('.jpg'):
                rgb_path = os.path.join(scene_rgb_dir, f)
                # GT has same filename as RGB (both .JPG)
                depth_path = os.path.join(scene_depth_dir, f)

                samples.append({
                    'rgb': rgb_path,
                    'depth': depth_path if os.path.exists(depth_path) else None,
                    'name': f"{scene}_{f.rsplit('.', 1)[0]}",
                    'scene': scene,
                    'dataset_type': 'eth3d',  # Mark for special GT loading
                })

    samples_with_gt = [s for s in samples if s['depth'] is not None]
    print(f"  ETH3D: {len(samples)} RGB images, {len(samples_with_gt)} with GT")

    return samples


def collect_sintel_samples(base_dir):
    """Collect all Sintel samples

    Sintel structure (uses partition file format):
    - RGB: base_dir/training/final/{scene}/frame_XXXX.png
    - GT:  base_dir/training/depth/{scene}/frame_XXXX.dpt
    """
    samples = []

    # Check for training subdirectory
    training_dir = base_dir
    if os.path.exists(os.path.join(base_dir, 'training')):
        training_dir = os.path.join(base_dir, 'training')

    final_dir = os.path.join(training_dir, 'final')
    depth_dir = os.path.join(training_dir, 'depth')

    if not os.path.exists(final_dir):
        print(f"  Sintel: final directory not found at {final_dir}")
        return samples

    for scene in os.listdir(final_dir):
        scene_rgb_dir = os.path.join(final_dir, scene)
        if not os.path.isdir(scene_rgb_dir):
            continue

        scene_depth_dir = os.path.join(depth_dir, scene)

        for f in os.listdir(scene_rgb_dir):
            if f.endswith('.png'):
                rgb_path = os.path.join(scene_rgb_dir, f)
                depth_name = f.replace('.png', '.dpt')
                depth_path = os.path.join(scene_depth_dir, depth_name)

                samples.append({
                    'rgb': rgb_path,
                    'depth': depth_path if os.path.exists(depth_path) else None,
                    'name': f"{scene}_{f.replace('.png', '')}",
                    'scene': scene,
                    'dataset_type': 'sintel',
                })

    samples_with_gt = [s for s in samples if s['depth'] is not None]
    print(f"  Sintel: {len(samples)} RGB images, {len(samples_with_gt)} with GT")

    return samples


def collect_drivingstereo_samples(base_dir, weather='foggy'):
    """Collect DrivingStereo samples (foggy/cloudy/rainy)

    Structure (from partition file):
    - RGB: base_dir/{weather}/{weather}/right-image-half-size/XXX.jpg
    - GT:  base_dir/{weather}/{weather}/depth-map-half-size/XXX.png
    """
    samples = []

    weather_dir = os.path.join(base_dir, weather, weather)
    if not os.path.exists(weather_dir):
        # Try alternative structure
        weather_dir = os.path.join(base_dir, weather)

    rgb_dir = os.path.join(weather_dir, 'right-image-half-size')
    depth_dir = os.path.join(weather_dir, 'depth-map-half-size')

    if not os.path.exists(rgb_dir):
        print(f"  DrivingStereo-{weather}: RGB directory not found at {rgb_dir}")
        return samples

    for f in os.listdir(rgb_dir):
        if f.endswith('.jpg') or f.endswith('.png'):
            rgb_path = os.path.join(rgb_dir, f)
            depth_name = f.replace('.jpg', '.png')
            depth_path = os.path.join(depth_dir, depth_name)

            samples.append({
                'rgb': rgb_path,
                'depth': depth_path if os.path.exists(depth_path) else None,
                'name': f"{weather}_{f.rsplit('.', 1)[0]}",
                'dataset_type': 'drivingstereo',
            })

    samples_with_gt = [s for s in samples if s['depth'] is not None]
    print(f"  DrivingStereo-{weather}: {len(samples)} RGB images, {len(samples_with_gt)} with GT")

    return samples


def collect_robotcar_samples(base_dir):
    """Collect Oxford RobotCar samples using AC partition file for exact alignment

    AC partition format: rgb_path gt_path (space separated)
    - RGB: rgb/XXX.png
    - GT: gt/XXX.npy

    Our structure:
    - RGB: base_dir/test/rgb/XXX.png
    - GT: base_dir/test/gt/XXX.npy
    """
    samples = []

    # Read AC partition file for exact alignment
    partition_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'DepthAnythingAC-master', 'partitions', 'robotcar', 'val.txt')

    # Try test subdirectory first
    test_dir = os.path.join(base_dir, 'test')
    if not os.path.exists(test_dir):
        test_dir = base_dir

    if os.path.exists(partition_file):
        # Use AC partition for exact alignment
        with open(partition_file, 'r') as f:
            lines = f.readlines()

        print(f"  Loading {len(lines)} samples from AC partition")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split(' ')
            if len(parts) < 2:
                continue

            ac_rgb_path, ac_gt_path = parts[0], parts[1]

            # Map to our paths
            rgb_path = os.path.join(test_dir, ac_rgb_path)
            gt_path = os.path.join(test_dir, ac_gt_path)

            filename = os.path.basename(ac_rgb_path)

            if os.path.exists(rgb_path):
                samples.append({
                    'rgb': rgb_path,
                    'depth': gt_path if os.path.exists(gt_path) else None,
                    'name': filename.rsplit('.', 1)[0],
                    'dataset_type': 'robotcar',
                })
    else:
        # Fallback: scan directory
        print(f"  AC partition not found, scanning directory")
        rgb_dir = os.path.join(test_dir, 'rgb')
        gt_dir = os.path.join(test_dir, 'gt')

        if not os.path.exists(rgb_dir):
            print(f"  RobotCar: RGB directory not found at {rgb_dir}")
            return samples

        for f in os.listdir(rgb_dir):
            if f.endswith('.png') or f.endswith('.jpg'):
                rgb_path = os.path.join(rgb_dir, f)
                depth_name = f.rsplit('.', 1)[0] + '.npy'
                depth_path = os.path.join(gt_dir, depth_name)

                samples.append({
                    'rgb': rgb_path,
                    'depth': depth_path if os.path.exists(depth_path) else None,
                    'name': f.rsplit('.', 1)[0],
                    'dataset_type': 'robotcar',
                })

    samples_with_gt = [s for s in samples if s['depth'] is not None]
    print(f"  RobotCar: {len(samples)} RGB images, {len(samples_with_gt)} with GT")

    return samples


def collect_robotcar_night_samples(base_dir):
    """Collect Oxford RobotCar Night samples (night scenes)

    Structure (same as robotcar but for night):
    - RGB: base_dir/test_night/rgb/XXX.png
    - GT: base_dir/test_night/gt/XXX.npy
    Or use robotcar_night partition if available.
    """
    samples = []

    partition_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'DepthAnythingAC-master', 'partitions', 'robotcar_night', 'val.txt')

    test_dir = os.path.join(base_dir, 'test_night')
    if not os.path.exists(test_dir):
        # Fallback: when dataset_dir=robotcar, use test/ (same structure as robotcar daytime)
        test_dir = os.path.join(base_dir, 'test')
    if not os.path.exists(test_dir):
        test_dir = base_dir

    if os.path.exists(partition_file):
        with open(partition_file, 'r') as f:
            lines = f.readlines()
        print(f"  Loading {len(lines)} samples from robotcar_night partition")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            if len(parts) < 2:
                continue
            ac_rgb_path, ac_gt_path = parts[0], parts[1]
            rgb_path = os.path.join(test_dir, ac_rgb_path)
            gt_path = os.path.join(test_dir, ac_gt_path)
            filename = os.path.basename(ac_rgb_path)
            if os.path.exists(rgb_path):
                samples.append({
                    'rgb': rgb_path,
                    'depth': gt_path if os.path.exists(gt_path) else None,
                    'name': filename.rsplit('.', 1)[0],
                    'dataset_type': 'robotcar',
                })
    else:
        rgb_dir = os.path.join(test_dir, 'rgb')
        gt_dir = os.path.join(test_dir, 'gt')
        if not os.path.exists(rgb_dir):
            print(f"  RobotCar-Night: RGB directory not found at {rgb_dir}")
            return samples
        for f in os.listdir(rgb_dir):
            if f.endswith('.png') or f.endswith('.jpg'):
                rgb_path = os.path.join(rgb_dir, f)
                depth_name = f.rsplit('.', 1)[0] + '.npy'
                depth_path = os.path.join(gt_dir, depth_name)
                samples.append({
                    'rgb': rgb_path,
                    'depth': depth_path if os.path.exists(depth_path) else None,
                    'name': f.rsplit('.', 1)[0],
                    'dataset_type': 'robotcar',
                })

    samples_with_gt = [s for s in samples if s['depth'] is not None]
    print(f"  RobotCar-Night: {len(samples)} RGB images, {len(samples_with_gt)} with GT")
    return samples


def collect_nuscene_samples(base_dir):
    """Collect nuScenes samples (night scenes)

    Actual structure:
    - RGB: base_dir/test_night/color/XXX.jpg
    - GT:  base_dir/test_night/gt/XXX.npy
    """
    samples = []

    # Try test_night subdirectory first
    test_dir = os.path.join(base_dir, 'test_night')
    if not os.path.exists(test_dir):
        test_dir = base_dir

    rgb_dir = os.path.join(test_dir, 'color')
    gt_dir = os.path.join(test_dir, 'gt')

    # Fallback to color_denoise if color doesn't exist
    if not os.path.exists(rgb_dir):
        rgb_dir = os.path.join(test_dir, 'color_denoise')

    if not os.path.exists(rgb_dir):
        print(f"  nuScenes: RGB directory not found at {rgb_dir}")
        return samples

    for f in os.listdir(rgb_dir):
        if f.endswith('.jpg') or f.endswith('.png'):
            rgb_path = os.path.join(rgb_dir, f)
            depth_name = f.rsplit('.', 1)[0] + '.npy'
            depth_path = os.path.join(gt_dir, depth_name)

            samples.append({
                'rgb': rgb_path,
                'depth': depth_path if os.path.exists(depth_path) else None,
                'name': f.rsplit('.', 1)[0],
                'dataset_type': 'nuscene',
            })

    samples_with_gt = [s for s in samples if s['depth'] is not None]
    print(f"  nuScenes: {len(samples)} RGB images, {len(samples_with_gt)} with GT")

    return samples


def _depth_eval_data_root_from_kitti_c_base(base_dir):
    """Resolve datasets/eval/depth from KITTI-C base_dir (e.g. .../kitti_c/kitti_c)."""
    d = os.path.abspath(base_dir)
    for _ in range(5):
        if os.path.isdir(os.path.join(d, 'kitti', 'kitti')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(os.path.abspath(base_dir))


def collect_kitti_c_samples(base_dir, corruption='fog', severity_levels=None):
    """Collect KITTI-C corrupted samples using AC partition files for exact alignment

    AC partition format: rgb_path gt_path (space separated)
    - RGB: kitti_c/{corruption}/{severity}/kitti_data/{date}/{drive}/image_02/data/*.png
    - GT: kitti_dataset/data_depth_annotated/train/{drive}/proj_depth/groundtruth/image_02/*.png

    Our structure:
    - RGB: datasets/eval/depth/kitti_c/kitti_c/{corruption}/{severity}/...
    - GT: datasets/eval/depth/kitti/kitti/{drive}/proj_depth/groundtruth/...

    Args:
        severity_levels: list of int, e.g. [1,2,3]. None means all levels.
    """
    samples = []

    allowed_severities = set(str(s) for s in severity_levels) if severity_levels else None

    corruption_map = {
        'fog': ('kitti_c_fog', 'fog'),
        'snow': ('kitti_c_snow', 'snow'),
        'dark': ('kitti_c_dark', 'dark'),
        'motion': ('kitti_c_motion', 'motion_blur'),
        'gaussian': ('kitti_c_gaussian', 'gaussian_noise'),
    }

    partition_name, folder_name = corruption_map.get(corruption, (f'kitti_c_{corruption}', corruption))

    partition_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'DepthAnythingAC-master', 'partitions', partition_name, 'val.txt')

    if not os.path.exists(partition_file):
        print(f"  KITTI-C-{corruption}: partition file not found at {partition_file}")
        return collect_kitti_c_samples_fallback(base_dir, corruption, severity_levels=severity_levels)

    data_root = _depth_eval_data_root_from_kitti_c_base(base_dir)

    with open(partition_file, 'r') as f:
        lines = f.readlines()

    if allowed_severities:
        print(f"  Loading from AC partition, filtering severity levels: {sorted(severity_levels)}")
    else:
        print(f"  Loading {len(lines)} samples from AC partition")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split(' ')
        if len(parts) < 2:
            continue

        ac_rgb_path, ac_gt_path = parts[0], parts[1]

        # Partition lines use "kitti_c/{corruption}/{severity}/kitti_data/...".
        # On disk RGB is under datasets/eval/depth/kitti_c/kitti_c/{corruption}/...
        # Joining base_dir (often .../kitti_c/kitti_c) with full ac_rgb_path would
        # incorrectly produce .../kitti_c/kitti_c/kitti_c/...
        rel_rgb = ac_rgb_path
        if rel_rgb.startswith('kitti_c/'):
            rel_rgb = rel_rgb[len('kitti_c/') :]
        try:
            severity = rel_rgb.split('/')[1]  # dark/1/kitti_data/... -> 1
        except Exception:
            severity = 'x'

        if allowed_severities and severity not in allowed_severities:
            continue

        our_rgb_path = os.path.join(base_dir, rel_rgb)
        if not os.path.exists(our_rgb_path):
            # Alternate layout: base_dir = .../depth/kitti_c (single kitti_c folder)
            alt = os.path.join(base_dir, 'kitti_c', rel_rgb)
            if os.path.exists(alt):
                our_rgb_path = alt
            else:
                our_rgb_path = os.path.join(data_root, 'kitti_c', 'kitti_c', rel_rgb)

        our_gt_path = ac_gt_path.replace('kitti_dataset/data_depth_annotated/train/', 'kitti/kitti/')
        our_gt_path = our_gt_path.replace('kitti_dataset/data_depth_annotated/val/', 'kitti/kitti/')
        our_gt_path = os.path.join(data_root, our_gt_path)

        filename = os.path.basename(ac_rgb_path)

        rgb_exists = os.path.exists(our_rgb_path)
        gt_exists = os.path.exists(our_gt_path)

        if rgb_exists:
            samples.append({
                'rgb': our_rgb_path,
                'depth': our_gt_path if gt_exists else None,
                'name': f"kitti_c_{corruption}_s{severity}_{filename.rsplit('.', 1)[0]}",
                'dataset_type': 'kitti_c',
            })

    samples_with_gt = [s for s in samples if s['depth'] is not None]
    sev_str = f" (severity {sorted(severity_levels)})" if severity_levels else ""
    print(f"  KITTI-C-{corruption}{sev_str}: {len(samples)} RGB images, {len(samples_with_gt)} with GT")

    return samples


def collect_kitti_c_samples_fallback(base_dir, corruption='fog', severity_levels=None):
    """Fallback: Collect KITTI-C samples by scanning directories

    Args:
        severity_levels: list of int, e.g. [1,2,3]. None means all levels 1-5.
    """
    samples = []

    corruption_map = {
        'fog': 'fog',
        'snow': 'snow',
        'dark': 'dark',
        'motion': 'motion_blur',
        'gaussian': 'gaussian_noise',
    }

    actual_corruption = corruption_map.get(corruption, corruption)

    kitti_c_dir = base_dir
    if os.path.exists(os.path.join(base_dir, 'kitti_c')):
        kitti_c_dir = os.path.join(base_dir, 'kitti_c')

    corruption_dir = os.path.join(kitti_c_dir, actual_corruption)
    if not os.path.exists(corruption_dir):
        print(f"  KITTI-C-{corruption}: directory not found at {corruption_dir}")
        return samples

    scan_levels = [str(s) for s in severity_levels] if severity_levels else ['1', '2', '3', '4', '5']
    rgb_files = []

    for severity in scan_levels:
        severity_dir = os.path.join(corruption_dir, severity, 'kitti_data')
        if os.path.exists(severity_dir):
            found_files = glob.glob(os.path.join(severity_dir, "**/image_02/data/*.png"), recursive=True)
            if found_files:
                rgb_files.extend(found_files)
                print(f"  Severity {severity}: {len(found_files)} images")

    kitti_gt_base = os.path.join(_depth_eval_data_root_from_kitti_c_base(base_dir), 'kitti', 'kitti')

    for rgb_path in rgb_files:
        filename = os.path.basename(rgb_path)
        parts = rgb_path.split('/')
        try:
            kitti_idx = parts.index('kitti_data')
            severity = parts[kitti_idx - 1]
            drive = parts[kitti_idx + 2]
            gt_path = os.path.join(kitti_gt_base, drive, 'proj_depth', 'groundtruth', 'image_02', filename)
            if not os.path.exists(gt_path):
                gt_path = None
        except (ValueError, IndexError):
            severity = 'x'
            gt_path = None

        samples.append({
            'rgb': rgb_path,
            'depth': gt_path,
            'name': f"kitti_c_{corruption}_s{severity}_{filename.rsplit('.', 1)[0]}",
            'dataset_type': 'kitti_c',
        })

    samples_with_gt = [s for s in samples if s['depth'] is not None]
    print(f"  KITTI-C-{corruption} (fallback): {len(samples)} RGB, {len(samples_with_gt)} with GT")

    return samples


def collect_da2k_samples(base_dir, variant='original'):
    """Collect DA-2K samples with point pair annotations

    DA-2K uses relative depth comparison (which point is closer)
    Structure:
    - DA-2K/images/{category}/*.jpg
    - DA-2K/annotations.json

    For perturbed variants:
    - DA-2K/images_dark, images_fog, images_snow, images_blur

    Args:
        base_dir: Path to DA-2K directory
        variant: 'original', 'dark', 'fog', 'snow', 'blur'
    """
    samples = []

    # Find annotations.json
    annotations_path = os.path.join(base_dir, 'annotations.json')
    if not os.path.exists(annotations_path):
        # Try DA-2K subdirectory
        annotations_path = os.path.join(base_dir, 'DA-2K', 'annotations.json')

    if not os.path.exists(annotations_path):
        print(f"  DA-2K: annotations.json not found")
        return samples

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Determine image directory
    if variant == 'original':
        img_dir_name = 'images'
    else:
        img_dir_name = f'images_{variant}'

    base_da2k = os.path.dirname(annotations_path)
    img_base = os.path.join(base_da2k, img_dir_name)

    if not os.path.exists(img_base):
        print(f"  DA-2K-{variant}: image directory not found at {img_base}")
        return samples

    for img_rel_path, point_pairs in annotations.items():
        # Original path: images/category/xxx.jpg
        # For variants, replace 'images' with 'images_{variant}'
        if variant != 'original':
            img_rel_path_variant = img_rel_path.replace('images/', f'{img_dir_name}/')
        else:
            img_rel_path_variant = img_rel_path

        img_path = os.path.join(base_da2k, img_rel_path_variant)

        if not os.path.exists(img_path):
            continue

        # Each sample has multiple point pairs
        for pair_idx, pair in enumerate(point_pairs):
            samples.append({
                'rgb': img_path,
                'point1': pair['point1'],  # [x, y]
                'point2': pair['point2'],  # [x, y]
                'closer_point': pair['closer_point'],  # 'point1' or 'point2'
                'name': f"da2k_{variant}_{Path(img_path).stem}_pair{pair_idx}",
                'dataset_type': 'da2k',
            })

    print(f"  DA-2K-{variant}: {len(samples)} point pairs from {len(annotations)} images")
    return samples


def evaluate_da2k_accuracy(image_gen, samples, args):
    """Evaluate DA-2K using relative depth accuracy

    For each point pair, check if the model correctly predicts which point is closer.
    """
    correct = 0
    total = 0

    # Group samples by image for efficiency
    from collections import defaultdict
    image_samples = defaultdict(list)
    for s in samples:
        image_samples[s['rgb']].append(s)

    print(f"Processing {len(image_samples)} unique images...")

    for rgb_path, img_samples in tqdm(image_samples.items(), desc="DA-2K Eval"):
        try:
            ref_image = Image.open(rgb_path).convert("RGB")
            orig_w, orig_h = ref_image.size

            # Run inference
            output_image = image_gen.generate_depth_discriminative(
                ref_images=ref_image,
                seed=args.seed,
                size_level=args.size_level,
                prompt=args.prompt,
                timestep=args.discriminative_timestep,
                show_progress=False,
            )

            if output_image is None:
                continue

            # Convert to depth (grayscale, lower value = closer)
            pred_depth = np.array(output_image.convert('L'), dtype=np.float32)
            pred_h, pred_w = pred_depth.shape

            # Scale coordinates if image was resized
            scale_x = pred_w / orig_w
            scale_y = pred_h / orig_h

            for s in img_samples:
                # Point coordinates are [x, y] format
                x1, y1 = s['point1']
                x2, y2 = s['point2']

                # Scale to prediction size
                x1_s, y1_s = int(x1 * scale_x), int(y1 * scale_y)
                x2_s, y2_s = int(x2 * scale_x), int(y2 * scale_y)

                # Clamp to valid range
                x1_s = min(max(x1_s, 0), pred_w - 1)
                y1_s = min(max(y1_s, 0), pred_h - 1)
                x2_s = min(max(x2_s, 0), pred_w - 1)
                y2_s = min(max(y2_s, 0), pred_h - 1)

                # Get depth values (Lotus encoding: higher value = closer)
                d1 = pred_depth[y1_s, x1_s]
                d2 = pred_depth[y2_s, x2_s]

                pred_closer = 'point1' if d1 > d2 else 'point2'

                if pred_closer == s['closer_point']:
                    correct += 1
                total += 1

        except Exception as e:
            print(f"Error processing {rgb_path}: {e}")
            continue

    accuracy = correct / max(total, 1)
    return {'accuracy': accuracy, 'correct': correct, 'total': total}


# ============== Metrics ==============

def abs_relative_difference(output, target, valid_mask=None):
    """AbsRel = mean(|pred - gt| / gt)"""
    abs_relative_diff = torch.abs(output - target) / (target + 1e-8)

    if valid_mask is not None:
        abs_relative_diff = abs_relative_diff * valid_mask
        n = valid_mask.sum()
        n = torch.clamp(n, min=1)
    else:
        n = output.numel()

    return (abs_relative_diff.sum() / n).item()


def threshold_percentage(output, target, threshold_val, valid_mask=None):
    """δ = % of pixels where max(pred/gt, gt/pred) < threshold"""
    d1 = output / (target + 1e-8)
    d2 = target / (output + 1e-8)
    max_d1_d2 = torch.max(d1, d2)

    bit_mat = (max_d1_d2 < threshold_val).float()

    if valid_mask is not None:
        bit_mat = bit_mat * valid_mask
        n = valid_mask.sum()
        n = torch.clamp(n, min=1)
    else:
        n = output.numel()

    return (bit_mat.sum() / n).item()


def align_depth_median(pred, gt, valid_mask=None):
    """Median alignment: scale = median(gt) / median(pred)"""
    if valid_mask is not None:
        m = valid_mask.bool() & torch.isfinite(pred) & torch.isfinite(gt)
        pred_valid = pred[m]
        gt_valid = gt[m]
    else:
        pred_valid = pred.flatten()
        gt_valid = gt.flatten()

    if len(pred_valid) < 10:
        return pred, 1.0

    pred_median = torch.median(pred_valid)
    gt_median = torch.median(gt_valid)

    scale = gt_median / (pred_median + 1e-8)
    aligned_pred = pred * scale

    return aligned_pred, scale.item()


def create_valid_mask(depth, min_depth=1e-3, max_depth=80.0):
    """Create valid mask"""
    valid = (depth > min_depth) & (depth < max_depth) & torch.isfinite(depth)
    return valid.float()


def align_depth_least_squares(pred, gt, valid_mask):
    """
    使用最小二乘法在深度空间直接对齐预测和GT
    公式: gt = scale * pred + shift
    参考: Lotus evaluation/util/alignment.py

    注意: 使用 float64 精度计算最小二乘，避免大规模数据时的数值不稳定
    """
    pred = pred.float().squeeze()
    gt = gt.float().squeeze()
    mask = valid_mask.float().squeeze().bool()

    # 获取有效区域的值 - 使用 float64 精度以确保数值稳定性
    pred_masked = pred[mask].reshape(-1, 1).double()
    gt_masked = gt[mask].reshape(-1, 1).double()

    # 构建 A = [pred, 1]，求解 gt = scale * pred + shift
    ones = torch.ones_like(pred_masked)
    A = torch.cat([pred_masked, ones], dim=-1)

    # 最小二乘解 (使用 float64 精度)
    result = torch.linalg.lstsq(A, gt_masked)
    X = result.solution
    scale, shift = X[0, 0].float(), X[1, 0].float()

    # 对齐
    aligned = pred * scale + shift
    return aligned


def align_depth_disparity_least_squares(pred, gt, valid_mask):
    """
    在视差空间进行最小二乘对齐 (Lotus 方式)
    公式: 1/gt = scale * pred + shift

    注意: 使用 float64 精度计算最小二乘，避免大规模数据时的数值不稳定
    """
    pred = pred.float().squeeze()
    gt = gt.float().squeeze()
    mask = valid_mask.float().squeeze().bool()

    # 转换 GT 到视差
    gt_disparity = torch.zeros_like(gt)
    gt_disparity[mask] = 1.0 / (gt[mask] + 1e-8)

    # 使用 float64 精度进行最小二乘
    pred_masked = pred[mask].reshape(-1, 1).double()
    gt_disp_masked = gt_disparity[mask].reshape(-1, 1).double()
    ones = torch.ones_like(pred_masked)
    A = torch.cat([pred_masked, ones], dim=-1)

    result = torch.linalg.lstsq(A, gt_disp_masked)
    X = result.solution
    scale, shift = X[0, 0].float(), X[1, 0].float()

    # 对齐并转回深度
    aligned_disp = pred * scale + shift
    aligned_disp = torch.clamp(aligned_disp, min=1e-8)  # 防止除零
    aligned_depth = 1.0 / aligned_disp

    return aligned_depth


def pred_log_stretch(pred, eps=1e-6):
    """
    对数拉伸：压缩近距离、拉伸远距离。
    公式: out = -log(1 - pred + eps)
    效果: 0.5→0.69, 0.9→2.30, 0.98→3.91, 0.99→4.61, 0.995→5.30
    让 0.98/0.99/0.995 等饱和值更分散，有利于对齐时区分远距离。
    """
    if torch.is_tensor(pred):
        return -torch.log(1.0 - pred + eps)
    else:
        return -np.log(1.0 - np.asarray(pred, dtype=np.float64) + eps).astype(np.float32)


def pred_power_stretch(pred, power=2.0):
    """
    幂次拉伸：让远距离更分散，仍保持 [0,1] 范围。
    公式: out = pred^power
    power=2: 0.98→0.9604, 0.99→0.9801
    power=1.7: 0.98→0.9663, 0.99→0.9831
    """
    if torch.is_tensor(pred):
        return pred ** power
    else:
        return (np.asarray(pred, dtype=np.float64) ** power).astype(np.float32)


def align_depth_log_least_squares(pred, gt, valid_mask, eps=1e-6):
    """
    在 log(depth) 空间做最小二乘对齐，远距离误差更均匀。
    模型: log(gt) = scale * log(pred_safe) + shift，即 aligned = exp(scale*log(pred)+shift)。
    pred: [0,1] 归一化（1=远），先 clamp 再取 log；GT 为米制深度。
    """
    pred = pred.float().squeeze()
    gt = gt.float().squeeze()
    mask = valid_mask.float().squeeze().bool()

    pred_safe = torch.clamp(pred, min=eps)
    gt_safe = torch.clamp(gt, min=eps)
    log_pred = torch.log(pred_safe)
    log_gt = torch.log(gt_safe)

    log_pred_masked = log_pred[mask].reshape(-1, 1).double()
    log_gt_masked = log_gt[mask].reshape(-1, 1).double()
    ones = torch.ones_like(log_pred_masked)
    A = torch.cat([log_pred_masked, ones], dim=-1)

    result = torch.linalg.lstsq(A, log_gt_masked)
    X = result.solution
    scale, shift = X[0, 0].float(), X[1, 0].float()

    aligned_log = scale * log_pred + shift
    aligned = torch.exp(aligned_log)
    return aligned


def save_visualization(rgb_path, pred_depth, gt_depth, mask, output_path,
                      cmap='Spectral_r'):
    """
    保存可视化结果（原图、预测、GT、mask 拼接）
    使用统一的 colormap 和归一化范围，便于视觉比较
    """
    # 读取原图
    img = cv2.imread(rgb_path)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize 到统一大小
    h, w = pred_depth.shape
    img_resized = cv2.resize(img, (w, h))

    # 使用统一的 vmin/vmax 进行归一化（基于 GT 的有效范围）
    valid_gt = gt_depth[mask.astype(bool)]
    if len(valid_gt) > 0:
        vmin = np.percentile(valid_gt, 2)  # 使用2%和98%分位数，避免异常值
        vmax = np.percentile(valid_gt, 98)
    else:
        vmin, vmax = gt_depth.min(), gt_depth.max()

    # 彩色化深度图 - 使用相同的 colormap 和范围
    pred_color = colorize_depth(pred_depth, cmap=cmap, vmin=vmin, vmax=vmax)
    gt_color = colorize_depth(gt_depth, cmap=cmap, vmin=vmin, vmax=vmax)

    # Mask 可视化
    mask_vis = (mask * 255).astype(np.uint8)
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)

    # 拼接
    top = np.hstack([img_resized, pred_color])
    bottom = np.hstack([gt_color, mask_vis])
    combined = np.vstack([top, bottom])

    # 保存
    cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/step1x-edit',
                        help='Path to base model directory')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path to LoRA weights')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['diode', 'eth3d', 'kitti', 'nyuv2', 'scannet',
                                 'sintel', 'robotcar', 'robotcar_night', 'nuscene',
                                 'drive_foggy', 'drive_cloudy', 'drive_rainy',
                                 'kitti_c_fog', 'kitti_c_snow', 'kitti_c_dark',
                                 'kitti_c_motion', 'kitti_c_gaussian',
                                 'da2k', 'da2k_dark', 'da2k_fog', 'da2k_snow', 'da2k_blur'],
                        help='Dataset to evaluate')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for predictions')
    parser.add_argument('--num_steps', type=int, default=28,
                        help='Number of diffusion steps')
    parser.add_argument('--cfg_guidance', type=float, default=4.0,
                        help='CFG guidance scale')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attention', choices=['auto', 'flash', 'torch'], default='auto')
    parser.add_argument('--encoding', choices=['white-near', 'white-far'], default='white-near',
                        help='Checkpoint output convention; epoch016 requires white-far')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples to process')
    parser.add_argument('--size_level', type=int, default=512,
                        help='Size level for inference')
    parser.add_argument('--prompt', type=str,
                        default="",
                        help='Prompt for depth estimation')
    parser.add_argument('--version', type=str, default='v1.0',
                        choices=['v1.0', 'v1.1'])
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip inference, only run evaluation')
    parser.add_argument('--align_method', type=str, default='disparity',
                        choices=['median', 'least_squares', 'disparity', 'log_depth'],
                        help='Depth alignment: median, least_squares (depth), disparity (Lotus), log_depth (log-space LS)')
    parser.add_argument('--crop_type', type=str, default='none',
                        choices=['none', 'garg', 'eigen'],
                        help='Evaluation crop type (garg for KITTI, eigen for NYUv2)')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='Minimum depth for evaluation')
    parser.add_argument('--max_depth', type=float, default=80.0,
                        help='Maximum depth for evaluation')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization results')
    parser.add_argument('--use_depth_mode', action='store_true',
                        help='Use dedicated depth estimation inference (no CFG, simplified)')
    parser.add_argument('--discriminative_mode', action='store_true',
                        help='Use discriminative inference (single-step, no noise, like Lotus-D)')
    parser.add_argument('--discriminative_timestep', type=float, default=1.0,
                        help='Fixed timestep for discriminative mode (default: 1.0)')
    parser.add_argument('--fix_sky', action='store_true',
                        help='Apply sky region post-processing fix for outdoor scenes')
    parser.add_argument('--sky_threshold', type=float, default=0.7,
                        help='Brightness threshold for sky detection (default: 0.7)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    parser.add_argument('--pred_transform', type=str, default=None,
                        choices=[None, 'log_stretch', 'power'],
                        help='Transform pred before alignment: log_stretch=-log(1-pred), power=pred^n (use --pred_power)')
    parser.add_argument('--pred_power', type=float, default=2.0,
                        help='Power for pred_transform=power (default: 2.0, use 1.7 for milder stretch)')
    parser.add_argument('--severity_levels', type=int, nargs='+', default=None,
                        help='KITTI-C severity levels to evaluate (e.g. --severity_levels 1 2 3). Default: all levels 1-5')
    parser.add_argument('--filter_ids_file', type=str, default=None,
                        help='Text file with image IDs (one per line) to filter samples. Only samples whose name ends with a matching ID are kept.')
    args = parser.parse_args()
    if not args.skip_inference and not args.lora_path:
        parser.error("--lora_path is required unless --skip_inference is set")

    # 当 cfg_guidance <= 0 时，自动使用深度估计模式
    if args.cfg_guidance <= 0:
        print(f"[INFO] cfg_guidance={args.cfg_guidance} <= 0, 自动启用 use_depth_mode (无 CFG 推理)")
        args.use_depth_mode = True

    if args.discriminative_mode and args.use_depth_mode:
        parser.error("Choose either --discriminative_mode or --use_depth_mode, not both")
    if not args.discriminative_mode and not args.use_depth_mode:
        args.discriminative_mode = True

    # 数据集特定的默认参数 (根据标准评估配置)
    # 参考: datasets/eval/depth/dataset_config/*.py
    if args.dataset == 'kitti':
        if args.crop_type == 'none':
            args.crop_type = 'garg'  # 默认使用 garg crop
        if args.min_depth == 0.1:
            args.min_depth = 1e-5
        if args.max_depth == 80.0:
            args.max_depth = 80.0
    elif args.dataset == 'nyuv2':
        if args.crop_type == 'none':
            args.crop_type = 'eigen'  # 默认使用 eigen crop
        if args.min_depth == 0.1:
            args.min_depth = 1e-3
        if args.max_depth == 80.0:
            args.max_depth = 10.0
    elif args.dataset == 'diode':
        if args.min_depth == 0.1:
            args.min_depth = 0.6
        if args.max_depth == 80.0:
            args.max_depth = 350.0
    elif args.dataset == 'scannet':
        if args.min_depth == 0.1:
            args.min_depth = 1e-3
        if args.max_depth == 80.0:
            args.max_depth = 10.0
    elif args.dataset == 'eth3d':
        if args.min_depth == 0.1:
            args.min_depth = 0.1  # ETH3D uses 0.1 as min
        if args.max_depth == 80.0:
            args.max_depth = 200.0  # ETH3D uses 200m max
    elif args.dataset == 'sintel':
        if args.min_depth == 0.1:
            args.min_depth = 1e-5
        if args.max_depth == 80.0:
            args.max_depth = 80.0
    elif args.dataset == 'robotcar':
        if args.min_depth == 0.1:
            args.min_depth = 1e-5
        if args.max_depth == 80.0:
            args.max_depth = 80.0
    elif args.dataset == 'robotcar_night':
        if args.min_depth == 0.1:
            args.min_depth = 1e-5
        if args.max_depth == 80.0:
            args.max_depth = 80.0
    elif args.dataset == 'nuscene':
        if args.min_depth == 0.1:
            args.min_depth = 1e-5
        if args.max_depth == 80.0:
            args.max_depth = 80.0
    elif args.dataset.startswith('drive_'):
        # DrivingStereo: foggy, cloudy, rainy
        if args.min_depth == 0.1:
            args.min_depth = 1e-5
        if args.max_depth == 80.0:
            args.max_depth = 80.0
    elif args.dataset.startswith('kitti_c_'):
        # KITTI-C corrupted variants
        if args.min_depth == 0.1:
            args.min_depth = 1e-5
        if args.max_depth == 80.0:
            args.max_depth = 80.0

    os.makedirs(args.output_dir, exist_ok=True)
    pred_dir = os.path.join(args.output_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    # Collect samples
    print(f"Collecting {args.dataset} samples...")
    if args.dataset == 'diode':
        samples = collect_diode_samples(args.dataset_dir)
    elif args.dataset == 'eth3d':
        samples = collect_eth3d_samples(args.dataset_dir)
    elif args.dataset == 'kitti':
        samples = collect_kitti_samples(args.dataset_dir)
    elif args.dataset == 'nyuv2':
        samples = collect_nyuv2_samples(args.dataset_dir)
    elif args.dataset == 'scannet':
        samples = collect_scannet_samples(args.dataset_dir)
    elif args.dataset == 'sintel':
        samples = collect_sintel_samples(args.dataset_dir)
    elif args.dataset == 'robotcar':
        samples = collect_robotcar_samples(args.dataset_dir)
    elif args.dataset == 'robotcar_night':
        samples = collect_robotcar_night_samples(args.dataset_dir)
    elif args.dataset == 'nuscene':
        samples = collect_nuscene_samples(args.dataset_dir)
    elif args.dataset == 'drive_foggy':
        samples = collect_drivingstereo_samples(args.dataset_dir, weather='foggy')
    elif args.dataset == 'drive_cloudy':
        samples = collect_drivingstereo_samples(args.dataset_dir, weather='cloudy')
    elif args.dataset == 'drive_rainy':
        samples = collect_drivingstereo_samples(args.dataset_dir, weather='rainy')
    elif args.dataset == 'kitti_c_fog':
        samples = collect_kitti_c_samples(args.dataset_dir, corruption='fog', severity_levels=args.severity_levels)
    elif args.dataset == 'kitti_c_snow':
        samples = collect_kitti_c_samples(args.dataset_dir, corruption='snow', severity_levels=args.severity_levels)
    elif args.dataset == 'kitti_c_dark':
        samples = collect_kitti_c_samples(args.dataset_dir, corruption='dark', severity_levels=args.severity_levels)
    elif args.dataset == 'kitti_c_motion':
        samples = collect_kitti_c_samples(args.dataset_dir, corruption='motion', severity_levels=args.severity_levels)
    elif args.dataset == 'kitti_c_gaussian':
        samples = collect_kitti_c_samples(args.dataset_dir, corruption='gaussian', severity_levels=args.severity_levels)
    elif args.dataset == 'da2k':
        samples = collect_da2k_samples(args.dataset_dir, variant='original')
    elif args.dataset == 'da2k_dark':
        samples = collect_da2k_samples(args.dataset_dir, variant='dark')
    elif args.dataset == 'da2k_fog':
        samples = collect_da2k_samples(args.dataset_dir, variant='fog')
    elif args.dataset == 'da2k_snow':
        samples = collect_da2k_samples(args.dataset_dir, variant='snow')
    elif args.dataset == 'da2k_blur':
        samples = collect_da2k_samples(args.dataset_dir, variant='blur')

    print(f"Found {len(samples)} samples")

    if args.filter_ids_file:
        with open(args.filter_ids_file, 'r') as f:
            filter_ids = set(line.strip() for line in f if line.strip())
        before = len(samples)
        exact = [s for s in samples if s['name'] in filter_ids]
        if exact:
            samples = exact
            print(f"Filtered by {args.filter_ids_file}: {before} -> {len(samples)} samples (exact name match)")
        else:
            samples = [s for s in samples if any(s['name'].endswith(fid) for fid in filter_ids)]
            print(f"Filtered by {args.filter_ids_file}: {before} -> {len(samples)} samples (suffix match, {len(filter_ids)} IDs)")

    seen_names = set()
    deduped = []
    for s in samples:
        if s['name'] not in seen_names:
            seen_names.add(s['name'])
            deduped.append(s)
    if len(deduped) < len(samples):
        print(f"Deduplicated: {len(samples)} -> {len(deduped)} samples ({len(samples)-len(deduped)} duplicates removed)")
        samples = deduped

    if not samples:
        raise SystemExit("No evaluation samples found; check --dataset_dir and dataset layout")

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Processing first {len(samples)} samples")

    # Run inference
    if not args.skip_inference:
        from opendepth.engine import ImageGenerator
        from modules.attention import flash_attn_func
        attention = args.attention
        if attention == 'auto':
            attention = 'flash' if flash_attn_func is not None else 'torch'
        if attention == 'flash' and flash_attn_func is None:
            raise SystemExit('Flash Attention is unavailable; use --attention torch.')
        torch.manual_seed(args.seed)
        print("Loading models...")

        if args.version == 'v1.0':
            ckpt_name = 'step1x-edit-i1258.safetensors'
        else:
            ckpt_name = 'step1x-edit-v1p1-official.safetensors'

        image_gen = ImageGenerator(
            dit_path=os.path.join(args.model_path, ckpt_name),
            ae_path=os.path.join(args.model_path, 'vae.safetensors'),
            qwen2vl_model_path=os.path.join(args.model_path, 'Qwen2.5-VL-7B-Instruct'),
            max_length=640,
            lora=args.lora_path,
            version=args.version,
            mode=attention,
        )

        cudagc()

        print(f"Running inference (batch_size={args.batch_size})...")

        # Filter out already processed samples
        pending_samples = []
        for sample in samples:
            pred_npy_path = os.path.join(pred_dir, sample['name'] + '.npy')
            if not os.path.exists(pred_npy_path):
                pending_samples.append(sample)

        print(f"  Pending: {len(pending_samples)}/{len(samples)} samples")

        # Batch inference for discriminative mode
        if args.discriminative_mode and args.batch_size > 1:
            # 批量推理
            for batch_start in tqdm(range(0, len(pending_samples), args.batch_size), desc="Inference (batch)"):
                batch_samples = pending_samples[batch_start:batch_start + args.batch_size]

                try:
                    # 加载批量图片
                    batch_images = []
                    batch_rgb_images = []
                    for sample in batch_samples:
                        ref_image = Image.open(sample['rgb']).convert("RGB")
                        batch_images.append(ref_image)
                        batch_rgb_images.append(ref_image)

                    # 批量推理
                    output_images = image_gen.generate_depth_discriminative_batch(
                        ref_images_list=batch_images,
                        seed=args.seed,
                        size_level=args.size_level,
                        prompt=args.prompt,
                        timestep=args.discriminative_timestep,
                        show_progress=False,
                    )

                    # 保存结果
                    for i, (sample, output_image) in enumerate(zip(batch_samples, output_images)):
                        if output_image is not None:
                            output_np = np.array(output_image.convert('L'), dtype=np.float32) / 255.0
                            if args.encoding == 'white-far':
                                output_np = 1.0 - output_np

                            if args.fix_sky:
                                rgb_np = np.array(batch_rgb_images[i])
                                output_np, _ = fix_sky_depth(output_np, rgb_np, sky_threshold=args.sky_threshold)

                            pred_npy_path = os.path.join(pred_dir, sample['name'] + '.npy')
                            pred_png_path = os.path.join(pred_dir, sample['name'] + '.png')
                            np.save(pred_npy_path, output_np)
                            output_img_fixed = Image.fromarray((output_np * 255).astype(np.uint8))
                            output_img_fixed.save(pred_png_path)

                except Exception as e:
                    import traceback
                    print(f"Batch error: {e}")
                    traceback.print_exc()
                    # 回退到单张处理
                    for sample in batch_samples:
                        try:
                            ref_image = Image.open(sample['rgb']).convert("RGB")
                            output_image = image_gen.generate_depth_discriminative(
                                ref_images=ref_image, seed=args.seed, size_level=args.size_level,
                                prompt=args.prompt, timestep=args.discriminative_timestep, show_progress=False,
                            )
                            if output_image is not None:
                                output_np = np.array(output_image.convert('L'), dtype=np.float32) / 255.0
                                if args.encoding == 'white-far':
                                    output_np = 1.0 - output_np
                                if args.fix_sky:
                                    rgb_np = np.array(ref_image)
                                    output_np, _ = fix_sky_depth(output_np, rgb_np, sky_threshold=args.sky_threshold)
                                pred_npy_path = os.path.join(pred_dir, sample['name'] + '.npy')
                                pred_png_path = os.path.join(pred_dir, sample['name'] + '.png')
                                np.save(pred_npy_path, output_np)
                                Image.fromarray((output_np * 255).astype(np.uint8)).save(pred_png_path)
                        except Exception as e2:
                            print(f"Error processing {sample['rgb']}: {e2}")

                # Periodic cleanup
                if batch_start % (args.batch_size * 5) == 0:
                    cudagc()
        else:
            # 单张推理（原逻辑）
            for sample in tqdm(pending_samples, desc="Inference"):
                rgb_path = sample['rgb']
                output_name = sample['name']
                pred_npy_path = os.path.join(pred_dir, output_name + '.npy')
                pred_png_path = os.path.join(pred_dir, output_name + '.png')

                try:
                    ref_image = Image.open(rgb_path).convert("RGB")

                    if args.discriminative_mode:
                        output_image = image_gen.generate_depth_discriminative(
                            ref_images=ref_image, seed=args.seed, size_level=args.size_level,
                            prompt=args.prompt, timestep=args.discriminative_timestep, show_progress=False,
                        )
                    elif args.use_depth_mode:
                        output_image = image_gen.generate_depth(
                            ref_images=ref_image, num_steps=args.num_steps, seed=args.seed,
                            size_level=args.size_level, prompt=args.prompt, show_progress=False,
                        )
                    else:
                        output_image = image_gen.generate_depth(
                            ref_images=ref_image, num_steps=args.num_steps, seed=args.seed,
                            size_level=args.size_level, prompt=args.prompt, show_progress=False,
                        )

                    if output_image is not None:
                        output_np = np.array(output_image.convert('L'), dtype=np.float32) / 255.0
                        if args.encoding == 'white-far':
                            output_np = 1.0 - output_np
                        if args.fix_sky:
                            rgb_np = np.array(ref_image)
                            output_np, _ = fix_sky_depth(output_np, rgb_np, sky_threshold=args.sky_threshold)
                        np.save(pred_npy_path, output_np)
                        Image.fromarray((output_np * 255).astype(np.uint8)).save(pred_png_path)

                except Exception as e:
                    import traceback
                    print(f"Error processing {rgb_path}: {e}")
                    traceback.print_exc()
                    continue

                if pending_samples.index(sample) % 20 == 0:
                    cudagc()

        del image_gen
        cudagc()

    # Special handling for DA2K datasets (point pair comparison)
    if args.dataset.startswith('da2k'):
        print("\n" + "="*60)
        print("Running DA-2K Evaluation (Point Pair Accuracy)...")
        print("="*60)

        # DA2K evaluation: check point pair predictions
        correct = 0
        total = 0

        # Group samples by image
        from collections import defaultdict
        image_samples = defaultdict(list)
        for s in samples:
            image_samples[s['rgb']].append(s)

        for rgb_path, img_samples in tqdm(image_samples.items(), desc="DA-2K Eval"):
            # Find any prediction file for this image
            sample_name_base = img_samples[0]['name'].rsplit('_pair', 1)[0]

            # Load the prediction image
            pred_png_path = None
            for s in img_samples:
                p = os.path.join(pred_dir, s['name'] + '.png')
                if os.path.exists(p):
                    pred_png_path = p
                    break

            if pred_png_path is None:
                continue

            try:
                # Load prediction depth (prefer NPY for better precision)
                pred_npy_path = pred_png_path.replace('.png', '.npy')
                if os.path.exists(pred_npy_path):
                    pred_depth = np.load(pred_npy_path).astype(np.float32)
                else:
                    pred_img = Image.open(pred_png_path).convert('L')
                    pred_depth = np.array(pred_img, dtype=np.float32)
                pred_h, pred_w = pred_depth.shape

                # Get original image size
                orig_img = Image.open(rgb_path)
                orig_w, orig_h = orig_img.size

                scale_x = pred_w / orig_w
                scale_y = pred_h / orig_h

                for s in img_samples:
                    # DA-2K annotations use [y, x] format (same as AC's h, w)
                    y1, x1 = s['point1']
                    y2, x2 = s['point2']

                    # Skip pairs with out-of-bounds coordinates
                    if x1 >= orig_w or y1 >= orig_h or x2 >= orig_w or y2 >= orig_h:
                        continue

                    x1_s = min(max(int(x1 * scale_x), 0), pred_w - 1)
                    y1_s = min(max(int(y1 * scale_y), 0), pred_h - 1)
                    x2_s = min(max(int(x2 * scale_x), 0), pred_w - 1)
                    y2_s = min(max(int(y2 * scale_y), 0), pred_h - 1)

                    d1 = pred_depth[y1_s, x1_s]
                    d2 = pred_depth[y2_s, x2_s]

                    # Lotus encoding: higher value = closer (disparity space)
                    pred_closer = 'point1' if d1 > d2 else 'point2'
                    if pred_closer == s['closer_point']:
                        correct += 1
                    total += 1
            except Exception as e:
                print(f"Error evaluating {rgb_path}: {e}")
                continue

        accuracy = correct / max(total, 1)
        print(f"\n{'='*60}")
        print(f"DA-2K Results:")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"{'='*60}")

        # Save results
        results = {
            'dataset': args.dataset,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }
        results_path = os.path.join(args.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
        return

    # Run evaluation (standard depth metrics)
    print("\n" + "="*60)
    print("Running Evaluation...")
    print(f"  Align method: {args.align_method}")
    if getattr(args, 'pred_transform', None):
        if args.pred_transform == 'power':
            print(f"  Pred transform: power={args.pred_power}")
        else:
            print(f"  Pred transform: {args.pred_transform}")
    print(f"  Crop type: {args.crop_type}")
    print(f"  Depth range: [{args.min_depth}, {args.max_depth}]")
    print("="*60)

    # 使用 RunningAverageDict 收集所有指标
    metrics = RunningAverageDict()
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    eval_count = 0
    for sample in tqdm(samples, desc="Evaluating"):
        pred_npy_path = os.path.join(pred_dir, sample['name'] + '.npy')
        gt_path = sample.get('depth')

        if not os.path.exists(pred_npy_path):
            continue

        if gt_path is None or not os.path.exists(gt_path):
            continue

        try:
            # Load prediction
            pred_depth = np.array(np.load(pred_npy_path), dtype=np.float32)
            pred_depth = torch.tensor(pred_depth, dtype=torch.float32)

            # Optional: 预测变换，拉伸远距离
            if getattr(args, 'pred_transform', None) == 'log_stretch':
                pred_depth = pred_log_stretch(pred_depth)
            elif getattr(args, 'pred_transform', None) == 'power':
                pred_depth = pred_power_stretch(pred_depth, getattr(args, 'pred_power', 2.0))

            # Load GT based on dataset type
            dataset_type = sample.get('dataset_type', args.dataset)

            if dataset_type == 'sintel' and gt_path.endswith('.dpt'):
                # Sintel: .dpt binary format
                gt_depth = read_depth_sintel(gt_path)
                gt_depth = torch.tensor(np.array(gt_depth, dtype=np.float32))
            elif dataset_type == 'eth3d' and gt_path.lower().endswith('.jpg'):
                # ETH3D: float32 binary depth (6048x4032)，扩展名可能是 .JPG 或 .jpg
                with open(gt_path, 'rb') as f:
                    gt_data = np.frombuffer(f.read(), dtype=np.float32)
                gt_depth = gt_data.reshape(4032, 6048)
                gt_depth = np.where(np.isinf(gt_depth), 0, gt_depth)
                gt_depth = torch.tensor(np.array(gt_depth, dtype=np.float32))
            elif gt_path.endswith('.npy'):
                # NPY format: DIODE, RobotCar, nuScenes
                gt_depth = np.load(gt_path)
                gt_depth = torch.tensor(np.array(gt_depth, dtype=np.float32))
            elif gt_path.endswith('.png'):
                gt_img = Image.open(gt_path)
                gt_depth = np.array(gt_img).astype(np.float32)

                # 根据数据集类型解码深度
                if args.dataset == 'kitti' or args.dataset.startswith('kitti_c_'):
                    # KITTI/KITTI-C: depth stored as 16-bit, decode by /256
                    gt_depth = gt_depth / 256.0
                elif args.dataset.startswith('drive_'):
                    # DrivingStereo: same as KITTI, decode by /256
                    gt_depth = gt_depth / 256.0
                elif args.dataset in ['nyuv2', 'scannet']:
                    # NYUv2/ScanNet: depth in mm, decode by /1000
                    if gt_depth.max() > 100:  # 16-bit depth
                        gt_depth = gt_depth / 1000.0
                # DIODE 使用 .npy 格式，不走这个分支

                gt_depth = torch.tensor(np.ascontiguousarray(gt_depth), dtype=torch.float32)
            else:
                continue

            # Squeeze if needed
            if gt_depth.dim() > 2:
                gt_depth = gt_depth.squeeze()
            if pred_depth.dim() > 2:
                pred_depth = pred_depth.squeeze()

            # Resize pred to GT size if needed
            if pred_depth.shape != gt_depth.shape:
                pred_depth = F.interpolate(
                    pred_depth.unsqueeze(0).unsqueeze(0),
                    size=gt_depth.shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

            # Load mask if available
            if sample.get('mask') and os.path.exists(sample['mask']):
                mask = np.load(sample['mask'])
                valid_mask = torch.tensor(np.array(mask, dtype=np.float32))
            else:
                valid_mask = create_valid_mask(gt_depth, args.min_depth, args.max_depth)

            if valid_mask.sum() < 100:
                continue

            # Apply crop BEFORE alignment (important for correct metrics)
            if args.crop_type != 'none':
                crop_mask = apply_crop(
                    gt_depth.numpy(),
                    valid_mask.numpy(),
                    args.dataset,
                    args.crop_type
                )
                valid_mask = torch.tensor(np.ascontiguousarray(crop_mask), dtype=torch.float32)

            if valid_mask.sum() < 100:
                continue

            # Align prediction to GT (on cropped region)
            if args.align_method == 'least_squares':
                pred_aligned = align_depth_least_squares(pred_depth, gt_depth, valid_mask)
            elif args.align_method == 'disparity':
                pred_aligned = align_depth_disparity_least_squares(pred_depth, gt_depth, valid_mask)
            elif args.align_method == 'log_depth':
                pred_aligned = align_depth_log_least_squares(pred_depth, gt_depth, valid_mask)
            else:
                pred_aligned, _ = align_depth_median(pred_depth, gt_depth, valid_mask)

            # Clamp
            pred_aligned = torch.clamp(pred_aligned, args.min_depth, args.max_depth)
            gt_clamped = torch.clamp(gt_depth, args.min_depth, args.max_depth)

            # 使用完整指标计算
            pred_np = pred_aligned.numpy()
            gt_np = gt_clamped.numpy()
            mask_np = valid_mask.numpy().astype(bool)

            sample_metrics = compute_errors(gt_np[mask_np], pred_np[mask_np])
            metrics.update(sample_metrics)
            eval_count += 1

            # Save visualization
            if args.save_vis and eval_count <= 50:  # 只保存前50个
                vis_path = os.path.join(vis_dir, f"{sample['name']}_vis.png")
                save_visualization(
                    sample['rgb'], pred_np, gt_np, mask_np.astype(float),
                    vis_path
                )

        except Exception as e:
            import traceback
            print(f"Error evaluating {sample['name']}: {e}")
            traceback.print_exc()
            continue

    # Print summary
    if eval_count == 0:
        raise SystemExit("No valid predictions were evaluated; check the prediction cache, masks, and depth paths")
    final_metrics = metrics.get_value()

    print("\n" + "="*70)
    print(f"Evaluation Results - {args.dataset.upper()}")
    print("="*70)
    print(f"Total samples evaluated: {eval_count}")
    print(f"LoRA: {args.lora_path}")
    print(f"Align method: {args.align_method}")
    if getattr(args, 'pred_transform', None):
        if args.pred_transform == 'power':
            print(f"Pred transform: power={args.pred_power}")
        else:
            print(f"Pred transform: {args.pred_transform}")
    print(f"Crop type: {args.crop_type}")
    print("-"*70)

    if eval_count > 0:
        # 主要指标
        print(f"{'Metric':<12} {'Value':>10}")
        print("-"*25)
        print(f"{'AbsRel ↓':<12} {final_metrics.get('abs_rel', 0):.4f}")
        print(f"{'SqRel ↓':<12} {final_metrics.get('sq_rel', 0):.4f}")
        print(f"{'RMSE ↓':<12} {final_metrics.get('rmse', 0):.4f}")
        print(f"{'RMSE_log ↓':<12} {final_metrics.get('rmse_log', 0):.4f}")
        print(f"{'SILog ↓':<12} {final_metrics.get('silog', 0):.4f}")
        print(f"{'log_10 ↓':<12} {final_metrics.get('log_10', 0):.4f}")
        print("-"*25)
        print(f"{'δ1 ↑':<12} {final_metrics.get('a1', 0):.4f}")
        print(f"{'δ2 ↑':<12} {final_metrics.get('a2', 0):.4f}")
        print(f"{'δ3 ↑':<12} {final_metrics.get('a3', 0):.4f}")
    else:
        print("No samples evaluated!")

    print("="*70)

    # Save results
    results = {
        'dataset': args.dataset,
        'lora_path': args.lora_path,
        'num_samples': eval_count,
        'align_method': args.align_method,
        'checkpoint_encoding': args.encoding,
        'crop_type': args.crop_type,
        'min_depth': args.min_depth,
        'max_depth': args.max_depth,
        'metrics': {k: float(v) for k, v in final_metrics.items()},
    }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/results.json")
    if args.save_vis:
        print(f"Visualizations saved to {vis_dir}/")


if __name__ == "__main__":
    main()
