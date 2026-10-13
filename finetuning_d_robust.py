"""
Step1X-Edit 抗干扰判别式训练 (Robust Discriminative Training)

核心策略：
- 每个 batch 同时包含原图和扰动图
- 两者共享同一个 GT 深度图
- 损失函数: L = L(model(x), y) + λ * L(model(perturb(x)), y)

扰动类型:
- 低光照 (Dark)
- 雾 (Fog)
- 雪 (Snow)
- 运动模糊 (Motion Blur)
- 对比度变化 (Contrast)

使用方法:
  python finetuning_d_robust.py \
    --discriminative_mode \
    --enable_perturbation \
    --perturbation_weight 1.0 \
    --dataset_config library/data_configs/lotus_disparity.toml \
    ...
"""

# ========== 限制 OpenCV 线程数，防止 "Can't spawn new thread" 错误 ==========
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OPENCV_THREAD_COUNT"] = "4"
# 必须在 import cv2 之前设置

import argparse
import copy
import math
import random
import time
from typing import Any, Optional, Union, List, Tuple
from multiprocessing import Value

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

from library import kohya_trainer
from library import (
    step1x_edit_train_utils,
    step1x_utils,
    strategy_step1x,
    strategy_base,
    train_util,
    config_util,
    huggingface_util,
)
from library.utils import setup_logging
from library.config_util import ConfigSanitizer, BlueprintGenerator
from library.perturbation import ImagePerturbation, perturb_tensor

# 导入判别式训练器作为基类
from finetuning_d import Step1XEditDiscriminativeTrainer

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _process_single_image_for_epoch(args_tuple):
    """
    处理单张图像的扰动（在子进程中运行）

    使用 epoch 作为随机种子的一部分，确保每个 epoch 的扰动不同
    返回：(source_path, output_path, status) 或 (source_path, (img_path, latents_path), status)
    """
    source_path, output_dir, perturber_cfg, perturbation_prob, epoch, cache_latents = args_tuple

    import random
    import os
    from PIL import Image
    import numpy as np

    # 构建输出路径
    basename = os.path.basename(source_path)
    name, ext = os.path.splitext(basename)
    output_path = os.path.join(output_dir, f"{name}_perturbed{ext}")
    latents_path = os.path.join(output_dir, f"{name}_perturbed.npz") if cache_latents else None

    # 使用 (源路径 + epoch) 作为随机种子，确保：
    # 1. 同一 epoch 内，同一图片的扰动决策一致
    # 2. 不同 epoch，同一图片可能有不同的扰动
    seed = hash(source_path) + epoch
    rng = random.Random(seed)

    # 按概率决定是否扰动
    if rng.random() >= perturbation_prob:
        return source_path, None, "skipped"

    try:
        # 惰性初始化扰动器
        if not hasattr(_process_single_image_for_epoch, 'perturber'):
            from library.perturbation import ImagePerturbation
            _process_single_image_for_epoch.perturber = ImagePerturbation(perturber_cfg)

        perturber = _process_single_image_for_epoch.perturber

        # 加载图像
        img = Image.open(source_path).convert('RGB')

        # 应用扰动
        perturbed_img = perturber(img)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 保存扰动图
        perturbed_img.save(output_path, quality=95)

        return source_path, output_path, "success"
    except Exception as e:
        return source_path, None, f"error: {e}"


class Step1XEditRobustDiscriminativeTrainer(Step1XEditDiscriminativeTrainer):
    """
    抗干扰判别式训练器

    在判别式训练的基础上，增加扰动一致性损失：
    L_total = L_clean + λ * L_perturbed

    其中：
    - L_clean: 原图预测与 GT 的损失
    - L_perturbed: 扰动图预测与 GT 的损失
    - λ: 扰动损失权重 (perturbation_weight)

    优化策略：
    - 每个 epoch 开始前预生成扰动图（多进程并行，~8分钟）
    - 训练时直接读取预生成的扰动图
    - epoch 结束时清理临时文件
    """

    def __init__(self):
        super().__init__()
        self.perturber = None
        self.perturbed_dir = None  # 预生成扰动图的目录
        self.perturbed_mapping = {}  # 源图路径 -> 扰动图路径的映射
        self.current_epoch_for_perturbation = -1  # 当前已预生成的 epoch
        self.perturbation_cfg = None  # 扰动配置

    def init_perturbation(self, args):
        """初始化扰动器"""
        if not getattr(args, 'enable_perturbation', False):
            return

        cfg = {
            'is_dark': getattr(args, 'perturb_dark', True),
            'is_fog': getattr(args, 'perturb_fog', True),
            'is_snow': getattr(args, 'perturb_snow', True),
            'is_motion_blur': getattr(args, 'perturb_motion_blur', True),
            'is_zoom_blur': True,  # 与 DepthAnythingAC 保持一致
            'is_contrast': getattr(args, 'perturb_contrast', True),

            # 概率设置（dark 是基础扰动，始终应用，不需要概率）
            'prob_fog': 0.2,
            'prob_snow': 0.2,
            'prob_motion_blur': 0.1,
            'prob_zoom_blur': 0.1,
            'prob_contrast': 0.2,

            'severity_lower_dark': getattr(args, 'perturb_severity_min', 1),
            'severity_upper_dark': getattr(args, 'perturb_severity_max', 5),
            'severity_lower_fog': getattr(args, 'perturb_severity_min', 1),
            'severity_upper_fog': min(getattr(args, 'perturb_severity_max', 3), 3),
            'severity_lower_snow': getattr(args, 'perturb_severity_min', 1),
            'severity_upper_snow': min(getattr(args, 'perturb_severity_max', 3), 2),
            'severity_lower_motion_blur': getattr(args, 'perturb_severity_min', 1),
            'severity_upper_motion_blur': min(getattr(args, 'perturb_severity_max', 3), 3),
            'severity_lower_zoom_blur': getattr(args, 'perturb_severity_min', 1),
            'severity_upper_zoom_blur': min(getattr(args, 'perturb_severity_max', 3), 3),
            'severity_lower_contrast': getattr(args, 'perturb_severity_min', 1),
            'severity_upper_contrast': min(getattr(args, 'perturb_severity_max', 3), 3),
        }

        self.perturber = ImagePerturbation(cfg)
        self.perturbation_cfg = cfg
        logger.info(f"Perturbation enabled with weight={args.perturbation_weight}")
        logger.info(f"  - Dark: {cfg['is_dark']}, Fog: {cfg['is_fog']}, Snow: {cfg['is_snow']}")
        logger.info(f"  - Motion Blur: {cfg['is_motion_blur']}, Zoom Blur: {cfg['is_zoom_blur']}, Contrast: {cfg['is_contrast']}")

    def _on_epoch_start(self, epoch: int, args, accelerator, vae):
        """
        Epoch 开始时的钩子 - 预生成扰动图

        这个方法在训练循环外被调用，预生成时间不计入训练时间
        """
        if not getattr(args, 'enable_perturbation', False):
            return

        # 初始化扰动器（首次调用时）
        if self.perturber is None:
            self.init_perturbation(args)

        # 保存 VAE 引用用于 latents 编码
        self.vae = vae

        # 预生成扰动图
        self.pregenerate_perturbations(epoch, args, accelerator)

    def pregenerate_perturbations(self, epoch: int, args, accelerator):
        """
        在 epoch 开始前预生成扰动图（多进程并行）

        Args:
            epoch: 当前 epoch 编号
            args: 训练参数
            accelerator: Accelerator 实例
        """
        if not getattr(args, 'enable_perturbation', False):
            return

        # 扰动图保存在项目目录下（避免系统盘空间不足）
        base_perturbed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perturbed_cache")
        existing_dir = os.path.join(base_perturbed_dir, f"epoch_{epoch}")
        existing_mapping = os.path.join(existing_dir, "mapping.json")

        # ========== 检查是否需要生成（所有进程同步检查） ==========
        need_generation = False
        if accelerator.is_main_process:
            if self.current_epoch_for_perturbation != epoch and not os.path.exists(existing_mapping):
                need_generation = True
                logger.info(f"[Perturbation] Will generate perturbations for epoch {epoch} (this may take 15-30 minutes)")

        # 广播生成标志到所有进程
        import torch.distributed as dist
        if dist.is_initialized():
            need_gen_tensor = torch.tensor([1 if need_generation else 0], device=accelerator.device)
            dist.broadcast(need_gen_tensor, src=0)
            need_generation = need_gen_tensor.item() == 1

        # ========== 检查缓存状态 ==========
        images_exist = os.path.exists(existing_mapping)
        latents_complete = False
        missing_latents_paths = []

        if images_exist:
            # 加载映射
            self.perturbed_dir = existing_dir
            self._load_perturbation_mapping(epoch, args)

            # 检查 latents 是否完整
            for ref_path, entry in self.perturbed_mapping.items():
                if isinstance(entry, dict):
                    latents_path = entry.get('latents')
                    img_path = entry.get('img')
                    if latents_path and img_path:
                        if not os.path.exists(latents_path) and os.path.exists(img_path):
                            missing_latents_paths.append((ref_path, img_path, latents_path))

            latents_complete = (len(missing_latents_paths) == 0)

            if accelerator.is_main_process:
                if latents_complete:
                    logger.info(f"[Perturbation] Found complete cache for epoch {epoch}, reusing")
                else:
                    logger.info(f"[Perturbation] Found images but {len(missing_latents_paths)} latents missing")

        if images_exist and latents_complete:
            # 缓存完整，直接复用
            self.current_epoch_for_perturbation = epoch
            return

        if images_exist and not latents_complete:
            # 扰动图存在但 latents 不完整：只做阶段2（分布式编码）
            self._distributed_encode_latents(missing_latents_paths, args, accelerator)
            self.current_epoch_for_perturbation = epoch
        else:
            # 需要从头生成：调用完整的分布式生成流程
            self._do_full_perturbation_generation(epoch, args, accelerator, existing_dir)
            # 加载映射
            self._load_perturbation_mapping(epoch, args)

        # 清理内存
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def _do_full_perturbation_generation(self, epoch: int, args, accelerator, existing_dir):
        """
        执行完整的扰动图生成流程

        分布式策略：
        - 阶段1（图像生成）：仅主进程执行（CPU 密集型，使用多进程）
        - 阶段2（latents 编码）：所有 GPU 参与（GPU 密集型，分布式加速）

        Args:
            epoch: 当前 epoch 编号
            args: 训练参数
            accelerator: Accelerator 实例
            existing_dir: 扰动图保存目录
        """
        self.perturbed_dir = existing_dir

        # 仅主进程执行阶段1（图像生成）
        perturbed_images_paths = []

        if accelerator.is_main_process:
            # 清理上一个 epoch 的扰动图
            base_perturbed_dir = os.path.dirname(existing_dir)
            if os.path.exists(base_perturbed_dir):
                for old_dir in os.listdir(base_perturbed_dir):
                    old_path = os.path.join(base_perturbed_dir, old_dir)
                    if old_path != existing_dir and os.path.isdir(old_path):
                        import shutil
                        logger.info(f"[Perturbation] Cleaning up: {old_path}")
                        shutil.rmtree(old_path, ignore_errors=True)

            os.makedirs(self.perturbed_dir, exist_ok=True)

            # 收集源图路径
            source_dirs = [
                ('dataset_disparity/hypersim_source', 'hypersim_source'),
                ('dataset_disparity/vkitti_source', 'vkitti_source')
            ]

            source_paths = []
            source_path_to_ref_path = {}

            for actual_dir, ref_prefix in source_dirs:
                if os.path.exists(actual_dir):
                    for f in os.listdir(actual_dir):
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                            actual_path = os.path.join(actual_dir, f)
                            ref_path = os.path.join(ref_prefix, f)
                            source_paths.append(actual_path)
                            source_path_to_ref_path[actual_path] = ref_path

            if not source_paths:
                logger.warning("[Perturbation] No source images found!")
                source_paths = []
                source_path_to_ref_path = {}

        # ========== 阶段1: 所有进程分布式生成扰动图 ==========
        # 每个 GPU 进程负责生成一部分图像
        perturbation_prob = getattr(args, 'perturbation_prob', 0.5)
        num_workers = getattr(args, 'perturbation_workers', 16)

        world_size = accelerator.num_processes
        rank = accelerator.process_index

        # 收集源图路径（所有进程都需要执行）
        source_dirs = [
            ('dataset_disparity/hypersim_source', 'hypersim_source'),
            ('dataset_disparity/vkitti_source', 'vkitti_source')
        ]

        all_source_paths = []
        all_source_path_to_ref_path = {}

        for actual_dir, ref_prefix in source_dirs:
            if os.path.exists(actual_dir):
                for f in os.listdir(actual_dir):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        actual_path = os.path.join(actual_dir, f)
                        ref_path = os.path.join(ref_prefix, f)
                        all_source_paths.append(actual_path)
                        all_source_path_to_ref_path[actual_path] = ref_path

        # 按 rank 分片
        my_source_paths = all_source_paths[rank::world_size]

        logger.info(f"[Perturbation] GPU {rank}: Processing {len(my_source_paths)}/{len(all_source_paths)} images")

        if my_source_paths:
            import subprocess

            start_time = time.time()
            # 每个进程使用自己的 mapping 文件
            my_mapping_file = os.path.join(self.perturbed_dir, f"mapping_rank{rank}.json")

            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_perturbations.py")
            cmd = [
                sys.executable, script_path,
                '--epoch', str(epoch),
                '--output_dir', self.perturbed_dir,
                '--perturbation_prob', str(perturbation_prob),
                '--workers', str(num_workers),
                '--mapping_file', my_mapping_file,
                '--perturb_severity_min', str(getattr(args, 'perturb_severity_min', 1)),
                '--perturb_severity_max', str(getattr(args, 'perturb_severity_max', 5)),
                '--rank', str(rank),
                '--world_size', str(world_size),
            ]

            if self.perturbation_cfg.get('is_dark', True):
                cmd.append('--perturb_dark')
            if self.perturbation_cfg.get('is_fog', True):
                cmd.append('--perturb_fog')
            if self.perturbation_cfg.get('is_snow', True):
                cmd.append('--perturb_snow')
            if self.perturbation_cfg.get('is_motion_blur', True):
                cmd.append('--perturb_motion_blur')
            if self.perturbation_cfg.get('is_zoom_blur', True):
                cmd.append('--perturb_zoom_blur')
            if self.perturbation_cfg.get('is_contrast', True):
                cmd.append('--perturb_contrast')

            # 设置超时：每张图像约 0.5 秒，最多 2 小时
            timeout_seconds = max(len(my_source_paths) * 0.5, 7200)
            logger.info(f"[Perturbation] GPU {rank}: Starting subprocess with {len(my_source_paths)} images...")
            try:
                result = subprocess.run(
                    cmd,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    timeout=timeout_seconds,
                    stdout=None,
                    stderr=None
                )

                elapsed_phase1 = time.time() - start_time
                if result.returncode == 0:
                    logger.info(f"[Perturbation] GPU {rank}: Phase 1 complete in {elapsed_phase1:.1f}s")
                else:
                    logger.error(f"[Perturbation] GPU {rank}: Phase 1 failed with code {result.returncode}")
            except subprocess.TimeoutExpired:
                logger.error(f"[Perturbation] GPU {rank}: Phase 1 timed out after {timeout_seconds}s")
            except Exception as e:
                logger.error(f"[Perturbation] GPU {rank}: Phase 1 failed with exception: {e}")

        # ========== 同步：等待所有进程完成阶段1 ==========
        accelerator.wait_for_everyone()

        # ========== 主进程合并所有 mapping 文件 ==========
        if accelerator.is_main_process:
            import json
            merged_mapping = {}
            for r in range(world_size):
                rank_mapping_file = os.path.join(self.perturbed_dir, f"mapping_rank{r}.json")
                if os.path.exists(rank_mapping_file):
                    with open(rank_mapping_file, 'r') as f:
                        rank_mapping = json.load(f)
                    merged_mapping.update(rank_mapping)
                    # 删除临时文件
                    os.remove(rank_mapping_file)

            # 保存合并后的 mapping
            final_mapping_file = os.path.join(self.perturbed_dir, "mapping.json")
            with open(final_mapping_file, 'w') as f:
                json.dump(merged_mapping, f)
            logger.info(f"[Perturbation] Merged {len(merged_mapping)} entries into mapping.json")

        # ========== 同步：等待主进程完成合并 ==========
        accelerator.wait_for_everyone()

        # ========== 阶段2: 所有 GPU 分布式编码 latents ==========
        mapping_file = os.path.join(self.perturbed_dir, "mapping.json")

        if os.path.exists(mapping_file):
            import json
            with open(mapping_file, 'r') as f:
                self.perturbed_mapping = json.load(f)

            # 收集需要编码的图像
            all_perturbed_paths = []
            total_entries = len(self.perturbed_mapping)
            img_exists_count = 0
            latents_exists_count = 0

            for ref_path, entry in self.perturbed_mapping.items():
                img_path = entry.get('img')
                latents_path = entry.get('latents')
                if img_path and latents_path:
                    if os.path.exists(img_path):
                        img_exists_count += 1
                        if os.path.exists(latents_path):
                            latents_exists_count += 1
                        else:
                            all_perturbed_paths.append((ref_path, img_path, latents_path))

            if accelerator.is_main_process:
                logger.info(f"[Perturbation] Phase 2 check: {total_entries} mapping entries, "
                          f"{img_exists_count} imgs exist, {latents_exists_count} latents exist, "
                          f"{len(all_perturbed_paths)} to encode")

            if all_perturbed_paths:
                # 分布式编码：每个 GPU 处理一部分
                world_size = accelerator.num_processes
                rank = accelerator.process_index

                # 按 rank 分片
                my_paths = all_perturbed_paths[rank::world_size]

                logger.info(f"[Perturbation] GPU {rank}: Encoding {len(my_paths)}/{len(all_perturbed_paths)} latents...")
                start_phase2 = time.time()

                self._encode_perturbed_latents(my_paths, args, accelerator)

                elapsed_phase2 = time.time() - start_phase2
                logger.info(f"[Perturbation] GPU {rank}: Phase 2 complete in {elapsed_phase2:.1f}s")

        # ========== 同步：等待所有 GPU 完成编码 ==========
        accelerator.wait_for_everyone()

        self.current_epoch_for_perturbation = epoch

        if accelerator.is_main_process:
            logger.info(f"[Perturbation] Epoch {epoch} perturbation generation complete")

    def _distributed_encode_latents(self, all_missing_paths, args, accelerator):
        """
        分布式编码缺失的 latents

        每个 GPU 处理 1/N 的图像，加速编码过程
        """
        if not all_missing_paths:
            return

        world_size = accelerator.num_processes
        rank = accelerator.process_index

        # 按 rank 分片
        my_paths = all_missing_paths[rank::world_size]

        if accelerator.is_main_process:
            logger.info(f"[Perturbation] Distributed encoding: {len(all_missing_paths)} total latents")

        logger.info(f"[Perturbation] GPU {rank}: Encoding {len(my_paths)} latents...")
        start_time = time.time()

        self._encode_perturbed_latents(my_paths, args, accelerator)

        elapsed = time.time() - start_time
        logger.info(f"[Perturbation] GPU {rank}: Encoding complete in {elapsed:.1f}s")

        # 等待所有 GPU 完成
        accelerator.wait_for_everyone()

    def _encode_perturbed_latents(self, perturbed_images_paths, args, accelerator):
        """
        GPU 批量编码扰动图的 latents 并保存到磁盘

        Args:
            perturbed_images_paths: List of (ref_path, img_path, latents_path)
            args: 训练参数
            accelerator: Accelerator 实例
        """
        if not perturbed_images_paths:
            return

        from PIL import Image
        import numpy as np

        # 获取 VAE 并确保在 GPU 上
        vae = self.vae
        vae_device = next(vae.parameters()).device
        vae_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

        # 如果 VAE 在 CPU 上，移动到 GPU
        if vae_device.type == 'cpu':
            vae_device = accelerator.device
            vae = vae.to(vae_device)
            logger.info(f"[Perturbation] Moved VAE to {vae_device} for latents encoding")

        # 按图像尺寸分组，同尺寸的可以批量编码
        # 跳过已存在的 latents 文件
        size_groups = {}  # (H, W) -> [(ref_path, img_path, latents_path), ...]
        skipped_count = 0

        for ref_path, img_path, latents_path in perturbed_images_paths:
            # 跳过已存在的 latents
            if os.path.exists(latents_path):
                skipped_count += 1
                continue
            try:
                img = Image.open(img_path)
                size = (img.height, img.width)
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append((ref_path, img_path, latents_path))
            except:
                continue

        total_to_encode = sum(len(items) for items in size_groups.values())
        logger.info(f"[Perturbation] Skipped {skipped_count} existing latents, {total_to_encode} to encode")

        if total_to_encode == 0:
            logger.info(f"[Perturbation] All latents already cached, skipping encoding")
            return

        logger.info(f"[Perturbation] Found {len(size_groups)} different image sizes")
        for size, items in size_groups.items():
            logger.info(f"  - {size[0]}x{size[1]}: {len(items)} images")

        total_encoded = 0
        torch.cuda.empty_cache()

        def load_image_to_tensor(img_path):
            """加载单张图像并转为 tensor，使用与原版相同的方式"""
            img = Image.open(img_path).convert('RGB')
            # 使用 PIL 直接转 tensor（与 strategy_step1x.py 相同的方式）
            img_bytes = img.tobytes()
            h, w = img.size[1], img.size[0]
            img_tensor = torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8).reshape(h, w, 3).permute(2, 0, 1).float() / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            return img_tensor

        with torch.no_grad():
            for size, items in size_groups.items():
                pixels = size[0] * size[1]
                if pixels <= 512 * 512:
                    batch_size = 32
                elif pixels <= 576 * 768:
                    batch_size = 16
                elif pixels <= 768 * 1024:
                    batch_size = 8
                else:
                    batch_size = 4

                logger.info(f"[Perturbation] Encoding {size[0]}x{size[1]} ({len(items)} imgs, batch_size={batch_size})")
                pbar = tqdm(range(0, len(items), batch_size),
                           desc=f"Encoding {size[0]}x{size[1]}",
                           disable=not accelerator.is_main_process)

                for i in pbar:
                    batch_items = items[i:i+batch_size]

                    # 加载批量图像
                    images = []
                    valid_items = []
                    for item in batch_items:
                        ref_path, img_path, latents_path = item
                        try:
                            img_tensor = load_image_to_tensor(img_path)
                            images.append(img_tensor)
                            valid_items.append(item)
                        except Exception as e:
                            pass

                    if not images:
                        continue

                    # 批量编码
                    images_batch = torch.stack(images).to(device=vae_device, dtype=vae_dtype)
                    latents = self.encode_images_to_latents(args, vae, images_batch)
                    latents = self.shift_scale_latents(args, latents)
                    latents_np = latents.float().cpu().numpy()

                    # 保存
                    for j, (ref_path, img_path, latents_path) in enumerate(valid_items):
                        np.save(latents_path, latents_np[j])
                        total_encoded += 1

        logger.info(f"[Perturbation] Encoded {total_encoded} latents")

    def _load_perturbation_mapping(self, epoch: int, args):
        """从文件加载扰动映射（供非主进程使用）"""
        base_perturbed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perturbed_cache")
        self.perturbed_dir = os.path.join(base_perturbed_dir, f"epoch_{epoch}")
        mapping_file = os.path.join(self.perturbed_dir, "mapping.json")

        if os.path.exists(mapping_file):
            import json
            with open(mapping_file, 'r') as f:
                self.perturbed_mapping = json.load(f)
            self.current_epoch_for_perturbation = epoch

    def get_perturbed_image_path(self, source_path: str) -> Optional[str]:
        """获取源图对应的预生成扰动图路径"""
        return self.perturbed_mapping.get(source_path, None)

    def _load_pregenerated_latents(self, batch, weight_dtype, device):
        """
        从预生成的 latents 缓存直接加载（跳过 VAE 编码）

        支持 Bucket 模式：当开启 bucket 时，训练图像会被 resize 到不同分辨率，
        此时需要将预生成的 latents resize 到匹配的尺寸。

        Returns:
            (perturbed_latents, valid_indices) 或 (None, None)
        """
        import numpy as np
        import torch.nn.functional as F

        if not self.perturbed_dir or not os.path.exists(self.perturbed_dir):
            return None, None

        ref_image_paths = batch.get("ref_image_paths", None)
        if ref_image_paths is None:
            return None, None

        # 获取目标 latent 尺寸（从 batch 中的 ref_latents 获取）
        # ref_latents 形状: (B, C, H, W)
        target_latent_size = None
        if "ref_latents" in batch and batch["ref_latents"] is not None:
            ref_latents = batch["ref_latents"]
            target_latent_size = (ref_latents.shape[2], ref_latents.shape[3])  # (H, W)
        elif "latents" in batch and batch["latents"] is not None:
            latents = batch["latents"]
            target_latent_size = (latents.shape[2], latents.shape[3])  # (H, W)

        latents_list = []
        valid_indices = []

        for idx, source_path in enumerate(ref_image_paths):
            # 路径匹配
            matched_key = None
            if source_path in self.perturbed_mapping:
                matched_key = source_path
            else:
                for suffix in ['hypersim_source/', 'vkitti_source/']:
                    if suffix in source_path:
                        path_idx = source_path.find(suffix)
                        relative_path = source_path[path_idx:]
                        if relative_path in self.perturbed_mapping:
                            matched_key = relative_path
                            break

            if matched_key:
                mapping_entry = self.perturbed_mapping[matched_key]

                # 新格式：{'img': ..., 'latents': ...}
                if isinstance(mapping_entry, dict):
                    latents_path = mapping_entry.get('latents')
                else:
                    # 旧格式兼容：直接是图像路径
                    continue

                if latents_path and os.path.exists(latents_path):
                    try:
                        # 支持 .npy 和 .npz 格式
                        if latents_path.endswith('.npy'):
                            latent_np = np.load(latents_path)
                        else:
                            data = np.load(latents_path)
                            latent_np = data['latents']
                        latent = torch.from_numpy(latent_np)
                        latents_list.append(latent)
                        valid_indices.append(idx)
                    except Exception as e:
                        pass

        if latents_list:
            # 检查所有 latents 尺寸是否一致（同一 batch 应来自同一数据集）
            shapes = [l.shape for l in latents_list]
            if len(set(shapes)) > 1:
                logger.warning(f"[Perturbation] Mixed latent sizes in batch: {set(shapes)}, skipping")
                return None, None

            try:
                result = torch.stack(latents_list).to(device=device, dtype=weight_dtype)

                # 如果目标尺寸与当前尺寸不同，进行 resize（支持 bucket 模式）
                if target_latent_size is not None:
                    current_size = (result.shape[2], result.shape[3])  # (H, W)
                    if current_size != target_latent_size:
                        # 使用 bilinear 插值对 latents 进行 resize
                        # latents 形状: (B, C, H, W)
                        result = F.interpolate(
                            result,
                            size=target_latent_size,
                            mode='bilinear',
                            align_corners=False
                        )
                        # 只在首次 resize 时打印日志
                        if not hasattr(self, '_logged_latent_resize'):
                            self._logged_latent_resize = True
                            logger.info(f"[Perturbation] Resized latents from {current_size} to {target_latent_size} (bucket support)")

                return result, valid_indices
            except Exception as e:
                logger.warning(f"[Perturbation] Failed to stack latents: {e}")
                return None, None

        return None, None

    def _load_pregenerated_perturbations(self, batch, vae_dtype, device):
        """
        从预生成的扰动图加载图像（备用方法，当没有 latents 缓存时使用）

        Returns:
            perturbed_images tensor 或 None
        """
        from PIL import Image
        import numpy as np

        if not self.perturbed_dir or not os.path.exists(self.perturbed_dir):
            return None

        ref_image_paths = batch.get("ref_image_paths", None)

        if ref_image_paths is not None:
            perturbed_list = []
            valid_indices = []

            # 获取 batch 中原图的尺寸（用于 resize）
            ref_images = batch.get("ref_images", None)
            if ref_images is not None and len(ref_images) > 0:
                # ref_images 的形状是 (B, C, H, W)
                target_size = (ref_images.shape[3], ref_images.shape[2])  # (W, H)
            else:
                target_size = None

            for idx, source_path in enumerate(ref_image_paths):
                matched_key = None
                if source_path in self.perturbed_mapping:
                    matched_key = source_path
                else:
                    for suffix in ['hypersim_source/', 'vkitti_source/']:
                        if suffix in source_path:
                            path_idx = source_path.find(suffix)
                            relative_path = source_path[path_idx:]
                            if relative_path in self.perturbed_mapping:
                                matched_key = relative_path
                                break

                if matched_key:
                    mapping_entry = self.perturbed_mapping[matched_key]
                    # 获取图像路径
                    if isinstance(mapping_entry, dict):
                        perturbed_path = mapping_entry.get('img')
                    else:
                        perturbed_path = mapping_entry

                    if perturbed_path and os.path.exists(perturbed_path):
                        try:
                            img = Image.open(perturbed_path).convert('RGB')

                            # 重要：resize 到与 batch 中原图相同的尺寸（bucket 尺寸）
                            if target_size is not None and img.size != target_size:
                                img = img.resize(target_size, Image.BILINEAR)

                            img_np = np.array(img).astype(np.float32) / 255.0
                            img_np = img_np * 2 - 1
                            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
                            perturbed_list.append(img_tensor)
                            valid_indices.append(idx)
                        except:
                            pass

            if perturbed_list:
                result = torch.stack(perturbed_list).to(device=device, dtype=vae_dtype)
                batch['_perturbed_indices'] = valid_indices
                return result

        return None
        # 这是一个备用方案，用于没有传递路径信息的情况
        batch_size = batch.get("latents", batch.get("images")).shape[0] if batch.get("latents") is not None else batch.get("images").shape[0]

        # 检查是否有足够的预生成扰动图
        if not self.perturbed_mapping:
            return None

        # 随机选择 batch_size 张扰动图
        available_paths = list(self.perturbed_mapping.values())
        if len(available_paths) < batch_size:
            return None

        selected_paths = random.sample(available_paths, batch_size)

        perturbed_list = []
        target_size = None

        for perturbed_path in selected_paths:
            if os.path.exists(perturbed_path):
                try:
                    img = Image.open(perturbed_path).convert('RGB')

                    # 确保所有图像尺寸一致
                    if target_size is None:
                        # 从 batch 中获取目标尺寸
                        if batch.get("ref_images") is not None:
                            _, _, h, w = batch["ref_images"].shape
                            target_size = (w, h)
                        else:
                            target_size = img.size

                    if img.size != target_size:
                        img = img.resize(target_size, Image.BILINEAR)

                    img_np = np.array(img).astype(np.float32) / 255.0
                    img_np = img_np * 2 - 1
                    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
                    perturbed_list.append(img_tensor)
                except Exception as e:
                    return None
            else:
                return None

        if perturbed_list:
            return torch.stack(perturbed_list).to(device=device, dtype=vae_dtype)

        return None

    def cleanup_perturbations(self):
        """清理扰动图临时目录"""
        if self.perturbed_dir and os.path.exists(self.perturbed_dir):
            import shutil
            logger.info(f"[Perturbation] Final cleanup: {self.perturbed_dir}")
            shutil.rmtree(self.perturbed_dir, ignore_errors=True)
            self.perturbed_dir = None
            self.perturbed_mapping = {}

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy=None,
        is_train=True,
        train_text_encoder=False,
        train_unet=True,
    ):
        """
        处理单个 batch - 抗干扰判别式训练版本 (特征一致性 + RGB重建一致性 Loss)

        五项 Loss 策略:
        1. L_gt_clean = MSE(pred_clean, GT)              — 原图拟合 GT (权重 1.0)
        2. L_gt_perturbed = MSE(pred_perturbed, GT)       — 扰动图拟合 GT (权重 α)
        3. L_feat = 1-CosSim(feat_perturbed, feat_clean.detach()) — 特征一致性 (权重 β)
        4. L_rgb_self = MSE(rgb_recon_clean, ref_latents) — RGB自重建 (权重 γ₁)
        5. L_rgb_cross = MSE(rgb_recon_perturbed, ref_latents_clean.detach()) — RGB跨域一致性 (权重 γ₂)

        L_total = L_clean + α*L_perturbed*r + β*L_feat*r + γ₁*L_rgb_self + γ₂*L_rgb_cross*r
        """
        # ========== 读取参数 ==========
        feature_loss_layers = getattr(args, 'feature_loss_layers', 4)
        feature_loss_weight = getattr(args, 'feature_loss_weight', 1.0)
        gt_perturbed_weight = getattr(args, 'gt_perturbed_weight', 0.5)
        perturbation_weight = getattr(args, 'perturbation_weight', 1.0)
        enable_rgb_reconstruction = getattr(args, 'enable_rgb_reconstruction', False)
        rgb_self_weight = getattr(args, 'rgb_self_weight', 1.0)
        rgb_cross_weight = getattr(args, 'rgb_cross_weight', 1.0)

        need_features = (self.perturber is not None and is_train
                         and bool(self.perturbed_mapping) and feature_loss_layers > 0)
        need_rgb_recon = (enable_rgb_reconstruction and is_train)

        # ========== 获取 Latents ==========
        use_cached_latents = "latents" in batch and batch["latents"] is not None

        with torch.no_grad():
            if use_cached_latents:
                # 使用预缓存 latents（快速）
                latents = batch["latents"].to(accelerator.device)
                ref_latents = batch["ref_latents"].to(accelerator.device)
                latents = self.shift_scale_latents(args, latents)
                ref_latents = self.shift_scale_latents(args, ref_latents)

                # 仍然加载原始图像用于扰动（如果 batch 中有）
                if "ref_images" in batch and batch["ref_images"] is not None:
                    ref_images = batch["ref_images"].to(accelerator.device, dtype=vae_dtype)
                else:
                    ref_images = None
            else:
                # VAE 编码（无缓存时）
                images = batch["images"].to(accelerator.device, dtype=vae_dtype)
                ref_images = batch["ref_images"].to(accelerator.device, dtype=vae_dtype)

                latents = self.encode_images_to_latents(args, vae, images)
                ref_latents = self.encode_images_to_latents(args, vae, ref_images)
                latents = self.shift_scale_latents(args, latents)
                ref_latents = self.shift_scale_latents(args, ref_latents)

        latents = latents.to(weight_dtype)
        ref_latents = ref_latents.to(weight_dtype)

        # ========== 获取文本编码 ==========
        text_encoder_conds = self._get_text_encoder_conds(
            batch, args, accelerator, text_encoders,
            text_encoding_strategy, tokenize_strategy,
            is_train, train_text_encoder, weight_dtype
        )

        # ========== 1. Clean 前向传播 ==========
        if need_features and need_rgb_recon:
            model_pred, target, timesteps, weighting, intermediate_features_clean, rgb_recon_clean = \
                self.get_noise_pred_and_target_discriminative(
                    args, accelerator, latents, ref_latents, batch,
                    text_encoder_conds, unet, network, weight_dtype, train_unet, is_train,
                    return_intermediate=feature_loss_layers, return_rgb_recon=True,
                )
        elif need_features:
            model_pred, target, timesteps, weighting, intermediate_features_clean = \
                self.get_noise_pred_and_target_discriminative(
                    args, accelerator, latents, ref_latents, batch,
                    text_encoder_conds, unet, network, weight_dtype, train_unet, is_train,
                    return_intermediate=feature_loss_layers, return_rgb_recon=False,
                )
            rgb_recon_clean = None
        elif need_rgb_recon:
            model_pred, target, timesteps, weighting, rgb_recon_clean = \
                self.get_noise_pred_and_target_discriminative(
                    args, accelerator, latents, ref_latents, batch,
                    text_encoder_conds, unet, network, weight_dtype, train_unet, is_train,
                    return_intermediate=0, return_rgb_recon=True,
                )
            intermediate_features_clean = None
        else:
            model_pred, target, timesteps, weighting = \
                self.get_noise_pred_and_target_discriminative(
                    args, accelerator, latents, ref_latents, batch,
                    text_encoder_conds, unet, network, weight_dtype, train_unet, is_train,
                )
            intermediate_features_clean = None
            rgb_recon_clean = None

        loss_clean = self._compute_masked_loss(model_pred, target, batch)

        loss_rgb_self = None
        if rgb_recon_clean is not None:
            loss_rgb_self = F.mse_loss(rgb_recon_clean.float(), ref_latents.float())

        loss_rgb_cross = None

        # ========== 2. Perturbed 前向传播 ==========
        loss_perturbed = None
        loss_feat = None
        perturbed_indices = []
        num_perturbed = 0

        # 添加扰动统计
        if not hasattr(self, 'perturbation_count'):
            self.perturbation_count = 0
            self.total_batch_count = 0

        self.total_batch_count += 1

        if self.perturber is not None and is_train and self.perturbed_mapping:
            import time as _time
            t0 = _time.time()

            # 优先尝试加载缓存的 latents（跳过 VAE 编码）
            perturbed_latents, perturbed_indices = self._load_pregenerated_latents(
                batch, weight_dtype, accelerator.device
            )

            use_cached_latents_for_perturb = perturbed_latents is not None

            # 如果没有缓存的 latents，回退到加载图像 + VAE 编码
            if perturbed_latents is None:
                perturbed_images = self._load_pregenerated_perturbations(batch, vae_dtype, accelerator.device)
                perturbed_indices = batch.get('_perturbed_indices', None)

                if perturbed_images is not None and perturbed_indices is not None:
                    with torch.no_grad():
                        vae_device = next(vae.parameters()).device
                        perturbed_latents = self.encode_images_to_latents(
                            args, vae, perturbed_images.to(device=vae_device, dtype=vae_dtype)
                        )
                        perturbed_latents = self.shift_scale_latents(args, perturbed_latents)
                        perturbed_latents = perturbed_latents.to(device=accelerator.device, dtype=weight_dtype)

            t1 = _time.time()

            # 调试：打印加载结果
            if self.perturbation_count < 5:
                if perturbed_latents is not None:
                    cache_status = "cached latents" if use_cached_latents_for_perturb else "VAE encoded"
                    logger.info(f"[DEBUG] Loaded perturbed latents: shape={perturbed_latents.shape}, "
                              f"method={cache_status}, load_time={t1-t0:.3f}s")
                else:
                    ref_paths = batch.get("ref_image_paths", [])
                    logger.info(f"[DEBUG] Failed to load perturbations. ref_image_paths={ref_paths[:2]}")

            # ========== 关键：分布式训练中，所有 GPU 必须执行相同的操作 ==========
            # 即使某些 GPU 没有有效的扰动数据，也要执行前向传播（使用原图）
            # 否则会导致 NCCL 死锁

            # 检查 perturbed_latents 是否有效
            valid_perturbation = False
            if perturbed_latents is not None:
                self.perturbation_count += 1
                num_perturbed = len(perturbed_indices)

                # 检查 NaN
                if torch.isnan(perturbed_latents).any():
                    if self.perturbation_count % 50 == 1:
                        logger.warning(f"[Perturbation] NaN detected in perturbed_latents")
                    perturbed_latents = None
                else:
                    # 确保索引有效
                    batch_size = latents.shape[0]
                    valid_indices = [i for i in perturbed_indices if 0 <= i < batch_size]
                    if len(valid_indices) != len(perturbed_indices):
                        perturbed_indices = valid_indices

                    # 检查尺寸是否匹配
                    if perturbed_latents is not None and len(perturbed_indices) > 0:
                        latent_shape = latents.shape[-2:]  # (H, W)
                        perturbed_shape = perturbed_latents.shape[-2:]
                        if latent_shape != perturbed_shape:
                            if self.perturbation_count % 50 == 1:
                                logger.warning(f"[Perturbation] Size mismatch: latents={latent_shape}, perturbed={perturbed_shape}")
                            perturbed_latents = None
                        else:
                            valid_perturbation = True

            # ========== 无论是否有有效扰动，都执行前向传播（保持 GPU 同步） ==========
            if valid_perturbation and perturbed_latents is not None and len(perturbed_indices) > 0:
                # 有效扰动：使用扰动 latents
                latents_subset = latents[perturbed_indices]
                perturbed_latents = perturbed_latents[:len(perturbed_indices)]

                # ========== 扰动损失目标：拟合 GT ==========
                # 对于扰动图像，模型应该输出与原图相同的 GT 深度
                # 这样训练模型对输入扰动的鲁棒性
                target_subset = target[perturbed_indices]  # GT 目标

                t2 = _time.time()

                sub_batch = {k: v for k, v in batch.items() if not k.startswith('_')}
                if 'alpha_masks' in sub_batch and sub_batch['alpha_masks'] is not None:
                    sub_batch['alpha_masks'] = sub_batch['alpha_masks'][perturbed_indices]

                batch_size = latents.shape[0]
                text_encoder_conds_subset = self._slice_text_encoder_conds(
                    text_encoder_conds, perturbed_indices, batch_size
                )

                _ret_feat = need_features and feature_loss_layers > 0
                _ret_rgb = need_rgb_recon

                if _ret_feat and _ret_rgb:
                    (model_pred_perturbed, _, _, _,
                     intermediate_features_perturbed, rgb_recon_perturbed) = \
                        self.get_noise_pred_and_target_discriminative(
                            args, accelerator, latents_subset, perturbed_latents, sub_batch,
                            text_encoder_conds_subset, unet, network, weight_dtype, train_unet, is_train,
                            return_intermediate=feature_loss_layers, return_rgb_recon=True,
                        )
                elif _ret_feat:
                    (model_pred_perturbed, _, _, _,
                     intermediate_features_perturbed) = \
                        self.get_noise_pred_and_target_discriminative(
                            args, accelerator, latents_subset, perturbed_latents, sub_batch,
                            text_encoder_conds_subset, unet, network, weight_dtype, train_unet, is_train,
                            return_intermediate=feature_loss_layers, return_rgb_recon=False,
                        )
                    rgb_recon_perturbed = None
                elif _ret_rgb:
                    (model_pred_perturbed, _, _, _,
                     rgb_recon_perturbed) = \
                        self.get_noise_pred_and_target_discriminative(
                            args, accelerator, latents_subset, perturbed_latents, sub_batch,
                            text_encoder_conds_subset, unet, network, weight_dtype, train_unet, is_train,
                            return_intermediate=0, return_rgb_recon=True,
                        )
                    intermediate_features_perturbed = None
                else:
                    model_pred_perturbed, _, _, _ = \
                        self.get_noise_pred_and_target_discriminative(
                            args, accelerator, latents_subset, perturbed_latents, sub_batch,
                            text_encoder_conds_subset, unet, network, weight_dtype, train_unet, is_train,
                        )
                    intermediate_features_perturbed = None
                    rgb_recon_perturbed = None

                loss_perturbed = self._compute_masked_loss(model_pred_perturbed, target_subset, sub_batch)

                if intermediate_features_perturbed is not None and intermediate_features_clean is not None:
                    loss_feat = self._compute_feature_consistency_loss(
                        intermediate_features_clean, intermediate_features_perturbed, perturbed_indices
                    )

                if rgb_recon_perturbed is not None:
                    loss_rgb_cross = F.mse_loss(
                        rgb_recon_perturbed.float(),
                        ref_latents[perturbed_indices].detach().float()
                    )
            else:
                dummy_indices = [0]
                latents_dummy = latents[dummy_indices]

                sub_batch = {k: v for k, v in batch.items() if not k.startswith('_')}
                if 'alpha_masks' in sub_batch and sub_batch['alpha_masks'] is not None:
                    sub_batch['alpha_masks'] = sub_batch['alpha_masks'][dummy_indices]

                batch_size = latents.shape[0]
                text_encoder_conds_dummy = self._slice_text_encoder_conds(
                    text_encoder_conds, dummy_indices, batch_size
                )

                _dummy_out = self.get_noise_pred_and_target_discriminative(
                    args, accelerator, latents_dummy, ref_latents[dummy_indices], sub_batch,
                    text_encoder_conds_dummy, unet, network, weight_dtype, train_unet, is_train,
                    return_intermediate=feature_loss_layers if need_features else 0,
                    return_rgb_recon=need_rgb_recon,
                )
                model_pred_dummy = _dummy_out[0]
                _ = self._compute_masked_loss(model_pred_dummy, latents[dummy_indices], sub_batch)
                loss_perturbed = None

            if loss_perturbed is not None and torch.isnan(loss_perturbed):
                loss_perturbed = None
            if loss_feat is not None and torch.isnan(loss_feat):
                loss_feat = None
            if loss_rgb_cross is not None and torch.isnan(loss_rgb_cross):
                loss_rgb_cross = None

            if loss_perturbed is not None and self.perturbation_count % 100 == 1:
                cache_str = "cached" if use_cached_latents_for_perturb else "VAE"
                feat_str = f", L_feat={loss_feat.item():.4f}" if loss_feat is not None else ""
                rgb_s_str = f", L_rgb_self={loss_rgb_self.item():.4f}" if loss_rgb_self is not None else ""
                rgb_c_str = f", L_rgb_cross={loss_rgb_cross.item():.4f}" if loss_rgb_cross is not None else ""
                logger.info(f"[Perturbation] #{self.perturbation_count} "
                            f"({num_perturbed}/{latents.shape[0]} samples, {cache_str}"
                            f"{feat_str}{rgb_s_str}{rgb_c_str})")

        if loss_rgb_self is not None and torch.isnan(loss_rgb_self):
            loss_rgb_self = None

        # ========== 3. 计算总损失 ==========
        num_perturbed_samples = len(perturbed_indices) if perturbed_indices else 0
        batch_size = latents.shape[0]
        sample_ratio = num_perturbed_samples / batch_size if batch_size > 0 else 0.0

        loss = loss_clean
        if loss_perturbed is not None:
            loss = loss + gt_perturbed_weight * loss_perturbed * sample_ratio
        if loss_feat is not None:
            loss = loss + feature_loss_weight * loss_feat * sample_ratio
        if loss_rgb_self is not None:
            loss = loss + rgb_self_weight * loss_rgb_self
        if loss_rgb_cross is not None:
            loss = loss + rgb_cross_weight * loss_rgb_cross * sample_ratio

        loss._loss_clean = loss_clean.detach().item()
        loss._loss_perturbed = loss_perturbed.detach().item() if loss_perturbed is not None else 0.0
        loss._loss_feat = loss_feat.detach().item() if loss_feat is not None else 0.0
        loss._loss_rgb_self = loss_rgb_self.detach().item() if loss_rgb_self is not None else 0.0
        loss._loss_rgb_cross = loss_rgb_cross.detach().item() if loss_rgb_cross is not None else 0.0
        loss._has_perturbed = loss_perturbed is not None
        loss._num_perturbed = num_perturbed_samples

        return loss

    def _get_text_encoder_conds(
        self, batch, args, accelerator, text_encoders,
        text_encoding_strategy, tokenize_strategy,
        is_train, train_text_encoder, weight_dtype
    ):
        """获取文本编码条件"""
        text_encoder_conds = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = text_encoder_outputs_list

        if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
            with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
                if args.weighted_captions:
                    input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                        tokenize_strategy,
                        self.get_models_for_text_encoding(args, accelerator, text_encoders),
                        input_ids_list,
                        weights_list,
                    )
                else:
                    tokenized_inputs = batch["tokenized_inputs"]
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                        tokenize_strategy,
                        self.get_models_for_text_encoding(args, accelerator, text_encoders),
                        tokenized_inputs,
                    )
                if args.full_fp16:
                    encoded_text_encoder_conds = [c.to(weight_dtype) if c is not None else None for c in encoded_text_encoder_conds]

            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        return text_encoder_conds

    def _compute_masked_loss(self, model_pred, target, batch):
        """计算带 mask 的损失（与 finetuning_d.py 保持一致）"""
        alpha_masks = batch.get("alpha_masks", None)

        if alpha_masks is not None:
            if alpha_masks.dtype != torch.bool:
                alpha_masks = alpha_masks > 0.5

            # 下采样 mask
            if alpha_masks.shape[-2:] != model_pred.shape[-2:]:
                if alpha_masks.dim() == 3:
                    alpha_masks = alpha_masks.unsqueeze(1)

                downsample_factor = alpha_masks.shape[-1] // model_pred.shape[-1]
                if downsample_factor > 1:
                    invalid_mask = ~alpha_masks
                    invalid_down = F.max_pool2d(
                        invalid_mask.float(),
                        downsample_factor,
                        downsample_factor
                    ) > 0.5
                    alpha_masks = ~invalid_down

            weight_mask = alpha_masks.float()
            if weight_mask.shape[1] != model_pred.shape[1]:
                weight_mask = weight_mask.expand(-1, model_pred.shape[1], -1, -1)

            weight_mask = weight_mask.to(model_pred.device, dtype=model_pred.dtype)

            # 加权 MSE（关键：使用 .float() 确保精度，与 finetuning_d.py 一致）
            mse = (model_pred.float() - target.float()).pow(2)
            valid_count = weight_mask.sum().clamp(min=1.0)
            loss = (mse * weight_mask).sum() / valid_count
        else:
            # 无 mask 时，直接计算 MSE（使用 .float() 确保精度）
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    def _compute_feature_consistency_loss(self, clean_features, perturbed_features, perturbed_indices):
        """Uses cosine similarity: Loss = 1 - cos_sim, averaged over layers"""
        loss_feat = torch.tensor(0.0, device=perturbed_features[0].device, dtype=torch.float32)
        num_layers = min(len(clean_features), len(perturbed_features))
        if num_layers == 0:
            return loss_feat
        for i in range(num_layers):
            clean_feat_subset = clean_features[i][perturbed_indices].detach()
            perturbed_feat = perturbed_features[i]
            clean_flat = clean_feat_subset.float().flatten(1)
            perturbed_flat = perturbed_feat.float().flatten(1)
            clean_norm = F.normalize(clean_flat, p=2, dim=1)
            perturbed_norm = F.normalize(perturbed_flat, p=2, dim=1)
            cosine_sim = (clean_norm * perturbed_norm).sum(dim=1)
            layer_loss = (1 - cosine_sim).mean()
            loss_feat = loss_feat + layer_loss
        return loss_feat / num_layers

    def _slice_text_encoder_conds(self, text_encoder_conds, indices, batch_size):
        result = []
        for cond in text_encoder_conds:
            if cond is not None:
                try:
                    if hasattr(cond, 'shape') and len(cond.shape) > 0 and cond.shape[0] >= batch_size:
                        result.append(cond[indices])
                    else:
                        result.append(cond)
                except:
                    result.append(cond)
            else:
                result.append(cond)
        return result


def setup_parser() -> argparse.ArgumentParser:
    """设置参数解析器"""
    from finetuning_d import setup_parser as base_setup_parser
    parser = base_setup_parser()

    # 扰动训练参数
    parser.add_argument(
        "--enable_perturbation",
        action="store_true",
        help="启用扰动训练"
    )
    parser.add_argument(
        "--perturbation_weight",
        type=float,
        default=1.0,
        help="扰动损失权重 λ (default: 1.0)"
    )
    parser.add_argument(
        "--perturbation_prob",
        type=float,
        default=0.5,
        help="每个样本应用扰动的概率 (default: 0.5)"
    )

    # 扰动类型开关
    parser.add_argument("--perturb_dark", action="store_true", default=True,
                       help="启用低光照扰动")
    parser.add_argument("--perturb_fog", action="store_true", default=True,
                       help="启用雾扰动")
    parser.add_argument("--perturb_snow", action="store_true", default=True,
                       help="启用雪扰动")
    parser.add_argument("--perturb_motion_blur", action="store_true", default=True,
                       help="启用运动模糊")
    parser.add_argument("--perturb_contrast", action="store_true", default=True,
                       help="启用对比度变化")

    # 扰动严重程度
    parser.add_argument("--perturb_severity_min", type=int, default=1,
                       help="扰动严重程度下限 (1-5)")
    parser.add_argument("--perturb_severity_max", type=int, default=3,
                       help="扰动严重程度上限 (1-5)")

    # 预生成扰动图的并行进程数
    parser.add_argument("--perturbation_workers", type=int, default=16,
                       help="预生成扰动图的并行进程数 (default: 16)")

    # RGB 重建一致性 Loss 参数
    parser.add_argument("--enable_rgb_reconstruction", action="store_true",
                       help="启用 RGB 重建一致性 Loss")
    parser.add_argument("--rgb_self_weight", type=float, default=1.0,
                       help="RGB 自重建权重 γ₁ (default: 1.0)")
    parser.add_argument("--rgb_cross_weight", type=float, default=1.0,
                       help="RGB 跨域一致性权重 γ₂ (default: 1.0)")

    # 特征一致性 Loss 参数
    parser.add_argument("--feature_loss_weight", type=float, default=1.0,
                       help="特征一致性权重 β (default: 1.0)")
    parser.add_argument("--gt_perturbed_weight", type=float, default=0.5,
                       help="GT 扰动损失权重 α (default: 0.5)")
    parser.add_argument("--feature_loss_layers", type=int, default=4,
                       help="特征一致性使用的中间层数 (default: 4)")

    return parser


if __name__ == "__main__":
    from library import train_util

    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    # 强制启用判别式模式
    args.discriminative_mode = True

    trainer = Step1XEditRobustDiscriminativeTrainer()
    trainer.train(args)
