"""
Step1X-Edit 判别式训练 (Discriminative Training, Lotus-D style)

核心策略：
- 不加噪声，输入零张量 + 参考图 latents
- 固定 timestep（默认 1.0）
- 目标 = latents 本身（直接回归，而非 noise - latents）
- 单步推理即可获得深度预测

使用方法:
  python finetuning_d.py \
    --discriminative_mode \
    --discriminative_timestep 1.0 \
    --dataset_config library/data_configs/lotus_disparity.toml \
    ...
"""

# ========== 限制线程数，防止 "Can't spawn new thread" 错误 ==========
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OPENCV_THREAD_COUNT"] = "4"

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
from finetuning_p import Step1XEditProbSamplingTrainer

setup_logging()
import logging

logger = logging.getLogger(__name__)


class Step1XEditDiscriminativeTrainer(Step1XEditProbSamplingTrainer):
    def __init__(self):
        super().__init__()

    def get_noise_pred_and_target_discriminative(
        self,
        args,
        accelerator,
        latents,
        ref_latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
        return_intermediate: int = 0,
        return_rgb_recon: bool = False,
    ):
        """
        判别式前向传播（Lotus-D style）。

        与生成式训练的区别：
        - 输入零张量而非带噪 latents
        - 固定 timestep（默认 1.0）
        - 目标 = latents 本身（直接回归）
        - unpack 时返回两个分区：depth 预测 和 RGB 重建

        Returns:
            base 4-tuple: (model_pred, target, timesteps, weighting)
            +intermediate → 5-tuple: (..., intermediate_features)
            +rgb_recon → 5-tuple: (..., rgb_recon)
            +both → 6-tuple: (..., intermediate_features, rgb_recon)
        """
        import einops

        bsz = latents.shape[0]

        zero_input = torch.zeros_like(latents)

        timesteps = torch.full(
            (bsz,),
            getattr(args, "discriminative_timestep", 1.0),
            device=accelerator.device,
            dtype=weight_dtype,
        )

        packed_latent_height = latents.shape[2] // 2
        packed_latent_width = latents.shape[3] // 2
        img_ids = step1x_utils.prepare_img_ids(
            bsz, packed_latent_height, packed_latent_width
        ).to(device=accelerator.device)

        if args.gradient_checkpointing:
            zero_input.requires_grad_(True)
            ref_latents.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)
            img_ids.requires_grad_(True)

        packed_zero_input = step1x_utils.pack_latents(zero_input)
        packed_ref = step1x_utils.pack_latents(ref_latents)
        packed_model_input = torch.cat([packed_zero_input, packed_ref], dim=1)
        img_ids = torch.cat([img_ids, img_ids], dim=1)

        embeds, masks = text_encoder_conds[0], text_encoder_conds[1]
        masks = masks.to(torch.long)
        txt_ids = torch.zeros(bsz, embeds.shape[1], 3).to(packed_model_input.device)

        unet_dtype = next(unet.parameters()).dtype
        packed_model_input = packed_model_input.to(unet_dtype)
        masks = masks.to(device=accelerator.device)

        with torch.set_grad_enabled(is_train):
            model_output = unet(
                img=packed_model_input,
                img_ids=img_ids,
                txt_ids=txt_ids,
                timesteps=timesteps / 1000,
                llm_embedding=embeds,
                t_vec=timesteps,
                mask=masks,
                return_intermediate=return_intermediate,
            )

        if return_intermediate > 0:
            model_pred, intermediate_features = model_output
        else:
            model_pred = model_output
            intermediate_features = None

        unpacked = einops.rearrange(
            model_pred,
            "b (p h w) (c ph pw) -> b p c (h ph) (w pw)",
            h=packed_latent_height,
            w=packed_latent_width,
            ph=2,
            pw=2,
            p=2,
        )
        model_pred = unpacked[:, 0]
        rgb_recon = unpacked[:, 1] if return_rgb_recon else None

        target = latents

        weighting = None

        if return_intermediate > 0 and return_rgb_recon:
            return model_pred, target, timesteps, weighting, intermediate_features, rgb_recon
        elif return_intermediate > 0:
            return model_pred, target, timesteps, weighting, intermediate_features
        elif return_rgb_recon:
            return model_pred, target, timesteps, weighting, rgb_recon
        else:
            return model_pred, target, timesteps, weighting

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
        处理批次数据（判别式训练版本）。

        与生成式训练的区别：
        - 调用 get_noise_pred_and_target_discriminative 而非 get_noise_pred_and_target
        - 使用加权 MSE 损失 sum(mse * mask) / sum(mask)，避免布尔索引导致 GPU 同步
        - mask 下采样使用 max_pool2d 作用于 invalid mask 后取反
        """
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = batch["latents"].to(accelerator.device)
                ref_latents = batch["ref_latents"].to(accelerator.device)
            else:
                if args.vae_batch_size is None or len(batch["images"]) <= args.vae_batch_size:
                    latents = self.encode_images_to_latents(
                        args, vae, batch["images"].to(accelerator.device, dtype=vae_dtype)
                    )
                    ref_latents = self.encode_images_to_latents(
                        args, vae, batch["ref_images"].to(accelerator.device, dtype=vae_dtype)
                    )
                else:
                    chunks = [
                        batch["images"][i : i + args.vae_batch_size]
                        for i in range(0, len(batch["images"]), args.vae_batch_size)
                    ]
                    ref_chunks = [
                        batch["ref_images"][i : i + args.vae_batch_size]
                        for i in range(0, len(batch["ref_images"]), args.vae_batch_size)
                    ]
                    list_latents = []
                    ref_list_latents = []
                    for chunk, ref_chunk in zip(chunks, ref_chunks):
                        list_latents.append(
                            self.encode_images_to_latents(
                                args, vae, chunk.to(accelerator.device, dtype=vae_dtype)
                            )
                        )
                        ref_list_latents.append(
                            self.encode_images_to_latents(
                                args, vae, ref_chunk.to(accelerator.device, dtype=vae_dtype)
                            )
                        )
                    latents = torch.cat(list_latents, dim=0)
                    ref_latents = torch.cat(ref_list_latents, dim=0)

            latents = self.shift_scale_latents(args, latents)
            ref_latents = self.shift_scale_latents(args, ref_latents)

        latents = latents.to(weight_dtype)
        ref_latents = ref_latents.to(weight_dtype)

        # 获取 text encoder 条件
        text_encoder_conds = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = text_encoder_outputs_list

        if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
            with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
                if args.weighted_captions:
                    input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(
                        batch["captions"]
                    )
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
                    encoded_text_encoder_conds = [
                        c.to(weight_dtype) if c is not None else None
                        for c in encoded_text_encoder_conds
                    ]

            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        # 判别式前向传播
        model_pred, target, timesteps, weighting = self.get_noise_pred_and_target_discriminative(
            args, accelerator, latents, ref_latents, batch,
            text_encoder_conds, unet, network, weight_dtype, train_unet, is_train,
        )

        # 带 mask 的加权 MSE 损失
        alpha_masks = batch.get("alpha_masks", None)

        if alpha_masks is not None:
            if alpha_masks.dtype != torch.bool:
                alpha_masks = alpha_masks > 0.5

            if alpha_masks.shape[-2:] != model_pred.shape[-2:]:
                if alpha_masks.dim() == 3:
                    alpha_masks = alpha_masks.unsqueeze(1)

                downsample_factor = alpha_masks.shape[-1] // model_pred.shape[-1]
                if downsample_factor > 1:
                    invalid_mask = ~alpha_masks
                    invalid_down = F.max_pool2d(
                        invalid_mask.float(), downsample_factor, downsample_factor
                    ) > 0.5
                    alpha_masks = ~invalid_down

            weight_mask = alpha_masks.float()
            if weight_mask.shape[1] != model_pred.shape[1]:
                weight_mask = weight_mask.expand(-1, model_pred.shape[1], -1, -1)

            weight_mask = weight_mask.to(model_pred.device, dtype=model_pred.dtype)

            mse = (model_pred.float() - target.float()).pow(2)
            valid_count = weight_mask.sum().clamp(min=1.0)
            loss = (mse * weight_mask).sum() / valid_count
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        loss_weights = batch.get("loss_weights", None)
        if loss_weights is not None and loss_weights.numel() > 0:
            loss_per_sample = (
                F.mse_loss(model_pred.float(), target.float(), reduction="none")
                .mean([1, 2, 3])
            )
            loss = (loss_per_sample * loss_weights.to(model_pred.device)).mean()

        loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

        return loss


def setup_parser() -> argparse.ArgumentParser:
    parser = kohya_trainer.setup_parser()
    train_util.add_dit_training_arguments(parser)
    step1x_edit_train_utils.add_step1x_edit_train_arguments(parser)
    parser.add_argument(
        "--qwen2p5vl",
        type=str,
        help="Path to Qwen2.5VL model / Qwen2.5VL 模型的路径",
    )
    parser.add_argument(
        "--mix_dataset",
        action="store_true",
        help="Enable mixed dataset training / 启用混合数据集训练",
    )
    parser.add_argument(
        "--prob_hypersim",
        type=float,
        default=0.5,
        help="Probability of sampling from Hypersim dataset / Hypersim 数据集采样概率",
    )
    parser.add_argument(
        "--dataset_probs",
        type=str,
        default=None,
        help="Dataset sampling probabilities, comma-separated / 数据集采样概率，逗号分隔",
    )
    parser.add_argument(
        "--discriminative_mode",
        action="store_true",
        help="Enable discriminative training mode (Lotus-D) / 启用判别式训练模式",
    )
    parser.add_argument(
        "--discriminative_timestep",
        type=float,
        default=1.0,
        help="Fixed timestep for discriminative training / 判别式训练固定 timestep (default: 1.0)",
    )
    parser.add_argument(
        "--epoch_offset",
        type=int,
        default=0,
        help="Epoch offset for resumed training / 恢复训练时的 epoch 偏移",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    args.discriminative_mode = True

    trainer = Step1XEditDiscriminativeTrainer()
    trainer.train(args)
