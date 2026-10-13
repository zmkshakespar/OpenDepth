"""
Step1X-Edit 概率采样训练器 (Probabilistic Sampling Trainer)

参考 Lotus-G 论文的数据集混合策略：
- 支持 N 个数据集按可配置概率采样
- 默认: P(Hypersim) = 0.9, P(VKitti) = 0.1
- 使用 infinite_dataloader 避免分布式训练中的 StopIteration 死锁
- 所有 GPU 共享同一个 sampling_rng 保证数据集选择同步

使用方法:
  python finetuning_p.py \
    --dataset_config library/data_configs/lotus_paper_prob.toml \
    --dataset_probs "0.9,0.1" \
    ...
"""

import argparse
import copy
import math
import random
import time
from typing import Any, Optional, Union, List
from multiprocessing import Value

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


import torch
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

from finetuning import Step1XEditNetworkTrainer

setup_logging()
import logging

logger = logging.getLogger(__name__)


def infinite_dataloader(dataloader):
    """
    将 dataloader 包装为无限迭代器，自动重启以避免 StopIteration。

    在分布式训练中，不同 GPU 上的 dataloader 长度可能不同（因为
    DistributedSampler 的 padding 行为），直接 for 循环迭代会导致
    部分 GPU 先耗尽数据触发 StopIteration，而其他 GPU 还在等待
    NCCL 通信，最终导致死锁。

    使用 infinite_dataloader 后，所有 GPU 通过外部的 steps_per_epoch
    计数来统一停止，而不依赖 dataloader 自身的长度。
    """
    while True:
        for batch in dataloader:
            yield batch


class Step1XEditProbSamplingTrainer(Step1XEditNetworkTrainer):
    """支持概率采样的 Step1X-Edit 训练器, 参考 Lotus-G 论文：P(Hypersim) = 0.9, P(VKitti) = 0.1"""

    def __init__(self):
        super().__init__()

    def _on_epoch_start(self, epoch: int, args, accelerator, vae):
        """
        Epoch 开始时的钩子，子类可覆盖以执行自定义逻辑
        （例如预生成扰动图、更新数据增强策略等）
        """
        pass

    def _save_model(self, network, save_dtype, output_dir, filename, metadata=None):
        """保存模型权重为 safetensors 格式"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        logger.info(f"Saving model to {filepath}")

        state_dict = network.state_dict()
        if save_dtype is not None:
            state_dict = {k: v.detach().clone().cpu().to(save_dtype) for k, v in state_dict.items()}
        else:
            state_dict = {k: v.detach().clone().cpu() for k, v in state_dict.items()}

        from safetensors.torch import save_file
        if metadata:
            save_file(state_dict, filepath, metadata=metadata)
        else:
            save_file(state_dict, filepath)

        return filepath

    def _train_with_prob_sampling(self, args):
        """
        使用概率采样的完整训练循环。

        与基类的 train() 方法不同之处：
        1. 为每个数据集创建独立的 dataloader，用 infinite_dataloader 包装
        2. 每个 step 通过 sampling_rng 按概率选择一个数据集
        3. 所有 GPU 使用相同的 sampling_rng seed，保证选择同步
        4. NaN loss 时使用 dummy loss 而非 continue，防止梯度不同步
        """
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        setup_logging(args, reset=True)

        # ========== 解析数据集概率 ==========
        dataset_probs_str = getattr(args, 'dataset_probs', None)
        if dataset_probs_str:
            dataset_probs = [float(p) for p in dataset_probs_str.split(',')]
        else:
            prob_h = getattr(args, 'prob_hypersim', 0.9)
            dataset_probs = [prob_h, 1.0 - prob_h]

        prob_sum = sum(dataset_probs)
        dataset_probs = [p / prob_sum for p in dataset_probs]
        logger.info(f"Dataset sampling probabilities: {dataset_probs}")

        cache_latents = args.cache_latents

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # ========== 策略初始化 ==========
        tokenize_strategy = self.get_tokenize_strategy(args)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(tokenize_strategy)

        latents_caching_strategy = self.get_latents_caching_strategy(args)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        # ========== 准备数据集 ==========
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True, True))
        logger.info(f"Loading dataset config from {args.dataset_config}")
        user_config = config_util.load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

        if args.debug_dataset:
            train_dataset_group.set_current_strategies()
            train_util.debug_dataset(train_dataset_group)
            return

        if len(train_dataset_group) == 0:
            logger.error("No data found. Please verify arguments.")
            return

        if cache_latents:
            assert train_dataset_group.is_latent_cacheable(), \
                "when caching latents, either color_aug or random_crop cannot be used"

        self.assert_extra_args(args, train_dataset_group, val_dataset_group)

        # ========== 准备 Accelerator ==========
        logger.info("preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

        # ========== 加载模型 ==========
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        # ========== 缓存 latents ==========
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()
            train_dataset_group.new_cache_latents(vae, accelerator)
            vae.to("cpu")
            clean_memory_on_device(accelerator.device)
            accelerator.wait_for_everyone()

        # ========== 文本编码 ==========
        text_encoding_strategy = self.get_text_encoding_strategy(args)
        strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

        text_encoder_outputs_caching_strategy = self.get_text_encoder_outputs_caching_strategy(args)
        if text_encoder_outputs_caching_strategy is not None:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)
        self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, train_dataset_group, weight_dtype)

        # ========== 准备 Network (LoRA) ==========
        import importlib
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            for i, weight_path in enumerate(args.base_weights):
                multiplier = 1.0
                if args.base_weights_multiplier is not None and len(args.base_weights_multiplier) > i:
                    multiplier = args.base_weights_multiplier[i]
                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")
                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")
            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoders, unet, **net_kwargs)
        else:
            if "dropout" not in net_kwargs:
                net_kwargs["dropout"] = args.network_dropout
            network = network_module.create_network(
                1.0, args.network_dim, args.network_alpha,
                vae, text_encoders, unet,
                neuron_dropout=args.network_dropout, **net_kwargs,
            )
        if network is None:
            return

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            logger.warning("scale_weight_norms is specified but the network does not support it")
            args.scale_weight_norms = False

        self.post_process_network(args, accelerator, network, text_encoders, unet)

        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            if args.cpu_offload_checkpointing:
                unet.enable_gradient_checkpointing(cpu_offload=True)
            else:
                unet.enable_gradient_checkpointing()
            for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders)):
                if flag and t_enc.supports_gradient_checkpointing:
                    t_enc.gradient_checkpointing_enable()
            network.enable_gradient_checkpointing()

        # ========== 优化器 ==========
        accelerator.print("prepare optimizer, data loader etc.")

        support_multiple_lrs = hasattr(network, "prepare_optimizer_params_with_multiple_te_lrs")
        if support_multiple_lrs:
            text_encoder_lr = args.text_encoder_lr
        else:
            if args.text_encoder_lr is None or isinstance(args.text_encoder_lr, (float, int)):
                text_encoder_lr = args.text_encoder_lr
            else:
                text_encoder_lr = None if len(args.text_encoder_lr) == 0 else args.text_encoder_lr[0]

        try:
            if support_multiple_lrs:
                results = network.prepare_optimizer_params_with_multiple_te_lrs(text_encoder_lr, args.unet_lr, args.learning_rate)
            else:
                results = network.prepare_optimizer_params(text_encoder_lr, args.unet_lr, args.learning_rate)
            if type(results) is tuple:
                trainable_params = results[0]
                lr_descriptions = results[1]
            else:
                trainable_params = results
                lr_descriptions = None
        except TypeError:
            trainable_params = network.prepare_optimizer_params(text_encoder_lr, args.unet_lr)
            lr_descriptions = None

        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)
        optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

        # ========== 创建各数据集独立的 Dataloader ==========
        train_dataset_group.set_current_strategies()

        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        datasets = train_dataset_group.datasets
        n_datasets = len(datasets)

        if len(dataset_probs) < n_datasets:
            remaining = 1.0 - sum(dataset_probs)
            extra = n_datasets - len(dataset_probs)
            dataset_probs.extend([remaining / extra] * extra)
        elif len(dataset_probs) > n_datasets:
            dataset_probs = dataset_probs[:n_datasets]
            prob_sum = sum(dataset_probs)
            dataset_probs = [p / prob_sum for p in dataset_probs]

        accelerator.print(f"Number of datasets: {n_datasets}")
        for i, ds in enumerate(datasets):
            accelerator.print(f"  Dataset {i}: {len(ds)} samples, prob={dataset_probs[i]:.3f}")

        dataloaders = []
        for ds in datasets:
            dl = torch.utils.data.DataLoader(
                ds,
                batch_size=1,
                shuffle=True,
                collate_fn=collator,
                num_workers=n_workers,
                persistent_workers=args.persistent_data_loader_workers if n_workers > 0 else False,
            )
            dataloaders.append(dl)

        total_dataloader_len = max(len(dl) for dl in dataloaders)
        accelerator.print(f"Total dataloader length (max dataset): {total_dataloader_len}")

        # steps_per_epoch: for 循环的迭代次数（每次 1 个前向传播）
        steps_per_epoch = math.ceil(total_dataloader_len / accelerator.num_processes)
        # num_update_steps_per_epoch: 每 epoch 的优化器更新次数
        num_update_steps_per_epoch = math.ceil(steps_per_epoch / args.gradient_accumulation_steps)

        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * num_update_steps_per_epoch

        accelerator.print(f"steps_per_epoch (forward passes): {steps_per_epoch}")
        accelerator.print(f"num_update_steps_per_epoch (optimizer updates): {num_update_steps_per_epoch}")
        accelerator.print(f"max_train_steps: {args.max_train_steps}")

        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # ========== LR Scheduler ==========
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        # ========== FP16/BF16 ==========
        if args.full_fp16:
            assert args.mixed_precision == "fp16", "full_fp16 requires mixed precision='fp16'"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert args.mixed_precision == "bf16", "full_bf16 requires mixed precision='bf16'"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet_weight_dtype = te_weight_dtype = weight_dtype
        if args.fp8_base or args.fp8_base_unet:
            assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0"
            assert args.mixed_precision != "no", "fp8_base requires mixed precision"
            accelerator.print("enable fp8 training for U-Net.")
            unet_weight_dtype = torch.float8_e4m3fn
            te_weight_dtype = weight_dtype if args.fp8_base_unet else torch.float8_e4m3fn
            logger.info(f"set U-Net weight dtype to {unet_weight_dtype}")
            unet.to(dtype=unet_weight_dtype)

        unet.requires_grad_(False)
        unet.to(dtype=unet_weight_dtype)
        for i, t_enc in enumerate(text_encoders):
            t_enc.requires_grad_(False)
            if t_enc.device.type != "cpu":
                t_enc.to(dtype=te_weight_dtype)
                if te_weight_dtype != weight_dtype:
                    self.prepare_text_encoder_fp8(i, t_enc, te_weight_dtype, weight_dtype)

        # ========== Accelerator prepare ==========
        if train_unet:
            unet = self.prepare_unet_with_accelerator(args, accelerator, unet)
        else:
            unet.to(accelerator.device, dtype=unet_weight_dtype)

        if train_text_encoder:
            text_encoders = [
                (accelerator.prepare(t_enc) if flag else t_enc)
                for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))
            ]
        # else: text_encoder already on correct device and dtype

        network, optimizer, lr_scheduler = accelerator.prepare(network, optimizer, lr_scheduler)
        dataloaders = [accelerator.prepare(dl) for dl in dataloaders]
        training_model = network

        if args.gradient_checkpointing:
            unet.train()
            for i, (t_enc, flag) in enumerate(zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))):
                t_enc.train()
                if flag:
                    self.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)
        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()

        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        if not cache_latents:
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        # ========== 计算 epoch 数 ==========
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        epoch_offset = getattr(args, 'epoch_offset', 0)

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train images * repeats: {train_dataset_group.num_train_images}")
        for i, ds in enumerate(datasets):
            accelerator.print(f"  dataset {i}: {ds.num_train_images} images, prob={dataset_probs[i]:.3f}")
        accelerator.print(f"  num batches per epoch (steps_per_epoch): {steps_per_epoch}")
        accelerator.print(f"  num update steps per epoch: {num_update_steps_per_epoch}")
        accelerator.print(f"  num epochs: {num_train_epochs}")
        accelerator.print(f"  gradient accumulation steps: {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps: {args.max_train_steps}")
        accelerator.print(f"  epoch offset: {epoch_offset}")

        noise_scheduler = self.get_noise_scheduler(args, accelerator.device)

        train_util.init_trackers(accelerator, args, "prob_sampling_train")

        loss_recorder = train_util.LossRecorder()
        loss_clean_recorder = train_util.LossRecorder()
        loss_perturbed_recorder = train_util.LossRecorder()

        if self.is_text_encoder_not_needed_for_training(args):
            logger.info("text_encoder is not needed for training. deleting to save memory.")
            for t_enc in text_encoders:
                del t_enc
            text_encoders = []
            text_encoder = None

        # dummy_param for NaN protection
        dummy_param = next(network.parameters())

        is_tracking = len(accelerator.trackers) > 0

        global_step = 0
        progress_bar = tqdm(
            range(args.max_train_steps),
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc="steps",
        )

        # ========== 训练循环 ==========
        for epoch in range(num_train_epochs):
            actual_epoch = epoch + epoch_offset
            accelerator.print(f"\nepoch {actual_epoch + 1} (internal {epoch + 1}/{num_train_epochs})")
            current_epoch.value = actual_epoch + 1

            accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)

            self._on_epoch_start(actual_epoch, args, accelerator, vae)

            # 创建 infinite_dataloader 迭代器（每个 epoch 重新创建以打乱数据）
            dl_iters = [iter(infinite_dataloader(dl)) for dl in dataloaders]

            # 所有 GPU 共享的采样 RNG，保证选择同步
            sampling_rng = random.Random(args.seed + actual_epoch)

            # 累积概率用于快速采样
            cum_probs = []
            cumsum = 0.0
            for p in dataset_probs:
                cumsum += p
                cum_probs.append(cumsum)

            for step in range(steps_per_epoch):
                current_step.value = global_step

                # 概率采样选择数据集
                r = sampling_rng.random()
                ds_idx = 0
                for i, cp in enumerate(cum_probs):
                    if r < cp:
                        ds_idx = i
                        break
                else:
                    ds_idx = n_datasets - 1

                batch = next(dl_iters[ds_idx])

                with accelerator.accumulate(training_model):
                    self.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True)

                    loss = self.process_batch(
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
                        tokenize_strategy,
                        is_train=True,
                        train_text_encoder=train_text_encoder,
                        train_unet=train_unet,
                    )

                    current_loss = loss.detach().item()

                    # NaN protection: 不能用 continue（会导致分布式训练中梯度不同步）
                    if torch.isnan(loss) or math.isnan(current_loss):
                        logger.warning(f"NaN loss at step {global_step}, using dummy loss")
                        loss = (dummy_param * 0).sum()
                        current_loss = 0.0

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if args.max_grad_norm != 0.0:
                            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # 提取 loss 分量（带安全默认值，兼容旧版 process_batch）
                    loss_clean_val = getattr(loss, '_loss_clean', current_loss)
                    loss_perturbed_val = getattr(loss, '_loss_perturbed', 0.0)
                    has_perturbed = getattr(loss, '_has_perturbed', False)
                    num_perturbed = getattr(loss, '_num_perturbed', 0)
                    loss_feat_val = getattr(loss, '_loss_feat', 0.0)
                    loss_rgb_self_val = getattr(loss, '_loss_rgb_self', 0.0)
                    loss_rgb_cross_val = getattr(loss, '_loss_rgb_cross', 0.0)

                    loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                    loss_clean_recorder.add(epoch=epoch, step=step, loss=loss_clean_val)
                    if has_perturbed:
                        loss_perturbed_recorder.add(epoch=epoch, step=step, loss=loss_perturbed_val)

                    avr_loss = loss_recorder.moving_average
                    avr_loss_clean = loss_clean_recorder.moving_average
                    avr_loss_perturbed = loss_perturbed_recorder.moving_average

                    # 获取当前学习率
                    current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler.get_last_lr() else 0.0

                    # 进度条
                    postfix = {
                        "loss": f"{current_loss:.4f}",
                        "avr": f"{avr_loss:.4f}",
                        "Lc": f"{loss_clean_val:.4f}",
                        "Lp": f"{loss_perturbed_val:.4f}({num_perturbed})" if has_perturbed else "N/A",
                        "Lf": f"{loss_feat_val:.4f}" if loss_feat_val > 0 else "-",
                        "LrS": f"{loss_rgb_self_val:.4f}" if loss_rgb_self_val > 0 else "-",
                        "LrX": f"{loss_rgb_cross_val:.4f}" if loss_rgb_cross_val > 0 else "-",
                        "ds": ds_idx,
                        "lr": f"{current_lr:.2e}",
                    }
                    progress_bar.set_postfix(postfix)

                    # TensorBoard logging
                    if is_tracking:
                        logs = {
                            "loss/total": current_loss,
                            "loss/average": avr_loss,
                            "loss/clean": loss_clean_val,
                            "loss/clean_avg": avr_loss_clean,
                            "lr": current_lr,
                        }
                        if has_perturbed:
                            logs["loss/perturbed"] = loss_perturbed_val
                            logs["loss/perturbed_avg"] = avr_loss_perturbed
                            logs["perturbed/num_samples"] = num_perturbed
                        if loss_rgb_self_val > 0:
                            logs["loss/rgb_self"] = loss_rgb_self_val
                        if loss_rgb_cross_val > 0:
                            logs["loss/rgb_cross"] = loss_rgb_cross_val
                        accelerator.log(logs, step=global_step)

                    # 定期保存模型
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if is_main_process:
                            ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                            ckpt_file = os.path.join(args.output_dir, ckpt_name)
                            os.makedirs(args.output_dir, exist_ok=True)
                            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
                            unwrapped_nw = accelerator.unwrap_model(network)
                            metadata_to_save = {"ss_steps": str(global_step), "ss_epoch": str(actual_epoch + 1)}
                            sai_metadata = self.get_sai_model_spec(args)
                            metadata_to_save.update(sai_metadata)
                            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)

                # 内存清理（每 200 步）
                if (step + 1) % 200 == 0:
                    clean_memory_on_device(accelerator.device)

                if global_step >= args.max_train_steps:
                    break

            # ========== Epoch 结束 ==========
            accelerator.wait_for_everyone()

            if args.save_every_n_epochs is not None:
                saving = (actual_epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, actual_epoch + 1)
                    ckpt_file = os.path.join(args.output_dir, ckpt_name)
                    os.makedirs(args.output_dir, exist_ok=True)
                    accelerator.print(f"\nsaving epoch checkpoint: {ckpt_file}")
                    unwrapped_nw = accelerator.unwrap_model(network)
                    metadata_to_save = {"ss_steps": str(global_step), "ss_epoch": str(actual_epoch + 1)}
                    sai_metadata = self.get_sai_model_spec(args)
                    metadata_to_save.update(sai_metadata)
                    unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)

            if global_step >= args.max_train_steps:
                break

        # ========== 训练结束 ==========
        accelerator.end_training()

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)
            os.makedirs(args.output_dir, exist_ok=True)
            accelerator.print(f"\nsaving final model: {ckpt_file}")
            unwrapped_nw = accelerator.unwrap_model(network)
            metadata_to_save = {
                "ss_steps": str(global_step),
                "ss_epoch": str(num_train_epochs + epoch_offset),
                "ss_training_finished_at": str(time.time()),
            }
            sai_metadata = self.get_sai_model_spec(args)
            metadata_to_save.update(sai_metadata)
            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
            logger.info("model saved.")

    def train(self, args):
        """覆盖基类的 train 方法，使用概率采样训练循环"""
        self._train_with_prob_sampling(args)


def setup_parser() -> argparse.ArgumentParser:
    """设置参数解析器"""
    parser = kohya_trainer.setup_parser()
    train_util.add_dit_training_arguments(parser)
    step1x_edit_train_utils.add_step1x_edit_train_arguments(parser)
    parser.add_argument('--qwen2p5vl', type=str, help='Path to Qwen2.5VL model / Qwen2.5VL模型的路径')

    parser.add_argument(
        "--mix_dataset", action="store_true",
        help="启用混合数据集训练（概率采样）",
    )
    parser.add_argument(
        "--prob_hypersim", type=float, default=0.9,
        help="Hypersim 数据集的采样概率 (default: 0.9)",
    )
    parser.add_argument(
        "--dataset_probs", type=str, default=None,
        help='数据集采样概率，逗号分隔 (e.g. "0.8,0.1,0.1")',
    )
    parser.add_argument(
        "--epoch_offset", type=int, default=0,
        help="Epoch 偏移量，用于从断点续训时保持 epoch 编号连续",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = Step1XEditProbSamplingTrainer()
    trainer.train(args)
