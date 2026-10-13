"""Depth inference engine adapted from Step1X-Edit; see NOTICE for provenance."""
import argparse
import datetime
import json
import itertools
import math
import os
import time
from pathlib import Path


import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torchvision.transforms import functional as F
from tqdm import tqdm

import sampling
from modules.autoencoder import AutoEncoder
from modules.conditioner import Qwen25VL_7b_Embedder as Qwen2VLEmbedder
from modules.model_edit import Step1XParams, Step1XEdit



def cudagc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def load_state_dict(model, ckpt_path, device="cuda", strict=False, assign=True):
    if Path(ckpt_path).suffix == ".safetensors":
        state_dict = load_file(ckpt_path, device)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(
        state_dict, strict=strict, assign=assign
    )
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    return model


def load_models(
    dit_path=None,
    ae_path=None,
    qwen2vl_model_path=None,
    device="cuda",
    max_length=256,
    dtype=torch.bfloat16,
    mode="flash",
    version="v1.0",
):
    qwen2vl_encoder = Qwen2VLEmbedder(
        qwen2vl_model_path,
        device=device,
        max_length=max_length,
        dtype=dtype,
    )

    with torch.device("meta"):
        ae = AutoEncoder(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        )

        step1x_params = Step1XParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            mode=mode,
            version=version,
        )
        dit = Step1XEdit(step1x_params)

    ae = load_state_dict(ae, ae_path, 'cpu')
    dit = load_state_dict(
        dit, dit_path, 'cpu'
    )

    ae = ae.to(dtype=torch.float32)

    return ae, dit, qwen2vl_encoder

def equip_dit_with_lora_sd_scripts(ae, text_encoders, dit, lora, device='cuda'):
    from safetensors.torch import load_file
    weights_sd = load_file(lora)
    is_lora = True
    from library import lora_module
    module = lora_module
    lora_model, _ = module.create_network_from_weights(1.0, None, ae, text_encoders, dit, weights_sd, True)
    lora_model.merge_to(text_encoders, dit, weights_sd)

    lora_model.set_multiplier(1.0)
    return lora_model

class ImageGenerator:
    def __init__(
        self,
        dit_path=None,
        ae_path=None,
        qwen2vl_model_path=None,
        device="cuda",
        max_length=640,
        dtype=torch.bfloat16,
        quantized=False,
        offload=False,
        lora=None,
        mode="flash",
        version="v1.0",
    ) -> None:
        self.device = torch.device(device)
        self.ae, self.dit, self.llm_encoder = load_models(
            dit_path=dit_path,
            ae_path=ae_path,
            qwen2vl_model_path=qwen2vl_model_path,
            max_length=max_length,
            dtype=dtype,
            device=self.device,
            mode=mode,
            version=version,
        )
        if not quantized:
            self.dit = self.dit.to(dtype=torch.bfloat16)
        else:
            self.dit = self.dit.to(dtype=torch.float8_e4m3fn)
        if not offload:
            self.dit = self.dit.to(device=self.device)
            self.ae = self.ae.to(device=self.device)
        self.quantized = quantized
        self.offload = offload
        if lora is not None:
            self.lora_module = equip_dit_with_lora_sd_scripts(
                self.ae,
                [self.llm_encoder],
                self.dit,
                lora,
                device=self.dit.device,
            )
        else:
            self.lora_module = None





    @staticmethod
    def unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    @staticmethod
    def load_image(image):
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, Image.Image):
            image = F.to_tensor(image.convert("RGB"))
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, str):
            image = F.to_tensor(Image.open(image).convert("RGB"))
            image = image.unsqueeze(0)
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def output_process_image(self, resize_img, image_size):
        res_image = resize_img.resize(image_size)
        return res_image

    def input_process_image(self, img, img_size=512):
        # 1. 打开图片
        w, h = img.size
        r = w / h

        if w > h:
            w_new = math.ceil(math.sqrt(img_size * img_size * r))
            h_new = math.ceil(w_new / r)
        else:
            h_new = math.ceil(math.sqrt(img_size * img_size / r))
            w_new = math.ceil(h_new * r)
        h_new = math.ceil(h_new) // 16 * 16
        w_new = math.ceil(w_new) // 16 * 16

        img_resized = img.resize((w_new, h_new))
        return img_resized, img.size


    @torch.inference_mode()
    def generate_depth(
        self,
        ref_images,
        num_steps=4,
        seed=42,
        size_level=768,
        prompt="",
        show_progress=False,
    ):
        """
        专门用于深度估计的推理函数
        参考 Lotus 的推理逻辑，去掉 CFG，简化流程

        Args:
            ref_images: PIL Image, 输入 RGB 图像
            num_steps: int, 去噪步数 (Lotus-G 用 1-10 步)
            seed: int, 随机种子
            size_level: int, 处理分辨率
            prompt: str, 可选的文本提示
            show_progress: bool, 是否显示进度条

        Returns:
            PIL Image: 深度图
        """
        # 1. 预处理输入图像
        ref_images_raw, img_info = self.input_process_image(ref_images, img_size=size_level)
        width, height = ref_images_raw.width, ref_images_raw.height

        ref_images_raw = self.load_image(ref_images_raw)
        ref_images_raw = ref_images_raw.to(self.device)

        if self.offload:
            self.ae = self.ae.to(self.device)
        ref_images_latent = self.ae.encode(ref_images_raw.to(self.device) * 2 - 1)
        if self.offload:
            self.ae = self.ae.cpu()
            cudagc()

        # 2. 初始化噪声
        seed = int(seed)
        seed = torch.Generator(device="cpu").seed() if seed < 0 else seed

        x = torch.randn(
            1,  # batch_size = 1
            16,
            height // 8,
            width // 8,
            device=self.device,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )

        # 3. 生成时间步 schedule
        timesteps = sampling.get_schedule(
            num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True
        )

        # 4. 准备输入 (不复制，只用单个 prompt)
        # 只传入一个 prompt，不做 CFG 的复制
        inputs = self._prepare_depth_inputs(prompt, x, ref_images_latent, ref_images_raw)

        # 5. 去噪
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self._denoise_depth(
                **inputs,
                timesteps=timesteps,
                show_progress=show_progress,
            )
            x = self.unpack(x.float(), height, width)

            if self.offload:
                self.ae = self.ae.to(self.device)
            x = self.ae.decode(x)
            if self.offload:
                self.ae = self.ae.cpu()
                cudagc()

            x = x.clamp(-1, 1)
            x = x.mul(0.5).add(0.5)

        # 6. 后处理
        output_image = self.output_process_image(F.to_pil_image(x[0].float()), img_info)
        return output_image

    def _prepare_depth_inputs(self, prompt, img, ref_image, ref_image_raw):
        """
        为深度估计准备输入，不做 CFG 复制
        """
        bs = 1  # 固定 batch_size = 1
        _, _, h, w = img.shape
        h_latent, w_latent = h * 8, w * 8  # latent 对应的图像尺寸

        # 对 latent 做 patchify
        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        ref_img = rearrange(ref_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        # 生成位置编码
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        ref_img_ids = torch.zeros(h // 2, w // 2, 3)
        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(h // 2)[:, None]
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(w // 2)[None, :]
        ref_img_ids = repeat(ref_img_ids, "h w c -> b (h w) c", b=bs)

        # 编码文本
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.offload:
            self.llm_encoder = self.llm_encoder.to(self.device)
        txt, mask = self.llm_encoder(prompt, ref_image_raw)
        if self.offload:
            self.llm_encoder = self.llm_encoder.cpu()
            cudagc()

        txt_ids = torch.zeros(bs, txt.shape[1], 3)

        # concat target 和 ref
        img = torch.cat([img, ref_img.to(device=img.device, dtype=img.dtype)], dim=-2)
        img_ids = torch.cat([img_ids, ref_img_ids], dim=-2)

        return {
            "img": img,
            "mask": mask,
            "img_ids": img_ids.to(img.device),
            "llm_embedding": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
        }

    def _denoise_depth(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        llm_embedding: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: list[float],
        mask=None,
        show_progress=False,
    ):
        """
        深度估计专用去噪函数，无 CFG
        参考 Lotus-G 的去噪逻辑
        """
        if self.offload:
            self.dit = self.dit.to(self.device)

        if show_progress:
            pbar = tqdm(itertools.pairwise(timesteps), desc='Depth denoising...')
        else:
            pbar = itertools.pairwise(timesteps)

        for t_curr, t_prev in pbar:
            t_vec = torch.full(
                (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
            )

            # 模型前向
            pred = self.dit(
                img=img,
                img_ids=img_ids,
                txt_ids=txt_ids,
                timesteps=t_vec,
                llm_embedding=llm_embedding,
                t_vec=t_vec,
                mask=mask,
            )

            # Flow matching 更新: x_{t-1} = x_t + (t_prev - t_curr) * v
            # 其中 v = pred 是模型预测的速度场
            img_input_length = img.shape[1] // 2

            # 只更新 target 部分，保持 ref 部分不变
            target_img = img[:, :img_input_length]
            ref_img = img[:, img_input_length:]

            # 只取预测的 target 部分
            pred_target = pred[:, :img_input_length]

            # 更新 target
            target_img = target_img + (t_prev - t_curr) * pred_target

            # 重新拼接
            img = torch.cat([target_img, ref_img], dim=1)

        if self.offload:
            self.dit = self.dit.cpu()
            cudagc()

        return img[:, :img.shape[1] // 2]

    @torch.inference_mode()
    def generate_depth_discriminative(
        self,
        ref_images,
        seed=42,
        size_level=768,
        prompt="",
        timestep=1.0,
        show_progress=False,
    ):
        """
        判别式深度估计推理函数 (单步推理)

        参考 Lotus-D 的判别式推理逻辑：
        - 不加噪声，输入为全零张量
        - 固定 timestep = 1
        - 单步推理直接获得深度预测

        Args:
            ref_images: PIL Image, 输入 RGB 图像
            seed: int, 随机种子（主要用于可重复性）
            size_level: int, 处理分辨率
            prompt: str, 可选的文本提示
            timestep: float, 固定的 timestep 值（默认 1.0）
            show_progress: bool, 是否显示进度条

        Returns:
            PIL Image: 深度图
        """
        # 1. 预处理输入图像
        ref_images_raw, img_info = self.input_process_image(ref_images, img_size=size_level)
        width, height = ref_images_raw.width, ref_images_raw.height

        ref_images_raw = self.load_image(ref_images_raw)
        ref_images_raw = ref_images_raw.to(self.device)

        if self.offload:
            self.ae = self.ae.to(self.device)
        ref_images_latent = self.ae.encode(ref_images_raw.to(self.device) * 2 - 1)
        if self.offload:
            self.ae = self.ae.cpu()
            cudagc()

        # 2. 初始化输入为全零张量（判别式模式的关键区别）
        x = torch.zeros(
            1,  # batch_size = 1
            16,
            height // 8,
            width // 8,
            device=self.device,
            dtype=torch.bfloat16,
        )

        # 3. 准备输入
        inputs = self._prepare_depth_inputs(prompt, x, ref_images_latent, ref_images_raw)

        # 4. 单步推理（判别式模式）
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self._forward_discriminative(
                **inputs,
                timestep=timestep,
            )
            x = self.unpack(x.float(), height, width)

            if self.offload:
                self.ae = self.ae.to(self.device)
            x = self.ae.decode(x)
            if self.offload:
                self.ae = self.ae.cpu()
                cudagc()

            x = x.clamp(-1, 1)
            x = x.mul(0.5).add(0.5)

        # 5. 后处理
        output_image = self.output_process_image(F.to_pil_image(x[0].float()), img_info)
        return output_image

    def _forward_discriminative(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        llm_embedding: torch.Tensor,
        txt_ids: torch.Tensor,
        timestep: float = 1.0,
        mask=None,
    ):
        """
        判别式前向推理（单步）

        与生成式的区别：
        - 输入已经是全零张量（不是噪声）
        - 固定 timestep
        - 单步直接输出预测结果（不是迭代去噪）

        注意：
        - timestep 需要除以 1000，与训练时保持一致
        - 参考 finetuning_d.py 中的 get_noise_pred_and_target_discriminative
        """
        if self.offload:
            self.dit = self.dit.to(self.device)

        # 固定 timestep（用于 t_vec，作为时间嵌入的原始值）
        t_vec = torch.full(
            (img.shape[0],), timestep, dtype=img.dtype, device=img.device
        )

        # 模型前向
        # 重要：timesteps 参数需要除以 1000，与训练时保持一致！
        pred = self.dit(
            img=img,
            img_ids=img_ids,
            txt_ids=txt_ids,
            timesteps=t_vec / 1000,  # 关键：除以 1000，与训练一致
            llm_embedding=llm_embedding,
            t_vec=t_vec,
            mask=mask,
        )

        # 判别式模式：直接返回预测结果（不需要迭代更新）
        # 只取 target 部分
        img_input_length = img.shape[1] // 2
        pred_target = pred[:, :img_input_length]

        if self.offload:
            self.dit = self.dit.cpu()
            cudagc()

        return pred_target

    @torch.inference_mode()
    def generate_depth_discriminative_batch(
        self,
        ref_images_list: list,
        seed: int = 42,
        size_level: int = 768,
        prompt: str = "Estimate the depth map of this image.",
        timestep: float = 1.0,
        show_progress: bool = False,
    ):
        """
        批量判别式深度估计推理

        Args:
            ref_images_list: PIL Image 列表
            seed: 随机种子
            size_level: 图像尺寸级别
            prompt: 文本提示
            timestep: 时间步
            show_progress: 是否显示进度条

        Returns:
            List[PIL Image]: 深度图列表
        """
        if len(ref_images_list) == 0:
            return []

        batch_size = len(ref_images_list)

        # 1. 预处理所有输入图像（需要相同尺寸才能 batch）
        processed_images = []
        img_infos = []
        for ref_img in ref_images_list:
            ref_img_processed, img_info = self.input_process_image(ref_img, img_size=size_level)
            processed_images.append(ref_img_processed)
            img_infos.append(img_info)

        # 检查尺寸是否一致（batch 推理要求相同尺寸）
        sizes = [(img.width, img.height) for img in processed_images]
        if len(set(sizes)) > 1:
            # 尺寸不一致，回退到逐张处理
            results = []
            for ref_img in ref_images_list:
                result = self.generate_depth_discriminative(
                    ref_images=ref_img,
                    seed=seed,
                    size_level=size_level,
                    prompt=prompt,
                    timestep=timestep,
                    show_progress=False,
                )
                results.append(result)
            return results

        width, height = sizes[0]

        # 2. 加载并编码所有图像
        ref_images_raw_list = []
        for img in processed_images:
            img_tensor = self.load_image(img).to(self.device)
            ref_images_raw_list.append(img_tensor)
        ref_images_raw_batch = torch.cat(ref_images_raw_list, dim=0)  # [B, 3, H, W]

        if self.offload:
            self.ae = self.ae.to(self.device)
        ref_images_latent = self.ae.encode(ref_images_raw_batch * 2 - 1)  # [B, 16, H/8, W/8]
        if self.offload:
            self.ae = self.ae.cpu()
            cudagc()

        # 3. 初始化输入为全零张量
        x = torch.zeros(
            batch_size,
            16,
            height // 8,
            width // 8,
            device=self.device,
            dtype=torch.bfloat16,
        )

        # 4. 准备批量输入
        inputs = self._prepare_depth_inputs_batch(prompt, x, ref_images_latent, ref_images_raw_batch)

        # 5. 批量推理
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self._forward_discriminative(
                **inputs,
                timestep=timestep,
            )
            x = self.unpack(x.float(), height, width)

            if self.offload:
                self.ae = self.ae.to(self.device)
            x = self.ae.decode(x)  # [B, 3, H, W]
            if self.offload:
                self.ae = self.ae.cpu()
                cudagc()

            x = x.clamp(-1, 1)
            x = x.mul(0.5).add(0.5)

        # 6. 后处理每张图
        output_images = []
        for i in range(batch_size):
            output_image = self.output_process_image(F.to_pil_image(x[i].float()), img_infos[i])
            output_images.append(output_image)

        return output_images

    def _prepare_depth_inputs_batch(self, prompt, img, ref_image, ref_image_raw):
        """
        为批量深度估计准备输入
        """
        bs = img.shape[0]
        _, _, h, w = img.shape
        h_latent, w_latent = h * 8, w * 8

        # 对 latent 做 patchify
        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        ref_img = rearrange(ref_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        # 生成位置编码
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        ref_img_ids = torch.zeros(h // 2, w // 2, 3)
        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(h // 2)[:, None]
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(w // 2)[None, :]
        ref_img_ids = repeat(ref_img_ids, "h w c -> b (h w) c", b=bs)

        # 编码文本（批量）
        prompts = [prompt] * bs
        if self.offload:
            self.llm_encoder = self.llm_encoder.to(self.device)
        txt, mask = self.llm_encoder(prompts, ref_image_raw)
        if self.offload:
            self.llm_encoder = self.llm_encoder.cpu()
            cudagc()

        txt_ids = torch.zeros(bs, txt.shape[1], 3)

        # concat target 和 ref
        img = torch.cat([img, ref_img.to(device=img.device, dtype=img.dtype)], dim=-2)
        img_ids = torch.cat([img_ids, ref_img_ids], dim=-2)

        return {
            "img": img,
            "mask": mask,
            "img_ids": img_ids.to(img.device),
            "llm_embedding": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
        }
