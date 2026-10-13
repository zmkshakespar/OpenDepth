import os
import glob
from typing import Any, List, Optional, Tuple, Union
import PIL
import torch
import numpy as np
from transformers import AutoProcessor
import time

from library import train_util, step1x_utils
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

from qwen_vl_utils import process_vision_info

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

Qwen25VL_7b_PREFIX = '''Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:
- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.
- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n
Here are examples of how to transform or refine prompts:
- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n
Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:
User Prompt:'''

def split_string(s):
    s = s.replace("“", '"').replace("”", '"').replace("'", '''"''')  # use english quotes
    result = []
    in_quotes = False
    temp = ""

    for idx,char in enumerate(s):
        if char == '"' and idx>155:
            temp += char
            if not in_quotes:
                result.append(temp)
                temp = ""

            in_quotes = not in_quotes
            continue
        if in_quotes:
            if char.isspace():
                pass  # have space token

            result.append("“" + char + "”")
        else:
            temp += char

    if temp:
        result.append(temp)

    return result

class Step1xEditTokenizeStrategy(TokenizeStrategy):
    def __init__(self, max_length: int = 640, tokenizer_cache_dir: Optional[str] = None) -> None:
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_cache_dir, min_pixels=256 * 28 * 28, max_pixels=324 * 28 * 28
        )
        self.prefix = Qwen25VL_7b_PREFIX

    def tokenize(self, text: Union[str, List[str]], ref_images: Union[PIL.Image.Image, List[PIL.Image.Image]]) -> List[torch.Tensor]:
        # 性能诊断：打印Tokenize调用
        start_time = time.time()
        # print(f"[DEBUG] Tokenize called for {len(text) if isinstance(text, list) else 1} items. Process ID: {os.getpid()}")

        text = [text] if isinstance(text, str) else text
        ref_images = [ref_images] if isinstance(ref_images, PIL.Image.Image) else ref_images

        res_list = []

        for idx, (txt, imgs) in enumerate(zip(text, ref_images)):
            messages = [{"role": "user", "content": []}]

            messages[0]["content"].append({"type": "text", "text": f"{self.prefix}"})

            messages[0]["content"].append({"type": "image", "image": imgs})

            # 再添加 text
            messages[0]["content"].append({"type": "text", "text": f"{txt}"})

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )

            # 直接使用原始PIL.Image对象，避免process_vision_info可能的batch处理问题
            # 确保imgs是PIL.Image对象
            if not isinstance(imgs, PIL.Image.Image):
                # 如果imgs不是PIL.Image，尝试从messages中提取
                for content_item in messages[0]["content"]:
                    if content_item.get("type") == "image" and isinstance(content_item.get("image"), PIL.Image.Image):
                        imgs = content_item["image"]
                        break

            # 对于单个图像，直接传递PIL.Image对象（不是列表），processor会自动处理
            inputs = self.processor(
                text=text,  # 不需要列表，processor会自动处理
                images=imgs,  # 直接传递单个PIL.Image对象
                videos=None,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            old_inputs_ids = inputs.input_ids
            text_split_list = split_string(text)

            token_list = []
            for text_each in text_split_list:
                txt_inputs = self.processor(
                    text=text_each,
                    images=None,
                    videos=None,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                token_each = txt_inputs.input_ids
                if token_each[0][0] == 2073 and token_each[0][-1] == 854:
                    token_each = token_each[:, 1:-1]
                    token_list.append(token_each)
                else:
                    token_list.append(token_each)

            new_txt_ids = torch.cat(token_list, dim=1)  # 保持在 CPU 上

            # 强制确保所有tensor都在CPU上，防止old_inputs_ids意外携带GPU设备信息
            new_txt_ids = new_txt_ids.to("cpu")
            old_inputs_ids = old_inputs_ids.to("cpu")

            idx1 = (old_inputs_ids == 151653).nonzero(as_tuple=True)[1][0]
            idx2 = (new_txt_ids == 151653).nonzero(as_tuple=True)[1][0]
            inputs.input_ids = (
                torch.cat([old_inputs_ids[0, :idx1], new_txt_ids[0, idx2:]], dim=0)
                .unsqueeze(0)
            )  # 保持在 CPU 上，由训练循环移动到 GPU
            inputs.attention_mask = (inputs.input_ids > 0).long()  # 保持在 CPU 上

            res_list.append(inputs)

        # 性能诊断：如果耗时超过1秒，打印警告
        elapsed = time.time() - start_time
        if elapsed > 1.0:
            print(f"[WARN] Slow Tokenize: {elapsed:.2f}s for {len(text)} items. PID: {os.getpid()}")

        return res_list

class Step1XEditEncodingStrategy(TextEncodingStrategy):
    def __init__(self, max_length=640, hidden_size=None) -> None:
        self.max_length = max_length
        self.hidden_size=hidden_size
        self.dtype = None

    def encode_tokens(self, tokenize_strategy, models, tokens):
        qwen2p5vl = models[0]
        if self.dtype is None:
            self.dtype = qwen2p5vl.model.lm_head.weight.dtype
        if self.hidden_size is None:
            self.hidden_size = qwen2p5vl.model.config.hidden_size
        embs = torch.zeros(
            len(tokens),
            self.max_length,
            self.hidden_size,
            dtype=self.dtype,
            device=torch.cuda.current_device()
        )
        masks = torch.zeros(
            len(tokens),
            self.max_length,
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        self.device = torch.device(torch.cuda.current_device())
        for idx, inputs in enumerate(tokens):
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(self.dtype)
            # with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            outputs = qwen2p5vl.model(
                input_ids = inputs.input_ids.to(torch.cuda.current_device()),
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values.to(torch.cuda.current_device()),
                image_grid_thw=inputs.image_grid_thw.to(torch.cuda.current_device()),
                output_hidden_states=True,
            )
            emb = outputs['hidden_states'][-1]
            embs[idx, : min(self.max_length, emb.shape[1] - 217)] = emb[0, 217:][
                : self.max_length
            ]

            masks[idx, : min(self.max_length, emb.shape[1] - 217)] = torch.ones(
                (min(self.max_length, emb.shape[1] - 217)),
                dtype=torch.long,
                device=torch.cuda.current_device(),
            )
        return embs, masks

class Step1xEditEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    Step1XEdit_ENCODER_OUTPUTS_NPZ_SUFFIX = "_step1x_te.npz"
    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)

        self.warn_fp8_weights = False

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + Step1xEditEncoderOutputsCachingStrategy.Step1XEdit_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        try:
            # 使用 mmap_mode='r' 减少内存占用
            with np.load(npz_path, mmap_mode='r') as npz:
                if 'embeds' not in npz:
                    return False
                if 'masks' not in npz:
                    return False
        except Exception as e:
            logger.error(f'Error loading file: {npz_path}')
            raise e
        return True

    def load_outputs_npz(self, npz_path):
        # 使用 mmap_mode='r' 减少内存占用，数据按需从磁盘读取
        with np.load(npz_path, mmap_mode='r') as data:
            # 需要 copy() 因为 mmap 返回的是只读视图
            embeds = data['embeds'].copy()
            masks = data['masks'].copy()
        return [embeds, masks]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List
    ):
        if not self.warn_fp8_weights:
            if models[0].model.lm_head.weight.dtype == torch.float8_e4m3fn:
                logger.warning(
                    "Qwen2VL model is using fp8 weights for caching. This may affect the quality of the cached outputs."
                )
            self.warn_fp8_weights = True
        step1x_edit_encoding_strategy: Step1XEditEncodingStrategy = text_encoding_strategy

        # 按图像尺寸分组批量处理（空caption场景优化）
        from collections import defaultdict

        captions = [info.caption for info in infos]
        all_same_caption = len(set(captions)) == 1

        if all_same_caption and len(infos) > 1:
            # 按图像尺寸分组
            size_groups = defaultdict(list)
            for info in infos:
                try:
                    img = PIL.Image.open(info.ref_absolute_path).convert('RGB')
                    size_key = (img.width, img.height)
                    size_groups[size_key].append((info, img))
                except:
                    pass  # 跳过无法打开的图像

            # 对每个尺寸组进行批量处理
            try:
                batch_size = 8  # 批次大小
                for size_key, group_items in size_groups.items():
                    logger.info(f"批量处理尺寸 {size_key[0]}x{size_key[1]}: {len(group_items)} 张图像")

                    for start_idx in range(0, len(group_items), batch_size):
                        end_idx = min(start_idx + batch_size, len(group_items))
                        batch_items = group_items[start_idx:end_idx]

                        batch_infos = [item[0] for item in batch_items]
                        batch_images = [item[1] for item in batch_items]
                        batch_captions = [batch_infos[0].caption] * len(batch_infos)

                        # 批量编码
                        tokens_and_masks = tokenize_strategy.tokenize(batch_captions, batch_images)
                        with torch.no_grad():
                            embs, masks = step1x_edit_encoding_strategy.encode_tokens(
                                tokenize_strategy, models, tokens_and_masks
                            )

                        if embs.dtype == torch.bfloat16:
                            embs = embs.float()
                        if masks.dtype == torch.bfloat16:
                            masks = masks.float()

                        # 保存每个样本的结果
                        for i, info in enumerate(batch_infos):
                            emb_i = embs[i]
                            mask_i = masks[i]

                            if self.cache_to_disk:
                                np.savez(
                                    info.text_encoder_outputs_npz,
                                    embeds=emb_i.cpu().numpy(),
                                    masks=mask_i.cpu().numpy()
                                )
                            else:
                                info.text_encoder_outputs = (emb_i, mask_i)
                return  # 批量处理完成
            except Exception as e:
                logger.warning(f"批量处理失败，回退到逐个处理: {e}")
                import traceback
                traceback.print_exc()

        # 逐个处理（回退方案或caption不同时）
        for info in infos:
            caption = info.caption
            image = PIL.Image.open(info.ref_absolute_path).convert('RGB')

            tokens_and_masks = tokenize_strategy.tokenize([caption], [image])
            with torch.no_grad():
                embs, masks = step1x_edit_encoding_strategy.encode_tokens(
                    tokenize_strategy, models, tokens_and_masks
                )

            if embs.dtype == torch.bfloat16:
                embs = embs.float()
            if masks.dtype == torch.bfloat16:
                masks = masks.float()

            emb_i = embs[0]
            mask_i = masks[0]

            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz,
                    embeds=emb_i.cpu().numpy(),
                    masks=mask_i.cpu().numpy()
                )
            else:
                info.text_encoder_outputs = (emb_i, mask_i)



from library.train_util import load_image, trim_and_resize_if_required, ImageInfo, IMAGE_TRANSFORMS
# for new_cache_latents
def step1x_load_images_and_masks_for_caching(
    image_infos: List[ImageInfo], use_alpha_mask: bool, random_crop: bool
) -> Tuple[torch.Tensor, List[np.ndarray], List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    r"""
    requires image_infos to have: [absolute_path or image], bucket_reso, resized_size

    returns: image_tensor, alpha_masks, original_sizes, crop_ltrbs

    image_tensor: torch.Tensor = torch.Size([B, 3, H, W]), ...], normalized to [-1, 1]
    alpha_masks: List[np.ndarray] = [np.ndarray([H, W]), ...], normalized to [0, 1]
    original_sizes: List[Tuple[int, int]] = [(W, H), ...]
    crop_ltrbs: List[Tuple[int, int, int, int]] = [(L, T, R, B), ...]
    """
    images: List[torch.Tensor] = []
    alpha_masks: List[np.ndarray] = []
    original_sizes: List[Tuple[int, int]] = []
    crop_ltrbs: List[Tuple[int, int, int, int]] = []

    ref_images: List[torch.Tensor] = []
    ref_alpha_masks: List[np.ndarray] = []

    # ============ 视差归一化深度图的 Mask 阈值 (与 Lotus 保持一致) ============
    # 编码 (Lotus 原始编码，无反转):
    #   高值(白/255) = 近距离 (高 disparity)
    #   低值(黑/0) = 远距离/天空 (低 disparity)
    #
    # 无效区域:
    #   - 天空/无穷远: 像素值太低 (< SKY_THRESHOLD)  → 在 Lotus 编码中黑色=远
    #   - 近距离盲区: 像素值太高 (> NEAR_THRESHOLD)  → 在 Lotus 编码中白色=近
    SKY_THRESHOLD = 2       # 像素值 < 2 被认为是天空/无穷远 (Lotus 编码: 黑色=远)
    NEAR_THRESHOLD = 250    # 像素值 > 250 被认为是近距离盲区截断 (Lotus 编码: 白色=近)

    for info in image_infos:
        is_16bit = '_16bit.png' in info.absolute_path

        # ============ 加载 target 深度图 ============
        image = load_image(info.absolute_path, use_alpha_mask) if info.image is None else np.array(info.image, np.uint8)
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop, image, info.bucket_reso, info.resized_size,
            resize_interpolation=info.resize_interpolation
        )

        # ============ Mask: 全 1（与 Lotus 一致，不屏蔽任何区域）============
        # 无效区域已在预处理时 clip 到 [-1,1] 边界，模型直接学习输出边界值
        alpha_mask = torch.ones((image.shape[0], image.shape[1]), dtype=torch.float32)

        # ============ 转为 tensor [-1, 1] ============
        if is_16bit:
            # 16-bit PNG 高精度路径: 重新读取 uint16 原始数据
            with PIL.Image.open(info.absolute_path) as img16:
                arr16 = np.array(img16, dtype=np.uint16)  # (H, W) uint16
            # resize 到目标尺寸（如果需要）
            if arr16.shape[0] != image.shape[0] or arr16.shape[1] != image.shape[1]:
                pil16 = PIL.Image.fromarray(arr16, mode='I;16')
                pil16 = pil16.resize((image.shape[1], image.shape[0]), PIL.Image.BILINEAR)
                arr16 = np.array(pil16, dtype=np.uint16)
            # uint16 [0, 65535] → float [-1, 1]
            depth_float = arr16.astype(np.float32) / 65535.0 * 2.0 - 1.0  # [-1, 1]
            # 扩展为 3 通道
            image_tensor = torch.from_numpy(np.stack([depth_float]*3, axis=0))  # (3, H, W)
        else:
            image = image[:, :, :3]
            image_pil = PIL.Image.fromarray(image)
            img_bytes = image_pil.tobytes()
            h, w = image_pil.size[1], image_pil.size[0]
            image_tensor = torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8).reshape(h, w, 3).permute(2, 0, 1).float() / 255.0
            image_tensor = (image_tensor - 0.5) / 0.5

        # ref_image (RGB 输入图) 始终走图片加载路径
        ref_image = load_image(info.ref_absolute_path, use_alpha_mask) if info.ref_image is None else np.array(info.ref_image, np.uint8)
        ref_image, ref_original_size, ref_crop_ltrb = trim_and_resize_if_required(
            random_crop, ref_image, info.bucket_reso, info.resized_size,
            resize_interpolation=info.resize_interpolation
        )

        original_sizes.append(original_size)
        crop_ltrbs.append(crop_ltrb)

        ref_alpha_mask = torch.ones((ref_image.shape[0], ref_image.shape[1]), dtype=torch.float32)

        alpha_masks.append(alpha_mask)
        ref_alpha_masks.append(ref_alpha_mask)
        images.append(image_tensor)

        ref_image = ref_image[:, :, :3]
        ref_image_pil = PIL.Image.fromarray(ref_image)
        img_bytes = ref_image_pil.tobytes()
        h, w = ref_image_pil.size[1], ref_image_pil.size[0]
        ref_image = torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8).reshape(h, w, 3).permute(2, 0, 1).float() / 255.0
        ref_image = (ref_image - 0.5) / 0.5
        ref_images.append(ref_image)

    img_tensor = torch.stack(images, dim=0)
    ref_img_tensor = torch.stack(ref_images, dim=0)
    return img_tensor, alpha_masks, ref_img_tensor, ref_alpha_masks, original_sizes, crop_ltrbs

class Step1XEditLatentsCachingStrategy(LatentsCachingStrategy):
    Step1XEdit_LATENTS_NPZ_SUFFIX = "_step1x_latents.npz"
    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return Step1XEditLatentsCachingStrategy.Step1XEdit_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + Step1XEditLatentsCachingStrategy.Step1XEdit_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)  # support multi-resolution

    def _default_load_latents_from_disk(
        self, latents_stride: Optional[int], npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        if latents_stride is None:
            key_reso_suffix = ""
        else:
            latents_size = (bucket_reso[1] // latents_stride, bucket_reso[0] // latents_stride)  # bucket_reso is (W, H)
            key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}"  # e.g. "_32x64", HxW

        # 使用 mmap_mode='r' 减少内存占用，数据按需从磁盘读取
        with np.load(npz_path, mmap_mode='r') as npz:
            if "latents" + key_reso_suffix not in npz:
                raise ValueError(f"latents{key_reso_suffix} not found in {npz_path}")

            # 需要 copy() 因为 mmap 模式返回只读视图，且退出 with 后失效
            latents = npz["latents" + key_reso_suffix].copy()
            ref_latents = npz["ref_latents" + key_reso_suffix].copy()
            original_size = npz["original_size" + key_reso_suffix].tolist()
            crop_ltrb = npz["crop_ltrb" + key_reso_suffix].tolist()
            flipped_latents = npz["latents_flipped" + key_reso_suffix].copy() if "latents_flipped" + key_reso_suffix in npz else None
            ref_flipped_latents = npz["ref_latents_flipped" + key_reso_suffix].copy() if "ref_latents_flipped" + key_reso_suffix in npz else None
            alpha_mask = npz["alpha_mask" + key_reso_suffix].copy() if "alpha_mask" + key_reso_suffix in npz else None
            ref_alpha_mask = npz["ref_alpha_mask" + key_reso_suffix].copy() if "ref_alpha_mask" + key_reso_suffix in npz else None
        return latents, ref_latents, original_size, crop_ltrb, flipped_latents, ref_flipped_latents, alpha_mask, ref_alpha_mask

    # TODO remove circular dependency for ImageInfo
    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).to("cpu")
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._custom_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)

    def _custom_cache_batch_latents(
        self,
        encode_by_vae,
        vae_device,
        vae_dtype,
        image_infos: List,
        flip_aug: bool,
        alpha_mask: bool,
        random_crop: bool,
        multi_resolution: bool = False,
    ):
        """
        Default implementation for cache_batch_latents. Image loading, VAE, flipping, alpha mask handling are common.
        """
        from library import train_util  # import here to avoid circular import

        img_tensor, alpha_masks, ref_img_tensor, ref_alpha_masks, original_sizes, crop_ltrbs = step1x_load_images_and_masks_for_caching(
            image_infos, alpha_mask, random_crop
        )
        img_tensor = img_tensor.to(device=vae_device, dtype=vae_dtype)
        ref_img_tensor = ref_img_tensor.to(device=vae_device, dtype=vae_dtype)

        with torch.no_grad():
            latents_tensors = encode_by_vae(img_tensor).to("cpu")
            ref_latents_tensors = encode_by_vae(ref_img_tensor).to("cpu")
        if flip_aug:
            img_tensor = torch.flip(img_tensor, dims=[3])
            ref_img_tensor = torch.flip(ref_img_tensor, dims=[3])
            with torch.no_grad():
                flipped_latents = encode_by_vae(img_tensor).to("cpu")
                ref_flipped_latents = encode_by_vae(ref_img_tensor).to("cpu")
        else:
            flipped_latents = [None] * len(latents_tensors)
            ref_flipped_latents = [None] * len(ref_latents_tensors)

        # for info, latents, flipped_latent, alpha_mask in zip(image_infos, latents_tensors, flipped_latents, alpha_masks):
        for i in range(len(image_infos)):
            info = image_infos[i]
            latents = latents_tensors[i]
            ref_latents = ref_latents_tensors[i]
            flipped_latent = flipped_latents[i]
            ref_flipped_latent = ref_flipped_latents[i]
            alpha_mask = alpha_masks[i]
            ref_alpha_mask = ref_alpha_masks[i]
            original_size = original_sizes[i]
            crop_ltrb = crop_ltrbs[i]

            latents_size = latents.shape[1:3]  # H, W

            key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}" if multi_resolution else ""  # e.g. "_32x64", HxW

            if self.cache_to_disk:
                self.save_latents_to_disk(
                    info.latents_npz, latents, ref_latents,
                    original_size, crop_ltrb,
                    flipped_latent, ref_flipped_latent,
                    alpha_mask, ref_alpha_mask,
                    key_reso_suffix
                )
            else:
                info.latents_original_size = original_size
                info.latents_crop_ltrb = crop_ltrb
                info.latents = latents
                info.ref_latents = ref_latents
                if flip_aug:
                    info.latents_flipped = flipped_latent
                    info.ref_latents_flipped = ref_flipped_latent
                info.alpha_mask = alpha_mask

    def save_latents_to_disk(
        self,
        npz_path,
        latents_tensor,
        ref_latents_tensor,
        original_size,
        crop_ltrb,
        flipped_latents_tensor=None,
        ref_flipped_latents_tensor=None,
        alpha_mask=None,
        ref_alpha_mask=None,
        key_reso_suffix="",
    ):
        kwargs = {}

        if os.path.exists(npz_path):
            # load existing npz and update it (使用 mmap 减少内存占用)
            with np.load(npz_path, mmap_mode='r') as npz:
                for key in npz.files:
                    kwargs[key] = npz[key].copy()  # 需要 copy() 因为 mmap 返回只读视图

        # 转换为标准 numpy 数组，避免版本冲突
        kwargs["latents" + key_reso_suffix] = np.ascontiguousarray(latents_tensor.float().cpu().numpy())
        kwargs["ref_latents" + key_reso_suffix] = np.ascontiguousarray(ref_latents_tensor.float().cpu().numpy())
        kwargs["original_size" + key_reso_suffix] = np.ascontiguousarray(np.array(original_size, dtype=np.int32))
        kwargs["crop_ltrb" + key_reso_suffix] = np.ascontiguousarray(np.array(crop_ltrb, dtype=np.int32))
        if flipped_latents_tensor is not None:
            kwargs["latents_flipped" + key_reso_suffix] = np.ascontiguousarray(flipped_latents_tensor.float().cpu().numpy())
            kwargs["ref_latents_flipped" + key_reso_suffix] = np.ascontiguousarray(ref_flipped_latents_tensor.float().cpu().numpy())
        if alpha_mask is not None:
            kwargs["alpha_mask" + key_reso_suffix] = np.ascontiguousarray(alpha_mask.float().cpu().numpy())
            kwargs["ref_alpha_mask" + key_reso_suffix] = np.ascontiguousarray(ref_alpha_mask.float().cpu().numpy())

        # 使用 with 语句确保文件正确关闭
        np.savez(npz_path, **kwargs)
