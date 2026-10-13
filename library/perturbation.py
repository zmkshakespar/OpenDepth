"""
图像扰动模块 - 用于抗干扰深度估计训练
参考: DepthAnythingAC

扰动类型:
1. 低光照 (Dark) - 降低亮度 + 泊松高斯噪声
2. 雾 (Fog)
3. 雪 (Snow)
4. 运动模糊 (Motion Blur)
5. 缩放模糊 (Zoom Blur)
6. 对比度变化 (Contrast)
"""

import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2

# 尝试导入 imagecorruptions，如果没有则使用简化版本
try:
    from imagecorruptions import corrupt
    HAS_IMAGECORRUPTIONS = True
except ImportError:
    HAS_IMAGECORRUPTIONS = False
    print("Warning: imagecorruptions not installed. Using simplified perturbations.")


def imadjust(x, a, b, c, d, gamma=1):
    """调整图像亮度范围"""
    y = (((x - a) / (b - a + 1e-5)) ** gamma) * (d - c) + c
    return y


def poisson_gaussian_noise(x, severity):
    """泊松-高斯噪声，模拟真实低光照噪声"""
    c_poisson = 10 * [60, 25, 12, 5, 3][severity]
    x = np.array(x)
    x = np.clip(x, 0, 255) / 255.

    try:
        x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255
        c_gauss = 0.1 * [.08, .12, 0.18, 0.26, 0.38][severity]
        x = np.array(x) / 255.
        x = np.clip(x + np.random.normal(size=x.shape, scale=c_gauss), 0, 1) * 255
    except Exception as e:
        print(f"Noise error: {e}")
        x = x * 255

    return Image.fromarray(np.uint8(x))


def create_dark(image, severity_lower=1, severity_upper=5):
    """低光照扰动"""
    severity = random.randint(severity_lower, severity_upper)
    c = [0.60, 0.50, 0.40, 0.30, 0.20][severity - 1]
    x = np.array(image) / 255.
    x_scaled = imadjust(x, x.min(), x.max(), 0, c, gamma=2) * 255
    x_scaled = poisson_gaussian_noise(x_scaled, severity=severity - 1)
    return x_scaled


def create_fog(image, severity_lower=1, severity_upper=3):
    """雾效果"""
    if not HAS_IMAGECORRUPTIONS:
        # 简化版: 添加白色遮罩
        severity = random.randint(severity_lower, severity_upper)
        alpha = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        img = np.array(image).astype(float)
        fog = np.ones_like(img) * 255
        result = img * (1 - alpha) + fog * alpha
        return Image.fromarray(np.uint8(result))

    severity = random.randint(severity_lower, severity_upper)
    image = np.array(image)
    corrupted = corrupt(image, corruption_name='fog', severity=severity)
    return Image.fromarray(corrupted)


def create_snow(image, severity_lower=1, severity_upper=2):
    """雪效果"""
    if not HAS_IMAGECORRUPTIONS:
        # 简化版: 添加白色噪点
        severity = random.randint(severity_lower, severity_upper)
        img = np.array(image).astype(float)
        noise_density = [0.01, 0.02, 0.03, 0.05, 0.08][severity - 1]
        snow_mask = np.random.random(img.shape[:2]) < noise_density
        img[snow_mask] = 255
        return Image.fromarray(np.uint8(img))

    severity = random.randint(severity_lower, severity_upper)
    image = np.array(image)
    corrupted = corrupt(image, corruption_name='snow', severity=severity)
    return Image.fromarray(corrupted)


def create_motion_blur(image, severity_lower=1, severity_upper=3):
    """运动模糊"""
    if not HAS_IMAGECORRUPTIONS:
        # 简化版: 使用 OpenCV 的 motion blur
        severity = random.randint(severity_lower, severity_upper)
        kernel_size = [5, 10, 15, 20, 25][severity - 1]
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        img = np.array(image)
        result = cv2.filter2D(img, -1, kernel)
        return Image.fromarray(result)

    severity = random.randint(severity_lower, severity_upper)
    image = np.array(image)
    corrupted = corrupt(image, corruption_name='motion_blur', severity=severity)
    return Image.fromarray(corrupted)


def create_zoom_blur(image, severity_lower=1, severity_upper=3):
    """缩放模糊"""
    if not HAS_IMAGECORRUPTIONS:
        # 简化版: 使用高斯模糊
        severity = random.randint(severity_lower, severity_upper)
        kernel_size = [3, 5, 7, 9, 11][severity - 1]
        img = np.array(image)
        result = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        return Image.fromarray(result)

    severity = random.randint(severity_lower, severity_upper)
    image = np.array(image)
    corrupted = corrupt(image, corruption_name='zoom_blur', severity=severity)
    return Image.fromarray(corrupted)


def create_contrast(image, severity_lower=1, severity_upper=3):
    """对比度变化"""
    if not HAS_IMAGECORRUPTIONS:
        severity = random.randint(severity_lower, severity_upper)
        factor = [0.8, 0.6, 0.4, 0.3, 0.2][severity - 1]
        img = np.array(image).astype(float)
        mean = img.mean()
        result = (img - mean) * factor + mean
        result = np.clip(result, 0, 255)
        return Image.fromarray(np.uint8(result))

    severity = random.randint(severity_lower, severity_upper)
    image = np.array(image)
    corrupted = corrupt(image, corruption_name='contrast', severity=severity)
    return Image.fromarray(corrupted)


class ImagePerturbation:
    """
    图像扰动类，用于训练时数据增强

    使用方法:
        perturber = ImagePerturbation(cfg)
        perturbed_image = perturber(image)
    """

    DEFAULT_CFG = {
        # 开关 (与 DepthAnythingAC 一致)
        'is_dark': True,        # 低光照 - 基础扰动，始终应用
        'is_fog': True,
        'is_frost': False,      # AC 中关闭
        'is_snow': True,
        'is_motion_blur': True,
        'is_zoom_blur': True,
        'is_contrast': True,

        # 概率 (与 DepthAnythingAC 一致)
        # 注意: dark 是基础扰动，始终应用，不是概率触发
        'prob_fog': 0.2,
        'prob_frost': 0.2,
        'prob_snow': 0.2,
        'prob_motion_blur': 0.1,
        'prob_zoom_blur': 0.1,
        'prob_contrast': 0.2,

        # 严重程度范围
        'severity_lower_dark': 1,
        'severity_upper_dark': 5,
        'severity_lower_fog': 1,
        'severity_upper_fog': 3,
        'severity_lower_snow': 1,
        'severity_upper_snow': 2,
        'severity_lower_motion_blur': 1,
        'severity_upper_motion_blur': 3,
        'severity_lower_zoom_blur': 1,
        'severity_upper_zoom_blur': 3,
        'severity_lower_contrast': 1,
        'severity_upper_contrast': 3,
    }

    def __init__(self, cfg=None):
        self.cfg = self.DEFAULT_CFG.copy()
        if cfg:
            self.cfg.update(cfg)

    def __call__(self, image):
        """
        对图像应用随机扰动

        Args:
            image: PIL Image 或 numpy array (H, W, 3)

        Returns:
            PIL Image: 扰动后的图像
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        corrupted = image

        # 1. 低光照（基础扰动，始终应用 - 与 AC 一致）
        # AC 中 Dark 是必选的基础扰动，不是概率触发
        if self.cfg['is_dark']:
            corrupted = create_dark(
                corrupted,
                severity_lower=self.cfg['severity_lower_dark'],
                severity_upper=self.cfg['severity_upper_dark']
            )

        # 2. 对比度
        if self.cfg['is_contrast'] and random.random() < self.cfg['prob_contrast']:
            corrupted = create_contrast(
                corrupted,
                severity_lower=self.cfg['severity_lower_contrast'],
                severity_upper=self.cfg['severity_upper_contrast']
            )

        # 3. 天气效果（雾/雪）- 互斥，随机选一个
        weather_effects = []
        if self.cfg['is_fog']:
            weather_effects.append(('fog', self.cfg['prob_fog']))
        if self.cfg['is_snow']:
            weather_effects.append(('snow', self.cfg['prob_snow']))

        if weather_effects:
            effect, prob = random.choice(weather_effects)
            if random.random() < prob:
                if effect == 'fog':
                    corrupted = create_fog(
                        corrupted,
                        severity_lower=self.cfg['severity_lower_fog'],
                        severity_upper=self.cfg['severity_upper_fog']
                    )
                elif effect == 'snow':
                    corrupted = create_snow(
                        corrupted,
                        severity_lower=self.cfg['severity_lower_snow'],
                        severity_upper=self.cfg['severity_upper_snow']
                    )

        # 4. 模糊效果（运动/缩放）- 互斥
        if self.cfg['is_motion_blur'] and self.cfg['is_zoom_blur']:
            if random.random() < 0.5:
                if random.random() < self.cfg['prob_motion_blur']:
                    corrupted = create_motion_blur(
                        corrupted,
                        severity_lower=self.cfg['severity_lower_motion_blur'],
                        severity_upper=self.cfg['severity_upper_motion_blur']
                    )
            else:
                if random.random() < self.cfg['prob_zoom_blur']:
                    corrupted = create_zoom_blur(
                        corrupted,
                        severity_lower=self.cfg['severity_lower_zoom_blur'],
                        severity_upper=self.cfg['severity_upper_zoom_blur']
                    )
        elif self.cfg['is_motion_blur'] and random.random() < self.cfg['prob_motion_blur']:
            corrupted = create_motion_blur(
                corrupted,
                severity_lower=self.cfg['severity_lower_motion_blur'],
                severity_upper=self.cfg['severity_upper_motion_blur']
            )
        elif self.cfg['is_zoom_blur'] and random.random() < self.cfg['prob_zoom_blur']:
            corrupted = create_zoom_blur(
                corrupted,
                severity_lower=self.cfg['severity_lower_zoom_blur'],
                severity_upper=self.cfg['severity_upper_zoom_blur']
            )

        return corrupted


def perturb_tensor(image_tensor, perturber):
    """
    对 tensor 格式的图像应用扰动

    Args:
        image_tensor: torch.Tensor (B, C, H, W) 或 (C, H, W)，范围 [0, 1]
        perturber: ImagePerturbation 实例

    Returns:
        torch.Tensor: 扰动后的图像，格式和输入相同
    """
    squeeze = False
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
        squeeze = True

    batch_size = image_tensor.shape[0]
    _, C, H, W = image_tensor.shape  # 保存原始尺寸
    device = image_tensor.device
    dtype = image_tensor.dtype

    perturbed_list = []
    for i in range(batch_size):
        # 转换为 PIL
        img = image_tensor[i].cpu()
        img = (img * 255).clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(img)

        # 应用扰动
        perturbed_pil = perturber(pil_img)

        # 确保尺寸一致（某些扰动可能改变尺寸）
        if perturbed_pil.size != (W, H):
            perturbed_pil = perturbed_pil.resize((W, H), Image.BILINEAR)

        # 转回 tensor (兼容不同 numpy 版本)
        perturbed_np = np.array(perturbed_pil)
        perturbed_np = perturbed_np.astype('float32') / 255.0
        perturbed_tensor = torch.tensor(perturbed_np, dtype=torch.float32).permute(2, 0, 1)
        perturbed_list.append(perturbed_tensor)

    result = torch.stack(perturbed_list).to(device=device, dtype=dtype)

    if squeeze:
        result = result.squeeze(0)

    return result


if __name__ == "__main__":
    # 测试
    import sys

    # 创建测试图像
    test_img = Image.new('RGB', (256, 256), color=(128, 128, 128))

    # 测试各种扰动
    perturber = ImagePerturbation()

    print("Testing perturbations...")
    for i in range(5):
        perturbed = perturber(test_img)
        print(f"  Test {i+1}: {type(perturbed)}, size={perturbed.size}")

    print("\nTesting tensor perturbation...")
    tensor_img = torch.rand(2, 3, 256, 256)
    perturbed_tensor = perturb_tensor(tensor_img, perturber)
    print(f"  Input: {tensor_img.shape}, Output: {perturbed_tensor.shape}")

    print("\nAll tests passed!")
