"""
Data augmentation for NeuralScribe v2.
Provides both precomputable (heavy) and on-the-fly (light) augmentations.
"""

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates
from typing import Tuple, Optional
import random


class AugmentationPipeline:
    """
    Configurable augmentation pipeline for 28×28 grayscale images.
    Can be applied on-the-fly during training or used to precompute variants.
    """

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-12, 12),
        translation_range: Tuple[int, int] = (-2, 2),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        shear_range: Tuple[float, float] = (-5, 5),
        elastic_alpha: float = 3.0,
        elastic_sigma: float = 0.5,
        elastic_prob: float = 0.3,
        erosion_dilation_prob: float = 0.15,
        erosion_dilation_kernel_range: Tuple[int, int] = (2, 3),
        noise_level: float = 0.02,
        blur_sigma_range: Tuple[float, float] = (0.0, 0.3),
    ):
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_prob = elastic_prob
        self.erosion_dilation_prob = erosion_dilation_prob
        self.erosion_dilation_kernel_range = erosion_dilation_kernel_range
        self.noise_level = noise_level
        self.blur_sigma_range = blur_sigma_range

    def __call__(self, img_array: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a 2D numpy array (H, W)."""
        img = img_array.copy().astype(np.float32)

        # Random affine
        img = self._affine_transform(img)

        # Elastic deformation
        if random.random() < self.elastic_prob:
            img = self._elastic_deform(img)

        # Erosion/dilation (morphological)
        if random.random() < self.erosion_dilation_prob:
            img = self._morphological(img)

        # Gaussian noise
        if self.noise_level > 0:
            noise = np.random.randn(*img.shape) * self.noise_level * 255
            img = np.clip(img + noise, 0, 255)

        # Light blur
        sigma = random.uniform(*self.blur_sigma_range)
        if sigma > 0.05:
            img = gaussian_filter(img, sigma=sigma)

        return img.astype(np.float32)

    def _affine_transform(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        cx, cy = w / 2, h / 2

        angle = random.uniform(*self.rotation_range)
        tx = random.randint(self.translation_range[0], self.translation_range[1])
        ty = random.randint(self.translation_range[0], self.translation_range[1])
        scale = random.uniform(*self.scale_range)
        shear = random.uniform(*self.shear_range) * np.pi / 180

        cos_a = np.cos(np.radians(angle)) * scale
        sin_a = np.sin(np.radians(angle)) * scale

        pil_img = Image.fromarray(img.astype(np.uint8))

        # Affine matrix coefficients for PIL
        a = cos_a + np.tan(shear) * sin_a
        b = -sin_a + np.tan(shear) * cos_a
        c = tx + cx - a * cx - b * cy
        d = sin_a
        e = cos_a
        f = ty + cy - d * cx - e * cy

        pil_img = pil_img.transform(
            (w, h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR, fillcolor=0
        )
        return np.array(pil_img, dtype=np.float32)

    def _elastic_deform(self, img: np.ndarray) -> np.ndarray:
        """Small elastic deformation for handwriting variance."""
        shape = img.shape
        dx = gaussian_filter(np.random.randn(*shape), self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter(np.random.randn(*shape), self.elastic_sigma) * self.elastic_alpha

        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        indices = [np.clip(y + dy, 0, shape[0] - 1), np.clip(x + dx, 0, shape[1] - 1)]
        return map_coordinates(img, indices, order=1, mode="constant", cval=0).astype(np.float32)

    def _morphological(self, img: np.ndarray) -> np.ndarray:
        """Random erosion or dilation to vary stroke thickness."""
        from scipy.ndimage import binary_erosion, binary_dilation

        k = random.randint(*self.erosion_dilation_kernel_range)
        struct = np.ones((k, k))
        binary = img > (img.max() * 0.3)

        if random.random() < 0.5:
            result = binary_dilation(binary, structure=struct)
        else:
            result = binary_erosion(binary, structure=struct)

        # Blend back
        return (result.astype(np.float32) * img.max()).clip(0, 255)


class TTAAugmentor:
    """
    Test-time augmentation for inference.
    Generates small variants (rotations, translations) and averages predictions.
    """

    def __init__(
        self,
        rotations: Tuple[float, ...] = (-4, 0, 4),
        translations: Tuple[Tuple[int, int], ...] = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)),
    ):
        self.rotations = rotations
        self.translations = translations

    def generate_variants(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate TTA variants from a single (1, 1, 28, 28) tensor.
        Returns: (N, 1, 28, 28) tensor of all variants.
        """
        # Move to CPU for numpy operations
        cpu_tensor = img_tensor.cpu()
        variants = [cpu_tensor.squeeze(0)]  # original

        img_np = (cpu_tensor.squeeze().numpy() * 0.3081 + 0.1307) * 255  # denormalize approx
        img_np = img_np.clip(0, 255).astype(np.uint8)

        for angle in self.rotations:
            if angle == 0:
                continue
            pil = Image.fromarray(img_np)
            rotated = pil.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            arr = np.array(rotated, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).float()
            t = (t - 0.1307) / 0.3081
            variants.append(t.unsqueeze(0))

        for tx, ty in self.translations:
            if tx == 0 and ty == 0:
                continue
            shifted = np.roll(np.roll(img_np.astype(np.float32), tx, axis=1), ty, axis=0)
            shifted = shifted / 255.0
            t = torch.from_numpy(shifted).float()
            t = (t - 0.1307) / 0.3081
            variants.append(t.unsqueeze(0))

        return torch.stack(variants, dim=0)