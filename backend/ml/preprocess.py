"""
Preprocessing pipeline for NeuralScribe v2.
Applies bounding-box crop, center-of-mass alignment, normalization, etc.
This module is the single source of truth for preprocessing — used by both
dataset preparation and inference to ensure consistency.
"""

import numpy as np
import torch
from PIL import Image, ImageFilter
from typing import Optional, Tuple
from scipy import ndimage


# Default normalization stats (EMNIST-like)
DEFAULT_MEAN = 0.1307
DEFAULT_STD = 0.3081


def bbox_crop_pad(img_array: np.ndarray, padding: int = 2) -> np.ndarray:
    """
    Crop to bounding box of non-zero pixels, then add uniform padding.
    Input: 2D numpy array (H, W), values 0–255 or 0–1.
    """
    if img_array.max() == 0:
        return img_array

    threshold = img_array.max() * 0.05
    rows = np.any(img_array > threshold, axis=1)
    cols = np.any(img_array > threshold, axis=0)

    if not rows.any() or not cols.any():
        return img_array

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = img_array[rmin:rmax + 1, cmin:cmax + 1]

    if padding > 0:
        cropped = np.pad(cropped, padding, mode="constant", constant_values=0)

    return cropped


def center_of_mass_align(img_array: np.ndarray, target_size: int = 28) -> np.ndarray:
    """
    Translate image so the center of mass is at the center of the target canvas.
    Input: 2D numpy array (any size).
    Output: 2D numpy array (target_size, target_size).
    """
    if img_array.max() == 0:
        return np.zeros((target_size, target_size), dtype=img_array.dtype)

    # Resize to target maintaining aspect ratio
    h, w = img_array.shape
    scale = min((target_size - 4) / h, (target_size - 4) / w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

    pil_img = Image.fromarray(img_array.astype(np.uint8))
    pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    resized = np.array(pil_img, dtype=np.float32)

    # Place on canvas
    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # Center of mass correction
    if canvas.sum() > 0:
        cy, cx = ndimage.center_of_mass(canvas)
        shift_y = target_size / 2 - cy
        shift_x = target_size / 2 - cx
        canvas = ndimage.shift(canvas, [shift_y, shift_x], mode="constant", cval=0)

    return canvas


def apply_emnist_orientation(img_array: np.ndarray) -> np.ndarray:
    """
    EMNIST images need transpose + horizontal flip to match visual orientation.
    Apply this correction so drawn inputs match training data.
    """
    return np.fliplr(img_array.T)


def smooth_image(img_array: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Light Gaussian blur to smooth synthetic/canvas strokes."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img_array, sigma=sigma)


def normalize(tensor: torch.Tensor, mean: float = DEFAULT_MEAN, std: float = DEFAULT_STD) -> torch.Tensor:
    """Normalize a float tensor with mean and std."""
    return (tensor - mean) / std


def preprocess_image(
    pil_image: Image.Image,
    target_size: int = 28,
    do_center_of_mass: bool = True,
    do_smoothing: bool = True,
    smoothing_sigma: float = 0.5,
    mean: float = DEFAULT_MEAN,
    std: float = DEFAULT_STD,
    is_emnist: bool = False,
) -> torch.Tensor:
    """
    Full preprocessing pipeline for a single PIL image.
    Returns: tensor of shape (1, 1, 28, 28), normalized and ready for model.
    """
    # Convert to grayscale numpy
    img = pil_image.convert("L")
    img_array = np.array(img, dtype=np.float32)

    # EMNIST orientation correction
    if is_emnist:
        img_array = apply_emnist_orientation(img_array)

    # Bounding box crop + padding
    img_array = bbox_crop_pad(img_array)

    # Center of mass alignment and resize
    if do_center_of_mass:
        img_array = center_of_mass_align(img_array, target_size)
    else:
        pil_resized = Image.fromarray(img_array.astype(np.uint8))
        pil_resized = pil_resized.resize((target_size, target_size), Image.BICUBIC)
        img_array = np.array(pil_resized, dtype=np.float32)

    # Smoothing
    if do_smoothing:
        img_array = smooth_image(img_array, smoothing_sigma)

    # To tensor [0, 1]
    tensor = torch.from_numpy(img_array).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Normalize
    tensor = normalize(tensor, mean, std)

    return tensor


def preprocess_tensor(
    img_tensor: torch.Tensor,
    target_size: int = 28,
    do_center_of_mass: bool = True,
    do_smoothing: bool = True,
    smoothing_sigma: float = 0.5,
    mean: float = DEFAULT_MEAN,
    std: float = DEFAULT_STD,
    is_emnist: bool = False,
) -> torch.Tensor:
    """
    Preprocess a raw tensor (e.g., from EMNIST loader).
    Input: tensor of shape (H, W) or (1, H, W), values 0–1 float.
    Returns: (1, 1, 28, 28) normalized tensor.
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.squeeze(0)

    img_array = (img_tensor.numpy() * 255).astype(np.float32)

    if is_emnist:
        img_array = apply_emnist_orientation(img_array)

    img_array = bbox_crop_pad(img_array)

    if do_center_of_mass:
        img_array = center_of_mass_align(img_array, target_size)
    else:
        pil_resized = Image.fromarray(img_array.astype(np.uint8))
        pil_resized = pil_resized.resize((target_size, target_size), Image.BICUBIC)
        img_array = np.array(pil_resized, dtype=np.float32)

    if do_smoothing:
        img_array = smooth_image(img_array, smoothing_sigma)

    tensor = torch.from_numpy(img_array).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = normalize(tensor, mean, std)

    return tensor


def preprocess_canvas_data(
    pixel_data: list,
    target_size: int = 28,
    mean: float = DEFAULT_MEAN,
    std: float = DEFAULT_STD,
) -> torch.Tensor:
    """
    Preprocess raw canvas pixel data from frontend.
    Input: flat list of 784 floats (28×28) or NxN grid.
    Returns: (1, 1, 28, 28) normalized tensor.
    """
    size = int(np.sqrt(len(pixel_data)))
    img_array = np.array(pixel_data, dtype=np.float32).reshape(size, size)

    # Scale to 0–255 for processing pipeline
    if img_array.max() <= 1.0:
        img_array = img_array * 255.0

    img_array = bbox_crop_pad(img_array)
    img_array = center_of_mass_align(img_array, target_size)
    img_array = smooth_image(img_array, sigma=0.5)

    tensor = torch.from_numpy(img_array).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = normalize(tensor, mean, std)

    return tensor