"""
Synthetic symbol image generator for NeuralScribe v2.
Renders Greek letters, math operators, and scientific symbols using TTF fonts
with varied stroke widths, sizes, and noise to match handwritten style.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.ndimage import gaussian_filter
import random

from ..utils.logging import get_logger

log = get_logger(__name__)


# Fallback system fonts that typically have good Unicode coverage
FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/opentype/stix/STIXGeneral.otf",
]


class SyntheticSymbolGenerator:
    """
    Generate synthetic training images for symbols not in EMNIST.
    Each symbol is rendered at multiple font sizes and stroke widths,
    with random augmentations to simulate handwriting.
    """

    def __init__(
        self,
        fonts_dir: str = "data/fonts",
        font_sizes: List[int] = None,
        stroke_widths: List[int] = None,
        samples_per_symbol: int = 1500,
        image_size: int = 28,
        canvas_size: int = 64,
        noise_level: float = 0.02,
        blur_sigma_range: Tuple[float, float] = (0.3, 0.8),
    ):
        self.fonts_dir = Path(fonts_dir)
        self.font_sizes = font_sizes or [18, 20, 22, 24]
        self.stroke_widths = stroke_widths or [1, 2, 3]
        self.samples_per_symbol = samples_per_symbol
        self.image_size = image_size
        self.canvas_size = canvas_size
        self.noise_level = noise_level
        self.blur_sigma_range = blur_sigma_range
        self._fonts: List[str] = []
        self._load_fonts()

    def _load_fonts(self):
        """Discover available TTF/OTF fonts."""
        # First check bundled fonts dir
        if self.fonts_dir.exists():
            for f in self.fonts_dir.glob("*.ttf"):
                self._fonts.append(str(f))
            for f in self.fonts_dir.glob("*.otf"):
                self._fonts.append(str(f))

        # Add system fallbacks that exist
        for fp in FALLBACK_FONTS:
            if os.path.exists(fp) and fp not in self._fonts:
                self._fonts.append(fp)

        if not self._fonts:
            log.warning("No fonts found! Synthetic generation will use PIL default font.")
        else:
            log.info(f"Loaded {len(self._fonts)} fonts for synthetic generation")

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get a random font at the given size."""
        if self._fonts:
            font_path = random.choice(self._fonts)
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass
        # Fallback to default
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def _render_symbol(
        self,
        char: str,
        font: ImageFont.FreeTypeFont,
        stroke_width: int = 0,
    ) -> np.ndarray:
        """Render a single character to a numpy array."""
        # Draw on larger canvas first
        canvas = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        draw = ImageDraw.Draw(canvas)

        # Get text bounding box
        try:
            bbox = draw.textbbox((0, 0), char, font=font, stroke_width=stroke_width)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = (self.canvas_size - tw) // 2 - bbox[0]
            y = (self.canvas_size - th) // 2 - bbox[1]
        except Exception:
            x, y = self.canvas_size // 4, self.canvas_size // 4

        draw.text(
            (x, y), char, fill=255, font=font,
            stroke_width=stroke_width, stroke_fill=255
        )

        return np.array(canvas, dtype=np.float32)

    def _add_variation(self, img: np.ndarray) -> np.ndarray:
        """Add random noise, blur, and small geometric variation to simulate handwriting."""
        # Random small rotation
        angle = random.uniform(-8, 8)
        pil = Image.fromarray(img.astype(np.uint8))
        pil = pil.rotate(angle, resample=Image.BILINEAR, fillcolor=0, expand=False)
        img = np.array(pil, dtype=np.float32)

        # Random translation
        tx = random.randint(-2, 2)
        ty = random.randint(-2, 2)
        img = np.roll(np.roll(img, tx, axis=1), ty, axis=0)

        # Random scale
        scale = random.uniform(0.85, 1.15)
        h, w = img.shape
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        pil = Image.fromarray(img.astype(np.uint8))
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        scaled = np.array(pil, dtype=np.float32)

        # Re-center on original canvas
        result = np.zeros_like(img)
        y_off = max(0, (h - new_h) // 2)
        x_off = max(0, (w - new_w) // 2)
        sy = max(0, (new_h - h) // 2)
        sx = max(0, (new_w - w) // 2)
        copy_h = min(new_h - sy, h - y_off)
        copy_w = min(new_w - sx, w - x_off)
        result[y_off:y_off + copy_h, x_off:x_off + copy_w] = scaled[sy:sy + copy_h, sx:sx + copy_w]
        img = result

        # Gaussian blur
        sigma = random.uniform(*self.blur_sigma_range)
        if sigma > 0.1:
            img = gaussian_filter(img, sigma=sigma)

        # Additive noise
        if self.noise_level > 0:
            noise = np.random.randn(*img.shape) * self.noise_level * 255
            img = np.clip(img + noise, 0, 255)

        # Random brightness/contrast
        brightness = random.uniform(0.8, 1.2)
        img = np.clip(img * brightness, 0, 255)

        return img

    def _to_28x28(self, img: np.ndarray) -> np.ndarray:
        """Resize to target image size with bounding box crop."""
        # Crop to content
        if img.max() > 0:
            threshold = img.max() * 0.05
            rows = np.any(img > threshold, axis=1)
            cols = np.any(img > threshold, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                img = img[rmin:rmax + 1, cmin:cmax + 1]

        # Pad to square
        h, w = img.shape
        if h != w:
            size = max(h, w) + 4
            canvas = np.zeros((size, size), dtype=np.float32)
            y_off = (size - h) // 2
            x_off = (size - w) // 2
            canvas[y_off:y_off + h, x_off:x_off + w] = img
            img = canvas

        # Resize
        pil = Image.fromarray(img.astype(np.uint8))
        pil = pil.resize((self.image_size, self.image_size), Image.BICUBIC)
        return np.array(pil, dtype=np.float32)

    def generate_for_symbol(
        self,
        char: str,
        class_id: int,
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic images for a single symbol.
        Returns: (images array (N, 28, 28), labels array (N,))
        """
        n = num_samples or self.samples_per_symbol
        images = []

        for _ in range(n):
            font_size = random.choice(self.font_sizes)
            stroke_width = random.choice(self.stroke_widths)
            font = self._get_font(font_size)

            img = self._render_symbol(char, font, stroke_width)
            img = self._add_variation(img)
            img = self._to_28x28(img)
            images.append(img)

        images_array = np.stack(images, axis=0)
        labels_array = np.full(n, class_id, dtype=np.int64)

        return images_array, labels_array

    def generate_all(
        self,
        symbols: List[Dict],
        progress_callback=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic images for all symbols.
        symbols: list of dicts with {id, display, label, category}
        Returns: (images (N, 28, 28), labels (N,))
        """
        all_images = []
        all_labels = []
        total = len(symbols)

        for i, sym in enumerate(symbols):
            char = sym["display"]
            class_id = sym["id"]
            label = sym["label"]

            log.info(f"Generating synthetic: {label} ({char}) — class {class_id}")
            imgs, lbls = self.generate_for_symbol(char, class_id)
            all_images.append(imgs)
            all_labels.append(lbls)

            if progress_callback:
                progress_callback(i + 1, total, label)

        return np.concatenate(all_images, axis=0), np.concatenate(all_labels, axis=0)