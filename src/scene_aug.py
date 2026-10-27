"""Scene-level augmentation for v3 training (needs a background bank, so it lives
at dataset level rather than in the torchvision transform chain).

  * background swap: images on a cut-out white background (c2h) get a random
    symptom-free background (soil, healthy foliage, plain colour) behind the plant,
    so no background is tied to one class.
  * zoom-out: any image is shrunk to 40-75% and placed on a random background,
    so a head or leaf that fills only part of the frame is still recognised.

Labels never change: backgrounds contain no disease (see src/background_bank.py).
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps


class SceneComposer:
    def __init__(self, bg_dir: Path, p_swap: float = 0.7, p_zoom: float = 0.2, zoom: tuple[float, float] = (0.4, 0.75)):
        self.paths = sorted(Path(bg_dir).glob("*.jpg"))
        self.p_swap, self.p_zoom, self.zoom = p_swap, p_zoom, zoom
        self._cache: dict[Path, Image.Image] = {}

    def _bank(self, path: Path) -> Image.Image:
        if path not in self._cache:
            self._cache[path] = Image.open(path).convert("RGB")
        return self._cache[path]

    def background(self, size: tuple[int, int]) -> Image.Image:
        w, h = size
        if not self.paths or random.random() < 0.25:
            c1 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
            c2 = np.clip(c1 + np.random.uniform(-60, 60, 3), 0, 255)
            t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
            grad = (c1 * (1 - t) + c2 * t) * np.ones((1, w, 1), np.float32)
            return Image.fromarray(grad.astype(np.uint8))
        bg = self._bank(random.choice(self.paths))
        s = int(min(bg.size) * random.uniform(0.4, 1.0))
        x, y = random.randint(0, bg.width - s), random.randint(0, bg.height - s)
        bg = bg.crop((x, y, x + s, y + s)).rotate(90 * random.randint(0, 3))
        if random.random() < 0.5:
            bg = ImageOps.mirror(bg)
        a = np.asarray(bg, np.float32) * np.random.uniform(0.75, 1.25, 3)
        return Image.fromarray(a.clip(0, 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)

    @staticmethod
    def cutout_mask(im: Image.Image) -> Image.Image:
        """Foreground = not near-white and not seen through an insect hole (the cleaf
        photos show a pale neutral-grey surface there, see src/insect_prelabel.py), so
        a swapped background also shows through the holes. Feathered: no hard halo."""
        a = np.asarray(im)
        fg = a.min(2) <= 235
        hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(np.int16)
        lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.int16)
        s, v = hsv[..., 1], hsv[..., 2]
        pale_grey = (s <= 50) & (v >= 150) & (np.abs(lab[..., 1] - 128) <= 5) & (lab[..., 2] - 128 <= 6)
        if (fg & ~pale_grey).any():
            pale_grey &= v >= np.median(v[fg & ~pale_grey]) + 45
        fg = (fg & ~pale_grey).astype(np.uint8) * 255
        return Image.fromarray(fg).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.5))

    def __call__(self, im: Image.Image, is_cutout: bool) -> Image.Image:
        mask = self.cutout_mask(im) if is_cutout else None
        if is_cutout and random.random() < self.p_swap:
            bg = self.background(im.size)
            bg.paste(im, (0, 0), mask)
            im, mask = bg, None
        if random.random() < self.p_zoom:
            k = random.uniform(*self.zoom)
            W, H = int(im.width / k), int(im.height / k)
            canvas = self.background((W, H))
            x, y = random.randint(0, W - im.width), random.randint(0, H - im.height)
            canvas.paste(im, (x, y), mask)
            im = canvas
        return im
