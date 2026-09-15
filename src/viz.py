"""Contact sheets for experiment error analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_thumb(abs_path: str | Path, size: int = 160) -> Image.Image:
    try:
        with Image.open(abs_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((size, size))
            canvas = Image.new("RGB", (size, size), (20, 20, 20))
            canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
            return canvas
    except Exception:
        canvas = Image.new("RGB", (size, size), (80, 0, 0))
        ImageDraw.Draw(canvas).text((8, size // 2), "UNREADABLE", fill=(255, 255, 255))
        return canvas


def save_image_grid(
    paths: list[str],
    labels: list[str],
    out_path: Path,
    title: str,
    cols: int = 5,
    thumb: int = 160,
) -> None:
    if not paths:
        return
    n = len(paths)
    rows = int(np.ceil(n / cols))
    pad = 8
    label_h = 40
    cell_w, cell_h = thumb, thumb + label_h
    grid = Image.new(
        "RGB",
        (cols * (cell_w + pad) + pad, rows * (cell_h + pad) + pad + 40),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    draw.text((pad, 10), title, fill=(0, 0, 0), font=font)
    for i, (p, lab) in enumerate(zip(paths, labels)):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = 40 + pad + r * (cell_h + pad)
        grid.paste(load_thumb(p, thumb), (x, y))
        text = lab if len(lab) <= 32 else lab[:29] + "..."
        draw.text((x, y + thumb + 4), text, fill=(20, 20, 20), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
