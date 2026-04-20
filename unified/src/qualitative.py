"""
Generate a qualitative side-by-side comparison figure.

Columns: SD v1.4 | Algorithm A | Algorithm B | Unified (+ optional CoSA+GBA)
Rows   : 3 erased concepts + 3 related concepts + 1 COCO

Uses the same seed for each prompt across methods so visual differences are
attributable to the method.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from unified.src.common import ERASED_ORDER, RELATED, pick_device


PROMPTS: list[tuple[str, str]] = [
    # (row_label, prompt)
    ("Superman (erased)", "a photo of superman flying over the city"),
    ("Van Gogh (erased)", "a painting in the style of van gogh"),
    ("Snoopy (erased)",   "a cartoon of snoopy on a doghouse"),
    ("Batman (related)",  "a close-up portrait of batman"),
    ("Picasso (related)", "a painting by picasso"),
    ("Mickey (related)",  "a cartoon of mickey mouse waving"),
    ("MS-COCO general",   "a yellow taxi driving past a skyscraper"),
]

# Methods to include in the figure; each entry: (column_label, method_key, kwargs)
METHODS_IN_FIG: list[tuple[str, str, dict]] = [
    ("SD v1.4",  "sd_v1.4",     {}),
    ("REPEL",    "algorithm_a", {}),
    ("UCE-CL",   "algorithm_b", {"path": "unified/output/algB"}),
    ("CoSA+GBA", "cosa_gba",    {}),
    ("Unified",  "unified",     {"edited_pipe": "unified/output/algB",
                                  "spm_dir": "unified/output/unified_spm"}),
]


def _tile_pil(images: list[Image.Image], n_cols: int, pad: int = 4,
              bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    if not images: return Image.new("RGB", (1, 1), bg)
    w, h = images[0].size
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    tile = Image.new("RGB", (n_cols * (w + pad) + pad, n_rows * (h + pad) + pad), bg)
    for idx, img in enumerate(images):
        r, c = divmod(idx, n_cols)
        tile.paste(img, (c * (w + pad) + pad, r * (h + pad) + pad))
    return tile


def _label_image(img: Image.Image, label: str, font_size: int = 18) -> Image.Image:
    """Add a top label bar to an image."""
    w, h = img.size
    bar_h = font_size + 12
    out = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 6), label, fill=(0, 0, 0), font=font)
    return out


def _label_row(img: Image.Image, label: str, font_size: int = 18) -> Image.Image:
    w, h = img.size
    bar_w = 220
    out = Image.new("RGB", (w + bar_w, h), (255, 255, 255))
    out.paste(img, (bar_w, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, h // 2 - font_size // 2), label, fill=(0, 0, 0), font=font)
    return out


def _to_pil(img_tensor: torch.Tensor) -> Image.Image:
    arr = (img_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="unified/to_human/qualitative.png")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    device = str(pick_device(args.device))

    # For each method, load it once, generate all prompts, then free
    per_method_imgs: dict[str, list[Image.Image]] = {}
    from unified.src.eval import make_generator

    for col_label, method_key, kwargs in METHODS_IN_FIG:
        print(f"\n===  Generating column: {col_label} ({method_key})")
        gen = make_generator(method_key, device, kwargs)
        imgs: list[Image.Image] = []
        for row_label, prompt in PROMPTS:
            t = gen(prompt, args.seed)
            pil = _to_pil(t).resize((args.image_size, args.image_size))
            imgs.append(pil)
        per_method_imgs[col_label] = imgs

        # Free VRAM for the next method
        del gen
        gc.collect()
        torch.cuda.empty_cache()

    # Build the grid: rows=PROMPTS, cols=METHODS
    # Each cell: labeled image with the column header on top row only
    tiles_rows: list[Image.Image] = []
    for i, (row_label, prompt) in enumerate(PROMPTS):
        row_imgs: list[Image.Image] = []
        for col_label, _, _ in METHODS_IN_FIG:
            img = per_method_imgs[col_label][i]
            if i == 0:
                img = _label_image(img, col_label)
            row_imgs.append(img)
        row_tile = _tile_pil(row_imgs, n_cols=len(METHODS_IN_FIG), pad=2)
        row_tile = _label_row(row_tile, row_label, font_size=18)
        tiles_rows.append(row_tile)

    # Stack rows vertically
    max_w = max(t.width for t in tiles_rows)
    total_h = sum(t.height for t in tiles_rows) + (len(tiles_rows) - 1) * 4
    final = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for t in tiles_rows:
        final.paste(t, (0, y))
        y += t.height + 4

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    final.save(args.out)
    print(f"[qualitative] saved → {args.out}  size={final.size}")


if __name__ == "__main__":
    main()
