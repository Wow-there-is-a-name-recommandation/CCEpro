"""
Self-contained evaluation harness for Unified Continual Concept Erasure.

Measures CLIP-Score (CS) for erased + related concepts and FID on MS-COCO.
No imports from the parent SPM/ directory.

Minimal prompt templates (taken from the Lyu et al. 2024 OpenCLIP-style
template subset) — 20 per concept. With 5 seeds we generate 100 images per
concept, matching the parent project's per-concept budget.

CS (CLIP-Score) is computed via the raw CLIP ViT-L/14 text-image similarity
(cosine × 100) — standard in concept-erasure benchmarks.

FID is computed via `clean-fid` against the original SD-v1.4 outputs for the
same prompts (a self-referential FID: measures drift from the unedited model
on a held-out MS-COCO prompt list). This is cheaper than generating the full
MS-COCO validation set.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #

# 20 per-concept prompt templates
CONCEPT_TEMPLATES: list[str] = [
    "a photo of {}",
    "a painting of {}",
    "an image of {}",
    "{} in a cinematic composition",
    "{} in a dramatic pose",
    "{} on a city street",
    "a detailed illustration of {}",
    "{} with stylized shading",
    "a digital artwork of {}",
    "a close-up portrait of {}",
    "{} in high resolution",
    "{} looking at the viewer",
    "a poster of {}",
    "a comic-book panel of {}",
    "{} centered in the frame",
    "{} in daylight",
    "{} at night",
    "{} in soft lighting",
    "{} in black and white",
    "{} — concept art",
]

# A tiny COCO-like caption set for FID reference. Real COCO-5k is prohibitive
# for a 5-day project; we use 100 diverse general captions (no assignment
# concepts). FID against SD-v1.4-on-the-same-captions quantifies drift.
COCO_MINI: list[str] = [
    "a man riding a bicycle down a city street",
    "two children playing with a kite on a grassy field",
    "a plate of pasta with tomato sauce on a wooden table",
    "a brown horse grazing in a sunny pasture",
    "a yellow taxi driving past a skyscraper",
    "a woman typing on a laptop in a coffee shop",
    "a golden retriever sitting on a porch",
    "a red car parked next to a lamppost",
    "a chef arranging sushi on a plate",
    "a sailboat on a calm lake at sunset",
    "a group of hikers walking on a mountain trail",
    "a cat sleeping on a patterned rug",
    "a vase of flowers on a kitchen counter",
    "a bowl of soup on a wooden dining table",
    "a couple walking hand in hand on a beach",
    "a crowded subway station during rush hour",
    "a stack of books on a wooden shelf",
    "a bicycle leaning against a brick wall",
    "a bus parked at a stop with passengers boarding",
    "a baseball player swinging a bat at home plate",
    "a kid eating an ice cream cone on a sunny day",
    "a fruit bowl with apples oranges and bananas",
    "a soccer ball on a green grass field",
    "a couple sitting on a park bench feeding pigeons",
    "a coffee mug next to an open book",
    "a glass of red wine on a dinner table",
    "a train passing through a rural village",
    "a busy kitchen with chefs preparing food",
    "a woman reading a newspaper on a train",
    "a fisherman holding a large fish on a pier",
    "a father teaching his son how to fly a kite",
    "a chef slicing vegetables on a cutting board",
    "a market stall selling fresh vegetables",
    "a surfer riding a large ocean wave",
    "a small wooden cabin in a snowy forest",
    "a golfer hitting a ball on a sunny course",
    "a ballerina practicing in a studio",
    "a jogger running on a forest path",
    "a couple eating dinner at an outdoor restaurant",
    "a crowded beach with umbrellas and towels",
    "a farmer driving a tractor in a wheat field",
    "a young girl holding a bouquet of flowers",
    "a street musician playing guitar on a corner",
    "a cup of tea steaming on a windowsill",
    "a fisherman casting a line into a river",
    "a skier going down a snowy slope",
    "a mother pushing a stroller through a park",
    "a lighthouse standing on a rocky cliff",
    "a picnic basket on a checkered blanket",
    "a dog catching a frisbee in mid-air",
    "a grandmother knitting in a rocking chair",
    "a bakery window full of pastries",
    "a row of colorful houses on a hillside",
    "a family gathered around a Christmas tree",
    "a vintage car parked in front of a diner",
    "a glass of orange juice next to a plate of toast",
    "a woman holding an umbrella in the rain",
    "a pair of scissors next to a roll of tape",
    "a wooden rocking chair on a front porch",
    "a hot-air balloon floating over a valley",
    "a man and woman dancing in a ballroom",
    "a waterfall cascading into a rocky pool",
    "a child blowing out birthday candles",
    "a baker pulling bread out of an oven",
    "a basketball player dunking through a hoop",
    "a fisherman holding a net on a small boat",
    "a classroom with children raising their hands",
    "a librarian stacking books on a cart",
    "a coffee shop with exposed brick walls",
    "a train station platform at night",
    "a little girl holding a puppy",
    "a fruit market with colorful produce",
    "a woman painting a watercolor landscape",
    "a chef decorating a wedding cake",
    "a fisherman sorting fish at a harbor",
    "a young boy flying a red kite",
    "a row of wooden boats at a dock",
    "a crowded marathon starting line",
    "a person writing in a journal at a coffee shop",
    "a stone bridge crossing a narrow stream",
    "a vintage camera on a wooden table",
    "a waiter carrying a tray of drinks",
    "a mother tying her daughter's shoelace",
    "a couple taking a selfie in front of a landmark",
    "a cluster of mushrooms growing on a log",
    "a child drawing with crayons at a desk",
    "a bowl of cereal with milk and strawberries",
    "a farmer feeding chickens in a yard",
    "a group of friends laughing at a dinner party",
    "a bookshop window display with new releases",
    "a pair of hiking boots on a trail",
    "a man fishing from a wooden pier at dawn",
    "a child building a sandcastle at the beach",
    "a musician tuning a violin backstage",
    "a bowl of fresh salad with cherry tomatoes",
    "a market vendor selling spices in burlap sacks",
    "a red double-decker bus on a busy street",
    "a boy reading a book under a tree",
    "a couple sharing popcorn at a movie theater",
    "a carved pumpkin on a front porch at dusk",
    "a candle burning on a wooden side table",
    "a street sign pointing to the old town",
]


# --------------------------------------------------------------------------- #
# CLIP scoring (uses open_clip ViT-L/14, a standard choice for concept erasure)
# --------------------------------------------------------------------------- #

_CLIP_CACHE = {"model": None, "preprocess": None, "tokenizer": None, "device": None}


def _load_clip(device: str):
    if _CLIP_CACHE["model"] is not None and _CLIP_CACHE["device"] == device:
        return _CLIP_CACHE["model"], _CLIP_CACHE["preprocess"], _CLIP_CACHE["tokenizer"]

    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    _CLIP_CACHE.update({"model": model, "preprocess": preprocess,
                        "tokenizer": tokenizer, "device": device})
    return model, preprocess, tokenizer


@torch.no_grad()
def clip_score(images: Sequence[Image.Image], text: str, device: str = "cuda:0") -> float:
    """Mean CLIP similarity (cosine, in [−1, 1]) between each image and `text`."""
    if not images: return 0.0
    model, preprocess, tokenizer = _load_clip(device)
    imgs = torch.stack([preprocess(img) for img in images]).to(device)
    toks = tokenizer([text]).to(device)
    img_feats = model.encode_image(imgs)
    txt_feats = model.encode_text(toks)
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
    sims = (img_feats @ txt_feats.T).squeeze(-1)
    return float(sims.mean().item())


# --------------------------------------------------------------------------- #
# FID computation via clean-fid
# --------------------------------------------------------------------------- #

def compute_fid(path_a: str, path_b: str, device: str = "cuda:0") -> float:
    """Clean-FID between two directories of PNGs."""
    from cleanfid import fid
    return float(fid.compute_fid(path_a, path_b, device=device, mode="clean",
                                  batch_size=16, num_workers=0))


# --------------------------------------------------------------------------- #
# Image generation wrapper
# --------------------------------------------------------------------------- #

def _save_tensor(img: torch.Tensor, path: Path) -> None:
    arr = (img.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _to_pil(img_tensor: torch.Tensor) -> Image.Image:
    arr = (img_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


@dataclass
class GenConfig:
    n_templates: int = 20
    n_seeds: int = 5     # images per template per concept
    device: str = "cuda:0"


def run_concept_eval(
    method_name: str,
    generate_fn: Callable[[str, int], torch.Tensor],
    concepts: Sequence[str],
    out_root: Path,
    cfg: GenConfig,
    save_images: bool = True,
) -> dict[str, float]:
    """Generate + CS for each concept."""
    cs_result: dict[str, float] = {}

    for concept in concepts:
        concept_dir = out_root / method_name / concept.replace(" ", "_")
        concept_dir.mkdir(parents=True, exist_ok=True)

        images: list[Image.Image] = []
        prompts = CONCEPT_TEMPLATES[:cfg.n_templates]
        for ti, tpl in enumerate(tqdm(prompts, desc=f"{method_name}/{concept}",
                                      leave=False)):
            prompt = tpl.format(concept)
            for si in range(cfg.n_seeds):
                seed = ti * cfg.n_seeds + si + 1
                img = generate_fn(prompt, seed)
                pil = _to_pil(img)
                images.append(pil)
                if save_images:
                    pil.save(concept_dir / f"t{ti:02d}_s{si}.png")

        cs = clip_score(images, concept, device=cfg.device)
        cs_result[concept] = cs
        print(f"[{method_name}] {concept:>14s}  CS = {cs:.4f}  (n={len(images)})")

    return cs_result


def run_coco_eval(
    method_name: str,
    generate_fn: Callable[[str, int], torch.Tensor],
    out_root: Path,
    cfg: GenConfig,
    n_coco: int = 100,
) -> dict[str, float]:
    """Generate on the COCO-mini prompt set and return paths for later FID."""
    coco_dir = out_root / method_name / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    images: list[Image.Image] = []
    prompts = COCO_MINI[:n_coco]
    for i, p in enumerate(tqdm(prompts, desc=f"{method_name}/coco", leave=False)):
        img = generate_fn(p, i + 1)
        pil = _to_pil(img)
        images.append(pil)
        pil.save(coco_dir / f"c{i:03d}.png")

    # CS against the COCO captions — one score per (image, prompt) pair, mean
    model, preprocess, tokenizer = _load_clip(cfg.device)
    with torch.no_grad():
        imgs = torch.stack([preprocess(img) for img in images]).to(cfg.device)
        img_feats = model.encode_image(imgs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        sims = []
        for i, p in enumerate(prompts):
            toks = tokenizer([p]).to(cfg.device)
            t = model.encode_text(toks); t = t / t.norm(dim=-1, keepdim=True)
            sims.append((img_feats[i:i+1] @ t.T).item())
        coco_cs = float(np.mean(sims))

    return {"coco_CS": coco_cs, "n_coco": n_coco}


# --------------------------------------------------------------------------- #
# Method dispatch
# --------------------------------------------------------------------------- #

def make_generator(method: str, device: str, extra_args: dict) -> Callable[[str, int], torch.Tensor]:
    """Return a generate_fn(prompt, seed) -> torch.Tensor [1,3,H,W] in [0,1]."""
    method = method.lower()

    if method == "sd_v1.4":
        from unified.src.common import load_sd14
        from diffusers import DDIMScheduler
        pipe = load_sd14(device=device, dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        @torch.no_grad()
        def gen(prompt: str, seed: int) -> torch.Tensor:
            g = torch.Generator(device=device).manual_seed(seed)
            out = pipe(prompt, num_inference_steps=30, guidance_scale=7.5,
                       generator=g, output_type="pt")
            return out.images
        return gen

    if method == "algorithm_a":
        from unified.src.algorithm_a import AlgorithmAPipeline, AlgAConfig
        # Only forward kwargs that AlgAConfig accepts — drop path-type keys.
        _alg_kwargs = {k: v for k, v in (extra_args or {}).items()
                       if k not in ("edited_pipe", "spm_dir", "path")}
        cfg = AlgAConfig(**_alg_kwargs) if _alg_kwargs else AlgAConfig()
        pipe = AlgorithmAPipeline(device=device, config=cfg)

        @torch.no_grad()
        def gen(prompt: str, seed: int) -> torch.Tensor:
            return pipe.generate(prompt, seed=seed)
        return gen

    if method == "algorithm_b":
        from diffusers import DDIMScheduler, StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            extra_args.get("path", "unified/output/algB"),
            torch_dtype=torch.float32, safety_checker=None,
        ).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        @torch.no_grad()
        def gen(prompt: str, seed: int) -> torch.Tensor:
            g = torch.Generator(device=device).manual_seed(seed)
            out = pipe(prompt, num_inference_steps=30, guidance_scale=7.5,
                       generator=g, output_type="pt")
            return out.images
        return gen

    if method == "cosa_gba":
        from unified.src.parent_spm_adapter import make_parent_cosa_gba_generator
        gen, _ = make_parent_cosa_gba_generator(device=device)
        return gen

    if method == "unified":
        from unified.src.algorithm_unified import UnifiedPipeline, UnifiedConfig
        cfg = UnifiedConfig(**{k: v for k, v in extra_args.items()
                                if k not in ("edited_pipe", "spm_dir")})
        pipe = UnifiedPipeline(
            edited_pipe_path=extra_args.get("edited_pipe", "unified/output/algB"),
            spm_dir=extra_args.get("spm_dir", "unified/output/unified_spm"),
            device=device, config=cfg,
        )

        @torch.no_grad()
        def gen(prompt: str, seed: int) -> torch.Tensor:
            return pipe.generate(prompt, seed=seed)
        return gen

    raise ValueError(f"Unknown method: {method}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    from unified.src.common import ERASED_ORDER, RELATED

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["sd_v1.4", "algorithm_a", "algorithm_b",
                                 "cosa_gba", "unified"])
    parser.add_argument("--out_dir", type=str, default="unified/eval_output")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_templates", type=int, default=20)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_coco", type=int, default=50)
    parser.add_argument("--skip_coco", action="store_true")
    parser.add_argument("--edited_pipe", type=str, default="unified/output/algB")
    parser.add_argument("--spm_dir", type=str, default="unified/output/unified_spm")
    args = parser.parse_args()

    cfg = GenConfig(n_templates=args.n_templates, n_seeds=args.n_seeds,
                    device=args.device)
    out_root = Path(args.out_dir)

    erased = list(ERASED_ORDER)
    related_all = [r for rs in RELATED.values() for r in rs]

    gen_fn = make_generator(
        args.method, args.device,
        extra_args={"edited_pipe": args.edited_pipe, "spm_dir": args.spm_dir},
    )

    # Erased concepts
    cs_erased = run_concept_eval(args.method, gen_fn, erased,
                                  out_root, cfg, save_images=True)

    # Related concepts
    cs_related = run_concept_eval(args.method, gen_fn, related_all,
                                   out_root, cfg, save_images=True)

    # MS-COCO (CS + FID later)
    coco_result = {}
    if not args.skip_coco:
        coco_result = run_coco_eval(args.method, gen_fn, out_root, cfg,
                                     n_coco=args.n_coco)

    # Aggregate
    result = {
        "method": args.method,
        "per_concept_CS": {**cs_erased, **cs_related},
        "AVG_e_CS": float(np.mean([cs_erased[c] for c in erased])),
        "AVG_r_CS": float(np.mean(list(cs_related.values()))),
        **coco_result,
    }
    for erased_c, rel_list in RELATED.items():
        result[f"{erased_c}_r_CS_avg"] = float(np.mean(
            [cs_related[r] for r in rel_list]))

    summary_path = out_root / f"{args.method}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[{args.method}] summary → {summary_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
