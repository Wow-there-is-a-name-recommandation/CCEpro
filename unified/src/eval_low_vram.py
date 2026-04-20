from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from unified.src.eval import (
    CONCEPT_TEMPLATES,
    COCO_MINI,
    _load_clip,
    clip_score,
    compute_fid,
    make_generator,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def flush() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


@dataclass
class GenConfig:
    n_templates: int = 20
    n_seeds: int = 3
    device: str = "cuda:0"
    flush_every: int = 1


def _concept_dir(out_root: Path, method: str, concept: str) -> Path:
    return out_root / method / concept.replace(" ", "_")


def _coco_dir(out_root: Path, method: str) -> Path:
    return out_root / method / "coco"


def _to_pil(img_tensor: torch.Tensor) -> Image.Image:
    arr = (img_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# --------------------------------------------------------------------------- #
# Phase 1: generation only
# --------------------------------------------------------------------------- #

def generate_concept_images(
    method_name: str,
    generate_fn: Callable[[str, int], torch.Tensor],
    concepts: Sequence[str],
    out_root: Path,
    cfg: GenConfig,
) -> None:
    """Generate images only. Save PNGs + prompt manifest. No scoring here."""
    for concept in concepts:
        concept_dir = _concept_dir(out_root, method_name, concept)
        concept_dir.mkdir(parents=True, exist_ok=True)

        manifest: list[dict] = []
        prompts = CONCEPT_TEMPLATES[:cfg.n_templates]
        img_count = 0

        for ti, tpl in enumerate(tqdm(prompts, desc=f"gen {method_name}/{concept}", leave=False)):
            prompt = tpl.format(concept)
            for si in range(cfg.n_seeds):
                seed = ti * cfg.n_seeds + si + 1
                file_name = f"t{ti:02d}_s{si}.png"
                file_path = concept_dir / file_name

                if not file_path.exists():
                    img = generate_fn(prompt, seed)
                    pil = _to_pil(img)
                    pil.save(file_path)
                    del img, pil
                    flush()

                manifest.append({
                    "file": file_name,
                    "prompt": prompt,
                    "seed": seed,
                    "concept": concept,
                })
                img_count += 1

                if cfg.flush_every > 0 and img_count % cfg.flush_every == 0:
                    flush()

        with open(concept_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)


def generate_coco_images(
    method_name: str,
    generate_fn: Callable[[str, int], torch.Tensor],
    out_root: Path,
    cfg: GenConfig,
    n_coco: int = 100,
) -> None:
    """Generate general-prompt images only. Save PNGs + prompt manifest."""
    coco_dir = _coco_dir(out_root, method_name)
    coco_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    prompts = COCO_MINI[:n_coco]

    for i, prompt in enumerate(tqdm(prompts, desc=f"gen {method_name}/coco", leave=False)):
        seed = i + 1
        file_name = f"c{i:03d}.png"
        file_path = coco_dir / file_name

        if not file_path.exists():
            img = generate_fn(prompt, seed)
            pil = _to_pil(img)
            pil.save(file_path)
            del img, pil
            flush()

        manifest.append({
            "file": file_name,
            "prompt": prompt,
            "seed": seed,
        })

        if cfg.flush_every > 0 and (i + 1) % cfg.flush_every == 0:
            flush()

    with open(coco_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# --------------------------------------------------------------------------- #
# Phase 2: scoring only
# --------------------------------------------------------------------------- #

def score_concept_dir(concept_dir: Path, text: str, device: str) -> float:
    image_paths = sorted([p for p in concept_dir.glob("*.png") if p.is_file()])
    if not image_paths:
        return 0.0

    model, preprocess, tokenizer = _load_clip(device)
    toks = tokenizer([text]).to(device)
    sims = []

    with torch.no_grad():
        txt_feat = model.encode_text(toks)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        for img_path in tqdm(image_paths, desc=f"score {concept_dir.parent.name}/{concept_dir.name}", leave=False):
            with Image.open(img_path).convert("RGB") as img:
                img_t = preprocess(img).unsqueeze(0).to(device)
            img_feat = model.encode_image(img_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims.append((img_feat @ txt_feat.T).item())
            del img_t, img_feat
            flush()

    del toks, txt_feat
    flush()
    return float(np.mean(sims))


def score_coco_dir(coco_dir: Path, device: str) -> float:
    image_paths = sorted([p for p in coco_dir.glob("*.png") if p.is_file()])
    prompts = COCO_MINI[:len(image_paths)]

    model, preprocess, tokenizer = _load_clip(device)
    sims = []
    with torch.no_grad():
        for img_path, prompt in tqdm(list(zip(image_paths, prompts)), desc=f"score coco {coco_dir.parent.name}", leave=False):
            with Image.open(img_path).convert("RGB") as img:
                img_t = preprocess(img).unsqueeze(0).to(device)
            toks = tokenizer([prompt]).to(device)

            img_feat = model.encode_image(img_t)
            txt_feat = model.encode_text(toks)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sims.append((img_feat @ txt_feat.T).item())

            del img_t, toks, img_feat, txt_feat
            flush()

    return float(np.mean(sims)) if sims else 0.0


def run_score_only(method: str, out_root: Path, device: str) -> dict:
    from unified.src.common import ERASED_ORDER, RELATED

    erased = list(ERASED_ORDER)
    related_all = [r for rs in RELATED.values() for r in rs]

    per_concept_cs: dict[str, float] = {}

    for concept in erased + related_all:
        cdir = _concept_dir(out_root, method, concept)
        cs = score_concept_dir(cdir, concept, device=device)
        per_concept_cs[concept] = cs
        print(f"[{method}] {concept:>14s}  CS = {cs:.4f}")

    coco_dir = _coco_dir(out_root, method)
    coco_cs = score_coco_dir(coco_dir, device=device)

    result = {
        "method": method,
        "per_concept_CS": per_concept_cs,
        "AVG_e_CS": float(np.mean([per_concept_cs[c] for c in erased])),
        "AVG_r_CS": float(np.mean([per_concept_cs[c] for c in related_all])),
        "coco_CS": coco_cs,
    }
    for erased_c, rel_list in RELATED.items():
        result[f"{erased_c}_r_CS_avg"] = float(np.mean([per_concept_cs[r] for r in rel_list]))

    summary_path = out_root / f"{method}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{method}] summary -> {summary_path}")
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    from unified.src.common import ERASED_ORDER, RELATED

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["generate", "score", "fid", "all"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["sd_v1.4", "algorithm_a", "algorithm_b", "cosa_gba", "unified"])
    parser.add_argument("--out_dir", type=str, default="unified/eval_output")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_templates", type=int, default=20)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_coco", type=int, default=50)
    parser.add_argument("--skip_coco", action="store_true")
    parser.add_argument("--flush_every", type=int, default=1)
    parser.add_argument("--edited_pipe", type=str, default="unified/output/algB")
    parser.add_argument("--spm_dir", type=str, default="unified/output/unified_spm")
    args = parser.parse_args()

    cfg = GenConfig(
        n_templates=args.n_templates,
        n_seeds=args.n_seeds,
        device=args.device,
        flush_every=args.flush_every,
    )
    out_root = Path(args.out_dir)

    erased = list(ERASED_ORDER)
    related_all = [r for rs in RELATED.values() for r in rs]

    if args.mode in ("generate", "all"):
        gen_fn = make_generator(
            args.method,
            args.device,
            extra_args={"edited_pipe": args.edited_pipe, "spm_dir": args.spm_dir},
        )
        generate_concept_images(args.method, gen_fn, erased, out_root, cfg)
        generate_concept_images(args.method, gen_fn, related_all, out_root, cfg)
        if not args.skip_coco:
            generate_coco_images(args.method, gen_fn, out_root, cfg, n_coco=args.n_coco)
        del gen_fn
        flush()

    if args.mode in ("score", "all"):
        run_score_only(args.method, out_root, args.device)

    if args.mode in ("fid", "all"):
        summary_path = out_root / f"{args.method}_summary.json"
        ref_dir = _coco_dir(out_root, "sd_v1.4")
        cur_dir = _coco_dir(out_root, args.method)
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary json not found: {summary_path}")
        with open(summary_path) as f:
            result = json.load(f)
        if args.method == "sd_v1.4":
            result["coco_FID"] = 0.0
        else:
            result["coco_FID"] = compute_fid(str(ref_dir), str(cur_dir), device=args.device)
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
