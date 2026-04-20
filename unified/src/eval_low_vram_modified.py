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
    compute_fid,
    make_generator,
)


# --------------------------------------------------------------------------- #
# Evaluation subset config
# --------------------------------------------------------------------------- #

# Keep erased concepts at 60 images each: 20 templates x 3 seeds
ERASED_TEMPLATES = 20
ERASED_SEEDS = 3

# Related concepts: choose only 2 per erased concept, 50 images each: 10 x 5
RELATED_SUBSET: dict[str, list[str]] = {
    "superman": ["batman", "thor"],
    "van gogh": ["picasso", "monet"],
    "snoopy": ["mickey", "pikachu"],
}
RELATED_TEMPLATES = 10
RELATED_SEEDS = 5

# General preservation set
DEFAULT_N_COCO = 50


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


def _load_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    with open(manifest_path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


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
                    "template_index": ti,
                    "seed_index": si,
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
    n_coco: int = DEFAULT_N_COCO,
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
            "prompt_index": i,
        })

        if cfg.flush_every > 0 and (i + 1) % cfg.flush_every == 0:
            flush()

    with open(coco_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# --------------------------------------------------------------------------- #
# Phase 2: scoring only
# --------------------------------------------------------------------------- #

def score_concept_dir(concept_dir: Path, text: str, device: str) -> tuple[float, list[dict]]:
    image_paths = sorted([p for p in concept_dir.glob("*.png") if p.is_file()])
    if not image_paths:
        return 0.0, []

    manifest = _load_manifest(concept_dir / "manifest.json")
    manifest_by_file = {row.get("file"): row for row in manifest if isinstance(row, dict) and row.get("file")}

    model, preprocess, tokenizer = _load_clip(device)
    toks = tokenizer([text]).to(device)
    rows: list[dict] = []

    with torch.no_grad():
        txt_feat = model.encode_text(toks)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        for img_path in tqdm(image_paths, desc=f"score {concept_dir.parent.name}/{concept_dir.name}", leave=False):
            with Image.open(img_path).convert("RGB") as img:
                img_t = preprocess(img).unsqueeze(0).to(device)
            img_feat = model.encode_image(img_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            cs = float((img_feat @ txt_feat.T).item())

            meta = manifest_by_file.get(img_path.name, {})
            rows.append({
                "file": img_path.name,
                "prompt": meta.get("prompt", ""),
                "seed": meta.get("seed"),
                "concept": text,
                "template_index": meta.get("template_index"),
                "seed_index": meta.get("seed_index"),
                "cs": cs,
            })
            del img_t, img_feat
            flush()

    del toks, txt_feat
    flush()

    rows_sorted_high = sorted(rows, key=lambda x: x["cs"], reverse=True)
    rows_sorted_low = sorted(rows, key=lambda x: x["cs"])
    rank_high = {row["file"]: i + 1 for i, row in enumerate(rows_sorted_high)}
    rank_low = {row["file"]: i + 1 for i, row in enumerate(rows_sorted_low)}

    for row in rows:
        row["rank_high"] = rank_high[row["file"]]
        row["rank_low"] = rank_low[row["file"]]

    with open(concept_dir / "cs_ranking.json", "w") as f:
        json.dump(sorted(rows, key=lambda x: x["cs"]), f, indent=2)

    mean_cs = float(np.mean([row["cs"] for row in rows])) if rows else 0.0
    return mean_cs, rows


def score_coco_dir(coco_dir: Path, device: str) -> tuple[float, list[dict]]:
    image_paths = sorted([p for p in coco_dir.glob("*.png") if p.is_file()])
    if not image_paths:
        return 0.0, []

    manifest = _load_manifest(coco_dir / "manifest.json")
    manifest_by_file = {row.get("file"): row for row in manifest if isinstance(row, dict) and row.get("file")}

    model, preprocess, tokenizer = _load_clip(device)
    rows: list[dict] = []
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc=f"score coco {coco_dir.parent.name}", leave=False):
            meta = manifest_by_file.get(img_path.name, {})
            prompt = meta.get("prompt", "")
            if not prompt:
                continue

            with Image.open(img_path).convert("RGB") as img:
                img_t = preprocess(img).unsqueeze(0).to(device)
            toks = tokenizer([prompt]).to(device)

            img_feat = model.encode_image(img_t)
            txt_feat = model.encode_text(toks)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            cs = float((img_feat @ txt_feat.T).item())
            rows.append({
                "file": img_path.name,
                "prompt": prompt,
                "seed": meta.get("seed"),
                "prompt_index": meta.get("prompt_index"),
                "cs": cs,
            })

            del img_t, toks, img_feat, txt_feat
            flush()

    rows_sorted_high = sorted(rows, key=lambda x: x["cs"], reverse=True)
    rows_sorted_low = sorted(rows, key=lambda x: x["cs"])
    rank_high = {row["file"]: i + 1 for i, row in enumerate(rows_sorted_high)}
    rank_low = {row["file"]: i + 1 for i, row in enumerate(rows_sorted_low)}

    for row in rows:
        row["rank_high"] = rank_high[row["file"]]
        row["rank_low"] = rank_low[row["file"]]

    with open(coco_dir / "cs_ranking.json", "w") as f:
        json.dump(sorted(rows, key=lambda x: x["cs"]), f, indent=2)

    return float(np.mean([row["cs"] for row in rows])) if rows else 0.0, rows


def run_score_only(method: str, out_root: Path, device: str) -> dict:
    from unified.src.common import ERASED_ORDER

    erased = list(ERASED_ORDER)
    related_all = [r for rs in RELATED_SUBSET.values() for r in rs]

    per_concept_cs: dict[str, float] = {}
    per_concept_extremes: dict[str, dict] = {}

    for concept in erased + related_all:
        cdir = _concept_dir(out_root, method, concept)
        cs, rows = score_concept_dir(cdir, concept, device=device)
        per_concept_cs[concept] = cs
        print(f"[{method}] {concept:>14s}  CS = {cs:.4f}")

        if rows:
            low = min(rows, key=lambda x: x["cs"])
            high = max(rows, key=lambda x: x["cs"])
            per_concept_extremes[concept] = {
                "min_cs": low["cs"],
                "min_cs_file": low["file"],
                "min_cs_prompt": low["prompt"],
                "max_cs": high["cs"],
                "max_cs_file": high["file"],
                "max_cs_prompt": high["prompt"],
                "n_images": len(rows),
            }

    coco_dir = _coco_dir(out_root, method)
    coco_cs, coco_rows = score_coco_dir(coco_dir, device=device)
    coco_extremes = None
    if coco_rows:
        low = min(coco_rows, key=lambda x: x["cs"])
        high = max(coco_rows, key=lambda x: x["cs"])
        coco_extremes = {
            "min_cs": low["cs"],
            "min_cs_file": low["file"],
            "min_cs_prompt": low["prompt"],
            "max_cs": high["cs"],
            "max_cs_file": high["file"],
            "max_cs_prompt": high["prompt"],
            "n_images": len(coco_rows),
        }

    result = {
        "method": method,
        "related_subset": RELATED_SUBSET,
        "per_concept_CS": per_concept_cs,
        "per_concept_extremes": per_concept_extremes,
        "AVG_e_CS": float(np.mean([per_concept_cs[c] for c in erased])) if erased else 0.0,
        "AVG_r_CS": float(np.mean([per_concept_cs[c] for c in related_all])) if related_all else 0.0,
        "coco_CS": coco_cs,
        "coco_extremes": coco_extremes,
    }
    for erased_c, rel_list in RELATED_SUBSET.items():
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
    from unified.src.common import ERASED_ORDER

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["generate", "score", "fid", "all"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["sd_v1.4", "algorithm_a", "algorithm_b", "cosa_gba", "unified"])
    parser.add_argument("--out_dir", type=str, default="unified/eval_output")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip_coco", action="store_true")
    parser.add_argument("--flush_every", type=int, default=1)
    parser.add_argument("--n_coco", type=int, default=DEFAULT_N_COCO)
    parser.add_argument("--edited_pipe", type=str, default="unified/output/algB")
    parser.add_argument("--spm_dir", type=str, default="unified/output/unified_spm")
    args = parser.parse_args()

    cfg_erased = GenConfig(
        n_templates=ERASED_TEMPLATES,
        n_seeds=ERASED_SEEDS,
        device=args.device,
        flush_every=args.flush_every,
    )
    cfg_related = GenConfig(
        n_templates=RELATED_TEMPLATES,
        n_seeds=RELATED_SEEDS,
        device=args.device,
        flush_every=args.flush_every,
    )
    cfg_coco = GenConfig(
        n_templates=0,
        n_seeds=0,
        device=args.device,
        flush_every=args.flush_every,
    )

    out_root = Path(args.out_dir)

    erased = list(ERASED_ORDER)
    related_all = [r for rs in RELATED_SUBSET.values() for r in rs]

    if args.mode in ("generate", "all"):
        gen_fn = make_generator(
            args.method,
            args.device,
            extra_args={"edited_pipe": args.edited_pipe, "spm_dir": args.spm_dir},
        )
        generate_concept_images(args.method, gen_fn, erased, out_root, cfg_erased)
        generate_concept_images(args.method, gen_fn, related_all, out_root, cfg_related)
        if not args.skip_coco:
            generate_coco_images(args.method, gen_fn, out_root, cfg_coco, n_coco=args.n_coco)
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
