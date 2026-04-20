from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

OUT_ROOT = Path("unified/eval_output_tiny")
DEVICE = "cuda:0"

PROMPTS = {
    "algorithm_a": "a photo of superman flying over the city",
    "sd_v1.4": "a photo of superman flying over the city",
    "coco": "a man riding a bicycle down a city street",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def list_pngs(folder: Path) -> list[Path]:
    return sorted([p for p in folder.glob("*.png") if p.is_file()])


# ------------------------------------------------------------
# CLIP score
# ------------------------------------------------------------

_CLIP_CACHE = {"model": None, "preprocess": None, "tokenizer": None, "device": None}


def load_clip(device: str):
    if _CLIP_CACHE["model"] is not None and _CLIP_CACHE["device"] == device:
        return _CLIP_CACHE["model"], _CLIP_CACHE["preprocess"], _CLIP_CACHE["tokenizer"]

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        device=device,
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()

    _CLIP_CACHE["model"] = model
    _CLIP_CACHE["preprocess"] = preprocess
    _CLIP_CACHE["tokenizer"] = tokenizer
    _CLIP_CACHE["device"] = device
    return model, preprocess, tokenizer


@torch.no_grad()
def clip_score_from_dir(image_dir: Path, text: str, device: str, batch_size: int = 8) -> float:
    files = list_pngs(image_dir)
    if not files:
        return float("nan")

    model, preprocess, tokenizer = load_clip(device)

    toks = tokenizer([text]).to(device)
    txt_feats = model.encode_text(toks)
    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

    sims: list[float] = []

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]

        imgs = []
        for fp in batch_files:
            with Image.open(fp) as img:
                imgs.append(preprocess(img.convert("RGB")))

        imgs_t = torch.stack(imgs).to(device)
        img_feats = model.encode_image(imgs_t)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        batch_sims = (img_feats @ txt_feats.T).squeeze(-1).detach().cpu().tolist()
        sims.extend([float(x) for x in batch_sims])

        del imgs, imgs_t, img_feats

    return float(np.mean(sims))


# ------------------------------------------------------------
# FID
# ------------------------------------------------------------

def compute_fid(dir_a: Path, dir_b: Path, device: str) -> float:
    from cleanfid import fid

    return float(
        fid.compute_fid(
            str(dir_a),
            str(dir_b),
            device=device,
            mode="clean",
            batch_size=16,
            num_workers=0,
        )
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    results = {}

    # CS for each folder
    for name, prompt in PROMPTS.items():
        folder = OUT_ROOT / name
        if not folder.exists():
            print(f"[WARN] missing folder: {folder}")
            results[name] = {
                "n_images": 0,
                "prompt": prompt,
                "CS": None,
                "FID_vs_sd_v1.4": None,
            }
            continue

        files = list_pngs(folder)
        cs = clip_score_from_dir(folder, prompt, DEVICE)
        results[name] = {
            "n_images": len(files),
            "prompt": prompt,
            "CS": cs,
            "FID_vs_sd_v1.4": None,
        }
        print(f"[CS] {name:>10s} | n={len(files):2d} | CS={cs:.4f}")

    # FID: algorithm_a vs sd_v1.4
    sd_dir = OUT_ROOT / "sd_v1.4"
    alg_dir = OUT_ROOT / "algorithm_a"

    if sd_dir.exists() and alg_dir.exists():
        fid_alg = compute_fid(sd_dir, alg_dir, DEVICE)
        results["algorithm_a"]["FID_vs_sd_v1.4"] = fid_alg
        results["sd_v1.4"]["FID_vs_sd_v1.4"] = 0.0
        print(f"[FID] algorithm_a vs sd_v1.4 = {fid_alg:.4f}")
    else:
        print("[WARN] FID skipped because sd_v1.4 or algorithm_a folder is missing")

    # coco is not a separate model; keep FID as None
    if "coco" in results:
        results["coco"]["note"] = (
            "COCO folder is a general-prompt sample set, not a separate model. "
            "FID_vs_sd_v1.4 is left as null in this tiny setup."
        )

    # Aggregate summary
    summary = {
        "out_root": str(OUT_ROOT),
        "device": DEVICE,
        "results": results,
    }

    summary_path = OUT_ROOT / "scores_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved summary to: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()