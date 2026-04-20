"""
One-command orchestrator to run all four evaluations (SD v1.4, Algorithm A,
Algorithm B, Unified) and compute FID against the SD v1.4 reference.

FID methodology: for each method we generate images for the same COCO-mini
prompt set, then compute Clean-FID(method_coco, sd_v1.4_coco). This measures
how much the edited model has drifted from the original on general prompts.
A low FID means the concept-erasure surgery did not damage overall generative
quality.
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path

import torch


METHODS_IN_ORDER = ["sd_v1.4", "algorithm_a", "algorithm_b", "unified"]


def run_method_eval(method: str, eval_dir: Path, device: str,
                    n_templates: int, n_seeds: int, n_coco: int) -> None:
    cmd = [
        "python", "-m", "unified.src.eval",
        "--method", method, "--out_dir", str(eval_dir),
        "--device", device,
        "--n_templates", str(n_templates),
        "--n_seeds", str(n_seeds),
        "--n_coco", str(n_coco),
    ]
    print(f"\n$$$ running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def add_fid_column(eval_dir: Path, device: str) -> None:
    """Compute Clean-FID of each method's COCO directory vs SD v1.4's."""
    from cleanfid import fid

    ref_dir = eval_dir / "sd_v1.4" / "coco"
    if not ref_dir.exists() or not any(ref_dir.iterdir()):
        print(f"[FID] SD v1.4 reference directory missing ({ref_dir}), skipping")
        return

    for method in METHODS_IN_ORDER:
        method_dir = eval_dir / method / "coco"
        if not method_dir.exists() or not any(method_dir.iterdir()):
            print(f"[FID] {method}: no coco dir, skip")
            continue

        summary_path = eval_dir / f"{method}_summary.json"
        if not summary_path.exists():
            print(f"[FID] {method}: no summary json, skip")
            continue

        # SD v1.4's FID against itself is by definition 0 — skip.
        if method == "sd_v1.4":
            with open(summary_path) as f:
                s = json.load(f)
            s["coco_FID"] = 0.0
            with open(summary_path, "w") as f:
                json.dump(s, f, indent=2)
            continue

        print(f"[FID] computing {method} vs sd_v1.4 ...")
        score = fid.compute_fid(
            str(ref_dir), str(method_dir),
            device=device, mode="clean", batch_size=16, num_workers=0,
        )
        print(f"[FID] {method}: FID = {score:.3f}")
        with open(summary_path) as f:
            s = json.load(f)
        s["coco_FID"] = float(score)
        with open(summary_path, "w") as f:
            json.dump(s, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="unified/eval_output")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_templates", type=int, default=20)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_coco", type=int, default=100)
    parser.add_argument("--skip_methods", type=str, default="")
    args = parser.parse_args()

    skip = {s.strip() for s in args.skip_methods.split(",") if s.strip()}
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Run each method independently in a subprocess so we fully free VRAM
    # between runs (very important given SD v1.4 + two editing states + SPMs).
    for method in METHODS_IN_ORDER:
        if method in skip:
            print(f"skipping {method}")
            continue
        run_method_eval(method, eval_dir, args.device,
                        args.n_templates, args.n_seeds, args.n_coco)
        gc.collect()
        torch.cuda.empty_cache()

    # Compute FID scores and update each summary json
    add_fid_column(eval_dir, args.device)

    # Finally re-compile the ablation table
    subprocess.check_call([
        "python", "-m", "unified.src.compile_ablation",
        "--eval_dir", str(eval_dir), "--out_dir", "unified/to_human",
    ])


if __name__ == "__main__":
    main()
