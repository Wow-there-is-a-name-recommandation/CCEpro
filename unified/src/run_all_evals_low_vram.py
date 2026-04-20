from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path

import torch

METHODS_IN_ORDER = ["sd_v1.4", "algorithm_a", "algorithm_b", "unified"]


def run_stage(method: str, mode: str, eval_dir: Path, device: str,
              n_templates: int, n_seeds: int, n_coco: int, flush_every: int) -> None:
    cmd = [
        "python", "-m", "unified.src.eval_low_vram",
        "--mode", mode,
        "--method", method,
        "--out_dir", str(eval_dir),
        "--device", device,
        "--n_templates", str(n_templates),
        "--n_seeds", str(n_seeds),
        "--n_coco", str(n_coco),
        "--flush_every", str(flush_every),
    ]
    print("\n$$$ running:", " ".join(cmd))
    subprocess.check_call(cmd)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="unified/eval_output")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_templates", type=int, default=20)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_coco", type=int, default=100)
    parser.add_argument("--flush_every", type=int, default=1)
    parser.add_argument("--skip_methods", type=str, default="")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in args.skip_methods.split(",") if s.strip()}

    # 1) Generate only, one method at a time.
    for method in METHODS_IN_ORDER:
        if method in skip:
            continue
        run_stage(method, "generate", eval_dir, args.device,
                  args.n_templates, args.n_seeds, args.n_coco, args.flush_every)

    # 2) Score only, one method at a time.
    for method in METHODS_IN_ORDER:
        if method in skip:
            continue
        run_stage(method, "score", eval_dir, args.device,
                  args.n_templates, args.n_seeds, args.n_coco, args.flush_every)

    # 3) Add FID, after SD reference is ready.
    for method in METHODS_IN_ORDER:
        if method in skip:
            continue
        run_stage(method, "fid", eval_dir, args.device,
                  args.n_templates, args.n_seeds, args.n_coco, args.flush_every)

    # 4) Optionally compile the final table if that script exists.
    compile_path = Path("unified/src/compile_ablation.py")
    if compile_path.exists():
        subprocess.check_call([
            "python", "-m", "unified.src.compile_ablation",
            "--eval_dir", str(eval_dir),
            "--out_dir", "unified/to_human",
        ])


if __name__ == "__main__":
    main()
