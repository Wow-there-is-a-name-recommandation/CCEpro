from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_tensor_image(img: torch.Tensor, out_path: Path):
    arr = (img.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)


def main():
    from diffusers import DDIMScheduler
    from unified.src.algorithm_a import AlgorithmAPipeline, AlgAConfig
    from unified.src.common import load_sd14

    out_root = Path("unified/eval_output_tiny")
    out_root.mkdir(parents=True, exist_ok=True)

    device = "cuda:0"
    n_samples = 15

    # --------------------------------------------------
    # 1) Algorithm A
    # --------------------------------------------------
    alg_cfg = AlgAConfig(image_size=384, n_steps=20, cfg=7.5)
    alg_pipe = AlgorithmAPipeline(device=device, dtype=torch.float32, config=alg_cfg)

    prompt_alg = "a photo of superman flying over the city"

    for i in range(n_samples):
        img = alg_pipe.generate(prompt_alg, seed=i)
        save_tensor_image(img, out_root / "algorithm_a" / f"{i:02d}.png")

        del img
        flush_memory()

    del alg_pipe
    flush_memory()

    # --------------------------------------------------
    # 2) SD v1.4
    # --------------------------------------------------
    sd_pipe = load_sd14(device=device, dtype=torch.float32)
    sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)

    prompt_sd = "a photo of superman flying over the city"

    for i in range(n_samples):
        g = torch.Generator(device=device).manual_seed(i)

        out = sd_pipe(
            prompt_sd,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=g,
            output_type="pt",
            height=384,
            width=384,
        )

        save_tensor_image(out.images, out_root / "sd_v1.4" / f"{i:02d}.png")

        del out
        flush_memory()

    del sd_pipe
    flush_memory()

    # --------------------------------------------------
    # 3) COCO (SD 기준)
    # --------------------------------------------------
    sd_pipe2 = load_sd14(device=device, dtype=torch.float32)
    sd_pipe2.scheduler = DDIMScheduler.from_config(sd_pipe2.scheduler.config)

    prompt_coco = "a man riding a bicycle down a city street"

    for i in range(n_samples):
        g2 = torch.Generator(device=device).manual_seed(i)

        out2 = sd_pipe2(
            prompt_coco,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=g2,
            output_type="pt",
            height=384,
            width=384,
        )

        save_tensor_image(out2.images, out_root / "coco" / f"{i:02d}.png")

        del out2
        flush_memory()

    del sd_pipe2
    flush_memory()

    # --------------------------------------------------
    # summary
    # --------------------------------------------------
    summary = {
        "n_samples_per_model": n_samples,
        "algorithm_a_prompt": prompt_alg,
        "sd_prompt": prompt_sd,
        "coco_prompt": prompt_coco,
        "saved_to": str(out_root),
    }

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"saved to {out_root}")


if __name__ == "__main__":
    main()