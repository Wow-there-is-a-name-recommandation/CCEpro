import os
import gc
import json
import torch
import numpy as np
from PIL import Image

from unified.src.algorithm_a import AlgorithmAPipeline, AlgAConfig

prompts = [
    "a photo of superman flying over the city",
    "a photo of batman flying over the city",
    "a photo of a person walking in a park",
]

os.makedirs("outputs/algA", exist_ok=True)

cfg = AlgAConfig(image_size=384, n_steps=20)
pipe = AlgorithmAPipeline(
    device="cuda:0",
    config=cfg,
)

pipe.pipe.enable_attention_slicing()
pipe.pipe.enable_vae_slicing()
pipe.pipe.enable_vae_tiling()

meta = []

for i, prompt in enumerate(prompts):
    seed = i
    img = pipe.generate(prompt, seed=seed)

    arr = (img.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    path = f"outputs/algA/{i:04d}.png"
    Image.fromarray(arr).save(path)

    meta.append({
        "id": i,
        "prompt": prompt,
        "seed": seed,
        "image_path": path,
    })

    del img, arr
    torch.cuda.empty_cache()
    gc.collect()

with open("outputs/algA/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)