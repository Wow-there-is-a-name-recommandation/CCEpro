# src/sampler.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


@torch.no_grad()
def sample_images(
    tokenizer,
    text_encoder,
    vae,
    unet,
    scheduler,
    prompts: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    output_dir: str = "samples",
    prefix: str = "sample",
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    batch_size = len(prompts)
    text_inputs = tokenizer(
        list(prompts),
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    uncond_inputs = tokenizer(
        [""] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_input_ids = uncond_inputs.input_ids.to(device)

    cond_embeds = text_encoder(text_input_ids)[0]
    uncond_embeds = text_encoder(uncond_input_ids)[0]
    encoder_hidden_states = torch.cat([uncond_embeds, cond_embeds], dim=0)

    latent_h = height // 8
    latent_w = width // 8
    latents = torch.randn(
        (batch_size, unet.config.in_channels, latent_h, latent_w),
        device=device,
        dtype=dtype,
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        latent_model_input = torch.cat([latents, latents], dim=0)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    image_latents = latents / 0.18215
    images = vae.decode(image_latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()

    paths: list[Path] = []
    for i, img in enumerate(images):
        pil = Image.fromarray((img * 255).astype("uint8"))
        save_path = output_path / f"{prefix}_{i:02d}.png"
        pil.save(save_path)
        paths.append(save_path)

    return paths