from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def tensor_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """
    images: [B, 3, H, W] in [0, 1]
    """
    images = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    pil_images = []
    for img in images:
        img = (img * 255).round().clip(0, 255).astype("uint8")
        pil_images.append(Image.fromarray(img))
    return pil_images


# ============================================================
# Model loading
# ============================================================
@dataclass
class SDModules:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    scheduler: DDIMScheduler
    device: torch.device
    dtype: torch.dtype


def load_sd14(
    model_id: str = "CompVis/stable-diffusion-v1-4",
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
) -> SDModules:
    if dtype is None:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

    device_obj = torch.device(device)

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    )
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype
    )
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    text_encoder.to(device_obj)
    vae.to(device_obj)
    unet.to(device_obj)

    text_encoder.eval()
    vae.eval()
    unet.eval()

    return SDModules(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        device=device_obj,
        dtype=dtype,
    )


# ============================================================
# Text embeddings
# ============================================================
@torch.no_grad()
def encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompts: List[str],
    device: torch.device,
) -> torch.Tensor:
    inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    embeds = text_encoder(input_ids)[0]
    return embeds


# ============================================================
# Sampling
# ============================================================
@torch.no_grad()
def generate_baseline(
    modules: SDModules,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 40,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    seed: int = 42,
) -> Image.Image:
    """
    Standard classifier-free guidance sampling.
    """
    device = modules.device
    dtype = modules.dtype
    tokenizer = modules.tokenizer
    text_encoder = modules.text_encoder
    vae = modules.vae
    unet = modules.unet
    scheduler = modules.scheduler

    set_seed(seed)

    batch_size = 1
    cond_embeds = encode_prompts(tokenizer, text_encoder, [prompt], device)
    uncond_embeds = encode_prompts(tokenizer, text_encoder, [negative_prompt], device)

    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        device=device,
        dtype=dtype,
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        latent_input = torch.cat([latents, latents], dim=0)
        latent_input = scheduler.scale_model_input(latent_input, t)

        encoder_hidden_states = torch.cat([uncond_embeds, cond_embeds], dim=0)

        noise_pred = unet(
            latent_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
        noise_pred_final = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        latents = scheduler.step(noise_pred_final, t, latents).prev_sample

    latents = latents / 0.18215
    images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)

    return tensor_to_pil(images)[0]


@torch.no_grad()
def generate_with_reverse_guidance(
    modules: SDModules,
    prompt: str,
    erase_term: str,
    related_terms: Optional[List[str]] = None,
    negative_prompt: str = "",
    num_inference_steps: int = 40,
    guidance_scale: float = 7.5,
    erase_scale: float = 2.5,
    related_scale: float = 1.0,
    height: int = 512,
    width: int = 512,
    seed: int = 42,
) -> Image.Image:
    """
    Reverse-guidance prototype:
      eps = eps_uncond
            + guidance_scale * (eps_prompt - eps_uncond)
            - erase_scale * (eps_erase - eps_uncond)
            - related_scale * sum_i (eps_rel_i - eps_uncond)

    Notes:
    - This is an inference-time control prototype.
    - No model parameters are updated.
    - related_terms can be empty or None.
    """
    if related_terms is None:
        related_terms = []

    device = modules.device
    dtype = modules.dtype
    tokenizer = modules.tokenizer
    text_encoder = modules.text_encoder
    vae = modules.vae
    unet = modules.unet
    scheduler = modules.scheduler

    set_seed(seed)

    batch_size = 1

    prompt_embeds = encode_prompts(tokenizer, text_encoder, [prompt], device)
    uncond_embeds = encode_prompts(tokenizer, text_encoder, [negative_prompt], device)
    erase_embeds = encode_prompts(tokenizer, text_encoder, [erase_term], device)

    related_embeds_list = []
    for term in related_terms:
        related_embeds_list.append(encode_prompts(tokenizer, text_encoder, [term], device))

    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        device=device,
        dtype=dtype,
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        branches = [uncond_embeds, prompt_embeds, erase_embeds] + related_embeds_list

        latent_input = torch.cat([latents] * len(branches), dim=0)
        latent_input = scheduler.scale_model_input(latent_input, t)

        encoder_hidden_states = torch.cat(branches, dim=0)

        noise_pred = unet(
            latent_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        preds = list(noise_pred.chunk(len(branches), dim=0))
        eps_uncond = preds[0]
        eps_prompt = preds[1]
        eps_erase = preds[2]
        eps_related = preds[3:] if len(preds) > 3 else []

        # Baseline prompt guidance
        eps_final = eps_uncond + guidance_scale * (eps_prompt - eps_uncond)

        # Reverse guidance for erase term
        eps_final = eps_final - erase_scale * (eps_erase - eps_uncond)

        # Reverse guidance for related terms
        for eps_rel in eps_related:
            eps_final = eps_final - related_scale * (eps_rel - eps_uncond)

        latents = scheduler.step(eps_final, t, latents).prev_sample

    latents = latents / 0.18215
    images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)

    return tensor_to_pil(images)[0]


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--erase_term", type=str, required=True)
    parser.add_argument("--related_terms", type=str, nargs="*", default=[])

    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--erase_scale", type=float, default=2.5)
    parser.add_argument("--related_scale", type=float, default=1.0)

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default="outputs_reverse_guidance")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    modules = load_sd14(
        model_id=args.model_id,
        device=args.device,
    )

    baseline_img = generate_baseline(
        modules=modules,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    reverse_img = generate_with_reverse_guidance(
        modules=modules,
        prompt=args.prompt,
        erase_term=args.erase_term,
        related_terms=args.related_terms,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        erase_scale=args.erase_scale,
        related_scale=args.related_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    baseline_path = os.path.join(args.output_dir, "baseline.png")
    reverse_path = os.path.join(args.output_dir, "reverse_guidance.png")

    baseline_img.save(baseline_path)
    reverse_img.save(reverse_path)

    print(f"[Saved] baseline: {baseline_path}")
    print(f"[Saved] reverse_guidance: {reverse_path}")


if __name__ == "__main__":
    main()