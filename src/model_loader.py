# src/model_loader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class SDComponents:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler
    device: torch.device
    dtype: torch.dtype


def load_sd14_components(
    model_id: str = "CompVis/stable-diffusion-v1-4",
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
) -> SDComponents:
    """
    Load Stable Diffusion v1.4 components for custom training loops.

    Notes:
    - We load the tokenizer, text encoder, VAE, UNet, and noise scheduler separately
      to keep the training/sampling loop fully customizable.
    - SD1.x commonly uses CLIP text encoder + latent scaling factor 0.18215.
    """
    if dtype is None:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

    torch_device = torch.device(device)

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=dtype,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    text_encoder.to(torch_device)
    vae.to(torch_device)
    unet.to(torch_device)

    # Freeze everything by default for baseline training skeleton.
    # Later you can selectively unfreeze UNet blocks or add LoRA/adapters.
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(True)

    return SDComponents(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        noise_scheduler=noise_scheduler,
        device=torch_device,
        dtype=dtype,
    )