# src/trainer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import retain_loss, erase_loss_placeholder, split_mode_masks
from .sampler import sample_images
from .utils import ensure_dir


@dataclass
class TrainConfig:
    output_dir: str = "outputs/base_train"
    learning_rate: float = 1e-5
    batch_size: int = 2
    num_epochs: int = 1
    save_every_steps: int = 500
    sample_every_steps: int = 200
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 4
    sample_prompts: tuple[str, ...] = (
        "a painting in the style of Van Gogh",
        "a photo of a beagle in a park",
        "a city skyline at night",
    )


class BaseTrainer:
    def __init__(
        self,
        components,
        train_dataloader: DataLoader,
        cfg: TrainConfig,
    ) -> None:
        self.tokenizer = components.tokenizer
        self.text_encoder = components.text_encoder
        self.vae = components.vae
        self.unet = components.unet
        self.noise_scheduler = components.noise_scheduler
        self.device = components.device
        self.dtype = components.dtype

        self.train_dataloader = train_dataloader
        self.cfg = cfg

        ensure_dir(cfg.output_dir)
        ensure_dir(Path(cfg.output_dir) / "checkpoints")
        ensure_dir(Path(cfg.output_dir) / "samples")

        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda" and cfg.mixed_precision))
        self.global_step = 0

    def _encode_prompt(self, prompts: list[str]) -> torch.Tensor:
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        with torch.no_grad():
            hidden_states = self.text_encoder(input_ids)[0]
        return hidden_states

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.dtype)
        prompts = batch["prompts"]
        modes = batch["modes"]

        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=self.device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = self._encode_prompt(prompts)

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        erase_mask, retain_mask = split_mode_masks(modes, self.device)

        total_loss = torch.tensor(0.0, device=self.device, dtype=noise_pred.dtype)

        if retain_mask.any():
            total_loss = total_loss + retain_loss(
                noise_pred[retain_mask],
                noise[retain_mask],
            )

        if erase_mask.any():
            total_loss = total_loss + erase_loss_placeholder(
                noise_pred[erase_mask],
                noise[erase_mask],
            )

        return total_loss

    def save_checkpoint(self, step: int) -> None:
        ckpt_dir = Path(self.cfg.output_dir) / "checkpoints" / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.unet.save_pretrained(ckpt_dir / "unet")

    def sample(self, step: int) -> None:
        sample_dir = Path(self.cfg.output_dir) / "samples" / f"step_{step}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        sample_images(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            scheduler=self.noise_scheduler,
            prompts=self.cfg.sample_prompts,
            device=self.device,
            dtype=self.dtype,
            output_dir=str(sample_dir),
            prefix="sample",
        )

    def train(self) -> None:
        self.unet.train()

        for epoch in range(self.cfg.num_epochs):
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.cfg.num_epochs}")

            for batch in pbar:
                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda" and self.cfg.mixed_precision)):
                    loss = self._compute_loss(batch)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.global_step += 1
                pbar.set_postfix(loss=float(loss.detach().cpu().item()), step=self.global_step)

                if self.global_step % self.cfg.sample_every_steps == 0:
                    self.sample(self.global_step)

                if self.global_step % self.cfg.save_every_steps == 0:
                    self.save_checkpoint(self.global_step)

        self.save_checkpoint(self.global_step)
        self.sample(self.global_step)