# src/losses.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def retain_loss(student_pred: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    """
    Standard diffusion noise prediction loss.
    """
    return F.mse_loss(student_pred, target_noise)


def erase_loss_placeholder(
    student_pred: torch.Tensor,
    target_noise: torch.Tensor,
) -> torch.Tensor:
    """
    Placeholder erase loss.
    For now, we keep a standard denoising form so the code path stays trainable.

    Replace later with:
    - surrogate-guided erasure
    - reverse-guidance-inspired training target
    - continual replay loss
    - associative concept suppression loss
    """
    return F.mse_loss(student_pred, target_noise)


def split_mode_masks(modes: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    erase_mask = torch.tensor([m == "erase" for m in modes], device=device, dtype=torch.bool)
    retain_mask = torch.tensor([m == "retain" for m in modes], device=device, dtype=torch.bool)
    return erase_mask, retain_mask