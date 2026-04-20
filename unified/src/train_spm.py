"""
SPM trainer with Gradient-Balanced Anchoring (GBA) for Phase 2 of the
Unified Algorithm.

The trainer attaches a fresh rank-1 SPMNetwork to the *edited* UNet (θ_edited
from Phase 1 / Algorithm B) and optimises two losses:

    L_erase  = ‖ε_{θ_edited + ϕ}(x_t, c)  −  ε_{θ_edited}(x_t, ∅)‖²
    L_anchor = ‖ε_{θ_edited + ϕ}(x_t, a)  −  ε_{θ_edited}(x_t, a)‖²

GBA dynamically reweights L_anchor by
    λ_spm = γ · EMA(‖∇L_erase‖) / EMA(‖∇L_anchor‖)
every T_p steps so we don't need to sweep λ manually.

Anchors are accumulated across concepts — when training SPM_k we anchor to
R(c_k) ∪ A_cumulative so previously-erased concepts' related words are
preserved too.

This file depends only on `unified/src/spm.py` and `unified/src/common.py`.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm

from unified.src.common import (
    COCO_ANCHORS, ERASED_ORDER, RELATED, concept_key,
    encode_prompt_sequence, flush, load_sd14, pick_device,
)
from unified.src.spm import SPMNetwork


# --------------------------------------------------------------------------- #
# GBA controller
# --------------------------------------------------------------------------- #

class GBAController:
    """Gradient-Balanced Anchoring — auto-tunes λ from EMA of gradient norms."""

    def __init__(self, gamma: float = 1.0, probe_interval: int = 50,
                 beta_ema: float = 0.9, lam_init: float = 100.0,
                 lam_min: float = 1.0, lam_max: float = 1e4):
        self.gamma = gamma
        self.probe_interval = probe_interval
        self.beta = beta_ema
        self.lam = lam_init
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.e_ema = 0.0
        self.a_ema = 0.0
        self._initialised = False

    def maybe_probe(self, step: int, spm: SPMNetwork,
                    loss_e: torch.Tensor, loss_a: torch.Tensor) -> None:
        """Compute grad norms for each loss separately and update λ."""
        if step % self.probe_interval != 0:
            return

        params = list(spm.parameters())
        # Separate grads: retain graph so we can backward twice
        g_e = torch.autograd.grad(loss_e, params, retain_graph=True, create_graph=False,
                                  allow_unused=True)
        g_a = torch.autograd.grad(loss_a, params, retain_graph=True, create_graph=False,
                                  allow_unused=True)
        g_e_norm = sum((g.detach().norm().item() ** 2 for g in g_e if g is not None)) ** 0.5
        g_a_norm = sum((g.detach().norm().item() ** 2 for g in g_a if g is not None)) ** 0.5

        if not self._initialised:
            self.e_ema = g_e_norm
            self.a_ema = g_a_norm
            self._initialised = True
        else:
            self.e_ema = self.beta * self.e_ema + (1 - self.beta) * g_e_norm
            self.a_ema = self.beta * self.a_ema + (1 - self.beta) * g_a_norm

        if self.a_ema > 1e-8:
            self.lam = max(self.lam_min,
                           min(self.lam_max, self.gamma * self.e_ema / self.a_ema))


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _embed(pipe, prompt: str, device: str) -> torch.Tensor:
    return encode_prompt_sequence(pipe, prompt, device)


def train_single_spm(
    pipe: StableDiffusionPipeline,
    target_concept: str,
    anchor_concepts: Sequence[str],
    iterations: int = 500,
    lr: float = 1e-4,
    rank: int = 1,
    alpha: float = 1.0,
    gba_gamma: float = 1.0,
    anchor_prob: float = 0.5,    # prob. of using a related anchor vs a COCO anchor
    device: str = "cuda:0",
    seed: int = 42,
    log_every: int = 50,
) -> tuple[SPMNetwork, dict]:
    """Train one SPM for one target concept. Returns (spm, training curves)."""
    torch.manual_seed(seed)
    random.seed(seed)

    unet = pipe.unet
    scheduler: DDIMScheduler = pipe.scheduler

    # Freeze base UNet, text encoder, VAE
    for m in (unet, pipe.text_encoder, pipe.vae):
        for p in m.parameters():
            p.requires_grad_(False)

    # Attach fresh rank-1 SPM
    spm = SPMNetwork(unet, rank=rank, alpha=alpha)
    optim = torch.optim.AdamW(spm.parameters(), lr=lr)

    gba = GBAController(gamma=gba_gamma)

    # Precompute text embeddings
    emb_concept = _embed(pipe, target_concept, device)
    emb_null = _embed(pipe, "", device)
    emb_anchors = {a: _embed(pipe, a, device) for a in anchor_concepts}
    emb_cocos = {c: _embed(pipe, c, device) for c in COCO_ANCHORS}
    anchor_keys = list(emb_anchors.keys())
    coco_keys = list(emb_cocos.keys())

    # Latent shape
    h = w = 64
    c_in = unet.config.in_channels

    curves = {"erase_loss": [], "anchor_loss": [], "lam_spm": [], "step": []}

    pbar = tqdm(range(iterations), desc=f"SPM[{target_concept}]")
    for step in pbar:
        # Sample anchor
        if random.random() < anchor_prob and anchor_keys:
            a_key = random.choice(anchor_keys)
            emb_anchor = emb_anchors[a_key]
        else:
            a_key = random.choice(coco_keys)
            emb_anchor = emb_cocos[a_key]

        # Sample x_t and timestep
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,),
                          device=device, dtype=torch.long)
        noise = torch.randn((1, c_in, h, w), device=device,
                            dtype=unet.dtype)
        # We treat `noise` directly as x_t — equivalent to training at t ∈ [0, T]
        # with the latent distribution replaced by the prior. This is the
        # standard SPM "fast" training mode.
        x_t = noise

        # ── Forward passes ──
        # 1) ε_θ(x_t, ∅)  (no SPM, null prompt) — erase target
        spm.set_multiplier(0.0)
        with torch.no_grad():
            eps_target_null = unet(x_t, t, encoder_hidden_states=emb_null).sample
            eps_target_anchor = unet(x_t, t, encoder_hidden_states=emb_anchor).sample

        # 2) ε_{θ+ϕ}(x_t, c) — SPM active, concept prompt
        spm.set_multiplier(1.0)
        eps_concept_spm = unet(x_t, t, encoder_hidden_states=emb_concept).sample

        # 3) ε_{θ+ϕ}(x_t, a) — SPM active, anchor prompt
        eps_anchor_spm = unet(x_t, t, encoder_hidden_states=emb_anchor).sample

        # ── Losses ──
        loss_e = F.mse_loss(eps_concept_spm, eps_target_null.detach())
        loss_a = F.mse_loss(eps_anchor_spm, eps_target_anchor.detach())

        # GBA probe (uses autograd.grad — no .backward() yet)
        if step > 0 and step % gba.probe_interval == 0:
            gba.maybe_probe(step, spm, loss_e, loss_a)

        loss = loss_e + gba.lam * loss_a

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Log
        if step % log_every == 0:
            curves["step"].append(step)
            curves["erase_loss"].append(float(loss_e.detach().cpu()))
            curves["anchor_loss"].append(float(loss_a.detach().cpu()))
            curves["lam_spm"].append(float(gba.lam))
            pbar.set_postfix(
                Le=f"{float(loss_e):.4f}",
                La=f"{float(loss_a):.4f}",
                lam=f"{gba.lam:.1f}",
            )

    return spm, curves


# --------------------------------------------------------------------------- #
# Sequential training entry point (Phase 2 of Unified)
# --------------------------------------------------------------------------- #

def train_unified_phase2(
    pipe_edited: StableDiffusionPipeline,
    output_dir: str,
    concepts: Sequence[str] = ERASED_ORDER,
    related_map: dict = RELATED,
    iterations: int = 500,
    lr: float = 1e-4,
    gba_gamma: float = 1.0,
    device: str = "cuda:0",
) -> dict:
    """Train one SPM per concept sequentially with cumulative anchors."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    a_cumulative: list[str] = []
    all_curves: dict[str, dict] = {}

    for concept in concepts:
        save_path = out / f"spm_{concept_key(concept)}.pt"
        # Resume: skip concepts whose checkpoint already exists. Cumulative
        # anchors must still be extended so later concepts see the right set.
        if save_path.exists():
            print(f"\n=== [Phase 2] SPM for '{concept}' already exists at "
                  f"{save_path}, skipping ===")
            a_cumulative += list(related_map[concept])
            continue

        anchors = list(related_map[concept]) + list(a_cumulative)
        anchors = list(dict.fromkeys(anchors))

        print(f"\n=== [Phase 2] Training SPM for '{concept}' "
              f"with {len(anchors)} anchors ===")
        spm, curves = train_single_spm(
            pipe_edited, target_concept=concept, anchor_concepts=anchors,
            iterations=iterations, lr=lr, gba_gamma=gba_gamma, device=device,
        )
        spm.save(save_path)
        all_curves[concept] = curves

        # CRITICAL: unload the SPM (restore base Linear.forward) before
        # training the next one, so the next SPMNetwork wraps the clean base.
        spm.unload()
        flush()

        a_cumulative += list(related_map[concept])

    with open(out / "training_curves.json", "w") as f:
        json.dump(all_curves, f, indent=2)
    print(f"[Phase 2] all SPMs saved to {out}")
    return all_curves


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str,
                        default="unified/output/algB",
                        help="Path to the Phase-1 edited SD pipeline")
    parser.add_argument("--output_dir", type=str,
                        default="unified/output/unified_spm")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gba_gamma", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"[Phase 2] loading edited SD pipeline from {args.base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.float32, safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    train_unified_phase2(
        pipe, output_dir=args.output_dir,
        iterations=args.iterations, lr=args.lr, gba_gamma=args.gba_gamma,
        device=str(device),
    )


if __name__ == "__main__":
    main()
