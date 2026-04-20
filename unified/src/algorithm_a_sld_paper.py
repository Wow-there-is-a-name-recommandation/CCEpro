"""
Algorithm A — Inference-only Continual Concept Erasure (paper-aligned SLD variant).

This version keeps the original three-layer structure:
    1. Prompt subspace projection + replacement (token-wise, score-gated)
    2. Forbidden direction aggregation (one negative prompt per target)
    3. Reverse guidance during sampling

But Layer 3 is changed to follow the Safe Latent Diffusion (SLD) paper more
closely than the original linear-decay implementation.

Key differences vs. the previous file:
    - Replaces linear reverse-guidance decay with step-based warm-up.
    - Uses element-wise thresholded masking for safety guidance.
    - Adds momentum that is accumulated during warm-up and applied afterwards.
    - Separates prompt-level score gating from SLD warm-up hyperparameters.

Notes:
    - The SLD paper defines safety guidance in epsilon/noise-prediction space:
          eps_bar = eps_uncond + s_g * (eps_cond - eps_uncond) - delta_t
          delta_t = mu * (eps_forbid - eps_uncond) + s_m * nu_t
          nu_{t+1} = beta_m * nu_t + (1 - beta_m) * delta_t
      where mu is an element-wise thresholded scaling term.
    - The exact OCR of Eq. (5)-(6) is imperfect in the parsed PDF text, so this
      implementation follows the intended behavior described in the paper text:
      only dimensions leaning toward the forbidden concept are suppressed.
    - We keep the existing prompt-level concept-score gate because it is part of
      your Algorithm A design, even though it is not part of vanilla SLD.

Usage:
    from unified.src.algorithm_a_sld_paper import AlgorithmAPipeline
    pipe = AlgorithmAPipeline(device="cuda:0")
    img  = pipe.generate("a photo of superman flying", seed=0)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from unified.src.common import (
    ERASED_ORDER,
    REPLACEMENT,
    SUBSPACE_PROMPTS,
    build_concept_subspace,
    build_forbidden_direction,
    build_replacement_direction,
    concept_score,
    encode_prompt_mean,
    encode_prompt_sequence,
    load_sd14,
    pick_device,
    project_onto_subspace,
    score_to_strength,
)


@dataclass
class ConceptMemory:
    """Precomputed memory for one erased concept."""

    U: torch.Tensor           # [d, k] orthonormal subspace
    v_rep: torch.Tensor       # [d] replacement direction (mean embedding)
    g_seq: torch.Tensor       # [1, 77, d] forbidden direction / negative cond
    e_mean: torch.Tensor      # [d] concept mean embedding (for residual scores)
    name: str


@dataclass
class AlgAConfig:
    # Layer 1 — prompt filtering
    alpha_proj: float = 10.0          # subspace-projection ramp steepness

    # Base diffusion / CFG
    cfg: float = 7.5                  # classifier-free guidance scale
    n_steps: int = 30                 # DDIM steps
    image_size: int = 512

    # Prompt-level gate (Algorithm A specific; NOT paper warm-up)
    score_gate: float = 0.3           # below this concept score, no reverse guidance
    gate_scale: float = 1.0           # optional extra gain on aggregated prompt gate

    # Paper-aligned SLD params
    safety_scale: float = 1000.0      # s_S, paper recommends roughly [100, 3000]
    safety_threshold: float = 0.01    # λ, paper recommends roughly [0.0, 0.03]
    warmup_steps: int = 10            # τ, paper recommends roughly [5, 20]
    momentum_scale: float = 0.3       # s_m, paper recommends [0, 0.5]
    momentum_beta: float = 0.4        # β_m, paper recommends [0.3, 0.7]

    # Stability helpers
    clip_mu_to_one: bool = True       # match paper intuition: clip scaling factor
    max_gate_eff: float | None = None # optionally cap aggregated prompt-level gate
    debug_steps: bool = True


class AlgorithmAPipeline:
    """Inference-only pipeline implementing Algorithm A with SLD-style guidance."""

    def __init__(
        self,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
        concepts: Sequence[str] = ERASED_ORDER,
        config: AlgAConfig | None = None,
        pipe: StableDiffusionPipeline | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.cfg = config or AlgAConfig()
        self.concepts = list(concepts)

        if pipe is None:
            self.pipe = load_sd14(device=device, dtype=dtype)
        else:
            self.pipe = pipe

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        self.memory: dict[str, ConceptMemory] = {}
        for c in self.concepts:
            self.memory[c] = ConceptMemory(
                U=build_concept_subspace(self.pipe, SUBSPACE_PROMPTS[c], device),
                v_rep=build_replacement_direction(self.pipe, REPLACEMENT[c], device),
                g_seq=build_forbidden_direction(self.pipe, SUBSPACE_PROMPTS[c], device),
                e_mean=encode_prompt_mean(self.pipe, c, device),
                name=c,
            )

        self.null_seq = encode_prompt_sequence(self.pipe, "", device)  # [1, 77, d]

    # ------------------------------------------------------------------ #
    # Layer 1 — prompt embedding filtering
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _filter_prompt(self, prompt_seq: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Apply sequential subspace projection + replacement across all concepts.

        Returns:
            filtered prompt sequence,
            dict of per-concept prompt scores for the later reverse-guidance gate.
        """
        p = prompt_seq.clone()
        scores: dict[str, float] = {}

        for c in self.concepts:
            mem = self.memory[c]
            p_mean = p.mean(dim=1).squeeze(0)  # [d]
            score = concept_score(p_mean.unsqueeze(0), mem.U).item()
            scores[c] = score

            beta = score_to_strength(score, self.cfg.alpha_proj)
            if isinstance(beta, torch.Tensor):
                beta = beta.item()

            if beta < 1e-3:
                continue

            proj = project_onto_subspace(p, mem.U)  # [1, 77, d]
            v_rep_broadcast = mem.v_rep.view(1, 1, -1)
            p = p - beta * proj + beta * v_rep_broadcast

        return p, scores

    # ------------------------------------------------------------------ #
    # Reverse-guidance helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _aggregate_forbidden_embedding(
        self,
        concept_scores: dict[str, float],
    ) -> tuple[torch.Tensor, dict[str, float], float]:
        """Aggregate forbidden prompt embeddings using Algorithm A prompt-level scores.

        This keeps your original idea:
            - concept scores computed from filtered prompt embeddings
            - concepts below score_gate are not activated
            - active concepts are mixed into a single forbidden embedding

        Returns:
            g_agg: [1, 77, d] aggregated forbidden conditioning
            gate_per: per-concept normalized gate strengths
            gate_eff: summed active strength (optionally clipped)
        """
        cfg = self.cfg

        gate_per: dict[str, float] = {}
        for c, s in concept_scores.items():
            gate_per[c] = max(0.0, s - cfg.score_gate) / max(1e-6, 1.0 - cfg.score_gate)

        gate_eff = cfg.gate_scale * sum(gate_per.values())
        if cfg.max_gate_eff is not None:
            gate_eff = min(gate_eff, cfg.max_gate_eff)

        if gate_eff <= 0.0:
            return self.null_seq, gate_per, 0.0

        total_w = sum(gate_per.values()) or 1.0
        g_agg = torch.zeros_like(self.null_seq)
        for c, w_c in gate_per.items():
            if w_c > 0.0:
                g_agg += (w_c / total_w) * self.memory[c].g_seq

        return g_agg, gate_per, gate_eff

    @torch.no_grad()
    def _build_mu(self, eps_cond, eps_forbid):
        diff = eps_cond - eps_forbid
        phi = self.cfg.mu_scale * diff   # 식 6 반영

        active = phi < self.cfg.safety_threshold
        scale = torch.clamp(phi.abs(), min=1.0)

        mu = torch.where(
            active,
            scale,
            torch.zeros_like(phi),
        )
        return mu

    # ------------------------------------------------------------------ #
    # Layer 3 — reverse guidance denoising loop
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _sample(
        self,
        prompt_seq: torch.Tensor,
        concept_scores: dict[str, float],
        seed: int = 0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Custom DDIM sampling with CFG + paper-aligned SLD reverse guidance."""
        device = self.device
        dtype = self.dtype
        cfg = self.cfg

        unet = self.pipe.unet
        vae = self.pipe.vae
        scheduler = self.pipe.scheduler

        scheduler.set_timesteps(cfg.n_steps, device=device)

        h = w = cfg.image_size // 8
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(seed)

        latents = torch.randn(
            (1, unet.config.in_channels, h, w),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        latents = latents * scheduler.init_noise_sigma

        g_agg, gate_per, gate_eff = self._aggregate_forbidden_embedding(concept_scores)

        timesteps = scheduler.timesteps
        n_total = len(timesteps)

        momentum: torch.Tensor | None = None

        for step_idx, t in enumerate(timesteps):
            latent_in = scheduler.scale_model_input(latents, t)

            # Three passes batched together: prompt / uncond / forbidden.
            hidden = torch.cat([prompt_seq, self.null_seq, g_agg], dim=0)
            latent_rep = latent_in.repeat(3, 1, 1, 1)
            eps_all = unet(latent_rep, t, encoder_hidden_states=hidden).sample
            eps_cond, eps_uncond, eps_forbid = eps_all.chunk(3)

            # Base classifier-free guidance.
            eps_cfg = eps_uncond + cfg.cfg * (eps_cond - eps_uncond)

            # If no concept is active after prompt-level gating, fall back to CFG.
            if gate_eff <= 0.0:
                eps_final = eps_cfg
                latents = scheduler.step(eps_final, t, latents).prev_sample
                if cfg.debug_steps and step_idx in [0, n_total // 2, n_total - 1]:
                    print(
                        f"step={step_idx}, gate_eff=0.0000, warmup={'on' if step_idx < cfg.warmup_steps else 'off'}, "
                        f"delta_abs_mean=0.000000"
                    )
                continue

            # Paper Eq. (4): unsafe direction in epsilon space.
            unsafe_dir = eps_forbid - eps_uncond

            # Paper Eq. (5)-style element-wise thresholding.
            mu = self._build_mu(eps_cond, eps_forbid)

            # Base safety guidance. We keep prompt-level gate_eff as an outer gain
            # because it is part of your Algorithm A design, not vanilla SLD.
            delta_base = gate_eff * mu * unsafe_dir

            if momentum is None:
                momentum = torch.zeros_like(delta_base)

            # Paper Eq. (7)-(8): accumulate momentum even during warm-up.
            delta_t = delta_base + cfg.momentum_scale * momentum
            momentum = cfg.momentum_beta * momentum + (1.0 - cfg.momentum_beta) * delta_t

            # Paper warm-up: do not apply safety guidance yet, but momentum has already
            # been updated above.
            if step_idx < cfg.warmup_steps:
                eps_final = eps_cfg
            else:
                eps_final = eps_cfg - delta_t

            if cfg.debug_steps and step_idx in [0, n_total // 2, n_total - 1]:
                active_frac = (mu > 0).float().mean().item()
                print(
                    f"step={step_idx}, gate_eff={gate_eff:.4f}, warmup={'on' if step_idx < cfg.warmup_steps else 'off'}, "
                    f"mu_active={active_frac:.4f}, delta_abs_mean={delta_t.abs().mean().item():.6f}"
                )

            latents = scheduler.step(eps_final, t, latents).prev_sample

        latents = latents / vae.config.scaling_factor
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(self, prompt: str, seed: int = 0) -> torch.Tensor:
        """Generate a single image. Returns a [1, 3, H, W] tensor in [0,1]."""
        p_raw = encode_prompt_sequence(self.pipe, prompt, self.device)
        p_filt, scores = self._filter_prompt(p_raw)
        img = self._sample(p_filt, scores, seed=seed)
        return img

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: Sequence[str],
        seeds: Sequence[int] | None = None,
    ) -> list[torch.Tensor]:
        """Generate images for a batch of prompts (sequential for simplicity)."""
        if seeds is None:
            seeds = list(range(len(prompts)))
        return [self.generate(p, seed=s) for p, s in zip(prompts, seeds)]


if __name__ == "__main__":
    import argparse
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of superman flying over metropolis",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="unified/output/algA_sld_paper_smoke.png",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dev = pick_device(args.device)
    pipe = AlgorithmAPipeline(device=str(dev))
    img = pipe.generate(args.prompt, seed=args.seed)
    arr = (img.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(args.out)
    print(f"[algA_sld_paper] smoke test done → {args.out}")
