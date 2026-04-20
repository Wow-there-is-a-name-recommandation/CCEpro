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
    cfg: float = 2.5                  # classifier-free guidance scale
    n_steps: int = 100                 # DDIM steps
    image_size: int = 512

    # Prompt-level gate (Algorithm A specific; NOT paper warm-up)
    score_gate: float = 0.4           # below this concept score, no reverse guidance
    #gate_scale: float = 1.0           # optional extra gain on aggregated prompt gate

    # Paper-aligned SLD params
    safety_scale: float = 500.0      # s_S, paper recommends roughly [100, 3000]
    safety_threshold: float = 0.01    # λ, paper recommends roughly [0.0, 0.03]
    warmup_steps: int = 10            # τ, paper recommends roughly [5, 20]
    momentum_scale: float = 0.3       # s_m, paper recommends [0, 0.5]
    momentum_beta: float = 0.4        # β_m, paper recommends [0.3, 0.7]

    # Stability helpers
    #clip_mu_to_one: bool = True       # match paper intuition: clip scaling factor
    #max_gate_eff: float | None = None # optionally cap aggregated prompt-level gate
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
        """Apply one-shot token-wise prompt filtering across all concepts.

        Redirect version:
            - remove forbidden subspace component
            - redirect the removed magnitude toward the replacement direction
        """
        p0 = prompt_seq.clone()                      # [1, 77, d]
        p0_tokens = p0.squeeze(0)                   # [77, d]

        scores: dict[str, float] = {}
        total_delta = torch.zeros_like(p0_tokens)   # [77, d]

        rep_scale = 0.5
        eps = 1e-8

        for c in self.concepts:
            mem = self.memory[c]

            # prompt-level summary score (logging / forbidden embedding aggregation)
            p0_mean = p0.mean(dim=1).squeeze(0)                     # [d]
            prompt_score = concept_score(p0_mean.unsqueeze(0), mem.U).item()
            scores[c] = prompt_score

            # token-wise scores from original prompt
            token_scores = concept_score(p0_tokens, mem.U)          # [77]
            token_betas = score_to_strength(token_scores, self.cfg.alpha_proj)
            token_betas = token_betas.to(p0_tokens.dtype)

            active_mask = token_betas > 1e-3
            if not active_mask.any():
                continue

            # forbidden component
            proj_tokens = project_onto_subspace(p0_tokens, mem.U)   # [77, d]

            # magnitude of forbidden component per token
            proj_norms = proj_tokens.norm(dim=-1, keepdim=True)     # [77, 1]

            # normalized replacement direction
            v_rep = mem.v_rep.view(1, -1)                           # [1, d]
            v_rep_unit = v_rep / v_rep.norm(dim=-1, keepdim=True).clamp_min(eps)

            token_betas = token_betas.unsqueeze(-1)                 # [77, 1]

            # redirect removed forbidden magnitude to replacement direction
            replacement_tokens = proj_norms * v_rep_unit            # [77, d]

            delta_c = (
                - token_betas * proj_tokens
                + rep_scale * token_betas * replacement_tokens
            )

            total_delta = total_delta + delta_c

        p_filt = p0_tokens + total_delta
        p_filt = p_filt.unsqueeze(0)

        return p_filt, scores

    # ------------------------------------------------------------------ #
    # Reverse-guidance helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _aggregate_forbidden_embedding(
        self,
        concept_scores: dict[str, float],
    ) -> dict[str, float]:
        """Build per-concept activation weights for reverse guidance.

        IMPORTANT:
            - We do NOT mix forbidden embeddings into one g_agg anymore.
            - Each concept keeps its own forbidden branch.
            - The returned weights are used only to decide which concepts are active
            (and optionally to scale each concept's delta if desired).
            - Reverse-guidance strength itself is still determined inside the SLD-style
            rule from unsafe_dir, mu, and momentum.

        Returns:
            gate_per: dict[str, float]
                Per-concept normalized activation weights in [0, 1].
        """
        cfg = self.cfg

        gate_per: dict[str, float] = {}
        for c, s in concept_scores.items():
            gate_per[c] = max(0.0, s - cfg.score_gate) / max(1e-6, 1.0 - cfg.score_gate)

        return gate_per

    @torch.no_grad()
    def _build_mu(self, eps_cond, eps_forbid):
        diff = eps_cond - eps_forbid
        phi = self.cfg.safety_scale * diff   # 식 6 반영

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
        """Custom DDIM sampling with CFG + per-concept SLD-style reverse guidance.

        In this version:
            - prompt filtering and reverse-guidance strength are separated
            - each concept keeps its own forbidden branch
            - for each concept c:
                eps_forbid_c -> unsafe_dir_c -> mu_c -> delta_c
            - final reverse guidance is the sum over concepts:
                delta_base_total = sum_c delta_c
        """
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

        # per-concept activation weights only
        gate_per = self._aggregate_forbidden_embedding(concept_scores)
        active_concepts = [c for c, w in gate_per.items() if w > 0.0]

        timesteps = scheduler.timesteps
        n_total = len(timesteps)

        momentum: torch.Tensor | None = None
        has_forbidden = len(active_concepts) > 0

        for step_idx, t in enumerate(timesteps):
            latent_in = scheduler.scale_model_input(latents, t)

            # --------------------------------------------------------------
            # Build batched hidden states:
            #   [prompt_seq, null_seq, g_seq(c1), g_seq(c2), ...]
            # --------------------------------------------------------------
            hidden_list = [prompt_seq, self.null_seq]
            for c in active_concepts:
                hidden_list.append(self.memory[c].g_seq)

            hidden = torch.cat(hidden_list, dim=0)
            latent_rep = latent_in.repeat(hidden.shape[0], 1, 1, 1)

            eps_all = unet(latent_rep, t, encoder_hidden_states=hidden).sample

            eps_cond = eps_all[0:1]
            eps_uncond = eps_all[1:2]
            eps_forbid_list = [eps_all[i:i+1] for i in range(2, eps_all.shape[0])]

            # Base classifier-free guidance.
            eps_cfg = eps_uncond + cfg.cfg * (eps_cond - eps_uncond)

            # If no forbidden concept is active, fall back to CFG.
            if not has_forbidden:
                eps_final = eps_cfg
                latents = scheduler.step(eps_final, t, latents).prev_sample
                if cfg.debug_steps and step_idx in [0, n_total // 2, n_total - 1]:
                    print(
                        f"step={step_idx}, forbidden=off, warmup={'on' if step_idx < cfg.warmup_steps else 'off'}, "
                        f"delta_abs_mean=0.000000"
                    )
                continue

            # --------------------------------------------------------------
            # Per-concept reverse guidance
            #   delta_c = mu_c * unsafe_dir_c
            #   delta_base_total = sum_c delta_c
            # --------------------------------------------------------------
            delta_base_total = torch.zeros_like(eps_cfg)
            mu_active_sum = 0.0

            for c, eps_forbid_c in zip(active_concepts, eps_forbid_list):
                unsafe_dir_c = eps_forbid_c - eps_uncond
                mu_c = self._build_mu(eps_cond, eps_forbid_c)

                # 기본형: 컨셉별 delta를 그냥 합산
                delta_c = mu_c * unsafe_dir_c

                # 원하면 아래처럼 컨셉 가중치를 곱할 수 있음
                # delta_c = gate_per[c] * mu_c * unsafe_dir_c

                delta_base_total = delta_base_total + delta_c
                mu_active_sum += (mu_c > 0).float().mean().item()

            if momentum is None:
                momentum = torch.zeros_like(delta_base_total)

            # Paper Eq. (7)-(8): accumulate momentum even during warm-up.
            delta_t = delta_base_total + cfg.momentum_scale * momentum
            momentum = cfg.momentum_beta * momentum + (1.0 - cfg.momentum_beta) * delta_t

            # Paper warm-up: do not apply safety guidance yet, but momentum has already
            # been updated above.
            if step_idx < cfg.warmup_steps:
                eps_final = eps_cfg
            else:
                eps_final = eps_cfg - delta_t

            if cfg.debug_steps and step_idx in [0, n_total // 2, n_total - 1]:
                avg_mu_active = mu_active_sum / max(1, len(active_concepts))
                print(
                    f"step={step_idx}, forbidden={len(active_concepts)} concepts, "
                    f"warmup={'on' if step_idx < cfg.warmup_steps else 'off'}, "
                    f"mu_active_avg={avg_mu_active:.4f}, delta_abs_mean={delta_t.abs().mean().item():.6f}"
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
