"""
Unified Continual Concept Erasure — 3-phase pipeline.

Phase 1 (Algorithm B): closed-form cross-attention edit → θ_edited
Phase 2 (SPM / GBA)  : rank-1 adapters trained on θ_edited         → Θ pool
Phase 3 (Inference)  : three-layer defense
    Layer 1 — prompt subspace projection + replacement (from Alg A)
    Layer 2 — SPM module composition (one per concept, activated by prompt match)
    Layer 3 — adaptive reverse guidance (λ ∝ residual concept score)

This module implements only the inference-time composition (Phase 3). Phases 1
and 2 are produced by `algorithm_b.py` and `train_spm.py` respectively.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from unified.src.algorithm_a import AlgAConfig, AlgorithmAPipeline, ConceptMemory
from unified.src.common import (
    ERASED_ORDER, RELATED, REPLACEMENT, SUBSPACE_PROMPTS,
    build_concept_subspace, build_forbidden_direction, build_replacement_direction,
    concept_key, concept_score, encode_prompt_mean, encode_prompt_sequence,
    load_sd14, pick_device, project_onto_subspace, score_to_strength,
)
from unified.src.spm import SPMNetwork, SPMPool


@dataclass
class UnifiedConfig(AlgAConfig):
    # Layer 2 — SPM activation
    spm_match_threshold: float = 0.3   # concept score above which SPM activates
    spm_max_strength: float = 1.0      # multiplier cap
    # Layer 3 — adaptive reverse guidance (overrides AlgA defaults)
    lam_reverse: float = 1.5           # softer than Alg A, since SPM carries load
    tau_safe: float = 0.35


class UnifiedPipeline:
    """End-to-end inference pipeline for the Unified Algorithm."""

    def __init__(
        self,
        edited_pipe_path: str,          # directory from algorithm_b.py output
        spm_dir: str,                   # directory from train_spm.py output
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
        concepts: Sequence[str] = ERASED_ORDER,
        config: UnifiedConfig | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.cfg = config or UnifiedConfig()
        self.concepts = list(concepts)

        # ── Load Phase-1 edited pipeline ──
        pipe = StableDiffusionPipeline.from_pretrained(
            edited_pipe_path, torch_dtype=dtype, safety_checker=None,
        ).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe

        # ── Phase-3 Layer 1 memory ──
        self.memory: dict[str, ConceptMemory] = {}
        for c in self.concepts:
            self.memory[c] = ConceptMemory(
                U=build_concept_subspace(pipe, SUBSPACE_PROMPTS[c], device),
                v_rep=build_replacement_direction(pipe, REPLACEMENT[c], device),
                g_seq=build_forbidden_direction(pipe, SUBSPACE_PROMPTS[c], device),
                e_mean=encode_prompt_mean(pipe, c, device),
                name=c,
            )
        self.null_seq = encode_prompt_sequence(pipe, "", device)

        # ── Phase-3 Layer 2: attach SPMs ──
        self.pool = SPMPool()
        for c in self.concepts:
            path = Path(spm_dir) / f"spm_{concept_key(c)}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing SPM checkpoint: {path}")
            net = SPMNetwork.from_file(path, pipe.unet)
            self.pool.add(c, net)
        self.pool.disable_all()

    # --------------------------------------------------------------------- #
    # Layer 1 — prompt subspace filtering (reuses Alg A logic)
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def _filter_prompt(self, prompt_seq: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        p = prompt_seq.clone()
        scores: dict[str, float] = {}

        for c in self.concepts:
            mem = self.memory[c]
            p_mean = p.mean(dim=1).squeeze(0)
            score = concept_score(p_mean.unsqueeze(0), mem.U).item()
            scores[c] = score
            beta = score_to_strength(score, self.cfg.alpha_proj)
            if isinstance(beta, torch.Tensor):
                beta = beta.item()
            if beta < 1e-3:
                continue
            proj = project_onto_subspace(p, mem.U)
            v_rep_b = mem.v_rep.view(1, 1, -1)
            p = p - beta * proj + beta * v_rep_b

        return p, scores

    # --------------------------------------------------------------------- #
    # Layer 2 — SPM activation via prompt matching
    # --------------------------------------------------------------------- #

    def _spm_weights(self, concept_scores: dict[str, float]) -> dict[str, float]:
        """Translate concept scores → SPM multiplier weights."""
        weights: dict[str, float] = {}
        thr = self.cfg.spm_match_threshold
        cap = self.cfg.spm_max_strength
        for c, s in concept_scores.items():
            if s < thr:
                weights[c] = 0.0
            else:
                # Linear ramp from 0 at s=thr to cap at s=1.0
                w = (s - thr) / max(1e-6, 1 - thr)
                weights[c] = min(cap, max(0.0, w))
        return weights

    # --------------------------------------------------------------------- #
    # Layer 3 — reverse guidance with adaptive λ
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def _sample(
        self,
        prompt_seq: torch.Tensor,
        concept_scores: dict[str, float],
        seed: int = 0,
    ) -> torch.Tensor:
        cfg = self.cfg
        unet = self.pipe.unet
        vae = self.pipe.vae
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(cfg.n_steps, device=self.device)

        # Sample initial latents
        h = w = cfg.image_size // 8
        g = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn((1, unet.config.in_channels, h, w),
                              generator=g, device=self.device, dtype=self.dtype)
        latents = latents * scheduler.init_noise_sigma

        # Reverse-guidance strengths
        lam_per: dict[str, float] = {}
        for c, s in concept_scores.items():
            lam_per[c] = cfg.lam_reverse * max(0.0, s - cfg.tau_safe) \
                         / max(1e-6, 1 - cfg.tau_safe)
        lam_eff = sum(lam_per.values())

        # Aggregated forbidden embedding
        if lam_eff > 0.0:
            total_w = sum(lam_per.values())
            g_agg = torch.zeros_like(self.null_seq)
            for c, w_c in lam_per.items():
                if w_c > 0:
                    g_agg += (w_c / total_w) * self.memory[c].g_seq
        else:
            g_agg = self.null_seq

        # Apply SPM multiplier schedule (fixed for the whole trajectory — a
        # natural choice; the prompt-score doesn't change during sampling)
        self.pool.set_weights(self._spm_weights(concept_scores))

        for t in scheduler.timesteps:
            latent_in = scheduler.scale_model_input(latents, t)
            hidden = torch.cat([prompt_seq, self.null_seq, g_agg], dim=0)
            eps_all = unet(latent_in.repeat(3, 1, 1, 1), t,
                           encoder_hidden_states=hidden).sample
            eps_cond, eps_uncond, eps_forbid = eps_all.chunk(3)

            eps_cfg = eps_uncond + cfg.cfg * (eps_cond - eps_uncond)
            eps_final = eps_cfg - lam_eff * (eps_forbid - eps_uncond)

            latents = scheduler.step(eps_final, t, latents).prev_sample

        # Disable SPM after sampling to avoid leaking into other batches
        self.pool.disable_all()

        latents = latents / vae.config.scaling_factor
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def generate(self, prompt: str, seed: int = 0) -> torch.Tensor:
        p_raw = encode_prompt_sequence(self.pipe, prompt, self.device)
        p_filt, scores = self._filter_prompt(p_raw)
        img = self._sample(p_filt, scores, seed=seed)
        return img

    @torch.no_grad()
    def generate_batch(self, prompts: Sequence[str],
                       seeds: Sequence[int] | None = None) -> list[torch.Tensor]:
        if seeds is None:
            seeds = list(range(len(prompts)))
        return [self.generate(p, seed=s) for p, s in zip(prompts, seeds)]


if __name__ == "__main__":
    # Smoke test
    import argparse
    from PIL import Image
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--edited_pipe", type=str, default="unified/output/algB")
    parser.add_argument("--spm_dir", type=str, default="unified/output/unified_spm")
    parser.add_argument("--prompt", type=str,
                        default="a photo of superman flying over metropolis")
    parser.add_argument("--out", type=str, default="unified/output/unified_smoke.png")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dev = str(pick_device(args.device))
    pipe = UnifiedPipeline(
        edited_pipe_path=args.edited_pipe, spm_dir=args.spm_dir, device=dev,
    )
    img = pipe.generate(args.prompt, seed=args.seed)
    arr = (img.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(args.out)
    print(f"[unified] smoke → {args.out}")
