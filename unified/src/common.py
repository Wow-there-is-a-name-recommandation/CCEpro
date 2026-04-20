"""
Self-contained utilities for the Unified Continual Concept Erasure project.

No imports from the parent SPM/ directory — this module is deliberately standalone
so the unified/ sub-project can be packaged independently.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, StableDiffusionPipeline


# --------------------------------------------------------------------------- #
# Benchmark specification (from AI_Algorithm_Assignment.pdf)
# --------------------------------------------------------------------------- #

ERASED_ORDER: list[str] = ["superman", "van gogh", "snoopy"]

RELATED: dict[str, list[str]] = {
    "superman": ["batman", "thor", "wonder woman", "shazam"],
    "van gogh": ["picasso", "monet", "paul gauguin", "caravaggio"],
    "snoopy": ["mickey", "spongebob", "pikachu", "hello kitty"],
}

# Replacement concepts used by Algorithm A's prompt filtering (semantically closest
# neutral categories). Kept deliberately generic so the replacement does not
# introduce a new bias.
REPLACEMENT: dict[str, str] = {
    "superman": "a person",
    "van gogh": "a painting",
    "snoopy": "a dog",
}

# Related prompt sets for concept subspace construction (S in the algorithms).
# We include the concept itself plus a few paraphrases so the subspace captures
# lexical variation.
SUBSPACE_PROMPTS: dict[str, list[str]] = {
    "superman": [
        "superman", "muscle hero with cloak", "man of steel",
        "man wearing blue and red cloth", "clark kent",
    ],
    "van gogh": [
        "van gogh", "strong contrast of deep blue and bright yellow", "in the style of van gogh",
        "expressive swirling brush strokes", "distorted perspective expressive painting",
    ],
    "snoopy": [
        "snoopy", "a cartoon of snoopy", "a small white cartoon dog with black ears",
        "minimalist cartoon line drawing dog", "a cartoon beagle holding a blanket",
    ],
}

# MS-COCO category anchor strings used for global preservation in closed-form edits.
# Small curated subset of common COCO classes.
COCO_ANCHORS: list[str] = [
    "a photo of a person", "a photo of a cat", "a photo of a dog",
    "a photo of a car", "a photo of a chair", "a photo of a bicycle",
    "a photo of a bird", "a photo of a horse", "a photo of a tree",
    "a photo of a house", "a photo of food on a plate", "a photo of a beach",
]


def concept_key(concept: str) -> str:
    """Normalize a concept string to a filesystem-safe key."""
    return concept.replace(" ", "_").lower()


# --------------------------------------------------------------------------- #
# Device / memory helpers
# --------------------------------------------------------------------------- #

def pick_device(prefer: str = "cuda:0") -> torch.device:
    """Pick a device, respecting the H100-NVL-forbidden constraint.

    The forbidden GPU (PCI 00000000:3D:00.0 / index 1) is masked out via
    CUDA_VISIBLE_DEVICES at the shell level — this function just returns the
    user-selected device when CUDA is available.
    """
    if torch.cuda.is_available():
        return torch.device(prefer)
    return torch.device("cpu")


def flush() -> None:
    torch.cuda.empty_cache()
    gc.collect()


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_sd14(
    device: torch.device | str = "cuda:0",
    dtype: torch.dtype = torch.float32,
    scheduler: str = "ddim",
) -> StableDiffusionPipeline:
    """Load Stable Diffusion v1.4 with an optional scheduler swap."""
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype,
        safety_checker=None,
    )
    if scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe


# --------------------------------------------------------------------------- #
# Text embedding utilities
# --------------------------------------------------------------------------- #

@torch.no_grad()
def encode_prompt_sequence(pipe: StableDiffusionPipeline, prompt: str, device: str) -> torch.Tensor:
    """Encode a prompt to its full token-sequence embedding ([1, 77, 768])."""
    tokens = pipe.tokenizer(
        prompt, padding="max_length",
        max_length=pipe.tokenizer.model_max_length, truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    emb = pipe.text_encoder(tokens)[0]
    return emb


@torch.no_grad()
def encode_prompt_mean(pipe: StableDiffusionPipeline, prompt: str, device: str) -> torch.Tensor:
    """Mean-pooled 1-D representation ([768,]) — used for closed-form editing.

    This matches the convention in the UCE reference implementation.
    """
    emb = encode_prompt_sequence(pipe, prompt, device)      # [1, 77, 768]
    return emb.mean(dim=1).squeeze(0)                        # [768]


@torch.no_grad()
def encode_prompts_mean(
    pipe: StableDiffusionPipeline, prompts: Sequence[str], device: str
) -> torch.Tensor:
    """Batch mean-pooled encodings. Returns [N, 768]."""
    rows = [encode_prompt_mean(pipe, p, device) for p in prompts]
    return torch.stack(rows, dim=0)


# --------------------------------------------------------------------------- #
# Subspace / direction builders (used by Algorithm A and Phase 3 of Unified)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def build_concept_subspace(
    pipe: StableDiffusionPipeline, prompts: Sequence[str], device: str,
    n_components: int = 3,
) -> torch.Tensor:
    """Build an orthonormal concept subspace U ∈ R^{d×k} via SVD of prompt embeddings.

    Returns U with columns being the top-k right singular vectors of the prompt
    embedding matrix (row-centered). k is clipped to at most len(prompts).
    """
    embs = encode_prompts_mean(pipe, list(prompts), device)   # [N, 768]
    centered = embs - embs.mean(dim=0, keepdim=True)
    # SVD: centered = U_l S V^T  →  concept subspace = columns of V_T^T
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    k = min(n_components, vt.shape[0])
    basis = vt[:k].T.contiguous()                             # [768, k]
    return basis


@torch.no_grad()
def build_replacement_direction(
    pipe: StableDiffusionPipeline, replacement_prompt: str, device: str,
) -> torch.Tensor:
    """Replacement direction v ∈ R^{d} — mean embedding of the replacement concept."""
    return encode_prompt_mean(pipe, replacement_prompt, device)


@torch.no_grad()
def build_forbidden_direction(
    pipe: StableDiffusionPipeline, prompts: Sequence[str], device: str,
) -> torch.Tensor:
    """Forbidden guidance direction — mean embedding of concept-related prompts.

    Used at inference time as the 'negative' prompt for reverse guidance.
    Returns a full sequence embedding [1, 77, 768] (for diffusion forward pass).
    """
    # We need a full token-sequence embedding (not mean-pooled) to feed into
    # the UNet's cross-attention. Use the first prompt as a canonical representative.
    return encode_prompt_sequence(pipe, prompts[0], device)


def project_onto_subspace(x: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Project x onto span(U). Works batch-wise along the last dim.

    x: [..., d], U: [d, k] with orthonormal columns.
    returns: [..., d] — the projection of x onto span(U).
    """
    # coeffs: x @ U → [..., k];  proj = coeffs @ U^T → [..., d]
    coeffs = x @ U
    return coeffs @ U.T


def concept_score(x: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Concept presence score: cosine between projection and x, in [0,1].

    Higher = prompt lives more inside the concept subspace.
    """
    proj = project_onto_subspace(x, U)
    x_norm = x.norm(dim=-1).clamp_min(1e-8)
    p_norm = proj.norm(dim=-1).clamp_min(1e-8)
    return (proj * x).sum(dim=-1) / (x_norm * p_norm).clamp_min(1e-8)


def score_to_strength(score: torch.Tensor | float, alpha: float) -> torch.Tensor | float:
    """Map a concept score in [0,1] to a subtraction strength β ∈ [0, 1].

    Uses a smooth ramp with steepness α:  β = sigmoid( α · (score - 0.5) ).
    When α is large, β sharply activates when score > 0.5.
    """
    import math
    if isinstance(score, torch.Tensor):
        return torch.sigmoid(alpha * (score - 0.5))
    return 1.0 / (1.0 + math.exp(-alpha * (score - 0.5)))
