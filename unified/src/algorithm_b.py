"""
Algorithm B — Closed-Form Cross-Attention Editing with Adaptive Feedback.

This edits the cross-attention key/value projections in the SD v1.4 UNet so that
each target concept maps to the null-embedding output while related / anchor
concepts remain anchored to the original model's output.

Differences from the UCE baseline (Gandikota et al., WACV 2024):
    1. Sequential processing — one concept is erased per outer step, with
       previously-erased concepts re-added as additional "target → null" terms.
    2. Adaptive Σ — diagonal per-concept importance weights adjusted after each
       step based on a cheap output-space proxy:
           - revival check: cos(W_curr·e_z , W_0·e_null) — if high, bump α_penalty.
           - damage  check: cos(W_curr·e_p , W_0·e_p)   — if low,  bump β_protect.
       The raw algorithm prescribes image-space CS, which is prohibitive; we
       use an output-embedding cosine as a tractable surrogate.
    3. Cumulative anchors — related concepts from previous steps are carried
       forward so preservation is enforced across the full task stream.

Usage:
    python -m unified.src.algorithm_b --output_dir unified/output/algB --lam 0.1

Outputs:
    A Stable Diffusion pipeline directory (diffusers format) that can be
    loaded with StableDiffusionPipeline.from_pretrained(...).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from unified.src.common import (
    COCO_ANCHORS,
    ERASED_ORDER,
    RELATED,
    concept_key,
    encode_prompt_mean,
    encode_prompts_mean,
    flush,
    load_sd14,
    pick_device,
)


# --------------------------------------------------------------------------- #
# Feedback helpers — cheap output-embedding proxies
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _avg_layer_cosine(
    unet: nn.Module, e_a: torch.Tensor, e_b: torch.Tensor,
) -> float:
    """Average cosine between (W · e_a) and (W · e_b) across all cross-attn
    to_k / to_v layers of the UNet.

    Works in float32 regardless of UNet precision so the number is stable
    across the feedback check. Each embedding is expected to be [d].
    """
    if e_a.dim() == 1: e_a = e_a.unsqueeze(0)   # [1, d]
    if e_b.dim() == 1: e_b = e_b.unsqueeze(0)

    sims: list[float] = []
    for name, module in unet.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not (name.endswith("to_k") or name.endswith("to_v")):
            continue
        if "attn2" not in name:
            continue

        W = module.weight.data.float()                # [out, in]
        # embed dim might not match W.in_features — e.g., cross-attn on
        # different resolutions. UNet cross-attn in SD v1.4 always takes
        # encoder_hidden_states of dim 768, so in_features == 768 everywhere.
        if W.shape[1] != e_a.shape[1]:
            continue
        a_out = (e_a @ W.T).squeeze(0)                # [out]
        b_out = (e_b @ W.T).squeeze(0)
        sim = torch.nn.functional.cosine_similarity(
            a_out.unsqueeze(0), b_out.unsqueeze(0)
        ).item()
        sims.append(sim)

    if not sims:
        return 0.0
    return float(sum(sims) / len(sims))


# --------------------------------------------------------------------------- #
# Core editing routine
# --------------------------------------------------------------------------- #

@torch.no_grad()
def closed_form_edit_one_step(
    unet: nn.Module,
    W0_cache: dict[str, torch.Tensor],
    targets_to_null: list[torch.Tensor],       # list of [d]
    sigmas_null: list[float],                   # per-target weights
    anchors_preserve: list[torch.Tensor],       # list of [d]
    sigmas_anchor: list[float],                 # per-anchor weights
    e_null: torch.Tensor,                       # [d]
    lam: float = 0.1,
    device: str = "cuda:0",
) -> int:
    """One closed-form update pass across all cross-attn K/V layers.

    Solves:
        W_new = (Σ_i σ_i (W_0 · e_null) · e_i^T  +  λ Σ_j σ_j (W_0 · e_j) · e_j^T)
                · inv( Σ_i σ_i e_i · e_i^T + λ Σ_j σ_j e_j · e_j^T + ε I )

    Note the σ weights appear on both sides so they act as importance weights
    in a weighted least-squares solve.
    """
    edited = 0
    for name, module in unet.named_modules():
        if not isinstance(module, nn.Linear): continue
        if not (name.endswith("to_k") or name.endswith("to_v")): continue
        if "attn2" not in name: continue

        W0 = W0_cache[name]                            # [out, in] fp32
        in_dim = W0.shape[1]

        if e_null.shape[0] != in_dim:
            continue

        num = torch.zeros_like(W0)
        den = torch.zeros(in_dim, in_dim, device=device, dtype=torch.float32)

        v_null = (W0 @ e_null.unsqueeze(1)).squeeze(1)   # [out]

        # Erase terms: map e_i to W0·e_null
        for e_i, sigma_i in zip(targets_to_null, sigmas_null):
            e_i_col = e_i.unsqueeze(1).float()
            num += sigma_i * (v_null.unsqueeze(1) @ e_i_col.T)
            den += sigma_i * (e_i_col @ e_i_col.T)

        # Preserve terms: map e_j to W0·e_j (anchor to original)
        for e_j, sigma_j in zip(anchors_preserve, sigmas_anchor):
            e_j_col = e_j.unsqueeze(1).float()
            v_j = (W0 @ e_j_col).float()               # [out, 1]
            num += lam * sigma_j * (v_j @ e_j_col.T)
            den += lam * sigma_j * (e_j_col @ e_j_col.T)

        den += 1e-6 * torch.eye(in_dim, device=device, dtype=torch.float32)
        W_new = num @ torch.linalg.inv(den)
        module.weight.data = W_new.to(module.weight.dtype)
        edited += 1
    return edited


@torch.no_grad()
def algorithm_b(
    pipe,
    concept_sequence: list[str] = ERASED_ORDER,
    related_map: dict[str, list[str]] = RELATED,
    coco_anchors: list[str] = COCO_ANCHORS,
    lam: float = 0.1,
    tau_erase: float = 0.35,          # revival threshold  (cos sim on W·e_z vs W0·e_null)
    tau_retain: float = 0.95,         # damage threshold   (cos sim on W·e_p vs W0·e_p)
    alpha_penalty: float = 1.0,       # bump for revival   (added to σ for that concept)
    beta_protect: float = 1.0,        # bump for damage    (added to σ for that anchor)
    device: str = "cuda:0",
    log_path: Path | None = None,
) -> dict:
    """Run Algorithm B on `pipe.unet` in-place. Returns a diagnostics dict."""

    unet = pipe.unet

    # -- Snapshot of W_0 for every edited layer (needed for anchoring & feedback) --
    W0_cache: dict[str, torch.Tensor] = {}
    for name, module in unet.named_modules():
        if not isinstance(module, nn.Linear): continue
        if not (name.endswith("to_k") or name.endswith("to_v")): continue
        if "attn2" not in name: continue
        W0_cache[name] = module.weight.data.clone().float()

    print(f"[algB] cached W_0 for {len(W0_cache)} cross-attn K/V layers")

    # -- Text encoding of all concepts we'll need --
    e_null = encode_prompt_mean(pipe, "", device).float()
    emb_cache: dict[str, torch.Tensor] = {}
    def enc(p: str) -> torch.Tensor:
        if p not in emb_cache:
            emb_cache[p] = encode_prompt_mean(pipe, p, device).float()
        return emb_cache[p]

    # Warm up cache
    for c in concept_sequence:
        enc(c)
    for c, rels in related_map.items():
        for r in rels: enc(r)
    for a in coco_anchors: enc(a)

    # -- State --
    Z: list[str] = []            # previously erased
    R_past: list[str] = []       # accumulated related
    sigma_map: dict[str, float] = {}   # concept → importance weight

    diagnostics = {"steps": [], "lam": lam, "tau_erase": tau_erase,
                   "tau_retain": tau_retain}

    for step, c_t in enumerate(concept_sequence):
        print(f"\n[algB] step {step + 1}/{len(concept_sequence)}  erasing: '{c_t}'")

        # Initialise σ for this concept if we haven't seen it
        sigma_map.setdefault(c_t, 1.0)
        for r in related_map[c_t] + R_past + Z + coco_anchors:
            sigma_map.setdefault(r, 1.0)

        # ── 1a. Feedback loop — check revival & collateral damage ──
        if Z:
            for z in Z:
                # Before-update revival: cos(W·e_z , W0·e_null). After successful
                # erasure of z at the previous step this should be close to 1.
                # If it has drifted down we bump α_penalty.
                cos_rev = _avg_layer_cosine(unet, enc(z), e_null)
                # "cos_rev far from 1" == revival; encode as a normalised score.
                # We call it 'revived' when cos_rev < tau_erase (e.g. 0.35).
                revived = cos_rev < tau_erase
                if revived:
                    sigma_map[z] += alpha_penalty
                    print(f"[algB] revival detected for '{z}' (cos={cos_rev:.3f})"
                          f"  → σ bumped to {sigma_map[z]:.2f}")

        for p in (related_map[c_t] + R_past):
            # "damaged" when W_curr · e_p has drifted from W_0 · e_p in output
            # space. Surrogate for the image-space CS check prescribed by the
            # algorithm spec (which would require expensive generations).
            cos_dmg = _avg_pair_W_vs_W0(unet, W0_cache, enc(p))
            damaged = cos_dmg < tau_retain
            if damaged:
                sigma_map[p] += beta_protect
                print(f"[algB] damage detected for '{p}' (cos={cos_dmg:.3f})"
                      f"  → σ bumped to {sigma_map[p]:.2f}")

        # ── 1b. Build constraint sets ──
        targets_to_null_embs: list[torch.Tensor] = [enc(c_t)]
        sigmas_null: list[float] = [sigma_map[c_t]]
        for z in Z:
            targets_to_null_embs.append(enc(z))
            sigmas_null.append(sigma_map[z])

        anchors_preserve_embs: list[torch.Tensor] = []
        sigmas_anchor: list[float] = []
        current_anchor_set = list(
            dict.fromkeys(related_map[c_t] + R_past + coco_anchors)
        )
        for p in current_anchor_set:
            anchors_preserve_embs.append(enc(p))
            sigmas_anchor.append(sigma_map[p])

        # ── 1c. Closed-form update ──
        n_edited = closed_form_edit_one_step(
            unet, W0_cache,
            targets_to_null=targets_to_null_embs, sigmas_null=sigmas_null,
            anchors_preserve=anchors_preserve_embs, sigmas_anchor=sigmas_anchor,
            e_null=e_null, lam=lam, device=device,
        )
        print(f"[algB] edited {n_edited} layers")

        # ── 1d. Update state ──
        Z.append(c_t)
        R_past += related_map[c_t]

        # Record diagnostics after this step
        step_diag = {
            "step": step + 1, "concept": c_t,
            "erased_so_far": list(Z),
            "n_edited_layers": n_edited,
            "sigmas": {k: sigma_map[k] for k in (Z + R_past[:8])},
        }
        # Post-step CS proxies
        step_diag["post_cos_null_to_erased"] = {
            z: round(_avg_layer_cosine(unet, enc(z), e_null), 4) for z in Z
        }
        step_diag["post_cos_preserved_vs_W0"] = {
            p: round(_avg_pair_W_vs_W0(unet, W0_cache, enc(p)), 4)
            for p in related_map[c_t]
        }
        diagnostics["steps"].append(step_diag)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        print(f"[algB] wrote diagnostics to {log_path}")
    return diagnostics


@torch.no_grad()
def _avg_pair_W_vs_W0(
    unet: nn.Module, W0_cache: dict[str, torch.Tensor], e: torch.Tensor,
) -> float:
    """Cosine between (W_curr · e) and (W_0 · e), averaged across edited layers.

    High value ≈ preservation; low value ≈ collateral damage.
    """
    if e.dim() == 1: e = e.unsqueeze(0)   # [1, d]
    sims: list[float] = []
    for name, module in unet.named_modules():
        if not isinstance(module, nn.Linear): continue
        if not (name.endswith("to_k") or name.endswith("to_v")): continue
        if "attn2" not in name: continue
        W = module.weight.data.float()
        W0 = W0_cache[name]
        if W.shape[1] != e.shape[1]:
            continue
        a_out = (e @ W.T).squeeze(0)
        b_out = (e @ W0.T).squeeze(0)
        sim = torch.nn.functional.cosine_similarity(
            a_out.unsqueeze(0), b_out.unsqueeze(0)
        ).item()
        sims.append(sim)
    if not sims: return 0.0
    return float(sum(sims) / len(sims))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="unified/output/algB")
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--tau_erase", type=float, default=0.35)
    parser.add_argument("--tau_retain", type=float, default=0.95)
    parser.add_argument("--alpha_penalty", type=float, default=1.0)
    parser.add_argument("--beta_protect", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = pick_device(args.device)
    pipe = load_sd14(device=device, dtype=torch.float32)

    algorithm_b(
        pipe,
        lam=args.lam,
        tau_erase=args.tau_erase, tau_retain=args.tau_retain,
        alpha_penalty=args.alpha_penalty, beta_protect=args.beta_protect,
        device=str(device),
        log_path=Path(args.output_dir) / "diagnostics.json",
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(args.output_dir)
    print(f"[algB] saved edited pipeline to {args.output_dir}")
    flush()


if __name__ == "__main__":
    main()
