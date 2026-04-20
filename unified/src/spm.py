"""
Self-contained rank-1 SPM (Semi-Permeable Membrane) adapter.

A minimal reimplementation of the rank-1 LoRA adapter used in CoSA+GBA.
It attaches to every cross-attention linear layer (to_q, to_k, to_v, to_out.0)
in the UNet and adds a low-rank delta: W ← W + α/r · B · A.

The class supports context-manager semantics (`with spm: ...`) to toggle the
delta on/off so we can cleanly run ε_{θ+ϕ} vs ε_θ without destructive edits.

Storage format is a plain dict of tensors saved via torch.save. No safetensors
dependency in this file — keeps the module self-contained.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# One rank-1 SPM layer (wraps a single Linear module)
# --------------------------------------------------------------------------- #

class SPMLayer(nn.Module):
    """Adds `multiplier · α/r · B · A · x` to the forward pass of `base`."""

    def __init__(self, base: nn.Linear, rank: int = 1, alpha: float = 1.0):
        super().__init__()
        assert isinstance(base, nn.Linear), "SPMLayer only wraps nn.Linear"
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.multiplier: float = 0.0   # start disabled

        in_feat, out_feat = base.in_features, base.out_features
        self.lora_down = nn.Linear(in_feat, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_feat, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

        # Save the original forward so we can restore it on unload.
        self._base_forward = base.forward
        base.forward = self.forward  # monkey-patch

    def forward(self, x):
        base_out = self._base_forward(x)
        if self.multiplier == 0.0:
            return base_out
        lora = self.lora_up(self.lora_down(x)) * (self.scale * self.multiplier)
        return base_out + lora

    def unload(self):
        self.base.forward = self._base_forward


# --------------------------------------------------------------------------- #
# Full SPM network — collection of rank-1 layers attached to every cross-attn
# --------------------------------------------------------------------------- #

# Targets: all linear projections inside every attn2 (cross-attention) block.
# SD v1.4 has 16 cross-attention modules × 4 projections = 64 wrapped layers.
_CROSS_ATTN_TARGET_SUFFIXES = ("to_q", "to_k", "to_v", "to_out.0")


def _is_target(name: str) -> bool:
    if "attn2" not in name:
        return False
    return any(name.endswith(s) for s in _CROSS_ATTN_TARGET_SUFFIXES)


class SPMNetwork(nn.Module):
    def __init__(self, unet: nn.Module, rank: int = 1, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.layers: list[SPMLayer] = []
        self.layer_names: list[str] = []

        for name, module in unet.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not _is_target(name):
                continue
            layer = SPMLayer(module, rank=rank, alpha=alpha)
            layer = layer.to(module.weight.device, dtype=module.weight.dtype)
            # Re-init with fp32 params (fp16/bf16 init would be lossy)
            layer.lora_down.weight.data = layer.lora_down.weight.data.float()
            layer.lora_up.weight.data = layer.lora_up.weight.data.float()
            self.layers.append(layer)
            self.layer_names.append(name)

        # Register as sub-modules so optimizer can find the LoRA params
        self._lora_modules = nn.ModuleList([l.lora_down for l in self.layers] +
                                           [l.lora_up   for l in self.layers])

    def __enter__(self):
        for l in self.layers: l.multiplier = 1.0
        return self

    def __exit__(self, *exc):
        for l in self.layers: l.multiplier = 0.0

    def set_multiplier(self, m: float):
        for l in self.layers: l.multiplier = m

    def parameters(self):
        for l in self.layers:
            yield from l.lora_down.parameters()
            yield from l.lora_up.parameters()

    def lora_state_dict(self) -> dict:
        out = {}
        for name, l in zip(self.layer_names, self.layers):
            out[name + ".lora_down.weight"] = l.lora_down.weight.detach().cpu()
            out[name + ".lora_up.weight"] = l.lora_up.weight.detach().cpu()
        return out

    def load_lora_state_dict(self, state: dict):
        for name, l in zip(self.layer_names, self.layers):
            kd = name + ".lora_down.weight"
            ku = name + ".lora_up.weight"
            if kd in state:
                l.lora_down.weight.data = state[kd].to(l.lora_down.weight.device,
                                                      dtype=l.lora_down.weight.dtype)
            if ku in state:
                l.lora_up.weight.data = state[ku].to(l.lora_up.weight.device,
                                                    dtype=l.lora_up.weight.dtype)

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.lora_state_dict(),
                    "rank": self.rank, "alpha": self.alpha,
                    "layer_names": self.layer_names}, path)

    def unload(self):
        for l in self.layers:
            l.unload()

    @classmethod
    def from_file(cls, path: str | Path, unet: nn.Module) -> "SPMNetwork":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        net = cls(unet, rank=blob["rank"], alpha=blob["alpha"])
        net.load_lora_state_dict(blob["state"])
        return net


# --------------------------------------------------------------------------- #
# Composition of multiple SPM modules at inference time (cumulative multiplier)
# --------------------------------------------------------------------------- #

class SPMPool:
    """Lightweight holder for multiple SPM networks attached to the same UNet.

    At inference time we set a per-network weight (multiplier) based on prompt
    matching; the sum of deltas adds up (since each network wraps the same
    base linear modules via independent LoRA layers).

    Note: SPMNetwork's layers monkey-patch `base.forward`, so the *last* network
    attached is the one whose forward is in the chain. We sequence attaching so
    that each network's SPMLayer.forward calls `self._base_forward` which
    points to the previous network's forward (or the original Linear if it's
    the first). Therefore the deltas compose additively.
    """

    def __init__(self):
        self.networks: dict[str, SPMNetwork] = {}
        self.concept_names: list[str] = []

    def add(self, concept: str, net: SPMNetwork):
        self.networks[concept] = net
        if concept not in self.concept_names:
            self.concept_names.append(concept)

    def set_weights(self, weights: dict[str, float]):
        for c in self.concept_names:
            w = weights.get(c, 0.0)
            self.networks[c].set_multiplier(w)

    def disable_all(self):
        for net in self.networks.values():
            net.set_multiplier(0.0)

    def enable_all(self, m: float = 1.0):
        for net in self.networks.values():
            net.set_multiplier(m)
