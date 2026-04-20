"""
Compatibility loader for the parent project's CoSA+GBA SPM safetensors.

The parent SPM saves to a different format than ``unified/src/spm.py``:
  - safetensors key naming: ``lora_unet_<module_path_with_underscores>.lora_{down,up}.weight``
  - separate ``.alpha`` tensors per layer
  - wraps Conv2d modules (proj_in/proj_out) in addition to Linear
  - covers self-attention (attn1) and feed-forward as well as cross-attention

This file loads such a checkpoint and monkey-patches the UNet's ``forward``
methods to add the LoRA delta, entirely without importing parent code. It is
a one-off reader used to produce the CoSA+GBA column in the qualitative
figure so we can compare against the team's reused result.

Usage:
    pipe = load_sd14(device, dtype)
    wrapper = ParentSPMAdapter(pipe.unet)
    wrapper.load_and_attach([
        "SPM/output/cosa_superman_gba_g05/superman_gba_g05_last.safetensors",
        "SPM/output/cosa_vangogh_gba_g05/van_gogh_gba_g05_last.safetensors",
        "SPM/output/cosa_snoopy_gba_g05/snoopy_gba_g05_last.safetensors",
    ])
    ...  # generate images as usual
    wrapper.detach()
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from safetensors.torch import load_file


def _module_path_to_prefix(path: str) -> str:
    """torch module path (e.g., 'down_blocks.0.attentions.0.proj_in')
    → parent safetensors prefix 'lora_unet_down_blocks_0_attentions_0_proj_in'."""
    return "lora_unet_" + path.replace(".", "_")


def _prefix_to_module_path(prefix: str) -> str:
    """Inverse of above. Strips 'lora_unet_' then replaces '_' with '.'.
    This is ambiguous (module names may contain '_'), so we use a second
    pass that matches against the actual UNet modules instead."""
    assert prefix.startswith("lora_unet_")
    return prefix[len("lora_unet_"):].replace("_", ".")


class _HookedLinear(nn.Module):
    """Wraps a Linear module and adds a LoRA delta on forward."""

    def __init__(self, base: nn.Linear, lora_down: torch.Tensor,
                 lora_up: torch.Tensor, scale: float):
        super().__init__()
        self.base = base
        self.scale = scale
        self.multiplier: float = 1.0
        self._base_forward = base.forward
        # Register buffers (no grad, move with module)
        self.register_buffer("ld", lora_down)
        self.register_buffer("lu", lora_up)
        base.forward = self.forward

    def forward(self, x):
        y = self._base_forward(x)
        if self.multiplier == 0.0:
            return y
        # x: [..., in];  ld: [rank, in];  lu: [out, rank]
        d = torch.nn.functional.linear(x, self.ld)    # [..., rank]
        u = torch.nn.functional.linear(d, self.lu)    # [..., out]
        return y + self.multiplier * self.scale * u

    def detach(self):
        self.base.forward = self._base_forward


class _HookedConv2d(nn.Module):
    """Wraps a Conv2d module and adds a LoRA delta on forward.

    Parent saves Conv2d LoRAs as two 1x1 conv weights. We apply them
    equivalently via two linear convolutions with kernel_size=1.
    """

    def __init__(self, base: nn.Conv2d, lora_down: torch.Tensor,
                 lora_up: torch.Tensor, scale: float):
        super().__init__()
        self.base = base
        self.scale = scale
        self.multiplier: float = 1.0
        self._base_forward = base.forward
        self.register_buffer("ld", lora_down)      # [rank, in, 1, 1]
        self.register_buffer("lu", lora_up)        # [out, rank, 1, 1]
        base.forward = self.forward

    def forward(self, x):
        y = self._base_forward(x)
        if self.multiplier == 0.0:
            return y
        d = torch.nn.functional.conv2d(x, self.ld)
        u = torch.nn.functional.conv2d(d, self.lu)
        return y + self.multiplier * self.scale * u

    def detach(self):
        self.base.forward = self._base_forward


class ParentSPMAdapter:
    """Load & attach one or more parent-format SPM safetensors to a UNet.

    Multiple SPMs compose additively (each wraps the already-hooked forward).
    """

    def __init__(self, unet: nn.Module):
        self.unet = unet
        # map: module_path → nn.Module
        self.modules_by_path: dict[str, nn.Module] = dict(unet.named_modules())
        # map: flattened_prefix → module_path (for each Linear/Conv2d with
        #   "attn" or "ff" or "proj_in"/"proj_out" in its path)
        self.prefix_map: dict[str, str] = {}
        for path, mod in self.modules_by_path.items():
            if not isinstance(mod, (nn.Linear, nn.Conv2d)): continue
            self.prefix_map[_module_path_to_prefix(path)] = path
        self.hooks: list = []

    def load_and_attach(self, paths: Iterable[str | Path],
                        multiplier: float = 1.0) -> int:
        """Attach every LoRA layer in each safetensors file.  Returns total
        count of hooked layers."""
        total = 0
        for p in paths:
            state = load_file(str(p))
            # Collect per-prefix sets of {lora_down, lora_up, alpha}
            layers: dict[str, dict[str, torch.Tensor]] = {}
            for k, v in state.items():
                # key format: <prefix>.alpha  or  <prefix>.lora_down.weight  etc.
                if k.endswith(".alpha"):
                    prefix = k[:-len(".alpha")]
                    layers.setdefault(prefix, {})["alpha"] = v
                elif k.endswith(".lora_down.weight"):
                    prefix = k[:-len(".lora_down.weight")]
                    layers.setdefault(prefix, {})["lora_down"] = v
                elif k.endswith(".lora_up.weight"):
                    prefix = k[:-len(".lora_up.weight")]
                    layers.setdefault(prefix, {})["lora_up"] = v

            for prefix, parts in layers.items():
                if "lora_down" not in parts or "lora_up" not in parts:
                    continue
                if prefix not in self.prefix_map:
                    # Unknown prefix — skip silently
                    continue
                path = self.prefix_map[prefix]
                base_mod = self.modules_by_path[path]
                # alpha default = rank → scale = 1
                rank = parts["lora_down"].shape[0]
                alpha = parts.get("alpha",
                                  torch.tensor(float(rank))).item()
                scale = float(alpha) / float(rank) * multiplier

                ld = parts["lora_down"].to(base_mod.weight.device,
                                           dtype=base_mod.weight.dtype)
                lu = parts["lora_up"].to(base_mod.weight.device,
                                         dtype=base_mod.weight.dtype)

                if isinstance(base_mod, nn.Linear):
                    hook = _HookedLinear(base_mod, ld, lu, scale)
                elif isinstance(base_mod, nn.Conv2d):
                    hook = _HookedConv2d(base_mod, ld, lu, scale)
                else:
                    continue
                self.hooks.append(hook)
                total += 1
        return total

    def detach(self) -> None:
        for h in self.hooks:
            h.detach()
        self.hooks.clear()

    def set_multiplier(self, m: float) -> None:
        for h in self.hooks:
            h.multiplier = m


# ---------------------------------------------------------------------------
# eval.py hook — returns a callable gen(prompt, seed) -> tensor
# ---------------------------------------------------------------------------

def make_parent_cosa_gba_generator(
    device: str,
    checkpoints: list[str] | None = None,
):
    """Create a CoSA+GBA image generator using parent safetensors."""
    from diffusers import DDIMScheduler
    import torch
    from unified.src.common import load_sd14

    if checkpoints is None:
        root = "/home/iiixr/Documents/users/seongrae/github/CoSA/SPM/output"
        checkpoints = [
            f"{root}/cosa_superman_gba_g05/superman_gba_g05_last.safetensors",
            f"{root}/cosa_vangogh_gba_g05/vangogh_gba_g05_last.safetensors",
            f"{root}/cosa_snoopy_gba_g05/snoopy_gba_g05_last.safetensors",
        ]

    pipe = load_sd14(device=device, dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    adapter = ParentSPMAdapter(pipe.unet)
    n = adapter.load_and_attach(checkpoints)
    print(f"[cosa_gba] attached {n} LoRA layers from {len(checkpoints)} ckpts")

    @torch.no_grad()
    def gen(prompt: str, seed: int) -> torch.Tensor:
        g = torch.Generator(device=device).manual_seed(seed)
        out = pipe(prompt, num_inference_steps=30, guidance_scale=7.5,
                   generator=g, output_type="pt")
        return out.images
    return gen, adapter
