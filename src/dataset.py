# src/dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class SampleRecord:
    image_path: str
    prompt: str
    mode: str   # "erase" | "retain" | "replay" ...


class ConceptEraseDataset(Dataset):
    """
    Expected metadata JSON format:
    [
      {
        "image_path": "data/erase/img_0001.png",
        "prompt": "a painting in the style of Van Gogh",
        "mode": "erase"
      },
      ...
    ]
    """

    def __init__(
        self,
        metadata_json: str,
        image_size: int = 512,
        center_crop: bool = False,
    ) -> None:
        self.metadata_path = Path(metadata_json)
        with self.metadata_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        self.records = [SampleRecord(**item) for item in raw]

        crop = transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size)
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            crop,
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # maps to [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert("RGB")
        pixel_values = self.image_transform(image)

        return {
            "pixel_values": pixel_values,
            "prompt": rec.prompt,
            "mode": rec.mode,
            "image_path": rec.image_path,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    prompts = [item["prompt"] for item in batch]
    modes = [item["mode"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "modes": modes,
        "image_paths": image_paths,
    }