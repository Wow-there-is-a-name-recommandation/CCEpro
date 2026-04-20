# train.py
from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from src.dataset import ConceptEraseDataset, collate_fn
from src.model_loader import load_sd14_components
from src.trainer import BaseTrainer, TrainConfig
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/base_train")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = ConceptEraseDataset(
        metadata_json=args.metadata_json,
        image_size=args.image_size,
        center_crop=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    components = load_sd14_components(
        model_id=args.model_id,
        device=args.device,
    )

    cfg = TrainConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
    )

    trainer = BaseTrainer(
        components=components,
        train_dataloader=dataloader,
        cfg=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()