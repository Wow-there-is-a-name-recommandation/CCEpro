from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from cleanfid import fid

TARGETS = ["superman", "van gogh", "snoopy"]
RELATED_SUBSET = {
    "superman": ["batman", "thor"],
    "van gogh": ["picasso", "monet"],
    "snoopy": ["mickey", "pikachu"],
}
DEFAULT_METHODS = ["sd_v1.4", "algorithm_a"]


def safe_stem(name: str) -> str:
    return name.replace(" ", "_")


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_folder_fid(ref_dir: Path, cur_dir: Path, device: str) -> float:
    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference folder missing: {ref_dir}")
    if not cur_dir.exists():
        raise FileNotFoundError(f"Current folder missing: {cur_dir}")
    ref_imgs = list(ref_dir.glob("*.png"))
    cur_imgs = list(cur_dir.glob("*.png"))
    if len(ref_imgs) == 0:
        raise ValueError(f"No PNGs found in reference folder: {ref_dir}")
    if len(cur_imgs) == 0:
        raise ValueError(f"No PNGs found in current folder: {cur_dir}")

    return float(
        fid.compute_fid(
            str(ref_dir),
            str(cur_dir),
            device=device,
            mode="clean",
            batch_size=16,
            num_workers=0,
        )
    )


def load_method_summary(eval_dir: Path, method: str) -> dict:
    sp = eval_dir / f"{method}_summary.json"
    if not sp.exists():
        raise FileNotFoundError(f"Summary missing: {sp}")
    return read_json(sp)


def build_metric_row(
    eval_dir: Path,
    method: str,
    device: str,
    update_summary: bool = True,
) -> dict:
    s = load_method_summary(eval_dir, method)
    pcs = s.get("per_concept_CS", {})

    ref_root = eval_dir / "sd_v1.4"
    cur_root = eval_dir / method

    row: Dict[str, object] = {"method": method}

    related_fid_detail: Dict[str, Dict[str, float]] = {}
    for target in TARGETS:
        row[f"{safe_stem(target)}_CS"] = pcs.get(target)

        rel_fids: List[float] = []
        related_fid_detail[target] = {}
        for rel in RELATED_SUBSET[target]:
            if method == "sd_v1.4":
                score = 0.0
            else:
                score = compute_folder_fid(
                    ref_root / safe_stem(rel),
                    cur_root / safe_stem(rel),
                    device=device,
                )
            related_fid_detail[target][rel] = score
            rel_fids.append(score)

        row[f"{safe_stem(target)}_related_avg_FID"] = float(np.mean(rel_fids)) if rel_fids else None

    row["coco_CS"] = s.get("coco_CS")
    if method == "sd_v1.4":
        coco_fid = 0.0
    else:
        coco_fid = compute_folder_fid(
            ref_root / "coco",
            cur_root / "coco",
            device=device,
        )
    row["coco_FID"] = coco_fid

    if update_summary:
        s["related_FID_detail"] = related_fid_detail
        for target in TARGETS:
            s[f"{safe_stem(target)}_related_avg_FID"] = row[f"{safe_stem(target)}_related_avg_FID"]
        s["coco_FID"] = coco_fid
        write_json(eval_dir / f"{method}_summary.json", s)

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="unified/eval_output")
    parser.add_argument("--out_dir", type=str, default="unified/report_assets_revised")
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_update_summary", action="store_true")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = []
    for method in methods:
        row = build_metric_row(
            eval_dir=eval_dir,
            method=method,
            device=args.device,
            update_summary=not args.no_update_summary,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    ordered_cols = [
        "method",
        "superman_CS",
        "superman_related_avg_FID",
        "van_gogh_CS",
        "van_gogh_related_avg_FID",
        "snoopy_CS",
        "snoopy_related_avg_FID",
        "coco_CS",
        "coco_FID",
    ]
    df = df[[c for c in ordered_cols if c in df.columns]]

    csv_path = out_dir / "target_cs_relatedfid_coco_table.csv"
    md_path = out_dir / "target_cs_relatedfid_coco_table.md"

    df.to_csv(csv_path, index=False)

    display_df = df.copy()
    for col in display_df.columns:
        if col != "method":
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.floating)) and x is not None else x
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(display_df.to_markdown(index=False))

    print(f"[ok] saved: {csv_path}")
    print(f"[ok] saved: {md_path}")
    print()
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()
