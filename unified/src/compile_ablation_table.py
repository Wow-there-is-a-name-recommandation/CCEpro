from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_summary(base_dir: Path, run_name: str, method: str = "algorithm_a") -> Path:
    p1 = base_dir / run_name / f"{method}_summary.json"
    if p1.exists():
        return p1
    p2 = base_dir / f"{run_name}_summary.json"
    if p2.exists():
        return p2
    raise FileNotFoundError(
        f"Could not find summary json for run='{run_name}'. "
        f"Tried: {p1} and {p2}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="unified/eval_ablation")
    parser.add_argument("--method", type=str, default="algorithm_a")
    parser.add_argument("--base_run", type=str, default="base_small")
    parser.add_argument("--filter_run", type=str, default="filter_only_small")
    parser.add_argument("--guidance_run", type=str, default="guidance_only_small")
    parser.add_argument("--full_run", type=str, default="full_small")
    parser.add_argument("--out_dir", type=str, default="unified/report_assets/ablation")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_map = [
        ("Base", args.base_run, False, False),
        ("Filter only", args.filter_run, True, False),
        ("Guidance only", args.guidance_run, False, True),
        ("Full", args.full_run, True, True),
    ]

    rows = []
    for label, run_name, use_filter, use_guidance in run_map:
        sp = find_summary(base_dir, run_name, method=args.method)
        s = read_json(sp)

        rows.append({
            "Setting": label,
            "Filter": "On" if use_filter else "Off",
            "Reverse Guidance": "On" if use_guidance else "Off",
            "AVG_e_CS": s.get("AVG_e_CS"),
            "AVG_r_CS": s.get("AVG_r_CS"),
            "coco_CS": s.get("coco_CS"),
            "coco_FID": s.get("coco_FID"),
            "summary_path": str(sp),
        })

    df = pd.DataFrame(rows)
    order = {"Base": 0, "Filter only": 1, "Guidance only": 2, "Full": 3}
    df["__order"] = df["Setting"].map(order)
    df = df.sort_values("__order").drop(columns="__order")

    csv_path = out_dir / "ablation_summary_table.csv"
    md_path = out_dir / "ablation_summary_table.md"

    df.to_csv(csv_path, index=False)

    display_df = df.copy()
    for col in ["AVG_e_CS", "AVG_r_CS", "coco_CS", "coco_FID"]:
        display_df[col] = display_df[col].map(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(display_df.to_markdown(index=False))

    print(f"[ok] saved: {csv_path}")
    print(f"[ok] saved: {md_path}")
    print()
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()
