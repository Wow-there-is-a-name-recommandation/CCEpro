"""
Compile the assignment ablation table.

Merges per-method summaries from `unified/eval_output/` with the parent
`CoSA+GBA` result from `SPM/eval_output_sweep/assignment_table.json`, then
emits a CSV + a LaTeX-ready table and an HTML report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from unified.src.common import ERASED_ORDER, RELATED

# --------------------------------------------------------------------------- #
# Keys that we want to expose in the final table
# --------------------------------------------------------------------------- #

ROW_ORDER = ["SD_v1.4", "UCE", "REPEL", "UCE-CL", "CoSA_GBA", "Unified"]

# Map internal summary-file names → display labels used in the table
DISPLAY_NAME = {
    "Algorithm_A": "REPEL",
    "Algorithm_B": "UCE-CL",
}

ASSIGNMENT_COLS = [
    "superman_CS_e",         # Superman_e  (erased target, CS↓)
    "superman_CS_r",         # Superman_r  (related, CS↑)
    "van_gogh_CS_e",
    "van_gogh_CS_r",
    "snoopy_CS_e",
    "snoopy_CS_r",
    "coco_CS",               # MS-COCO CS↑
    "coco_FID",              # MS-COCO FID↓
    "AVG_e_CS",
    "AVG_r_CS",
]


# --------------------------------------------------------------------------- #
# Reading self-generated and parent data
# --------------------------------------------------------------------------- #

def _read_local_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _row_from_local_summary(summary: dict) -> dict:
    """Map unified/eval_output summary → assignment table row."""
    per = summary["per_concept_CS"]
    row = {
        "superman_CS_e": per.get("superman", float("nan")),
        "van_gogh_CS_e": per.get("van gogh", float("nan")),
        "snoopy_CS_e": per.get("snoopy", float("nan")),
        "AVG_e_CS": summary["AVG_e_CS"],
        "AVG_r_CS": summary["AVG_r_CS"],
        # The summary JSON uses the same key-naming as RELATED (with space for
        # "van gogh") so accept both forms.
        "superman_CS_r": summary.get("superman_r_CS_avg", float("nan")),
        "van_gogh_CS_r": (summary.get("van gogh_r_CS_avg")
                          if "van gogh_r_CS_avg" in summary
                          else summary.get("van_gogh_r_CS_avg", float("nan"))),
        "snoopy_CS_r": summary.get("snoopy_r_CS_avg", float("nan")),
        "coco_CS": summary.get("coco_CS", float("nan")),
        "coco_FID": summary.get("coco_FID", float("nan")),
    }
    return row


def _row_from_parent(parent_entry: dict) -> dict:
    """Map parent assignment_table.json entry → our table row."""
    row = {
        "superman_CS_e": parent_entry.get("superman_CS", float("nan")),
        "van_gogh_CS_e": parent_entry.get("van_gogh_CS", float("nan")),
        "snoopy_CS_e": parent_entry.get("snoopy_CS", float("nan")),
        "AVG_e_CS": parent_entry.get("AVG_e_CS", float("nan")),
        "AVG_r_CS": parent_entry.get("AVG_r_CS", float("nan")),
        "superman_CS_r": parent_entry.get("superman_r_CS_avg", float("nan")),
        "van_gogh_CS_r": parent_entry.get("van_gogh_r_CS_avg", float("nan")),
        "snoopy_CS_r": parent_entry.get("snoopy_r_CS_avg", float("nan")),
        "coco_CS": parent_entry.get("coco_CS", float("nan")),
        "coco_FID": parent_entry.get("coco_FID", float("nan")),
    }
    return row


def build_table(
    eval_dir: Path = Path("unified/eval_output"),
    parent_json: Path = Path(
        "/home/iiixr/Documents/users/seongrae/github/CoSA/"
        "SPM/eval_output_sweep/assignment_table.json"
    ),
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}

    # Parent baselines
    if parent_json.exists():
        with open(parent_json) as f:
            parent = json.load(f)
        for name in ("SD_v1.4", "UCE", "CoSA_GBA"):
            if name in parent:
                rows[name] = _row_from_parent(parent[name])

    # Local results
    local_map = {
        "REPEL":    "algorithm_a_summary.json",
        "UCE-CL":   "algorithm_b_summary.json",
        "CoSA_GBA": "cosa_gba_summary.json",   # overrides parent entry if present
        "Unified":  "unified_summary.json",
    }
    for method, fn in local_map.items():
        s = _read_local_summary(eval_dir / fn)
        if s is not None:
            rows[method] = _row_from_local_summary(s)

    # Also merge a local sd_v1.4 summary if present — it uses the same CLIP
    # model as our other local methods (ViT-L/14 open_clip) so the numbers
    # are comparable.
    s = _read_local_summary(eval_dir / "sd_v1.4_summary.json")
    if s is not None:
        rows["SD_v1.4_local"] = _row_from_local_summary(s)

    return rows


# --------------------------------------------------------------------------- #
# Emitters
# --------------------------------------------------------------------------- #

def _fmt(v: float, precision: int = 4) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "—"
    return f"{v:.{precision}f}"


def emit_csv(rows: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["method"] + ASSIGNMENT_COLS
    lines = [",".join(header)]
    order = [r for r in ROW_ORDER if r in rows] + \
            [r for r in rows if r not in ROW_ORDER]
    for name in order:
        row = rows[name]
        lines.append(",".join([name] + [_fmt(row.get(c, float("nan")), 4)
                                         for c in ASSIGNMENT_COLS]))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def emit_markdown(rows: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("| Method | Superman_e↓ | Superman_r | VanGogh_e↓ | VanGogh_r | "
              "Snoopy_e↓ | Snoopy_r | COCO_CS↑ | COCO_FID↓ | AVG_e↓ | AVG_r |")
    sep = "|" + "|".join(["---"] * 11) + "|"
    body = []
    order = [r for r in ROW_ORDER if r in rows] + \
            [r for r in rows if r not in ROW_ORDER]
    for name in order:
        r = rows[name]
        body.append("| " + " | ".join([
            name,
            _fmt(r.get("superman_CS_e"), 4),
            _fmt(r.get("superman_CS_r"), 4),
            _fmt(r.get("van_gogh_CS_e"), 4),
            _fmt(r.get("van_gogh_CS_r"), 4),
            _fmt(r.get("snoopy_CS_e"), 4),
            _fmt(r.get("snoopy_CS_r"), 4),
            _fmt(r.get("coco_CS"), 4),
            _fmt(r.get("coco_FID"), 2),
            _fmt(r.get("AVG_e_CS"), 4),
            _fmt(r.get("AVG_r_CS"), 4),
        ]) + " |")
    with open(path, "w") as f:
        f.write("\n".join([header, sep] + body) + "\n")


def emit_html(rows: dict[str, dict[str, float]], path: Path) -> None:
    """Standalone HTML report with the table + caveats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    order = [r for r in ROW_ORDER if r in rows] + \
            [r for r in rows if r not in ROW_ORDER]
    parts = ["""<!doctype html><html><head><meta charset='utf-8'>
<title>Unified CE — Ablation Table</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;
     max-width:1100px;margin:2em auto;padding:1em;color:#111;}
h1,h2{font-weight:600}
table{border-collapse:collapse;width:100%;margin:1em 0}
th,td{border:1px solid #ccc;padding:.5em .7em;text-align:right;font-variant-numeric:tabular-nums;font-size:.9em}
th:first-child,td:first-child{text-align:left}
tr:nth-child(odd) td{background:#fafafa}
.good{color:#067d39;font-weight:600}
.bad{color:#a33;font-weight:600}
small{color:#555}
</style>
</head><body>
<h1>Unified Continual Concept Erasure — Ablation Table</h1>
<p><small>Main metric: lower CS on erased concepts (e_CS), higher CS on related
(r_CS), lower COCO FID. CoSA+GBA from parent run; Algorithm A / B / Unified
implemented locally. Numbers in the parent-sourced rows use the CLIP-Score
convention from Lyu et al. 2024 (open_clip ViT-L/14).</small></p>
<table><thead><tr>
<th>Method</th><th>Sup_e↓</th><th>Sup_r</th>
<th>VG_e↓</th><th>VG_r</th>
<th>Sno_e↓</th><th>Sno_r</th>
<th>COCO_CS↑</th><th>COCO_FID↓</th>
<th>AVG_e↓</th><th>AVG_r</th>
</tr></thead><tbody>"""]

    for name in order:
        r = rows[name]
        parts.append("<tr><td>" + name + "</td>" +
                     "".join(f"<td>{_fmt(r.get(k, float('nan')), 4)}</td>" for k in [
                         "superman_CS_e", "superman_CS_r",
                         "van_gogh_CS_e", "van_gogh_CS_r",
                         "snoopy_CS_e", "snoopy_CS_r",
                     ]) +
                     f"<td>{_fmt(r.get('coco_CS'), 4)}</td>" +
                     f"<td>{_fmt(r.get('coco_FID'), 2)}</td>" +
                     f"<td>{_fmt(r.get('AVG_e_CS'), 4)}</td>" +
                     f"<td>{_fmt(r.get('AVG_r_CS'), 4)}</td></tr>")
    parts.append("</tbody></table></body></html>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="unified/eval_output")
    parser.add_argument("--out_dir", type=str, default="unified/to_human")
    args = parser.parse_args()

    rows = build_table(eval_dir=Path(args.eval_dir))
    out = Path(args.out_dir)
    emit_csv(rows, out / "ablation_table.csv")
    emit_markdown(rows, out / "ablation_table.md")
    emit_html(rows, out / "ablation_table.html")

    print(f"Wrote ablation outputs to {out}/")
    print(f"  CSV:      {out/'ablation_table.csv'}")
    print(f"  Markdown: {out/'ablation_table.md'}")
    print(f"  HTML:     {out/'ablation_table.html'}")


if __name__ == "__main__":
    main()
