from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

TARGETS = ["superman", "van gogh", "snoopy"]
RELATED_SUBSET = {
    "superman": ["batman", "thor"],
    "van gogh": ["picasso", "monet"],
    "snoopy": ["mickey", "pikachu"],
}
DEFAULT_METHODS = ["sd_v1.4", "algorithm_a"]


def safe_stem(name: str) -> str:
    return name.replace(" ", "_")


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_summary_rows(eval_dir: Path, methods: list[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        sp = eval_dir / f"{method}_summary.json"
        if not sp.exists():
            print(f"[warn] summary missing: {sp}")
            continue
        s = read_json(sp)
        row = {
            "method": method,
            "AVG_e_CS": s.get("AVG_e_CS"),
            "AVG_r_CS": s.get("AVG_r_CS"),
            "coco_CS": s.get("coco_CS"),
            "coco_FID": s.get("coco_FID"),
        }
        for target in TARGETS:
            row[f"{target}_CS"] = s.get("per_concept_CS", {}).get(target)
            row[f"{target}_r_CS_avg"] = s.get(f"{target}_r_CS_avg")
        rows.append(row)
    return pd.DataFrame(rows)


def collect_per_concept_rows(eval_dir: Path, methods: list[str]) -> pd.DataFrame:
    rows = []
    concepts = TARGETS + [c for rels in RELATED_SUBSET.values() for c in rels]
    for method in methods:
        sp = eval_dir / f"{method}_summary.json"
        if not sp.exists():
            continue
        s = read_json(sp)
        pcs = s.get("per_concept_CS", {})
        for concept in concepts:
            group = "target" if concept in TARGETS else "related"
            rows.append({
                "method": method,
                "concept": concept,
                "group": group,
                "CS": pcs.get(concept),
            })
    return pd.DataFrame(rows)


def save_tables(eval_dir: Path, out_dir: Path, methods: list[str]) -> None:
    ensure_dir(out_dir)
    summary_df = collect_summary_rows(eval_dir, methods)
    concept_df = collect_per_concept_rows(eval_dir, methods)

    summary_csv = out_dir / "quant_summary_table.csv"
    concept_csv = out_dir / "per_concept_cs_table.csv"
    summary_md = out_dir / "quant_summary_table.md"
    concept_md = out_dir / "per_concept_cs_table.md"

    if not summary_df.empty:
        summary_df.to_csv(summary_csv, index=False)
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write(summary_df.to_markdown(index=False))
    if not concept_df.empty:
        concept_df.to_csv(concept_csv, index=False)
        with open(concept_md, "w", encoding="utf-8") as f:
            f.write(concept_df.to_markdown(index=False))

    print(f"[ok] saved: {summary_csv}")
    print(f"[ok] saved: {concept_csv}")


def top_bottom_rows(ranking_rows: list[dict], k: int, prefer_low: bool) -> list[dict]:
    rows = [r for r in ranking_rows if "cs" in r]
    rows = sorted(rows, key=lambda x: x["cs"], reverse=not prefer_low)
    return rows[:k]


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if len(trial) <= max_chars:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def load_font(size: int):
    for cand in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        p = Path(cand)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def build_grid(
    image_infos: list[dict],
    title: str,
    out_path: Path,
    ncols: int = 3,
    thumb_w: int = 320,
    thumb_h: int = 320,
    pad: int = 20,
    header_h: int = 70,
    caption_h: int = 110,
) -> None:
    if not image_infos:
        return

    title_font = load_font(28)
    cap_font = load_font(18)
    small_font = load_font(16)

    n = len(image_infos)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    canvas_w = pad + ncols * (thumb_w + pad)
    cell_h = thumb_h + caption_h
    canvas_h = header_h + pad + nrows * (cell_h + pad)

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18), title, fill="black", font=title_font)

    for idx, info in enumerate(image_infos):
        r = idx // ncols
        c = idx % ncols
        x = pad + c * (thumb_w + pad)
        y = header_h + pad + r * (cell_h + pad)

        img = Image.open(info["path"]).convert("RGB")
        img.thumbnail((thumb_w, thumb_h))
        img_x = x + (thumb_w - img.width) // 2
        img_y = y + (thumb_h - img.height) // 2
        draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline="black", width=1)
        canvas.paste(img, (img_x, img_y))

        cy = y + thumb_h + 8
        label = info.get("concept") or info.get("group") or ("coco" if "coco" in str(info.get("path", "")).lower() else "sample")
        line1 = f"{label} | CS={info['cs']:.4f}"
        line2 = f"seed={info.get('seed', '-')}, file={info.get('file', info['path'].name)}"
        draw.text((x, cy), line1, fill="black", font=cap_font)
        draw.text((x, cy + 24), line2, fill="black", font=small_font)

        prompt_lines = wrap_text(info.get("prompt", ""), 38)[:3]
        py = cy + 48
        for pl in prompt_lines:
            draw.text((x, py), pl, fill="black", font=small_font)
            py += 18

    ensure_dir(out_path.parent)
    canvas.save(out_path)
    print(f"[ok] saved grid: {out_path}")


def collect_infos_for_grid(concept_dir: Path, mode: str, k: int) -> list[dict]:
    ranking_path = concept_dir / "cs_ranking.json"
    if not ranking_path.exists():
        print(f"[warn] ranking missing: {ranking_path}")
        return []
    rows = read_json(ranking_path)
    if mode == "lowest":
        selected = top_bottom_rows(rows, k, prefer_low=True)
    elif mode == "highest":
        selected = top_bottom_rows(rows, k, prefer_low=False)
    else:
        low = top_bottom_rows(rows, k, prefer_low=True)
        high = top_bottom_rows(rows, k, prefer_low=False)
        selected = low + high

    infos = []
    for row in selected:
        img_path = concept_dir / row["file"]
        if not img_path.exists():
            continue
        infos.append({**row, "path": img_path})
    return infos


def build_method_grids(eval_dir: Path, out_dir: Path, methods: list[str], k: int) -> None:
    grid_root = out_dir / "grids"
    ensure_dir(grid_root)
    concepts = TARGETS + [c for rels in RELATED_SUBSET.values() for c in rels] + ["coco"]
    for method in methods:
        for concept in concepts:
            concept_dir = eval_dir / method / (safe_stem(concept) if concept != "coco" else "coco")
            if not concept_dir.exists():
                continue

            if concept == "coco":
                both_infos = collect_infos_for_grid(concept_dir, mode="both", k=k)
                if both_infos:
                    build_grid(
                        both_infos,
                        title=f"{method} | COCO prompts | lowest {k} + highest {k} CLIP matches",
                        out_path=grid_root / f"{method}_coco_low{k}_high{k}.png",
                        ncols=min(3, len(both_infos)),
                    )
                continue

            low_infos = collect_infos_for_grid(concept_dir, mode="lowest", k=k)
            high_infos = collect_infos_for_grid(concept_dir, mode="highest", k=k)
            if low_infos:
                build_grid(
                    low_infos,
                    title=f"{method} | {concept} | lowest {k} CS samples",
                    out_path=grid_root / f"{method}_{safe_stem(concept)}_lowest{k}.png",
                )
            if high_infos:
                build_grid(
                    high_infos,
                    title=f"{method} | {concept} | highest {k} CS samples",
                    out_path=grid_root / f"{method}_{safe_stem(concept)}_highest{k}.png",
                )


def load_ranking_rows(eval_dir: Path, method: str, concept: str) -> list[dict]:
    concept_dir = eval_dir / method / safe_stem(concept)
    ranking_path = concept_dir / "cs_ranking.json"
    if not ranking_path.exists():
        print(f"[warn] ranking missing: {ranking_path}")
        return []
    rows = read_json(ranking_path)
    out = []
    for row in rows:
        file_name = row.get("file")
        if not file_name:
            continue
        img_path = concept_dir / file_name
        if not img_path.exists():
            continue
        out.append({**row, "path": img_path, "concept": concept})
    return out


def load_coco_rows(eval_dir: Path, method: str) -> list[dict]:
    coco_dir = eval_dir / method / "coco"
    ranking_path = coco_dir / "cs_ranking.json"
    if not ranking_path.exists():
        print(f"[warn] ranking missing: {ranking_path}")
        return []
    rows = read_json(ranking_path)
    out = []
    for row in rows:
        file_name = row.get("file")
        if not file_name:
            continue
        img_path = coco_dir / file_name
        if not img_path.exists():
            continue
        out.append({**row, "path": img_path, "concept": "coco"})
    return out


def pick_one(rows: list[dict], mode: str = "highest") -> dict | None:
    rows = [r for r in rows if "cs" in r]
    if not rows:
        return None
    reverse = mode == "highest"
    rows_sorted = sorted(rows, key=lambda x: x["cs"], reverse=reverse)
    return rows_sorted[0]


def pick_target_representative(eval_dir: Path, method: str, target: str) -> dict | None:
    rows = load_ranking_rows(eval_dir, method, target)
    mode = "highest" if method == "sd_v1.4" else "lowest"
    pick = pick_one(rows, mode=mode)
    if pick is not None:
        pick = dict(pick)
        pick["panel_kind"] = "target"
        pick["target_group"] = target
        pick["display_label"] = f"{target} target"
    return pick


def choose_shared_related_concept(eval_dir: Path, target: str) -> str | None:
    """
    Choose ONE related concept name per target group using Algorithm A:
    among the related subset, choose the concept whose highest-CS sample under
    Algorithm A is the best. This concept name is then shared across both rows.
    """
    candidates = []
    for rel in RELATED_SUBSET[target]:
        rows = load_ranking_rows(eval_dir, "algorithm_a", rel)
        best = pick_one(rows, mode="highest")
        if best is not None:
            candidates.append((rel, best["cs"]))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def pick_related_representative(eval_dir: Path, method: str, target: str, fixed_related_concept: str) -> dict | None:
    """
    After the related concept name is fixed using Algorithm A,
    select the highest-CS sample for BOTH SD and Algorithm A.
    """
    rows = load_ranking_rows(eval_dir, method, fixed_related_concept)
    pick = pick_one(rows, mode="highest")
    if pick is not None:
        pick = dict(pick)
        pick["panel_kind"] = "related"
        pick["target_group"] = target
        pick["display_label"] = f"{target} related ({fixed_related_concept})"
    return pick


def pick_coco_representative(eval_dir: Path, method: str) -> dict | None:
    rows = load_coco_rows(eval_dir, method)
    pick = pick_one(rows, mode="highest")
    if pick is not None:
        pick = dict(pick)
        pick["panel_kind"] = "coco"
        pick["target_group"] = "coco"
        pick["display_label"] = "coco"
    return pick


def select_panel_items_for_method(eval_dir: Path, method: str, shared_related_map: dict[str, str | None]) -> list[dict | None]:
    items: list[dict | None] = []
    for target in TARGETS:
        items.append(pick_target_representative(eval_dir, method, target))
        fixed_rel = shared_related_map.get(target)
        if fixed_rel is None:
            items.append(None)
        else:
            items.append(pick_related_representative(eval_dir, method, target, fixed_rel))
    items.append(pick_coco_representative(eval_dir, method))
    return items


def build_comparison_panel(eval_dir: Path, out_dir: Path, methods: list[str]) -> None:
    needed = ["sd_v1.4", "algorithm_a"]
    if not all(m in methods for m in needed):
        print("[warn] comparison panel requires methods: sd_v1.4, algorithm_a")
        return

    shared_related_map = {target: choose_shared_related_concept(eval_dir, target) for target in TARGETS}

    panel_map = {
        "sd_v1.4": select_panel_items_for_method(eval_dir, "sd_v1.4", shared_related_map),
        "algorithm_a": select_panel_items_for_method(eval_dir, "algorithm_a", shared_related_map),
    }

    cols = 7
    pad = 18
    left_label_w = 120
    top_header_h = 48
    thumb_w = 220
    thumb_h = 220
    caption_h = 88

    cell_w = thumb_w
    cell_h = thumb_h + caption_h

    canvas_w = left_label_w + pad + cols * (cell_w + pad)
    canvas_h = top_header_h + pad + 2 * (cell_h + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    header_font = load_font(18)
    row_font = load_font(22)
    cap_font = load_font(16)
    small_font = load_font(14)

    rel_headers = [shared_related_map.get(t) or "related" for t in TARGETS]
    col_headers = [
        "Superman\n(target)",
        f"Superman\n({rel_headers[0]})",
        "Van Gogh\n(target)",
        f"Van Gogh\n({rel_headers[1]})",
        "Snoopy\n(target)",
        f"Snoopy\n({rel_headers[2]})",
        "COCO",
    ]
    row_headers = ["SD v1.4", "Algorithm A"]

    for c, hdr in enumerate(col_headers):
        x = left_label_w + pad + c * (cell_w + pad)
        lines = hdr.split("\n")
        for i, line in enumerate(lines):
            draw.text((x + 10, 8 + i * 18), line, fill="black", font=header_font)

    for r, row_name in enumerate(row_headers):
        y = top_header_h + pad + r * (cell_h + pad)
        draw.text((10, y + thumb_h // 2), row_name, fill="black", font=row_font)

    ordered_methods = ["sd_v1.4", "algorithm_a"]
    for r, method in enumerate(ordered_methods):
        items = panel_map[method]
        for c, info in enumerate(items):
            x = left_label_w + pad + c * (cell_w + pad)
            y = top_header_h + pad + r * (cell_h + pad)

            draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline="black", width=1)

            if info is None:
                draw.text((x + 10, y + 10), "missing", fill="red", font=cap_font)
                continue

            img = Image.open(info["path"]).convert("RGB")
            img.thumbnail((thumb_w - 6, thumb_h - 6))
            img_x = x + (thumb_w - img.width) // 2
            img_y = y + (thumb_h - img.height) // 2
            canvas.paste(img, (img_x, img_y))

            cy = y + thumb_h + 6
            label = info.get("display_label", info.get("concept", "sample"))
            draw.text((x, cy), label[:28], fill="black", font=cap_font)
            draw.text((x, cy + 18), f"CS={info['cs']:.4f}", fill="black", font=cap_font)
            extra = f"{info.get('concept', '-')} | seed={info.get('seed', '-')}"
            draw.text((x, cy + 36), extra[:30], fill="black", font=small_font)
            prompt = info.get("prompt", "")
            prompt_line = wrap_text(prompt, 28)
            if prompt_line:
                draw.text((x, cy + 54), prompt_line[0], fill="black", font=small_font)

    ensure_dir(out_dir / "grids")
    out_path = out_dir / "grids" / "sd_vs_algorithm_a_representative_panel.png"
    canvas.save(out_path)
    print(f"[ok] saved panel: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="unified/eval_output")
    parser.add_argument("--out_dir", type=str, default="unified/report_assets")
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS),
                        help="Comma-separated method names, e.g. sd_v1.4,algorithm_a")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of lowest/highest images to select per concept")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    save_tables(eval_dir, out_dir, methods)
    build_method_grids(eval_dir, out_dir, methods, k=args.k)
    build_comparison_panel(eval_dir, out_dir, methods)
    print(f"[done] assets written to: {out_dir}")


if __name__ == "__main__":
    main()
